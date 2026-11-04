"""Perfiles de prop firm del USUARIO (guardrail ``onboarding-c``, fase C-1).

El usuario crea perfiles DUPLICANDO uno existente y cambiando sus numeros; nunca edita los
oficiales (``config/propfirms.yaml``) ni puede pedir reglas que el motor no sabe simular.

- Fichero propio y gitignorado: ``config/propfirms.user.yaml`` (``USER_PROFILES_PATH``), mismo
  formato que el oficial (``firms: {id: spec}``) mas una clave ``meta`` por perfil que el
  loader ignora: ``{based_on, verified_on, notes, created, updated}``.
- ``based_on`` es SIEMPRE el id del oficial RAIZ: de el salen el modelo, el tipo y los campos
  bloqueados.
- Los campos editables viven en una LISTA BLANCA explicita por modelo (``_RULES``): cada
  entrada lleva ``{path, label, unit, type, min, max, section}``. Todo lo demas (modelo de
  fondeada, tipos de drawdown, ejes de dias, moneda de la tarifa, numero de fases...) esta
  bloqueado y se rechaza nombrando el camino. La validacion es estricta por camino: ``meta``
  queda fuera del diff; se rechazan NaN, Infinity, booleanos como texto, caminos fuera de la
  lista blanca o inexistentes en la raiz, y cambios de forma (clave anadida/quitada, lista de
  otra longitud).
- Guardrail ``perfiles-v2``: cada perfil lleva ``company`` (compañía, editable); el fichero de
  usuario pasa a ``{firms: {...}, hidden: [ids]}`` (``hidden`` = oficiales y propios ocultos, no
  borrados); y existen 4 PLANTILLAS ("Nuevo perfil" desde cero, ``meta.template``) derivadas en cada
  llamada de la spec cruda de un oficial base (``template_root``).
- ``main.py`` es dueño de ``_firms`` y de su ``RLock``; este modulo construye el diccionario
  fusionado (oficiales primero, luego los de usuario) y avisa a ``main`` tras cada escritura
  mediante el gancho ``set_reload_hook`` (evita el import circular).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from fastapi import APIRouter, HTTPException, Request

from poc_web.api.model_notes import normalize_company
from src.risk.propfirm import (
    _SUPPORTED_FEE_CCY,
    PropFirmConfig,
    parse_firm_spec,
    validate_runnable,
)

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PATH = ROOT / "config" / "propfirms.yaml"
USER_PROFILES_PATH = ROOT / "config" / "propfirms.user.yaml"  # sustituible en tests

KEY_RE = re.compile(r"^[a-z0-9_]{3,40}$")
EXPORT_FORMAT = "risk-app-profile"
EXPORT_VERSION = 1
_NOTES_MAX = 2000
_NAME_MAX = 80
_COMPANY_MAX = 40

router = APIRouter(prefix="/api/profiles", tags=["profiles"])
_FILE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Lista blanca explicita por modelo (enmiendas 2 y 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    pattern: str            # camino con ``*`` por indice de lista, p. ej. phases.*.max_loss_pct
    label: str              # en espanol; ``{n}`` = indice (1-based) del ULTIMO comodin
    unit: str               # %, USD, dias, ciclos, contratos, fraccion, moneda de la tarifa...
    type: str               # number | integer | boolean | string | enum
    min: float | None
    max: float | None
    section: str            # Cuenta | Fase | Fondeada | Suscripción (Fase -> "Fase {i+1}")
    models: tuple[str, ...]
    min_exclusive: bool = False
    options: tuple[str, ...] | None = None      # solo type == "enum"


_ALL = ("cycle", "ladder", "futures")
_LF = ("ladder", "futures")
_CYC = ("cycle",)
_BIG = 1.0e7


def _pct(pattern, label, section, models=_ALL, lo=0.0, excl=True):
    return Rule(pattern, label, "%", "number", lo, 100.0, section, models, excl)


def _usd(pattern, label, section, models=_ALL, lo=0.0, excl=True):
    return Rule(pattern, label, "USD", "number", lo, _BIG, section, models, excl)


def _int(pattern, label, unit, lo, hi, section, models=_ALL):
    return Rule(pattern, label, unit, "integer", float(lo), float(hi), section, models, False)


def _share(pattern, label, section, models=_ALL):
    return Rule(pattern, label, "fracción", "number", 0.0, 1.0, section, models, True)


_F = "Fase"
_PA = "Fondeada"
_RULES: tuple[Rule, ...] = (
    # --- Cuenta (todos los modelos; account_size SOLO en cycle) ---------------------------
    Rule("display_name", "Nombre", "", "string", 1, _NAME_MAX, "Cuenta", _ALL),
    Rule("company", "Compañía", "", "string", 1, _COMPANY_MAX, "Cuenta", _ALL),
    Rule("account_size", "Tamaño de la cuenta", "USD", "number", 0.0, _BIG, "Cuenta", _CYC, True),
    Rule("fee", "Tarifa del reto", "fee_ccy", "number", 0.0, 1.0e6, "Cuenta", _ALL),
    _pct("profit_split_pct", "Reparto de beneficios (tu parte)", "Cuenta"),
    Rule("refund_fee_on_first_payout", "Devolver la tarifa con el primer payout", "", "boolean",
         None, None, "Cuenta", _ALL),
    # --- Fases ---------------------------------------------------------------------------
    _pct("phases.*.profit_target_pct", "Objetivo de beneficio", _F),
    _pct("phases.*.max_daily_loss_pct", "Pérdida diaria máxima", _F),
    _pct("phases.*.max_loss_pct", "Pérdida máxima total", _F),
    _int("phases.*.min_trading_days", "Días mínimos de trading", "días", 0, 365, _F),
    _int("phases.*.lifetime_days", "Días de vida de la fase", "días", 1, 3650, _F),
    _usd("phases.*.daily_loss_usd", "Pérdida diaria máxima (USD, fatal)", _F),
    _usd("phases.*.drawdown.amount_usd", "Drawdown máximo", _F),
    _usd("phases.*.drawdown.lock_at_balance", "Balance de bloqueo del drawdown", _F),
    _int("phases.*.max_contracts.max_mini", "Máx. contratos mini", "contratos", 1, 100000, _F),
    _int("phases.*.max_contracts.max_micro", "Máx. contratos micro", "contratos", 1, 1000000, _F),
    _share("phases.*.consistency.max_best_day_share", "Cuota máxima del mejor día", _F),
    # --- Fondeada: cycle ------------------------------------------------------------------
    _int("funded.payout_cycle_days", "Días entre payouts", "días", 1, 365, _PA, _CYC),
    _pct("funded.min_payout_profit_pct", "Beneficio mínimo para cobrar", _PA, _CYC, 0.0, False),
    _int("funded.max_payout_cycles", "Tope de ciclos de payout (truncado del modelo)",
         "ciclos", 1, 1000, _PA, _CYC),
    # --- Fondeada: extendidos (ladder y futures) -------------------------------------------
    _usd("funded.drawdown.amount_usd", "Drawdown máximo", _PA, _LF),
    _usd("funded.drawdown.lock_at_balance", "Balance de bloqueo del drawdown", _PA, _LF),
    _usd("funded.drawdown.frozen_floor", "Suelo congelado del drawdown", _PA, ("ladder",)),
    _usd("funded.daily_loss.by_tier_usd.*", "Pérdida diaria máxima (tramo {n})", _PA, _LF),
    _int("funded.scaling_tiers.*.max_mini", "Contratos mini máx. (tramo {n})", "contratos",
         1, 100000, _PA, _LF),
    _int("funded.scaling_tiers.*.max_micro", "Contratos micro máx. (tramo {n})", "contratos",
         1, 1000000, _PA, _LF),
    _usd("funded.payouts.qualifying_day_min_usd", "Beneficio mínimo de un día válido", _PA,
         ("ladder",), 0.0, False),
    _int("funded.payouts.days_per_payout", "Días válidos por payout", "días", 1, 365, _PA,
         ("ladder",)),
    _usd("funded.payouts.ladder_usd.*", "Escalera de retiros (retiro {n})", _PA, ("ladder",)),
    _usd("funded.payouts.min_payout_usd", "Payout mínimo", _PA, ("ladder",)),
    _usd("funded.payouts.min_balance_first", "Balance mínimo para el primer retiro", _PA,
         ("ladder",)),
    _usd("funded.payouts.residual_floor", "Suelo residual (nunca se retira por debajo)", _PA,
         ("ladder",), 0.0, False),
    _share("funded.payouts.consistency.max_best_day_share", "Cuota máxima del mejor día", _PA,
           ("ladder",)),
    _int("funded.inactivity.window_days", "Ventana de inactividad", "días", 1, 365, _PA, _LF),
    _int("funded.inactivity.min_days", "Días activos mínimos en la ventana", "días", 1, 365,
         _PA, _LF),
    _usd("funded.inactivity.min_profit_usd", "Beneficio mínimo por día activo", _PA, _LF, 0.0,
         False),
    _usd("funded.payouts_futures.qualifying_day_min_usd", "Beneficio mínimo de un día válido",
         _PA, ("futures",), 0.0, False),
    _int("funded.payouts_futures.days_per_payout", "Días válidos por payout", "días", 1, 365,
         _PA, ("futures",)),
    _share("funded.payouts_futures.withdraw_share", "Fracción retirable del beneficio", _PA,
           ("futures",)),
    _usd("funded.payouts_futures.request_cap_usd", "Tope por solicitud de retiro", _PA,
         ("futures",)),
    _share("funded.payouts_futures.min_new_profit_frac", "Fracción mínima de beneficio nuevo",
           _PA, ("futures",)),
    _usd("funded.payouts_futures.min_payout_usd", "Payout mínimo", _PA, ("futures",), 0.0, False),
    _usd("funded.payouts_futures.buffer_usd", "Colchón a preservar", _PA, ("futures",)),
    # --- Activación y suscripción ---------------------------------------------------------
    _usd("activation.fee_usd", "Tarifa de activación de la cuenta fondeada", _PA),
    _usd("subscription.monthly_fee_usd", "Cuota mensual", "Suscripción"),
    _usd("subscription.reset_fee_usd", "Coste de reset de pago", "Suscripción", _ALL, 0.0, False),
    _int("subscription.billing_period_days", "Días del periodo de facturación", "días", 1, 365,
         "Suscripción"),
    _int("subscription.campaign_cap_days", "Tope de días de la campaña (truncado del modelo)",
         "días", 1, 3650, "Suscripción"),
    _int("subscription.funded_cap_days", "Tope de días de la fondeada (truncado del modelo)",
         "días", 1, 3650, "Suscripción"),
    _int("subscription.funded_start_delay_days", "Días hasta empezar la fondeada", "días", 0,
         365, "Suscripción"),
)

# Reglas EXTRA de los perfiles creados desde plantilla (``editable_map(root, template=True)``):
# en "desde cero" el usuario escribe todos los importes en USD, así que ``account_size`` es editable
# en todos los modelos (las relaciones R-1..R-5 de ``check_relations`` acotan la coherencia) y
# ``fee_ccy`` es un enumerado. En los duplicados la lista blanca no cambia (``_BLOCKED``).
_TEMPLATE_RULES: tuple[Rule, ...] = (
    Rule("account_size", "Tamaño de la cuenta", "USD", "number", 0.0, _BIG, "Cuenta", _ALL, True),
    Rule("fee_ccy", "Moneda de la tarifa", "", "enum", None, None, "Cuenta", _ALL, False,
         tuple(_SUPPORTED_FEE_CCY)),
)


def _rules_for(template: bool) -> tuple[Rule, ...]:
    if not template:
        return _RULES
    over = {r.pattern: r for r in _TEMPLATE_RULES}
    out: list[Rule] = []
    for r in _RULES:
        out.append(over.get(r.pattern, r))
        if r.pattern == "fee":                      # fee_ccy justo detrás de la tarifa
            out.append(over["fee_ccy"])
    return tuple(out)


# Motivos de los caminos bloqueados (patron con ``*`` por indice). Sirven para el mensaje de
# error; un camino bloqueado SIN entrada aqui recibe el motivo generico.
_BLOCKED: dict[str, str] = {
    "funded.model": "el modelo de fondeada no se puede cambiar: requiere código",
    # PLAN-DESVIO (lectura literal de §2.4): max_payout_cycles solo es editable en `cycle`.
    "funded.max_payout_cycles": "tope de truncado de la simulación en ladder y futures: lo fija "
                                "el perfil oficial",
    "phases.*.drawdown.type": "el tipo de drawdown es semántico (requiere código)",
    "phases.*.drawdown.breach_on": "el tipo de drawdown es semántico (requiere código)",
    "funded.drawdown.type": "el tipo de drawdown es semántico (requiere código)",
    "funded.drawdown.breach_on": "el tipo de drawdown es semántico (requiere código)",
    "day_axis": "el eje de días es semántico (requiere código)",
    "day_axis_cut_hhmm": "el corte del eje de días es semántico (requiere código)",
    "fee_ccy": "la moneda de la tarifa está ligada al perfil oficial",
    "ruleset_version": "la versión de reglas la fija el perfil oficial",
    "phases.*.name": "el número de fases y sus nombres no se pueden cambiar",
    "phases.*.on_horizon_exhausted": "es semántico (requiere código)",
    "activation.deadline_days": "el motor no usa este campo (aviso de Apex)",
    "phases.*.consistency.strict": "la semántica de la consistencia requiere código",
    "phases.*.consistency.window": "la semántica de la consistencia requiere código",
    "funded.payouts.consistency.strict": "la semántica de la consistencia requiere código",
    "funded.payouts.consistency.window": "la semántica de la consistencia requiere código",
    "funded.daily_loss.type": "la semántica del daily loss requiere código",
    "funded.daily_loss.reset": "la semántica del daily loss requiere código",
    "funded.daily_loss.post_hit": "la semántica del daily loss requiere código",
    "funded.payouts_futures.withdrawal_policy": "es una política por petición, no del perfil",
    # PLAN-DESVIO: el plan (§2.4) lista `payouts.*` numéricos como editables; max_payouts SE
    # BLOQUEA porque `_parse_ladder_payouts` exige max_payouts == len(ladder_usd) y la longitud
    # de la lista es fija: cualquier cambio solo produciría un error.
    "funded.payouts.max_payouts": "debe coincidir con la longitud de ladder_usd (fija)",
    "funded.scaling_tiers.*.level": "la estructura de tramos es fija",
    "funded.scaling_tiers.*.profit_range": "los rangos de los tramos son fijos",
    "funded.scaling_tiers.*.profit_range.*": "los rangos de los tramos son fijos",
    "meta": "meta no es una regla",
}
_ACCOUNT_SIZE_MOTIVE = (
    "en ladder y futures las reglas van en USD absolutos ligados al tamaño de la cuenta: "
    "para otro tamaño duplica el perfil oficial de ese tamaño"
)


def _idx_pattern(path: str) -> str:
    return ".".join("*" if t.isdigit() else t for t in path.split("."))


def blocked_reason(path: str, model: str) -> str:
    if path == "account_size" and model != "cycle":
        return _ACCOUNT_SIZE_MOTIVE
    return _BLOCKED.get(_idx_pattern(path), "campo no editable (fuera de la lista blanca)")


def _expand(pattern: str, spec: Any) -> list[str]:
    """Caminos concretos de ``pattern`` que EXISTEN en ``spec`` con valor escalar no nulo."""
    tokens = pattern.split(".")
    out: list[str] = []

    def rec(node: Any, i: int, acc: list[str]) -> None:
        if i == len(tokens):
            if node is not None and not isinstance(node, (dict, list)):
                out.append(".".join(acc))
            return
        t = tokens[i]
        if t == "*":
            if isinstance(node, list):
                for j, x in enumerate(node):
                    rec(x, i + 1, acc + [str(j)])
        elif isinstance(node, dict) and t in node:
            rec(node[t], i + 1, acc + [t])

    rec(spec, 0, [])
    return out


def _get(spec: Any, path: str) -> Any:
    node = spec
    for t in path.split("."):
        node = node[int(t)] if isinstance(node, list) else node[t]
    return node


def root_model(root_spec: dict) -> str:
    return str((root_spec.get("funded") or {}).get("model", "cycle"))


def editable_map(root_spec: dict, template: bool = False) -> dict[str, Rule]:
    """``{camino concreto: Rule}`` de la spec RAIZ (existencia con valor no nulo). ``template=True``
    = perfil creado desde plantilla (``_TEMPLATE_RULES``: ``account_size`` en todos los modelos y
    ``fee_ccy`` enumerado)."""
    model = root_model(root_spec)
    out: dict[str, Rule] = {}
    for rule in _rules_for(template):
        if model not in rule.models:
            continue
        for path in _expand(rule.pattern, root_spec):
            out[path] = rule
    return out


def _indices(path: str, rule: Rule) -> list[int]:
    toks, pats = path.split("."), rule.pattern.split(".")
    return [int(t) for t, p in zip(toks, pats) if p == "*"]


def _label(path: str, rule: Rule) -> str:
    idx = _indices(path, rule)
    return rule.label.format(n=(idx[-1] + 1) if idx else 0)


def _section(path: str, rule: Rule) -> str:
    if rule.section == _F:
        return f"Fase {_indices(path, rule)[0] + 1}"
    return rule.section


def editable_fields(root_spec: dict, template: bool = False) -> list[dict]:
    """Esquema del formulario: ``[{path,label,unit,type,min,max,min_exclusive,section,reference}]``
    (+ ``options`` en los ``enum``)."""
    fee_ccy = str(root_spec.get("fee_ccy", "EUR")).upper()
    fields = []
    for path, rule in editable_map(root_spec, template).items():
        f = {
            "path": path, "label": _label(path, rule),
            "unit": fee_ccy if rule.unit == "fee_ccy" else rule.unit,
            "type": rule.type, "min": rule.min, "max": rule.max,
            "min_exclusive": rule.min_exclusive, "section": _section(path, rule),
            "reference": _get(root_spec, path),
        }
        if rule.options is not None:
            f["options"] = list(rule.options)
        fields.append(f)
    return fields


# ---------------------------------------------------------------------------
# Diff estricto contra la raiz
# ---------------------------------------------------------------------------
_MISSING = object()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def diff_paths(root: Any, new: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any, Any]]:
    """Caminos que difieren: ``[(camino, valor_raiz, valor_nuevo)]``.

    Un cambio de FORMA (clave anadida o quitada, lista de otra longitud, dict por escalar, bool
    por numero) se reporta en el camino del propio nodo, no de sus hojas.
    """
    path = ".".join(prefix)
    if isinstance(root, dict) and isinstance(new, dict):
        out: list[tuple[str, Any, Any]] = []
        keys = list(root) + [k for k in new if k not in root]
        for k in keys:
            a, b = root.get(k, _MISSING), new.get(k, _MISSING)
            if a is _MISSING or b is _MISSING:
                out.append((".".join(prefix + (str(k),)), a, b))
            else:
                out.extend(diff_paths(a, b, prefix + (str(k),)))
        return out
    if isinstance(root, list) and isinstance(new, list):
        if len(root) != len(new):
            return [(path, root, new)]
        out = []
        for j, (a, b) in enumerate(zip(root, new)):
            out.extend(diff_paths(a, b, prefix + (str(j),)))
        return out
    return [] if _same(root, new) else [(path, root, new)]


def strip_meta(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "meta"}


def blocked_changes(root_spec: dict, spec: dict, template: bool = False) -> list[str]:
    """Caminos cambiados respecto a la raiz que NO estan en la lista blanca (``meta`` fuera)."""
    emap = editable_map(root_spec, template)
    return [p for p, _, _ in diff_paths(root_spec, strip_meta(spec)) if p not in emap]


def _check_value(path: str, rule: Rule, v: Any, fee_ccy: str) -> tuple[Any, str | None]:
    """(valor normalizado, error|None) de un camino editable cambiado."""
    label = f"{path} ({_label(path, rule)})"
    unit = fee_ccy if rule.unit == "fee_ccy" else rule.unit
    if v is None:                       # plantillas: campo sin rellenar
        return v, f"{label}: falta el valor"
    if rule.type == "enum":
        if not isinstance(v, str) or v not in (rule.options or ()):
            return v, f"{label}: debe ser uno de {', '.join(rule.options or ())}"
        return v, None
    if rule.type == "boolean":
        if not isinstance(v, bool):
            return v, f"{label}: debe ser true o false (booleano real, no texto ni número)"
        return v, None
    if rule.type == "string":
        if not isinstance(v, str) or not v.strip():
            return v, f"{label}: debe ser un texto no vacío"
        v = normalize_company(v) if rule.pattern == "company" else v.strip()
        if len(v) > rule.max:
            return v, f"{label}: máximo {int(rule.max)} caracteres"
        return v, None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v, f"{label}: debe ser un número (no texto ni booleano)"
    if not math.isfinite(v):
        return v, f"{label}: debe ser un número finito (NaN e Infinity no valen)"
    if rule.type == "integer":
        if v != int(v):
            return v, f"{label}: debe ser un número entero"
        v = int(v)
    else:
        v = float(v) if isinstance(v, float) else v
    lo, hi = rule.min, rule.max
    bad_lo = lo is not None and (v <= lo if rule.min_exclusive else v < lo)
    if bad_lo or (hi is not None and v > hi):
        lo_txt = "" if lo is None else f"{'>' if rule.min_exclusive else '>='} {lo:g}"
        hi_txt = "" if hi is None else f"<= {hi:g}"
        rng = " y ".join(t for t in (lo_txt, hi_txt) if t)
        return v, f"{label}: fuera de rango, debe ser {rng} {unit}".rstrip()
    return v, None


def _set(spec: dict, path: str, value: Any) -> None:
    node: Any = spec
    toks = path.split(".")
    for t in toks[:-1]:
        node = node[int(t)] if isinstance(node, list) else node[t]
    last = toks[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


# ---------------------------------------------------------------------------
# Relaciones entre importes de futuros (Enmienda v2 4): condicion para abrir ``account_size``
# ---------------------------------------------------------------------------
def _opt_num(spec: Any, path: str) -> float | None:
    try:
        v = _get(spec, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def check_relations(body: dict, model: str) -> list[str]:
    """R-1..R-5 entre importes de un perfil ``ladder``/``futures`` (solo si existen los caminos):

    - R-1: todo ``*.drawdown.amount_usd`` < ``account_size``.
    - R-2: todo ``*.drawdown.lock_at_balance`` >= ``account_size``.
    - R-3: ``account_size - funded.drawdown.amount_usd < funded.drawdown.frozen_floor <=
      funded.drawdown.lock_at_balance``.
    - R-4: ``account_size <= funded.payouts.residual_floor <= funded.payouts.min_balance_first``.
    - R-5: todo ``funded.daily_loss.by_tier_usd.*`` <= ``funded.drawdown.amount_usd``.
    """
    if model not in _LF:
        return []
    acc = _opt_num(body, "account_size")
    if acc is None:
        return []
    errs: list[str] = []
    dd_prefixes = [f"phases.{i}.drawdown" for i in range(len(body.get("phases") or []))]
    dd_prefixes.append("funded.drawdown")
    for pre in dd_prefixes:
        amount, lock = _opt_num(body, f"{pre}.amount_usd"), _opt_num(body, f"{pre}.lock_at_balance")
        if amount is not None and not amount < acc:
            errs.append(f"R-1: {pre}.amount_usd ({amount:g}) debe ser menor que el tamaño de la "
                        f"cuenta ({acc:g})")
        if lock is not None and not lock >= acc:
            errs.append(f"R-2: {pre}.lock_at_balance ({lock:g}) debe ser mayor o igual que el "
                        f"tamaño de la cuenta ({acc:g})")
    amount = _opt_num(body, "funded.drawdown.amount_usd")
    frozen = _opt_num(body, "funded.drawdown.frozen_floor")
    lock = _opt_num(body, "funded.drawdown.lock_at_balance")
    if frozen is not None and amount is not None and not frozen > acc - amount:
        errs.append(f"R-3: funded.drawdown.frozen_floor ({frozen:g}) debe ser mayor que tamaño de la "
                    f"cuenta - drawdown máximo ({acc - amount:g})")
    if frozen is not None and lock is not None and not frozen <= lock:
        errs.append(f"R-3: funded.drawdown.frozen_floor ({frozen:g}) debe ser menor o igual que "
                    f"lock_at_balance ({lock:g})")
    residual = _opt_num(body, "funded.payouts.residual_floor")
    first = _opt_num(body, "funded.payouts.min_balance_first")
    if residual is not None and not residual >= acc:
        errs.append(f"R-4: funded.payouts.residual_floor ({residual:g}) debe ser mayor o igual que "
                    f"el tamaño de la cuenta ({acc:g})")
    if residual is not None and first is not None and not residual <= first:
        errs.append(f"R-4: funded.payouts.residual_floor ({residual:g}) debe ser menor o igual que "
                    f"min_balance_first ({first:g})")
    funded = body.get("funded") if isinstance(body.get("funded"), dict) else {}
    dloss = funded.get("daily_loss") if isinstance(funded.get("daily_loss"), dict) else {}
    tiers = dloss.get("by_tier_usd")
    if amount is not None and isinstance(tiers, list):
        for j in range(len(tiers)):
            t = _opt_num(body, f"funded.daily_loss.by_tier_usd.{j}")
            if t is not None and not t <= amount:
                errs.append(f"R-5: funded.daily_loss.by_tier_usd.{j} ({t:g}) debe ser menor o igual "
                            f"que el drawdown máximo de la fondeada ({amount:g})")
    return errs


# ---------------------------------------------------------------------------
# Plantillas ("Nuevo perfil" desde cero; D3 de perfiles-v2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Template:
    id: str
    label: str
    kind: str                  # cfd | futures
    base: str                  # oficial del que se deriva la raíz de validación
    n_phases: int | None = None  # None = las de la base; 1 = ``phases[:1]``


TEMPLATES: dict[str, Template] = {t.id: t for t in (
    Template("cfd_2f", "Reto de 2 fases, drawdown estático y cobros por ciclo", "cfd", "ftmo_50k"),
    Template("cfd_1f", "Reto de 1 fase, drawdown estático y cobros por ciclo", "cfd", "ftmo_50k", 1),
    Template("fut_suscripcion", "Futuros con suscripción mensual (como FTMO Futures)", "futures",
             "ftmo_futures_growth_50k"),
    Template("fut_trailing", "Futuros con trailing intradía y escalera de retiros (como Apex)",
             "futures", "apex_150k_intraday"),
)}


def get_template(tid: Any) -> Template:
    if not isinstance(tid, str) or tid not in TEMPLATES:
        raise ValueError(f"plantilla desconocida: {tid}")
    return TEMPLATES[tid]


def template_root(tid: Any) -> dict:
    """Raíz de validación de una plantilla: copia profunda de la spec CRUDA ACTUAL del oficial base
    (sigue sus cambios; conserva sus valores reales). ``cfd_1f`` = ``phases[:1]``. Si la base trae
    ``ruleset_version`` queda en ``""`` (el perfil nuevo no hereda la versión verificada). Si no
    trae ``fee_ccy`` (CFD) se explicita ``EUR`` (default del loader) para que sea editable."""
    t = get_template(tid)
    off_raw, _ = official()
    if t.base not in off_raw:
        raise ValueError(f"plantilla {tid}: el perfil oficial base '{t.base}' ya no existe")
    root = copy.deepcopy(off_raw[t.base])
    if t.n_phases is not None:
        if not isinstance(root.get("phases"), list) or len(root["phases"]) < t.n_phases:
            raise ValueError(f"plantilla {tid}: el oficial base '{t.base}' no tiene {t.n_phases} fase(s)")
        root["phases"] = root["phases"][: t.n_phases]
    if "ruleset_version" in root:
        root["ruleset_version"] = ""
    # PLAN-DESVIO: las raices CFD no traen `fee_ccy` y `_expand` solo ve caminos con valor, asi que `fee_ccy`
    # (enum, Enm. 6) no seria editable en cfd_1f/cfd_2f; se explicita el default del loader (EUR).
    root.setdefault("fee_ccy", "EUR")
    return root


def template_empty_spec(root: dict) -> dict:
    """La raíz con cada camino editable (modo plantilla) a ``None``: punto de partida del formulario."""
    spec = copy.deepcopy(root)
    for path in editable_map(root, True):
        _set(spec, path, None)
    return spec


def template_info(tid: str) -> dict:
    t = get_template(tid)
    root = template_root(tid)
    return {"id": t.id, "label": t.label, "kind": t.kind, "based_on": t.base,
            "n_phases": len(root.get("phases") or []), "n_fields": len(editable_map(root, True))}


def template_detail(tid: str) -> dict:
    root = template_root(tid)
    fields = editable_fields(root, True)
    schema = []
    for f in fields:
        f = dict(f)
        f["example"] = f.pop("reference")
        schema.append(f)
    return {**template_info(tid), "spec": template_empty_spec(root), "schema": schema}


def inherit_company(spec: Any, root: dict) -> Any:
    """Compatibilidad hacia atrás: una entrada sin ``company`` (creada antes de ``perfiles-v2``) o
    una importación sin ella hereda la de su raíz ANTES de validar (normalización, no error)."""
    if isinstance(spec, dict) and "company" not in spec and "company" in root:
        return {**spec, "company": root["company"]}
    return spec


def _root_of_meta(meta: dict, off_raw: dict) -> tuple[dict | None, bool]:
    """(raíz de validación, es_plantilla) de una entrada de usuario."""
    tid = meta.get("template")
    if tid:
        try:
            return template_root(tid), True
        except ValueError:
            return None, True
    return off_raw.get(meta["based_on"]), False


# ---------------------------------------------------------------------------
# Oficiales (cache por mtime) y fichero de usuario
# ---------------------------------------------------------------------------
_official_cache: dict[str, Any] = {}
_OFFICIAL_LOCK = threading.RLock()


def official() -> tuple[dict[str, dict], dict[str, PropFirmConfig]]:
    """(specs crudas, configs) de los oficiales; cache por (ruta, mtime_ns), bajo lock y con UNA
    sola lectura del YAML (raws y cfgs salen del mismo texto). Los llamadores no mutan los raws."""
    with _OFFICIAL_LOCK:
        stamp = (str(OFFICIAL_PATH), OFFICIAL_PATH.stat().st_mtime_ns)
        if _official_cache.get("stamp") != stamp:
            doc = yaml.safe_load(OFFICIAL_PATH.read_text(encoding="utf-8"))
            raws = (doc or {}).get("firms") or {}
            if not raws:
                raise ValueError(f"{OFFICIAL_PATH}: no hay perfiles bajo 'firms:'")
            cfgs = {k: parse_firm_spec(k, copy.deepcopy(sp)) for k, sp in raws.items()}
            _official_cache.update(stamp=stamp, raws=raws, cfgs=cfgs)
        return _official_cache["raws"], _official_cache["cfgs"]


@dataclass
class UserState:
    cfgs: dict[str, PropFirmConfig] = field(default_factory=dict)
    specs: dict[str, dict] = field(default_factory=dict)    # spec cruda SIN meta
    metas: dict[str, dict] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)        # [{key, error}]
    hidden: list[str] = field(default_factory=list)         # ids ocultos del fichero (TODOS, incl. desconocidos)


_state = UserState()
_reload_hook: Callable[[], Any] | None = None


def set_reload_hook(fn: Callable[[], Any] | None) -> None:
    global _reload_hook
    _reload_hook = fn


def set_state(state: UserState) -> None:
    global _state
    _state = state


def get_errors() -> list[dict]:
    return list(_state.errors)


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    return v.isoformat() if isinstance(v, (date, datetime)) else str(v)


def read_user_file() -> dict[str, Any]:
    """DOCUMENTO del fichero de usuario: ``{"firms": {id: entrada cruda}, "hidden": [ids], ...}``.
    Falta -> documento vacío; corrupto -> ValueError. Conserva las demás claves de primer nivel."""
    empty: dict[str, Any] = {"firms": {}, "hidden": []}
    if not USER_PROFILES_PATH.exists():
        return empty
    try:
        raw = yaml.safe_load(USER_PROFILES_PATH.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{USER_PROFILES_PATH.name} ilegible: {exc}") from exc
    if raw is None:
        return empty
    if not isinstance(raw, dict) or not isinstance(raw.get("firms", {}), (dict, type(None))):
        raise ValueError(f"{USER_PROFILES_PATH.name}: se esperaba 'firms:' con un diccionario")
    hidden = raw.get("hidden", [])
    if hidden is None:
        hidden = []
    if not isinstance(hidden, list) or not all(isinstance(h, str) for h in hidden):
        raise ValueError(f"{USER_PROFILES_PATH.name}: 'hidden' debe ser una lista de ids")
    doc = {k: v for k, v in raw.items() if k not in ("firms", "hidden")}
    return {"firms": dict(raw.get("firms") or {}), "hidden": list(dict.fromkeys(hidden)), **doc}


def load_user_profiles() -> UserState:
    """Parsea el fichero de usuario ENTRADA A ENTRADA (enmienda 6): las validas se cargan, los
    errores se listan. Un fichero corrupto o ausente nunca rompe (estado vacio + aviso)."""
    st = UserState()
    try:
        doc = read_user_file()
    except ValueError as exc:
        st.errors.append({"key": None, "error": str(exc)})
        return st
    entries = doc["firms"]
    st.hidden = list(doc["hidden"])         # TODOS, también los de entradas con error / desconocidos
    off_raw, off_cfg = official()
    for key, entry in entries.items():
        try:
            if not isinstance(key, str) or not KEY_RE.fullmatch(key):
                raise ValueError("id inválido (debe cumplir ^[a-z0-9_]{3,40}$)")
            if key in off_cfg:
                raise ValueError(
                    "el id coincide con un perfil oficial (se omite el tuyo; duplícalo con "
                    "otro id)")
            if not isinstance(entry, dict):
                raise ValueError("la entrada no es un diccionario")
            meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
            based = meta.get("based_on")
            if based not in off_cfg:
                raise ValueError(f"meta.based_on '{based}' no es un perfil oficial")
            tmpl = meta.get("template") or None
            if tmpl is not None:
                tpl = get_template(tmpl)            # ValueError "plantilla desconocida: X"
                if based != tpl.base:
                    raise ValueError(f"meta.based_on '{based}' no coincide con la plantilla "
                                     f"'{tmpl}' (base {tpl.base})")
                root, is_tpl = template_root(tmpl), True
            else:
                root, is_tpl = off_raw[based], False
            verified = _iso(meta.get("verified_on"))
            notes = meta.get("notes")
            meta_errs = check_meta(verified, notes)
            res = validate_spec_body(key, inherit_company(entry, root), root, template=is_tpl)
            errs = res.errors + meta_errs
            if errs:
                raise ValueError("; ".join(errs))
            spec, cfg = res.spec, res.cfg
        except Exception as exc:  # fichero editable a mano: cualquier fallo es de ESA entrada
            st.errors.append({"key": key if isinstance(key, str) else str(key), "error": str(exc)})
            continue
        st.cfgs[key] = cfg
        st.specs[key] = spec
        st.metas[key] = {
            "based_on": based, "verified_on": verified, "notes": notes,
            "created": _iso(meta.get("created")), "updated": _iso(meta.get("updated")),
            "template": tmpl,
        }
    return st


def build_firms() -> tuple[dict[str, PropFirmConfig], UserState]:
    """Diccionario fusionado (oficiales PRIMERO, luego los de usuario) y el estado de usuario."""
    _, off_cfg = official()
    st = load_user_profiles()
    merged = dict(off_cfg)
    merged.update(st.cfgs)
    return merged, st


def based_on(key: str) -> str:
    """Id del oficial raiz: el propio id si es oficial (o desconocido), ``meta.based_on`` si es de
    usuario."""
    meta = _state.metas.get(key)
    return meta["based_on"] if meta else key


def is_user_profile(key: str) -> bool:
    return key in _state.cfgs


def company_of(key: str) -> str | None:
    """Compañía (normalizada) de un perfil: de su spec cruda (oficial) o de su spec guardada
    (usuario). ``None`` si el id es desconocido o la spec no la trae."""
    st = _state
    if key in st.specs:
        c = st.specs[key].get("company")
    else:
        try:
            c = (official()[0].get(key) or {}).get("company")
        except (OSError, ValueError):
            c = None
    return normalize_company(c) if isinstance(c, str) and c.strip() else None


def note_context(key: str) -> dict:
    """Contexto de notas del modelo de un perfil (``perfiles-v2``): ``{company, tags, base_nombre}``.

    ``tags``: ``usuario`` (perfil escrito por el usuario) y ``copia`` (perfil de usuario de
    ``futures``/``ladder`` cuya compañía difiere de la del oficial raíz: copia su mecánica).
    ``base_nombre`` = ``display_name`` de ese oficial raíz (solo con ``copia``)."""
    st = _state                                  # UNA lectura del estado
    meta = st.metas.get(key)
    company = company_of(key)
    tags: set[str] = set()
    base_nombre = None
    if meta is not None:
        tags.add("usuario")
        raws, _ = official()
        base = raws.get(meta["based_on"]) or {}
        cfg = st.cfgs[key]
        root_company = base.get("company")
        if cfg.funded.model in _LF and isinstance(root_company, str) \
                and company != normalize_company(root_company):
            tags.add("copia")
            base_nombre = str(base.get("display_name") or meta["based_on"])
    return {"company": company, "tags": tags, "base_nombre": base_nombre}


def catalog_fields(key: str) -> dict:
    """Campos que el catálogo añade a cada firma (``official``, ``based_on``, ``verified_on``,
    ``company``, ``hidden``)."""
    st = _state
    meta = st.metas.get(key)
    extra = {"company": company_of(key), "hidden": key in st.hidden}
    if meta is None:
        # PLAN-DESVIO: §2.3 dice "(None en los oficiales)" sobre los tres campos; se lee como
        # based_on = None y verified_on = None para los oficiales (``based_on(key)`` sí devuelve su id).
        return {"official": True, "based_on": None, "verified_on": None, **extra}
    return {"official": False, "based_on": meta["based_on"], "verified_on": meta["verified_on"],
            **extra}


def spec_sha256(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def snapshot_for(key: str) -> dict | None:
    """Instantanea del perfil de USUARIO para ``meta.profile_snapshot`` (None con oficiales)."""
    st = _state                       # UNA lectura del estado: meta y spec coherentes
    meta = st.metas.get(key)
    if meta is None:
        return None
    spec = copy.deepcopy(st.specs[key])
    return {"key": key, "based_on": meta["based_on"], "verified_on": meta["verified_on"],
            "spec": spec, "sha256": spec_sha256(spec)}


# ---------------------------------------------------------------------------
# Validacion de un perfil completo
# ---------------------------------------------------------------------------
@dataclass
class Check:
    errors: list[str]
    spec: dict | None = None            # spec normalizada (sin meta)
    cfg: PropFirmConfig | None = None


def check_meta(verified_on: Any, notes: Any) -> list[str]:
    errs = []
    if verified_on is not None:
        try:
            if not isinstance(verified_on, str):
                raise ValueError
            date.fromisoformat(verified_on)
        except ValueError:
            errs.append("verified_on: debe ser una fecha AAAA-MM-DD válida (o null)")
    if notes is not None and (not isinstance(notes, str) or len(notes) > _NOTES_MAX):
        errs.append(f"notes: debe ser un texto de máximo {_NOTES_MAX} caracteres (o null)")
    return errs


def validate_spec_body(key: str, spec: Any, root: dict, errs: list[str] | None = None,
                       *, template: bool = False) -> Check:
    """Nucleo de validacion COMPARTIDO por la API (``check_spec``) y la carga del fichero de usuario
    (``load_user_profiles``): diff contra la raiz sin ``meta``, lista blanca por camino (tipo y
    rango), ``parse_firm_spec`` y ``validate_runnable``. ``errs`` = errores previos del llamador
    (id, colision...), que se conservan. Devuelve la spec NORMALIZADA (enteros, texto recortado).
    ``template=True`` = la raíz es la de una plantilla (``editable_map(root, True)``). Un valor
    ``null`` en un camino editable difiere SIEMPRE de la raíz (que conserva valores reales) y
    sale como "falta el valor"."""
    errs = list(errs or [])
    if not isinstance(spec, dict):
        errs.append("spec debe ser un objeto")
        return Check(errs)
    model = root_model(root)
    emap = editable_map(root, template)
    fee_ccy = str(root.get("fee_ccy", "EUR")).upper()
    body = copy.deepcopy(strip_meta(spec))
    if template and body.get("fee_ccy") in _SUPPORTED_FEE_CCY:
        fee_ccy = body["fee_ccy"]
    changed = diff_paths(root, body)
    changed_paths = {c[0] for c in changed}
    for path, _, new in changed:
        rule = emap.get(path)
        if rule is None:
            errs.append(f"{path}: campo bloqueado ({blocked_reason(path, model)})")
            continue
        if new is _MISSING:
            errs.append(f"{path} ({_label(path, rule)}): no se puede quitar un campo")
            continue
        value, err = _check_value(path, rule, new, fee_ccy)
        if err:
            errs.append(err)
        else:
            _set(body, path, value)
    if errs:
        return Check(errs)
    for path in emap:                  # sin cambio de valor: tipo normalizado al de la raíz (4.0 -> 4)
        if path not in changed_paths:
            _set(body, path, _get(root, path))
    rel_errs = check_relations(body, model)
    if rel_errs:
        return Check(rel_errs)
    try:
        cfg = parse_firm_spec(key, body)
    except ValueError as exc:
        return Check([str(exc)])
    run_errs = validate_runnable(cfg)
    if run_errs:
        return Check(run_errs)
    return Check([], body, cfg)


def check_spec(key: Any, spec: Any, based: Any, *, editing: bool = False,
               template: Any = None) -> Check:
    """Validacion completa: id, colision, raiz y ``validate_spec_body``. ``editing=True`` permite
    que ``key`` ya exista entre los perfiles de usuario. ``template`` (id de plantilla): la raíz es
    ``template_root`` y ``based`` (si se da) debe ser su oficial base."""
    errs: list[str] = []
    off_raw, off_cfg = official()
    st = _state
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        errs.append("id inválido: debe cumplir ^[a-z0-9_]{3,40}$ (minúsculas, dígitos y _)")
    elif key in off_cfg:
        errs.append(f"el id '{key}' ya es un perfil oficial")
    elif not editing and key in st.cfgs:
        errs.append(f"el id '{key}' ya existe entre tus perfiles")
    if template is not None:
        try:
            tpl = get_template(template)
            root = template_root(template)
        except ValueError as exc:
            errs.append(str(exc))
            return Check(errs)
        if based is not None and based != tpl.base:
            errs.append(f"based_on '{based}' no coincide con la plantilla '{template}' "
                        f"(base {tpl.base})")
            return Check(errs)
        return validate_spec_body(key, spec, root, errs, template=True)
    if not isinstance(based, str) or based not in off_raw:
        errs.append(f"based_on '{based}' no es un perfil oficial")
        return Check(errs)
    return validate_spec_body(key, spec, off_raw[based], errs)


# ---------------------------------------------------------------------------
# Escritura (atomica) y recarga
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_user_file(doc: dict[str, Any]) -> None:
    path = USER_PROFILES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Perfiles de prop firm del USUARIO (gitignorado; lo escribe la vista Perfiles).\n"
              "# Mismo formato que propfirms.yaml + `meta` por perfil. No edites a mano con la app abierta.\n")
    out = {"firms": doc.get("firms") or {}, "hidden": list(doc.get("hidden") or [])}
    out.update({k: v for k, v in doc.items() if k not in ("firms", "hidden")})
    text = header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True,
                                   default_flow_style=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _reload() -> None:
    if _reload_hook is not None:
        _reload_hook()
    else:  # uso aislado (sin main)
        set_state(build_firms()[1])


def _entry(spec: dict, meta: dict) -> dict:
    return {"meta": meta, **spec}


def _mutate_doc(fn: Callable[[dict], None]) -> None:
    """Lee el DOCUMENTO crudo (``{firms, hidden}``; conserva entradas invalidas y ids ocultos
    desconocidos), aplica ``fn(doc)``, escribe y recarga, todo dentro de ``_FILE_LOCK``.

    Orden de locks: ``_FILE_LOCK`` -> ``_FIRMS_LOCK`` (la recarga de ``main`` toma el segundo);
    ningún camino los toma al revés. ``fn`` puede lanzar ``HTTPException`` antes de escribir
    (atómico: todo o nada)."""
    with _FILE_LOCK:
        try:
            doc = read_user_file()
        except ValueError as exc:
            raise HTTPException(
                409, f"{exc}. Arréglalo o bórralo antes de guardar más perfiles.")
        fn(doc)
        _write_user_file(doc)
        _reload()


# ---------------------------------------------------------------------------
# Entrada JSON estricta
# ---------------------------------------------------------------------------
def _reject_constant(name: str):
    raise ValueError(f"constante JSON no permitida: {name}")


def _parse_float(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f"número fuera de rango: {text}")
    return v


def parse_json_strict(raw: bytes | str) -> Any:
    return json.loads(raw, parse_constant=_reject_constant, parse_float=_parse_float)


async def _body(request: Request) -> dict:
    try:
        data = parse_json_strict(await request.body())
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, detail={"errors": [f"JSON inválido: {exc}"]})
    if not isinstance(data, dict):
        raise HTTPException(422, detail={"errors": ["el cuerpo debe ser un objeto JSON"]})
    return data


# ---------------------------------------------------------------------------
# Presentacion
# ---------------------------------------------------------------------------
def _kind(model: str) -> str:
    return "cfd" if model == "cycle" else "futures"


def _profile_view(key: str) -> dict:
    off_raw, off_cfg = official()
    st = _state
    if key in st.cfgs:
        cfg, spec, meta = st.cfgs[key], st.specs[key], st.metas[key]
        root, is_tpl = _root_of_meta(meta, off_raw)
        needs = root is None or bool(blocked_changes(root, spec, is_tpl))
        return {
            "key": key, "display_name": cfg.display_name, "official": False,
            "company": spec.get("company"), "hidden": key in st.hidden,
            "template": meta.get("template"),
            "based_on": meta["based_on"], "verified_on": meta["verified_on"],
            "notes": meta["notes"], "created": meta["created"], "updated": meta["updated"],
            "model": cfg.funded.model, "kind": _kind(cfg.funded.model),
            "ruleset_version": cfg.ruleset_version or None,
            "needs_recreate": needs, "spec": copy.deepcopy(spec),
        }
    cfg = off_cfg[key]
    return {
        "key": key, "display_name": cfg.display_name, "official": True, "based_on": None,
        "company": off_raw[key].get("company"), "hidden": key in st.hidden, "template": None,
        "verified_on": None, "notes": None, "created": None, "updated": None,
        "model": cfg.funded.model, "kind": _kind(cfg.funded.model),
        "ruleset_version": cfg.ruleset_version or None,
        "needs_recreate": False, "spec": copy.deepcopy(off_raw[key]),
    }


def _find(key: str) -> str:
    """'official' | 'user'; 404 si no existe."""
    _, off_cfg = official()
    if key in off_cfg:
        return "official"
    if key in _state.cfgs:
        return "user"
    raise HTTPException(404, f"perfil desconocido: '{key}'")


def _unique_key(base: str) -> str:
    """``<base>_<n>`` dentro del formato (se trunca la base) sin chocar con oficiales ni usuario."""
    _, off_cfg = official()
    n = 2
    while True:
        suffix = f"_{n}"
        cand = base[: 40 - len(suffix)] + suffix
        if KEY_RE.fullmatch(cand) and cand not in off_cfg and cand not in _state.cfgs:
            return cand
        n += 1


def _save_new(key: str, spec: dict, based: str, display_name: str | None, verified_on,
              notes, template: str | None = None) -> dict:
    spec = copy.deepcopy(spec)
    if display_name is not None:
        spec["display_name"] = display_name
    now = _now()
    meta = {"based_on": based, "template": template, "verified_on": verified_on, "notes": notes,
            "created": now, "updated": now}

    def fn(doc: dict) -> None:
        doc["firms"][key] = _entry(spec, meta)

    _mutate_doc(fn)
    return _profile_view(key)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("")
def list_profiles():
    # PLAN-DESVIO: §2.3 describe una lista, pero la enmienda 6 exige `errors` en este endpoint:
    # la respuesta es {"profiles": [...], "errors": [{key, error}]}.
    off_raw, off_cfg = official()
    keys = list(off_cfg) + list(_state.cfgs)
    return {"profiles": [_profile_view(k) for k in keys], "errors": get_errors()}


@router.get("/templates")
def list_templates():
    """Plantillas de "Nuevo perfil" desde cero (derivadas en cada llamada de la spec cruda actual)."""
    try:
        return [template_info(t) for t in TEMPLATES]
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/templates/{tid}")
def template_spec(tid: str):
    """``{spec con cada camino editable a null, schema (``example`` = valor del oficial base),
    n_fields, ...}`` de una plantilla."""
    if tid not in TEMPLATES:
        raise HTTPException(404, f"plantilla desconocida: '{tid}'")
    try:
        return template_detail(tid)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/visibility")
async def set_visibility(request: Request):
    """Oculta (``hidden: true``) o restaura (``false``) perfiles oficiales y de usuario CARGADOS.
    Atómico (todos o ninguno); ocultar no borra. 404 con ids desconocidos, 422 con lista vacía y
    409 si no quedara ningún perfil visible. Respuesta: ``{hidden: [ids ocultos existentes]}``.

    # PLAN-DESVIO: el plan solo dice `{hidden: [ids]}`; se devuelven los ocultos que EXISTEN (los ids
    # desconocidos que conserva el fichero no se listan). El 404 lleva `detail={errors, unknown}`."""
    body = await _body(request)
    keys, hidden = body.get("keys"), body.get("hidden")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise HTTPException(422, detail={"errors": ["keys: lista no vacía de ids de perfil"]})
    if not isinstance(hidden, bool):
        raise HTTPException(422, detail={"errors": ["hidden: debe ser true o false"]})
    keys = list(dict.fromkeys(keys))
    result: dict[str, list[str]] = {}

    def fn(doc: dict) -> None:
        _, off_cfg = official()
        valid = list(off_cfg) + [k for k in _state.cfgs if k not in off_cfg]
        unknown = [k for k in keys if k not in valid]
        if unknown:
            raise HTTPException(404, detail={
                "errors": [f"perfiles desconocidos: {', '.join(unknown)}"], "unknown": unknown})
        cur = list(doc["hidden"])
        new = cur + [k for k in keys if k not in cur] if hidden else [h for h in cur if h not in keys]
        if not [k for k in valid if k not in new]:
            raise HTTPException(409, "deja al menos un perfil visible")
        doc["hidden"] = new
        result["hidden"] = [h for h in new if h in valid]

    _mutate_doc(fn)
    return {"hidden": result["hidden"]}


@router.get("/{key}/schema")
def profile_schema(key: str):
    kind = _find(key)
    off_raw, _ = official()
    if kind == "official":
        return editable_fields(off_raw[key])
    root, is_tpl = _root_of_meta(_state.metas[key], off_raw)
    if root is None:
        raise HTTPException(409, "el perfil oficial base ya no existe; duplícalo de nuevo")
    return editable_fields(root, is_tpl)


@router.post("/validate")
async def validate_profile(request: Request):
    # PLAN-DESVIO: campo opcional `editing` (bool) en el cuerpo: sin él, validar el id de un perfil
    # de usuario YA guardado (edición en vivo) daría siempre "ya existe".
    body = await _body(request)
    tmpl = body.get("template")
    if tmpl is not None and body.get("based_on") is not None:
        return {"ok": False, "errors": ["usa 'based_on' (duplicar) o 'template' (desde cero), no ambos"]}
    res = check_spec(body.get("key"), body.get("spec"), body.get("based_on"),
                     editing=bool(body.get("editing", False)), template=tmpl)
    errs = list(res.errors)
    if "verified_on" in body or "notes" in body:
        errs += check_meta(body.get("verified_on"), body.get("notes"))
    return {"ok": not errs, "errors": errs}


@router.post("")
async def create_profile(request: Request):
    """Crea un perfil de usuario: DUPLICANDO (``base`` + ``key`` [+ ``display_name``]) o DESDE CERO
    (``template`` + ``key`` + ``spec`` completa [+ ``verified_on``, ``notes``]). Excluyentes.
    Con ``template``, ``company`` y ``display_name`` van DENTRO de ``spec`` (Enmienda v2 8)."""
    body = await _body(request)
    key, base, tmpl, name = body.get("key"), body.get("base"), body.get("template"), \
        body.get("display_name")
    if (base is None) == (tmpl is None):
        raise HTTPException(422, detail={"errors": [
            "usa 'base' (duplicar un perfil) o 'template' (crear desde cero), uno y solo uno"]})
    off_raw, off_cfg = official()
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise HTTPException(
            422, detail={"errors": ["id inválido: debe cumplir ^[a-z0-9_]{3,40}$"]})
    if key in off_cfg or key in _state.cfgs:
        raise HTTPException(409, f"el id '{key}' ya existe")
    if tmpl is not None:
        stray = [k for k in ("display_name", "company", "based_on") if k in body]
        if stray:
            raise HTTPException(422, detail={"errors": [
                f"{', '.join(stray)}: con 'template' van dentro de 'spec' (company y display_name) "
                "o se deducen de la plantilla"]})
        errs = check_meta(body.get("verified_on"), body.get("notes"))
        res = check_spec(key, body.get("spec"), None, template=tmpl)
        errs = res.errors + errs
        if errs:
            raise HTTPException(422, detail={"errors": errs})
        return _save_new(key, res.spec, TEMPLATES[tmpl].base, None, body.get("verified_on"),
                         body.get("notes"), template=tmpl)
    if not isinstance(base, str):
        raise HTTPException(422, detail={"errors": ["base: falta el id del perfil a duplicar"]})
    kind = _find(base)
    if name is not None and (not isinstance(name, str) or not name.strip()
                             or len(name.strip()) > _NAME_MAX):
        raise HTTPException(422, detail={"errors": [
            f"display_name: texto de 1 a {_NAME_MAX} caracteres"]})
    root_key = base if kind == "official" else _state.metas[base]["based_on"]
    if root_key not in off_raw:
        raise HTTPException(409, "el perfil oficial base ya no existe")
    base_tmpl = None if kind == "official" else _state.metas[base].get("template")
    src = off_raw[base] if kind == "official" else _state.specs[base]
    name = name.strip() if isinstance(name, str) else f"{src.get('display_name', base)} (tuyo)"
    res = check_spec(key, {**copy.deepcopy(src), "display_name": name}, root_key,
                     template=base_tmpl)
    if res.errors:
        raise HTTPException(422, detail={"errors": res.errors})
    return _save_new(key, res.spec, root_key, None, None, None, template=base_tmpl)


@router.put("/{key}")
async def update_profile(key: str, request: Request):
    kind = _find(key)
    if kind == "official":
        raise HTTPException(403, "los perfiles oficiales son de solo lectura: duplícalo")
    body = await _body(request)
    meta = _state.metas[key]
    off_raw, _ = official()
    root, is_tpl = _root_of_meta(meta, off_raw)
    if root is None or blocked_changes(root, _state.specs[key], is_tpl):
        raise HTTPException(422, detail={"errors": [
            "el perfil oficial base cambió; duplícalo de nuevo"]})
    errs = check_meta(body.get("verified_on"), body.get("notes"))
    res = check_spec(key, body.get("spec"), meta["based_on"], editing=True,
                     template=meta.get("template"))
    errs = res.errors + errs
    if errs:
        raise HTTPException(422, detail={"errors": errs})
    verified = body["verified_on"] if "verified_on" in body else meta["verified_on"]
    notes = body["notes"] if "notes" in body else meta["notes"]
    new_meta = {"based_on": meta["based_on"], "template": meta.get("template"),
                "verified_on": verified, "notes": notes,
                "created": meta["created"] or _now(), "updated": _now()}

    def fn(doc: dict) -> None:
        doc["firms"][key] = _entry(res.spec, new_meta)

    _mutate_doc(fn)
    return _profile_view(key)


@router.delete("/{key}")
def delete_profile(key: str):
    if _find(key) == "official":
        raise HTTPException(403, "los oficiales no se borran: ocúltalo")

    def fn(doc: dict) -> None:
        _, off_cfg = official()
        hidden = set(doc["hidden"])
        if not [k for k in list(off_cfg) + list(_state.cfgs) if k != key and k not in hidden]:
            raise HTTPException(409, "deja al menos un perfil visible")   # mismo criterio que /visibility
        doc["firms"].pop(key, None)
        doc["hidden"] = [h for h in doc["hidden"] if h != key]      # borrar lo quita de `hidden`

    _mutate_doc(fn)
    return {"deleted": key}


@router.get("/{key}/export")
def export_profile(key: str):
    _find(key)
    view = _profile_view(key)
    return {"format": EXPORT_FORMAT, "version": EXPORT_VERSION, "key": key,
            "based_on": view["based_on"] or key, "template": view["template"],
            "verified_on": view["verified_on"], "notes": view["notes"], "spec": view["spec"]}


@router.post("/import")
async def import_profile(request: Request):
    body = await _body(request)
    if body.get("format") != EXPORT_FORMAT:
        raise HTTPException(422, detail={"errors": [f"format debe ser '{EXPORT_FORMAT}'"]})
    ver = body.get("version")
    if isinstance(ver, bool) or not isinstance(ver, int) or ver != EXPORT_VERSION:
        raise HTTPException(422, detail={"errors": [f"version debe ser {EXPORT_VERSION}"]})
    key, based, spec = body.get("key"), body.get("based_on"), body.get("spec")
    tmpl = body.get("template")
    off_raw, off_cfg = official()
    if not isinstance(based, str) or based not in off_raw:
        raise HTTPException(422, detail={"errors": [
            f"based_on '{based}' no existe como perfil oficial en esta instalación"]})
    if tmpl is not None:
        try:
            root = template_root(tmpl)
        except ValueError as exc:
            raise HTTPException(422, detail={"errors": [str(exc)]})
        if based != TEMPLATES[tmpl].base:
            raise HTTPException(422, detail={"errors": [
                f"based_on '{based}' no coincide con la plantilla '{tmpl}' "
                f"(base {TEMPLATES[tmpl].base})"]})
    else:
        root = off_raw[based]
    spec = inherit_company(spec, root)          # export sin `company` (pre perfiles-v2): hereda
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise HTTPException(
            422, detail={"errors": ["id inválido: debe cumplir ^[a-z0-9_]{3,40}$"]})
    verified, notes = body.get("verified_on"), body.get("notes")
    meta_errs = check_meta(verified, notes)
    final = key
    if key in off_cfg or key in _state.cfgs:
        final = _unique_key(key)
    res = check_spec(final, spec, based, template=tmpl)
    errs = res.errors + meta_errs
    if errs:
        raise HTTPException(422, detail={"errors": errs})
    view = _save_new(final, res.spec, based, None, verified, notes, template=tmpl)
    return {"profile": view, "renamed_from": key if final != key else None,
            "message": (f"el id '{key}' ya existía: importado como '{final}'"
                        if final != key else "importado")}
