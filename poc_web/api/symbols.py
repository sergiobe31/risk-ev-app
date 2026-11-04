"""Fichas de símbolo por bróker (guardrail ``entrada-4-bloques``: F1 = modelo y API; F2 = la carga de patas
``.sqx`` de ``main.py`` lee estas fichas, y este módulo añade el alta guiada desde un ``.sqx``).

Una FICHA (:class:`SymbolSheet`) reúne lo que la app necesita saber de un símbolo de un bróker: valor del punto,
pasos de lote, comisión, spread típico y el fichero OHLC M1. Se identifica por ``sqx_symbol``, el nombre del
símbolo EXACTAMENTE como figura en ``SymbolInfo/@symbolName`` del ``.sqx``. Es único entre TODOS los brókers
(oficiales y del usuario) para que un ``.sqx`` resuelva una sola ficha; es una limitación declarada de la v1
(enmienda E7).

Concurrencia: ``_FILE_LOCK`` y ``_STATE_LOCK`` son locks INTRA-PROCESO. La app corre en un solo proceso
(``run_app.py``, uvicorn sin ``workers``), igual que ``profiles.py``; con varios procesos escribiendo el mismo
``config/symbols.user.yaml`` los locks no bastarían (la escritura sigue siendo atómica, pero una escritura podría
pisar los cambios de otra).

- OFICIALES (solo lectura): se LEEN con adaptadores de los ficheros de especificaciones que ya existen
  (``ftmo_cfd_costs.yaml``, ``ftmo_futures_costs.yaml``, ``broker_axi.yaml``) sin modificarlos. Calculan con las
  MISMAS expresiones y en el mismo orden que el resto de la app (``tick_value_usd / tick_size``,
  ``typical_spread_points * tick_size``): el test de paridad compara con ``==`` (enmienda E4).
- DEL USUARIO: ``config/symbols.user.yaml`` (``USER_SYMBOLS_PATH``, gitignorado, sustituible en tests) con
  ``{brokers: {...}, symbols: {<broker>: {<sqx_symbol>: {...}}}, hidden: ["<broker>/<sqx_symbol>", ...]}``. El
  usuario puede añadir símbolos a un bróker oficial o a uno suyo; NO puede redefinir uno oficial (solo ocultarlo).
- ``config/brokers.yaml`` (oficial, versionado) declara los brókers, su adaptador, el bróker de cada perfil
  oficial y el de cada lente.
- ``main.py`` es dueño de ``_leg_cache`` y avisa a este módulo con ``set_reload_hook`` (evita el import
  circular): tras cada escritura se invalida el registro y se llama al gancho.

Códigos HTTP de ``/api/symbols``: 400 = ficha o bróker inválido (campos, unicidad de ``sqx_symbol``, redefinir o
borrar una ficha oficial); 404 = bróker o ficha inexistente; 409 = el fichero de usuario está ilegible (no se
pisa); 422 = cuerpo que no es JSON estricto o no es un objeto. Los errores de contenido llevan
``detail={"errors": [...]}``.

Contrato de la API (rutas bajo ``/api/symbols``; ``sqx_symbol`` va codificado en la URL, p. ej. ``%40micro_oro``):

- ``GET ""`` -> ``{brokers: [Broker], symbols: [ficha + hidden], errors: [str], profiles: {perfil: bróker|null},
  lenses: {lente: bróker}}``. Las fichas ocultas salen con ``hidden: true``.
- ``POST ""`` (alta) con ``broker``, ``sqx_symbol`` y los campos de la ficha. ``PUT /{broker}/{sqx_symbol}`` es
  PARCIAL: lo no enviado se conserva y ``null`` borra un campo opcional (así se acepta un conflicto campo a
  campo, E11). ``DELETE`` solo de fichas de usuario. ``POST /visibility`` con ``{keys: ["<bróker>/<símbolo>"],
  hidden: bool}``. ``POST /brokers`` crea un bróker de usuario. ``GET /schema`` describe el formulario y
  ``POST /validate`` valida sin guardar (``editing: true`` = cuerpo parcial).
- ``GET /{broker}/{sqx_symbol}/export`` y ``POST /import``: sobre ``{format: "risk-app-symbols", version: 1,
  broker, sqx_symbol, broker_def, sheet}``; ``broker_def`` (solo brókers de usuario) permite crear el bróker.
- ``POST /mt5-sheet`` con ``{broker, text, mapping?}`` -> ``{broker, drafts}``: un borrador plano por fila (campos
  de ficha + ``guardable``, ``motivo``, ``faltan``, ``warnings``, ``existing``, ``editable``, ``conflicts``,
  ``candidates``). No guarda nada.
- ``GET /draft-from-sqx?path=`` -> un borrador plano (la misma forma) con lo que un ``.sqx`` del catálogo propone:
  ``pointValue``, ``orderSizeStep``, ``tickSize``, la comisión ``SizeBased`` y ``defaultSpread · tickSize``. Todo
  con procedencia ``sqx`` y en ``needs_review`` (el ``.sqx`` solo propone); el lote mínimo no está en el ``.sqx``
  y queda en ``faltan``. 400 si la ruta no está en el catálogo (``set_catalog_provider``). No guarda nada.

API interna que usa ``main.py``: :func:`load_registry`, :func:`resolve`, :func:`catalog_entry`,
:func:`broker_for_profile`, :func:`lens_broker`, :func:`sheet_signature`, :func:`reader_spec`,
:func:`parse_mt5_sheet`, :func:`draft_from_sqx`, :func:`set_reload_hook` y :func:`set_catalog_provider`.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

import yaml
from fastapi import APIRouter, HTTPException, Query, Request

from poc_web.api import profiles
from sqx.d1.extract_sqx import SQXFormatError
from sqx.sqx_trades import read_instrument_info
from src.broker_config import load_broker_config

ROOT = Path(__file__).resolve().parents[2]
BROKERS_PATH = ROOT / "config" / "brokers.yaml"
USER_SYMBOLS_PATH = ROOT / "config" / "symbols.user.yaml"  # sustituible en tests

SCHEMA_VERSION = 1
EXPORT_FORMAT = "risk-app-symbols"
EXPORT_VERSION = 1

BROKER_ID_RE = re.compile(r"^[a-z0-9_]{3,40}$")
SQX_SYMBOL_RE = re.compile(r"^[^\s/\\\x00-\x1f\x7f]{1,60}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ASSET_CLASSES = ("cfd", "futures")
CLOCKS = ("ny_plus_7", "ny_wall")
SOURCE_KINDS = ("official", "mt5_sheet", "sqx", "manual")
USER_SOURCE_KINDS = ("mt5_sheet", "sqx", "manual")      # "official" lo reservan los adaptadores
COMMISSION_KEYS = ("per_lot_rt", "pct_notional_rt")
# Campos de datos de una ficha, con procedencia (``sources``) y marca de revisión (``needs_review``).
DATA_FIELDS = ("ticker", "point_value_usd", "lot_step", "min_lot", "max_lot", "commission", "spread_px",
               "ohlc_m1", "tick_size")

_NAME_MAX = 80
_COMPANY_MAX = 40
_STR_MAX = 60
_NOTES_MAX = 2000
_NUM_MAX = 1.0e9
_LOT_MAX = 1.0e6
_LOT_MIN = 1.0e-8                   # cota inferior de lot_step y min_lot (un valor subnormal desborda el cociente)
_STEP_TOL = 1.0e-9                  # tolerancia de "min_lot es múltiplo de lot_step" (en pasos)
_CONFLICT_REL_TOL = 1.0e-12         # E11: igualdad de valores entre fuentes
_LABELS = {
    "sqx_symbol": "Símbolo SQX", "ticker": "Ticker en el bróker", "point_value_usd": "Valor del punto",
    "lot_step": "Paso de lote", "min_lot": "Lote mínimo", "max_lot": "Lote máximo",
    "commission": "Comisión", "spread_px": "Spread típico", "ohlc_m1": "Fichero OHLC M1",
    "tick_size": "Tamaño del tick", "verified_on": "Verificada el", "notes": "Notas",
}
# Claves que el cuerpo de una ficha puede traer además de los campos de datos.
_BODY_KEYS = frozenset({"broker", "sqx_symbol", "asset_class", "currency", "sources", "needs_review",
                        "verified_on", "notes", *DATA_FIELDS})
# Claves de solo lectura (las devuelven el listado y los borradores): se aceptan y se ignoran.
_ECHO_KEYS = frozenset({"official", "usable", "motivo", "hidden", "guardable", "conflicts", "editable",
                        "existing", "candidates", "faltan", "warnings", "sqx_symbol_suggested"})

# Etiquetas de los enumerados para el formulario (``GET /schema``).
_ENUM_LABELS = {
    "ny_plus_7": "Hora del servidor = Nueva York + 7 h", "ny_wall": "Reloj de pared de Nueva York",
    "cfd": "CFD", "futures": "Futuros",
    "official": "Oficial", "mt5_sheet": "Hoja de MT5", "sqx": "Del .sqx (sin verificar)", "manual": "Manual",
}

router = APIRouter(prefix="/api/symbols", tags=["symbols"])
_FILE_LOCK = threading.RLock()      # serializa lecturas-modificaciones-escrituras del fichero de usuario
_STATE_LOCK = threading.RLock()     # protege las cachés de este módulo (orden de locks: _FILE_LOCK -> _STATE_LOCK)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Broker:
    id: str
    display_name: str
    company: str
    currency: str
    asset_class: str                  # "cfd" | "futures"
    clock: str                        # "ny_plus_7" | "ny_wall"
    ohlc_dir: str | None = None
    official: bool = True

    def as_dict(self) -> dict:
        return {"id": self.id, "display_name": self.display_name, "company": self.company,
                "currency": self.currency, "asset_class": self.asset_class, "clock": self.clock,
                "ohlc_dir": self.ohlc_dir, "official": self.official}


@dataclass(frozen=True)
class SymbolSheet:
    broker: str
    sqx_symbol: str
    ticker: str
    asset_class: str
    point_value_usd: float
    lot_step: float
    min_lot: float
    max_lot: float | None = None
    commission: dict = field(default_factory=dict)       # {"per_lot_rt": x} | {"pct_notional_rt": x}
    spread_px: float | None = None                       # unidades de PRECIO (no pips ni puntos)
    ohlc_m1: str | None = None                           # nombre de fichero dentro de ``broker.ohlc_dir``
    tick_size: float | None = None
    sources: dict = field(default_factory=dict)          # {campo: {"from": ..., "date": ...}}
    needs_review: tuple = ()
    verified_on: str | None = None
    notes: str | None = None
    official: bool = False
    usable: bool = True
    motivo: str | None = None

    def as_dict(self) -> dict:
        return {
            "broker": self.broker, "sqx_symbol": self.sqx_symbol, "ticker": self.ticker,
            "asset_class": self.asset_class, "point_value_usd": self.point_value_usd,
            "lot_step": self.lot_step, "min_lot": self.min_lot, "max_lot": self.max_lot,
            "commission": dict(self.commission), "spread_px": self.spread_px, "ohlc_m1": self.ohlc_m1,
            "tick_size": self.tick_size, "sources": copy.deepcopy(self.sources),
            "needs_review": list(self.needs_review), "verified_on": self.verified_on, "notes": self.notes,
            "official": self.official, "usable": self.usable, "motivo": self.motivo,
        }


@dataclass
class Registry:
    brokers: dict[str, Broker] = field(default_factory=dict)
    sheets: dict[str, SymbolSheet] = field(default_factory=dict)     # clave = sqx_symbol (único)
    hidden: set[str] = field(default_factory=set)                    # "<broker>/<sqx_symbol>"
    errors: list[str] = field(default_factory=list)
    profiles: dict[str, str | None] = field(default_factory=dict)    # perfil OFICIAL raíz -> bróker | None
    lenses: dict[str, str] = field(default_factory=dict)
    user_file_error: str | None = None                               # fichero de usuario ilegible


def _hidden_key(broker: str, sqx_symbol: str) -> str:
    return f"{broker}/{sqx_symbol}"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    return v.isoformat() if isinstance(v, (date, datetime)) else str(v)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _resolve_file(name: Any) -> Path:
    p = Path(str(name))
    return p if p.is_absolute() else ROOT / p


def _fmt(x: Any) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_CONFLICT_REL_TOL, abs_tol=0.0)


# ---------------------------------------------------------------------------
# Adaptadores oficiales (E4: mismas expresiones y mismo orden que hoy)
# ---------------------------------------------------------------------------
def _official_sources(sheet_fields: dict, when: str | None) -> dict:
    return {f: {"from": "official", "date": when} for f in DATA_FIELDS if sheet_fields.get(f) not in (None, {})}


def _costs_sheet(broker: Broker, key: str, spec: Any, policy: dict, file_class: str) -> SymbolSheet:
    """Ficha de una entrada ``symbols.<key>`` de un YAML de costes (``ftmo_*_costs.yaml``)."""
    if not isinstance(spec, dict) or not spec.get("sqx_symbol"):
        raise ValueError("la entrada no tiene sqx_symbol")
    sqx_symbol = str(spec["sqx_symbol"])
    tick_size = float(spec["tick_size"])                     # como _spec_float de sqx/sqx_trades.py
    tick_value = float(spec["tick_value_usd"])
    if tick_size <= 0:
        raise ValueError("tick_size <= 0")
    point_value_usd = tick_value / tick_size                 # como pv_yaml de sqx/sqx_trades.py
    if spec.get("commission_pct") is not None:               # misma precedencia que read_trade_table
        if file_class != "cfd":
            raise ValueError("commission_pct solo vale en CFD")
        commission = {"pct_notional_rt": float(spec["commission_pct"])}
    else:
        commission = {"per_lot_rt": float(spec["total_round_trip"])}
    if file_class == "cfd":
        lot_step, min_lot = float(spec["lot_step"]), float(spec["min_lot"])
    else:
        lot_step = min_lot = 1.0
    pts = (policy.get(sqx_symbol) or {}).get("typical_spread_points")
    # CFD: ``typical_spread_points`` va en pips del instrumento SQX, así que ``spread_px = pts · tick_size`` (la
    # expresión de main._sqx_leg_minutes). Futuros: en ``ftmo_futures_costs.yaml`` ya va en UNIDADES DE PRECIO
    # (cabecera de ``sqx_policy``: MNQ 0,25 = 1 tick), de modo que ``spread_px = pts`` (corrección F2 de la
    # revisión de código de la F1, hallazgo H6; el producto daba 0,0625 para MNQ).
    if pts is None:
        spread_px = None
    elif file_class == "futures":
        spread_px = float(pts)
    else:
        spread_px = float(pts) * tick_size
    ohlc_m1 = str(spec["ohlc_m1"]) if spec.get("ohlc_m1") else None
    data = {"ticker": key, "point_value_usd": point_value_usd, "lot_step": lot_step, "min_lot": min_lot,
            "commission": commission, "spread_px": spread_px, "ohlc_m1": ohlc_m1, "tick_size": tick_size}
    return SymbolSheet(
        broker=broker.id, sqx_symbol=sqx_symbol, ticker=key, asset_class=broker.asset_class,
        point_value_usd=point_value_usd, lot_step=lot_step, min_lot=min_lot, max_lot=None,
        commission=commission, spread_px=spread_px, ohlc_m1=ohlc_m1, tick_size=tick_size,
        sources=_official_sources(data, None), needs_review=(), official=True)


def _adapter_costs_yaml(broker: Broker, src: dict, errors: list[str]) -> list[SymbolSheet]:
    path = _resolve_file(src["file"])
    try:
        costs = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        errors.append(f"{broker.id}: no se pudo leer {path.name}: {exc}")
        return []
    if not isinstance(costs, dict):
        errors.append(f"{broker.id}: {path.name} no es un diccionario")
        return []
    file_class = str(costs.get("asset_class", "futures"))
    if file_class != broker.asset_class:
        errors.append(f"{broker.id}: {path.name} es de clase '{file_class}' y el bróker declara "
                      f"'{broker.asset_class}'")
        return []
    policy = costs.get("sqx_policy") or {}
    out: list[SymbolSheet] = []
    for key, spec in (costs.get("symbols") or {}).items():
        try:
            out.append(_costs_sheet(broker, str(key), spec, policy, file_class))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{broker.id}/{key}: {path.name}: {type(exc).__name__}: {exc}")
    return out


def _adapter_broker_axi(broker: Broker, src: dict, errors: list[str]) -> list[SymbolSheet]:
    path = _resolve_file(src["file"])
    suffix = str(src.get("sqx_suffix") or "")
    try:
        cfg = load_broker_config(path)
    except Exception as exc:  # noqa: BLE001 - Pydantic, YAML o disco: el registro lo lista, no rompe
        errors.append(f"{broker.id}: no se pudo leer {path.name}: {type(exc).__name__}: {exc}")
        return []
    when = _iso(cfg.captured_at)
    out: list[SymbolSheet] = []
    for s in cfg.symbols:
        # Aislado por símbolo, como el adaptador de costes: un símbolo defectuoso va a ``errors`` y el resto carga.
        label = getattr(s, "ticker_axi", "?")
        try:
            if not s.ticker_cascade:
                errors.append(f"{broker.id}/{label}: sin ticker_cascade no tiene nombre SQX: se omite")
                continue
            out.append(_axi_sheet(broker, s, suffix, when))
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de UN símbolo se lista; no tumba el registro
            errors.append(f"{broker.id}/{label}: {path.name}: {type(exc).__name__}: {exc}")
    return out


def _axi_sheet(broker: Broker, s: Any, suffix: str, when: str | None) -> SymbolSheet:
    """Ficha de un ``BrokerSymbol`` de ``broker_axi.yaml`` (sin tocar el modelo ni el fichero)."""
    sqx_symbol = s.ticker_cascade + suffix
    usable = str(s.quote_currency).upper() == "USD"
    commission = {"per_lot_rt": s.commission_round_trip_usd()}          # 2 * comisión por lado
    spread_px = s.typical_spread_points * s.tick_size
    ohlc_m1 = f"{sqx_symbol}.csv"
    # ``tick_size`` queda sin rellenar a propósito: el .sqx de AXI trae el PIP como tickSize (EURUSD 0,0001) y la
    # ficha de AXI guarda el tick de 5 decimales (0,00001); compararlos daría un aviso falso en cada pata de FX.
    data = {"ticker": s.ticker_axi, "point_value_usd": s.point_value_usd, "lot_step": s.lot_step,
            "min_lot": s.min_lot, "max_lot": s.max_lot, "commission": commission,
            "spread_px": spread_px, "ohlc_m1": ohlc_m1}
    return SymbolSheet(
        broker=broker.id, sqx_symbol=sqx_symbol, ticker=s.ticker_axi, asset_class=broker.asset_class,
        point_value_usd=float(s.point_value_usd), lot_step=float(s.lot_step), min_lot=float(s.min_lot),
        max_lot=float(s.max_lot), commission=commission, spread_px=spread_px, ohlc_m1=ohlc_m1,
        tick_size=None, sources=_official_sources(data, when), needs_review=(), official=True,
        usable=usable, motivo=None if usable else "divisa no USD (v1)")


_ADAPTERS: dict[str, Callable[[Broker, dict, list[str]], list[SymbolSheet]]] = {
    "costs_yaml": _adapter_costs_yaml,
    "broker_axi": _adapter_broker_axi,
}


# ---------------------------------------------------------------------------
# Validación (compartida por la API y por la carga del fichero de usuario)
# ---------------------------------------------------------------------------
def _clean_text(v: Any, name: str, maxlen: int, errs: list[str]) -> str | None:
    if not isinstance(v, str) or not v.strip():
        errs.append(f"{name}: debe ser un texto no vacío")
        return None
    v = " ".join(v.split())
    if len(v) > maxlen:
        errs.append(f"{name}: máximo {maxlen} caracteres")
        return None
    return v


def _num_field(errs: list[str], name: str, v: Any, *, label: str | None = None, gt: float | None = None,
               ge: float | None = None, hi: float | None = None) -> float | None:
    tag = f"{name} ({label or _LABELS.get(name, name)})"
    if not _is_num(v):
        errs.append(f"{tag}: debe ser un número (no texto ni booleano)")
        return None
    try:
        x = float(v)
    except OverflowError:                       # un entero JSON gigantesco
        errs.append(f"{tag}: número fuera de rango")
        return None
    if not math.isfinite(x):
        errs.append(f"{tag}: debe ser un número finito (NaN e Infinity no valen)")
        return None
    if gt is not None and not x > gt:
        errs.append(f"{tag}: debe ser mayor que {gt:g}")
        return None
    if ge is not None and not x >= ge:
        errs.append(f"{tag}: debe ser mayor o igual que {ge:g}")
        return None
    if hi is not None and x > hi:
        errs.append(f"{tag}: debe ser menor o igual que {hi:g}")
        return None
    return x


def _check_date(v: Any, name: str, errs: list[str]) -> str | None:
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    if isinstance(v, str) and ISO_DATE_RE.fullmatch(v):
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            pass
    errs.append(f"{name}: debe ser una fecha AAAA-MM-DD válida (o null)")
    return None


def _check_fields(m: dict, broker: Broker, errs: list[str]) -> dict:
    """Valida y normaliza los campos de datos (``m`` = campos ya fusionados). Devuelve solo los presentes."""
    out: dict[str, Any] = {}
    if m.get("ticker") is not None:
        t = _clean_text(m["ticker"], "ticker (Ticker en el bróker)", _STR_MAX, errs)
        if t is not None:
            out["ticker"] = t
    # valor del punto
    if m.get("point_value_usd") is None:
        errs.append("point_value_usd (Valor del punto): falta el valor")
    else:
        x = _num_field(errs, "point_value_usd", m["point_value_usd"], gt=0.0, hi=_NUM_MAX)
        if x is not None:
            out["point_value_usd"] = x
    # pasos de lote
    if broker.asset_class == "futures":
        for name in ("lot_step", "min_lot"):
            if m.get(name) is None:
                out[name] = 1.0
                continue
            x = _num_field(errs, name, m[name], gt=0.0)
            if x is None:
                continue
            if x != 1.0:
                errs.append(f"{name} ({_LABELS[name]}): en un bróker de futuros vale 1 (fijo), no {_fmt(x)}")
            else:
                out[name] = 1.0
    else:
        for name in ("lot_step", "min_lot"):
            if m.get(name) is None:
                errs.append(f"{name} ({_LABELS[name]}): falta el valor (obligatorio en CFD)")
                continue
            x = _num_field(errs, name, m[name], ge=_LOT_MIN, hi=_LOT_MAX)
            if x is not None:
                out[name] = x
        if "lot_step" in out and "min_lot" in out:
            ratio = out["min_lot"] / out["lot_step"]
            if abs(ratio - round(ratio)) > _STEP_TOL:
                errs.append(f"min_lot (Lote mínimo): {_fmt(out['min_lot'])} no es múltiplo de lot_step "
                            f"({_fmt(out['lot_step'])})")
            elif out["min_lot"] < out["lot_step"] * (1.0 - _STEP_TOL):
                # H4: ni siquiera "múltiplo" vale si el mínimo no llega a un paso (un múltiplo 0)
                errs.append(f"min_lot (Lote mínimo): {_fmt(out['min_lot'])} es menor que un paso de lote "
                            f"(lot_step {_fmt(out['lot_step'])})")
    if m.get("max_lot") is not None:
        x = _num_field(errs, "max_lot", m["max_lot"], gt=0.0, hi=_LOT_MAX)
        if x is not None:
            if "min_lot" in out and x < out["min_lot"] - 1e-12:
                errs.append(f"max_lot (Lote máximo): {_fmt(x)} es menor que min_lot ({_fmt(out['min_lot'])})")
            else:
                out["max_lot"] = x
    # comisión: exactamente una
    c = m.get("commission")
    if not isinstance(c, dict) or not c:
        errs.append("commission (Comisión): falta; indica exactamente una de per_lot_rt (USD por lote, ida y "
                    "vuelta) o pct_notional_rt (% del nocional, ida y vuelta); sin comisión, per_lot_rt = 0")
    else:
        unknown = sorted(str(k) for k in c if k not in COMMISSION_KEYS)
        if unknown:
            errs.append(f"commission (Comisión): clave desconocida {unknown}; usa per_lot_rt o pct_notional_rt")
        elif len(c) != 1:
            errs.append("commission (Comisión): un solo tipo de comisión a la vez (per_lot_rt o "
                        "pct_notional_rt), no los dos")
        else:
            kind, raw = next(iter(c.items()))
            if kind == "pct_notional_rt":
                if broker.asset_class != "cfd":
                    errs.append("commission (Comisión): pct_notional_rt solo vale en CFD")
                else:
                    x = _num_field(errs, "commission.pct_notional_rt", raw, label="% del nocional, ida y vuelta",
                                   ge=0.0, hi=100.0)
                    if x is not None:
                        out["commission"] = {kind: x}
            else:
                x = _num_field(errs, "commission.per_lot_rt", raw, label="USD por lote, ida y vuelta",
                               ge=0.0, hi=_LOT_MAX)
                if x is not None:
                    out["commission"] = {kind: x}
    if m.get("spread_px") is not None:
        x = _num_field(errs, "spread_px", m["spread_px"], ge=0.0, hi=_NUM_MAX)
        if x is not None:
            out["spread_px"] = x
    if m.get("ohlc_m1") is not None:
        v = m["ohlc_m1"]
        if not isinstance(v, str) or not v.strip() or v != v.strip() or len(v) > 120 \
                or "/" in v or "\\" in v or v in (".", "..") or v.startswith("."):
            errs.append("ohlc_m1 (Fichero OHLC M1): debe ser el NOMBRE de un fichero (sin carpetas) dentro de "
                        "la carpeta de OHLC del bróker")
        elif not broker.ohlc_dir:
            errs.append(f"ohlc_m1 (Fichero OHLC M1): el bróker '{broker.id}' no tiene carpeta de OHLC (ohlc_dir)")
        else:
            out["ohlc_m1"] = v
    if m.get("tick_size") is not None:
        x = _num_field(errs, "tick_size", m["tick_size"], gt=0.0, hi=_NUM_MAX)
        if x is not None:
            out["tick_size"] = x
    return out


def _check_source(v: Any, name: str, errs: list[str]) -> dict | None:
    if not isinstance(v, dict) or set(v) - {"from", "date"} or v.get("from") not in USER_SOURCE_KINDS:
        errs.append(f"sources.{name}: debe ser {{from: {' | '.join(USER_SOURCE_KINDS)}, date: AAAA-MM-DD o null}}")
        return None
    when = _check_date(v.get("date"), f"sources.{name}.date", errs)
    return {"from": v["from"], "date": when}


def _build_entry(body: dict, broker: Broker, *, base: dict | None, default_date: str | None,
                 errs: list[str]) -> dict | None:
    """Entrada normalizada de una ficha (lo que se guarda en el YAML de usuario) o ``None`` con ``errs``.

    ``body`` = cuerpo de la petición (o entrada del fichero); ``base`` = entrada vigente al EDITAR: los
    campos ausentes del cuerpo se conservan y un ``null`` explícito borra un campo opcional (PUT parcial,
    E11). ``default_date`` = fecha de los campos manuales nuevos (``None`` al leer el fichero)."""
    unknown = sorted(str(k) for k in body if k not in _BODY_KEYS and k not in _ECHO_KEYS)
    if unknown:
        errs.append(f"campos no reconocidos: {', '.join(unknown)}")
    cur = body.get("currency")
    if cur is not None and cur != "USD":
        errs.append(f"currency: solo USD en la v1 (recibido {cur!r}): la cuenta debe estar en USD")
    ac = body.get("asset_class")
    if ac is not None and ac != broker.asset_class:
        errs.append(f"asset_class: la hereda del bróker ('{broker.id}' es {broker.asset_class}), no {ac!r}")
    merged = copy.deepcopy(base) if base else {}
    touched: set[str] = set()
    for f in DATA_FIELDS:
        if f in body:
            if not base or body[f] != base.get(f):
                touched.add(f)
            merged[f] = body[f]
    fields = _check_fields(merged, broker, errs)
    verified = _check_date(body["verified_on"] if "verified_on" in body else (base or {}).get("verified_on"),
                           "verified_on", errs)
    notes = body["notes"] if "notes" in body else (base or {}).get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes) > _NOTES_MAX):
        errs.append(f"notes (Notas): debe ser un texto de máximo {_NOTES_MAX} caracteres (o null)")
        notes = None
    # procedencia
    body_src = body.get("sources")
    if body_src is not None and not isinstance(body_src, dict):
        errs.append("sources: debe ser un objeto {campo: {from, date}}")
        body_src = None
    for k in (body_src or {}):
        if k not in DATA_FIELDS:
            errs.append(f"sources.{k}: campo desconocido")
    base_src = (base or {}).get("sources") or {}
    sources: dict[str, dict] = {}
    for f in DATA_FIELDS:
        if f not in fields:
            continue
        if body_src and f in body_src:
            chk = _check_source(body_src[f], f, errs)
            if chk is not None:
                sources[f] = chk
        elif f in touched or f not in base_src:
            sources[f] = {"from": "manual", "date": default_date}
        else:
            sources[f] = copy.deepcopy(base_src[f])
    # campos sin verificar
    if "needs_review" in body:
        nr = body["needs_review"]
        if not isinstance(nr, list) or not all(isinstance(x, str) for x in nr):
            errs.append("needs_review: debe ser una lista de nombres de campo")
            nr = []
        bad = sorted({x for x in nr if x not in DATA_FIELDS})
        if bad:
            errs.append(f"needs_review: campos desconocidos {bad}")
        needs = [x for x in dict.fromkeys(nr) if x in DATA_FIELDS]
    else:
        needs = [x for x in (base or {}).get("needs_review", [])
                 if x in DATA_FIELDS and not (x in touched and sources.get(x, {}).get("from") == "manual")]
    if errs:
        return None
    entry: dict[str, Any] = dict(fields)
    entry["sources"] = sources
    entry["needs_review"] = needs
    if verified is not None:
        entry["verified_on"] = verified
    if notes is not None:
        entry["notes"] = notes
    return entry


def _sheet_from_entry(broker: Broker, sqx_symbol: str, entry: dict) -> SymbolSheet:
    return SymbolSheet(
        broker=broker.id, sqx_symbol=sqx_symbol, ticker=entry.get("ticker") or sqx_symbol,
        asset_class=broker.asset_class, point_value_usd=entry["point_value_usd"], lot_step=entry["lot_step"],
        min_lot=entry["min_lot"], max_lot=entry.get("max_lot"), commission=dict(entry["commission"]),
        spread_px=entry.get("spread_px"), ohlc_m1=entry.get("ohlc_m1"), tick_size=entry.get("tick_size"),
        sources=copy.deepcopy(entry["sources"]), needs_review=tuple(entry.get("needs_review") or ()),
        verified_on=entry.get("verified_on"), notes=entry.get("notes"), official=False, usable=True, motivo=None)


def _check_sqx_symbol(v: Any) -> str | None:
    if not isinstance(v, str) or not SQX_SYMBOL_RE.fullmatch(v) or v in (".", ".."):
        return ("sqx_symbol (Símbolo SQX): 1 a 60 caracteres sin espacios ni barras; debe ser igual, carácter "
                "a carácter, que el símbolo del .sqx")
    return None


def _dup_message(sqx_symbol: str, existing: SymbolSheet) -> str:
    """E7: la unicidad de ``sqx_symbol`` entre brókers es una limitación de la v1; el error dice qué hacer."""
    msg = (f"el símbolo SQX '{sqx_symbol}' ya tiene ficha en el bróker '{existing.broker}'; si es otro "
           f"bróker, dale a sus datos en SQX un nombre propio (p. ej. {sqx_symbol}_mibroker)")
    if existing.official:
        msg += ". Las fichas oficiales no se editan: si no la quieres, ocúltala"
    return msg


def _parse_broker(bid: Any, d: Any, *, official: bool) -> Broker:
    """Bróker de ``brokers.yaml`` o del fichero de usuario; ``ValueError`` con todos los errores."""
    errs: list[str] = []
    if not isinstance(bid, str) or not BROKER_ID_RE.fullmatch(bid):
        errs.append("id inválido: debe cumplir ^[a-z0-9_]{3,40}$")
    if not isinstance(d, dict):
        raise ValueError("; ".join(errs + ["la definición no es un diccionario"]))
    allowed = {"display_name", "company", "currency", "asset_class", "clock", "ohlc_dir"}
    if official:
        allowed |= {"source"}
    extra = sorted(str(k) for k in d if k not in allowed)
    if extra:
        errs.append(f"claves no reconocidas: {', '.join(extra)}")
    name = _clean_text(d.get("display_name"), "display_name", _NAME_MAX, errs)
    company = _clean_text(d.get("company"), "company", _COMPANY_MAX, errs)
    currency = d.get("currency", "USD")
    if currency != "USD":
        errs.append(f"currency: solo USD en la v1 (recibido {currency!r})")
    ac, clock = d.get("asset_class"), d.get("clock")
    if ac not in ASSET_CLASSES:
        errs.append(f"asset_class: debe ser uno de {', '.join(ASSET_CLASSES)}")
    if clock not in CLOCKS:
        errs.append(f"clock: debe ser uno de {', '.join(CLOCKS)}")
    ohlc_dir = d.get("ohlc_dir")
    if ohlc_dir is not None:
        p = Path(str(ohlc_dir)) if isinstance(ohlc_dir, str) else None
        # absoluta en CUALQUIER sistema: en Windows «/datos» no es is_absolute() (no tiene unidad) pero tiene raíz
        if p is None or not ohlc_dir.strip() or PurePosixPath(ohlc_dir).is_absolute() or PureWindowsPath(ohlc_dir).anchor \
                or ".." in p.parts or "\\" in ohlc_dir:
            errs.append("ohlc_dir: debe ser una ruta relativa a la raíz del repo, sin '..'")
    if errs:
        raise ValueError("; ".join(errs))
    return Broker(id=bid, display_name=name, company=company, currency="USD", asset_class=ac, clock=clock,
                  ohlc_dir=str(ohlc_dir) if ohlc_dir is not None else None, official=official)


# ---------------------------------------------------------------------------
# Fichero de usuario
# ---------------------------------------------------------------------------
def read_user_file() -> dict[str, Any]:
    """DOCUMENTO del fichero de usuario: ``{"brokers": {...}, "symbols": {broker: {sqx: entrada}}, "hidden":
    [...]}`` (más las demás claves de primer nivel). Falta -> documento vacío; ilegible o de forma
    inesperada -> ``ValueError``."""
    name = USER_SYMBOLS_PATH.name
    empty: dict[str, Any] = {"brokers": {}, "symbols": {}, "hidden": []}
    if not USER_SYMBOLS_PATH.exists():
        return empty
    try:
        raw = yaml.safe_load(USER_SYMBOLS_PATH.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} ilegible: {exc}") from exc
    if raw is None:
        return empty
    if not isinstance(raw, dict):
        raise ValueError(f"{name} ilegible: se esperaba un diccionario con 'brokers:', 'symbols:' y 'hidden:'")
    brokers, symbols, hidden = raw.get("brokers"), raw.get("symbols"), raw.get("hidden")
    if brokers is not None and not isinstance(brokers, dict):
        raise ValueError(f"{name} ilegible: 'brokers' debe ser un diccionario")
    if symbols is not None and (not isinstance(symbols, dict)
                                or not all(isinstance(v, (dict, type(None))) for v in symbols.values())):
        raise ValueError(f"{name} ilegible: 'symbols' debe ser {{bróker: {{símbolo SQX: ficha}}}}")
    if hidden is not None and (not isinstance(hidden, list) or not all(isinstance(h, str) for h in hidden)):
        raise ValueError(f"{name} ilegible: 'hidden' debe ser una lista de '<bróker>/<símbolo SQX>'")
    rest = {k: v for k, v in raw.items() if k not in ("brokers", "symbols", "hidden")}
    return {"brokers": dict(brokers or {}), "symbols": {b: dict(s or {}) for b, s in (symbols or {}).items()},
            "hidden": list(dict.fromkeys(hidden or [])), **rest}


def _write_user_file(doc: dict[str, Any]) -> None:
    """Escritura atómica: fichero temporal completo + ``os.replace`` (nunca queda a medias)."""
    path = USER_SYMBOLS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Fichas de símbolo y brókers del USUARIO (gitignorado; los escribe la vista Símbolos).\n"
              "# No edites a mano con la app abierta.\n")
    out: dict[str, Any] = {
        "brokers": doc.get("brokers") or {},
        "symbols": {b: s for b, s in (doc.get("symbols") or {}).items() if s},
        "hidden": list(doc.get("hidden") or []),
    }
    out.update({k: v for k, v in doc.items() if k not in ("brokers", "symbols", "hidden")})
    text = header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, default_flow_style=False)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"no se pudo escribir {path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Registro (cacheado por mtime)
# ---------------------------------------------------------------------------
_doc_cache: dict[str, Any] = {"stamp": None, "doc": None, "error": None}
_reg_cache: dict[str, Any] = {"stamp": None, "registry": None}
_reload_hook: Callable[[], Any] | None = None


_catalog_provider: Callable[[], Any] | None = None


def set_reload_hook(fn: Callable[[], Any] | None) -> None:
    global _reload_hook
    _reload_hook = fn


def set_catalog_provider(fn: Callable[[], Any] | None) -> None:
    """``main`` registra aquí una función que devuelve las rutas ``.sqx`` de su catálogo (cadenas, relativas a la
    raíz del repo o absolutas). ``draft-from-sqx`` solo lee rutas de ese catálogo (nunca ficheros arbitrarios);
    así este módulo no importa ``main``."""
    global _catalog_provider
    _catalog_provider = fn


def _official_doc() -> tuple[dict | None, str | None]:
    """``(documento, error)`` de ``brokers.yaml``; cacheado por (ruta, mtime)."""
    with _STATE_LOCK:
        stamp = (str(BROKERS_PATH), _mtime_ns(BROKERS_PATH))
        if _doc_cache["stamp"] != stamp:
            doc, err = None, None
            try:
                doc = yaml.safe_load(BROKERS_PATH.read_text(encoding="utf-8"))
                if not isinstance(doc, dict):
                    doc, err = None, f"{BROKERS_PATH.name}: se esperaba un diccionario"
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                err = f"{BROKERS_PATH.name} ilegible: {exc}"
            _doc_cache.update(stamp=stamp, doc=doc, error=err)
        return _doc_cache["doc"], _doc_cache["error"]


def _adapter_files(doc: dict | None) -> list[Path]:
    out: list[Path] = []
    brokers = (doc or {}).get("brokers")
    for bdef in (brokers.values() if isinstance(brokers, dict) else ()):
        src = bdef.get("source") if isinstance(bdef, dict) else None
        if isinstance(src, dict) and src.get("file"):
            out.append(_resolve_file(src["file"]))
    return out


def _stamp() -> tuple:
    doc, _ = _official_doc()
    return (str(BROKERS_PATH), _mtime_ns(BROKERS_PATH), str(USER_SYMBOLS_PATH), _mtime_ns(USER_SYMBOLS_PATH),
            tuple((str(p), _mtime_ns(p)) for p in _adapter_files(doc)))


def _load_official(doc: dict, reg: Registry) -> None:
    errors = reg.errors
    name = BROKERS_PATH.name
    if doc.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{name}: schema_version {doc.get('schema_version')!r} no soportada "
                      f"(esperada {SCHEMA_VERSION})")
        return
    raw_brokers = doc.get("brokers")
    if not isinstance(raw_brokers, dict) or not raw_brokers:
        errors.append(f"{name}: falta 'brokers:' con al menos un bróker")
        return
    for bid, bdef in raw_brokers.items():
        try:
            broker = _parse_broker(bid, bdef, official=True)
            src = bdef.get("source")
            if not isinstance(src, dict) or src.get("adapter") not in _ADAPTERS or not src.get("file"):
                raise ValueError(f"source: adapter debe ser uno de {', '.join(_ADAPTERS)} y llevar 'file'")
        except ValueError as exc:
            errors.append(f"{name}: bróker '{bid}': {exc}")
            continue
        reg.brokers[broker.id] = broker
        for sheet in _ADAPTERS[src["adapter"]](broker, src, errors):
            prev = reg.sheets.get(sheet.sqx_symbol)
            if prev is not None:
                errors.append(f"símbolo SQX '{sheet.sqx_symbol}' repetido entre brókers oficiales "
                              f"('{prev.broker}' y '{broker.id}'): se ignora el segundo")
                continue
            reg.sheets[sheet.sqx_symbol] = sheet
    for section, target in (("profiles", reg.profiles), ("lenses", reg.lenses)):
        raw = doc.get(section) or {}
        if not isinstance(raw, dict):
            errors.append(f"{name}: '{section}' debe ser un diccionario")
            continue
        for key, bid in raw.items():
            # un perfil puede apuntar a ``null`` (sin bróker para patas .sqx); una lente, no
            if (bid is None and section == "profiles") or (isinstance(bid, str) and bid in reg.brokers):
                target[str(key)] = bid
            else:
                errors.append(f"{name}: {section}['{key}'] apunta a un bróker desconocido: {bid!r}")


def _load_user(reg: Registry) -> None:
    errors = reg.errors
    try:
        doc = read_user_file()
    except ValueError as exc:
        reg.user_file_error = str(exc)
        errors.append(str(exc))
        return
    name = USER_SYMBOLS_PATH.name
    reg.hidden = set(doc["hidden"])
    for bid, bdef in doc["brokers"].items():
        try:
            if bid in reg.brokers:
                raise ValueError("el id coincide con un bróker oficial o repetido (se omite el tuyo)")
            reg.brokers[bid] = _parse_broker(bid, bdef, official=False)
        except ValueError as exc:
            errors.append(f"{name}: bróker '{bid}': {exc}")
        except Exception as exc:  # noqa: BLE001 - un bróker defectuoso no tumba el registro (H3)
            errors.append(f"{name}: bróker '{bid}': {type(exc).__name__}: {exc}")
    for bid, entries in doc["symbols"].items():
        broker = reg.brokers.get(bid)
        if broker is None:
            errors.append(f"{name}: fichas del bróker '{bid}': el bróker no existe o no es válido")
            continue
        for sym, entry in entries.items():
            # H3: cada entrada se valida AISLADA; una excepción cualquiera va a ``errors`` y las demás cargan.
            try:
                errs: list[str] = []
                clean: dict | None = None
                bad_sym = _check_sqx_symbol(sym)
                if bad_sym:
                    errs.append(bad_sym)
                elif not isinstance(entry, dict):
                    errs.append("la ficha no es un diccionario")
                else:
                    clean = _build_entry(entry, broker, base=None, default_date=None, errs=errs)
                    prev = reg.sheets.get(sym)
                    if clean is not None and prev is not None:
                        errs.append(_dup_message(sym, prev))
                if errs or clean is None:
                    errors.append(f"{name}: {bid}/{sym}: " + "; ".join(errs))
                    continue
                reg.sheets[sym] = _sheet_from_entry(broker, sym, clean)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {bid}/{sym}: {type(exc).__name__}: {exc}")


def _build_registry() -> Registry:
    reg = Registry()
    doc, err = _official_doc()
    if err:
        reg.errors.append(err)
    else:
        _load_official(doc, reg)
    _load_user(reg)
    return reg


def load_registry() -> Registry:
    """Registro vigente (oficiales + usuario). Cacheado por los mtimes de ``brokers.yaml``, de los ficheros
    fuente de los adaptadores y del fichero de usuario. Solo lectura: no mutar el objeto devuelto."""
    with _STATE_LOCK:
        stamp = _stamp()
        if _reg_cache["stamp"] != stamp:
            _reg_cache.update(stamp=stamp, registry=_build_registry())
        return _reg_cache["registry"]


def _reload() -> None:
    with _STATE_LOCK:
        _reg_cache.update(stamp=None, registry=None)
        # también el documento oficial: dos escrituras seguidas pueden compartir mtime (resolución gruesa en Windows)
        _doc_cache.update(stamp=None, doc=None, error=None)
    if _reload_hook is not None:
        _reload_hook()


def resolve(sqx_symbol: str) -> SymbolSheet | None:
    """Ficha de un símbolo SQX (``None`` si no existe o está oculta)."""
    reg = load_registry()
    sheet = reg.sheets.get(sqx_symbol)
    if sheet is None or _hidden_key(sheet.broker, sheet.sqx_symbol) in reg.hidden:
        return None
    return sheet


def catalog_entry(sqx_symbol: str, reg: Registry | None = None) -> dict:
    """Estado de ficha de un símbolo SQX para el catálogo de patas: ``{"ficha", "broker", "asset", "motivo"}``.

    - ``ficha = "ok"``: hay ficha usable y visible; ``broker`` es su bróker y ``asset`` el ``asset_class`` de ese
      bróker;
    - ``"no_usable"``: hay ficha pero está marcada no usable (p. ej. divisa no USD); ``motivo`` lo explica;
    - ``"sin_ficha"``: no hay ficha (o está oculta; el ``motivo`` lo dice): ``broker`` y ``asset`` son ``None``.

    ``reg`` permite reutilizar un registro ya cargado al recorrer muchas patas."""
    reg = reg or load_registry()
    sheet = reg.sheets.get(sqx_symbol)
    if sheet is None:
        return {"ficha": "sin_ficha", "broker": None, "asset": None, "motivo": None}
    if _hidden_key(sheet.broker, sheet.sqx_symbol) in reg.hidden:
        return {"ficha": "sin_ficha", "broker": None, "asset": None,
                "motivo": f"la ficha del símbolo '{sqx_symbol}' (bróker '{sheet.broker}') está oculta"}
    broker = reg.brokers.get(sheet.broker)
    return {"ficha": "ok" if sheet.usable else "no_usable", "broker": sheet.broker,
            "asset": broker.asset_class if broker is not None else sheet.asset_class,
            "motivo": None if sheet.usable else sheet.motivo}


def broker_for_profile(key: str) -> str | None:
    """Bróker de un perfil de prop firm: el de su perfil OFICIAL raíz (``profiles.based_on``); ``None`` si la
    raíz no tiene bróker (p. ej. Apex) o no figura en ``brokers.yaml``."""
    return load_registry().profiles.get(profiles.based_on(key))


def lens_broker(lens: str) -> str | None:
    return load_registry().lenses.get(lens)


def reader_spec(sheet: SymbolSheet) -> dict:
    """``spec`` de ``sqx.sqx_trades.read_trade_table_spec`` para una ficha: ``symbol``, ``asset_class``,
    ``point_value_usd``, ``tick_size`` (``None`` = no se compara), ``commission`` (exactamente una de
    ``per_lot_rt`` o ``pct_notional_rt``), ``lot_step``, ``min_lot`` y ``source`` (``{campo: procedencia}``)."""
    return {"symbol": sheet.sqx_symbol, "asset_class": sheet.asset_class,
            "point_value_usd": sheet.point_value_usd, "tick_size": sheet.tick_size,
            "commission": dict(sheet.commission), "lot_step": sheet.lot_step, "min_lot": sheet.min_lot,
            "source": {f: s.get("from") for f, s in sheet.sources.items()}}


def sheet_signature(sheet: SymbolSheet, broker: Broker | str, ohlc_path: Any, ohlc_mtime: Any,
                    synth_mode: Any) -> str:
    """sha256 del JSON canónico de todo lo que, si cambia, invalida una pata ``.sqx`` cacheada (E4): la ficha
    normalizada, el id, la clase de activo, el reloj y la carpeta de OHLC del bróker (``broker`` = el
    :class:`Broker` o su id), la ruta resuelta del OHLC y su mtime, y el modo de la serie sintética.

    ``tick_size`` entra aunque la lista de E4 no lo nombre: el lector compara el tickSize del .sqx con el de la
    ficha y emite un aviso, que forma parte del dict cacheado de la pata (aceptado en la revisión de la F1)."""
    if isinstance(broker, str):
        broker = load_registry().brokers[broker]
    payload = {
        "sheet": {"point_value_usd": sheet.point_value_usd, "lot_step": sheet.lot_step,
                  "min_lot": sheet.min_lot, "max_lot": sheet.max_lot, "commission": dict(sheet.commission),
                  "spread_px": sheet.spread_px, "ohlc_m1": sheet.ohlc_m1, "tick_size": sheet.tick_size},
        "broker": {"id": broker.id, "asset_class": broker.asset_class, "clock": broker.clock,
                   "ohlc_dir": broker.ohlc_dir},
        # ruta normalizada con «/» en todos los sistemas (sin segmentos redundantes): la firma no depende de si llega una cadena
        # o un Path (en Windows str(Path) usa «\\»)
        "ohlc": {"path": None if ohlc_path is None else Path(str(ohlc_path)).as_posix(), "mtime": ohlc_mtime},
        "synth_mode": synth_mode,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Hoja del script de SQX para MT5 (§2.2 + E6 + E11)
# ---------------------------------------------------------------------------
_SHEET_REQUIRED = ("Instrument", "Point Value", "Min Lot Size", "Max Lot Size", "Min Lot Step",
                   "Pip (Tick Size)", "Spread (In Pips)", "Commission Value", "1 Pip Worth")


def _float_or_none(v: Any) -> float | None:
    try:
        x = float(str(v).strip())
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _root(name: str) -> str:
    """Raíz de un símbolo: prefijo alfanumérico antes de ``_``, ``.`` o ``(`` (``US100.cash`` -> ``US100``)."""
    m = re.match(r"[A-Za-z0-9]+", name)
    return m.group(0).upper() if m else ""


def _comm_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(_close(float(a[k]), float(b[k])) for k in a)


# fuente del borrador -> (sujeto de la frase, "el valor de ..." que el usuario puede conservar)
_PROPOSER = {"mt5_sheet": ("tu hoja", "el valor de la hoja"), "sqx": ("tu .sqx", "el valor del .sqx")}
# H5 (revisión de código de la F2): el POST de alta (``_add_sheet``) rechaza cualquier ``sqx_symbol`` que ya tenga ficha,
# OCULTA incluida (E7: la unicidad cuenta las ocultas), así que un borrador sobre una ficha de usuario oculta no es
# guardable ni editable hasta que se muestre.
HIDDEN_MOTIVO = "ficha oculta: muéstrala para editarla"


def _conflicts(existing: SymbolSheet, draft: dict, *, source: str = "mt5_sheet", broker_id: str | None = None,
               hidden: bool = False) -> list[dict]:
    """E11: campos presentes en la ficha y en el borrador (``source``: la hoja de MT5 o el ``.sqx``) con valor
    distinto (tolerancia relativa 1e-12).

    ``ticker`` no entra en la comparación (decisión aceptada en la revisión de la F1): es solo informativo y
    difiere siempre en la forma (``US100`` frente a ``US100.cash``); daría un conflicto falso en cada fila.

    La frase final de cada conflicto dice qué se puede hacer, y debe ser coherente con ``editable`` del borrador
    (H3, revisión de código de la F2): una ficha OFICIAL no se edita (crea un bróker propio); una ficha de usuario de
    OTRO bróker que el de destino (``broker_id``; ``None`` = sin destino fijado) tampoco (crea un bróker propio o dale a
    tus datos en SQX un nombre propio); una ficha de usuario OCULTA hay que mostrarla antes; solo una ficha de usuario
    visible del propio bróker se puede aceptar campo a campo."""
    out: list[dict] = []
    subject, keep = _PROPOSER[source]
    own = "la ficha oficial" if existing.official else "tu ficha"
    other_broker = broker_id is not None and existing.broker != broker_id
    if existing.official:
        tail = "; si tu cuenta es distinta, crea un bróker propio"
    elif other_broker:
        tail = (f"; esa ficha es de otro bróker ('{existing.broker}'): crea un bróker propio o dale a tus datos en "
                "SQX un nombre propio")
    elif hidden:
        tail = f"; la ficha está oculta: muéstrala para quedarte con {keep}, campo a campo"
    else:
        tail = f"; puedes quedarte con {keep}, campo a campo"
    cands: list[tuple[str, Any, Any]] = [(f, getattr(existing, f), draft.get(f)) for f in
                                         ("point_value_usd", "lot_step", "min_lot", "max_lot", "spread_px",
                                          "tick_size")]
    cands.append(("commission", dict(existing.commission) or None, dict(draft.get("commission") or {}) or None))
    for fname, actual, proposed in cands:
        if actual is None or proposed is None:
            continue
        same = _comm_equal(actual, proposed) if fname == "commission" else _close(actual, proposed)
        if same:
            continue
        out.append({
            "field": fname, "actual": actual, "propuesto": proposed,
            "fuente_actual": (existing.sources.get(fname) or {}).get("from"), "fuente_propuesta": source,
            "texto": f"{fname}: {own} dice {_fmt(actual)}, {subject} dice {_fmt(proposed)}{tail}"
                     if fname != "commission" else
                     f"{fname}: {own} dice {actual}, {subject} dice {proposed}{tail}"})
    return out


def _apply_existing(draft: dict, existing: SymbolSheet | None, broker_id: str | None, *, source: str,
                    hidden: bool = False) -> None:
    """E7, E11, H2 y H5 en un borrador plano: si su ``sqx_symbol`` YA tiene ficha, anota ``existing``, los
    ``conflicts`` y, si guardarlo es imposible, ``guardable=False``, ``editable=False`` y ``motivo``. Es imposible
    cuando la ficha existente:

    - es OFICIAL (no se edita ni se redefine): ``motivo`` = el texto E7;
    - está en OTRO bróker que el de destino (``broker_id``; ``None`` = sin destino fijado, como en el alta desde un
      ``.sqx``): ``motivo`` = el texto E7;
    - es de usuario, del propio bróker, pero está OCULTA (``hidden``; H5): el POST de alta la rechaza igual que a
      cualquier duplicado, así que el borrador no es guardable; ``motivo`` = :data:`HIDDEN_MOTIVO`.

    Una ficha del propio usuario, visible y en el mismo bróker, sigue siendo editable (aceptar el conflicto campo
    a campo)."""
    if existing is None:
        return
    draft["existing"] = {"broker": existing.broker, "sqx_symbol": existing.sqx_symbol, "official": existing.official}
    other_broker = broker_id is not None and existing.broker != broker_id
    hidden_user = hidden and not existing.official and not other_broker
    blocked = existing.official or other_broker or hidden_user
    draft["editable"] = not blocked
    draft["conflicts"] = _conflicts(existing, draft, source=source, broker_id=broker_id, hidden=hidden)
    dup = _dup_message(existing.sqx_symbol, existing)
    if other_broker:
        draft["warnings"].append(dup)
    if blocked:
        draft["guardable"] = False
        why = HIDDEN_MOTIVO if hidden_user else dup
        if draft["motivo"] is None:
            draft["motivo"] = why
        elif why not in draft["warnings"]:
            draft["warnings"].append(why)


def parse_mt5_sheet(text: str, broker_id: str, mapping: dict[str, str] | None = None, *,
                    today: str | None = None) -> list[dict]:
    """BORRADORES de ficha desde el texto de "Human Readable Exported Information Sheet.csv" (script de SQX
    para MT5). Puro: no guarda nada. Una fila = un borrador PLANO (campos de ficha + ``guardable``,
    ``motivo``, ``faltan``, ``warnings``, ``existing``, ``editable``, ``conflicts``, ``candidates``).

    - Las columnas se leen por NOMBRE de cabecera. La moneda del valor del punto es la columna SIN NOMBRE
      inmediatamente posterior a ``1 Pip Worth``; si no es ``USD`` (o está vacía) el borrador lleva
      ``currency`` en ``needs_review`` y NO es guardable (E6: nunca se asume USD).
    - ``spread_px = Spread (In Pips) * Pip (Tick Size)``; comisión 0 -> comisión vacía y en ``needs_review``.
    - ``sqx_symbol``: ``mapping[ticker]`` si se da; si no, la sugerencia por raíz entre las fichas del bróker
      destino; si no hay, ``None`` (no guardable hasta que el usuario lo ponga).
    - E11: si el ``sqx_symbol`` ya tiene ficha, ``conflicts`` lista los campos que difieren; una ficha oficial
      no es editable (``editable: False``).
    - H2: si esa ficha es OFICIAL o está en OTRO bróker que el de destino, el borrador no es guardable
      (``guardable: False``, ``editable: False``) y su ``motivo`` es el texto E7. Las fichas ocultas no se
      sugieren en ``candidates``.
    - H3: la frase de cada conflicto es coherente con ``editable`` (una ficha de usuario de OTRO bróker no se acepta
      «campo a campo»: crea un bróker propio o dale a tus datos en SQX un nombre propio).
    - H5: una ficha de USUARIO del propio bróker pero OCULTA tampoco es guardable ni editable (el POST de alta
      rechaza cualquier duplicado, oculta incluida): ``motivo`` = ``HIDDEN_MOTIVO``."""
    reg = load_registry()
    broker = reg.brokers.get(broker_id)
    if broker is None:
        raise ValueError(f"bróker desconocido: '{broker_id}'")
    if broker.asset_class != "cfd":
        raise ValueError("la hoja de MT5 solo vale para brókers CFD (en futuros el paso y el lote mínimo son 1)")
    mapping = dict(mapping or {})
    for k, v in mapping.items():
        if not isinstance(k, str) or _check_sqx_symbol(v):
            raise ValueError(f"mapping['{k}']: {_check_sqx_symbol(v) or 'ticker inválido'}")
    when = today or _today()
    text = text.lstrip("\ufeff")
    first = text.split("\n", 1)[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("la hoja está vacía")
    header = [h.strip() for h in rows[0]]
    col: dict[str, int] = {}
    for i, h in enumerate(header):
        if h and h not in col:
            col[h] = i                              # la PRIMERA columna con ese nombre
    missing = [c for c in _SHEET_REQUIRED if c not in col]
    if missing:
        raise ValueError(f"a la hoja le faltan columnas: {', '.join(missing)}")
    cur_idx = col["1 Pip Worth"] + 1
    has_cur = cur_idx < len(header) and header[cur_idx] == ""

    def cell(row: list[str], name: str) -> str:
        i = col.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    drafts: list[dict] = []
    for row in rows[1:]:
        ticker = cell(row, "Instrument")
        if not ticker:
            continue
        currency = row[cur_idx].strip() if has_cur and cur_idx < len(row) else ""
        usd_ok = currency.upper() == "USD"          # vacía u otra: nunca se asume USD (E6)
        pv = _float_or_none(cell(row, "Point Value")) if usd_ok else None
        min_lot, max_lot = _float_or_none(cell(row, "Min Lot Size")), _float_or_none(cell(row, "Max Lot Size"))
        lot_step, tick = _float_or_none(cell(row, "Min Lot Step")), _float_or_none(cell(row, "Pip (Tick Size)"))
        pips = _float_or_none(cell(row, "Spread (In Pips)"))
        # PLAN-DESVIO (menor): ``Spread (In Pips) * Pip (Tick Size)`` redondeado a 12 cifras significativas para
        # quitar el ruido binario (41 * 0,01 = 0,41000000000000003) y no guardarlo ni mostrarlo así
        spread_px = float(f"{pips * tick:.12g}") if pips is not None and tick is not None else None
        warnings: list[str] = []
        needs: list[str] = []
        commission: dict = {}
        comm_value, comm_type = _float_or_none(cell(row, "Commission Value")), cell(row, "Commission Type")
        if not comm_value:
            needs.append("commission")              # 0 (o ausente): el script la exporta a 0 si no se teclea
        elif comm_type == "SizeBased":
            commission = {"per_lot_rt": comm_value}
        elif comm_type == "PercentageBased":
            commission = {"pct_notional_rt": comm_value}
        else:
            needs.append("commission")
            warnings.append(f"tipo de comisión no soportado en la hoja: {comm_type!r}")
        motivo: str | None = None
        if not usd_ok:
            needs.append("currency")
            motivo = ("divisa no USD (v1)" if currency
                      else "la hoja no trae la moneda del valor del punto: no se asume USD")
        # sqx_symbol: el del mapping o la sugerencia por raíz entre las fichas del bróker destino
        candidates = [s.sqx_symbol for s in reg.sheets.values()
                      if s.broker == broker_id and _root(ticker) and _root(s.sqx_symbol) == _root(ticker)
                      and _hidden_key(s.broker, s.sqx_symbol) not in reg.hidden]      # H2: las ocultas no se sugieren
        if ticker in mapping:
            sqx_symbol, suggested = mapping[ticker], False
        else:
            sqx_symbol, suggested = (candidates[0], True) if candidates else (None, False)
        data = {"ticker": ticker, "point_value_usd": pv, "lot_step": lot_step, "min_lot": min_lot,
                "max_lot": max_lot, "commission": commission, "spread_px": spread_px, "tick_size": tick}
        faltan = [f for f in ("point_value_usd", "lot_step", "min_lot", "commission") if not data.get(f)]
        if sqx_symbol is None:
            faltan.insert(0, "sqx_symbol")
        guardable = usd_ok and sqx_symbol is not None and None not in (pv, lot_step, min_lot)
        if motivo is None and not guardable:
            motivo = ("falta el símbolo SQX" if sqx_symbol is None else
                      "faltan datos obligatorios en la hoja: " + ", ".join(f for f in faltan if f != "commission"))
        existing = reg.sheets.get(sqx_symbol) if sqx_symbol is not None else None
        draft = {
            "broker": broker_id, "ticker": ticker, "sqx_symbol": sqx_symbol, "sqx_symbol_suggested": suggested,
            "candidates": candidates, "asset_class": broker.asset_class, "currency": currency,
            **{f: data.get(f) for f in ("point_value_usd", "lot_step", "min_lot", "max_lot", "commission",
                                         "spread_px")},
            "ohlc_m1": None, "tick_size": tick,
            "sources": {f: {"from": "mt5_sheet", "date": when} for f, v in data.items()
                        if v not in (None, {})},
            "needs_review": needs, "verified_on": None, "notes": None,
            "guardable": guardable, "motivo": motivo, "faltan": faltan, "warnings": warnings,
            "existing": None, "editable": True, "conflicts": [],
        }
        _apply_existing(draft, existing, broker_id, source="mt5_sheet",
                        hidden=existing is not None and _hidden_key(existing.broker, existing.sqx_symbol) in reg.hidden)
        drafts.append(draft)
    return drafts


# ---------------------------------------------------------------------------
# Alta guiada desde un .sqx (§2.3 + E11 + H2)
# ---------------------------------------------------------------------------
def _catalog_sqx_paths() -> dict[str, Path]:
    """``{ruta resuelta: ruta}`` de los ``.sqx`` del catálogo que registró ``main`` (vacío sin proveedor)."""
    if _catalog_provider is None:
        return {}
    out: dict[str, Path] = {}
    for raw in _catalog_provider() or ():
        p = Path(str(raw))
        p = p if p.is_absolute() else ROOT / p
        out[str(p.resolve())] = p
    return out


def draft_from_sqx(path: str, *, today: str | None = None) -> dict:
    """BORRADOR de ficha desde el ``InstrumentInfo`` de un ``.sqx`` del catálogo (puro: no guarda nada).

    Mismo formato plano que los borradores de la hoja de MT5. El ``.sqx`` solo PROPONE: cada campo que trae va con
    procedencia ``sqx`` y en ``needs_review`` (sin verificar).

    | Campo del ``.sqx``                     | Campo del borrador         |
    | -------------------------------------- | -------------------------- |
    | ``pointValue``                         | ``point_value_usd``        |
    | ``orderSizeStep``                      | ``lot_step``               |
    | ``tickSize``                           | ``tick_size``              |
    | ``commissions`` ``SizeBased``          | ``commission.per_lot_rt``  |
    | ``defaultSpread`` · ``tickSize``       | ``spread_px``              |

    El lote mínimo no está en el ``.sqx``: ``min_lot`` queda vacío y en ``faltan``; tampoco hay bróker destino
    (``broker`` en ``faltan``). Si el símbolo ya tiene ficha: ``existing`` y ``conflicts`` (E11); una ficha
    oficial lo hace no guardable (H2, texto E7) y una ficha de usuario OCULTA también (H5, ``HIDDEN_MOTIVO``).
    ``ValueError`` si la ruta no está en el catálogo o el ``.sqx`` no se puede leer (incluido un ``settings.xml``
    sin ``SymbolInfo``/``InstrumentInfo``, H1)."""
    allowed = _catalog_sqx_paths()
    raw = Path(str(path))
    key = str((raw if raw.is_absolute() else ROOT / raw).resolve())
    if key not in allowed:
        raise ValueError("la ruta no está en el catálogo de patas .sqx: solo se lee un .sqx que la app ya lista")
    try:
        info = read_instrument_info(allowed[key])
    except (SQXFormatError, OSError) as exc:          # OSError: el fichero desapareció entre el escaneo y la lectura
        raise ValueError(f"{allowed[key].name}: {exc}") from exc
    when = today or _today()
    warnings: list[str] = []

    def positive(x: float | None) -> float | None:
        return x if x is not None and x > 0 else None

    pv, lot_step, tick = positive(info["point_value"]), positive(info["order_size_step"]), positive(info["tick_size"])
    spread = info["default_spread"]
    # ``defaultSpread · tickSize`` redondeado a 12 cifras significativas para quitar el ruido binario
    spread_px = float(f"{spread * tick:.12g}") if spread is not None and spread >= 0 and tick is not None else None
    commission: dict = {}
    comm = info["commission"]
    if comm is not None:
        if comm["type"] == "SizeBased" and comm["value"] is not None and comm["value"] >= 0:
            commission = {"per_lot_rt": comm["value"]}
        else:
            warnings.append(f"comisión del .sqx de tipo {comm['type']!r} (o sin valor): no se propone; "
                            "tecléala a mano")
    data = {"point_value_usd": pv, "lot_step": lot_step, "tick_size": tick, "commission": commission,
            "spread_px": spread_px}
    sources = {f: {"from": "sqx", "date": when} for f in DATA_FIELDS if data.get(f) not in (None, {})}
    sqx_symbol = info["symbol"]
    # PLAN-DESVIO (menor): el brief pide ``min_lot`` en ``faltan``; aquí también va ``broker`` (primero) porque el borrador
    # no sabe en qué bróker se guardará (el símbolo SQX de un .sqx sin ficha no pertenece a ninguno todavía).
    faltan = ["broker"] + [f for f in ("point_value_usd", "lot_step", "min_lot", "commission")
                           if not data.get(f)]
    motivo = ("el .sqx no trae un valor del punto" if pv is None else _check_sqx_symbol(sqx_symbol))
    draft = {
        "broker": None, "ticker": None, "sqx_symbol": sqx_symbol, "sqx_symbol_suggested": False,
        "candidates": [], "asset_class": None, "currency": None,
        **{f: data.get(f) for f in ("point_value_usd", "lot_step")}, "min_lot": None, "max_lot": None,
        "commission": commission, "spread_px": spread_px, "ohlc_m1": None, "tick_size": tick,
        "sources": sources, "needs_review": list(sources), "verified_on": None, "notes": None,
        "guardable": motivo is None, "motivo": motivo,
        "faltan": faltan, "warnings": warnings, "existing": None, "editable": True, "conflicts": [],
    }
    existing = _registry_sheet(sqx_symbol)
    _apply_existing(draft, existing, None, source="sqx",
                    hidden=existing is not None
                    and _hidden_key(existing.broker, existing.sqx_symbol) in load_registry().hidden)
    return draft


def _registry_sheet(sqx_symbol: str) -> SymbolSheet | None:
    """Ficha con ese ``sqx_symbol`` en el registro, ocultas incluidas (la unicidad también las cuenta)."""
    return load_registry().sheets.get(sqx_symbol)


# ---------------------------------------------------------------------------
# Escritura del fichero de usuario
# ---------------------------------------------------------------------------
def _mutate_doc(fn: Callable[[dict], None]) -> None:
    """Lee el DOCUMENTO crudo, aplica ``fn(doc)``, escribe y recarga, todo dentro de ``_FILE_LOCK``.
    ``fn`` puede lanzar ``HTTPException`` antes de escribir (todo o nada)."""
    with _FILE_LOCK:
        try:
            doc = read_user_file()
        except ValueError as exc:
            raise HTTPException(409, f"{exc}. Arréglalo o bórralo antes de guardar más fichas.")
        fn(doc)
        _write_user_file(doc)
        _reload()


def _errors_400(errs: list[str]) -> HTTPException:
    return HTTPException(400, detail={"errors": errs})


async def _body(request: Request) -> dict:
    try:
        data = profiles.parse_json_strict(await request.body())
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, detail={"errors": [f"JSON inválido: {exc}"]})
    if not isinstance(data, dict):
        raise HTTPException(422, detail={"errors": ["el cuerpo debe ser un objeto JSON"]})
    return data


def _sheet_view(reg: Registry, sqx_symbol: str) -> dict:
    sheet = reg.sheets[sqx_symbol]
    return {**sheet.as_dict(), "hidden": _hidden_key(sheet.broker, sqx_symbol) in reg.hidden}


def _broker_or_404(reg: Registry, broker_id: Any) -> Broker:
    if not isinstance(broker_id, str):
        raise _errors_400(["broker: falta el id del bróker"])
    broker = reg.brokers.get(broker_id)
    if broker is None:
        raise HTTPException(404, f"bróker desconocido: '{broker_id}'")
    return broker


def _add_sheet(doc: dict, reg: Registry, broker: Broker, sqx_symbol: Any, body: dict) -> str:
    """Valida y añade una ficha NUEVA al documento. Devuelve su ``sqx_symbol``."""
    errs: list[str] = []
    bad = _check_sqx_symbol(sqx_symbol)
    if bad:
        raise _errors_400([bad])
    prev = reg.sheets.get(sqx_symbol)
    if prev is not None:
        raise _errors_400([_dup_message(sqx_symbol, prev)])
    entry = _build_entry(body, broker, base=None, default_date=_today(), errs=errs)
    if entry is None:
        raise _errors_400(errs)
    entry.setdefault("ticker", sqx_symbol)
    entry.setdefault("sources", {}).setdefault("ticker", {"from": "manual", "date": _today()})
    doc["symbols"].setdefault(broker.id, {})[sqx_symbol] = entry
    return sqx_symbol


# ---------------------------------------------------------------------------
# Esquema del formulario
# ---------------------------------------------------------------------------
def _schema() -> dict:
    num = {"type": "number"}
    return {
        "version": SCHEMA_VERSION,
        "sheet_fields": [
            {"name": "broker", "label": "Bróker", "type": "broker", "required": True, "section": "Identificación",
             "editable_on_edit": False},
            {"name": "sqx_symbol", "label": _LABELS["sqx_symbol"], "type": "string", "required": True,
             "max": 60, "section": "Identificación", "editable_on_edit": False,
             "help": "Igual, carácter a carácter, que el símbolo del .sqx (SymbolInfo/@symbolName)."},
            {"name": "ticker", "label": _LABELS["ticker"], "type": "string", "required": False, "max": _STR_MAX,
             "section": "Identificación", "help": "Nombre del símbolo en el bróker; solo informativo."},
            {"name": "point_value_usd", "label": _LABELS["point_value_usd"], **num,
             "unit": "USD por 1,0 de precio y lote", "min": 0.0, "min_exclusive": True, "max": _NUM_MAX,
             "required": True, "section": "Contrato"},
            {"name": "lot_step", "label": _LABELS["lot_step"], **num, "unit": "lotes", "min": _LOT_MIN,
             "min_exclusive": False, "max": _LOT_MAX, "required": True, "section": "Contrato",
             "fixed": {"futures": 1}, "help": "En futuros vale 1 (fijo)."},
            {"name": "min_lot", "label": _LABELS["min_lot"], **num, "unit": "lotes", "min": _LOT_MIN,
             "min_exclusive": False, "max": _LOT_MAX, "required": True, "section": "Contrato",
             "fixed": {"futures": 1},
             "help": "Múltiplo del paso de lote y al menos un paso. En futuros vale 1 (fijo)."},
            {"name": "max_lot", "label": _LABELS["max_lot"], **num, "unit": "lotes", "min": 0.0,
             "min_exclusive": True, "max": _LOT_MAX, "required": False, "section": "Contrato"},
            {"name": "tick_size", "label": _LABELS["tick_size"], **num, "unit": "precio", "min": 0.0,
             "min_exclusive": True, "max": _NUM_MAX, "required": False, "section": "Contrato"},
            {"name": "commission", "label": _LABELS["commission"], "type": "commission", "required": True,
             "section": "Costes", "options": [
                 {"id": "per_lot_rt", "label": "USD por lote, ida y vuelta", "unit": "USD", "min": 0.0,
                  "max": _LOT_MAX, "applies_to": ["cfd", "futures"]},
                 {"id": "pct_notional_rt", "label": "% del nocional, ida y vuelta", "unit": "%", "min": 0.0,
                  "max": 100.0, "applies_to": ["cfd"]}],
             "help": "Exactamente una. Sin comisión: per_lot_rt = 0."},
            {"name": "spread_px", "label": _LABELS["spread_px"], **num, "unit": "unidades de precio", "min": 0.0,
             "max": _NUM_MAX, "required": False, "section": "Costes",
             "help": "En unidades de precio (no pips ni puntos). Sin él no hay flotante por minuto."},
            {"name": "ohlc_m1", "label": _LABELS["ohlc_m1"], "type": "string", "required": False, "max": 120,
             "section": "Datos", "help": "Nombre de fichero dentro de la carpeta de OHLC del bróker."},
            {"name": "verified_on", "label": _LABELS["verified_on"], "type": "date", "required": False,
             "section": "Notas"},
            {"name": "notes", "label": _LABELS["notes"], "type": "text", "required": False, "max": _NOTES_MAX,
             "section": "Notas"},
        ],
        "broker_fields": [
            {"name": "id", "label": "Id", "type": "string", "pattern": BROKER_ID_RE.pattern, "required": True},
            {"name": "display_name", "label": "Nombre", "type": "string", "max": _NAME_MAX, "required": True},
            {"name": "company", "label": "Compañía", "type": "string", "max": _COMPANY_MAX, "required": True},
            {"name": "currency", "label": "Divisa de la cuenta", "type": "enum", "options": ["USD"],
             "required": False, "default": "USD"},
            {"name": "asset_class", "label": "Clase de activo", "type": "enum", "options": list(ASSET_CLASSES),
             "required": True},
            {"name": "clock", "label": "Reloj del servidor", "type": "enum", "options": list(CLOCKS),
             "required": True},
            {"name": "ohlc_dir", "label": "Carpeta de OHLC", "type": "string", "required": False,
             "help": "Ruta relativa a la raíz del repo."},
        ],
        "enums": {
            "clock": [{"id": k, "label": _ENUM_LABELS[k]} for k in CLOCKS],
            "asset_class": [{"id": k, "label": _ENUM_LABELS[k]} for k in ASSET_CLASSES],
            "source_from": [{"id": k, "label": _ENUM_LABELS[k]} for k in SOURCE_KINDS],
        },
        "limits": {"min_lot_multiple_tolerance": _STEP_TOL, "export_format": EXPORT_FORMAT,
                   "export_version": EXPORT_VERSION},
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _listing(reg: Registry) -> dict:
    return {
        "brokers": [b.as_dict() for b in reg.brokers.values()],
        "symbols": [_sheet_view(reg, s) for s in reg.sheets],
        "errors": list(reg.errors),
        "profiles": dict(reg.profiles),
        "lenses": dict(reg.lenses),
    }


@router.get("")
def list_symbols():
    return _listing(load_registry())


@router.get("/schema")
def symbols_schema():
    return _schema()


@router.post("/validate")
async def validate_symbol(request: Request):
    """Valida una ficha sin guardar. Con ``editing: true`` el cuerpo es parcial y se fusiona con la ficha de
    usuario vigente (como ``PUT``). Respuesta: ``{ok, errors, sheet}`` (``sheet`` = la ficha normalizada)."""
    body = await _body(request)
    reg = load_registry()
    broker = _broker_or_404(reg, body.get("broker"))
    sqx_symbol = body.get("sqx_symbol")
    errs: list[str] = []
    bad = _check_sqx_symbol(sqx_symbol)
    if bad:
        return {"ok": False, "errors": [bad], "sheet": None}
    editing = bool(body.get("editing", False))
    prev = reg.sheets.get(sqx_symbol)
    base = None
    if prev is not None:
        if not editing or prev.official or prev.broker != broker.id:
            return {"ok": False, "errors": [_dup_message(sqx_symbol, prev)], "sheet": None}
        base = _entry_of(prev)
    elif editing:
        return {"ok": False, "errors": [f"la ficha '{broker.id}/{sqx_symbol}' no existe"], "sheet": None}
    payload = {k: v for k, v in body.items() if k != "editing"}
    entry = _build_entry(payload, broker, base=base, default_date=_today(), errs=errs)
    if entry is None:
        return {"ok": False, "errors": errs, "sheet": None}
    entry.setdefault("ticker", sqx_symbol)
    return {"ok": True, "errors": [], "sheet": _sheet_from_entry(broker, sqx_symbol, entry).as_dict()}


def _entry_of(sheet: SymbolSheet) -> dict:
    """Entrada almacenable (sin identificadores) de una ficha: base de una edición."""
    e: dict[str, Any] = {"ticker": sheet.ticker, "point_value_usd": sheet.point_value_usd,
                         "lot_step": sheet.lot_step, "min_lot": sheet.min_lot,
                         "commission": dict(sheet.commission), "sources": copy.deepcopy(sheet.sources),
                         "needs_review": list(sheet.needs_review)}
    for f in ("max_lot", "spread_px", "ohlc_m1", "tick_size", "verified_on", "notes"):
        if getattr(sheet, f) is not None:
            e[f] = getattr(sheet, f)
    return e


@router.post("")
async def create_symbol(request: Request):
    """Alta de una ficha de usuario en un bróker oficial o propio. Cuerpo: ``broker``, ``sqx_symbol`` y los
    campos de la ficha."""
    body = await _body(request)
    out: dict[str, str] = {}

    def fn(doc: dict) -> None:
        reg = load_registry()
        broker = _broker_or_404(reg, body.get("broker"))
        out["sqx"] = _add_sheet(doc, reg, broker, body.get("sqx_symbol"), body)

    _mutate_doc(fn)
    return _sheet_view(load_registry(), out["sqx"])


@router.put("/{broker_id}/{sqx_symbol}")
async def update_symbol(broker_id: str, sqx_symbol: str, request: Request):
    """Edita una ficha de USUARIO. Parcial: los campos ausentes se conservan, ``null`` borra un campo
    opcional (E11: aceptar un conflicto campo a campo). ``broker`` y ``sqx_symbol`` no se pueden cambiar."""
    body = await _body(request)

    def fn(doc: dict) -> None:
        reg = load_registry()
        broker = _broker_or_404(reg, broker_id)
        sheet = reg.sheets.get(sqx_symbol)
        if sheet is None or sheet.broker != broker_id:
            raise HTTPException(404, f"ficha desconocida: '{broker_id}/{sqx_symbol}'")
        if sheet.official:
            raise _errors_400([f"la ficha oficial '{broker_id}/{sqx_symbol}' no se edita: ocúltala o "
                               "crea un bróker propio"])
        for k in ("broker", "sqx_symbol"):
            if k in body and body[k] != (broker_id if k == "broker" else sqx_symbol):
                raise _errors_400([f"{k}: no se puede cambiar; borra la ficha y crea otra"])
        errs: list[str] = []
        entry = _build_entry(body, broker, base=_entry_of(sheet), default_date=_today(), errs=errs)
        if entry is None:
            raise _errors_400(errs)
        doc["symbols"].setdefault(broker_id, {})[sqx_symbol] = entry

    _mutate_doc(fn)
    return _sheet_view(load_registry(), sqx_symbol)


@router.delete("/{broker_id}/{sqx_symbol}")
def delete_symbol(broker_id: str, sqx_symbol: str):
    def fn(doc: dict) -> None:
        reg = load_registry()
        _broker_or_404(reg, broker_id)
        sheet = reg.sheets.get(sqx_symbol)
        if sheet is None or sheet.broker != broker_id:
            raise HTTPException(404, f"ficha desconocida: '{broker_id}/{sqx_symbol}'")
        if sheet.official:
            raise _errors_400([f"la ficha oficial '{broker_id}/{sqx_symbol}' no se borra: ocúltala"])
        (doc["symbols"].get(broker_id) or {}).pop(sqx_symbol, None)
        key = _hidden_key(broker_id, sqx_symbol)
        doc["hidden"] = [h for h in doc["hidden"] if h != key]       # borrar la quita de `hidden`

    _mutate_doc(fn)
    return {"deleted": _hidden_key(broker_id, sqx_symbol)}


@router.post("/visibility")
async def set_visibility(request: Request):
    """Oculta (``hidden: true``) o muestra (``false``) fichas oficiales y de usuario. ``keys`` = lista de
    ``"<bróker>/<símbolo SQX>"``. Atómico (todas o ninguna); ocultar no borra. 404 con claves desconocidas."""
    body = await _body(request)
    keys, hidden = body.get("keys"), body.get("hidden")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise HTTPException(422, detail={"errors": ["keys: lista no vacía de '<bróker>/<símbolo SQX>'"]})
    if not isinstance(hidden, bool):
        raise HTTPException(422, detail={"errors": ["hidden: debe ser true o false"]})
    keys = list(dict.fromkeys(keys))
    result: dict[str, list[str]] = {}

    def fn(doc: dict) -> None:
        reg = load_registry()
        valid = {_hidden_key(s.broker, s.sqx_symbol) for s in reg.sheets.values()}
        unknown = [k for k in keys if k not in valid]
        if unknown:
            raise HTTPException(404, detail={"errors": [f"fichas desconocidas: {', '.join(unknown)}"],
                                             "unknown": unknown})
        cur = list(doc["hidden"])
        new = cur + [k for k in keys if k not in cur] if hidden else [h for h in cur if h not in keys]
        doc["hidden"] = new
        result["hidden"] = [h for h in new if h in valid]

    _mutate_doc(fn)
    return {"hidden": result["hidden"]}


@router.post("/brokers")
async def create_broker(request: Request):
    """Alta de un bróker de usuario: ``id``, ``display_name``, ``company``, ``asset_class``, ``clock`` y,
    opcionales, ``currency`` (solo USD) y ``ohlc_dir``."""
    body = await _body(request)
    bid = body.get("id")
    defn = {k: v for k, v in body.items() if k != "id"}
    out: dict[str, Broker] = {}

    def fn(doc: dict) -> None:
        reg = load_registry()
        try:
            broker = _parse_broker(bid, defn, official=False)
        except ValueError as exc:
            raise _errors_400(str(exc).split("; "))
        if broker.id in reg.brokers:
            kind = "un bróker oficial" if reg.brokers[broker.id].official else "uno de tus brókers"
            raise _errors_400([f"el id '{broker.id}' ya es {kind}"])
        stored = {k: v for k, v in broker.as_dict().items() if k not in ("id", "official") and v is not None}
        doc["brokers"][broker.id] = stored
        out["broker"] = broker

    _mutate_doc(fn)
    return out["broker"].as_dict()


@router.get("/{broker_id}/{sqx_symbol}/export")
def export_symbol(broker_id: str, sqx_symbol: str):
    reg = load_registry()
    broker = _broker_or_404(reg, broker_id)
    sheet = reg.sheets.get(sqx_symbol)
    if sheet is None or sheet.broker != broker_id:
        raise HTTPException(404, f"ficha desconocida: '{broker_id}/{sqx_symbol}'")
    body = _entry_of(sheet)
    return {"format": EXPORT_FORMAT, "version": EXPORT_VERSION, "broker": broker_id, "sqx_symbol": sqx_symbol,
            "broker_def": None if broker.official else {k: v for k, v in broker.as_dict().items()
                                                         if k not in ("id", "official")},
            "sheet": body}


@router.post("/import")
async def import_symbol(request: Request):
    """Importa un sobre ``risk-app-symbols`` v1 como ficha de usuario. Si el bróker no existe y el sobre trae
    ``broker_def`` (bróker de usuario), se crea en la misma escritura. Las procedencias ``official`` pasan a
    ``manual`` (la ficha importada ya no es la oficial)."""
    body = await _body(request)
    if body.get("format") != EXPORT_FORMAT:
        raise HTTPException(422, detail={"errors": [f"format debe ser '{EXPORT_FORMAT}'"]})
    ver = body.get("version")
    if isinstance(ver, bool) or not isinstance(ver, int) or ver != EXPORT_VERSION:
        raise HTTPException(422, detail={"errors": [f"version debe ser {EXPORT_VERSION}"]})
    sheet = body.get("sheet")
    if not isinstance(sheet, dict):
        raise HTTPException(422, detail={"errors": ["sheet: debe ser un objeto"]})
    sheet = copy.deepcopy(sheet)
    if isinstance(sheet.get("sources"), dict):
        for fname, src in list(sheet["sources"].items()):
            if isinstance(src, dict) and src.get("from") == "official":
                sheet["sources"][fname] = {**src, "from": "manual"}
    out: dict[str, Any] = {"broker_created": False}

    def fn(doc: dict) -> None:
        reg = load_registry()
        bid = body.get("broker")
        broker = reg.brokers.get(bid) if isinstance(bid, str) else None
        if broker is None:
            bdef = body.get("broker_def")
            if not isinstance(bdef, dict):
                raise HTTPException(404, f"bróker desconocido: '{bid}' (y el sobre no trae broker_def)")
            try:
                broker = _parse_broker(bid, bdef, official=False)
            except ValueError as exc:
                raise _errors_400(str(exc).split("; "))
            doc["brokers"][broker.id] = {k: v for k, v in broker.as_dict().items()
                                         if k not in ("id", "official") and v is not None}
            out["broker_created"] = True
        _add_sheet(doc, reg, broker, body.get("sqx_symbol"), sheet)
        out["sqx"] = body["sqx_symbol"]

    _mutate_doc(fn)
    return {"sheet": _sheet_view(load_registry(), out["sqx"]), "broker_created": out["broker_created"],
            "message": "importado"}


@router.post("/mt5-sheet")
async def mt5_sheet(request: Request):
    """Borradores desde el texto de la hoja del script de SQX para MT5. Cuerpo: ``broker`` (destino), ``text``
    y, opcional, ``mapping`` ``{ticker: sqx_symbol}``. NO guarda nada: se guarda con ``POST ""`` por ficha."""
    body = await _body(request)
    text, mapping = body.get("text"), body.get("mapping")
    if not isinstance(text, str):
        raise HTTPException(422, detail={"errors": ["text: falta el texto de la hoja"]})
    if mapping is not None and not isinstance(mapping, dict):
        raise HTTPException(422, detail={"errors": ["mapping: debe ser {ticker: símbolo SQX}"]})
    broker_id = body.get("broker")
    _broker_or_404(load_registry(), broker_id)
    try:
        drafts = parse_mt5_sheet(text, broker_id, mapping)
    except ValueError as exc:
        raise _errors_400([str(exc)])
    return {"broker": broker_id, "drafts": drafts}


@router.get("/draft-from-sqx")
def draft_from_sqx_endpoint(path: str = Query(..., description="ruta de un .sqx del catálogo")):
    """Borrador de ficha desde un ``.sqx`` del catálogo (alta guiada). No guarda nada: se guarda con ``POST ""``
    una vez que el usuario elige bróker, teclea el lote mínimo y revisa los campos que propone el ``.sqx``.
    400 si la ruta no está en el catálogo o el ``.sqx`` no se puede leer."""
    try:
        return draft_from_sqx(path)
    except ValueError as exc:
        raise _errors_400([str(exc)])
