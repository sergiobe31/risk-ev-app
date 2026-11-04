"""Perfiles de prop firm del usuario (guardrail ``onboarding-c``, fase C-1, backend).

Todo sintético: el fichero de usuario es un temporal (``USER_PROFILES_PATH`` en ``tmp_path``) y las
simulaciones e2e usan ``.sqx`` CFD escritos con el writer de ``tests/test_sqx_trades.py`` (mismo
patrón que ``test_cfd_sqx_legs.py``). Cubre I2-I9 y las enmiendas 2, 3 y 5-11 del plan v2.

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_profiles.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from poc_web.api import main, profiles
from poc_web.api.main import (
    PortfolioLoadRequest,
    PropfirmSimRequest,
    portfolio_load,
    simulate_propfirm,
    simulate_propfirm_pair,
)

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "config" / "propfirms.yaml"
FTMO = "ftmo_50k"
SWING = "ftmo_50k_swing"
APEX = "apex_150k_intraday"
FUT = "ftmo_futures_growth_50k"
PRO = "ftmo_futures_pro_50k"
HEAD_CATALOG_KEYS = {"key", "display_name", "account_size", "fee", "fee_ccy", "ruleset_version",
                     "day_axis", "funded_model", "rules", "caveats"}


def _raw() -> dict[str, dict]:
    return copy.deepcopy(yaml.safe_load(OFFICIAL.read_text(encoding="utf-8"))["firms"])


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def user_file(tmp_path, monkeypatch):
    path = tmp_path / "propfirms.user.yaml"
    monkeypatch.setattr(profiles, "USER_PROFILES_PATH", path)
    main.reload_firms()
    yield path
    monkeypatch.undo()
    main.reload_firms()          # restaura el estado del proceso (sin perfiles de usuario)


@pytest.fixture()
def api(user_file):
    return TestClient(main.app)


def _send(c, method, url, obj=None, *, text=None):
    body = text if text is not None else json.dumps(obj)
    return c.request(method, url, content=body, headers={"content-type": "application/json"})


def _spec_with(base_key, **paths):
    """spec del oficial ``base_key`` con ``{camino.punteado: valor}`` aplicado."""
    spec = _raw()[base_key]
    for path, value in paths.items():
        node = spec
        toks = path.split("__")
        for t in toks[:-1]:
            node = node[int(t)] if isinstance(node, list) else node[t]
        if isinstance(node, list):
            node[int(toks[-1])] = value
        else:
            node[toks[-1]] = value
    return spec


def _put_path(spec, dotted, value):
    node = spec
    toks = dotted.split(".")
    for t in toks[:-1]:
        node = node[int(t)] if isinstance(node, list) else node[t]
    if isinstance(node, list):
        node[int(toks[-1])] = value
    else:
        node[toks[-1]] = value
    return spec


def _create(c, base=FTMO, key="mi_perfil", name="Mi perfil"):
    r = _send(c, "POST", "/api/profiles", {"base": base, "key": key, "display_name": name})
    assert r.status_code == 200, r.text
    return r.json()


def _errors(r):
    return r.json()["detail"]["errors"]


# ---------------------------------------------------------------------------
# I2: sin fichero de usuario, el catálogo es el de HEAD + 3 campos
# ---------------------------------------------------------------------------
def test_i2_sin_fichero_catalog_head_mas_tres_campos(api):
    firms = api.get("/api/catalog").json()["firms"]
    assert [f["key"] for f in firms] == list(_raw())          # los 10, en el orden del YAML oficial
    for f in firms:
        assert HEAD_CATALOG_KEYS <= set(f)
        assert f["official"] is True and f["based_on"] is None and f["verified_on"] is None
    assert list(main._firms) == list(_raw())


def test_i2_sin_fichero_el_primer_perfil_por_defecto_es_el_primer_oficial(api):
    assert next(iter(main._firms)) == "ftmo_50k"
    assert api.get("/api/health").json()["user_profiles_errors"] == []


def test_catalog_oficiales_primero_y_campos_del_perfil_de_usuario(api):
    _create(api, SWING, "mi_swing", "Mi swing")
    r = _send(api, "PUT", "/api/profiles/mi_swing",
              {"spec": _raw()[SWING] | {"display_name": "Mi swing"}, "verified_on": "2026-10-20"})
    assert r.status_code == 200, r.text
    firms = api.get("/api/catalog").json()["firms"]
    assert [f["key"] for f in firms][-1] == "mi_swing" and len(firms) == 11
    mine = firms[-1]
    assert mine["official"] is False and mine["based_on"] == SWING
    assert mine["verified_on"] == "2026-10-20" and mine["funded_model"] == "cycle"


# ---------------------------------------------------------------------------
# Listado y esquema (enmienda 3: lista blanca explícita)
# ---------------------------------------------------------------------------
def test_listado_10_oficiales_con_spec_cruda_y_tipo(api):
    body = api.get("/api/profiles").json()
    assert body["errors"] == []
    by = {p["key"]: p for p in body["profiles"]}
    assert list(by) == list(_raw())
    assert by[FTMO]["official"] is True and by[FTMO]["kind"] == "cfd" and by[FTMO]["model"] == "cycle"
    assert by[APEX]["kind"] == "futures" and by[APEX]["model"] == "ladder"
    assert by[FUT]["kind"] == "futures" and by[FUT]["model"] == "futures"
    assert by[FTMO]["spec"] == _raw()[FTMO] and "meta" not in by[FTMO]["spec"]
    assert by[FTMO]["based_on"] is None and by[FTMO]["needs_recreate"] is False


@pytest.mark.parametrize("key", list(_raw()))
def test_esquema_forma_y_valores_de_referencia(api, key):
    fields = api.get(f"/api/profiles/{key}/schema").json()
    root = _raw()[key]
    assert fields
    paths = [f["path"] for f in fields]
    assert len(paths) == len(set(paths))
    for f in fields:
        assert {"path", "label", "unit", "type", "min", "max", "section"} <= set(f)
        assert f["label"] and f["type"] in ("number", "integer", "boolean", "string")
        assert f["section"] in ("Cuenta", "Fase 1", "Fase 2", "Fondeada", "Suscripción")
        node = root
        for t in f["path"].split("."):
            node = node[int(t)] if isinstance(node, list) else node[t]
        assert node is not None and node == f["reference"]
        if f["type"] in ("number", "integer"):          # el valor oficial cabe en el rango declarado
            lo, hi, v = f["min"], f["max"], f["reference"]
            assert (v > lo if f["min_exclusive"] else v >= lo) and v <= hi, f["path"]


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for j, v in enumerate(node):
            yield from _leaves(v, f"{prefix}.{j}")
    else:
        yield prefix, node


@pytest.mark.parametrize("key", list(_raw()))
def test_nunca_se_oculta_un_campo_editable_cada_hoja_es_editable_o_bloqueada_con_motivo(key):
    """Cada hoja escalar no nula de la raíz está en la lista blanca o tiene motivo de bloqueo."""
    root = _raw()[key]
    emap = profiles.editable_map(root)
    model = profiles.root_model(root)
    for path, v in _leaves(root):
        if v is None or path in emap:
            continue
        reason = profiles.blocked_reason(path, model)
        assert not reason.startswith("campo no editable"), f"{key}: {path} sin motivo de bloqueo"


def test_esquema_por_modelo_account_size_solo_en_cycle(api):
    paths = lambda k: {f["path"] for f in api.get(f"/api/profiles/{k}/schema").json()}   # noqa: E731
    assert "account_size" in paths(FTMO) and "account_size" in paths(SWING)
    for k in (APEX, FUT, PRO):
        assert "account_size" not in paths(k)


def test_esquema_incluye_los_anadidos_y_excluye_los_bloqueados(api):
    apex = {f["path"] for f in api.get(f"/api/profiles/{APEX}/schema").json()}
    assert {"funded.drawdown.frozen_floor", "activation.fee_usd",
            "funded.payouts.consistency.max_best_day_share",
            "funded.payouts.ladder_usd.0", "funded.scaling_tiers.4.max_micro",
            "funded.daily_loss.by_tier_usd.4", "phases.0.max_contracts.max_mini"} <= apex
    for blocked in ("activation.deadline_days", "funded.payouts.consistency.strict",
                    "funded.payouts.consistency.window", "funded.daily_loss.type",
                    "funded.daily_loss.reset", "funded.daily_loss.post_hit", "funded.model",
                    "funded.drawdown.type", "fee_ccy", "day_axis", "ruleset_version",
                    "funded.payouts.max_payouts", "phases.0.name", "phases.0.on_horizon_exhausted",
                    "phases.0.max_loss_pct"):                        # null en la raíz
        assert blocked not in apex, blocked
    fut = {f["path"] for f in api.get(f"/api/profiles/{PRO}/schema").json()}
    assert "funded.payouts_futures.withdrawal_policy" not in fut
    assert {"subscription.monthly_fee_usd", "phases.0.daily_loss_usd",
            "phases.0.consistency.max_best_day_share", "funded.payouts_futures.withdraw_share"} <= fut
    assert "day_axis_cut_hhmm" not in fut


def test_esquema_secciones_en_espanol(api):
    fut = api.get(f"/api/profiles/{FUT}/schema").json()
    assert {f["section"] for f in fut} == {"Cuenta", "Fase 1", "Fondeada", "Suscripción"}
    cyc = api.get(f"/api/profiles/{FTMO}/schema").json()
    assert {f["section"] for f in cyc} == {"Cuenta", "Fase 1", "Fase 2", "Fondeada"}
    units = {f["path"]: f["unit"] for f in cyc}
    assert units["phases.0.profit_target_pct"] == "%" and units["fee"] == "EUR"
    assert units["funded.payout_cycle_days"] == "días" and units["funded.max_payout_cycles"] == "ciclos"
    assert {f["path"]: f["unit"] for f in fut}["fee"] == "USD"


def test_esquema_de_perfil_desconocido_es_404(api):
    assert api.get("/api/profiles/no_existe/schema").status_code == 404


# ---------------------------------------------------------------------------
# /validate: validación estricta por camino (enmienda 2) y reglas del motor
# ---------------------------------------------------------------------------
def _validate(c, key, spec, based, **extra):
    r = _send(c, "POST", "/api/profiles/validate",
              {"key": key, "spec": spec, "based_on": based, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_validate_sin_cambios_ok(api):
    assert _validate(api, "mi_ftmo", _raw()[FTMO], FTMO) == {"ok": True, "errors": []}


def test_validate_meta_queda_fuera_del_diff(api):
    spec = _raw()[FTMO]
    spec["meta"] = {"based_on": "otra_cosa", "verified_on": "2026-01-01", "notes": "x",
                    "created": "a", "updated": "b"}
    assert _validate(api, "mi_ftmo", spec, FTMO)["ok"] is True


@pytest.mark.parametrize("base, path, value, needle", [
    # --- caminos válidos ---
    (FTMO, "phases.0.profit_target_pct", 12, None),
    (FTMO, "phases.1.min_trading_days", 0, None),
    (FTMO, "fee", 400.5, None),
    (FTMO, "account_size", 100000, None),
    (FTMO, "refund_fee_on_first_payout", False, None),
    (FTMO, "funded.min_payout_profit_pct", 5, None),
    (FTMO, "display_name", "Otro nombre", None),
    (APEX, "activation.fee_usd", 60, None),
    (APEX, "funded.drawdown.frozen_floor", 150200, None),
    (APEX, "funded.payouts.ladder_usd.0", 2600, None),
    (APEX, "funded.scaling_tiers.0.max_mini", 6, None),
    (APEX, "funded.payouts.consistency.max_best_day_share", 0.4, None),
    (APEX, "phases.0.max_contracts.max_mini", 13, None),
    (FUT, "subscription.monthly_fee_usd", 120, None),
    (FUT, "funded.daily_loss.by_tier_usd.0", 1100, None),
    (FUT, "funded.payouts_futures.withdraw_share", 0.6, None),
    (PRO, "phases.0.daily_loss_usd", 1200, None),
    (PRO, "phases.0.consistency.max_best_day_share", 0.55, None),
    # --- tipo y rango por camino ---
    (FTMO, "phases.0.profit_target_pct", "10", "phases.0.profit_target_pct"),
    (FTMO, "phases.0.profit_target_pct", True, "número"),
    (FTMO, "phases.0.profit_target_pct", None, "falta el valor"),   # perfiles-v2: null = campo sin rellenar
    (FTMO, "phases.0.profit_target_pct", 0, "fuera de rango"),
    (FTMO, "phases.0.profit_target_pct", -5, "fuera de rango"),
    (FTMO, "phases.0.profit_target_pct", 101, "fuera de rango"),
    (FTMO, "phases.0.min_trading_days", 2.5, "entero"),
    (FTMO, "phases.0.min_trading_days", -1, "fuera de rango"),
    (FTMO, "profit_split_pct", 0, "fuera de rango"),
    (FTMO, "profit_split_pct", 150, "fuera de rango"),
    (FTMO, "fee", -1, "fuera de rango"),
    (FTMO, "fee", "345", "número"),
    (FTMO, "account_size", 0, "fuera de rango"),
    (FTMO, "refund_fee_on_first_payout", "false", "booleano"),
    (FTMO, "refund_fee_on_first_payout", "true", "booleano"),
    (FTMO, "refund_fee_on_first_payout", 0, "booleano"),
    (FTMO, "refund_fee_on_first_payout", 1, "booleano"),
    (FTMO, "display_name", "", "texto no vacío"),
    (FTMO, "display_name", "x" * 81, "máximo 80"),
    (FTMO, "display_name", 5, "texto"),
    (FTMO, "funded.max_payout_cycles", 0, "fuera de rango"),
    (FTMO, "funded.payout_cycle_days", 366, "fuera de rango"),
    (FTMO, "funded.min_payout_profit_pct", -1, "fuera de rango"),
    (APEX, "funded.payouts.consistency.max_best_day_share", 1.5, "fuera de rango"),
    (APEX, "funded.payouts.consistency.max_best_day_share", 0, "fuera de rango"),
    (FUT, "funded.payouts_futures.withdraw_share", 1.5, "fuera de rango"),
    (FUT, "subscription.billing_period_days", 0, "fuera de rango"),
    (FUT, "subscription.billing_period_days", 30.5, "entero"),
    (APEX, "phases.0.max_contracts.max_mini", 0, "fuera de rango"),
    # --- bloqueados (nombran el camino) ---
    (FTMO, "funded.model", "ladder", "funded.model"),
    (APEX, "account_size", 200000, "ligados al tamaño"),
    (FUT, "account_size", 100000, "account_size"),
    (APEX, "phases.0.max_loss_pct", 5, "phases.0.max_loss_pct"),             # null en la raíz
    (APEX, "phases.0.max_daily_loss_pct", 5, "phases.0.max_daily_loss_pct"),
    (APEX, "activation.deadline_days", 9, "el motor no usa"),
    (APEX, "funded.payouts.consistency.strict", False, "consistency"),
    (APEX, "funded.payouts.consistency.window", "x", "consistency"),
    (APEX, "funded.drawdown.type", "eod_trailing", "tipo de drawdown"),
    (APEX, "phases.0.drawdown.type", "eod_trailing", "tipo de drawdown"),
    (APEX, "funded.daily_loss.type", "hard_fatal", "daily loss"),
    (APEX, "funded.daily_loss.reset", "x", "daily loss"),
    (APEX, "funded.daily_loss.post_hit", "x", "daily loss"),
    (APEX, "fee_ccy", "EUR", "fee_ccy"),
    (APEX, "day_axis", None, "day_axis"),
    (APEX, "ruleset_version", "otra", "ruleset_version"),
    (APEX, "phases.0.on_horizon_exhausted", "exclude", "on_horizon_exhausted"),
    (APEX, "phases.0.name", "Otra", "phases.0.name"),
    (APEX, "funded.payouts.max_payouts", 5, "longitud de ladder_usd"),
    (APEX, "funded.scaling_tiers.0.level", 9, "funded.scaling_tiers.0.level"),
    (APEX, "funded.scaling_tiers.4.profit_range.0", 1, "rangos de los tramos"),
    (FUT, "day_axis_cut_hhmm", "16:20", "day_axis_cut_hhmm"),
    (FUT, "funded.payouts_futures.withdrawal_policy", "buffer_preserving", "política por petición"),
    (FUT, "funded.drawdown.breach_on", "balance", "breach_on"),
    # --- el motor/parser rechaza lo que la lista blanca deja pasar ---
    (FTMO, "phases.0.max_daily_loss_pct", 12, "daily loss"),                  # >= max loss
    (APEX, "funded.payouts.min_balance_first", 100, "min_balance_first"),
    (APEX, "funded.drawdown.frozen_floor", 154100, "frozen_floor"),
    (FUT, "phases.0.drawdown.lock_at_balance", 40000, "lock_at_balance"),
    (FUT, "phases.0.drawdown.amount_usd", 60000, "drawdown.amount_usd"),     # >= account_size
    (PRO, "phases.0.daily_loss_usd", -5, "fuera de rango"),
])
def test_validate_por_camino(api, base, path, value, needle):
    spec = _put_path(_raw()[base], path, value)
    out = _validate(api, "mi_perfil", spec, base)
    if needle is None:
        assert out == {"ok": True, "errors": []}, out
    else:
        assert out["ok"] is False and out["errors"], out
        assert any(needle in e for e in out["errors"]), out["errors"]


def test_validate_estructura_clave_anadida_quitada_y_fases(api):
    spec = _raw()[FTMO]
    spec["extra"] = 1
    out = _validate(api, "mi_perfil", spec, FTMO)
    assert not out["ok"] and any(e.startswith("extra:") for e in out["errors"])
    spec = _raw()[FTMO]
    del spec["fee"]
    assert any("fee" in e and "quitar" in e for e in _validate(api, "mi_perfil", spec, FTMO)["errors"])
    spec = _raw()[FTMO]
    spec["phases"].append(copy.deepcopy(spec["phases"][0]))                  # 3 fases
    assert any(e.startswith("phases:") for e in _validate(api, "mi_perfil", spec, FTMO)["errors"])
    spec = _raw()[FTMO]
    spec["phases"].pop()                                                     # 1 fase
    assert any(e.startswith("phases:") for e in _validate(api, "mi_perfil", spec, FTMO)["errors"])
    spec = _raw()[FTMO]
    spec["phases"] = {"x": 1}                                                # forma distinta
    assert not _validate(api, "mi_perfil", spec, FTMO)["ok"]
    assert not _validate(api, "mi_perfil", [1, 2], FTMO)["ok"]               # spec no es objeto
    spec = _raw()[APEX]
    spec["funded"]["payouts"]["ladder_usd"].append(3000)                     # lista más larga
    assert any("ladder_usd" in e for e in _validate(api, "mi_perfil", spec, APEX)["errors"])


@pytest.mark.parametrize("text", [
    '{"key":"mi_x","based_on":"ftmo_50k","spec":{"fee":NaN}}',
    '{"key":"mi_x","based_on":"ftmo_50k","spec":{"fee":Infinity}}',
    '{"key":"mi_x","based_on":"ftmo_50k","spec":{"fee":-Infinity}}',
    '{"key":"mi_x","based_on":"ftmo_50k","spec":{"fee":1e999}}',
    '{"key":"mi_x"',
    '[1,2]',
])
def test_json_estricto_rechaza_nan_infinity_y_basura(api, text):
    for method, url in (("POST", "/api/profiles/validate"), ("POST", "/api/profiles"),
                        ("POST", "/api/profiles/import")):
        assert _send(api, method, url, text=text).status_code == 422
    _create(api)
    assert _send(api, "PUT", "/api/profiles/mi_perfil", text=text).status_code == 422


def test_check_spec_en_proceso_rechaza_no_finitos():
    for bad in (float("nan"), float("inf"), -float("inf")):
        res = profiles.check_spec("mi_x", _spec_with(FTMO, fee=bad), FTMO)
        assert res.errors and "finito" in res.errors[0]


@pytest.mark.parametrize("key, needle", [
    ("AB", "id inválido"), ("ab", "id inválido"), ("Mi_Perfil", "id inválido"),
    ("mi-perfil", "id inválido"), ("a" * 41, "id inválido"), ("", "id inválido"),
    (None, "id inválido"), (5, "id inválido"),
    ("ftmo_100k", "ya es un perfil oficial"),
])
def test_validate_formato_y_colision_de_id(api, key, needle):
    out = _validate(api, key, _raw()[FTMO], FTMO)
    assert not out["ok"] and any(needle in e for e in out["errors"])


def test_validate_id_en_los_limites_del_formato_es_valido(api):
    assert _validate(api, "abc", _raw()[FTMO], FTMO)["ok"]
    assert _validate(api, "a" * 40, _raw()[FTMO], FTMO)["ok"]


def test_validate_colision_con_perfil_de_usuario_salvo_editing(api):
    _create(api, key="mi_perfil")
    out = _validate(api, "mi_perfil", _raw()[FTMO], FTMO)
    assert not out["ok"] and "ya existe entre tus perfiles" in out["errors"][0]
    assert _validate(api, "mi_perfil", _raw()[FTMO], FTMO, editing=True)["ok"]


def test_validate_based_on_debe_ser_oficial(api):
    _create(api, key="mi_perfil")
    for based in ("mi_perfil", "no_existe", None, 7):
        out = _validate(api, "otro_perfil", _raw()[FTMO], based)
        assert not out["ok"] and "based_on" in out["errors"][0]


def test_validate_meta_verified_on_y_notes(api):
    ok = _validate(api, "mi_perfil", _raw()[FTMO], FTMO, verified_on="2026-10-20", notes="hola")
    assert ok["ok"]
    for bad in ({"verified_on": "2026-13-40"}, {"verified_on": 5}, {"verified_on": "20/10/2026"},
                {"notes": "x" * 2001}, {"notes": 5}):
        assert not _validate(api, "mi_perfil", _raw()[FTMO], FTMO, **bad)["ok"], bad


def test_todos_los_campos_editables_aceptan_un_cambio_pequeno_sin_error_de_tipo_ni_rango():
    """Para cada campo de cada oficial: valor+1 (o el booleano opuesto) no da error de tipo/rango
    (puede dar error del motor, que es otra capa). Y dejar el valor de la raíz siempre es válido."""
    for key, root in _raw().items():
        for f in profiles.editable_fields(root):
            ref = f["reference"]
            assert profiles.check_spec("mi_x", copy.deepcopy(root), key).errors == []
            if f["type"] == "boolean":
                new = not ref
            elif f["type"] == "string":
                new = ref + " x"
            elif f["type"] == "integer":
                new = ref + 1 if ref + 1 <= f["max"] else ref - 1
            else:
                new = ref + 0.5 if ref + 0.5 <= f["max"] else ref - 0.25
            res = profiles.check_spec("mi_x", _put_path(copy.deepcopy(root), f["path"], new), key)
            bad = [e for e in res.errors if "fuera de rango" in e or "debe ser un" in e
                   or "bloqueado" in e]
            assert not bad, (key, f["path"], new, res.errors)


# ---------------------------------------------------------------------------
# I4 / I5: ciclo de vida
# ---------------------------------------------------------------------------
def test_i4_put_campo_bloqueado_422_nombrando_el_camino(api):
    _create(api, key="mi_perfil")
    spec = _put_path(_raw()[FTMO] | {"display_name": "Mi perfil"}, "funded.model", "ladder")
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec})
    assert r.status_code == 422 and any("funded.model" in e for e in _errors(r))
    assert main._firms["mi_perfil"].funded.model == "cycle"


def test_i4_put_campo_editable_200_y_efecto_en_firms(api):
    _create(api, key="mi_perfil")
    ref = main._firms                                              # R3: referencia previa a la recarga
    spec = _put_path(_raw()[FTMO] | {"display_name": "Mi perfil"}, "phases.0.profit_target_pct", 12)
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec})
    assert r.status_code == 200, r.text
    assert main._firms["mi_perfil"].phases[0].profit_target_pct == 12.0
    assert ref is main._firms and ref["mi_perfil"].phases[0].profit_target_pct == 12.0
    assert main._firms[FTMO].phases[0].profit_target_pct == 10.0           # el oficial no cambia


def test_i4_put_errores_del_motor_422_y_no_guarda(api):
    _create(api, key="mi_perfil")
    before = profiles.USER_PROFILES_PATH.read_text(encoding="utf-8")
    spec = _put_path(_raw()[FTMO] | {"display_name": "Mi perfil"}, "phases.0.max_daily_loss_pct", 12)
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec})
    assert r.status_code == 422 and any("daily loss" in e for e in _errors(r))
    assert profiles.USER_PROFILES_PATH.read_text(encoding="utf-8") == before


def test_put_verified_on_notes_conservar_y_borrar(api):
    _create(api, key="mi_perfil")
    spec = _raw()[FTMO] | {"display_name": "Mi perfil"}
    r = _send(api, "PUT", "/api/profiles/mi_perfil",
              {"spec": spec, "verified_on": "2026-10-20", "notes": "ok web FTMO"})
    assert r.status_code == 200
    assert (r.json()["verified_on"], r.json()["notes"]) == ("2026-10-20", "ok web FTMO")
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec})            # claves ausentes: conserva
    assert (r.json()["verified_on"], r.json()["notes"]) == ("2026-10-20", "ok web FTMO")
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec, "verified_on": None})
    assert r.json()["verified_on"] is None and r.json()["notes"] == "ok web FTMO"
    for bad in ({"verified_on": "2026-13-40"}, {"verified_on": 7}, {"notes": "x" * 2001}):
        assert _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec, **bad}).status_code == 422


def test_put_y_delete_sobre_oficial_403_y_desconocido_404(api):
    spec = _raw()[FTMO]
    assert _send(api, "PUT", f"/api/profiles/{FTMO}", {"spec": spec}).status_code == 403
    assert api.delete(f"/api/profiles/{FTMO}").status_code == 403
    assert _send(api, "PUT", "/api/profiles/no_existe", {"spec": spec}).status_code == 404
    assert api.delete("/api/profiles/no_existe").status_code == 404
    assert list(main._firms) == list(_raw())


def test_i5_ciclo_de_vida_crear_editar_exportar_borrar_importar(api, user_file):
    created = _create(api, SWING, "mi_swing", "Mi swing")
    assert created["official"] is False and created["based_on"] == SWING
    assert created["spec"] == _raw()[SWING] | {"display_name": "Mi swing"}
    assert "mi_swing" in main._firms
    spec = copy.deepcopy(created["spec"])
    _put_path(spec, "phases.0.profit_target_pct", 8)
    r = _send(api, "PUT", "/api/profiles/mi_swing", {"spec": spec, "verified_on": "2026-10-20",
                                                      "notes": "n"})
    assert r.status_code == 200, r.text
    edited = r.json()
    exp = api.get("/api/profiles/mi_swing/export").json()
    assert exp == {"format": "risk-app-profile", "version": 1, "key": "mi_swing", "based_on": SWING,
                   "template": None,          # perfiles-v2: el export lleva `template` (null en duplicados)
                   "verified_on": "2026-10-20", "notes": "n", "spec": edited["spec"]}
    assert api.delete("/api/profiles/mi_swing").json() == {"deleted": "mi_swing"}
    assert "mi_swing" not in main._firms and list(main._firms) == list(_raw())
    r = _send(api, "POST", "/api/profiles/import", exp)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["renamed_from"] is None and out["profile"]["key"] == "mi_swing"
    assert out["profile"]["spec"] == edited["spec"]                           # vuelve al mismo spec
    assert (out["profile"]["verified_on"], out["profile"]["notes"]) == ("2026-10-20", "n")
    assert main._firms["mi_swing"].phases[0].profit_target_pct == 8.0
    # el fichero: formato del oficial + meta, sin restos .tmp
    on_disk = yaml.safe_load(user_file.read_text(encoding="utf-8"))["firms"]["mi_swing"]
    assert on_disk["meta"]["based_on"] == SWING and on_disk["meta"]["verified_on"] == "2026-10-20"
    assert {k: v for k, v in on_disk.items() if k != "meta"} == edited["spec"]
    assert not list(user_file.parent.glob("*.tmp"))


def test_crear_colisiones_formato_y_base_desconocida(api):
    _create(api, key="mi_perfil")
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "mi_perfil", "display_name": "x"})
    assert r.status_code == 409
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "ftmo_100k", "display_name": "x"})
    assert r.status_code == 409
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "MAL ID", "display_name": "x"})
    assert r.status_code == 422
    r = _send(api, "POST", "/api/profiles", {"base": "no_existe", "key": "otro_id", "display_name": "x"})
    assert r.status_code == 404
    r = _send(api, "POST", "/api/profiles", {"key": "otro_id"})
    assert r.status_code == 422
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "otro_id", "display_name": " "})
    assert r.status_code == 422


def test_duplicar_un_perfil_de_usuario_hereda_el_based_on_raiz(api):
    _create(api, SWING, "mi_swing", "Mi swing")
    spec = _put_path(_raw()[SWING] | {"display_name": "Mi swing"}, "phases.0.profit_target_pct", 9)
    assert _send(api, "PUT", "/api/profiles/mi_swing", {"spec": spec}).status_code == 200
    dup = _create(api, "mi_swing", "mi_swing_2", "Copia")
    assert dup["based_on"] == SWING and dup["spec"]["phases"][0]["profit_target_pct"] == 9
    assert dup["spec"]["display_name"] == "Copia"


def test_importar_colision_sufijo_dentro_del_formato_enmienda_11(api):
    _create(api, SWING, "mi_swing", "Mi swing")
    exp = api.get("/api/profiles/mi_swing/export").json()
    r1 = _send(api, "POST", "/api/profiles/import", exp).json()
    assert r1["profile"]["key"] == "mi_swing_2" and r1["renamed_from"] == "mi_swing"
    assert "ya existía" in r1["message"]
    r2 = _send(api, "POST", "/api/profiles/import", exp).json()
    assert r2["profile"]["key"] == "mi_swing_3"
    # colisión con un oficial
    r = _send(api, "POST", "/api/profiles/import", exp | {"key": "ftmo_100k"}).json()
    assert r["profile"]["key"] == "ftmo_100k_2" and r["renamed_from"] == "ftmo_100k"
    # base larga: se trunca para que `<base>_<n>` cumpla ^[a-z0-9_]{3,40}$
    long_key = "a" * 40
    _send(api, "POST", "/api/profiles/import", exp | {"key": long_key})
    r = _send(api, "POST", "/api/profiles/import", exp | {"key": long_key}).json()
    assert r["profile"]["key"] == "a" * 38 + "_2" and len(r["profile"]["key"]) == 40
    r = _send(api, "POST", "/api/profiles/import", exp | {"key": long_key}).json()
    assert r["profile"]["key"] == "a" * 38 + "_3"
    assert all(profiles.KEY_RE.fullmatch(k) for k in profiles._state.cfgs)


def test_importar_envolvente_y_validaciones(api):
    _create(api, key="mi_perfil")
    good = api.get("/api/profiles/mi_perfil/export").json()
    api.delete("/api/profiles/mi_perfil")
    def post(obj):
        return _send(api, "POST", "/api/profiles/import", obj)
    assert post(good).status_code == 200
    api.delete("/api/profiles/mi_perfil")
    for bad in (good | {"format": "otro"}, good | {"version": 2}, good | {"version": "1"},
                good | {"version": True}, good | {"version": 1.0 + 1}, {k: v for k, v in good.items()
                                                                       if k != "format"}):
        assert post(bad).status_code == 422, bad
    r = post(good | {"based_on": "no_existe"})
    assert r.status_code == 422 and "no existe como perfil oficial" in _errors(r)[0]
    assert post(good | {"based_on": "mi_perfil"}).status_code == 422             # solo oficiales
    assert post(good | {"key": "MAL"}).status_code == 422
    bad_spec = _put_path(copy.deepcopy(good["spec"]), "funded.model", "ladder")
    r = post(good | {"spec": bad_spec})
    assert r.status_code == 422 and any("funded.model" in e for e in _errors(r))
    assert post(good | {"verified_on": "ayer"}).status_code == 422
    assert "mi_perfil" not in main._firms


def test_exportar_oficial_se_reimporta_como_copia(api):
    exp = api.get(f"/api/profiles/{FTMO}/export").json()
    assert exp["format"] == "risk-app-profile" and exp["based_on"] == FTMO and exp["spec"] == _raw()[FTMO]
    out = _send(api, "POST", "/api/profiles/import", exp).json()
    assert out["profile"]["key"] == "ftmo_50k_2" and out["profile"]["based_on"] == FTMO
    assert api.get("/api/profiles/no_existe/export").status_code == 404


# ---------------------------------------------------------------------------
# I6: firm_key (enmienda 7) en el setup común del 1D y el par 2D
# ---------------------------------------------------------------------------
@pytest.fixture()
def setup_stub(monkeypatch):
    """Portafolio vacío + ``resolve_fee_eur`` que registra el cfg resuelto y corta con 400."""
    monkeypatch.setitem(main._portfolios, "pf_stub", {})
    seen = []

    def fake(cfg, fx):
        seen.append(cfg.key)
        raise ValueError("stop")

    monkeypatch.setattr(main, "resolve_fee_eur", fake)
    return seen


def _req_stub(firm_key):
    return PropfirmSimRequest.model_construct(portfolio_id="pf_stub", firm_key=firm_key)


@pytest.mark.parametrize("fn", [main._propfirm_setup, simulate_propfirm, simulate_propfirm_pair])
def test_i6_firm_key_none_o_vacio_mantiene_el_primer_oficial(api, setup_stub, fn):
    for fk in (None, ""):
        with pytest.raises(HTTPException) as ei:
            fn(_req_stub(fk))
        assert ei.value.status_code == 400 and ei.value.detail == "stop"
    assert setup_stub == ["ftmo_50k", "ftmo_50k"]


@pytest.mark.parametrize("fn", [main._propfirm_setup, simulate_propfirm, simulate_propfirm_pair])
def test_i6_firm_key_no_vacia_y_desconocida_es_404(api, setup_stub, fn):
    with pytest.raises(HTTPException) as ei:
        fn(_req_stub("no_existe"))
    assert ei.value.status_code == 404 and "perfil desconocido" in ei.value.detail
    assert setup_stub == []


def test_i6_firm_key_de_usuario_valida_y_luego_borrada_da_404(api, setup_stub):
    _create(api, key="mi_perfil")
    with pytest.raises(HTTPException) as ei:
        main._propfirm_setup(_req_stub("mi_perfil"))
    assert ei.value.status_code == 400 and setup_stub == ["mi_perfil"]
    assert api.delete("/api/profiles/mi_perfil").status_code == 200
    with pytest.raises(HTTPException) as ei:
        main._propfirm_setup(_req_stub("mi_perfil"))
    assert ei.value.status_code == 404 and setup_stub == ["mi_perfil"]


def test_i6_por_http_404_de_perfil_desconocido(api, monkeypatch):
    monkeypatch.setitem(main._portfolios, "pf_stub", {})
    for url in ("/api/simulate/propfirm", "/api/simulate/propfirm_pair"):
        r = api.post(url, json={"portfolio_id": "pf_stub", "firm_key": "no_existe"})
        assert r.status_code == 404 and "perfil desconocido" in r.json()["detail"]


def test_la_carga_de_cartera_sigue_avisando_sin_404(api, legs):
    # `portfolio_load` con firm_key desconocida solo avisa (el 404 es del setup de simulación)
    res = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key="no_existe"))
    assert any("firm_key 'no_existe' desconocida" in w for w in res["warnings"]), res["warnings"]
    assert res["portfolio_id"] in main._portfolios                      # la carga se hizo (no hay 404)
    res_ok = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key=SWING))
    assert not any("desconocida" in w for w in res_ok["warnings"])


# ---------------------------------------------------------------------------
# I7: _cfd_profile_ok vía based_on
# ---------------------------------------------------------------------------
def test_i7_cfd_profile_ok_hereda_la_identidad_del_oficial_raiz(api):
    _create(api, SWING, "mi_swing", "Mi swing")
    _create(api, FTMO, "mi_ftmo", "Mi ftmo")
    _create(api, FUT, "mi_futuros", "Mi fut")
    _create(api, APEX, "mi_apex", "Mi apex")
    assert main._cfd_profile_ok(main._firms[SWING]) is True
    assert main._cfd_profile_ok(main._firms["mi_swing"]) is True
    assert main._cfd_profile_ok(main._firms["mi_ftmo"]) is True
    assert main._cfd_profile_ok(main._firms["mi_futuros"]) is False
    assert main._cfd_profile_ok(main._firms["mi_apex"]) is False
    assert profiles.based_on("mi_swing") == SWING and profiles.based_on(SWING) == SWING
    assert profiles.based_on("desconocida") == "desconocida"


# ---------------------------------------------------------------------------
# I9 y enmiendas 6, 9: fichero de usuario corrupto / parcial / en colisión
# ---------------------------------------------------------------------------
def test_i9_fichero_corrupto_no_rompe_el_arranque_y_se_avisa_en_health(api, user_file):
    user_file.write_text("firms: [sin cerrar\n  - x: {", encoding="utf-8")
    main.reload_firms()
    assert list(main._firms) == list(_raw())
    errs = api.get("/api/health").json()["user_profiles_errors"]
    assert len(errs) == 1 and errs[0]["key"] is None and "ilegible" in errs[0]["error"]
    body = api.get("/api/profiles").json()
    assert len(body["profiles"]) == 10 and body["errors"] == errs
    for text in ("- a\n- b\n", "firms: 5\n", "firms: [1, 2]\n"):
        user_file.write_text(text, encoding="utf-8")
        main.reload_firms()
        assert list(main._firms) == list(_raw()) and api.get("/api/health").json()["user_profiles_errors"]


def test_fichero_corrupto_bloquea_escrituras_con_409_sin_tocarlo(api, user_file):
    user_file.write_text("firms: [sin cerrar", encoding="utf-8")
    main.reload_firms()
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "nuevo_id", "display_name": "x"})
    assert r.status_code == 409 and "ilegible" in r.json()["detail"]
    assert user_file.read_text(encoding="utf-8") == "firms: [sin cerrar"


def test_enmienda6_errores_parciales_entrada_a_entrada(api, user_file):
    good = _raw()[FTMO] | {"display_name": "Buena", "meta": {"based_on": FTMO, "verified_on": "2026-10-20"}}
    bad_parse = _put_path(_raw()[FTMO], "phases.0.max_daily_loss_pct", 12) | {"meta": {"based_on": FTMO}}
    bad_run = _raw()[FTMO] | {"meta": {"based_on": FTMO}}
    bad_run["funded"]["max_payout_cycles"] = 0
    no_base = _raw()[FTMO] | {"meta": {"based_on": "no_existe"}}
    missing = _raw()[FTMO] | {"meta": {"based_on": FTMO}}
    del missing["fee"]
    user_file.write_text(yaml.safe_dump({"firms": {
        "perfil_bueno": good, "mal_parse": bad_parse, "mal_runnable": bad_run, "sin_base": no_base,
        "falta_fee": missing, "MAL ID": good, "no_dict": 5}}), encoding="utf-8")
    main.reload_firms()
    assert "perfil_bueno" in main._firms and len(main._firms) == 11
    errs = {e["key"]: e["error"] for e in api.get("/api/health").json()["user_profiles_errors"]}
    assert set(errs) == {"mal_parse", "mal_runnable", "sin_base", "falta_fee", "MAL ID", "no_dict"}
    assert "daily loss" in errs["mal_parse"] and "max_payout_cycles" in errs["mal_runnable"]
    assert "fee" in errs["falta_fee"] and "quitar" in errs["falta_fee"] and "no es un perfil oficial" in errs["sin_base"]
    assert {e["key"] for e in api.get("/api/profiles").json()["errors"]} == set(errs)
    # guardar otro perfil NO pierde las entradas inválidas del fichero
    _create(api, key="nuevo_id")
    on_disk = yaml.safe_load(user_file.read_text(encoding="utf-8"))["firms"]
    assert {"perfil_bueno", "mal_parse", "mal_runnable", "sin_base", "falta_fee", "nuevo_id"} <= set(on_disk)


def test_enmienda9_id_de_usuario_que_coincide_con_un_oficial_se_omite_y_se_avisa(api, user_file):
    entry = _raw()[FTMO] | {"display_name": "Impostor", "meta": {"based_on": FTMO}}
    user_file.write_text(yaml.safe_dump({"firms": {"ftmo_100k": entry}}), encoding="utf-8")
    main.reload_firms()
    assert main._firms["ftmo_100k"].display_name == "FTMO Challenge 100k"      # gana el oficial
    errs = api.get("/api/health").json()["user_profiles_errors"]
    assert errs[0]["key"] == "ftmo_100k" and "oficial" in errs[0]["error"]
    assert api.get("/api/catalog").json()["firms"][2]["official"] is True


# ---------------------------------------------------------------------------
# Enmienda 10: raíz cambiada
# ---------------------------------------------------------------------------
def test_enmienda10_raiz_cambiada_needs_recreate_y_put_422(api, tmp_path, monkeypatch):
    _create(api, SWING, "mi_swing", "Mi swing")
    assert api.get("/api/profiles").json()["profiles"][-1]["needs_recreate"] is False
    changed = _raw()
    changed[SWING]["phases"][0]["name"] = "Reto"                  # camino BLOQUEADO distinto en la raíz
    new_root = tmp_path / "propfirms_nueva.yaml"
    new_root.write_text(yaml.safe_dump({"firms": changed}), encoding="utf-8")
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", new_root)
    main.reload_firms()
    # carga estricta (C1): el diff contra la raíz NUEVA toca un camino bloqueado -> el perfil se
    # excluye de _firms y se lista en errors (el fichero no se toca: se puede recuperar)
    assert "mi_swing" not in main._firms
    errs = {e["key"]: e["error"] for e in api.get("/api/profiles").json()["errors"]}
    assert "phases.0.name" in errs["mi_swing"] and "bloqueado" in errs["mi_swing"]
    assert "mi_swing" in yaml.safe_load(profiles.USER_PROFILES_PATH.read_text(encoding="utf-8"))["firms"]
    # un cambio en un camino EDITABLE de la raíz no marca nada
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", OFFICIAL)
    changed2 = _raw()
    changed2[SWING]["fee"] = 400
    new_root.write_text(yaml.safe_dump({"firms": changed2}), encoding="utf-8")
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", new_root)
    main.reload_firms()
    assert api.get("/api/profiles").json()["profiles"][-1]["needs_recreate"] is False
    assert "mi_swing" in main._firms
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", OFFICIAL)


# ---------------------------------------------------------------------------
# Enmienda 5: concurrencia y mutación en sitio
# ---------------------------------------------------------------------------
def test_enmienda5_recarga_en_otro_hilo_mientras_se_pide_el_catalog(api):
    _create(api, key="mi_perfil")
    errors: list[BaseException] = []

    def reloader():
        try:
            for _ in range(40):
                main.reload_firms()
        except BaseException as exc:                              # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=reloader)
    t.start()
    try:
        for _ in range(25):
            firms = main.catalog()["firms"]
            keys = [f["key"] for f in firms]
            assert keys[:10] == list(_raw()) and keys[10:] in ([], ["mi_perfil"])
            assert main._firms_snapshot()
    finally:
        t.join()
    assert not errors


def test_r3_reload_firms_muta_en_sitio(api):
    ref = main._firms
    main.reload_firms()
    assert ref is main._firms and list(ref) == list(_raw())


# ---------------------------------------------------------------------------
# Enmienda 8 / I3 (API): e2e con .sqx CFD sintéticos
# ---------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer_profiles", ROOT / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx

ACCOUNT, PV, COMM, LOTS = 50000.0, 100000.0, 5.0, 0.5


def _f32(x):
    return float(np.float32(x))


def _cfd_trades(slot_hour, seed, n_days=100):
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.bdate_range("2023-01-02", periods=n_days):
        dist = float(rng.choice([0.0010, 0.0015, 0.0020, 0.0025]))
        open_px = _f32(1.10 + 0.0001 * float(rng.integers(-40, 40)))
        win = rng.random() < 0.6
        sl_px = _f32(open_px + dist)
        close_px = _f32(open_px - 1.5 * dist) if win else sl_px
        gross = (open_px - close_px) * PV * LOTS
        mae = (float(rng.uniform(0.0, 0.8)) * dist * PV * LOTS) if win else -gross
        out.append(dict(
            open=datetime(d.year, d.month, d.day, slot_hour, 0),
            close=datetime(d.year, d.month, d.day, slot_hour + 1, 30),
            lots=LOTS, open_px=open_px, close_px=close_px, sl_px=sl_px,
            pnl=gross - COMM * LOTS, mae=max(mae, 0.0), flag=1))
    return out


@pytest.fixture()
def legs(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")
    root = tmp_path / "extra" / "bank"
    root.mkdir(parents=True)
    return [str(write_sqx(root / f"Strategy 8.{k}.1.sqx", _cfd_trades(10 + 2 * k, 200 + k),
                          symbol="EURUSD_ftmo", instrument="EURUSD_cfd_ftmo", pv=PV, tick=0.0001,
                          capital=ACCOUNT)) for k in range(3)]


def _req(res, firm_key):
    return PropfirmSimRequest(portfolio_id=res["portfolio_id"], firm_key=firm_key, f_lo=0.0025,
                              f_hi=0.0075, f_step=0.0025, n_paths=200, n_challenges=5,
                              n_sequences=100, capital0_eur=3000.0, fx_eur_usd=1.08,
                              day_axis="ftmo")


def test_i3_i7_api_perfil_duplicado_sin_cambios_simula_igual_y_pasa_la_guarda_cfd(api, legs):
    _create(api, SWING, "mi_swing", "Mi swing")
    res_u = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key="mi_swing"))
    res_o = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key=SWING))
    assert res_u["ok"], res_u                                           # I7: guarda CFD OK vía based_on
    out_u = simulate_propfirm(_req(res_u, "mi_swing"))
    out_o = simulate_propfirm(_req(res_o, SWING))
    assert out_u["reports"] == out_o["reports"] and out_u["recommendation"] == out_o["recommendation"]
    mu, mo = dict(out_u["meta"]), dict(out_o["meta"])
    # enmienda 8: instantánea SOLO con perfil de usuario, con el cfg que se simuló
    assert "profile_snapshot" not in mo
    snap = mu.pop("profile_snapshot")
    assert snap["key"] == "mi_swing" and snap["based_on"] == SWING and snap["verified_on"] is None
    assert snap["spec"] == _raw()[SWING] | {"display_name": "Mi swing"} and "meta" not in snap["spec"]
    # C3: los booleanos del snapshot son booleanos JSON reales (no 0/1 de ``_jsonable``) y el sha
    # se puede recalcular desde el spec SERIALIZADO
    assert snap["spec"]["refund_fee_on_first_payout"] is True
    assert snap["sha256"] == profiles.spec_sha256(snap["spec"])
    assert snap["sha256"] == profiles.spec_sha256(_raw()[SWING] | {"display_name": "Mi swing"})
    wire = json.loads(json.dumps(snap))
    assert wire["spec"]["refund_fee_on_first_payout"] is True
    assert wire["sha256"] == profiles.spec_sha256(wire["spec"])
    for k in ("elapsed_s", "firm"):
        mu.pop(k), mo.pop(k)
    assert mu == mo
    pair_u = simulate_propfirm_pair(_req(res_u, "mi_swing"))
    assert pair_u["meta"]["profile_snapshot"]["key"] == "mi_swing"
    assert pair_u["meta"]["profile_snapshot"]["spec"]["refund_fee_on_first_payout"] is True
    assert pair_u["meta"]["profile_snapshot"]["sha256"] == snap["sha256"]
    pair_o = simulate_propfirm_pair(_req(res_o, SWING))
    assert "profile_snapshot" not in pair_o["meta"]
    assert pair_u["cells"] == pair_o["cells"]


def test_enmienda8_la_instantanea_lleva_verified_on_y_refleja_la_edicion(api, legs):
    _create(api, SWING, "mi_swing", "Mi swing")
    spec = _put_path(_raw()[SWING] | {"display_name": "Mi swing"}, "phases.0.profit_target_pct", 8)
    assert _send(api, "PUT", "/api/profiles/mi_swing",
                 {"spec": spec, "verified_on": "2026-10-20"}).status_code == 200
    res = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key="mi_swing"))
    snap = simulate_propfirm(_req(res, "mi_swing"))["meta"]["profile_snapshot"]
    assert snap["verified_on"] == "2026-10-20" and snap["spec"]["phases"][0]["profit_target_pct"] == 8
    assert snap["sha256"] == profiles.spec_sha256(spec)
    json.dumps(snap)                                                   # serializable para el historial
    assert snap["sha256"] == profiles.spec_sha256(snap["spec"])


def test_c3_snapshot_conserva_false_real_y_el_resto_de_meta_sigue_con_jsonable(api, legs):
    """Un perfil con `refund_fee_on_first_payout: false` -> `is False` en el snapshot; `_jsonable`
    no cambia para el resto (los bool de `meta`, p. ej. `trailing`, siguen saliendo como 0/1)."""
    _create(api, SWING, "mi_swing", "Mi swing")
    spec = _raw()[SWING] | {"display_name": "Mi swing", "refund_fee_on_first_payout": False}
    assert _send(api, "PUT", "/api/profiles/mi_swing", {"spec": spec}).status_code == 200
    res = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key="mi_swing"))
    meta = simulate_propfirm(_req(res, "mi_swing"))["meta"]
    snap = meta["profile_snapshot"]
    assert snap["spec"]["refund_fee_on_first_payout"] is False
    assert snap["sha256"] == profiles.spec_sha256(snap["spec"])
    assert type(meta["trailing"]) is int                                # `_jsonable` intacto (I1)


def test_i7_cartera_cfd_con_perfil_de_usuario_futuros_es_409(api, legs):
    _create(api, FUT, "mi_fut", "Mi fut")
    res = portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT, firm_key=SWING))
    with pytest.raises(HTTPException) as ei:
        simulate_propfirm(_req(res, "mi_fut"))
    assert ei.value.status_code == 409


def test_snapshot_for_none_con_oficiales_y_desconocidos(api):
    assert profiles.snapshot_for(FTMO) is None and profiles.snapshot_for("no_existe") is None
    assert profiles.catalog_fields(FTMO) == {"official": True, "based_on": None, "verified_on": None,
                                             "company": "FTMO", "hidden": False}   # perfiles-v2


def test_ruta_por_defecto_del_fichero_de_usuario():
    assert profiles.USER_PROFILES_PATH == ROOT / "config" / "propfirms.user.yaml"


# ---------------------------------------------------------------------------
# C1: el fichero editado a mano pasa por la MISMA validación estricta que la API
# ---------------------------------------------------------------------------
def _put_path_many(spec, paths):
    for dotted, value in paths.items():
        _put_path(spec, dotted, value)
    return spec


def _yaml_body(spec, **top):
    """Cuerpo YAML (indentado 4) de una spec, con claves de primer nivel sustituidas en crudo."""
    lines = []
    for k, v in spec.items():
        if k in top:
            lines.append(f"    {k}: {top[k]}\n")
        else:
            block = yaml.safe_dump({k: v}, sort_keys=False, default_flow_style=False)
            lines.append("".join("    " + ln + "\n" for ln in block.splitlines()))
    return "".join(lines)


def test_c1_fichero_editado_a_mano_se_valida_con_el_mismo_rigor_que_la_api(api, user_file):
    def entry(**paths):
        spec = _put_path_many(_raw()[FTMO], paths)
        spec["meta"] = {"based_on": FTMO, "verified_on": "2026-10-20"}
        return spec

    bad_meta = entry()
    bad_meta["meta"]["verified_on"] = "ayer"
    extra = entry()
    extra["clave_extra"] = 1
    quitado = entry()
    del quitado["fee"]
    user_file.write_text(yaml.safe_dump({"firms": {
        "bueno": entry(display_name="Bueno"),
        "entero_float": entry(**{"phases.0.min_trading_days": 4.0}),
        "bool_texto": entry(refund_fee_on_first_payout="false"),
        "bool_numero": entry(refund_fee_on_first_payout=1),
        "bloqueado": entry(fee_ccy="USD"),
        "modelo": entry(**{"funded.model": "ladder"}),
        "rango": entry(**{"phases.0.profit_target_pct": 150}),
        "tipo": entry(fee="345"),
        "extra": extra,
        "quitado": quitado,
        "meta_mala": bad_meta,
    }}), encoding="utf-8")
    main.reload_firms()
    assert {"bueno", "entero_float"} <= set(main._firms) and len(main._firms) == 12
    errs = {e["key"]: e["error"] for e in api.get("/api/health").json()["user_profiles_errors"]}
    assert set(errs) == {"bool_texto", "bool_numero", "bloqueado", "modelo", "rango", "tipo", "extra",
                         "quitado", "meta_mala"}
    assert "refund_fee_on_first_payout" in errs["bool_texto"] and "booleano" in errs["bool_texto"]
    assert "booleano" in errs["bool_numero"]
    assert "fee_ccy" in errs["bloqueado"] and "bloqueado" in errs["bloqueado"]
    assert "funded.model" in errs["modelo"]
    assert "fuera de rango" in errs["rango"] and "phases.0.profit_target_pct" in errs["rango"]
    assert "fee" in errs["tipo"] and "número" in errs["tipo"]
    assert errs["extra"].startswith("clave_extra:") and "quitar" in errs["quitado"]
    assert "verified_on" in errs["meta_mala"]
    # se guarda el spec NORMALIZADO (entero real), no el crudo
    norm = profiles._state.specs["entero_float"]["phases"][0]["min_trading_days"]
    assert norm == 4 and type(norm) is int
    assert main._firms["entero_float"].phases[0].min_trading_days == 4
    # el fichero no se toca (las entradas inválidas siguen ahí para corregirlas)
    assert set(yaml.safe_load(user_file.read_text(encoding="utf-8"))["firms"]) >= set(errs)


def test_c1_nan_infinity_en_el_fichero_y_fecha_de_yaml(api, user_file):
    user_file.write_text(
        "firms:\n"
        "  nan_fee:\n    meta: {based_on: ftmo_50k}\n" + _yaml_body(_raw()[FTMO], fee=".nan")
        + "  inf_fee:\n    meta: {based_on: ftmo_50k}\n" + _yaml_body(_raw()[FTMO], fee=".inf")
        + "  fecha_yaml:\n    meta: {based_on: ftmo_50k, verified_on: 2026-10-20}\n"
        + _yaml_body(_raw()[FTMO]), encoding="utf-8")
    main.reload_firms()
    errs = {e["key"]: e["error"] for e in api.get("/api/health").json()["user_profiles_errors"]}
    assert set(errs) == {"nan_fee", "inf_fee"} and "finito" in errs["nan_fee"]
    assert profiles._state.metas["fecha_yaml"]["verified_on"] == "2026-10-20"   # date de YAML -> str


# ---------------------------------------------------------------------------
# C2: instantánea atómica (firms, metas) en catalog() y _cfd_profile_ok
# ---------------------------------------------------------------------------
def test_c2_catalog_lee_firms_y_metas_bajo_el_lock(api, monkeypatch):
    _create(api, SWING, "mi_swing", "Mi swing")
    held = []
    real = profiles.catalog_fields

    def spy(key):
        held.append(main._FIRMS_LOCK._is_owned())
        return real(key)

    monkeypatch.setattr(profiles, "catalog_fields", spy)
    firms = main.catalog()["firms"]
    assert len(held) == len(firms) == 11 and all(held)
    assert firms[-1]["key"] == "mi_swing" and firms[-1]["official"] is False
    held.clear()
    real_based = profiles.based_on

    def spy_based(key):
        held.append(main._FIRMS_LOCK._is_owned())
        return real_based(key)

    monkeypatch.setattr(profiles, "based_on", spy_based)
    assert main._cfd_profile_ok(main._firms["mi_swing"]) is True
    assert held == [True]


def test_c2_catalog_nunca_mezcla_firms_y_metas_de_momentos_distintos(api, user_file):
    """Una recarga que alterna la existencia del perfil de usuario no deja nunca una entrada con
    `official`/`based_on` de otro momento."""
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    with_profile = user_file.read_text(encoding="utf-8")
    stop = threading.Event()
    errors: list[BaseException] = []

    def flipper():
        try:
            while not stop.is_set():
                user_file.write_text("firms: {}\n", encoding="utf-8")
                main.reload_firms()
                user_file.write_text(with_profile, encoding="utf-8")
                main.reload_firms()
        except BaseException as exc:                              # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=flipper)
    t.start()
    try:
        for _ in range(60):
            for f in main.catalog()["firms"]:
                if f["key"] == "mi_perfil":
                    assert f["official"] is False and f["based_on"] == FTMO
                else:
                    assert f["official"] is True and f["based_on"] is None
    finally:
        stop.set()
        t.join()
    assert not errors


# ---------------------------------------------------------------------------
# C4: official() = una lectura del YAML, cache coherente y bajo lock
# ---------------------------------------------------------------------------
def test_c4_official_una_sola_lectura_y_cache(tmp_path, monkeypatch):
    root = tmp_path / "propfirms.yaml"
    root.write_text(OFFICIAL.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", root)
    reads = []
    real_read = Path.read_text

    def counting(self, *a, **k):
        if self == root:
            reads.append(1)
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", counting)
    raws, cfgs = profiles.official()
    assert len(reads) == 1 and list(raws) == list(cfgs) == list(_raw())
    profiles.official()
    assert len(reads) == 1                                              # cache por mtime
    assert hasattr(profiles._OFFICIAL_LOCK, "acquire")
    monkeypatch.setattr(profiles, "OFFICIAL_PATH", OFFICIAL)
    profiles.official()


# ===========================================================================
# guardrail `perfiles-v2`, fase 1: compañía (AC1), ocultar (AC2) y documento {firms, hidden}
# ===========================================================================
import hashlib   # noqa: E402

OFFICIAL_COMPANY = {k: ("Apex" if k.startswith("apex") else "FTMO") for k in _raw()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hide(c, keys, hidden=True):
    return _send(c, "POST", "/api/profiles/visibility", {"keys": keys, "hidden": hidden})


def _doc(user_file):
    return yaml.safe_load(user_file.read_text(encoding="utf-8"))


# --- AC1: company ---------------------------------------------------------------------------
def test_ac1_los_oficiales_son_9_ftmo_y_1_apex_en_el_yaml_el_listado_de_firmas_y_el_de_perfiles(api):
    raw = _raw()
    assert {k: v["company"] for k, v in raw.items()} == OFFICIAL_COMPANY
    assert sorted(set(OFFICIAL_COMPANY.values())) == ["Apex", "FTMO"]
    assert list(OFFICIAL_COMPANY.values()).count("FTMO") == 9
    firms = api.get("/api/catalog").json()["firms"]
    assert {f["key"]: f["company"] for f in firms} == OFFICIAL_COMPANY
    assert all(f["hidden"] is False for f in firms)            # booleano JSON real (no 0/1)
    profs = api.get("/api/profiles").json()["profiles"]
    assert {p["key"]: p["company"] for p in profs} == OFFICIAL_COMPANY


def test_ac1_propfirms_yaml_solo_gana_las_lineas_company():
    """El YAML oficial sin las líneas `company:` es el de 3e79519: sha256 del fichero reconstruido."""
    lines = OFFICIAL.read_text(encoding="utf-8").split("\n")
    company = [ln for ln in lines if ln.startswith("    company:")]
    assert len(company) == 10
    rest = "\n".join(ln for ln in lines if not ln.startswith("    company:"))
    assert hashlib.sha256(rest.encode("utf-8")).hexdigest() == \
        "01a335a7553ecb565e2669e2bd718db2e465099df634cef25e6dcd2f1af05be4"


def test_ac1_el_loader_ignora_company_el_golden_lo_cubre_test_risk_propfirm_parse_spec():
    """Referencia: el golden del loader lo cubre `tests/test_risk_propfirm_parse_spec.py::
    test_i1_load_propfirm_config_igual_al_golden` (la clave `company` la ignora el loader, C1)."""
    from src.risk.propfirm import load_propfirm_config
    assert len(load_propfirm_config()) == 10
    assert not hasattr(load_propfirm_config()[FTMO], "company")


def test_ac1_entrada_formato_c1_sin_company_hereda_la_de_su_raiz_y_no_se_reescribe(api, user_file):
    entry = _raw()[APEX]
    del entry["company"]
    entry["display_name"] = "Mi Apex viejo"
    entry["meta"] = {"based_on": APEX, "verified_on": "2026-10-20"}
    swing = _raw()[SWING]
    del swing["company"]
    swing["meta"] = {"based_on": SWING}
    text = yaml.safe_dump({"firms": {"apex_viejo": entry, "swing_viejo": swing}})   # formato C-1
    user_file.write_text(text, encoding="utf-8")
    main.reload_firms()
    assert api.get("/api/health").json()["user_profiles_errors"] == []
    assert profiles._state.specs["apex_viejo"]["company"] == "Apex"
    assert profiles._state.specs["swing_viejo"]["company"] == "FTMO"
    firms = {f["key"]: f for f in api.get("/api/catalog").json()["firms"]}
    assert firms["apex_viejo"]["company"] == "Apex" and firms["swing_viejo"]["company"] == "FTMO"
    assert user_file.read_text(encoding="utf-8") == text                  # I3: leer no reescribe
    r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": "otro_nuevo"})
    assert r.status_code == 200
    assert {"apex_viejo", "swing_viejo", "otro_nuevo"} <= set(_doc(user_file)["firms"])
    assert api.get("/api/health").json()["user_profiles_errors"] == []     # siguen cargando


def test_ac1_company_se_copia_al_duplicar_y_es_editable_y_se_normaliza(api):
    created = _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert created["company"] == "FTMO" and created["template"] is None and created["hidden"] is False
    spec = created["spec"] | {"company": "  Funded   Next  "}
    r = _send(api, "PUT", "/api/profiles/mi_perfil", {"spec": spec})
    assert r.status_code == 200, r.text
    assert r.json()["company"] == "Funded Next"                      # strip + espacios colapsados
    firms = {f["key"]: f for f in api.get("/api/catalog").json()["firms"]}
    assert firms["mi_perfil"]["company"] == "Funded Next" and firms["mi_perfil"]["official"] is False
    for bad, needle in (("", "texto no vacío"), ("   ", "texto no vacío"), (5, "texto no vacío"),
                        (None, "falta el valor"), ("x" * 41, "máximo 40")):
        out = _validate(api, "mi_perfil", created["spec"] | {"company": bad}, FTMO, editing=True)
        assert out["ok"] is False and any(needle in e and "company" in e for e in out["errors"]), \
            (bad, out)


def test_ac1_company_en_el_esquema_de_todos_los_modelos(api):
    for key in _raw():
        fields = api.get(f"/api/profiles/{key}/schema").json()
        c = [f for f in fields if f["path"] == "company"]
        assert len(c) == 1 and c[0]["type"] == "string" and c[0]["section"] == "Cuenta"


def test_ac1_import_sin_company_hereda_y_con_company_la_respeta(api):
    exp = api.get(f"/api/profiles/{PRO}/export").json()
    assert exp["spec"]["company"] == "FTMO" and exp["template"] is None
    old = copy.deepcopy(exp)
    del old["spec"]["company"]                                # export v1 anterior a perfiles-v2
    old["key"] = "pro_importado"
    r = _send(api, "POST", "/api/profiles/import", old)
    assert r.status_code == 200, r.text
    assert r.json()["profile"]["company"] == "FTMO"
    new = copy.deepcopy(exp) | {"key": "pro_otro"}
    new["spec"]["company"] = "Otra Casa"
    r = _send(api, "POST", "/api/profiles/import", new)
    assert r.status_code == 200 and r.json()["profile"]["company"] == "Otra Casa"


# --- AC2: ocultar -----------------------------------------------------------------------------
def test_ac2_ocultar_y_restaurar_oficiales_no_excluye_de_las_firmas(api, user_file):
    r = _hide(api, [APEX, FUT])
    assert r.status_code == 200 and r.json() == {"hidden": [APEX, FUT]}
    firms = {f["key"]: f for f in api.get("/api/catalog").json()["firms"]}
    assert len(firms) == 10 and firms[APEX]["hidden"] is True and firms[FUT]["hidden"] is True
    assert firms[FTMO]["hidden"] is False and firms[APEX]["rules"]            # el detalle usa sus rules
    profs = {p["key"]: p for p in api.get("/api/profiles").json()["profiles"]}
    assert profs[APEX]["hidden"] is True and profs[FTMO]["hidden"] is False
    assert _doc(user_file)["hidden"] == [APEX, FUT]
    assert list(main._firms) == list(_raw())                              # ocultar no saca nada de _firms
    r = _hide(api, [APEX], False)
    assert r.json() == {"hidden": [FUT]}
    assert _hide(api, [FUT], False).json() == {"hidden": []}
    assert all(not f["hidden"] for f in api.get("/api/catalog").json()["firms"])


def test_ac2_ids_repetidos_y_ocultar_dos_veces_es_idempotente(api, user_file):
    assert _hide(api, [APEX, APEX]).json() == {"hidden": [APEX]}
    assert _hide(api, [APEX]).json() == {"hidden": [APEX]}
    assert _doc(user_file)["hidden"] == [APEX]
    assert _hide(api, [FUT], False).json() == {"hidden": [APEX]}         # restaurar uno no oculto: no-op


def test_ac2_atomico_404_con_ids_desconocidos_no_cambia_nada(api, user_file):
    r = _hide(api, [APEX, "no_existe", "otro_raro"])
    assert r.status_code == 404
    d = r.json()["detail"]
    assert d["unknown"] == ["no_existe", "otro_raro"] and "no_existe" in d["errors"][0]
    assert not user_file.exists()                                          # ni se escribió
    assert not any(f["hidden"] for f in api.get("/api/catalog").json()["firms"])


@pytest.mark.parametrize("body", [
    {"keys": [], "hidden": True}, {"hidden": True}, {"keys": "ftmo_50k", "hidden": True},
    {"keys": [5], "hidden": True}, {"keys": [FTMO]}, {"keys": [FTMO], "hidden": "si"},
    {"keys": [FTMO], "hidden": 1}])
def test_ac2_cuerpo_invalido_es_422(api, body):
    r = _send(api, "POST", "/api/profiles/visibility", body)
    assert r.status_code == 422, r.text


def test_ac2_409_si_no_quedara_ningun_perfil_visible_y_el_estado_no_cambia(api, user_file):
    r = _hide(api, list(_raw()))
    assert r.status_code == 409 and "al menos un perfil visible" in r.json()["detail"]
    assert not user_file.exists()
    assert _hide(api, [k for k in _raw() if k != APEX]).status_code == 200
    before = user_file.read_text(encoding="utf-8")
    assert _hide(api, [APEX]).status_code == 409
    assert user_file.read_text(encoding="utf-8") == before
    # un perfil de usuario cargado cuenta como visible
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert _hide(api, [APEX]).status_code == 200
    assert _hide(api, ["mi_perfil"]).status_code == 409                    # ahora sí sería el último
    assert _hide(api, [APEX], False).status_code == 200


def test_ac2_hidden_sobrevive_a_crear_editar_borrar_e_importar(api, user_file):
    assert _hide(api, [APEX, PRO]).status_code == 200
    _create(api, SWING, "mi_swing", "Mi swing")
    assert _doc(user_file)["hidden"] == [APEX, PRO]
    spec = _raw()[SWING] | {"display_name": "Mi swing 2"}
    assert _send(api, "PUT", "/api/profiles/mi_swing", {"spec": spec}).status_code == 200
    assert _doc(user_file)["hidden"] == [APEX, PRO]
    exp = api.get("/api/profiles/mi_swing/export").json()
    r = _send(api, "POST", "/api/profiles/import", exp)
    assert r.status_code == 200
    assert _doc(user_file)["hidden"] == [APEX, PRO]
    assert api.delete("/api/profiles/mi_swing").status_code == 200
    assert _doc(user_file)["hidden"] == [APEX, PRO]
    firms = {f["key"]: f["hidden"] for f in api.get("/api/catalog").json()["firms"]}
    assert firms[APEX] is True and firms[PRO] is True and firms[FTMO] is False


def test_ac2_borrar_un_perfil_de_usuario_lo_quita_de_hidden(api, user_file):
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    r = _hide(api, ["mi_perfil", APEX])
    assert r.json() == {"hidden": ["mi_perfil", APEX]}
    firms = {f["key"]: f["hidden"] for f in api.get("/api/catalog").json()["firms"]}
    assert firms["mi_perfil"] is True
    assert api.delete("/api/profiles/mi_perfil").status_code == 200
    assert _doc(user_file)["hidden"] == [APEX]


def test_ac2_delete_de_un_oficial_es_403_con_mensaje_de_ocultar(api, user_file):
    r = api.delete(f"/api/profiles/{APEX}")
    assert r.status_code == 403 and "ocúltalo" in r.json()["detail"] and "no se borran" in r.json()["detail"]
    assert APEX in main._firms and not user_file.exists()


def test_ac2_ids_desconocidos_del_fichero_se_conservan_y_se_ignoran(api, user_file):
    user_file.write_text(yaml.safe_dump({"firms": {}, "hidden": ["fantasma", APEX]}), encoding="utf-8")
    main.reload_firms()
    assert api.get("/api/health").json()["user_profiles_errors"] == []
    firms = {f["key"]: f["hidden"] for f in api.get("/api/catalog").json()["firms"]}
    assert firms[APEX] is True and "fantasma" not in firms and sum(firms.values()) == 1
    r = _hide(api, [FUT])
    assert r.json() == {"hidden": [APEX, FUT]}                            # la respuesta solo lista existentes
    assert _doc(user_file)["hidden"] == ["fantasma", APEX, FUT]            # el fichero conserva el fantasma
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert _hide(api, [APEX, FUT], False).json() == {"hidden": []}
    assert _doc(user_file)["hidden"] == ["fantasma"]
    assert _hide(api, ["fantasma"]).status_code == 404                     # no es un perfil: 404


def test_ac2_hidden_de_entradas_con_error_se_conserva_y_visibility_las_da_por_desconocidas(api, user_file):
    bad = _raw()[FTMO] | {"meta": {"based_on": "no_existe"}}
    user_file.write_text(yaml.safe_dump({"firms": {"roto": bad}, "hidden": ["roto"]}), encoding="utf-8")
    main.reload_firms()
    assert [e["key"] for e in api.get("/api/health").json()["user_profiles_errors"]] == ["roto"]
    assert _hide(api, ["roto"], False).status_code == 404                  # no está cargada
    assert _hide(api, [APEX]).status_code == 200
    d = _doc(user_file)
    assert d["hidden"] == ["roto", APEX] and "roto" in d["firms"]


def test_ac2_el_documento_conserva_otras_claves_de_primer_nivel(api, user_file):
    user_file.write_text(yaml.safe_dump({"firms": {}, "hidden": [], "futuro": {"x": 1}}), encoding="utf-8")
    main.reload_firms()
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert _hide(api, [APEX]).status_code == 200
    d = _doc(user_file)
    assert d["futuro"] == {"x": 1} and d["hidden"] == [APEX] and "mi_perfil" in d["firms"]


@pytest.mark.parametrize("hidden", ["apex_150k_intraday", 5, [1, 2], {"a": 1}])
def test_ac2_hidden_malformado_no_rompe_el_arranque_y_avisa(api, user_file, hidden):
    user_file.write_text(yaml.safe_dump({"firms": {}, "hidden": hidden}), encoding="utf-8")
    main.reload_firms()                                      # no lanza
    errs = api.get("/api/health").json()["user_profiles_errors"]
    assert errs and "hidden" in errs[0]["error"]
    assert list(main._firms) == list(_raw())
    assert _hide(api, [APEX]).status_code == 409             # no se pisa un fichero ilegible


def test_ac2_fichero_c1_sin_hidden_se_lee_y_la_primera_escritura_añade_hidden_vacio(api, user_file):
    entry = _raw()[SWING] | {"meta": {"based_on": SWING}}
    user_file.write_text(yaml.safe_dump({"firms": {"viejo_c1": entry}}), encoding="utf-8")
    main.reload_firms()
    assert "viejo_c1" in main._firms and profiles._state.hidden == []
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    d = _doc(user_file)
    assert d["hidden"] == [] and {"viejo_c1", "mi_perfil"} == set(d["firms"])


# --- Atomicidad y locks (Enmienda v2 5) -------------------------------------------------------
def test_enmienda5_la_recarga_ocurre_dentro_del_file_lock(api, user_file, monkeypatch):
    owned = []
    real = profiles._reload

    def spy():
        owned.append(profiles._FILE_LOCK._is_owned())
        return real()

    monkeypatch.setattr(profiles, "_reload", spy)
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert _hide(api, [APEX]).status_code == 200
    assert api.delete("/api/profiles/mi_perfil").status_code == 200
    assert owned == [True, True, True]


def test_enmienda5_orden_de_locks_file_antes_que_firms_y_main_no_toma_el_file_lock(api, user_file, monkeypatch):
    """La recarga (dentro de `_FILE_LOCK`) toma `_FIRMS_LOCK`; ningún camino de `main.py` toma el
    `_FILE_LOCK` (que solo existe en profiles.py), así que el orden inverso es imposible."""
    seen = []
    real = main.reload_firms

    def spy():
        seen.append((profiles._FILE_LOCK._is_owned(), main._FIRMS_LOCK._is_owned()))
        return real()

    monkeypatch.setattr(profiles, "_reload_hook", spy)
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert seen == [(True, False)]            # con `_FILE_LOCK` ya tomado y `_FIRMS_LOCK` aún libre
    assert "_FILE_LOCK" not in (ROOT / "poc_web" / "api" / "main.py").read_text(encoding="utf-8")


def test_enmienda5_escrituras_concurrentes_no_pierden_ni_corrompen_el_documento(api, user_file):
    errors: list[BaseException] = []

    def creator(prefix):
        try:
            for i in range(6):
                r = _send(api, "POST", "/api/profiles", {"base": FTMO, "key": f"{prefix}_{i}"})
                assert r.status_code == 200, r.text
        except BaseException as exc:                              # pragma: no cover
            errors.append(exc)

    def toggler():
        try:
            for i in range(12):
                assert _hide(api, [APEX, PRO], i % 2 == 0).status_code == 200
        except BaseException as exc:                              # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=creator, args=("aa",)), threading.Thread(target=creator, args=("bb",)),
          threading.Thread(target=toggler)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    d = _doc(user_file)
    assert len(d["firms"]) == 12 and d["hidden"] == []              # i=11 es impar -> restaurados
    assert api.get("/api/health").json()["user_profiles_errors"] == []
    assert len(main._firms) == 22


# --- AC10: propfirms.yaml nunca se escribe ------------------------------------------------------
def test_ac10_propfirms_yaml_identico_byte_a_byte_tras_crear_editar_borrar_ocultar_e_importar(api, user_file):
    before = _sha(OFFICIAL)
    root_user = ROOT / "config" / "propfirms.user.yaml"
    existed = root_user.exists()
    _create(api, SWING, "mi_swing", "Mi swing")
    spec = _raw()[SWING] | {"display_name": "Mi swing 2", "company": "Otra"}
    assert _send(api, "PUT", "/api/profiles/mi_swing", {"spec": spec, "notes": "n"}).status_code == 200
    assert _hide(api, [APEX]).status_code == 200
    exp = api.get("/api/profiles/mi_swing/export").json()
    assert _send(api, "POST", "/api/profiles/import", exp).status_code == 200
    assert _hide(api, [APEX], False).status_code == 200
    assert api.delete("/api/profiles/mi_swing").status_code == 200
    assert _sha(OFFICIAL) == before
    assert root_user.exists() == existed                           # los tests nunca tocan el fichero real


def test_ac2_borrar_el_ultimo_perfil_visible_es_409_y_no_toca_el_fichero(api, user_file):
    _create(api, FTMO, "mi_perfil", "Mi perfil")
    assert _hide(api, list(_raw())).status_code == 200            # solo `mi_perfil` visible
    before = user_file.read_text(encoding="utf-8")
    r = api.delete("/api/profiles/mi_perfil")
    assert r.status_code == 409 and "al menos un perfil visible" in r.json()["detail"]
    assert user_file.read_text(encoding="utf-8") == before and "mi_perfil" in main._firms
    # restaurando uno, ya se puede borrar; y borrar uno oculto o con otros visibles no tiene guarda
    assert _hide(api, [FTMO], False).status_code == 200
    assert api.delete("/api/profiles/mi_perfil").status_code == 200
    assert "mi_perfil" not in main._firms and _doc(user_file)["hidden"] == [k for k in _raw() if k != FTMO]


def test_ac2_borrar_un_perfil_oculto_no_cuenta_como_visible(api, user_file):
    _create(api, FTMO, "perfil_a", "A")
    _create(api, FTMO, "perfil_b", "B")
    assert _hide(api, list(_raw()) + ["perfil_b"]).status_code == 200      # solo `perfil_a` visible
    assert api.delete("/api/profiles/perfil_b").status_code == 200          # borrar el oculto: queda perfil_a
    assert api.delete("/api/profiles/perfil_a").status_code == 409
