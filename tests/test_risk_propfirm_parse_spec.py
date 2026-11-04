"""Guardián de ``parse_firm_spec`` / ``validate_runnable`` (guardrail ``onboarding-c``, fase C-1).

- I1: los 10 perfiles oficiales dan EXACTAMENTE el ``asdict`` del golden (HEAD ``6e66324``,
  ``tests/fixtures/propfirm_loader_golden.json``, sha256 ``542311b1…25416``) tanto por
  ``parse_firm_spec`` como por ``load_propfirm_config``, y ``validate_runnable(oficial) == []``.
- Enmienda 1: ``KeyError`` (clave ausente) -> ``ValueError`` con perfil y clave; ``TypeError`` /
  ``AttributeError`` se propagan; las claves desconocidas (``meta``) se ignoran.
- Enmienda 4: un test por regla nueva de ``validate_runnable`` y un test por invariante que YA
  cubren los ``_parse_*`` (documentan qué no se duplica).
- I3 (motor): un perfil duplicado sin cambios da el mismo ``PropFirmConfig`` salvo ``key`` y
  ``display_name`` y una simulación idéntica sobre ``day_arrays`` sintéticos.

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest tests/test_risk_propfirm_parse_spec.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.risk.mc import extend_stationary_rows, stationary_block_indices
from src.risk.propfirm import (
    BlockDayArrays,
    load_propfirm_config,
    parse_firm_spec,
    simulate_firm_paths,
    validate_runnable,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "config" / "propfirms.yaml"
GOLDEN = ROOT / "tests" / "fixtures" / "propfirm_loader_golden.json"

FTMO = "ftmo_50k"
APEX = "apex_150k_intraday"
FUT = "ftmo_futures_growth_50k"


def _raw() -> dict[str, dict]:
    return copy.deepcopy(yaml.safe_load(OFFICIAL.read_text(encoding="utf-8"))["firms"])


def _as_golden_json(cfgs: dict) -> dict:
    """Misma serialización que el golden: ``asdict`` -> json con ``default=str`` y claves ordenadas."""
    return json.loads(json.dumps({k: dataclasses.asdict(v) for k, v in cfgs.items()},
                                 default=str, sort_keys=True))


def _golden() -> dict:
    if not GOLDEN.is_file():
        # El fixture no viaja en el export compartible (su manifest no lo incluye): ahí se omite.
        pytest.skip("fixture del golden ausente (export compartible)")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# I1: golden del loader
# ---------------------------------------------------------------------------
def test_i1_load_propfirm_config_igual_al_golden():
    golden = _golden()
    assert len(golden) == 10
    assert _as_golden_json(load_propfirm_config()) == golden


def test_i1_parse_firm_spec_igual_al_golden_en_los_10_oficiales():
    golden = _golden()
    raw = _raw()
    assert set(raw) == set(golden)
    cfgs = {k: parse_firm_spec(k, spec) for k, spec in raw.items()}
    assert _as_golden_json(cfgs) == golden


def test_i1_validate_runnable_oficiales_vacio():
    cfgs = load_propfirm_config()
    assert len(cfgs) == 10
    assert {k: validate_runnable(c) for k, c in cfgs.items()} == {k: [] for k in cfgs}


# ---------------------------------------------------------------------------
# Enmienda 1: errores de parse_firm_spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path, clave", [
    (("account_size",), "account_size"),
    (("fee",), "fee"),
    (("profit_split_pct",), "profit_split_pct"),
    (("phases",), "phases"),
    (("funded",), "funded"),
    (("phases", 0, "profit_target_pct"), "profit_target_pct"),
    (("phases", 0, "name"), "name"),
    (("phases", 0, "min_trading_days"), "min_trading_days"),
])
def test_clave_obligatoria_ausente_es_valueerror_con_perfil_y_clave(path, clave):
    spec = _raw()[FTMO]
    node = spec
    for p in path[:-1]:
        node = node[p]
    del node[path[-1]]
    with pytest.raises(ValueError) as ei:
        parse_firm_spec("mi_perfil", spec)
    msg = str(ei.value)
    assert msg.startswith("mi_perfil:") and f"'{clave}'" in msg


def test_clave_ausente_en_bloque_anidado_es_valueerror():
    spec = _raw()[APEX]
    del spec["funded"]["payouts"]["ladder_usd"]  # sin ladder -> "no vacia" (ValueError propio)
    with pytest.raises(ValueError):
        parse_firm_spec("apex_x", spec)
    spec = _raw()[APEX]
    del spec["funded"]["payouts"]["max_payouts"]  # KeyError interno -> ValueError traducido
    with pytest.raises(ValueError, match="falta 'max_payouts'"):
        parse_firm_spec("apex_x", spec)


def test_typeerror_y_attributeerror_se_propagan_sin_envolver():
    spec = _raw()[FTMO]
    spec["phases"] = [5]                      # 5["name"] -> TypeError
    with pytest.raises(TypeError):
        parse_firm_spec("mi_perfil", spec)
    spec = _raw()[FTMO]
    spec["funded"] = 7                        # 7.get -> AttributeError / TypeError
    with pytest.raises((TypeError, AttributeError)):
        parse_firm_spec("mi_perfil", spec)
    with pytest.raises(TypeError):
        parse_firm_spec("mi_perfil", None)    # None["phases"]


def test_claves_desconocidas_y_meta_se_ignoran():
    spec = _raw()[FTMO]
    base = parse_firm_spec(FTMO, copy.deepcopy(spec))
    spec["meta"] = {"based_on": FTMO, "verified_on": "2026-10-20", "notes": "x"}
    spec["clave_rara"] = {"a": 1}
    assert parse_firm_spec(FTMO, spec) == base


def test_load_propfirm_config_mantiene_errores_de_fichero(tmp_path):
    vacio = tmp_path / "vacio.yaml"
    vacio.write_text("firms: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no hay perfiles bajo 'firms:'"):
        load_propfirm_config(vacio)
    sin_clave = tmp_path / "sin_clave.yaml"
    sin_clave.write_text("otra: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no hay perfiles"):
        load_propfirm_config(sin_clave)
    malo = tmp_path / "malo.yaml"
    spec = _raw()[FTMO]
    del spec["fee"]
    malo.write_text(yaml.safe_dump({"firms": {"x_malo": spec}}), encoding="utf-8")
    with pytest.raises(ValueError, match="x_malo: falta 'fee'"):
        load_propfirm_config(malo)


# ---------------------------------------------------------------------------
# Enmienda 4: validate_runnable, un test por regla nueva
# ---------------------------------------------------------------------------
def _cfg(key, mutate):
    spec = _raw()[key]
    mutate(spec)
    return parse_firm_spec("mi_perfil", spec)


def test_runnable_cycle_fase_sin_drawdown_exige_max_loss():
    cfg = _cfg(FTMO, lambda s: s["phases"][0].update(max_loss_pct=None))
    errs = validate_runnable(cfg)
    assert len(errs) == 1 and "Challenge" in errs[0] and "max_loss_pct" in errs[0]


def test_runnable_cycle_fase_sin_drawdown_exige_max_daily_loss():
    cfg = _cfg(FTMO, lambda s: s["phases"][1].update(max_daily_loss_pct=None))
    errs = validate_runnable(cfg)
    assert len(errs) == 1 and "Verification" in errs[0]


def test_runnable_cycle_con_daily_loss_usd_sin_max_daily_sigue_siendo_error():
    def mut(s):
        s["phases"][0].update(max_daily_loss_pct=None, daily_loss_usd=1000)
    assert validate_runnable(_cfg(FTMO, mut))


def test_runnable_ladder_max_payouts_no_cabe_en_max_payout_cycles():
    def mut(s):
        s["funded"]["max_payout_cycles"] = 5      # max_payouts = 6
    errs = validate_runnable(_cfg(APEX, mut))
    assert len(errs) == 1 and "max_payouts" in errs[0] and "max_payout_cycles" in errs[0]


def test_runnable_ladder_max_payouts_igual_al_tope_es_valido():
    def mut(s):
        s["funded"]["max_payout_cycles"] = 6
    assert validate_runnable(_cfg(APEX, mut)) == []


def test_runnable_activation_con_fee_ccy_eur_no_es_ejecutable():
    errs = validate_runnable(_cfg(APEX, lambda s: s.update(fee_ccy="EUR")))
    assert len(errs) == 1 and "fee_ccy" in errs[0]


def test_runnable_subscription_con_fee_ccy_eur_no_es_ejecutable():
    errs = validate_runnable(_cfg(FUT, lambda s: s.update(fee_ccy="EUR")))
    assert len(errs) == 1 and "fee_ccy" in errs[0]


def test_runnable_drawdown_de_fase_menor_que_account_size():
    def mut(s):
        s["phases"][0]["drawdown"]["amount_usd"] = s["account_size"]
    errs = validate_runnable(_cfg(APEX, mut))
    assert len(errs) == 1 and "drawdown.amount_usd" in errs[0] and "Evaluation" in errs[0]


def test_runnable_drawdown_de_fondeada_menor_que_account_size():
    def mut(s):
        s["funded"]["drawdown"]["amount_usd"] = s["account_size"] + 1
    errs = validate_runnable(_cfg(FUT, mut))
    assert len(errs) == 1 and "funded" in errs[0]


def test_runnable_max_payout_cycles_al_menos_1_en_cycle():
    errs = validate_runnable(_cfg(FTMO, lambda s: s["funded"].update(max_payout_cycles=0)))
    assert len(errs) == 1 and "max_payout_cycles" in errs[0]


def test_runnable_max_payout_cycles_al_menos_1_en_futures():
    errs = validate_runnable(_cfg(FUT, lambda s: s["funded"].update(max_payout_cycles=-3)))
    assert len(errs) == 1 and "max_payout_cycles" in errs[0]


# --- invariantes que YA cubren los _parse_* / el loader (no se duplican en validate_runnable) --
def test_ya_cubierto_ladder_usd_distinto_de_max_payouts():
    with pytest.raises(ValueError, match="max_payouts.*len"):
        _cfg(APEX, lambda s: s["funded"]["payouts"].update(max_payouts=5))


def test_ya_cubierto_min_balance_first_menor_que_residual_mas_min_payout():
    def mut(s):
        s["funded"]["payouts"]["min_balance_first"] = 154_100
    with pytest.raises(ValueError, match="min_balance_first"):
        _cfg(APEX, mut)


def test_ya_cubierto_frozen_floor_menor_que_lock():
    def mut(s):
        s["funded"]["drawdown"]["frozen_floor"] = 154_100  # == lock
    with pytest.raises(ValueError, match="frozen_floor"):
        _cfg(APEX, mut)


def test_ya_cubierto_lock_y_frozen_van_juntos():
    def mut(s):
        del s["funded"]["drawdown"]["frozen_floor"]
    with pytest.raises(ValueError, match="van juntos"):
        _cfg(APEX, mut)


def test_ya_cubierto_by_tier_usd_todos_positivos():
    def mut(s):
        s["funded"]["daily_loss"]["by_tier_usd"][2] = 0
    with pytest.raises(ValueError, match="by_tier_usd"):
        _cfg(APEX, mut)


def test_ya_cubierto_payout_cycle_days_positivo_en_cycle():
    with pytest.raises(ValueError, match="payout_cycle_days"):
        _cfg(FTMO, lambda s: s["funded"].update(payout_cycle_days=0))


# ---------------------------------------------------------------------------
# I3 (motor): duplicado sin cambios == oficial salvo key y display_name
# ---------------------------------------------------------------------------
def _synthetic_world(n_blocks=40, n_paths=60, seed=3):
    rng = np.random.default_rng(seed)
    days_arrays = []
    for _ in range(n_blocks):
        n_days = int(rng.integers(1, 4))
        pnl = rng.normal(250.0, 900.0, n_days)
        rel_closed = np.cumsum(pnl)
        rel_before = np.concatenate([[0.0], rel_closed[:-1]])
        rel_float = np.minimum(rel_before, rel_closed) - np.abs(rng.normal(150.0, 100.0, n_days))
        days_arrays.append(BlockDayArrays(
            day_ord=np.arange(1, n_days + 1), rel_before=rel_before, rel_closed=rel_closed,
            rel_float=rel_float, opens_day=np.ones(n_days, dtype=int),
            total_pnl=float(rel_closed[-1])))
    durations = np.array([float(len(d.day_ord)) for d in days_arrays])
    idx = stationary_block_indices(n_blocks, n_paths, 1.0, np.random.default_rng(0))
    idx_ext = extend_stationary_rows(idx, n_blocks, 1.0, 2 * n_blocks,
                                     np.random.default_rng([0, 2])).astype(np.int32)
    return days_arrays, durations, idx, idx_ext


def test_i3_duplicado_sin_cambios_es_el_mismo_config_salvo_key_y_nombre():
    off = load_propfirm_config()["ftmo_50k"]
    spec = _raw()["ftmo_50k"]
    spec["display_name"] = "Mi FTMO 50k"
    spec["meta"] = {"based_on": "ftmo_50k", "verified_on": None, "notes": None}
    dup = parse_firm_spec("mi_ftmo_50k", spec)
    assert dup != off
    assert dataclasses.replace(dup, key=off.key, display_name=off.display_name) == off


def test_i3_simulacion_identica_con_day_arrays_sinteticos():
    off = load_propfirm_config()["ftmo_50k"]
    spec = _raw()["ftmo_50k"]
    spec["display_name"] = "Mi FTMO 50k"
    dup = parse_firm_spec("mi_ftmo_50k", spec)
    arrays, dur, idx, idx_ext = _synthetic_world()
    a = simulate_firm_paths(idx, arrays, dur, off, seed=0, idx_ext=idx_ext)
    b = simulate_firm_paths(idx, arrays, dur, dup, seed=0, idx_ext=idx_ext)
    assert dataclasses.is_dataclass(a)
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=f.name)
        else:
            assert repr(va) == repr(vb), f.name
