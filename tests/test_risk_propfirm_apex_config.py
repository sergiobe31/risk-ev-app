"""Tests del esquema de config tipado Apex (Fase 1 del plan Apex 150K).

Cubre: parseo del perfil apex_150k_intraday del YAML real, validaciones
duras del loader (ValueError), resolve_fee_eur (USD→EUR una vez) y la
regresion FTMO (debe parsear exactamente igual que antes).
"""
from __future__ import annotations

import copy

import pytest
import yaml

from src.risk.propfirm import (
    ActivationSpec,
    ConsistencySpec,
    DailyLossSpec,
    InactivitySpec,
    LadderPayouts,
    ScalingTier,
    TrailingDrawdown,
    load_propfirm_config,
    resolve_fee_eur,
)


# ---------------------------------------------------------------------------
# Spec Apex valida (espejo del YAML) para ejercitar las validaciones duras
# ---------------------------------------------------------------------------

def _apex_spec() -> dict:
    return {
        "display_name": "Apex 150K Intraday Trail (4.0)",
        "ruleset_version": "apex_4.0_verified_2026-09",
        "account_size": 150000,
        "fee": 49.80,
        "fee_ccy": "USD",
        "profit_split_pct": 100,
        "refund_fee_on_first_payout": False,
        "activation": {"fee_usd": 59, "deadline_days": 7},
        "day_axis": "et_session",
        "phases": [
            {
                "name": "Evaluation",
                "profit_target_pct": 6.0,
                "min_trading_days": 0,
                "max_daily_loss_pct": None,
                "max_loss_pct": None,
                "drawdown": {"type": "trailing_intraday_unrealized", "amount_usd": 4000},
                "lifetime_days": 30,
                "on_horizon_exhausted": "fail",
            }
        ],
        "funded": {
            "model": "ladder",
            "drawdown": {
                "type": "trailing_intraday_unrealized",
                "amount_usd": 4000,
                "lock_at_balance": 154100,
                "frozen_floor": 150100,
            },
            "daily_loss": {
                "type": "pause_not_fatal",
                "reset": "session",
                "by_tier_usd": [2500, 2500, 2500, 3000, 4000],
                "post_hit": "clamp",
            },
            "scaling_tiers": [
                {"level": 1, "profit_range": [0, 1999], "max_mini": 4, "max_micro": 40},
                {"level": 2, "profit_range": [2000, 2999], "max_mini": 5, "max_micro": 50},
                {"level": 3, "profit_range": [3000, 4999], "max_mini": 7, "max_micro": 70},
                {"level": 4, "profit_range": [5000, 9999], "max_mini": 10, "max_micro": 100},
                {"level": 5, "profit_range": [10000, None], "max_mini": 10, "max_micro": 100},
            ],
            "payouts": {
                "qualifying_day_min_usd": 300,
                "days_per_payout": 5,
                "ladder_usd": [2500, 3000, 3000, 4000, 4000, 5000],
                "min_payout_usd": 500,
                "min_balance_first": 154600,
                "residual_floor": 154100,
                "max_payouts": 6,
                "consistency": {
                    "max_best_day_share": 0.50,
                    "strict": True,
                    "window": "since_last_payout",
                },
            },
            "inactivity": {"window_days": 30, "min_days": 2, "min_profit_usd": 50},
            "max_payout_cycles": 50,
        },
    }


def _load_spec(tmp_path, spec: dict):
    path = tmp_path / "firms.yaml"
    path.write_text(yaml.safe_dump({"firms": {"apex_test": spec}}), encoding="utf-8")
    return load_propfirm_config(path)["apex_test"]


def _load_bad(tmp_path, mutate):
    spec = _apex_spec()
    mutate(spec)
    with pytest.raises(ValueError):
        _load_spec(tmp_path, spec)


# ---------------------------------------------------------------------------
# Parseo del perfil real del YAML
# ---------------------------------------------------------------------------

def test_apex_yaml_parsea_y_valores_aterrizan():
    cfgs = load_propfirm_config()
    cfg = cfgs["apex_150k_intraday"]

    assert cfg.display_name == "Apex 150K Intraday Trail (4.0)"
    assert cfg.ruleset_version == "apex_4.0_verified_2026-09"
    assert cfg.account_size == 150_000.0
    assert cfg.fee == pytest.approx(49.80)
    assert cfg.fee_ccy == "USD"
    assert cfg.profit_split_pct == 100.0
    assert cfg.refund_fee_on_first_payout is False
    assert cfg.activation == ActivationSpec(fee_usd=59.0, deadline_days=7)
    assert cfg.day_axis == "et_session"

    assert len(cfg.phases) == 1
    ph = cfg.phases[0]
    assert ph.name == "Evaluation"
    assert ph.profit_target_pct == 6.0
    assert ph.min_trading_days == 0
    assert ph.max_daily_loss_pct is None
    assert ph.max_loss_pct is None
    assert ph.drawdown == TrailingDrawdown(amount_usd=4000.0)
    assert ph.drawdown.type == "trailing_intraday_unrealized"
    assert ph.lifetime_days == 30
    assert ph.on_horizon_exhausted == "fail"

    fnd = cfg.funded
    assert fnd.model == "ladder"
    assert fnd.max_payout_cycles == 50
    assert fnd.drawdown == TrailingDrawdown(
        amount_usd=4000.0, lock_at_balance=154100.0, frozen_floor=150100.0
    )
    assert fnd.daily_loss == DailyLossSpec(
        type="pause_not_fatal",
        by_tier_usd=(2500.0, 2500.0, 2500.0, 3000.0, 4000.0),
        reset="session",
        post_hit="clamp",
    )
    assert len(fnd.scaling_tiers) == 5
    assert fnd.scaling_tiers[0] == ScalingTier(
        level=1, profit_lo=0.0, profit_hi=1999.0, max_mini=4, max_micro=40
    )
    assert fnd.scaling_tiers[4] == ScalingTier(
        level=5, profit_lo=10000.0, profit_hi=None, max_mini=10, max_micro=100
    )
    pay = fnd.payouts
    assert pay == LadderPayouts(
        qualifying_day_min_usd=300.0,
        days_per_payout=5,
        ladder_usd=(2500.0, 3000.0, 3000.0, 4000.0, 4000.0, 5000.0),
        min_payout_usd=500.0,
        min_balance_first=154600.0,
        residual_floor=154100.0,
        max_payouts=6,
        consistency=ConsistencySpec(
            max_best_day_share=0.50, strict=True, window="since_last_payout"
        ),
    )
    assert fnd.inactivity == InactivitySpec(
        window_days=30, min_days=2, min_profit_usd=50.0
    )


# ---------------------------------------------------------------------------
# Validaciones duras del loader
# ---------------------------------------------------------------------------

def test_trailing_exige_max_loss_null(tmp_path):
    def mut(spec):
        spec["phases"][0]["max_loss_pct"] = 3.0
    _load_bad(tmp_path, mut)


def test_daily_loss_cero_prohibido(tmp_path):
    def mut(spec):
        spec["phases"][0]["max_daily_loss_pct"] = 0.0
    _load_bad(tmp_path, mut)


def test_daily_loss_negativo_prohibido(tmp_path):
    def mut(spec):
        spec["phases"][0]["max_daily_loss_pct"] = -1.0
    _load_bad(tmp_path, mut)


def test_ladder_sin_bloque_payouts(tmp_path):
    def mut(spec):
        del spec["funded"]["payouts"]
    _load_bad(tmp_path, mut)


def test_ladder_max_payouts_debe_igualar_len(tmp_path):
    def mut(spec):
        spec["funded"]["payouts"]["max_payouts"] = 5
    _load_bad(tmp_path, mut)


def test_ladder_escalera_no_vacia_y_positiva(tmp_path):
    def mut_vacia(spec):
        spec["funded"]["payouts"]["ladder_usd"] = []
    _load_bad(tmp_path, mut_vacia)


def test_ladder_escalera_con_no_positivos(tmp_path):
    def mut(spec):
        spec["funded"]["payouts"]["ladder_usd"][2] = 0
    _load_bad(tmp_path, mut)


def test_min_balance_first_respeta_safety_net(tmp_path):
    def mut(spec):
        # residual_floor (154100) + min_payout (500) = 154600 > 154500
        spec["funded"]["payouts"]["min_balance_first"] = 154500
    _load_bad(tmp_path, mut)


def test_fee_ccy_invalida(tmp_path):
    def mut(spec):
        spec["fee_ccy"] = "GBP"
    _load_bad(tmp_path, mut)


def test_on_horizon_exhausted_invalido(tmp_path):
    def mut(spec):
        spec["phases"][0]["on_horizon_exhausted"] = "retry"
    _load_bad(tmp_path, mut)


def test_activation_incoherente(tmp_path):
    def mut(spec):
        spec["activation"]["fee_usd"] = 0
    _load_bad(tmp_path, mut)


def test_drawdown_type_no_soportado(tmp_path):
    def mut(spec):
        spec["phases"][0]["drawdown"]["type"] = "trailing_eod"
    _load_bad(tmp_path, mut)


def test_drawdown_lock_y_floor_van_juntos(tmp_path):
    def mut(spec):
        del spec["funded"]["drawdown"]["frozen_floor"]
    _load_bad(tmp_path, mut)


def test_cycle_exige_payout_cycle_days(tmp_path):
    def mut(spec):
        spec["funded"]["model"] = "cycle"
        del spec["funded"]["payouts"]
        spec["funded"].pop("payout_cycle_days", None)
    _load_bad(tmp_path, mut)


# ---------------------------------------------------------------------------
# resolve_fee_eur: conversion USD->EUR una sola vez
# ---------------------------------------------------------------------------

def test_resolve_fee_eur_convierte_usd():
    cfg = load_propfirm_config()["apex_150k_intraday"]
    out = resolve_fee_eur(cfg, fx_eur_usd=1.10)

    assert out.fee == pytest.approx(49.80 / 1.10)
    assert out.fee_ccy == "EUR"
    assert out.activation.fee_eur == pytest.approx(59.0 / 1.10)
    assert out.activation.fee_usd == 59.0  # se conserva lo declarado
    # El config original (frozen) no se muta
    assert cfg.fee == pytest.approx(49.80)
    assert cfg.fee_ccy == "USD"
    assert cfg.activation.fee_eur is None
    # Idempotente: una vez en EUR, devuelve sin tocar aunque falte fx
    assert resolve_fee_eur(out, fx_eur_usd=None) is out


def test_resolve_fee_eur_deja_eur_intacto():
    ftmo = load_propfirm_config()["ftmo_50k"]
    assert resolve_fee_eur(ftmo, fx_eur_usd=None) is ftmo
    assert resolve_fee_eur(ftmo, fx_eur_usd=1.10) is ftmo


def test_resolve_fee_eur_raise_sin_fx():
    cfg = load_propfirm_config()["apex_150k_intraday"]
    with pytest.raises(ValueError):
        resolve_fee_eur(cfg, fx_eur_usd=None)
    with pytest.raises(ValueError):
        resolve_fee_eur(cfg, fx_eur_usd=0.0)
    with pytest.raises(ValueError):
        resolve_fee_eur(cfg, fx_eur_usd=-1.1)


# ---------------------------------------------------------------------------
# Regresion FTMO: parsea exactamente igual que antes
# ---------------------------------------------------------------------------

def test_ftmo_parsea_igual_campo_a_campo():
    cfgs = load_propfirm_config()
    ftmo = cfgs["ftmo_50k"]

    assert ftmo.display_name == "FTMO Challenge 50k"
    assert ftmo.account_size == 50_000.0
    assert ftmo.fee == 345.0
    assert ftmo.fee_ccy == "EUR"
    assert ftmo.profit_split_pct == 80.0
    assert ftmo.refund_fee_on_first_payout is True
    assert ftmo.ruleset_version == ""
    assert ftmo.activation is None
    assert ftmo.day_axis is None

    assert [p.name for p in ftmo.phases] == ["Challenge", "Verification"]
    assert [p.profit_target_pct for p in ftmo.phases] == [10.0, 5.0]
    for p in ftmo.phases:
        assert p.max_daily_loss_pct == 5.0
        assert p.max_loss_pct == 10.0
        assert p.min_trading_days == 4
        assert p.drawdown is None
        assert p.lifetime_days is None
        assert p.on_horizon_exhausted == "exclude"

    fnd = ftmo.funded
    assert fnd.model == "cycle"
    assert fnd.payout_cycle_days == 14
    assert fnd.min_payout_profit_pct == 0.0
    assert fnd.max_payout_cycles == 50
    assert fnd.drawdown is None
    assert fnd.daily_loss is None
    assert fnd.scaling_tiers == ()
    assert fnd.payouts is None
    assert fnd.inactivity is None

    assert cfgs["ftmo_100k"].fee == 540.0
    assert cfgs["ftmo_50k_swing"].fee == 379.0 and cfgs["ftmo_50k_swing"].fee_ccy == "EUR"   # pagado en EUR (2026-10-14)
