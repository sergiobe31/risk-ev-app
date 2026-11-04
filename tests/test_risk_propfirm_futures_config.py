"""Tests de config FTMO Futures (AC-1 del plan): los 6 perfiles cargan con
todos los números del ruleset verificado (tabla §2.8 del plan, ground truth
en skill propfirm-sim) y las validaciones duras del loader rechazan los
perfiles inválidos (v2 #10 incluye cap >= account_size − amount para
eod_trailing). El eje 16:10 en sí se testea en test_risk_session_axis_futures.
"""
from __future__ import annotations

import pytest
import yaml

from src.risk.propfirm import (
    ContractCap,
    DailyLossSpec,
    TrailingDrawdown,
    load_propfirm_config,
    resolve_fee_eur,
)

RULESET = "ftmo_futures_verified_2026-09-29"

# Tabla §2.8 del plan (ruleset verificado en ftmo.com 2026-09-29).
PROFILE_TABLE = {
    "ftmo_futures_growth_50k": dict(
        account=50_000, monthly=119, reset=109, amount=2_000,
        dll_eval=None, dll_funded=("pause_not_fatal", 1_000),
        share=0.40, qdays=4, qmin=150, wshare=0.50, cap=2_500,
        mini=5, micro=50),
    "ftmo_futures_growth_100k": dict(
        account=100_000, monthly=169, reset=159, amount=3_500,
        dll_eval=None, dll_funded=("pause_not_fatal", 2_000),
        share=0.40, qdays=4, qmin=250, wshare=0.50, cap=3_000,
        mini=10, micro=100),
    "ftmo_futures_growth_150k": dict(
        account=150_000, monthly=229, reset=219, amount=5_000,
        dll_eval=None, dll_funded=("pause_not_fatal", 3_000),
        share=0.40, qdays=4, qmin=300, wshare=0.50, cap=4_000,
        mini=15, micro=150),
    "ftmo_futures_pro_50k": dict(
        account=50_000, monthly=139, reset=129, amount=3_000,
        dll_eval=1_000, dll_funded=("hard_fatal", 1_000),
        share=0.50, qdays=5, qmin=200, wshare=1.00, cap=5_000,
        mini=5, micro=50),
    "ftmo_futures_pro_100k": dict(
        account=100_000, monthly=199, reset=189, amount=4_500,
        dll_eval=1_500, dll_funded=("hard_fatal", 1_500),
        share=0.50, qdays=5, qmin=300, wshare=1.00, cap=6_000,
        mini=10, micro=100),
    "ftmo_futures_pro_150k": dict(
        account=150_000, monthly=269, reset=259, amount=6_000,
        dll_eval=2_000, dll_funded=("hard_fatal", 2_000),
        share=0.50, qdays=5, qmin=500, wshare=1.00, cap=8_000,
        mini=15, micro=150),
}


def _growth_spec() -> dict:
    """Spec válida espejo de ftmo_futures_growth_50k (para mutar)."""
    return {
        "display_name": "FTMO Futures Growth 50K",
        "ruleset_version": RULESET,
        "account_size": 50000,
        "fee": 0,
        "fee_ccy": "USD",
        "profit_split_pct": 90,
        "refund_fee_on_first_payout": False,
        "day_axis": "et_session",
        "day_axis_cut_hhmm": "16:10",
        "subscription": {"monthly_fee_usd": 119, "reset_fee_usd": 109,
                         "billing_period_days": 30, "campaign_cap_days": 183,
                         "funded_cap_days": 365, "funded_start_delay_days": 3},
        "phases": [{
            "name": "Evaluation",
            "profit_target_pct": 6.0,
            "min_trading_days": 0,
            "max_daily_loss_pct": None,
            "max_loss_pct": None,
            "drawdown": {"type": "eod_trailing", "amount_usd": 2000,
                         "breach_on": "equity", "lock_at_balance": 50000},
            "consistency": {"max_best_day_share": 0.40, "strict": False},
            "max_contracts": {"max_mini": 5, "max_micro": 50},
        }],
        "funded": {
            "model": "futures",
            "drawdown": {"type": "eod_trailing", "amount_usd": 2000,
                         "breach_on": "equity", "lock_at_balance": 50000},
            "daily_loss": {"type": "pause_not_fatal", "by_tier_usd": [1000],
                           "post_hit": "clamp"},
            "payouts_futures": {
                "qualifying_day_min_usd": 150, "days_per_payout": 4,
                "withdraw_share": 0.50, "request_cap_usd": 2500,
                "min_new_profit_frac": 0.50, "min_payout_usd": 20,
                "withdrawal_policy": "greedy", "buffer_usd": 2000},
            "max_payout_cycles": 50,
        },
    }


def _load_spec(tmp_path, spec: dict):
    path = tmp_path / "firms.yaml"
    path.write_text(yaml.safe_dump({"firms": {"futures_test": spec}}),
                    encoding="utf-8")
    return load_propfirm_config(path)["futures_test"]


def _load_bad(tmp_path, mutate):
    spec = _growth_spec()
    mutate(spec)
    with pytest.raises(ValueError):
        _load_spec(tmp_path, spec)


# ---------------------------------------------------------------------------
# AC-1: los 6 perfiles cargan y cada campo numérico iguala la tabla §2.8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(PROFILE_TABLE))
def test_perfiles_ftmo_futures_tabla_verificada(key):
    t = PROFILE_TABLE[key]
    cfg = load_propfirm_config()[key]

    assert cfg.ruleset_version == RULESET
    assert cfg.account_size == float(t["account"])
    assert cfg.fee == 0.0                    # fee-único NO aplica (suscripción)
    assert cfg.fee_ccy == "USD"
    assert cfg.profit_split_pct == 90.0      # ratio 90/10 sobre lo solicitado
    assert cfg.refund_fee_on_first_payout is False
    assert cfg.day_axis == "et_session"
    assert cfg.day_axis_cut_hhmm == "16:10"
    assert cfg.activation is None

    sub = cfg.subscription
    assert sub is not None
    assert sub.monthly_fee_usd == float(t["monthly"])
    assert sub.reset_fee_usd == float(t["reset"])
    assert sub.billing_period_days == 30
    assert sub.campaign_cap_days == 183      # horizonte de truncado (v2 C1)
    assert sub.funded_cap_days == 365
    assert sub.funded_start_delay_days == 3

    (ph,) = cfg.phases
    assert ph.name == "Evaluation"
    assert ph.profit_target_pct == 6.0
    assert ph.min_trading_days == 0
    assert ph.max_daily_loss_pct is None
    assert ph.max_loss_pct is None
    assert ph.daily_loss_usd == t["dll_eval"]      # Growth None / Pro hard
    assert ph.consistency is not None
    assert ph.consistency.max_best_day_share == t["share"]
    assert ph.consistency.strict is False          # "or less" = inclusiva
    assert ph.max_contracts == ContractCap(max_mini=t["mini"], max_micro=t["micro"])
    assert ph.drawdown == TrailingDrawdown(
        amount_usd=float(t["amount"]), lock_at_balance=float(t["account"]),
        frozen_floor=None, type="eod_trailing", breach_on="equity",
    )

    fnd = cfg.funded
    assert fnd.model == "futures"
    assert fnd.max_payout_cycles == 50             # cap de truncado (v2 C1)
    assert fnd.payouts is None and fnd.inactivity is None
    assert fnd.scaling_tiers == ()                 # tablas funded sin verificar
    assert fnd.drawdown == TrailingDrawdown(
        amount_usd=float(t["amount"]), lock_at_balance=float(t["account"]),
        frozen_floor=None, type="eod_trailing", breach_on="equity",
    )
    dll_type, dll_usd = t["dll_funded"]
    assert fnd.daily_loss == DailyLossSpec(
        type=dll_type, by_tier_usd=(float(dll_usd),), reset="session",
        post_hit="clamp",
    )
    pay = fnd.payouts_futures
    assert pay is not None
    assert pay.qualifying_day_min_usd == float(t["qmin"])
    assert pay.days_per_payout == t["qdays"]
    assert pay.withdraw_share == t["wshare"]
    assert pay.request_cap_usd == float(t["cap"])
    assert pay.min_new_profit_frac == 0.5
    assert pay.min_payout_usd == 20.0
    assert pay.withdrawal_policy == "greedy"
    assert pay.buffer_usd == float(t["amount"])    # default = amount Max DD


def test_perfil_base_espejo_carga(tmp_path):
    # Sanity: la spec usada para los casos de rechazo ES válida (los raises
    # de abajo no son falsos positivos de otra cosa).
    cfg = _load_spec(tmp_path, _growth_spec())
    assert cfg.subscription.monthly_fee_usd == 119.0


# ---------------------------------------------------------------------------
# AC-1: validaciones duras del loader (7 casos §2.8 + extras futures)
# ---------------------------------------------------------------------------

def test_loader_raises_frozen_floor_con_eod_trailing(tmp_path):
    def mut(spec):
        spec["funded"]["drawdown"]["frozen_floor"] = 48000.0
    _load_bad(tmp_path, mut)


def test_loader_raises_daily_loss_usd_xor_max_daily_loss_pct(tmp_path):
    def mut(spec):
        spec["phases"][0]["daily_loss_usd"] = 1000.0
        spec["phases"][0]["max_daily_loss_pct"] = 2.0
    _load_bad(tmp_path, mut)


def test_loader_raises_breach_on_invalido(tmp_path):
    def mut(spec):
        spec["funded"]["drawdown"]["breach_on"] = "flotante"
    _load_bad(tmp_path, mut)


def test_loader_raises_futures_sin_payouts_futures(tmp_path):
    def mut(spec):
        del spec["funded"]["payouts_futures"]
    _load_bad(tmp_path, mut)


def test_loader_raises_futures_sin_subscription(tmp_path):
    def mut(spec):
        del spec["subscription"]
    _load_bad(tmp_path, mut)


def test_loader_raises_cut_hhmm_sin_et_session(tmp_path):
    def mut(spec):
        del spec["day_axis"]
    _load_bad(tmp_path, mut)


def test_loader_raises_cap_menor_que_account_size_menos_amount(tmp_path):
    # v2 #10: cap (lock_at_balance) >= account_size − amount.
    # 47999 < 50000 − 2000 = 48000 → rechazo (la cuenta nacería breacheada).
    def mut(spec):
        spec["phases"][0]["drawdown"]["lock_at_balance"] = 47999.0
    _load_bad(tmp_path, mut)


def test_loader_raises_cap_ok_en_el_borde(tmp_path):
    # El borde exacto cap == account_size − amount SÍ carga.
    spec = _growth_spec()
    spec["phases"][0]["drawdown"]["lock_at_balance"] = 48000.0
    cfg = _load_spec(tmp_path, spec)
    assert cfg.phases[0].drawdown.lock_at_balance == 48000.0


def test_loader_raises_futures_drawdown_no_eod_trailing(tmp_path):
    def mut(spec):
        spec["funded"]["drawdown"] = {"type": "trailing_intraday_unrealized",
                                      "amount_usd": 2000}
    _load_bad(tmp_path, mut)


def test_loader_raises_futures_daily_loss_multitier(tmp_path):
    def mut(spec):
        spec["funded"]["daily_loss"]["by_tier_usd"] = [1000, 1500]
    _load_bad(tmp_path, mut)


def test_loader_raises_daily_loss_usd_negativo(tmp_path):
    def mut(spec):
        spec["phases"][0]["daily_loss_usd"] = -100.0
    _load_bad(tmp_path, mut)


# ---------------------------------------------------------------------------
# resolve_fee_eur con suscripción (mismo sentido de conversión que fee/activation)
# ---------------------------------------------------------------------------

def test_resolve_fee_eur_subscription_usd():
    cfg = load_propfirm_config()["ftmo_futures_growth_50k"]
    out = resolve_fee_eur(cfg, fx_eur_usd=1.10)
    assert out.fee == 0.0
    assert out.fee_ccy == "EUR"
    assert out.subscription.monthly_fee_eur == pytest.approx(119.0 / 1.10)
    assert out.subscription.reset_fee_eur == pytest.approx(109.0 / 1.10)
    assert out.subscription.monthly_fee_usd == 119.0   # lo declarado se conserva
    # El original (frozen) no se muta
    assert cfg.subscription.monthly_fee_eur is None
    assert cfg.fee_ccy == "USD"
    with pytest.raises(ValueError):
        resolve_fee_eur(cfg, fx_eur_usd=None)


def test_perfiles_existentes_intactos():
    # Regresión dentro de AC-1: Apex y FTMO CFD no cambian de parseo.
    cfgs = load_propfirm_config()
    apex = cfgs["apex_150k_intraday"]
    assert apex.day_axis_cut_hhmm == "18:00"
    assert apex.subscription is None
    assert apex.funded.payouts_futures is None
    assert apex.funded.daily_loss.by_tier_usd == (2500.0, 2500.0, 2500.0,
                                                  3000.0, 4000.0)
    ftmo = cfgs["ftmo_50k"]
    assert ftmo.day_axis is None
    assert ftmo.day_axis_cut_hhmm == "18:00"
    assert resolve_fee_eur(ftmo, fx_eur_usd=None) is ftmo
