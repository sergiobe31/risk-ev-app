"""Fase 2 (plan ftmo-futures-ui): adaptador de reporte futures (puro).

Tests unitarios de ``poc_web/api/futures_report.py`` sobre outcomes
sintéticos (builder copiado de ``tests/test_risk_propfirm_futures_economics``
— el motor está congelado y aquí solo se traduce su salida):

- mapeo de campos (aliasing ev_per_attempt == ev_per_campaign,
  median_days_pass1 == median_days_to_pass, ICs wilson a mano sobre los N
  paths completos, warnings del motor propagadas, n_paths = paths);
- sizing_invalid con umbral ESTRICTO (== tol no invalida, > tol invalida +
  warning SIZING_INVALID);
- withdrawal_policy_cfg: replace que solo toca ese campo.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_futures_report.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.propfirm import (
    ContractCapDiag,
    DailyLossSpec,
    FirmPathOutcomes,
    FundedRules,
    FuturesPayouts,
    PropFirmConfig,
    PropFirmPhase,
    SubscriptionSpec,
    TrailingDrawdown,
    wilson_ci,
)
from src.risk.propfirm_futures import sequence_ev_futures

from poc_web.api.futures_report import firm_report_futures, withdrawal_policy_cfg

FX = 1.10
ACC = 50_000.0
MONTHLY = 119.0
RESET = 109.0
CAP_DAYS = 183.0


def _eval_phase() -> PropFirmPhase:
    return PropFirmPhase(
        name="Evaluation", profit_target_pct=6.0, max_daily_loss_pct=None,
        max_loss_pct=None, min_trading_days=0,
        drawdown=TrailingDrawdown(amount_usd=2000, lock_at_balance=ACC,
                                  type="eod_trailing", breach_on="equity"),
    )


def _funded_rules() -> FundedRules:
    return FundedRules(
        payout_cycle_days=0, min_payout_profit_pct=0.0, max_payout_cycles=50,
        model="futures",
        drawdown=TrailingDrawdown(amount_usd=2000, lock_at_balance=ACC,
                                  type="eod_trailing", breach_on="equity"),
        daily_loss=DailyLossSpec(type="pause_not_fatal", by_tier_usd=(1000.0,),
                                 post_hit="clamp"),
        payouts_futures=FuturesPayouts(
            qualifying_day_min_usd=150.0, days_per_payout=4,
            withdraw_share=0.50, request_cap_usd=2500.0,
            min_new_profit_frac=0.5, min_payout_usd=20.0,
            withdrawal_policy="greedy", buffer_usd=2000.0),
    )


def _cfg() -> PropFirmConfig:
    sub = SubscriptionSpec(
        monthly_fee_usd=MONTHLY, reset_fee_usd=RESET,
        billing_period_days=30, campaign_cap_days=int(CAP_DAYS),
        funded_cap_days=365, funded_start_delay_days=3,
        monthly_fee_eur=MONTHLY / FX, reset_fee_eur=RESET / FX,
    )
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=0.0,
        profit_split_pct=90.0, refund_fee_on_first_payout=False,
        phases=(_eval_phase(),), funded=_funded_rules(), subscription=sub,
        fee_ccy="EUR", ruleset_version="ftmo_futures_verified_2026-09-29",
        day_axis="et_session", day_axis_cut_hhmm="16:10",
    )


def _outcomes(pass1, neither1, days_eval1, payouts=None, alive_at_cap=None,
              days_first_payout=None, dll_hits=None,
              payout_deferrals=None) -> FirmPathOutcomes:
    n = len(pass1)
    pass1 = np.asarray(pass1, dtype=bool)
    neither1 = np.asarray(neither1, dtype=bool)
    days_eval1 = np.asarray(days_eval1, dtype=float)
    return FirmPathOutcomes(
        pass1=pass1, pass2=np.ones(n, dtype=bool),
        neither1=neither1, neither2=np.zeros(n, dtype=bool),
        funded_payouts=(np.asarray(payouts, dtype=float).reshape(n, -1)
                        if payouts is not None
                        else np.zeros((n, 3))),
        days_first_payout=(np.asarray(days_first_payout, dtype=float)
                           if days_first_payout is not None
                           else np.full(n, np.nan)),
        alive_at_cap=(np.asarray(alive_at_cap, dtype=bool)
                      if alive_at_cap is not None
                      else np.zeros(n, dtype=bool)),
        days_to_pass1=np.where(pass1, days_eval1, np.nan),
        expired1=np.zeros(n, dtype=bool), expired2=np.zeros(n, dtype=bool),
        dll_hits=(np.asarray(dll_hits, dtype=float)
                  if dll_hits is not None else None),
        payout_deferrals=(np.asarray(payout_deferrals, dtype=float)
                          if payout_deferrals is not None else None),
        days_eval1=days_eval1,
    )


def _seq(outcomes, n_paths=60, seed=0, reset_policy="billing_retry"):
    return sequence_ev_futures(
        outcomes, _cfg(), fx_eur_usd=FX, n_paths=n_paths, n_boot=500,
        initial_capital_eur=5_000.0, seed=seed, reset_policy=reset_policy,
    )


# --- Mapeo de campos ----------------------------------------------------------
def test_mapeo_campos():
    # Pool: pass con payout / fail rápido / 3 neither (HIGH_NEITHER_ATTEMPTS).
    oc = _outcomes(
        pass1=[True, False, False, False, False],
        neither1=[False, False, True, True, True],
        days_eval1=[20.0, 10.0, 30.0, 30.0, 30.0],
        payouts=[[300.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        alive_at_cap=[False, False, False, False, False],
        days_first_payout=[35.0, np.nan, np.nan, np.nan, np.nan],
        dll_hits=[0, 1, 0, 0, 0],
        payout_deferrals=[0, 0, 0, 0, 0],
    )
    seq = _seq(oc, n_paths=100, seed=0)
    assert seq.warnings and any(w.startswith("HIGH_NEITHER_ATTEMPTS")
                                for w in seq.warnings)
    r = firm_report_futures(0.005, oc, seq, cfg=_cfg(),
                            reset_policy="billing_retry",
                            withdrawal_policy="greedy")
    # Aliasing futures (meta.ev_semantics lo etiqueta).
    assert r.ev_per_attempt == seq.ev_per_campaign
    assert r.median_days_pass1 == seq.median_days_to_pass
    assert r.ev_ci95 == tuple(seq.ev_ci95)
    assert r.net_mean == seq.net_mean_eur
    assert r.p_net_negative == seq.p_net_negative
    assert r.median_days_first_payout == seq.median_days_first_payout
    assert r.alive_at_cap_frac == seq.alive_at_cap_frac
    assert r.p_payout_given_funded == seq.p_payout_given_funded
    # n_paths = paths del backtest, NO campañas (E9).
    assert r.n_paths == 5 == len(oc.pass1)
    assert r.n_paths != seq.n_paths
    # ICs wilson INCONDICIONALES sobre los N paths completos (1/5 pass,
    # 3/5 neither) — a mano contra wilson_ci directo.
    assert r.p_pass1 == pytest.approx(0.2)
    assert r.p_pass1_ci == wilson_ci(1, 5)
    assert r.p_both == pytest.approx(0.2)
    assert r.p_both_ci == wilson_ci(1, 5)
    assert r.p_unresolved == pytest.approx(0.6)
    # Warnings del motor propagadas.
    for w in seq.warnings:
        assert w in r.warnings
    # Políticas ecoadas; sin diagnóstico de cap acoplado no hay sizing.
    assert r.reset_policy == "billing_retry"
    assert r.withdrawal_policy == "greedy"
    assert r.sizing_invalid is False
    assert np.isnan(r.frac_cap_violations_eval)


# --- sizing_invalid: umbral ESTRICTO -------------------------------------------
def test_sizing_invalid_umbral_estricto():
    fail10 = _outcomes([False], [False], [10.0])
    seq = _seq(fail10, n_paths=20)
    diag = ContractCapDiag(scope="evaluation", cap_micros=50.0, n_paths=1,
                           frac_violating=0.5, max_observed=60.0,
                           path_max=np.array([60.0]))
    fail10.contract_cap_eval = diag
    kw = dict(cfg=_cfg(), reset_policy="billing_retry",
              withdrawal_policy="greedy")
    # frac == tol: NO invalida (umbral estricto, cap inclusivo).
    r_eq = firm_report_futures(0.005, fail10, seq, cap_violation_tol=0.5, **kw)
    assert r_eq.sizing_invalid is False
    assert r_eq.frac_cap_violations_eval == 0.5
    assert not any(w.startswith("SIZING_INVALID") for w in r_eq.warnings)
    # frac > tol: invalida + warning con el formato del motor.
    r_gt = firm_report_futures(0.005, fail10, seq, cap_violation_tol=0.49, **kw)
    assert r_gt.sizing_invalid is True
    assert any(w.startswith("SIZING_INVALID[evaluation]")
               for w in r_gt.warnings)
    # Sin diagnóstico acoplado: silencio (cap PA no aplica a futures).
    fail10.contract_cap_eval = None
    r_none = firm_report_futures(0.005, fail10, seq, cap_violation_tol=0.0,
                                 **kw)
    assert r_none.sizing_invalid is False
    assert np.isnan(r_none.frac_cap_violations_eval)


# --- withdrawal_policy_cfg: replace quirúrgico ---------------------------------
def test_withdrawal_policy_cfg_replace():
    cfg = _cfg()
    assert cfg.funded.payouts_futures.withdrawal_policy == "greedy"
    new = withdrawal_policy_cfg(cfg, "buffer_preserving")
    assert new.funded.payouts_futures.withdrawal_policy == "buffer_preserving"
    # Round-trip: reverter deja la cfg IGUAL — solo ese campo cambió.
    assert withdrawal_policy_cfg(new, "greedy") == cfg
    assert cfg.funded.payouts_futures.withdrawal_policy == "greedy"
