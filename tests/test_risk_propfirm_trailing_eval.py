"""Tests de la evaluación trailing + economía de evaluación (plan Apex Fase 3).

Todo calculado a mano. Cubre: breach trailing sin pérdida cerrada (pico
flotante +9,000 que da back 4,000), precedencia breach > target el mismo día,
target en el último día del lifetime = pass (no expired), expiración = outcome
"expired" con EV = −fee en sequence_ev, neither (horizonte de backtest) sigue
excluido y es lo único que alimenta p_unresolved, veto de recommend_f_pair
intacto, activation fee solo en paths pass (con conversión EUR vía
resolve_fee_eur y raise si no resuelta), hwm cross-bloque (breach intradía que
un check de cerrado no vería), dispatch portfolio/mono en simulate_challenge,
raise ruidoso sin escalares trailing y regresión FTMO explícita.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.mc import extend_stationary_rows
from src.risk.propfirm import (
    ActivationSpec,
    BlockDayArrays,
    FirmPathOutcomes,
    FirmReport,
    FundedRules,
    PropFirmConfig,
    PropFirmPhase,
    TrailingDrawdown,
    evaluate_phase_path_trailing,
    ev_vector_all,
    firm_report,
    recommend_f_pair,
    build_phase_tape,
    resolve_fee_eur,
    sequence_ev,
    simulate_challenge,
)

ACC = 150_000.0
TARGET = ACC * 1.06  # 159,000
AMOUNT = 4_000.0

APEX_PHASE = PropFirmPhase(
    name="Evaluation",
    profit_target_pct=6.0,
    max_daily_loss_pct=None,
    max_loss_pct=None,
    min_trading_days=0,
    drawdown=TrailingDrawdown(amount_usd=AMOUNT),
    lifetime_days=30,
    on_horizon_exhausted="fail",
)

FUNDED_DUMMY = FundedRules(
    payout_cycle_days=14, min_payout_profit_pct=0.0, max_payout_cycles=3
)


def _tbda(days, rel_closed, rel_low, rel_high, rel_exc, opens=None):
    """BlockDayArrays sintético CON campos trailing (a mano).

    Precondiciones de plausibilidad (el constructor real las cumple):
    rel_low[d] <= rel_closed[d] (lower al último minuto = realized) y
    rel_exc[d] con cummax de bloque.
    """
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_low = np.asarray(rel_low, dtype=float)
    rel_high = np.asarray(rel_high, dtype=float)
    rel_exc = np.asarray(rel_exc, dtype=float)
    n = len(days)
    return BlockDayArrays(
        day_ord=days,
        rel_before=np.concatenate([[0.0], rel_closed[:-1]]),
        rel_closed=rel_closed,
        rel_float=np.concatenate([[0.0], rel_closed[:-1]]),
        opens_day=np.asarray(opens if opens is not None else [1] * n, dtype=int),
        total_pnl=float(rel_closed[-1]),
        rel_low=rel_low,
        rel_high=rel_high,
        rel_exc=rel_exc,
        block_low=float(rel_low.min()),
        block_high=float(rel_high.max()),
        block_exc=float(rel_exc.max()),
    )


def _eval(bas, phase=APEX_PHASE, account=ACC, order=None, durations=None,
          gaps=None):
    n = len(bas)
    return evaluate_phase_path_trailing(
        np.arange(n) if order is None else np.asarray(order),
        bas,
        np.ones(n) if durations is None else np.asarray(durations, dtype=float),
        gaps,
        account_size=account, phase=phase,
    )


def _cfg(fee=50.0, activation=None, phase=APEX_PHASE, key="apex_t"):
    return PropFirmConfig(
        key=key, display_name=key, account_size=ACC, fee=fee,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(phase,), funded=FUNDED_DUMMY, activation=activation,
    )


# ---------------------------------------------------------------------------
# Breach trailing
# ---------------------------------------------------------------------------

def test_breach_trailing_sin_perdida_cerrada():
    # Ejemplo canónico: pico flotante +9,000 que da back 4,000. El balance
    # cerrado (+5,000) nunca tocó el target (159,000) NI el suelo antiguo
    # (146,000) — y sin embargo el trailing liquida: excursión 9,000−5,000.
    ba = _tbda([1], [5000.0], rel_low=[0.0], rel_high=[9000.0], rel_exc=[4000.0])
    r = _eval([ba])
    assert r.outcome == "fail"
    assert not r.expired
    assert r.days == 1.0
    assert r.terminal == pytest.approx(ACC + 5000.0)


def test_breach_y_target_mismo_dia_gana_breach():
    # Cierra +9,500 (>= target +9,000) pero intradía: pico +12,000, valle
    # +8,000 → excursión 4,000. Orden adjudicado: breach → target.
    ba = _tbda([1], [9500.0], rel_low=[8000.0], rel_high=[12000.0],
               rel_exc=[4000.0])
    r = _eval([ba])
    assert r.outcome == "fail"


def test_hwm_cross_block_breach_intradia():
    # Bloque 0: pico +3,000, cierre +2,000 → hwm = 153,000, saldo = 2,000.
    # Bloque 1 (S = 152,000): valle rel −3,000 → (hwm−S) − rel_low = 4,000.
    # El cierre del bloque 1 es 149,500, POR ENCIMA del suelo post-pico
    # (149,000): un check de cerrado NO lo flaggearía — la cota intradía sí.
    b0 = _tbda([1], [2000.0], rel_low=[0.0], rel_high=[3000.0], rel_exc=[0.0])
    b1 = _tbda([2], [-2500.0], rel_low=[-3000.0], rel_high=[0.0], rel_exc=[500.0])
    r = _eval([b0, b1])
    assert r.outcome == "fail"
    assert r.terminal == pytest.approx(149_500.0)
    # Sin el pico del bloque 0 (hwm plano) no habría breach: 152,000−3,000
    # queda a 3,000 del ancla < 4,000.
    b0_flat = _tbda([1], [2000.0], rel_low=[0.0], rel_high=[2000.0],
                    rel_exc=[0.0])
    r2 = _eval([b0_flat, b1])
    assert r2.outcome != "fail"


# ---------------------------------------------------------------------------
# Lifetime: pass en el día 30 válido; expiración = "expired"
# ---------------------------------------------------------------------------

def test_target_en_ultimo_dia_del_lifetime_es_pass():
    # Día 30 (elapsed+1 == 30): el target se chequea ANTES que la expiración.
    ba = _tbda([1, 30], [1000.0, 9000.0],
               rel_low=[0.0, 8000.0], rel_high=[1000.0, 9000.0],
               rel_exc=[0.0, 1000.0])
    r = _eval([ba])
    assert r.outcome == "pass"
    assert not r.expired
    assert r.days == 30.0
    assert r.terminal == pytest.approx(TARGET)


def test_expiracion_sin_target_es_expired():
    # Día 31 (elapsed+1 == 31 > 30) sin target → expired (≠ neither).
    ba = _tbda([1, 31], [1000.0, 5000.0],
               rel_low=[0.0, 4000.0], rel_high=[1000.0, 5000.0],
               rel_exc=[0.0, 1000.0])
    r = _eval([ba])
    assert r.outcome == "expired"
    assert r.expired
    assert r.days == 31.0
    assert r.trading_days == 2
    assert r.terminal == pytest.approx(ACC + 5000.0)


def test_expiracion_via_gaps_wall_clock():
    # El reloj de 30 días naturales avanza por durations + gaps: bloque 1 tras
    # un gap de 29 días (elapsed_prev = 1 + 29 = 30) → su primer día es el 31.
    b0 = _tbda([1], [1000.0], rel_low=[0.0], rel_high=[1000.0], rel_exc=[0.0])
    b1 = _tbda([1], [500.0], rel_low=[0.0], rel_high=[500.0], rel_exc=[0.0])
    r = _eval([b0, b1], durations=[1.0, 1.0], gaps=np.array([29.0, 0.0]))
    assert r.outcome == "expired"
    assert r.days == 31.0
    # Sin gap el mismo par de bloques no expira (día 2) ni pasa: neither.
    r2 = _eval([b0, b1], durations=[1.0, 1.0])
    assert r2.outcome == "neither"
    assert not r2.expired


def test_min_trading_days_cero_pasa_el_dia_del_target():
    ba = _tbda([1], [9000.0], rel_low=[9000.0], rel_high=[9000.0], rel_exc=[0.0])
    r = _eval([ba])
    assert r.outcome == "pass"
    assert r.days == 1.0
    assert r.trading_days == 1


def test_fase_trailing_sin_escalares_raise_ruidoso():
    ba = BlockDayArrays(
        day_ord=np.array([1]), rel_before=np.array([0.0]),
        rel_closed=np.array([1000.0]), rel_float=np.array([0.0]),
        opens_day=np.array([1]), total_pnl=1000.0,
    )
    with pytest.raises(ValueError, match="trailing=True"):
        _eval([ba])


# ---------------------------------------------------------------------------
# sequence_ev: expired = resuelto con EV = −fee; neither sigue excluido
# ---------------------------------------------------------------------------

def _firm_outcomes(pass1, expired1=None, neither1=None, payout_usd=0.0):
    n = len(pass1)
    pass1 = np.asarray(pass1, dtype=bool)
    funded = np.zeros((n, FUNDED_DUMMY.max_payout_cycles))
    funded[pass1, 0] = payout_usd
    return FirmPathOutcomes(
        pass1=pass1, pass2=np.ones(n, dtype=bool),
        neither1=(np.zeros(n, dtype=bool) if neither1 is None
                  else np.asarray(neither1, dtype=bool)),
        neither2=np.zeros(n, dtype=bool),
        funded_payouts=funded,
        days_first_payout=np.where(pass1, 14.0, np.nan),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.where(pass1, 10.0, np.nan),
        expired1=(None if expired1 is None else np.asarray(expired1, dtype=bool)),
    )


def _seq(outcomes, cfg, **kw):
    args = dict(fx_eur_usd=1.0, n_challenges=5, n_sequences=50,
                initial_capital_eur=5000.0, seed=1)
    args.update(kw)
    return sequence_ev(outcomes, cfg, **args)


def test_expired_es_resuelto_con_ev_menos_fee():
    pass1 = np.zeros(100, dtype=bool)
    exp = np.zeros(100, dtype=bool)
    exp[:50] = True
    res = _seq(_firm_outcomes(pass1, expired1=exp), _cfg(fee=50.0))
    assert res.ev_per_attempt == pytest.approx(-50.0)
    assert res.p_expired == pytest.approx(0.5)
    assert res.p_unresolved == 0.0
    assert res.p_pass1 == 0.0


def test_todo_expired_no_dispara_el_guard_de_resueltos():
    exp = np.ones(100, dtype=bool)
    res = _seq(_firm_outcomes(np.zeros(100, dtype=bool), expired1=exp),
               _cfg(fee=50.0))
    assert res.ev_per_attempt == pytest.approx(-50.0)
    assert res.p_expired == pytest.approx(1.0)


def test_neither_excluido_y_expired_no_alimenta_p_unresolved():
    # 25 pass (payout 1,000) / 25 fail / 25 expired / 25 neither.
    n, q = 100, 25
    pass1 = np.zeros(n, dtype=bool)
    pass1[:q] = True
    exp = np.zeros(n, dtype=bool)
    exp[2 * q:3 * q] = True
    nei = np.zeros(n, dtype=bool)
    nei[3 * q:] = True
    res = _seq(_firm_outcomes(pass1, expired1=exp, neither1=nei,
                              payout_usd=1000.0),
               _cfg(fee=50.0))
    # EV sobre los 75 resueltos: (25×950 + 50×(−50)) / 75.
    assert res.ev_per_attempt == pytest.approx((25 * 950.0 - 50 * 50.0) / 75)
    assert res.p_unresolved == pytest.approx(0.25)   # SOLO neither
    assert res.p_expired == pytest.approx(25 / 75)   # condicional a resueltos
    assert res.p_pass1 == pytest.approx(25 / 75)


def test_veto_recommend_f_pair_intacto_con_expired():
    # Celda A: EV alto pero p_unresolved (neither) > 5% → vetada. Celda B:
    # p_expired = 0.9 pero p_unresolved = 0 → NO vetada (expired es resuelto).
    def _rep(f, ev, p_unres, p_exp):
        return FirmReport(
            f=f, n_paths=100, p_pass1=0.5, p_pass1_ci=(0.4, 0.6),
            p_both=0.3, p_both_ci=(0.2, 0.4), p_unresolved=p_unres,
            ev_per_attempt=ev, ev_ci95=(ev - 10, ev + 10), net_mean=ev * 5,
            p_net_negative=0.3, net_p05=-1.0, net_p50=0.0, net_p95=1.0,
            mean_payout_eur=1000.0, p_payout_given_funded=0.8,
            challenges_per_payout=3.0, median_days_first_payout=30.0,
            median_days_pass1=15.0, alive_at_cap_frac=0.1, p_expired=p_exp,
        )

    reports = {(0.01, 0.01): _rep(0.01, 500.0, 0.20, 0.0),
               (0.02, 0.02): _rep(0.02, 100.0, 0.0, 0.9)}
    n = 10
    cells = {
        (0.01, 0.01): (np.full(n, 500.0), np.ones(n, dtype=bool)),
        (0.02, 0.02): (np.full(n, 100.0), np.ones(n, dtype=bool)),
    }
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    assert rec is not None
    # La celda vetada (neither) no puede ganar aunque su EV sea 5× mayor; la
    # de p_expired=0.9 sobrevive al veto (expired no es "sin resolver").
    assert rec.f_challenge == 0.02
    assert rec.p_unresolved == 0.0
    assert rec.feasible


# ---------------------------------------------------------------------------
# Activation fee: solo paths pass, EUR resuelta, raise si no
# ---------------------------------------------------------------------------

def test_activation_solo_la_pagan_los_paths_pass():
    act = ActivationSpec(fee_usd=59.0, deadline_days=7, fee_eur=50.0)
    cfg = _cfg(fee=40.0, activation=act)
    pass1 = np.zeros(100, dtype=bool)
    pass1[:50] = True
    ev = ev_vector_all(_firm_outcomes(pass1, payout_usd=1000.0), cfg,
                       fx_eur_usd=1.0)
    # pass: −40 (fee) + 1,000 (payout) − 50 (activation) = 910.
    assert ev[0] == pytest.approx(910.0)
    # fail: −40, SIN activation.
    assert ev[99] == pytest.approx(-40.0)
    res = _seq(_firm_outcomes(pass1, payout_usd=1000.0), cfg)
    assert res.ev_per_attempt == pytest.approx((50 * 910.0 - 50 * 40.0) / 100)


def test_activation_conversion_eur_via_resolve_fee_eur():
    cfg_usd = PropFirmConfig(
        key="apex_t", display_name="apex_t", account_size=ACC, fee=49.80,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(APEX_PHASE,), funded=FUNDED_DUMMY, fee_ccy="USD",
        activation=ActivationSpec(fee_usd=59.0, deadline_days=7),
    )
    cfg = resolve_fee_eur(cfg_usd, fx_eur_usd=1.10)
    pass1 = np.ones(50, dtype=bool)
    ev = ev_vector_all(_firm_outcomes(pass1, payout_usd=1100.0), cfg,
                       fx_eur_usd=1.10)
    # (−49.80 + 1,100 − 59) / 1.10 = 901.0909...
    assert ev[0] == pytest.approx((1100.0 - 49.80 - 59.0) / 1.10)


def test_activation_sin_resolver_raise():
    cfg = _cfg(fee=40.0, activation=ActivationSpec(fee_usd=59.0, deadline_days=7))
    oc = _firm_outcomes(np.ones(10, dtype=bool), payout_usd=1000.0)
    with pytest.raises(ValueError, match="resolve_fee_eur"):
        ev_vector_all(oc, cfg, fx_eur_usd=1.0)
    with pytest.raises(ValueError, match="resolve_fee_eur"):
        _seq(oc, cfg)


# ---------------------------------------------------------------------------
# Dispatch en simulate_challenge
# ---------------------------------------------------------------------------

def _blocks_dispatch():
    b_fail = _tbda([1], [5000.0], rel_low=[0.0], rel_high=[9000.0],
                   rel_exc=[4000.0])
    b_pass = _tbda([1], [9000.0], rel_low=[9000.0], rel_high=[9000.0],
                   rel_exc=[0.0])
    return [b_fail, b_pass]


def _tape(idx, n_trades):
    """PhaseTape sintetica (sin gaps) para el dispatch: la F1 es ``idx``."""
    ext = extend_stationary_rows(idx, n_trades, 1.0, 2 * idx.shape[1],
                                 np.random.default_rng(0))
    return build_phase_tape(idx, ext, None, None)


def test_dispatch_trailing_portfolio_end_to_end():
    day_arrays = _blocks_dispatch()
    idx = np.array([[0], [1]])  # path 0 → bloque breach; path 1 → bloque pass
    ch = simulate_challenge(_tape(idx, 2), day_arrays, np.ones(2), _cfg(),
                            portfolio=True)
    np.testing.assert_array_equal(ch.pass1, [False, True])
    np.testing.assert_array_equal(ch.expired1, [False, False])
    np.testing.assert_array_equal(ch.neither1, [False, False])
    assert np.isnan(ch.days_to_pass1[0])
    assert ch.days_to_pass1[1] == pytest.approx(1.0)
    # expired es resuelto: no alimenta unresolved.
    assert ch.resolved.all()


def test_dispatch_trailing_portfolio_expiracion():
    b_exp = _tbda([1, 31], [1000.0, 5000.0],
                  rel_low=[0.0, 4000.0], rel_high=[1000.0, 5000.0],
                  rel_exc=[0.0, 1000.0])
    idx = np.zeros((3, 1), dtype=int)
    ch = simulate_challenge(_tape(idx, 1), [b_exp], np.ones(1), _cfg(),
                            portfolio=True)
    np.testing.assert_array_equal(ch.expired1, [True, True, True])
    assert not ch.pass1.any() and not ch.unresolved.any()


def test_dispatch_trailing_mono_raise_claro():
    day_arrays = _blocks_dispatch()
    idx = np.array([[0]])
    with pytest.raises(ValueError, match="portfolio"):
        simulate_challenge(_tape(idx, 2), day_arrays, np.ones(2), _cfg())
    with pytest.raises(ValueError, match="portfolio"):
        simulate_challenge(_tape(idx, 2), day_arrays, np.ones(2), _cfg(),
                           portfolio=False)


# ---------------------------------------------------------------------------
# Regresión FTMO explícita: sequence_ev + firm_report byte-idénticos
# ---------------------------------------------------------------------------

def test_regresion_ftmo_sequence_ev_y_firm_report():
    phase_ftmo = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)
    cfg = PropFirmConfig(
        key="ftmo_t", display_name="ftmo_t", account_size=50_000.0, fee=345.0,
        profit_split_pct=80.0, refund_fee_on_first_payout=True,
        phases=(phase_ftmo,), funded=FUNDED_DUMMY,
    )
    n = 100
    pass1 = np.zeros(n, dtype=bool)
    pass1[:60] = True
    oc = _firm_outcomes(pass1, payout_usd=1000.0)
    res = _seq(oc, cfg)
    # pass: −345 + 1,000 + 345 (refund) = 1,000; fail: −345.
    assert res.ev_per_attempt == pytest.approx(0.6 * 1000.0 - 0.4 * 345.0)
    assert res.p_pass1 == pytest.approx(0.6)
    assert res.p_both == pytest.approx(0.6)
    assert res.p_unresolved == 0.0
    assert res.p_expired == 0.0  # FTMO no expira: diagnóstico en 0
    rep = firm_report(0.01, oc, res)
    assert rep.p_pass1 == pytest.approx(0.6)
    assert rep.ev_per_attempt == pytest.approx(res.ev_per_attempt)
    assert rep.p_expired == 0.0
    assert rep.p_pass1_ci[0] < 0.6 < rep.p_pass1_ci[1]
