"""Tests del barrido 2D (f_challenge, f_funded) — PLAN_RECOMMEND_F_PAIR.

T1 regresion legacy (goldens del simulate_firm_paths original), T2 equivalencia
refactor (wrapper == challenge+sweep+combine), T3 CRN (mascara depende solo de
f_c), T4 IC honesto con N_f=1, T5 veto por p_unresolved, T6 filtros simetricos
en la diagonal, T7 flag on_boundary, T8 fallback sin factibles. Mas la
consistencia de ev_vector_all con sequence_ev.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk import propfirm as _pf
from src.risk.mc import extend_stationary_rows
from src.risk.propfirm import (
    BlockDayArrays,
    FirmPathOutcomes,
    FirmReport,
    FundedRules,
    PropFirmConfig,
    PropFirmPhase,
    combine_challenge_funded,
    ev_vector_all,
    firm_report_pair,
    recommend_f_pair,
    sequence_ev,
    _draw_gap_tape,
    build_phase_tape,
    simulate_challenge,
    simulate_firm_paths,
    sweep_funded,
)

ACC = 50_000.0
PHASE = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)
PHASE2 = PropFirmPhase("Verification", 5.0, 5.0, 10.0, 4)
FUNDED = FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0, max_payout_cycles=3)


def _bda(days, rel_closed, rel_float=None, opens=None):
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_before = np.concatenate([[0.0], rel_closed[:-1]])
    if rel_float is None:
        rel_float = rel_before.copy()
    return BlockDayArrays(
        day_ord=days, rel_before=rel_before, rel_closed=rel_closed,
        rel_float=np.asarray(rel_float, dtype=float),
        opens_day=np.asarray(opens if opens is not None else [1] * len(days), dtype=int),
        total_pnl=float(rel_closed[-1]),
    )


def _cfg(refund=True):
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=345.0,
        profit_split_pct=80.0, refund_fee_on_first_payout=refund,
        phases=(PHASE, PHASE2), funded=FUNDED,
    )


BLOCKS = [
    _bda([1, 2], [500.0, 1500.0]),      # b0: win pequeno
    _bda([10], [4000.0]),               # b1: win grande
    _bda([20, 21], [-1000.0, -2600.0]),  # b2: perdida (2x -> breach)
    _bda([30], [200.0]),                # b3: minimo
]
DURATIONS = np.array([2.0, 1.0, 2.0, 1.0])
IDX = np.array([
    [0, 1, 2],
    [2, 2, 1],
    [1, 0, 3],
    [2, 3, 0],
    [0, 3, 1],
    [1, 2, 0],
    [3, 3, 0],
    [0, 1, 1],
])
GAP_DAYS = np.array([0.0, 3.0, 7.0])
SEED = 7
# Continuacion de la secuencia (plan ftmo-fases-secuencias): la F2 y la fondeada son
# ventanas posteriores de la fila prolongada. Cinta sintetica determinista.
IDX_EXT = extend_stationary_rows(IDX, len(BLOCKS), 2.0, 2 * IDX.shape[1],
                                 np.random.default_rng(11))


def _tape(seed=SEED):
    g1, g_ext = _draw_gap_tape(np.random.default_rng(seed), GAP_DAYS, *IDX.shape)
    return build_phase_tape(IDX, IDX_EXT, g1, g_ext)


def _legacy(monkeypatch):
    """Reconstruye el comportamiento anterior: F2 y fondeada sobre la fila de la F1."""
    monkeypatch.setattr(_pf, "_next_start", lambda e: 0)


def _assert_outcomes_equal(a: FirmPathOutcomes, b: FirmPathOutcomes):
    for f in ("pass1", "pass2", "neither1", "neither2", "alive_at_cap"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    for f in ("funded_payouts", "days_first_payout", "days_to_pass1"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f),
                                   equal_nan=True, err_msg=f)


# ---------------------------------------------------------------------------
# T1: regresion legacy — goldens del simulate_firm_paths original (pre-refactor)
# ---------------------------------------------------------------------------

# Goldens de la continuacion (plan ftmo-fases-secuencias; regresion de la cinta sintetica).
GOLDEN_CONT_PASS2 = [False, True, True, True, True, True, True, False]
GOLDEN_CONT_NEITHER2 = [True, False, False, False, False, False, False, True]
GOLDEN_CONT_PAYOUTS = [[0.0, 0.0, 0.0]] * 4 + [[1280.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 3
GOLDEN_CONT_DFP = [np.nan] * 4 + [14.0] + [np.nan] * 3


def _golden_f1(out):
    """Goldens de la F1 (identicos antes y despues de la continuacion: I1)."""
    np.testing.assert_array_equal(
        out.pass1, [True, False, True, False, True, False, False, True])
    np.testing.assert_array_equal(
        out.neither1, [False, False, False, True, False, True, True, False])
    np.testing.assert_allclose(
        out.days_to_pass1, [14.0, np.nan, 11.0, np.nan, 11.0, np.nan, np.nan, 7.0],
        equal_nan=True)


def test_t1_wrapper_reproduce_goldens_legacy(monkeypatch):
    """Con ``_next_start`` -> 0 (F2 y fondeada sobre la fila de la F1) el wrapper
    reproduce BIT A BIT los goldens del simulate_firm_paths original."""
    _legacy(monkeypatch)
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(),
                              gap_days=GAP_DAYS, seed=SEED, idx_ext=IDX_EXT)
    _golden_f1(out)
    np.testing.assert_array_equal(
        out.pass2, [True, False, True, False, True, True, False, True])
    np.testing.assert_array_equal(out.neither2, np.zeros(8, dtype=bool))
    np.testing.assert_allclose(
        out.funded_payouts,
        [[3600.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 7, equal_nan=True)
    np.testing.assert_allclose(
        out.days_first_payout, [14.0] + [np.nan] * 7, equal_nan=True)
    np.testing.assert_array_equal(out.alive_at_cap, np.zeros(8, dtype=bool))


def test_t1_goldens_continuacion():
    """Re-baseline (plan ftmo-fases-secuencias): la F2 y la fondeada continuan la fila.
    La F1 NO cambia (mismos goldens que arriba)."""
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(),
                              gap_days=GAP_DAYS, seed=SEED, idx_ext=IDX_EXT)
    _golden_f1(out)
    np.testing.assert_array_equal(
        out.pass2, GOLDEN_CONT_PASS2)
    np.testing.assert_array_equal(out.neither2, GOLDEN_CONT_NEITHER2)
    np.testing.assert_allclose(out.funded_payouts, GOLDEN_CONT_PAYOUTS, equal_nan=True)
    np.testing.assert_allclose(out.days_first_payout, GOLDEN_CONT_DFP, equal_nan=True)
    np.testing.assert_array_equal(out.alive_at_cap, np.zeros(8, dtype=bool))


# ---------------------------------------------------------------------------
# T2: equivalencia refactor — wrapper == challenge + sweep + combine
# ---------------------------------------------------------------------------

def test_t2_wrapper_legacy_draw_equivale_a_gaps_inyectados():
    # El sorteo interno del wrapper (default_rng(seed) -> _draw_gap_tape) es el legacy;
    # los gaps de la F1 son la MISMA llamada rng.choice de siempre.
    g = np.random.default_rng(SEED).choice(GAP_DAYS, size=IDX.shape, replace=True)
    g1, g_ext = _draw_gap_tape(np.random.default_rng(SEED), GAP_DAYS, *IDX.shape)
    np.testing.assert_array_equal(g1, g)
    a = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(),
                            gap_days=GAP_DAYS, seed=SEED, idx_ext=IDX_EXT)
    b = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(), gaps_fixed=g1,
                            idx_ext=IDX_EXT, gaps_ext=g_ext)
    _assert_outcomes_equal(a, b)


def test_t2_wrapper_equivale_a_challenge_sweep_combine():
    tape = _tape()
    ref = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(),
                              gap_days=GAP_DAYS, seed=SEED, idx_ext=IDX_EXT)
    ch = simulate_challenge(tape, BLOCKS, DURATIONS, _cfg())
    fu = sweep_funded(tape=tape, funded_start=ch.funded_start,
                      day_arrays_f_grid=[BLOCKS], passers_mask=ch.both,
                      durations=DURATIONS, cfg=_cfg(), f_grid=[0.0])[0.0]
    _assert_outcomes_equal(ref, combine_challenge_funded(ch, fu))


# ---------------------------------------------------------------------------
# T3: CRN — la mascara de pase depende solo de f_c; la fondeada solo de passers
# ---------------------------------------------------------------------------

def test_t3_mascara_solo_depende_de_fc_y_fondeada_solo_en_passers():
    tape = _tape()
    cfg = _cfg()
    ch = simulate_challenge(tape, BLOCKS, DURATIONS, cfg)
    # Dos "f_f" distintos (misma grilla de arrays duplicada): las mascaras no
    # pueden cambiar; solo cambiaria el monto si cambiaran los arrays.
    fus = sweep_funded(tape=tape, funded_start=ch.funded_start,
                       day_arrays_f_grid=[BLOCKS, BLOCKS], passers_mask=ch.both,
                       durations=DURATIONS, cfg=cfg, f_grid=[0.01, 0.02])
    oc_a = combine_challenge_funded(ch, fus[0.01])
    oc_b = combine_challenge_funded(ch, fus[0.02])
    np.testing.assert_array_equal(oc_a.both, oc_b.both)
    np.testing.assert_array_equal(oc_a.resolved, oc_b.resolved)
    np.testing.assert_array_equal(oc_a.both, ch.both)
    # Fondeada evaluada SOLO en passers: filas de no-passers a cero.
    passers = ch.both
    assert not passers.all()  # el escenario tiene no-passers (fail/neither)
    np.testing.assert_array_equal(fus[0.01].funded_payouts[~passers], 0.0)
    assert np.isnan(fus[0.01].days_first_payout[~passers]).all()


# ---------------------------------------------------------------------------
# Helpers para recommend_f_pair
# ---------------------------------------------------------------------------

def _report_2d(f_c, f_f, ev, p_neg=0.3, p_unres=0.0, alive=0.1):
    return FirmReport(
        f=f_c, n_paths=100, p_pass1=0.5, p_pass1_ci=(0.4, 0.6),
        p_both=0.3, p_both_ci=(0.2, 0.4), p_unresolved=p_unres,
        ev_per_attempt=ev, ev_ci95=(ev - 10, ev + 10), net_mean=ev * 10,
        p_net_negative=p_neg, net_p05=-1, net_p50=0, net_p95=1,
        mean_payout_eur=1000.0, p_payout_given_funded=0.8,
        challenges_per_payout=3.0, median_days_first_payout=30.0,
        median_days_pass1=15.0, alive_at_cap_frac=alive, seq=None,
        f_challenge=f_c, f_funded=f_f,
    )


def _cell_const(ev, n=50, resolved=True):
    """Celda (ev_vec_all, resolved) degenerada: IC del ΔEV de ancho cero."""
    return (np.full(n, float(ev)), np.full(n, resolved, dtype=bool))


def _grid(evs: dict, **kw):
    """reports_2d + cells a partir de {(f_c,f_f): ev}. kw: p_neg/p_unres por celda."""
    reports, cells = {}, {}
    for k, ev in evs.items():
        reports[k] = _report_2d(k[0], k[1], ev,
                                p_neg=kw.get("p_neg", {}).get(k, 0.3),
                                p_unres=kw.get("p_unres", {}).get(k, 0.0))
        cells[k] = _cell_const(ev)
    return reports, cells


# ---------------------------------------------------------------------------
# T4: IC honesto — con N_f=1, ΔEV == 0 y el IC contiene 0
# ---------------------------------------------------------------------------

def test_t4_grilla_de_una_celda_delta_cero():
    reports, cells = _grid({(0.01, 0.01): 100.0})
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    assert rec is not None
    assert rec.f_challenge == 0.01 and rec.f_funded == 0.01
    assert rec.delta_ev == pytest.approx(0.0)
    assert rec.delta_ev_ci[0] <= 0.0 <= rec.delta_ev_ci[1]
    assert not rec.recommend_pair


# ---------------------------------------------------------------------------
# T5: veto por p_unresolved — la celda con EV maximo pero neither alto no gana
# ---------------------------------------------------------------------------

def test_t5_veto_p_unresolved():
    reports, cells = _grid(
        {(0.01, 0.01): 100.0, (0.01, 0.02): 120.0,
         (0.02, 0.01): 130.0, (0.02, 0.02): 999.0},
        p_unres={(0.02, 0.02): 0.5},
    )
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    assert (rec.f_challenge, rec.f_funded) == (0.02, 0.01)
    assert rec.ev_pair == pytest.approx(130.0)
    assert rec.recommend_pair  # IC degenerado (30, 30) excluye 0


# ---------------------------------------------------------------------------
# T6: filtros simetricos — la diagonal optima sin filtro queda vetada
# ---------------------------------------------------------------------------

def test_t6_diagonal_con_mismos_filtros():
    reports, cells = _grid(
        {(0.01, 0.01): 100.0, (0.01, 0.02): 150.0,
         (0.02, 0.01): 140.0, (0.02, 0.02): 500.0},
        p_unres={(0.02, 0.02): 0.5},
    )
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    # Si la diagonal no aplicara el veto, ev_single seria 500.
    assert rec.ev_single == pytest.approx(100.0)
    assert (rec.f_challenge, rec.f_funded) == (0.01, 0.02)
    assert rec.delta_ev == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# T7: on_boundary — optimo en el borde de la grilla
# ---------------------------------------------------------------------------

def test_t7_optimo_en_esquina_marca_flag():
    reports, cells = _grid(
        {(0.01, 0.01): 100.0, (0.01, 0.02): 150.0,
         (0.02, 0.01): 140.0, (0.02, 0.02): 160.0},
    )
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    assert (rec.f_challenge, rec.f_funded) == (0.02, 0.02)
    assert rec.on_boundary == (True, True)


# ---------------------------------------------------------------------------
# T8: fallback — ninguna celda factible -> menor P(neto<0)
# ---------------------------------------------------------------------------

def test_t8_fallback_sin_factible():
    reports, cells = _grid(
        {(0.01, 0.01): 100.0, (0.02, 0.02): 200.0},
        p_neg={(0.01, 0.01): 0.8, (0.02, 0.02): 0.9},
    )
    rec = recommend_f_pair(reports, cells, tol_p_negative=0.5)
    assert not rec.feasible
    assert (rec.f_challenge, rec.f_funded) == (0.01, 0.01)


def test_recommend_f_pair_vacio_devuelve_none():
    assert recommend_f_pair({}, {}, tol_p_negative=0.5) is None


# ---------------------------------------------------------------------------
# ev_vector_all: misma formula que sequence_ev
# ---------------------------------------------------------------------------

def _outcomes(pass1, pass2, payouts, n=100, neither1=False):
    pass1 = np.full(n, pass1, dtype=bool)
    pass2 = np.full(n, pass2, dtype=bool)
    ne1 = np.zeros(n, dtype=bool)
    if neither1:
        ne1[n // 2:] = True
        pass1[n // 2:] = False
    funded = np.zeros((n, 3))
    funded[:, 0] = payouts
    return FirmPathOutcomes(
        pass1=pass1, pass2=pass2, neither1=ne1, neither2=np.zeros(n, dtype=bool),
        funded_payouts=funded,
        days_first_payout=np.where(np.asarray(payouts) > 0, np.full(n, 14.0), np.full(n, np.nan)),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.where(pass1, 10.0, np.nan),
    )


def test_ev_vector_all_coincide_con_sequence_ev():
    cfg = _cfg(refund=False)
    oc = _outcomes(True, True, 1000.0, neither1=True)
    ev_all = ev_vector_all(oc, cfg, fx_eur_usd=1.0)
    seq = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    assert ev_all[oc.resolved].mean() == pytest.approx(seq.ev_per_attempt)
    # Solo los pass cobran: -345 fee + 1000 payout = 655 en resueltos.
    assert ev_all[oc.resolved].mean() == pytest.approx(655.0)
    assert (ev_all[~oc.resolved] == -345.0).all()
