"""Tests del motor propfirm a nivel CARTERA (plan propfirm-carteras v3 paso 2).

Cubre: golden de equivalencia 1 pata (cartera == mono, certifica el puerto, no
la semántica — M2-7), golden 2 patas asimétricas (pv/lotes/cierres distintos),
casos N6 (mismo día de cierre con pv distinto; matriz de lotes no cuadrada),
close-then-open en el mismo instante y cierre en el borde del bloque, paridad
BAJO-3 con contador n_no_valley, trading day por unión, breach correlacionado
end-to-end, fix de IC sobre resolved (M2-10) y el guardián count_crossing con
trade de duración 0 (artefacto del export, RESULTADOS_EXP2.md:69).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import StrategyMC, count_crossing, detect_flatflat_blocks
from src.risk.propfirm import (
    FirmPathOutcomes,
    FundedRules,
    PropFirmConfig,
    PropFirmPhase,
    block_day_arrays,
    block_day_arrays_portfolio,
    build_day_blocks,
    build_day_blocks_portfolio,
    evaluate_phase_path,
    firm_report,
    precompute_day_arrays_portfolio,
    sequence_ev,
)
from src.risk.ruin import wilson_ci

ACC = 50_000.0
PHASE = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)


def _strat(key, trades, pv, lots, valles=None):
    """StrategyMC sintético.

    trades: lista de (open, close, pnl_per_lot) con fechas str ISO.
    lots: array (n_f, n_trades). valles: {trade_idx: {date: valle_precio}}.
    """
    open_t = np.array([np.datetime64(pd.Timestamp(t[0])) for t in trades])
    close_t = np.array([np.datetime64(pd.Timestamp(t[1])) for t in trades])
    close_d = np.array([pd.Timestamp(t[1]).date() for t in trades])
    pnl = np.array([t[2] for t in trades], dtype=float)
    n = len(trades)
    return StrategyMC(
        key=key, open_t=open_t, close_t=close_t, close_d=close_d,
        valle=valles or {}, pnl_per_lot=pnl, is_buy=np.ones(n, dtype=bool),
        pv=float(pv), lots=np.asarray(lots, dtype=float), n_trades=n,
    )


def _assert_bda_equal(a, b):
    np.testing.assert_array_equal(a.day_ord, b.day_ord)
    np.testing.assert_array_equal(a.rel_before, b.rel_before)
    np.testing.assert_array_equal(a.rel_closed, b.rel_closed)
    np.testing.assert_array_equal(a.rel_float, b.rel_float)
    np.testing.assert_array_equal(a.opens_day, b.opens_day)
    assert a.total_pnl == b.total_pnl


# ---------------------------------------------------------------------------
# 1. Golden de equivalencia: cartera de 1 pata == mono-estrategia (exacto)
# ---------------------------------------------------------------------------

def _golden_strat():
    trades = [
        ("2024-01-02 10:00", "2024-01-03 15:00", 500.0),
        ("2024-01-03 16:00", "2024-01-04 09:00", -200.0),
        ("2024-01-10 10:00", "2024-01-10 18:00", 300.0),  # sin valle (BAJO-3)
    ]
    valles = {
        0: {pd.Timestamp("2024-01-02").date(): -2.0, pd.Timestamp("2024-01-03").date(): -1.0},
        1: {pd.Timestamp("2024-01-03").date(): -3.0},
    }
    lots = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]  # (n_f=2, n_trades=3)
    return _strat("g", trades, pv=100.0, lots=lots, valles=valles)


def test_golden_equivalencia_1_pata_exacta():
    s = _golden_strat()
    mono_blocks = build_day_blocks(s)
    port_blocks = build_day_blocks_portfolio({0: s})
    # 3 episodios flat-flat: t0 (cierra 15:00) y t1 (abre 16:00) están separados
    # por 1h flat; t2 va aparte.
    assert len(mono_blocks) == len(port_blocks) == 3
    for fi in range(2):
        for mb, pb in zip(mono_blocks, port_blocks):
            a = block_day_arrays(mb, s.lots[fi], s.pv)
            b = block_day_arrays_portfolio(pb, {0: s.lots}, {0: s.pv}, fi)
            _assert_bda_equal(a, b)


def test_precompute_day_arrays_portfolio_estructura():
    s = _golden_strat()
    blocks = build_day_blocks_portfolio({0: s})
    grid = precompute_day_arrays_portfolio(blocks, {0: s.lots}, {0: s.pv}, n_f=2)
    assert len(grid) == 2 and all(len(g) == len(blocks) for g in grid)
    # La f 0 del precomputo coincide con la llamada directa.
    _assert_bda_equal(
        grid[0][0], block_day_arrays_portfolio(blocks[0], {0: s.lots}, {0: s.pv}, 0)
    )


# ---------------------------------------------------------------------------
# 2. Golden 2 patas ASIMÉTRICAS (pv, lotes y días de cierre distintos)
# ---------------------------------------------------------------------------

def test_golden_2_patas_asimetricas_calculo_a_mano():
    a = _strat("A", [
        ("2024-01-02 09:00", "2024-01-03 12:00", 500.0),
        ("2024-01-04 10:00", "2024-01-04 18:00", -100.0),
    ], pv=100.0, lots=[[2.0, 2.0]], valles={
        0: {pd.Timestamp("2024-01-02").date(): -2.0},
        1: {pd.Timestamp("2024-01-04").date(): -1.0},
    })
    b = _strat("B", [
        ("2024-01-02 14:00", "2024-01-02 20:00", 50.0),
        ("2024-01-03 09:00", "2024-01-04 09:00", 200.0),
    ], pv=20.0, lots=[[3.0, 3.0]], valles={
        0: {pd.Timestamp("2024-01-02").date(): -5.0},
        1: {pd.Timestamp("2024-01-03").date(): -10.0},
    })
    strats = {0: a, 1: b}
    blocks = build_day_blocks_portfolio(strats)
    assert len(blocks) == 2  # flat 01-04 09:00→10:00 parte el busy en dos
    lots_by_sym = {0: a.lots, 1: b.lots}
    pv_by_sym = {0: 100.0, 1: 20.0}

    # Bloque 1 (a0, b0, b1): cierres 01-03, 01-02, 01-04.
    b1 = block_day_arrays_portfolio(blocks[0], lots_by_sym, pv_by_sym, 0)
    d = [pd.Timestamp(x).toordinal() for x in ("2024-01-02", "2024-01-03", "2024-01-04")]
    np.testing.assert_array_equal(b1.day_ord, d)
    # PnL cerrado: 01-02 → b0 50×3=150; 01-03 → a0 500×2=1000; 01-04 → b1 200×3=600.
    np.testing.assert_array_equal(b1.rel_closed, [150.0, 1150.0, 1750.0])
    np.testing.assert_array_equal(b1.rel_before, [0.0, 150.0, 1150.0])
    # Flotante: 01-02 → a0(-2×2×100) + b0(-5×3×20) = -700; 01-03 → b1(-10×3×20) = -600.
    np.testing.assert_array_equal(b1.rel_float, [-700.0, -450.0, 1150.0])
    np.testing.assert_array_equal(b1.opens_day, [2, 1, 0])
    assert b1.total_pnl == 1750.0

    # Bloque 2 (a1 solo): cierra 01-04, valle -1×2×100 = -200.
    b2 = block_day_arrays_portfolio(blocks[1], lots_by_sym, pv_by_sym, 0)
    np.testing.assert_array_equal(b2.day_ord, [d[2]])
    np.testing.assert_array_equal(b2.rel_closed, [-200.0])
    np.testing.assert_array_equal(b2.rel_float, [-200.0])
    np.testing.assert_array_equal(b2.opens_day, [1])


# ---------------------------------------------------------------------------
# 3. (N6a) Dos patas cerrando el MISMO día con pv distinto
# ---------------------------------------------------------------------------

def test_mismo_dia_cierre_pv_distinto_caza_pv_swap():
    a = _strat("A", [("2024-01-02 09:00", "2024-01-03 17:00", 100.0)],
               pv=100.0, lots=[[1.0]],
               valles={0: {pd.Timestamp("2024-01-03").date(): -1.0}})
    b = _strat("B", [("2024-01-02 10:00", "2024-01-03 16:00", 100.0)],
               pv=50.0, lots=[[1.0]],
               valles={0: {pd.Timestamp("2024-01-03").date(): -2.0}})
    blocks = build_day_blocks_portfolio({0: a, 1: b})
    assert len(blocks) == 1
    ba = block_day_arrays_portfolio(blocks[0], {0: a.lots, 1: b.lots},
                                    {0: 100.0, 1: 50.0}, 0)
    # Flotante 01-03 = -1×1×100 (A) + -2×1×50 (B) = -200; con pv cruzados sería -250.
    assert ba.rel_float[-1] == -200.0
    assert ba.rel_closed[-1] == 200.0


# ---------------------------------------------------------------------------
# 4. (N6b) Matriz de lotes (n_f=2, n_trades=3) — caza transpuesta
# ---------------------------------------------------------------------------

def test_matriz_lotes_no_cuadrada_usa_la_fila_f_correcta():
    trades = [
        ("2024-01-02 09:00", "2024-01-03 10:00", 1.0),
        ("2024-01-02 10:00", "2024-01-03 11:00", 1.0),
        ("2024-01-02 11:00", "2024-01-03 12:00", 1.0),
    ]
    lots = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    s = _strat("L", trades, pv=100.0, lots=lots)
    blocks = build_day_blocks_portfolio({0: s})
    assert len(blocks) == 1
    ba0 = block_day_arrays_portfolio(blocks[0], {0: s.lots}, {0: s.pv}, 0)
    ba1 = block_day_arrays_portfolio(blocks[0], {0: s.lots}, {0: s.pv}, 1)
    assert ba0.total_pnl == 6.0    # 1+2+3
    assert ba1.total_pnl == 60.0   # 10+20+30


# ---------------------------------------------------------------------------
# 5. Close-then-open en el mismo instante; cierre exactamente en el borde
# ---------------------------------------------------------------------------

def test_close_then_open_mismo_timestamp_corta_en_el_borde():
    # Semántica real de detect_flatflat_blocks (mc_portfolio.py:152-164): el
    # close (-1) se procesa antes que el open (+1) del mismo instante → active
    # llega a 0 y el episodio SE CORTA en T. El trade que cierra queda en el
    # bloque anterior y el que abre en el siguiente; nadie cruza (0 crossings).
    a = _strat("A", [("2024-01-02 09:00", "2024-01-03 12:00", 100.0)],
               pv=100.0, lots=[[1.0]])
    b = _strat("B", [("2024-01-03 12:00", "2024-01-04 09:00", 100.0)],
               pv=100.0, lots=[[1.0]])
    strats = {0: a, 1: b}
    blocks = build_day_blocks_portfolio(strats)
    assert len(blocks) == 2
    assert blocks[0].end == pd.Timestamp("2024-01-03 12:00")
    assert blocks[1].start == pd.Timestamp("2024-01-03 12:00")
    assert blocks[0].n_trades == 1 and blocks[0].sym_id[0] == 0
    assert blocks[1].n_trades == 1 and blocks[1].sym_id[0] == 1
    raw = detect_flatflat_blocks({k: (s.open_t, s.close_t) for k, s in strats.items()})
    assert count_crossing(strats, raw) == 0


def test_cierre_exactamente_en_el_borde_entra_en_el_bloque():
    a = _strat("A", [("2024-01-02 10:00", "2024-01-02 12:00", 100.0)],
               pv=100.0, lots=[[1.0]])
    b = _strat("B", [("2024-01-02 11:00", "2024-01-02 12:00", 100.0)],
               pv=100.0, lots=[[1.0]])
    blocks = build_day_blocks_portfolio({0: a, 1: b})
    assert len(blocks) == 1
    assert blocks[0].n_trades == 2  # ct == bend cumple ct <= bend


# ---------------------------------------------------------------------------
# 6. Paridad BAJO-3: trade sin valle entra, cuenta y se contabiliza
# ---------------------------------------------------------------------------

def test_bajo3_trade_sin_valle_cuenta_pnl_y_se_contabiliza():
    s = _strat("S", [
        ("2024-01-02 09:00", "2024-01-03 12:00", 400.0),   # sin valle
        ("2024-01-03 10:00", "2024-01-03 18:00", -100.0),  # con valle
    ], pv=100.0, lots=[[1.0, 1.0]],
        valles={1: {pd.Timestamp("2024-01-03").date(): -2.0}})
    blocks = build_day_blocks_portfolio({0: s})
    assert len(blocks) == 1
    assert blocks[0].n_trades == 2
    assert blocks[0].n_no_valley == 1
    ba = block_day_arrays_portfolio(blocks[0], {0: s.lots}, {0: s.pv}, 0)
    assert ba.total_pnl == 300.0          # el PnL del trade sin valle cuenta
    assert ba.rel_float[-1] == -200.0     # solo aporta flotante el trade con valle


# ---------------------------------------------------------------------------
# 7. Trading day por unión: dos patas abren el mismo día → cuenta 1
# ---------------------------------------------------------------------------

def test_trading_day_union_end_to_end():
    a = _strat("A", [("2024-01-02 09:00", "2024-01-02 17:00", 3000.0)],
               pv=100.0, lots=[[1.0]])
    b = _strat("B", [("2024-01-02 14:00", "2024-01-02 20:00", 3000.0)],
               pv=100.0, lots=[[1.0]])
    strats = {0: a, 1: b}
    blocks = build_day_blocks_portfolio(strats)
    assert len(blocks) == 1
    ba = block_day_arrays_portfolio(blocks[0], {0: a.lots, 1: b.lots},
                                    {0: 100.0, 1: 100.0}, 0)
    assert ba.opens_day[0] == 2  # dos APERTURAS ese día…
    phase1 = PropFirmPhase("T", 10.0, 5.0, 10.0, 1)
    r = evaluate_phase_path(np.array([0]), [ba], np.array([1.0]),
                            account_size=ACC, phase=phase1)
    assert r.outcome == "pass"        # 6000 ≥ target 5000
    assert r.trading_days == 1        # …pero el DÍA de trading cuenta 1


# ---------------------------------------------------------------------------
# 8. Breach correlacionado end-to-end (lo que el motor mono no puede ver)
# ---------------------------------------------------------------------------

def test_breach_correlacionado_dos_patas_mismo_dia():
    a = _strat("A", [("2024-01-02 09:00", "2024-01-03 17:00", 100.0)],
               pv=100.0, lots=[[1.0]],
               valles={0: {pd.Timestamp("2024-01-02").date(): -15.0}})
    b = _strat("B", [("2024-01-02 10:00", "2024-01-03 16:00", 100.0)],
               pv=100.0, lots=[[1.0]],
               valles={0: {pd.Timestamp("2024-01-02").date(): -15.0}})
    strats = {0: a, 1: b}
    blocks = build_day_blocks_portfolio(strats)
    ba = block_day_arrays_portfolio(blocks[0], {0: a.lots, 1: b.lots},
                                    {0: 100.0, 1: 100.0}, 0)
    r = evaluate_phase_path(np.array([0]), [ba], np.array([2.0]),
                            account_size=ACC, phase=PHASE)
    assert r.outcome == "fail"  # -3000 flotante conjunto < suelo -2500
    # Control: una sola pata (-1500) NO rompe el daily loss.
    blocks_a = build_day_blocks_portfolio({0: a})
    ba_a = block_day_arrays_portfolio(blocks_a[0], {0: a.lots}, {0: 100.0}, 0)
    r_a = evaluate_phase_path(np.array([0]), [ba_a], np.array([2.0]),
                              account_size=ACC, phase=PHASE)
    assert r_a.outcome == "neither"


# ---------------------------------------------------------------------------
# 9. Fix M2-10: IC Wilson sobre paths resolved (mismo denominador que el punto)
# ---------------------------------------------------------------------------

def _cfg():
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=345.0,
        profit_split_pct=80.0, refund_fee_on_first_payout=False,
        phases=(PHASE,),
        funded=FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0,
                           max_payout_cycles=3),
    )


def test_ic_wilson_sobre_resolved_no_sobre_todos():
    n = 10
    pass1 = np.array([True] * 6 + [False] * 4)
    oc = FirmPathOutcomes(
        pass1=pass1,
        pass2=np.ones(n, dtype=bool),
        neither1=np.array([False] * 8 + [True] * 2),  # 2 unresolved
        neither2=np.zeros(n, dtype=bool),
        funded_payouts=np.zeros((n, 3)),
        days_first_payout=np.full(n, np.nan),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.where(pass1, 10.0, np.nan),
    )
    seq = sequence_ev(oc, _cfg(), fx_eur_usd=1.0, n_challenges=5, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    assert seq.p_pass1 == pytest.approx(6 / 8)  # sobre resolved
    rep = firm_report(0.01, oc, seq)
    assert rep.p_pass1_ci == pytest.approx(wilson_ci(6, 8))
    assert rep.p_both_ci == pytest.approx(wilson_ci(6, 8))
    # El denominador viejo (todos los paths) daría un IC distinto.
    assert rep.p_pass1_ci != pytest.approx(wilson_ci(6, 10))


# ---------------------------------------------------------------------------
# 10. Guardián count_crossing: trade de duración 0 (artefacto del export)
# ---------------------------------------------------------------------------

def test_count_crossing_trade_duracion_cero_en_flat():
    s = _strat("S", [
        ("2024-01-02 09:00", "2024-01-02 17:00", 100.0),
        ("2024-01-05 12:00", "2024-01-05 12:00", 50.0),  # duración 0, en flat
    ], pv=100.0, lots=[[1.0, 1.0]])
    blocks = detect_flatflat_blocks({0: (s.open_t, s.close_t)})
    assert len(blocks) == 1
    assert count_crossing({0: s}, blocks) == 1
    # Y el trade de duración 0 no entra en ningún bloque (queda fuera, contado).
    pblocks = build_day_blocks_portfolio({0: s})
    assert len(pblocks) == 1 and pblocks[0].n_trades == 1
