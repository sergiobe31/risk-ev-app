"""Tests del flotante conjunto intradía del motor propfirm (``float_mode="joint"``).

Veredicto M2-1 (diagnóstico ``sqx/exp_ftmo_float_diag.py`` sobre 15 carteras
reales, 2026-08-20): la cota corr=1 intra-día (``float_mode="bound"``) sesga
P(pass) hasta +26pp a f=1.5% → se implementa el flotante conjunto con
timestamp en el motor. Cubre: coherencia de ``attach_minute_valleys`` con la
convención de valle de ``precompute_strategy``; golden 1 pata (joint ≡ bound ≡
mono en días de un solo trade vivo); caso a mano con mínimos no coincidentes;
ffill físico cross-minuto y cero tras el cierre; precompute joint vs bound
(solo difiere ``rel_float``); e integración con ``simulate_challenge``
(P(pass) joint ≥ bound en un caso construido donde la cota breacha el daily y
el conjunto no).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.mc_portfolio import (
    StrategyMC,
    attach_minute_valleys,
    precompute_strategy,
)
from src.risk.propfirm import (
    PropFirmConfig,
    PropFirmPhase,
    block_day_arrays,
    block_day_arrays_portfolio,
    build_day_blocks,
    build_phase_tape,
    build_day_blocks_portfolio,
    precompute_day_arrays_portfolio,
    simulate_challenge,
)

D = pd.Timestamp("2024-01-02").date()
D_ORD = pd.Timestamp(D).toordinal()


def _strat(key, trades, pv, lots, valles=None):
    """StrategyMC sintético (mismo patrón que tests/test_risk_propfirm_portfolio)."""
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


def _minute(entries):
    """minute_valle sintético: {trade_idx: (ts, vals, ot, ct)}."""
    out = {}
    for i, (ts, vals, ot, ct) in entries.items():
        out[i] = (pd.DatetimeIndex(ts).values, np.asarray(vals, dtype=float),
                  pd.Timestamp(ot), pd.Timestamp(ct))
    return out


def _ohlc(index, low, high, close):
    return pd.DataFrame({"o": close, "h": high, "l": low, "c": close},
                        index=pd.DatetimeIndex(index))


class _TLFake:
    """TradeList mínima para precompute_strategy (df + open_price + pnl)."""

    def __init__(self, df, open_price):
        self.df = df
        self.open_price = np.asarray(open_price, dtype=float)
        self.n_trades = len(df)

    def pnl_per_lot(self):
        return np.ones(self.n_trades)


def _strat_from_ohlc(key, trades, ohlc, pv, lots):
    """StrategyMC vía precompute_strategy + attach_minute_valleys (convención
    compartida por construcción entre ``valle`` diario y series M1)."""
    df = pd.DataFrame({
        "open_time": [t[0] for t in trades],
        "close_time": [t[1] for t in trades],
        "type": [t[3] for t in trades],
    })
    tl = _TLFake(df, [t[2] for t in trades])
    s = precompute_strategy(tl, ohlc, pv, key=key)
    s.lots = np.asarray(lots, dtype=float)
    attach_minute_valleys(s, ohlc, tl.open_price)
    return s


# ---------------------------------------------------------------------------
# 0. attach_minute_valleys reproduce la convención de valle diaria
# ---------------------------------------------------------------------------

def test_attach_minute_valleys_coherente_con_valle_diario():
    idx = pd.date_range("2024-01-02 09:00", periods=120, freq="min")
    low = np.full(120, 99.0); low[10] = 98.0; low[80] = 97.5
    ohlc = _ohlc(idx, low, np.full(120, 100.5), np.full(120, 99.0))
    s = _strat_from_ohlc("S", [
        ("2024-01-02 09:00", "2024-01-02 09:30", 100.0, "buy"),
        ("2024-01-02 10:00", "2024-01-02 10:59", 100.0, "buy"),
    ], ohlc, pv=10.0, lots=[[2.0, 2.0]])
    assert set(s.minute_valle) == set(s.valle) == {0, 1}
    for i, (ts, vals, ot, ct) in s.minute_valle.items():
        por_dia = pd.Series(vals, index=ts).groupby(pd.DatetimeIndex(ts).date).min()
        assert dict(por_dia) == s.valle[i]


# ---------------------------------------------------------------------------
# a. Golden 1 pata: joint ≡ bound ≡ mono-estrategia en días de 1 trade vivo
# ---------------------------------------------------------------------------

def test_golden_una_pata_joint_igual_bound_en_dias_un_trade():
    idx = pd.date_range("2024-01-02 09:00", periods=120, freq="min")
    low = np.full(120, 99.0); low[10] = 98.0; low[80] = 97.5
    ohlc = _ohlc(idx, low, np.full(120, 100.5), np.full(120, 99.0))
    s = _strat_from_ohlc("S", [
        ("2024-01-02 09:00", "2024-01-02 09:30", 100.0, "buy"),
        ("2024-01-02 10:00", "2024-01-02 10:59", 100.0, "buy"),
    ], ohlc, pv=10.0, lots=[[2.0, 2.0]])
    port_blocks = build_day_blocks_portfolio({0: s})
    mono_blocks = build_day_blocks(s)
    assert len(port_blocks) == len(mono_blocks) == 2  # flat entre trades
    for pb, mb in zip(port_blocks, mono_blocks):
        b_bound = block_day_arrays_portfolio(pb, {0: s.lots}, {0: s.pv}, 0,
                                             float_mode="bound")
        b_joint = block_day_arrays_portfolio(pb, {0: s.lots}, {0: s.pv}, 0,
                                             float_mode="joint",
                                             minute_by_sym={0: s.minute_valle})
        b_mono = block_day_arrays(mb, s.lots[0], s.pv)
        # 1 trade por bloque -> joint ≡ bound ≡ mono en TODOS los días.
        np.testing.assert_allclose(b_joint.day_ord, b_bound.day_ord)
        np.testing.assert_allclose(b_joint.rel_float, b_bound.rel_float,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(b_joint.rel_float, b_mono.rel_float,
                                   rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# b. Mínimos no coincidentes entre 2 patas: bound -5.0 vs joint -3.5 (a mano)
# ---------------------------------------------------------------------------

def _caso_mins_no_coincidentes():
    a = _strat("A", [("2024-01-02 09:00", "2024-01-02 09:59", 10.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -2.0}})
    b = _strat("B", [("2024-01-02 09:00", "2024-01-02 09:59", 10.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -3.0}})
    idx = pd.date_range("2024-01-02 09:00", periods=60, freq="min")
    va = np.full(60, -0.5); va[30] = -2.0   # valle de A a las 09:30
    vb = np.full(60, -0.5); vb[45] = -3.0   # valle de B a las 09:45
    mv_a = _minute({0: (idx, va, "2024-01-02 09:00", "2024-01-02 09:59")})
    mv_b = _minute({0: (idx, vb, "2024-01-02 09:00", "2024-01-02 09:59")})
    return a, b, mv_a, mv_b


def test_joint_mins_no_coincidentes_calculo_a_mano():
    a, b, mv_a, mv_b = _caso_mins_no_coincidentes()
    strats = {"A": a, "B": b}
    blk = build_day_blocks_portfolio(strats)[0]
    lots_by = {0: a.lots, 1: b.lots}
    pv_by = {0: 1.0, 1: 1.0}
    bound = block_day_arrays_portfolio(blk, lots_by, pv_by, 0, float_mode="bound")
    joint = block_day_arrays_portfolio(blk, lots_by, pv_by, 0, float_mode="joint",
                                       minute_by_sym={0: mv_a, 1: mv_b})
    i = list(bound.day_ord).index(D_ORD)
    # Cota: -2 + -3 = -5. Conjunto: 09:30 -> -2.5; 09:45 -> -3.5; resto -1.
    assert abs(bound.rel_float[i] - (-5.0)) < 1e-12
    assert abs(joint.rel_float[i] - (-3.5)) < 1e-12
    # Solo rel_float difiere; el resto de campos son idénticos.
    np.testing.assert_array_equal(joint.day_ord, bound.day_ord)
    np.testing.assert_allclose(joint.rel_closed, bound.rel_closed, rtol=0, atol=0)
    assert joint.total_pnl == bound.total_pnl


# ---------------------------------------------------------------------------
# c. ffill físico cross-minuto y cero tras el cierre
# ---------------------------------------------------------------------------

def test_joint_ffill_y_cero_tras_cierre():
    # A: barras cada minuto 09:00-09:59, trade vivo [09:00, 09:30], vp=-1.
    idx_a = pd.date_range("2024-01-02 09:00", periods=60, freq="min")
    va = np.full(60, -1.0)
    mv_a = _minute({0: (idx_a, va, "2024-01-02 09:00", "2024-01-02 09:30")})
    # B: barras solo 09:15/09:30/09:45, trade vivo [09:15, 09:45], vp=-1/-2/-3.
    idx_b = pd.DatetimeIndex(["2024-01-02 09:15", "2024-01-02 09:30",
                              "2024-01-02 09:45"])
    mv_b = _minute({0: (idx_b, [-1.0, -2.0, -3.0],
                        "2024-01-02 09:15", "2024-01-02 09:45")})
    a = _strat("A", [("2024-01-02 09:00", "2024-01-02 09:30", 10.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -1.0}})
    b = _strat("B", [("2024-01-02 09:15", "2024-01-02 09:45", 10.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -3.0}})
    blk = build_day_blocks_portfolio({"A": a, "B": b})[0]
    joint = block_day_arrays_portfolio(
        blk, {0: a.lots, 1: b.lots}, {0: 1.0, 1: 1.0}, 0,
        float_mode="joint", minute_by_sym={0: mv_a, 1: mv_b})
    i = list(joint.day_ord).index(D_ORD)
    # Suma: 09:00-14 -> -1; 09:15-29 -> -2 (ffill de B); 09:30 -> -3 (A aún
    # viva en su minuto de cierre); 09:31-44 -> -2; 09:45 -> -3; 09:46+ -> 0
    # (B cerrada: no aporta). min = -3.
    assert abs(joint.rel_float[i] - (-3.0)) < 1e-12
    # La cota suma los valles: -1 + -3 = -4.
    bound = block_day_arrays_portfolio(blk, {0: a.lots, 1: b.lots},
                                       {0: 1.0, 1: 1.0}, 0, float_mode="bound")
    assert abs(bound.rel_float[i] - (-4.0)) < 1e-12


# ---------------------------------------------------------------------------
# d. precompute_day_arrays_portfolio: joint vs bound solo difiere en rel_float
# ---------------------------------------------------------------------------

def test_precompute_joint_vs_bound_solo_rel_float():
    a, b, mv_a, mv_b = _caso_mins_no_coincidentes()
    strats = {"A": a, "B": b}
    blocks = build_day_blocks_portfolio(strats)
    lots2 = np.array([[1.0], [2.0]])
    a2 = _strat("A", [("2024-01-02 09:00", "2024-01-02 09:59", 10.0)],
                pv=1.0, lots=lots2, valles={0: {D: -2.0}})
    b2 = _strat("B", [("2024-01-02 09:00", "2024-01-02 09:59", 10.0)],
                pv=1.0, lots=lots2, valles={0: {D: -3.0}})
    blocks = build_day_blocks_portfolio({"A": a2, "B": b2})
    lots_by = {0: a2.lots, 1: b2.lots}
    pv_by = {0: 1.0, 1: 1.0}
    minute_by = {0: mv_a, 1: mv_b}
    grid_bound = precompute_day_arrays_portfolio(blocks, lots_by, pv_by, 2,
                                                 float_mode="bound")
    grid_joint = precompute_day_arrays_portfolio(blocks, lots_by, pv_by, 2,
                                                 float_mode="joint",
                                                 minute_by_sym=minute_by)
    assert len(grid_bound) == len(grid_joint) == 2
    for fi in range(2):
        for bb, bj in zip(grid_bound[fi], grid_joint[fi]):
            np.testing.assert_array_equal(bb.day_ord, bj.day_ord)
            np.testing.assert_allclose(bb.rel_before, bj.rel_before, rtol=0, atol=0)
            np.testing.assert_allclose(bb.rel_closed, bj.rel_closed, rtol=0, atol=0)
            np.testing.assert_array_equal(bb.opens_day, bj.opens_day)
            assert bb.total_pnl == bj.total_pnl
            # El conjunto nunca es peor que la cota en este caso (mismos mins
            # por minuto que por día al compartir calendario las dos patas).
            assert np.all(bj.rel_float >= bb.rel_float - 1e-12)
            assert np.any(bj.rel_float > bb.rel_float)


# ---------------------------------------------------------------------------
# e. Integración: simulate_challenge con joint pasa donde bound breacha daily
# ---------------------------------------------------------------------------

def test_simulate_challenge_joint_pass_donde_bound_breahea():
    # Un día, 2 patas: valles -1300 cada una en minutos distintos -> cota
    # -2600 (breach daily: 47400 < suelo 47500) vs conjunto -2400 (no breach).
    # PnL cerrado conjunto +5100 -> closed 55100 > target 55000.0000...07
    # (1.1 no es exacto en binario; margen para no caer en neither por float).
    idx = pd.date_range("2024-01-02 09:00", periods=60, freq="min")
    va = np.full(60, -1100.0); va[30] = -1300.0
    vb = np.full(60, -1100.0); vb[45] = -1300.0
    mv_a = _minute({0: (idx, va, "2024-01-02 09:00", "2024-01-02 09:59")})
    mv_b = _minute({0: (idx, vb, "2024-01-02 09:00", "2024-01-02 09:59")})
    a = _strat("A", [("2024-01-02 09:00", "2024-01-02 09:59", 3000.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -1300.0}})
    b = _strat("B", [("2024-01-02 09:00", "2024-01-02 09:59", 2100.0)],
               pv=1.0, lots=[[1.0]], valles={0: {D: -1300.0}})
    strats = {"A": a, "B": b}
    blocks = build_day_blocks_portfolio(strats)
    lots_by = {0: a.lots, 1: b.lots}
    pv_by = {0: 1.0, 1: 1.0}
    bound = precompute_day_arrays_portfolio(blocks, lots_by, pv_by, 1,
                                            float_mode="bound")[0]
    joint = precompute_day_arrays_portfolio(blocks, lots_by, pv_by, 1,
                                            float_mode="joint",
                                            minute_by_sym={0: mv_a, 1: mv_b})[0]
    phase = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 1)
    cfg = PropFirmConfig("t", "t", 50_000.0, 0.0, 0.0, False, (phase,), None)
    idx_matrix = np.zeros((4, len(blocks)), dtype=int)  # 4 paths idénticos
    durations = np.array([blk.duration_days for blk in blocks])
    gaps = np.zeros((4, len(blocks)))
    # Cinta (la F1 es idx_matrix; cfg de 1 fase, la prolongacion no interviene).
    tape = build_phase_tape(idx_matrix, np.zeros((4, 2 * len(blocks)), dtype=int),
                            gaps, np.zeros((4, 2 * len(blocks))))
    ch_bound = simulate_challenge(tape, bound, durations, cfg)
    ch_joint = simulate_challenge(tape, joint, durations, cfg)
    assert ch_bound.pass1.mean() == 0.0     # la cota breacha el daily
    assert ch_joint.pass1.mean() == 1.0     # el flotante conjunto, no
    assert ch_joint.pass1.mean() >= ch_bound.pass1.mean()
