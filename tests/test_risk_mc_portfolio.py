"""Tests del MC por bloques flat-flat para portfolios (src/risk/mc_portfolio.py).

Datos sintéticos (no dependen de asset_data ni de los CSVs del Desktop). Cubren:
  - detect_flatflat_blocks: cortes correctos.
  - V2: count_crossing == 0 con datos coherentes.
  - V1: merged_series_identity == composición continua (bit-exact).
  - evaluate_path: pass/ruin por cruce de barrera.
  - run_portfolio_mc: smoke end-to-end, P(pass)+P(ruin)≈1, sin importar sqx.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import (
    StrategyMC, BlockRecord, block_total_pnls, calibrate_block_size,
    detect_flatflat_blocks, build_block_records,
    count_crossing, evaluate_path, merged_series_identity, precompute_strategy,
    run_portfolio_mc,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _strategy(key, trades, valle, ppl, pv, lots_grid):
    """StrategyMC sintética.

    trades: list[(open_ts, close_ts, close_date)].
    valle: {i: {date: vp}}.
    lots_grid: list[n_f] → array (n_f, n_trades).
    """
    open_t = np.array([_ts(t[0]) for t in trades])
    close_t = np.array([_ts(t[1]) for t in trades])
    close_d = np.array([t[2] for t in trades])
    n = len(trades)
    lots = np.array(lots_grid)  # (n_f, n)
    return StrategyMC(
        key=key, open_t=open_t, close_t=close_t, close_d=close_d,
        valle=valle, pnl_per_lot=np.array(ppl, dtype=float),
        is_buy=np.ones(n, dtype=bool), pv=pv, lots=lots, n_trades=n,
    )


class _TinyTradeList:
    def __init__(self):
        self.df = pd.DataFrame({
            "open_time": [_ts("2024-01-01 10:00")],
            "close_time": [_ts("2024-01-01 10:02")],
            "type": ["Buy"],
        })
        self.n_trades = 1
        self.open_price = np.array([100.0])

    def pnl_per_lot(self):
        return np.array([10.0])


def test_precompute_strategy_accepts_short_ohlc_columns():
    idx = pd.date_range("2024-01-01 10:00", periods=3, freq="min")
    ohlc = pd.DataFrame({
        "o": [100.0, 100.0, 99.5],
        "h": [100.5, 100.2, 99.8],
        "l": [99.8, 99.4, 99.2],
        "c": [100.0, 99.5, 99.3],
    }, index=idx)

    s = precompute_strategy(_TinyTradeList(), ohlc, pv=100.0, key="short-cols")

    assert s.key == "short-cols"
    assert s.valle[0][date(2024, 1, 1)] == pytest.approx(-0.8)


# ---------------------------------------------------------------------------
# detect_flatflat_blocks
# ---------------------------------------------------------------------------

def test_detect_flatflat_blocks_non_overlapping():
    # 3 episodios disjuntos (A dos veces, B una) → 3 bloques busy.
    oc = {
        "A": (np.array([_ts("2024-01-01 10:00"), _ts("2024-01-02 10:00")]),
              np.array([_ts("2024-01-01 12:00"), _ts("2024-01-02 12:00")])),
        "B": (np.array([_ts("2024-01-03 10:00")]),
              np.array([_ts("2024-01-03 12:00")])),
    }
    blocks = detect_flatflat_blocks(oc)
    assert len(blocks) == 3
    # cada bloque es (start_busy, end_flat)
    starts = [b[0] for b in blocks]
    assert starts == sorted(starts)


def test_detect_flatflat_blocks_overlapping_one_block():
    # A y B solapados → 1 solo bloque busy (active nunca baja a 0 entre medias).
    oc = {
        "A": (np.array([_ts("2024-01-01 10:00")]),
              np.array([_ts("2024-01-01 14:00")])),
        "B": (np.array([_ts("2024-01-01 11:00")]),
              np.array([_ts("2024-01-01 13:00")])),
    }
    blocks = detect_flatflat_blocks(oc)
    assert len(blocks) == 1
    assert blocks[0][0] == _ts("2024-01-01 10:00")
    assert blocks[0][1] == _ts("2024-01-01 14:00")


# ---------------------------------------------------------------------------
# V2: no crossing
# ---------------------------------------------------------------------------

def test_count_crossing_zero_when_coherent():
    trades = [(_ts("2024-01-01 10:00"), _ts("2024-01-01 12:00"), date(2024, 1, 1))]
    valle = {0: {date(2024, 1, 1): -10.0}}
    s = _strategy("A", trades, valle, [1.0], 100.0, [[0.1]])
    oc = {"A": (s.open_t, s.close_t)}
    blocks = detect_flatflat_blocks(oc)
    assert count_crossing({"A": s}, blocks) == 0


# ---------------------------------------------------------------------------
# V1: identity == composición continua
# ---------------------------------------------------------------------------

def test_v1_identity_equals_continuous():
    # 2 bloques disjuntos, 1 estrategia, 2 trades; valle y pnl coherentes.
    d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
    trades = [
        (_ts("2024-01-01 10:00"), _ts("2024-01-01 12:00"), d1),
        (_ts("2024-01-02 10:00"), _ts("2024-01-02 12:00"), d2),
    ]
    valle = {0: {d1: -100.0}, 1: {d2: -50.0}}
    s = _strategy("A", trades, valle, ppl=[2.0, 3.0], pv=100.0, lots_grid=[[0.1, 0.2]])
    strategies = {"A": s}
    blocks = build_block_records(strategies, detect_flatflat_blocks({"A": (s.open_t, s.close_t)}))
    lots_by = {0: s.lots}
    pv_by = {0: s.pv}
    f_idx = 0
    merged_id = merged_series_identity(blocks, lots_by, pv_by, f_idx)

    # Composición continua de los mismos trades (oráculo):
    # merged(d) = saldo_cerrado_antes(d) + day_float(d); el PnL del close del propio
    # día NO entra (el flotante worst ocurre antes del close).
    # d1: saldo 0 (trade0 cierra en d1, su PnL entra tras el flotante worst)
    #     day_float(d1) = -100*0.1*100 = -1000 → merged(d1) = -1000
    # d2: saldo = 0.2 (trade0 ya cerrado), day_float(d2) = -50*0.2*100 = -1000 → -999.8
    assert pytest.approx(merged_id.loc[d1], abs=1e-6) == -1000.0
    assert pytest.approx(merged_id.loc[d2], abs=1e-6) == 0.2 + (-1000.0)


def test_block_total_pnls_suma_identity():
    """block_total_pnls = PnL cerrado por bloque; su suma = terminal del identity."""
    d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
    trades = [
        (_ts("2024-01-01 10:00"), _ts("2024-01-01 12:00"), d1),
        (_ts("2024-01-02 10:00"), _ts("2024-01-02 12:00"), d2),
    ]
    valle = {0: {d1: -100.0}, 1: {d2: -50.0}}
    s = _strategy("A", trades, valle, ppl=[2.0, 3.0], pv=100.0, lots_grid=[[0.1, 0.2]])
    strategies = {"A": s}
    blocks = build_block_records(strategies, detect_flatflat_blocks({"A": (s.open_t, s.close_t)}))
    pnls = block_total_pnls(blocks, {0: s.lots}, {0: s.pv}, 0)
    # trade0: 0.1*2.0 = 0.2 ; trade1: 0.2*3.0 = 0.6 ; 1 bloque cada uno
    np.testing.assert_allclose(pnls, [0.2, 0.6], atol=1e-9)
    assert pytest.approx(pnls.sum(), abs=1e-9) == 0.8


# ---------------------------------------------------------------------------
# evaluate_path: pass / ruin por cruce
# ---------------------------------------------------------------------------

def test_evaluate_path_ruin_then_pass():
    # 1 bloque con 1 trade cuyo valle worst = -1000 (≤ -350) → ruin en d1.
    d1 = date(2024, 1, 1)
    b = BlockRecord(
        idx=0, start=_ts("2024-01-01 10:00"), end=_ts("2024-01-01 12:00"),
        duration_days=2.0 / 24.0,
        sym_id=np.array([0]), trade_local_id=np.array([0]),
        close_d=np.array([d1]), valle_precio=[{d1: -10.0}],
        pnl_per_lot=np.array([2.0]), n_unique_trades=1,
    )
    lots_by = {0: np.array([[0.1]])}  # size 0.1
    pv_by = {0: 100.0}
    # day_float(d1) = -10*0.1*100 = -100 ≤ -350? no (-100 > -350). Forzamos valle más profundo:
    b.valle_precio = [{d1: -40.0}]  # -40*0.1*100 = -400 ≤ -350 → ruin
    out = evaluate_path(np.array([0]), [b], lots_by, pv_by, 0,
                        pass_line=350.0, ruin_line=-350.0, baseline=5000.0)
    assert out.outcome == "ruin"


def test_evaluate_path_pass():
    # 1 bloque cuyo saldo cerrado sube a +400 (≥ +350) → pass.
    d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
    b = BlockRecord(
        idx=0, start=_ts("2024-01-01 10:00"), end=_ts("2024-01-02 12:00"),
        duration_days=2.0,
        sym_id=np.array([0, 0]), trade_local_id=np.array([0, 1]),
        close_d=np.array([d1, d2]),
        valle_precio=[{d1: 0.0}, {d2: 0.0}],  # sin flotante adverso
        pnl_per_lot=np.array([2000.0, 0.0]),  # trade0 gana 0.1*2000 = 200
        n_unique_trades=2,
    )
    lots_by = {0: np.array([[0.1, 0.1]])}
    pv_by = {0: 100.0}
    # d1: saldo 0 + day_float 0 = 0; tras close trade0 saldo = +200
    # d2: saldo +200 + day_float 0 = +200 (< 350). No pasa con 1 bloque.
    # Ampliamos pnl_per_lot para que pase en d1.
    b.pnl_per_lot = np.array([4000.0, 0.0])  # 0.1*4000 = 400 ≥ 350 al cierre d1
    # Pero el cruce se evalúa en day_float(d), no al cierre. En d1 day_float=0, merged=0.
    # Para forzar pass, hacemos day_float(d1) = +400.
    b.valle_precio = [{d1: 40.0}, {d2: 0.0}]  # +40*0.1*100 = +400
    out = evaluate_path(np.array([0]), [b], lots_by, pv_by, 0,
                        pass_line=350.0, ruin_line=-350.0, baseline=5000.0)
    assert out.outcome == "pass"


# ---------------------------------------------------------------------------
# Smoke end-to-end
# ---------------------------------------------------------------------------

def test_run_portfolio_mc_smoke():
    # 2 estrategias sintéticas, 2 trades cada una, no solapadas entre sí.
    rng = np.random.default_rng(0)
    d = [date(2024, 1, 1 + k) for k in range(4)]
    strategies = {}
    for ki, (key, offset) in enumerate([("A", 0), ("B", 2)]):
        trades = [
            (_ts(f"2024-01-{1+offset:02d} 10:00"), _ts(f"2024-01-{1+offset:02d} 12:00"), d[offset]),
            (_ts(f"2024-01-{2+offset:02d} 10:00"), _ts(f"2024-01-{2+offset:02d} 12:00"), d[offset + 1]),
        ]
        valle = {0: {d[offset]: -5.0}, 1: {d[offset + 1]: -5.0}}
        lots = np.tile(rng.uniform(0.05, 0.1, size=(1, 2)), (3, 1))  # (n_f=3, n_trades=2)
        strategies[key] = _strategy(key, trades, valle, [1.0, 1.0], 100.0, lots.tolist())
    f_grid = np.array([0.01, 0.02, 0.03])
    res = run_portfolio_mc(strategies, f_grid, n_paths=200, block_size=1.0, seed=1)
    assert len(res.reports) == 3
    assert res.n_crossing == 0
    for r in res.reports:
        total = r.p_pass + r.p_ruin + r.p_neither
        assert 0.999 <= total <= 1.001
        assert 0.0 <= r.p_pass <= 1.0


# ---------------------------------------------------------------------------
# calibrate_block_size — fidelidad ACF del barajado (criterio Alan)
# ---------------------------------------------------------------------------

def test_calibrate_ruido_blanco_sin_clustering():
    """Serie sin estructura: has_real_clustering False, recomienda el bs mínimo."""
    rng = np.random.default_rng(7)
    pnl = rng.normal(0, 1, size=400)
    cal = calibrate_block_size(pnl, n_paths=100, seed=1)
    assert cal.has_real_clustering is False
    assert cal.recommended == 1.0
    assert "ruido" in cal.reason
    assert len(cal.per_bs) == 5


def test_calibrate_detecta_clustering_real():
    """Regímenes de volatilidad -> ACF² con lags cortos significativos -> clustering."""
    rng = np.random.default_rng(11)
    sigma = np.repeat([3.0, 3.0, 3.0, 0.5, 0.5, 0.5, 3.0, 0.5], 50)
    pnl = sigma * rng.normal(0, 1, size=len(sigma))
    cal = calibrate_block_size(pnl, n_paths=100, seed=1)
    assert cal.has_real_clustering is True
    assert cal.sig_lags_sq and min(cal.sig_lags_sq) <= 3
    assert cal.recommended in (1.0, 2.0, 3.0, 5.0, 10.0)
    assert "Clustering real" in cal.reason
