"""Golden diferencial del valle flotante diario (src/risk/mc_portfolio.py).

El valle por dia que consume el motor de prop firms (via ``StrategyMC.valle``)
se calcula en ``precompute_strategy``. Historicamente se valido contra un
oraculo externo (scripts/diagnostics/mini_sim_daily_equity.py); este test lo
fija con datos sinteticos minimos donde el valor esperado se calcula A MANO
en el comentario (long y short), incluido el papel de ``prev_close``:

    precio_worst = min(low, prev_close)   (long)
    precio_worst = max(high, prev_close)  (short)
    valle_d = min_dia( signo * (precio_worst - open_price) )

``prev_close`` de la primera barra del tramo es el open_price del trade; el
resto viene del close de la barra M1 anterior (serie continua entre dias).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import precompute_strategy


class _TradeList:
    """TradeList sintetico minimo para precompute_strategy."""

    def __init__(self, open_ts, close_ts, tipo, open_price):
        self.df = pd.DataFrame({
            "open_time": [pd.Timestamp(open_ts)],
            "close_time": [pd.Timestamp(close_ts)],
            "type": [tipo],
        })
        self.n_trades = 1
        self.open_price = np.array([open_price], dtype=float)

    def pnl_per_lot(self):
        return np.array([0.0])


def _ohlc(rows, index, short_cols=False):
    """rows: list[(h, l, c)]; index: timestamps M1."""
    if short_cols:
        cols = ("h", "l", "c")
    else:
        cols = ("High", "Low", "Close")
    return pd.DataFrame(
        {c: [r[i] for r in rows] for c, i in zip(cols, range(3))},
        index=pd.DatetimeIndex(index),
    )


def test_valle_long_hand_computed():
    # Trade LONG: abre 10:00 @ 100.0, cierra 10:04. Barras M1 (h, l, c):
    #   10:00 (100.5, 99.9, 100.2)
    #   10:01 (100.4, 99.5,  99.6)
    #   10:02 (100.0, 99.0,  98.8)
    #   10:03 ( 99.8, 99.1,  99.5)
    #   10:04 ( 99.9, 99.4,  99.7)
    #
    # Calculo manual (long: precio_worst = min(low, prev_close)):
    #   prev_close = [100.0(open), 100.2, 99.6, 98.8, 99.5]
    #   10:00: min(99.9, 100.0) = 99.9 -> vp = -0.1
    #   10:01: min(99.5, 100.2) = 99.5 -> vp = -0.5
    #   10:02: min(99.0,  99.6) = 99.0 -> vp = -1.0
    #   10:03: min(99.1,  98.8) = 98.8 -> vp = -1.2   <- ATA prev_close:
    #          el low de la barra (99.1) NO llega al peor flotante, que ya
    #          quedo fijado por el close adverso de la barra anterior (98.8).
    #   10:04: min(99.4,  99.5) = 99.4 -> vp = -0.6
    #   valle(dia) = min = -1.2
    idx = pd.date_range("2024-01-01 10:00", periods=5, freq="min")
    ohlc = _ohlc(
        [(100.5, 99.9, 100.2), (100.4, 99.5, 99.6), (100.0, 99.0, 98.8),
         (99.8, 99.1, 99.5), (99.9, 99.4, 99.7)],
        idx,
    )
    tl = _TradeList("2024-01-01 10:00", "2024-01-01 10:04", "Buy", 100.0)

    s = precompute_strategy(tl, ohlc, pv=100.0, key="long-golden")

    assert s.is_buy[0]
    assert s.valle[0] == {date(2024, 1, 1): pytest.approx(-1.2)}


def test_valle_short_hand_computed():
    # Trade SHORT: abre 10:00 @ 200.0, cierra 10:02. Barras M1 (h, l, c):
    #   10:00 (200.3, 199.8, 200.1)
    #   10:01 (200.2, 199.5, 200.8)
    #   10:02 (200.5, 199.9, 200.2)
    #
    # Calculo manual (short: precio_worst = max(high, prev_close), vp = -(pw-op)):
    #   prev_close = [200.0(open), 200.1, 200.8]
    #   10:00: max(200.3, 200.0) = 200.3 -> vp = -0.3
    #   10:01: max(200.2, 200.1) = 200.2 -> vp = -0.2
    #   10:02: max(200.5, 200.8) = 200.8 -> vp = -0.8   <- ATA prev_close:
    #          el high (200.5) no supera el close adverso previo (200.8).
    #   valle(dia) = min = -0.8
    idx = pd.date_range("2024-01-01 10:00", periods=3, freq="min")
    ohlc = _ohlc(
        [(200.3, 199.8, 200.1), (200.2, 199.5, 200.8), (200.5, 199.9, 200.2)],
        idx, short_cols=True,  # alias cortos h/l/c tambien tolerados
    )
    tl = _TradeList("2024-01-01 10:00", "2024-01-01 10:02", "Sell", 200.0)

    s = precompute_strategy(tl, ohlc, pv=100.0, key="short-golden")

    assert not s.is_buy[0]
    assert s.valle[0] == {date(2024, 1, 1): pytest.approx(-0.8)}


def test_valle_se_agrega_por_dia_calendario():
    # Trade LONG que cruza medianoche: abre D1 23:59 @ 100.0, cierra D2 00:02.
    # Barras M1 (h, l, c):
    #   D1 23:59 (100.2, 99.7,  99.8)
    #   D2 00:00 (100.1, 99.9, 100.0)
    #   D2 00:01 (100.3, 99.4,  99.6)
    #   D2 00:02 ( 99.8, 99.5,  99.7)
    #
    # Calculo manual (prev_close continuo entre dias: la 1a barra de D2 usa
    # el close de D1 23:59 = 99.8):
    #   prev_close = [100.0(open), 99.8, 100.0, 99.6]
    #   D1 23:59: min(99.7, 100.0) = 99.7 -> vp = -0.3
    #   D2 00:00: min(99.9,  99.8) = 99.8 -> vp = -0.2
    #   D2 00:01: min(99.4, 100.0) = 99.4 -> vp = -0.6
    #   D2 00:02: min(99.5,  99.6) = 99.5 -> vp = -0.5
    #   valle(D1) = -0.3 ; valle(D2) = min(-0.2, -0.6, -0.5) = -0.6
    idx = pd.DatetimeIndex([
        "2024-01-01 23:59", "2024-01-02 00:00",
        "2024-01-02 00:01", "2024-01-02 00:02",
    ])
    ohlc = _ohlc(
        [(100.2, 99.7, 99.8), (100.1, 99.9, 100.0),
         (100.3, 99.4, 99.6), (99.8, 99.5, 99.7)],
        idx,
    )
    tl = _TradeList("2024-01-01 23:59", "2024-01-02 00:02", "Buy", 100.0)

    s = precompute_strategy(tl, ohlc, pv=100.0, key="long-2d")

    assert set(s.valle[0]) == {date(2024, 1, 1), date(2024, 1, 2)}
    assert s.valle[0][date(2024, 1, 1)] == pytest.approx(-0.3)
    assert s.valle[0][date(2024, 1, 2)] == pytest.approx(-0.6)
