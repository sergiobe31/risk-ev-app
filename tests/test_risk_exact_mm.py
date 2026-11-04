"""Tests de risk_app/exact_mm.py — modo exacto por pata (ATR real desde .sqx)."""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from risk_app.exact_mm import exact_sizing_from_sqx
from src.risk.trades import load_trade_list

_LASTSETTINGS_ATR = """<root>
  <Methods>
    <Method type="ATRRiskBasedSizing" use="true"><Params>
      <Param key="Risk" className="x">2</Param>
      <Param key="ATRPeriod" className="x">3</Param>
      <Param key="ATRMult" className="x">2</Param>
      <Param key="Decimals" className="x">2</Param>
      <Param key="LotsIfNoMM" className="x">0.01</Param>
      <Param key="MaxLots" className="x">5</Param>
    </Params></Method>
  </Methods>
  <InitialCapital>5000</InitialCapital>
  <Chart symbol="XAUUSD_axi" timeframe="H4"/>
</root>"""

_LASTSETTINGS_FIXED = _LASTSETTINGS_ATR.replace(
    'type="ATRRiskBasedSizing"', 'type="FixedSize"'
)

_HEADER = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
)


def _sqx_bytes(lastsettings: str) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("lastSettings.xml", lastsettings)
        z.writestr("Results/Main: XAUUSD_axi_LOM_H4/dailyEquity.bin", b"PK")
    return buf.getvalue()


def _ohlc_m1_constant_range() -> pd.DataFrame:
    """3 días de M1 donde cada barra H4 tiene high-low = 10 -> ATR(3) = 10."""
    idx = pd.date_range("2024.01.02 00:00", periods=3 * 24 * 60, freq="1min")
    return pd.DataFrame({"o": 100.0, "h": 105.0, "l": 95.0, "c": 100.0}, index=idx)


def _tl_open(open_time: str) -> object:
    csv = _HEADER + (
        f"1;XAUUSD.pa;Buy;{open_time};2000.00;1.00;2024.01.06 00:00;2001.00;100;5100;"
        "IST;Exit After 1 Bars;-10;60;1;-\n"
    )
    return load_trade_list(csv.encode())


def test_sl_in_money_exacto_atr_constante():
    ohlc = _ohlc_m1_constant_range()
    tl = _tl_open("2024.01.05 10:00")
    sl, mm = exact_sizing_from_sqx(_sqx_bytes(_LASTSETTINGS_ATR), tl, ohlc, 100.0)
    # ATR(3)=10 (rango constante), mult=2, pv=100 -> slInMoney = 2000
    assert sl.shape == (1,)
    np.testing.assert_allclose(sl, [2000.0])
    assert mm.risk == 2.0 and mm.timeframe == "H4" and mm.initial_capital == 5000.0


def test_mm_no_atr_rechazado():
    ohlc = _ohlc_m1_constant_range()
    tl = _tl_open("2024.01.05 10:00")
    with pytest.raises(ValueError, match="ATRRiskBasedSizing"):
        exact_sizing_from_sqx(_sqx_bytes(_LASTSETTINGS_FIXED), tl, ohlc, 100.0)


def test_portfolio_sqx_rechazado():
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("lastSettings.xml", _LASTSETTINGS_ATR)
        z.writestr("Results/Portfolio/dailyEquity.bin", b"PK")
    ohlc = _ohlc_m1_constant_range()
    tl = _tl_open("2024.01.05 10:00")
    with pytest.raises(ValueError, match="PORTFOLIO"):
        exact_sizing_from_sqx(buf.getvalue(), tl, ohlc, 100.0)


def test_sqx_de_otro_simbolo_rechazado():
    """El .sqx es de DEUIDXEUR pero el CSV es de XAUUSD -> aviso (no corresponden)."""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("lastSettings.xml", _LASTSETTINGS_ATR)
        z.writestr("Results/Main: DEUIDXEUR_axi_LOM_H4/dailyEquity.bin", b"PK")
    ohlc = _ohlc_m1_constant_range()
    tl = _tl_open("2024.01.05 10:00")  # CSV de XAUUSD.pa
    with pytest.raises(ValueError, match="no corresponden"):
        exact_sizing_from_sqx(buf.getvalue(), tl, ohlc, 100.0)
