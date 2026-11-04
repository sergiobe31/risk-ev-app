"""Tests del eje de sesiones ET (``src.risk.session_axis``, plan Apex Fase 0).

Regla adjudicada: etiqueta de sesión = ``(ts → America/New_York + 6h).date()``
— la fecha ET del cierre 16:59 de la sesión que contiene el timestamp (18:00 ET
del día N−1 → 16:59 ET del día N, cruzando medianoche; domingo 18:00 ET →
lunes = trade date CME). Conversión tz-aware con zoneinfo (DST correcto): los
tests de DST usan timestamps ABSOLUTOS (UTC) que discriminan contra un offset
fijo de horas. Cubre también: naive sin source_tz → raise; coherencia
open_ord/close_ord/claves de valle con el labeler activo; paridad bit-idéntica
con labeler None; y los helpers de compliance (sin overnight, mantenimiento
[17:00, 18:00) ET).
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import precompute_strategy
from src.risk.propfirm import (
    block_day_arrays,
    build_day_blocks,
    build_day_blocks_portfolio,
)
from src.risk.session_axis import (
    crosses_session_close,
    in_maintenance_window,
    make_session_day_labeler,
    overnight_violations,
    session_label,
    session_labels,
)

ET = "America/New_York"


def _et(ts: str) -> pd.Timestamp:
    """Timestamp naive interpretado como hora ET (vía source_tz)."""
    return pd.Timestamp(ts)


class _TLFake:
    """TradeList mínima para precompute_strategy (df + open_price + pnl)."""

    def __init__(self, df, open_price):
        self.df = df
        self.open_price = np.asarray(open_price, dtype=float)
        self.n_trades = len(df)

    def pnl_per_lot(self):
        return np.ones(self.n_trades)


def _ohlc(index):
    n = len(index)
    return pd.DataFrame(
        {"o": np.full(n, 100.0), "h": np.full(n, 100.5),
         "l": np.full(n, 99.0), "c": np.full(n, 100.0)},
        index=pd.DatetimeIndex(index),
    )


def _strat_overnight_session(day_labeler=None):
    """Un trade 19:00 ET → 02:00 ET del día siguiente (sesión que cruza
    medianoche), con OHLC M1 cubriendo todo el tramo (timestamps naive ET)."""
    df = pd.DataFrame({
        "open_time": ["2024-01-02 19:00"],
        "close_time": ["2024-01-03 02:00"],
        "type": ["buy"],
    })
    tl = _TLFake(df, [100.0])
    idx = pd.date_range("2024-01-02 18:00", "2024-01-03 03:00", freq="15min")
    return precompute_strategy(tl, _ohlc(idx), pv=10.0, key="S",
                               day_labeler=day_labeler)


# ---------------------------------------------------------------------------
# Etiquetado escalar y vectorizado
# ---------------------------------------------------------------------------

def test_sesion_que_cruza_medianoche_es_un_dia_sesion():
    # Sesión N: 18:00 ET del 2 ene → 16:59 ET del 3 ene; etiqueta 3 ene.
    assert session_label(_et("2024-01-02 18:00"), source_tz=ET) == date(2024, 1, 3)
    assert session_label(_et("2024-01-02 22:30"), source_tz=ET) == date(2024, 1, 3)
    assert session_label(_et("2024-01-03 03:00"), source_tz=ET) == date(2024, 1, 3)
    assert session_label(_et("2024-01-03 16:59"), source_tz=ET) == date(2024, 1, 3)
    labels = session_labels(
        pd.to_datetime(["2024-01-02 18:00", "2024-01-03 03:00"]), source_tz=ET
    )
    assert list(labels) == [date(2024, 1, 3), date(2024, 1, 3)]


def test_dst_marzo_timestamps_absolutos():
    # Spring forward US: domingo 2024-03-10 02:00 ET. El martes 12 rige EDT (-4):
    # 22:30 UTC = 18:30 EDT → etiqueta del día siguiente. Un offset fijo -5
    # daría 17:30 → etiqueta del mismo día (fallaría).
    assert session_label(pd.Timestamp("2024-03-12T22:30:00Z")) == date(2024, 3, 13)
    # Antes de la transición rige EST (-5): 23:30 UTC = 18:30 EST.
    assert session_label(pd.Timestamp("2024-03-08T23:30:00Z")) == date(2024, 3, 9)
    # El mismo instante de pared ET debe dar la misma etiqueta naive o tz-aware.
    assert session_label(_et("2024-03-12 18:30"), source_tz=ET) == date(2024, 3, 13)


def test_dst_noviembre_timestamps_absolutos():
    # Fall back US: domingo 2024-11-03 02:00 ET. El martes 5 rige EST (-5):
    # 22:30 UTC = 17:30 EST → misma sesión. Un offset fijo -4 daría 18:30 →
    # etiqueta del día siguiente (fallaría).
    assert session_label(pd.Timestamp("2024-11-05T22:30:00Z")) == date(2024, 11, 5)
    assert session_label(pd.Timestamp("2024-11-05T23:30:00Z")) == date(2024, 11, 6)
    # Antes de la transición rige EDT (-4): 22:30 UTC = 18:30 EDT.
    assert session_label(pd.Timestamp("2024-10-31T22:30:00Z")) == date(2024, 11, 1)


def test_domingo_18h_et_es_lunes():
    domingo = _et("2024-01-07 18:00")  # 2024-01-07 fue domingo
    assert domingo.dayofweek == 6
    lab = session_label(domingo, source_tz=ET)
    assert lab == date(2024, 1, 8)
    assert lab.weekday() == 0  # lunes = trade date CME


def test_festivo_cierre_anticipado_mapea_bien():
    # 2024-07-03 (víspera del 4 jul): cierre anticipado 13:00 ET; el trade
    # 09:30 → 13:00 ET vive entero en la sesión del 3 jul.
    assert session_label(_et("2024-07-03 09:30"), source_tz=ET) == date(2024, 7, 3)
    assert session_label(_et("2024-07-03 13:00"), source_tz=ET) == date(2024, 7, 3)
    assert not crosses_session_close(
        _et("2024-07-03 09:30"), _et("2024-07-03 13:00"), source_tz=ET
    )


def test_naive_sin_source_tz_raise():
    with pytest.raises(ValueError):
        session_label(_et("2024-01-02 19:00"))
    with pytest.raises(ValueError):
        session_labels(pd.to_datetime(["2024-01-02 19:00"]))
    with pytest.raises(ValueError):
        crosses_session_close(_et("2024-01-02 19:00"), _et("2024-01-02 20:00"))
    with pytest.raises(ValueError):
        in_maintenance_window(_et("2024-01-02 17:30"))


def test_trade_19h_a_02h_ambos_extremos_misma_etiqueta():
    o, c = _et("2024-01-02 19:00"), _et("2024-01-03 02:00")
    assert session_label(o, source_tz=ET) == date(2024, 1, 3)
    assert session_label(c, source_tz=ET) == date(2024, 1, 3)
    assert not crosses_session_close(o, c, source_tz=ET)


# ---------------------------------------------------------------------------
# Integración con el motor (un solo punto de remap)
# ---------------------------------------------------------------------------

def test_open_close_valle_mismo_ordinal_con_labeler_activo():
    labeler = make_session_day_labeler(source_tz=ET)
    s = _strat_overnight_session(day_labeler=labeler)
    ord3 = pd.Timestamp(date(2024, 1, 3)).toordinal()
    assert set(s.valle) == {0}
    assert {pd.Timestamp(d).toordinal() for d in s.valle[0]} == {ord3}
    assert pd.Timestamp(s.close_d[0]).toordinal() == ord3

    (blk,) = build_day_blocks(s, day_labeler=labeler)
    assert blk.open_ord[0] == ord3
    assert blk.close_ord[0] == ord3

    (pblk,) = build_day_blocks_portfolio({0: s}, day_labeler=labeler)
    assert pblk.open_ord[0] == ord3
    assert pblk.close_ord[0] == ord3

    bda = block_day_arrays(blk, np.ones(s.n_trades), pv=s.pv)
    assert list(bda.day_ord) == [ord3]  # un solo día-sesión pese a 2 días civiles


def test_paridad_labeler_none_vs_labeler_calendario():
    # Un labeler que reproduce el día calendario debe dar EXACTAMENTE lo de hoy.
    def cal_labeler(ts):
        return pd.DatetimeIndex(pd.to_datetime(ts)).date

    s_none = _strat_overnight_session(day_labeler=None)
    s_cal = _strat_overnight_session(day_labeler=cal_labeler)
    assert s_none.valle == s_cal.valle
    assert list(s_none.close_d) == list(s_cal.close_d)

    (b_none,) = build_day_blocks(s_none)
    (b_cal,) = build_day_blocks(s_cal, day_labeler=cal_labeler)
    np.testing.assert_array_equal(b_none.open_ord, b_cal.open_ord)
    np.testing.assert_array_equal(b_none.close_ord, b_cal.close_ord)

    (pb_none,) = build_day_blocks_portfolio({0: s_none})
    (pb_cal,) = build_day_blocks_portfolio({0: s_cal}, day_labeler=cal_labeler)
    np.testing.assert_array_equal(pb_none.open_ord, pb_cal.open_ord)
    np.testing.assert_array_equal(pb_none.close_ord, pb_cal.close_ord)


def test_labeler_cambia_la_agregacion_respecto_a_calendario():
    # Sanidad inversa: sin labeler el trade toca 2 días civiles; con labeler, 1.
    s_none = _strat_overnight_session(day_labeler=None)
    s_et = _strat_overnight_session(day_labeler=make_session_day_labeler(ET))
    assert len(s_none.valle[0]) == 2
    assert len(s_et.valle[0]) == 1


# ---------------------------------------------------------------------------
# Helpers de compliance (perfiles Apex: sin overnight, sin mantenimiento)
# ---------------------------------------------------------------------------

def test_compliance_bloque_overnight_detectado():
    intervals = [
        (_et("2024-01-02 19:00"), _et("2024-01-03 02:00")),   # limpio (misma sesión)
        (_et("2024-01-02 16:00"), _et("2024-01-03 09:00")),   # overnight 17:00 ET
        (_et("2024-01-02 17:10"), _et("2024-01-02 17:45")),   # en mantenimiento
        SimpleNamespace(start=_et("2024-01-02 10:00"), end=_et("2024-01-02 15:30")),
    ]
    viol = overnight_violations(intervals, source_tz=ET)
    assert list(viol) == [1, 2]


def test_compliance_bloque_intradia_limpio():
    intervals = [
        (_et("2024-01-02 09:30"), _et("2024-01-02 16:00")),
        (_et("2024-01-02 19:00"), _et("2024-01-03 02:00")),
        (_et("2024-01-02 18:00"), _et("2024-01-02 23:59")),
    ]
    assert overnight_violations(intervals, source_tz=ET).size == 0
    assert overnight_violations([], source_tz=ET).size == 0
    # Extremos NaT: ignorados (el motor ya filtra esos trades).
    nan_int = [(pd.NaT, _et("2024-01-02 10:00"))]
    assert overnight_violations(nan_int, source_tz=ET).size == 0


def test_ventana_mantenimiento_1700_1800_et():
    assert in_maintenance_window(_et("2024-01-02 17:00"), source_tz=ET)
    assert in_maintenance_window(_et("2024-01-02 17:59"), source_tz=ET)
    assert not in_maintenance_window(_et("2024-01-02 16:59"), source_tz=ET)
    assert not in_maintenance_window(_et("2024-01-02 18:00"), source_tz=ET)
    m = in_maintenance_window(
        pd.to_datetime(["2024-01-02 16:59", "2024-01-02 17:30"]), source_tz=ET
    )
    assert list(m) == [False, True]
