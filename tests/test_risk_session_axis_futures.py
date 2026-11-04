"""Tests del eje de sesiones FTMO Futures (cut 16:10 ET) — AC-2 del plan.

Golden del borde de sesión con cut "16:10": ts ET 16:09 → etiqueta N;
16:10 → N+1 (corte exclusivo por arriba; la última barra M1 del día-regla
es la 16:09); 17:59 → N+1; 18:00 → N+1. Los tests de DST usan timestamps
UTC ABSOLUTOS en las transiciones 2026-03-08 (spring forward, EDT = UTC−4
a partir de las 02:00 ET) y 2026-11-01 (fall back, EST = UTC−5): un offset
fijo de horas desplazaría el corte y fallarían. Cubre también la regresión
AC-20: default cut "18:00" bit-idéntico (mismo resultado que la aritmética
histórica ts_ET + 6h), y la validación del formato del cut.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.risk.session_axis import (
    make_session_day_labeler,
    session_label,
    session_labels,
)

ET = "America/New_York"

# ---------------------------------------------------------------------------
# Golden del borde con cut 16:10 (timestamps UTC absolutos, días DST)
# ---------------------------------------------------------------------------

def test_axis_ftmo_cut_1610_borde_marzo_dst():
    # 2026-03-08: spring forward US a las 02:00 ET; desde entonces EDT (UTC−4).
    assert session_label(pd.Timestamp("2026-03-08T20:09:00Z"),
                         cut_hhmm="16:10") == date(2026, 3, 8)   # 16:09 ET
    assert session_label(pd.Timestamp("2026-03-08T20:10:00Z"),
                         cut_hhmm="16:10") == date(2026, 3, 9)   # 16:10 ET
    assert session_label(pd.Timestamp("2026-03-08T21:59:00Z"),
                         cut_hhmm="16:10") == date(2026, 3, 9)   # 17:59 ET
    assert session_label(pd.Timestamp("2026-03-08T22:00:00Z"),
                         cut_hhmm="16:10") == date(2026, 3, 9)   # 18:00 ET


def test_axis_ftmo_cut_1610_borde_noviembre_dst():
    # 2026-11-01: fall back US a las 02:00 ET; desde entonces EST (UTC−5).
    assert session_label(pd.Timestamp("2026-11-01T21:09:00Z"),
                         cut_hhmm="16:10") == date(2026, 11, 1)  # 16:09 ET
    assert session_label(pd.Timestamp("2026-11-01T21:10:00Z"),
                         cut_hhmm="16:10") == date(2026, 11, 2)  # 16:10 ET
    assert session_label(pd.Timestamp("2026-11-01T22:59:00Z"),
                         cut_hhmm="16:10") == date(2026, 11, 2)  # 17:59 ET
    assert session_label(pd.Timestamp("2026-11-01T23:00:00Z"),
                         cut_hhmm="16:10") == date(2026, 11, 2)  # 18:00 ET


def test_axis_ftmo_cut_1610_naive_source_tz():
    # Mismo instante de pared ET vía naive + source_tz: idéntico al tz-aware.
    assert session_label(pd.Timestamp("2024-01-02 16:09"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 2)
    assert session_label(pd.Timestamp("2024-01-02 16:10"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)
    # Post-cierre pre-apertura: atribuido al día-regla siguiente.
    assert session_label(pd.Timestamp("2024-01-02 17:59"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)
    # La apertura 18:00 ya es del día-regla siguiente.
    assert session_label(pd.Timestamp("2024-01-02 18:00"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)
    # Sesión que cruza medianoche: 18:00 ET del 2 → 16:09 ET del 3, misma etiqueta.
    assert session_label(pd.Timestamp("2024-01-02 18:00"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)
    assert session_label(pd.Timestamp("2024-01-03 03:00"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)
    assert session_label(pd.Timestamp("2024-01-03 16:09"), source_tz=ET,
                         cut_hhmm="16:10") == date(2024, 1, 3)


def test_axis_ftmo_cut_1610_labeler_escalar_y_vector():
    labeler = make_session_day_labeler(ET, cut_hhmm="16:10")
    ts = pd.to_datetime(["2024-01-02 16:09", "2024-01-02 16:10",
                         "2024-01-02 23:00"])
    assert labeler(ts[0]) == date(2024, 1, 2)
    assert labeler(pd.Timestamp("2024-01-02 16:10")) == date(2024, 1, 3)
    np.testing.assert_array_equal(
        labeler(ts), np.array([date(2024, 1, 2), date(2024, 1, 3),
                               date(2024, 1, 3)], dtype=object)
    )


# ---------------------------------------------------------------------------
# Regresión AC-20: default "18:00" bit-idéntico (roll 6h histórico)
# ---------------------------------------------------------------------------

def test_axis_default_1800_bit_identico():
    # Panel fijo que cruza AMBAS transiciones DST de 2026 (marzo y noviembre)
    # más días lisos: el default debe igualar a la aritmética histórica
    # (ts_ET + 6h).date() en cada timestamp, y a cut_hhmm="18:00" explícito.
    idx = pd.date_range("2026-03-06", "2026-03-13", freq="37min", tz="UTC")
    idx = idx.append(pd.date_range("2026-10-28", "2026-11-06", freq="41min",
                                   tz="UTC"))
    expected = (idx.tz_convert(ET) + pd.Timedelta(hours=6)).date
    np.testing.assert_array_equal(session_labels(idx), expected)
    np.testing.assert_array_equal(session_labels(idx, cut_hhmm="18:00"), expected)

    def legacy_label(ts):
        scalar = isinstance(ts, pd.Timestamp)
        i = pd.DatetimeIndex([ts] if scalar else pd.to_datetime(ts))
        if i.tz is None:
            i = i.tz_localize(ET)
        out = (i.tz_convert(ET) + timedelta(hours=6)).date
        return out[0] if scalar else out

    np.testing.assert_array_equal(
        make_session_day_labeler(ET)(idx), legacy_label(idx)
    )
    # Escalar, con y sin source_tz
    assert session_label(pd.Timestamp("2026-11-05T22:30:00Z")) == legacy_label(
        pd.Timestamp("2026-11-05T22:30:00Z"))
    assert session_label(pd.Timestamp("2024-01-02 18:00"),
                         source_tz=ET) == date(2024, 1, 3)


def test_axis_cut_1610_equivale_offset_7h50m():
    # El cut 16:10 es exactamente un offset aditivo de 7h50m (24h − cut).
    idx = pd.date_range("2026-03-06", "2026-03-13", freq="23min", tz="UTC")
    idx = idx.append(pd.date_range("2026-10-28", "2026-11-06", freq="29min",
                                   tz="UTC"))
    expected = (idx.tz_convert(ET) + pd.Timedelta(hours=7, minutes=50)).date
    np.testing.assert_array_equal(session_labels(idx, cut_hhmm="16:10"),
                                  expected)


# ---------------------------------------------------------------------------
# Validación del formato del cut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["16:6", "24:00", "16:10:00", "abcd",
                                 "18:60", "00:00"])
def test_axis_cut_formato_invalido_raise(bad):
    with pytest.raises(ValueError):
        session_label(pd.Timestamp("2024-01-02 12:00"), source_tz=ET,
                      cut_hhmm=bad)
    with pytest.raises(ValueError):
        session_labels(pd.to_datetime(["2024-01-02 12:00"]), source_tz=ET,
                       cut_hhmm=bad)
    with pytest.raises(ValueError):
        make_session_day_labeler(ET, cut_hhmm=bad)


# ---------------------------------------------------------------------------
# Plan ftmo-futures-ui Fase 1 (enmienda E6): con cut "16:10" vs "18:00" solo
# cambian de etiqueta los ts ET en [16:10, 18:00) (pasan al día siguiente);
# el resto de la rejilla queda idéntico. La identidad default == "18:00" ya la
# cubre test_axis_default_1800_bit_identico (no se duplica).
# ---------------------------------------------------------------------------

def test_axis_cut_1610_vs_1800_solo_difiere_ventana_1610_1800():
    idx = pd.date_range("2026-03-06", periods=10 * 24, freq="h", tz="UTC")
    et = idx.tz_convert(ET)
    # Ventana ET [16:10, 18:00): hora 16 con minuto >= 10, o hora 17.
    ventana = ((et.hour == 16) & (et.minute >= 10)) | (et.hour == 17)
    assert ventana.any() and (~ventana).any()

    l1610 = make_session_day_labeler(ET, cut_hhmm="16:10")(idx)
    l1800 = make_session_day_labeler(ET, cut_hhmm="18:00")(idx)
    dif = np.asarray([a != b for a, b in zip(l1610, l1800)])
    np.testing.assert_array_equal(dif, ventana)
    # Además: donde difieren, la etiqueta 16:10 es exactamente el día siguiente.
    nxt = np.asarray([a == b + timedelta(days=1)
                      for a, b in zip(l1610[ventana], l1800[ventana])])
    assert nxt.all()
