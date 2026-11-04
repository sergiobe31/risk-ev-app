"""Eje de sesiones ET (CME) para perfiles intradía tipo Apex (plan Fase 0).

Las reglas de Apex (DLL, día cualificado, tier, inactividad) son POR SESIÓN
CME: 18:00 ET del día N−1 → 16:59 ET del día N, cruzando medianoche; el
motor (``mc_portfolio`` / ``propfirm``) agrupa por día calendario. Regla de
etiquetado adjudicada:

    session_label(ts) = (ts convertido a America/New_York + roll).date()

con ``roll = 24h − cut`` (cut = hora ET de corte de la sesión; default
"18:00" → roll 6h, bit-idéntico al histórico) — la fecha ET del cierre de la
sesión que contiene ``ts`` (con cut 18:00: 18:00 ET del día N−1 y 16:59 ET
del día N llevan ambos etiqueta N; domingo 18:00 ET → lunes = trade date
CME). FTMO Futures usa cut "16:10" → roll 7h50m: la partición de etiquetado
es [16:10 ET día N, 16:10 ET día N+1) → etiqueta N+1, así los minutos
post-cierre/pre-apertura (16:10–17:59) se atribuyen al día-regla siguiente,
cuyo snapshot EOD de balance ya se tomó a las 16:10. La conversión es
tz-aware con ``zoneinfo`` (DST correcto; un offset fijo de horas desplazaría
el corte una hora durante medio año). El offset aditivo es DST-seguro por
estructura: las transiciones US son a las 02:00 ET y la FECHA del resultado
no cambia aunque la hora wall-clock cayera en el hueco/repetición.

Integración: el etiquetado se inyecta al motor como ``day_labeler``
(callable ts→date, ver :func:`make_session_day_labeler`) en
``mc_portfolio.precompute_strategy`` (groupby de envolventes y ``close_d``)
y en ``propfirm.build_day_blocks`` / ``build_day_blocks_portfolio``
(``open_ord``). Con ``None`` el motor sigue en día calendario bit-idéntico.
La ventana joint-float de FTMO (``propfirm._joint_float_columns``) permanece
en día calendario a propósito: los perfiles Apex no pasan por el evaluador
FTMO (Fases 2-3 del plan).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = "America/New_York"

_DEFAULT_CUT = "18:00"
_CUT_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_MAINT_START_H = 17   # ventana de mantenimiento CME: [17:00, 18:00) ET
_MAINT_END_H = 18

_SCALAR_TYPES = (pd.Timestamp, datetime, np.datetime64)


def _roll_from_cut(cut_hhmm: str) -> timedelta:
    """Offset aditivo de etiquetado: ``roll = 24h − cut``.

    Cut "18:00" → 6h (bit-idéntico al histórico); cut "16:10" → 7h50m
    (FTMO Futures). El corte es EXCLUSIVO por arriba: un ts exactamente en
    el cut ya etiqueta al día siguiente (barras M1 con timestamp de
    apertura: la última barra del día-regla FTMO es la 16:09).
    """
    m = _CUT_RE.match(str(cut_hhmm))
    if m is None:
        raise ValueError(
            f"cut_hhmm '{cut_hhmm}': formato invalido "
            f"(se espera 'HH:MM', 00:00-23:59)"
        )
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 0 and mi == 0:
        raise ValueError("cut_hhmm '00:00' prohibido: degeneraria el roll a 24h")
    return timedelta(hours=24) - timedelta(hours=h, minutes=mi)


# Roll de los helpers de compliance (semántica Apex: cruce de cierre de
# sesión CME). NO aceptan cut: los perfiles FTMO Futures no los consumen.
_SESSION_ROLL = _roll_from_cut(_DEFAULT_CUT)


def _as_et_index(ts, source_tz: str | None, close_tz: str) -> pd.DatetimeIndex:
    """DatetimeIndex tz-aware en ``close_tz``; ``source_tz`` obligatorio si naive."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    if idx.tz is None:
        if source_tz is None:
            raise ValueError(
                "timestamps naive sin source_tz: la zona horaria de origen es "
                "obligatoria para etiquetar sesiones ET"
            )
        idx = idx.tz_localize(ZoneInfo(source_tz))
    return idx.tz_convert(ZoneInfo(close_tz))


def session_label(ts, source_tz: str | None = None, close_tz: str = ET,
                  cut_hhmm: str = _DEFAULT_CUT) -> date:
    """Etiqueta de sesión (date) de UN timestamp: fecha ET de su cierre."""
    idx = _as_et_index([ts], source_tz, close_tz)
    return (idx[0] + _roll_from_cut(cut_hhmm)).date()


def session_labels(ts, source_tz: str | None = None, close_tz: str = ET,
                   cut_hhmm: str = _DEFAULT_CUT) -> np.ndarray:
    """Etiquetas de sesión (array de date) para un array/Series de timestamps."""
    idx = _as_et_index(ts, source_tz, close_tz)
    return (idx + _roll_from_cut(cut_hhmm)).date


def make_session_day_labeler(source_tz: str, close_tz: str = ET,
                             cut_hhmm: str = _DEFAULT_CUT):
    """Callable ts→date (escalar o vector) para el ``day_labeler`` del motor.

    ``source_tz`` es la zona de los timestamps de entrada cuando llegan naive
    (p.ej. la tz documentada de los CSV de futuros, Fase 7 del plan).
    ``cut_hhmm`` es la hora ET de corte de la sesión (default "18:00" =
    Apex, bit-idéntico; FTMO Futures usa "16:10", ver :func:`_roll_from_cut`).
    """
    roll = _roll_from_cut(cut_hhmm)

    def label(ts):
        if isinstance(ts, _SCALAR_TYPES):
            idx = _as_et_index([ts], source_tz, close_tz)
            return (idx[0] + roll).date()
        idx = _as_et_index(ts, source_tz, close_tz)
        return (idx + roll).date
    return label


def _maintenance_mask(idx_et: pd.DatetimeIndex) -> np.ndarray:
    h = idx_et.hour
    return (h >= _MAINT_START_H) & (h < _MAINT_END_H)


def in_maintenance_window(ts, source_tz: str | None = None, close_tz: str = ET):
    """True (escalar o máscara) para timestamps en [17:00, 18:00) ET."""
    scalar = isinstance(ts, _SCALAR_TYPES)
    idx = _as_et_index([ts] if scalar else ts, source_tz, close_tz)
    m = _maintenance_mask(idx)
    return bool(m[0]) if scalar else m


def crosses_session_close(open_ts, close_ts, source_tz: str | None = None,
                          close_tz: str = ET):
    """True si el intervalo [open, close] cruza un cierre de sesión (17:00 ET).

    Un trade compliant Apex (flat a las 16:59 ET) tiene open y close en la
    MISMA sesión y ningún extremo en la ventana de mantenimiento. Acepta
    escalares o arrays pareados.
    """
    scalar = isinstance(open_ts, _SCALAR_TYPES)
    o = _as_et_index([open_ts] if scalar else open_ts, source_tz, close_tz)
    c = _as_et_index([close_ts] if scalar else close_ts, source_tz, close_tz)
    cross = ((o + _SESSION_ROLL).date != (c + _SESSION_ROLL).date) | \
        _maintenance_mask(o) | _maintenance_mask(c)
    return bool(cross[0]) if scalar else np.asarray(cross, dtype=bool)


def overnight_violations(intervals, source_tz: str | None = None,
                         close_tz: str = ET) -> np.ndarray:
    """Índices de los intervalos que cruzan un cierre de sesión (17:00 ET).

    ``intervals``: iterable de pares ``(open, close)`` o de objetos con
    ``.start``/``.end`` (p.ej. bloques). Extremos NaT → intervalo ignorado
    (el motor ya filtra esos trades). Vacío ⇒ compliant (assert "sin
    overnight" de los perfiles Apex: ``len(...) == 0``).
    """
    opens, closes, keep = [], [], []
    for k, it in enumerate(intervals):
        if hasattr(it, "start") and hasattr(it, "end"):
            o, c = it.start, it.end
        else:
            o, c = it
        if pd.isna(o) or pd.isna(c):
            continue
        opens.append(o)
        closes.append(c)
        keep.append(k)
    if not keep:
        return np.empty(0, dtype=int)
    cross = crosses_session_close(opens, closes, source_tz=source_tz,
                                  close_tz=close_tz)
    return np.asarray(keep, dtype=int)[cross]
