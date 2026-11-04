"""Eje de días para el lente prop firm (F2c-2).

Puerto fiel de ``risk_app/propfirm_mode.apply_day_axis_shift`` (:50-74) — NO se
importa desde ahí porque ``propfirm_mode`` importa Streamlit a nivel módulo y
este shim FastAPI no puede arrastrarlo. El motor ``src/risk`` agrupa por fecha
naive del timestamp (medianoche de servidor AXI, EET/EEST); FTMO resetea el
daily loss a 00:00 CE(S)T = 01:00 servidor, así que ``hours=-1`` rebucketa al
eje FTMO.

Sensibilidad medida 2026-08-21 (adjudicación joint float, attack 5;
``sqx/exp_ftmo_tz_sensitivity.py``): a eje servidor (hours=0) P(pass) queda
~1.3pp optimista de media (máx ~5pp) frente al eje FTMO.

Se aplica en ``main._load_leg`` ANTES del precomputo (bloques, valles M1 y
sizing viven todos en el mismo eje); la clave de ``_leg_cache`` incluye las
horas de shift → entradas separadas por eje.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

DAY_AXIS_HOURS = {"servidor": 0.0, "ftmo": -1.0}


def apply_day_axis_shift(tl, ohlc: pd.DataFrame | None,
                         hours: float):
    """Shift global de timestamps al eje elegido; devuelve (tl, ohlc).

    Copia ``tl.df`` y el índice OHLC (mismo comportamiento que Streamlit —
    el OHLC llega cacheado y NO debe mutarse in situ).
    """
    if not hours:
        return tl, ohlc
    delta = pd.Timedelta(hours=hours)
    df_s = tl.df.copy()
    for col in ("open_time", "close_time"):
        if col in df_s.columns:
            df_s[col] = pd.to_datetime(df_s[col], errors="coerce") + delta
    tl = dataclasses.replace(
        tl, df=df_s,
        close_time=pd.to_datetime(tl.close_time, errors="coerce") + delta)
    if ohlc is not None:
        ohlc = ohlc.copy()
        ohlc.index = ohlc.index + delta
    return tl, ohlc
