"""Modo EXACTO: slInMoney desde el ATR real (Wilder, shift 1) sobre el OHLC.

Ground truth = codigo MQL5 que SQX exporta para ATRRiskBasedSizing (verificado
empiricamente, ratio ATR_mio/ATR_SQX ~0.998)::

    atrValue  = iATR(symbol, TF, ATRPeriod) buffer[1]   # Wilder (RMA), shift 1
    slInMoney = atrValue * ATRMult * PointValue          # tickSize se cancela
    lots      = roundDown(equity*Risk/slInMoney, sizeStep)   # FLOOR

Aqui calculamos el ATR de Wilder sobre el OHLC y lo mapeamos (gap-aware, shift 1)
al entry de cada trade. Esto da el ``slInMoney`` EXACTO sin back-derivo, asi que
el sweep es fiel al sizing de SQX sin importar el regimen de lotes.

Modulo puro: NO importa ``sqx`` (guardian AST). Lee un CSV OHLC (artefacto).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Raiz del repo (para resolver asset_data/ como fallback de OHLC).
_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSET_DATA = _ROOT / "asset_data"
_INPUTS_OHLC = _ROOT / "risk_app" / "inputs" / "ohlc"

# TF de SQX -> string de resample de pandas.
_TF_TO_FREQ: dict[str, str] = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H2": "2h", "H4": "4h", "D1": "1D", "W1": "1W",
}


def resolve_ohlc_path(symbol: str) -> Path:
    """Busca el CSV M1 del simbolo en risk_app/inputs/ohlc/ y luego en asset_data/.

    Matchea por raiz (XAUUSD_axi / XAUUSD.pa / XAUUSD -> todos XAUUSD).
    """
    root = symbol.split(".")[0].split("_")[0]
    for d in (_INPUTS_OHLC, _ASSET_DATA):
        if not d.exists():
            continue
        for p in d.glob("*.csv"):
            if p.stem.split(".")[0].split("_")[0] == root:
                return p
    raise FileNotFoundError(
        f"OHLC no encontrado para '{symbol}' (raiz '{root}') en "
        f"{_INPUTS_OHLC} ni {_ASSET_DATA}."
    )


def load_ohlc_m1(path: Path | str) -> pd.DataFrame:
    """Carga un CSV M1 formato MT5 (sin header): date,time,O,H,L,C,...

    Devuelve DataFrame con columnas o,h,l,c indexado por datetime NAIVE en la
    hora del CSV — que es la del SERVIDOR del bróker del export, NO UTC
    (verificado empíricamente 2026-08-21 en risk_app/inputs/ohlc/*_axi.csv:
    XAUUSD abre lunes 01:00, 0 barras de domingo → AXI EET/EEST esquema
    NY-close; ver skill mt5-sqx-deploy, offset confirmado en vivo 2026-07-22).
    Para reglas con reset diario en otro reloj (FTMO: 00:00 CE(S)T = 01:00
    servidor) hay que rebucketar el eje de días — ver
    ``LegCache(tz_shift_hours)`` / ``apply_day_axis_shift``.
    """
    df = pd.read_csv(
        path, header=None, usecols=[0, 1, 2, 3, 4, 5],
        names=["d", "t", "o", "h", "l", "c"], dtype=str,
    )
    df.index = pd.to_datetime(df["d"] + " " + df["t"], format="%Y.%m.%d %H:%M")
    return df[["o", "h", "l", "c"]].astype(float)


def resample_ohlc(ohlc: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M1 -> TF (H4, D1, ...). Barras etiquetadas por open (left)."""
    freq = _TF_TO_FREQ.get(timeframe.upper())
    if freq is None:
        raise ValueError(f"TF no soportado: {timeframe!r}. Probar: {list(_TF_TO_FREQ)}")
    agg = ohlc.resample(freq, label="left", closed="left").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
    )
    return agg.dropna()


def wilder_atr(ohlc_tf: pd.DataFrame, period: int) -> pd.Series:
    """ATR de Wilder (RMA) — replica iATR de MT5.

    Seed = SMA de los primeros ``period`` True Ranges; luego
    ``ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period``.
    """
    if period < 1:
        raise ValueError("period debe ser >= 1")
    h, l, c = ohlc_tf["h"], ohlc_tf["l"], ohlc_tf["c"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1,
    ).max(axis=1)
    tr = tr.dropna()
    if len(tr) < period:
        raise ValueError(f"Solo {len(tr)} barras; ATR({period}) necesita >= {period}")
    vals = tr.to_numpy(dtype=float)
    atr = np.full(len(vals), np.nan)
    atr[period - 1] = vals[:period].mean()
    for i in range(period, len(vals)):
        atr[i] = (atr[i - 1] * (period - 1) + vals[i]) / period
    out = pd.Series(atr, index=tr.index, name="atr")
    return out.dropna()


def compute_atr_series(
    ohlc_m1: pd.DataFrame, timeframe: str, period: int, tz_shift_hours: float = 0.0,
) -> pd.Series:
    """Lo pesado: resample M1 -> TF + ATR de Wilder. Cacheable por (path, tf, period).

    Devuelve la serie ATR indexada por el open de cada barra del TF.
    """
    if tz_shift_hours:
        ohlc_m1 = ohlc_m1.copy()
        ohlc_m1.index = ohlc_m1.index + pd.Timedelta(hours=tz_shift_hours)
    h4 = resample_ohlc(ohlc_m1, timeframe)
    return wilder_atr(h4, period)


def sl_in_money_from_atr(
    atr_series: pd.Series,
    entries: pd.Series,
    *,
    atr_mult: float,
    point_value: float,
    atr_shift: int = 1,
) -> np.ndarray:
    """Mapea la serie ATR (cacheada) a los entries -> ``slInMoney`` por trade.

    Barra T = ultimo bar con open <= entry; ``atr_shift`` barras atras (1 = SQX).
    """
    atr_idx = atr_series.index
    atr_vals = atr_series.to_numpy()
    ent = pd.to_datetime(entries).to_numpy()
    pos = np.searchsorted(atr_idx.values, ent, side="right") - 1  # bar T
    sel = np.clip(pos - atr_shift, 0, len(atr_vals) - 1)
    return atr_vals[sel] * atr_mult * point_value


def sl_in_money_at_entries(
    ohlc_m1: pd.DataFrame,
    entries: pd.Series,
    *,
    timeframe: str,
    atr_period: int,
    atr_mult: float,
    point_value: float,
    atr_shift: int = 1,
    tz_shift_hours: float = 0.0,
) -> np.ndarray:
    """``slInMoney`` por trade = ATR(shift) @ entry * atr_mult * point_value.

    Conveniente: compone :func:`compute_atr_series` + :func:`sl_in_money_from_atr`.
    """
    atr = compute_atr_series(ohlc_m1, timeframe, atr_period, tz_shift_hours)
    return sl_in_money_from_atr(
        atr, entries, atr_mult=atr_mult, point_value=point_value, atr_shift=atr_shift,
    )

