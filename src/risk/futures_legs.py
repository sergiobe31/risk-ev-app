"""Patas de futuros nativas desde una TABLA DE TRADES (plan float-valleys-ventanas, Fase B).

Frontera con ``sqx``: este módulo NO importa de ``sqx`` (guardián
``tests/test_risk_no_sqx_import.py``). El extractor ``sqx/sqx_trades.py``
(Fase A) produce un ``DataFrame`` por estrategia y este módulo lo consume.

Contrato de la tabla (una fila por trade, orden = id local del trade en la pata):

- ``open_ts``, ``close_ts``: datetime64[ns] naive, reloj de pared ET.
- ``lots`` (> 0), ``pnl_net`` (USD, NETO de comisión RT).
- ``mae_usd`` (USD, >= 0): peor flotante BRUTO de TODA la vida del trade.
- ``open_px``, ``close_px``, ``sl_px`` (SL inicial), ``is_buy`` (bool).
- ``comm_usd``: comisión RT total del trade (USD).
- ``swap_usd`` (opcional, patas CFD; 2026-10-12): swap acumulado del trade, ya
  incluido en ``pnl_net`` (positivo = abono). Si es un cargo, entra entero en el
  peor flotante del trade (cota conservadora: se supone cobrado desde la apertura).

Metadatos (dict): ``symbol``, ``pv``, ``tick_size``, ``backtest_precision``,
``strategy_name``.

Dos piezas:

- :func:`strategy_from_trade_table` → ``(StrategyMC, FuturesLegData)``. La
  ``StrategyMC`` lleva ``valle = {}`` (el flotante NO viene de valles OHLC sino
  de la cota de ventanas: ``propfirm.block_day_arrays_portfolio(...,
  float_mode="windows", adverse_by_sym=...)``) y ``lots`` vacío (los lotes los
  fija el sizing).
- :func:`fixed_risk_micros`: sizing "Opción A" (debate MM FTMO 2026-10-03):
  riesgo fijo en USD por trade, ``n = min(cap, max(min_lot, floor(X / riesgo)))``
  con ``X = f · account_size`` estático (sin compounding). Micros enteros por
  defecto; CFD con ``lot_step``/``min_lot`` = 0,01 (2026-10-12).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.risk.mc_portfolio import StrategyMC

REQUIRED_COLUMNS = (
    "open_ts", "close_ts", "lots", "pnl_net", "mae_usd",
    "open_px", "close_px", "sl_px", "is_buy", "comm_usd",
)

# Tolerancia numérica del cociente X/riesgo antes del floor: f·account_size
# puede quedar 1 ulp por debajo de un entero exacto (p.ej. 174.99999999999997)
# y bajar un micro por ruido de coma flotante. 1e-9 en el cociente es órdenes de
# magnitud menor que cualquier diferencia real de riesgo (precios float32 · pv).
_FLOOR_EPS = 1e-9


@dataclass
class FuturesLegData:
    """Datos por trade de una pata de futuros, alineados con la ``StrategyMC``.

    - ``adverse_per_lot``: ``(mae_usd + comm_usd + cargo de swap) / lots`` (>= 0):
      peor flotante adverso NETO de la comisión RT (y del swap si es cargo),
      USD por lote. Es lo que consume ``float_mode="windows"`` (``adverse_by_sym``).
    - ``risk_per_micro``: ``|open_px - sl_px| · pv``: riesgo inicial del SL por
      micro (USD); la comisión NO está incluida (ver ``p95_total_adverse_over_X``).
    - ``comm_per_lot``: ``comm_usd / lots``; ``pnl_per_lot``: ``pnl_net / lots``.
    """

    adverse_per_lot: np.ndarray
    risk_per_micro: np.ndarray
    comm_per_lot: np.ndarray
    pnl_per_lot: np.ndarray
    symbol: str
    pv: float
    backtest_precision: str | None = None


def strategy_from_trade_table(df: pd.DataFrame, meta: dict, key: str,
                              day_labeler=None) -> tuple[StrategyMC, FuturesLegData]:
    """Construye ``(StrategyMC, FuturesLegData)`` desde la tabla de trades.

    ``close_d`` = ``day_labeler(close_ts)`` si hay labeler (eje de sesión ET,
    :func:`src.risk.session_axis.make_session_day_labeler`); si no, las fechas
    de calendario (mismo convenio que
    :func:`src.risk.mc_portfolio.precompute_strategy`). El orden de filas se
    conserva: la fila i es el trade local i de la pata.

    ``ValueError`` si faltan columnas o metadatos, hay ``lots <= 0`` o valores
    no finitos / negativos donde el contrato exige >= 0.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"tabla de trades sin columnas requeridas: {missing}")
    if "pv" not in meta:
        raise ValueError("meta sin 'pv' (USD por unidad de precio y lote)")
    n = len(df)
    lots = df["lots"].to_numpy(dtype=float)
    if not np.all(np.isfinite(lots)) or (lots <= 0).any():
        raise ValueError("tabla de trades con lots <= 0 o no finitos")
    pnl_net = df["pnl_net"].to_numpy(dtype=float)
    mae = df["mae_usd"].to_numpy(dtype=float)
    comm = df["comm_usd"].to_numpy(dtype=float)
    open_px = df["open_px"].to_numpy(dtype=float)
    sl_px = df["sl_px"].to_numpy(dtype=float)
    for name, arr in (("pnl_net", pnl_net), ("mae_usd", mae), ("comm_usd", comm),
                      ("open_px", open_px), ("sl_px", sl_px)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"tabla de trades con {name} no finito")
    if (mae < 0).any() or (comm < 0).any():
        raise ValueError("tabla de trades con mae_usd o comm_usd negativos "
                         "(el contrato exige >= 0)")
    swap_cost = np.zeros(n)
    if "swap_usd" in df.columns:
        swap = df["swap_usd"].to_numpy(dtype=float)
        if not np.all(np.isfinite(swap)):
            raise ValueError("tabla de trades con swap_usd no finito")
        swap_cost = np.maximum(-swap, 0.0)
    if (open_px == sl_px).any():
        raise ValueError("tabla de trades con sl_px == open_px (riesgo inicial "
                         "nulo): el sizing Opción A no está definido")
    pv = float(meta["pv"])

    open_ts = pd.to_datetime(df["open_ts"])
    close_ts = pd.to_datetime(df["close_ts"])
    if open_ts.isna().any() or close_ts.isna().any():
        raise ValueError("tabla de trades con open_ts/close_ts NaT")
    open_t = open_ts.to_numpy(dtype="datetime64[ns]")
    close_t = close_ts.to_numpy(dtype="datetime64[ns]")
    if day_labeler is None:
        close_d = close_ts.dt.date.to_numpy()
    else:
        close_d = np.asarray(day_labeler(close_ts))

    pnl_per_lot = pnl_net / lots
    is_buy = df["is_buy"].to_numpy(dtype=bool)
    strat = StrategyMC(
        key=key, open_t=open_t, close_t=close_t, close_d=close_d,
        valle={}, pnl_per_lot=pnl_per_lot, is_buy=is_buy, pv=pv,
        lots=np.empty((0, n)), n_trades=n,
    )
    leg = FuturesLegData(
        adverse_per_lot=(mae + comm + swap_cost) / lots,
        risk_per_micro=np.abs(open_px - sl_px) * pv,
        comm_per_lot=comm / lots,
        pnl_per_lot=pnl_per_lot,
        symbol=str(meta.get("symbol", "")),
        pv=pv,
        backtest_precision=meta.get("backtest_precision"),
    )
    return strat, leg


def fixed_risk_micros(risk_per_micro, f_grid, account_size: float,
                      cap: float | None = None,
                      comm_per_micro=None, *, lot_step: float = 1.0,
                      min_lot: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Sizing Opción A: riesgo fijo en USD por trade, en lotes.

    Por f: ``X = f · account_size`` (estático, sin compounding) y
    ``n_j = min(cap, max(min_lot, floor(X / (risk_j · lot_step)) · lot_step))``
    (``cap=None`` → sin tope). Por defecto micros enteros (``lot_step`` =
    ``min_lot`` = 1, el comportamiento original); CFD: 0,01 y 0,01, que es como
    redondea el MM FixedAmount del EA (al paso, hacia abajo).
    Devuelve ``(lots, diag)`` con ``lots`` de forma ``(n_f, n_trades)`` (float)
    y ``diag`` = una lista (una entrada por f) de dicts:

    - ``f``, ``X``;
    - ``frac_clamp_up``: fracción de trades con ``X < risk · min_lot`` (el clamp
      al lote mínimo sube el riesgo efectivo por encima de X; el error NO está
      acotado por este porcentaje — lección f-grid, R5 del plan);
    - ``p95_risk_eff_over_X``: p95 de ``n · risk / X``;
    - ``frac_capped``: fracción de trades en que el cap recorta (``floor`` > cap);
    - ``p95_total_adverse_over_X`` (solo con ``comm_per_micro``, escalar o
      array por trade; enmienda V5): p95 de ``n · (risk + comm) / X``.

    ``risk_per_micro`` debe ser > 0 y finito; ``f`` y ``account_size`` > 0
    (``ValueError`` si no).
    """
    risk = np.asarray(risk_per_micro, dtype=float)
    if risk.ndim != 1:
        raise ValueError("risk_per_micro debe ser un vector 1-D (n_trades,)")
    if not np.all(np.isfinite(risk)) or (risk <= 0).any():
        raise ValueError("risk_per_micro debe ser > 0 y finito en todo trade "
                         "(SL == precio de apertura o dato corrupto)")
    if not account_size > 0:
        raise ValueError("account_size debe ser > 0")
    if not (lot_step > 0 and min_lot > 0 and np.isfinite(lot_step)
            and np.isfinite(min_lot)):
        raise ValueError("lot_step y min_lot deben ser > 0 y finitos")
    if cap is not None and float(cap) < float(min_lot):
        raise ValueError(f"cap ({cap}) menor que el lote mínimo ({min_lot})")
    comm = None
    if comm_per_micro is not None:
        comm = np.broadcast_to(np.asarray(comm_per_micro, dtype=float),
                               risk.shape)
    fs = np.atleast_1d(np.asarray(f_grid, dtype=float))
    if (fs <= 0).any() or not np.all(np.isfinite(fs)):
        raise ValueError("f_grid debe ser > 0 y finito")
    lots = np.empty((len(fs), len(risk)), dtype=float)
    diag: list[dict] = []
    for i, f in enumerate(fs):
        x = float(f) * float(account_size)
        if lot_step == 1.0 and min_lot == 1.0:   # micros enteros: camino original
            raw = np.maximum(1.0, np.floor(x / risk + _FLOOR_EPS))
        else:
            steps = np.floor(x / (risk * lot_step) + _FLOOR_EPS)
            # round: limpia el ruido de coma flotante de k·0,01 (p.ej. 0,30000000000000004)
            raw = np.maximum(float(min_lot), np.round(steps * lot_step, 10))
        n = raw if cap is None else np.minimum(float(cap), raw)
        lots[i] = n
        d = {
            "f": float(f),
            "X": x,
            "frac_clamp_up": (float(np.mean(x < risk * min_lot))
                              if len(risk) else 0.0),
            "p95_risk_eff_over_X": (float(np.percentile(n * risk / x, 95))
                                    if len(risk) else 0.0),
            "frac_capped": (0.0 if cap is None or not len(risk)
                            else float(np.mean(raw > float(cap)))),
        }
        if comm is not None:
            d["p95_total_adverse_over_X"] = (
                float(np.percentile(n * (risk + comm) / x, 95))
                if len(risk) else 0.0
            )
        diag.append(d)
    return lots, diag
