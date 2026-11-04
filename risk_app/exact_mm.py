"""Modo EXACTO por pata (portfolio): sizing ATR real desde el .sqx de la estrategia.

El modo rápido back-deriva ``slInMoney = f_ref·E/Size`` del CSV, lo que sesga
~20% en lotes chicos (y más si la referencia estuvo clamped). El modo exacto
recomputa ``slInMoney`` desde el ATR real (Wilder shift1, réplica iATR MT5)
sobre el OHLC M1, con los params leídos del propio ``.sqx`` (risk, ATRPeriod,
ATRMult, MaxLots) — el mismo ground truth que el modo exacto single, ya
validado bit-exact contra la columna Size del CSV (``test_risk_sizing_parity``).

Vive en ``risk_app/`` (no en ``src/risk``) porque importa ``sqx.risk_mm_parse``;
el guardián AST prohíbe ``src/* → sqx``. No importa streamlit (testeable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sqx.risk_mm_parse import StrategyMM, parse_strategy_mm
from src.risk.atr import sl_in_money_at_entries
from src.risk.trades import TradeList

# Campos del MM imprescindibles para el modo exacto.
_REQUIRED_MM = ("risk", "atr_period", "atr_mult")

# NOTA (hardening 2026-10-06): se evaluó y REFUTÓ un veto de mismatch TF por
# rejilla de entradas ("todas las entradas en boundaries del TF del .sqx"):
# las patas reales tienen fills INTRABAR (stops/limits) — en
# `nasdaq_2h/Strategy 4.49.131(1)` 529/532 entradas caen fuera de la rejilla
# H2 (p.ej. 21:59, 14:03) — así que el check daría falso veto. La protección
# contra .sqx de otro TF sigue siendo `_exact_size_crosscheck` (banda
# [0.90, 1.01], warning no veto) en el caller de la API.


def exact_sizing_from_sqx(
    sqx_source: bytes | str,
    tl: TradeList,
    ohlc_m1: pd.DataFrame,
    point_value: float,
) -> tuple[np.ndarray, StrategyMM]:
    """``slInMoney`` por trade desde el ATR real + el MM parseado del .sqx.

    Args:
        sqx_source: bytes o ruta del .sqx de la estrategia (NO de un portfolio;
            ``parse_strategy_mm`` los rechaza).
        tl: trades de la pata (sus ``open_time`` mapean el ATR shift-1).
        ohlc_m1: OHLC M1 del símbolo (mismo feed que el backtest).
        point_value: USD por 1.0 de precio por lote (broker_config).

    Returns:
        (sl_in_money, mm): array (n_trades,) USD de riesgo por lote, y el
        StrategyMM parseado (risk en %, max_lots, etc.) para que el caller
        derive ``f_ref = mm.risk/100`` y ``max_lots``.

    Raises:
        ValueError: si el MM no es ATRRiskBasedSizing o le faltan params.
    """
    mm = parse_strategy_mm(sqx_source)
    if mm.mm_type != "ATRRiskBasedSizing":
        raise ValueError(
            f"MM '{mm.mm_type}' no soportado en modo exacto (solo ATRRiskBasedSizing). "
            "Deja la pata en modo rápido o re-exporta con el bloque MM correcto."
        )
    missing = [f for f in _REQUIRED_MM if getattr(mm, f) is None]
    if missing:
        raise ValueError(f"MM del .sqx incompleto: faltan {missing}.")
    if not mm.timeframe:
        raise ValueError("MM del .sqx sin timeframe; no puedo resamplear el ATR.")

    # Correspondencia .sqx <-> CSV: mismo símbolo (raíz). Un .sqx de OTRA
    # estrategia/símbolo aplicaría un ATR de otro instrumento a estos trades.
    if mm.symbol:
        csv_root = str(tl.df["symbol"].iloc[0]).split(".")[0].split("_")[0]
        sqx_root = str(mm.symbol).split(".")[0].split("_")[0]
        if csv_root != sqx_root:
            raise ValueError(
                f"el .sqx es de '{mm.symbol}' pero el CSV es de "
                f"'{tl.df['symbol'].iloc[0]}' — no corresponden. Sube el .sqx "
                f"de ESTA estrategia."
            )

    entries = pd.to_datetime(tl.df["open_time"], errors="coerce")
    if entries.isna().any():
        raise ValueError("Trades sin open_time parseable; no puedo mapear el ATR.")
    sl = sl_in_money_at_entries(
        ohlc_m1,
        entries,
        timeframe=mm.timeframe,
        atr_period=int(mm.atr_period),
        atr_mult=float(mm.atr_mult),
        point_value=float(point_value),
    )
    sl = np.asarray(sl, dtype=float)
    # isfinite tras el mapeo ATR: NaN/inf solo pueden venir de params
    # corruptos del .sqx (ATRMult/point_value) — hoy downstream los
    # silenciaría (nan_to_num → riesgo 0) o simularía con inf.
    n_bad = int((~np.isfinite(sl)).sum())
    if n_bad:
        raise ValueError(
            f"{n_bad}/{len(sl)} slInMoney no finitos (NaN/inf) tras el mapeo "
            "ATR: revisa los params del .sqx (ATRMult), el point_value y el "
            "OHLC."
        )
    return sl, mm
