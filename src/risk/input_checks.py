"""Checks de cordura de inputs para la app de riesgo (modo portfolio).

El motor MC está validado; el riesgo práctico restante es un INPUT malo que
produce un número de aspecto plausible sobre datos basura. Tres guards:

1. **CSV multi-símbolo** — el CSV del Portfolio Master colado como "una
   estrategia" (los trades de otros símbolos se valoran con el OHLC/point
   value del primero → flotante basura). Veto explícito.
2. **Capital inicial del backtest != baseline de la etapa** — los sizes del
   CSV escalan con la equity del backtest; si no es la Allocation ($5k Seed),
   las barreras ±7% se evalúan contra lotes de otra escala. Warning.
3. **Coste implícito por trade vs esperado del yaml** — comisión o point
   value mal configurados en SQX (el bug "Per trade", comm>0 en un símbolo
   comm-0) dejan un coste por trade incoherente con el yaml. El coste
   implícito = bruto (precios × pv × size) − neto del CSV ≈ comisión + swap
   (el SPREAD ya va embebido en los precios de ejecución del CSV — NO es
   detectable desde el CSV; su guard es la calibración de 1 trade en SQX).
   Warning si el ratio sale de [0.5, 2]× o si hay coste donde se espera ~0.

Módulo puro: NO importa ``sqx`` (guardián AST).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.broker_config import BrokerSymbol
from src.risk.trades import TradeList

# Tolerancias del check de coste implícito.
_COST_RATIO_HI = 2.0
_COST_RATIO_LO = 0.5


def multi_symbol_counts(tl: TradeList) -> dict[str, int] | None:
    """``{simbolo: n_trades}`` si el CSV mezcla >1 símbolo; None si es limpio.

    El modo portfolio espera UN CSV por estrategia. Un CSV del Portfolio
    Master (o una concatenación manual) mezcla símbolos y contamina el
    símbolo → OHLC/point value de toda la pata.
    """
    vals, counts = np.unique(tl.symbol, return_counts=True)
    if len(vals) <= 1:
        return None
    return {str(v): int(c) for v, c in zip(vals, counts)}


def initial_capital_mismatch(
    tl: TradeList, baseline: float, rtol: float = 0.05
) -> str | None:
    """Mensaje de warning si la equity inicial del backtest difiere de la baseline.

    El sizing del CSV escala con la equity del backtest; el MC evalúa
    barreras ±7% sobre la Allocation. Para que cuadren, el backtest debe
    haberse corrido con la Allocation como equity inicial (Seed = $5.000).
    """
    e0 = tl.initial_equity
    if e0 <= 0 or baseline <= 0:
        return None
    if abs(e0 - baseline) / baseline <= rtol:
        return None
    return (
        f"capital inicial del backtest ${e0:,.0f} != baseline de la etapa "
        f"${baseline:,.0f} — los lotes del CSV están escalados a otra equity; "
        f"el P(pass) no es fiable. Corre el backtest con la Allocation como "
        f"capital inicial."
    )


@dataclass(frozen=True)
class LegDiagnostics:
    """Diagnóstico por pata: clamp del min_lot + coste implícito vs esperado."""

    n_trades: int
    frac_min_lot_ref: float        # fracción de trades del CSV con Size == min_lot
    floor_risk_pct: float          # riesgo suelo mediano (min_lot × slInMoney), % de baseline
    implied_cost_per_lot: float    # coste implícito medio por lote (USD)
    expected_cost_per_lot: float   # comm round-trip + spread + swap estimado (USD/lote)
    cost_ratio: float              # implied/expected (NaN si no computable)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _is_buy_series(tl: TradeList) -> np.ndarray:
    return tl.df["type"].astype(str).str.lower().str.startswith("b").to_numpy()


def _nights_in_trade(tl: TradeList) -> np.ndarray:
    """Noches held por trade (0 si no hay timestamps). Triple swap ignorado."""
    if "open_time" not in tl.df.columns:
        return np.zeros(tl.n_trades)
    ot = pd.to_datetime(tl.df["open_time"], errors="coerce")
    ct = pd.to_datetime(tl.df["close_time"], errors="coerce")
    nights = (ct.dt.normalize() - ot.dt.normalize()).dt.days
    return np.nan_to_num(nights.to_numpy(dtype=float), nan=0.0)


def leg_diagnostics(
    tl: TradeList,
    sym: BrokerSymbol,
    *,
    f_ref: float,
    baseline: float,
) -> LegDiagnostics:
    """Diagnóstico de una pata: clamp min_lot, riesgo suelo y coste implícito.

    - ``frac_min_lot_ref``: fracción de trades con ``Size == min_lot``. Alta =
      la pata estaba en el suelo de lote en el backtest (bajar ``f`` no la
      des-riesga; su riesgo real es mayor que el nominal).
    - ``floor_risk_pct``: ``mediana(min_lot × slInMoney) / baseline``. El
      riesgo indeleble por trade aunque ``f → 0``. Caveat: ``slInMoney``
      back-derivado sesga BAJO si el Size de referencia ya estaba clamped
      (el suelo real es aún mayor) — es una cota inferior.
    - ``cost_ratio``: coste implícito por lote (bruto de precios − neto del
      CSV ≈ comisión + swap) vs esperado del yaml. ~1 = config de costes SQX
      sana; fuera de [0.5, 2] = revisar comisión/point value en SQX. En
      símbolos comm-0/swap-0 (.fs) el check es absoluto: cualquier coste
      implícito delata un point value distinto al del yaml. El spread NO es
      detectable aquí (va embebido en los precios del CSV).
    """
    n = tl.n_trades
    warnings: list[str] = []

    # --- clamp min_lot en la referencia + riesgo suelo ---
    frac_min = float(np.mean(np.abs(tl.size - sym.min_lot) <= 1e-9)) if n else 0.0
    if frac_min > 0.20:
        warnings.append(
            f"{frac_min:.0%} de trades con Size == min_lot ({sym.min_lot}): la pata "
            f"está en el suelo de lote; bajar f NO reduce su riesgo."
        )
    floor_pct = float("nan")
    if n and baseline > 0 and f_ref > 0:
        sl = tl.sl_in_money(f_ref)
        sl = sl[np.isfinite(sl) & (sl > 0)]
        if len(sl):
            floor_pct = float(np.median(sym.min_lot * sl) / baseline * 100.0)

    # --- coste implícito vs esperado ---
    implied = float("nan")
    expected = float("nan")
    ratio = float("nan")
    has_prices = (
        {"open_price", "close_price", "type"} <= set(tl.df.columns)
        and tl.df["close_price"].notna().any()
    )
    if n and has_prices:
        is_buy = _is_buy_series(tl)
        signo = np.where(is_buy, 1.0, -1.0)
        open_p = tl.df["open_price"].to_numpy(dtype=float)
        close_p = tl.df["close_price"].to_numpy(dtype=float)
        gross = signo * (close_p - open_p) * sym.point_value_usd * tl.size
        implied_total = float(np.sum(gross - tl.pnl))  # coste total pagado
        total_lots = float(np.sum(tl.size))
        if total_lots > 0:
            implied = implied_total / total_lots
        # esperado: comisión round-trip + swap por noches (el SPREAD no entra:
        # ya va embebido en los precios de ejecución del CSV).
        spread_typ_usd, _ = sym.spread_usd_per_lot()
        expected = sym.commission_round_trip_usd()
        nights = _nights_in_trade(tl)
        if nights.any() and (sym.swap_long_value or sym.swap_short_value):
            price_med = float(np.nanmedian(open_p))
            swap_long, swap_short = sym.swap_per_day_per_lot(current_price=price_med)
            swap_side = np.where(is_buy, swap_long, swap_short)
            # el swap es coste si negativo; el esperado suma el coste medio
            expected += float(-np.mean(swap_side * nights))
        if expected > 0 and np.isfinite(implied):
            ratio = implied / expected
            if ratio > _COST_RATIO_HI:
                warnings.append(
                    f"coste implícito {ratio:.1f}× lo esperado del yaml "
                    f"(${implied:.2f} vs ${expected:.2f}/lote): revisa comisión / "
                    f"point value de {sym.ticker_axi} en SQX."
                )
            elif ratio < _COST_RATIO_LO:
                warnings.append(
                    f"coste implícito {ratio:.2f}× lo esperado del yaml "
                    f"(${implied:.2f} vs ${expected:.2f}/lote): sospecha de point "
                    f"value o comisión mal configurados en SQX."
                )
        elif np.isfinite(implied):
            # Símbolo comm-0/swap-0 (.fs): todo coste implícito es sospechoso
            # (típicamente point value del yaml != el que usó SQX).
            if implied > 0.5 * spread_typ_usd:
                warnings.append(
                    f"coste implícito ${implied:.2f}/lote con comisión y swap 0 "
                    f"esperados: sospecha de point value de {sym.ticker_axi} "
                    f"distinto en SQX (yaml ${sym.point_value_usd})."
                )
            elif implied < -0.5 * spread_typ_usd:
                warnings.append(
                    f"coste implícito NEGATIVO (${implied:.2f}/lote): inconsistente; "
                    f"revisa point value y que precios y PnL sean de la misma corrida."
                )

    return LegDiagnostics(
        n_trades=n,
        frac_min_lot_ref=frac_min,
        floor_risk_pct=floor_pct,
        implied_cost_per_lot=implied,
        expected_cost_per_lot=expected,
        cost_ratio=ratio,
        warnings=tuple(warnings),
    )
