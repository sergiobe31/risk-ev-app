"""Sweep del % de riesgo ``f`` (ATR Risk Sized) con re-floor de lot-step.

Linealidad (exacta a ``f = f_ref`` por construction; aproximada a ``f != f_ref``
si la referencia fue clamped/floored — ver caveat en :mod:`trades` y
:func:`reference_clamping_fraction`): con sizing ATR-risk dinamico, el
retorno % por trade es ``r_n(f) = f * y_n``. El naive ``prod(1 + f*y_n)`` (sin re-floor) es
silenciosamente erróneo (hasta 0.24% por la discretizacion de lot-step). Aqui
replayamos trade-a-trade con:

    slInMoney_n = f_ref * equity_before_n / Size_n      (riesgo USD por lote)
    pnl_per_lot_n = PnL_n / Size_n                       (comision size-based incluida)
    lots_cont   = f * E_n / slInMoney_n
    lots_margin = E_n * leverage / (contract_size * price_n)   # clip 100:1 real AXI
    L_n         = floor(min(lots_cont, lots_margin) / lot_step) * lot_step   # RE-FLOOR
    # guarda LotsIfNoMM: si L_n < min_lot -> clamp a min_lot + flag (NUNCA 1.0 default)
    PnL_n       = L_n * pnl_per_lot_n
    E_{n+1}     = E_n + PnL_n

PRECONDICION (no verificable desde el CSV): exits price/time/ATR. Si una regla
tiene exit equity-contingent, esta funcion NO es valida para esa estrategia.

El replay se vectoriza sobre la grilla de ``f`` (un array (n_f,) de equity que
avanza trade a trade), O(n_trades * n_f) con numpy.

Base del lote (``compound``, guardrail csv-d1-d2, defecto D1, 2026-10-19). Por defecto
(``compound=True``) el lote compone sobre la equity del replay, ``E_n`` en la receta de arriba:
comportamiento historico, bit a bit. Con ``compound=False`` el lote continuo es
``lots_cont = f * E_0 / slInMoney_n`` con ``E_0 = tl.initial_equity`` FIJO (tamano de cuenta, sin
reinvertir): el lote de cada orden no depende del PnL de las anteriores. Todo lo demas (margen
opcional, re-floor, ``LotsIfNoMM``, ``MaxLots``, lote minimo, PnL) es identico; la equity del replay
sigue avanzando con el PnL de esos lotes fijos (diagnostico) y describe la estrategia a lote fijo,
no la cuenta (ver :class:`SweepResult`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import re

import numpy as np

from src.broker_config import BrokerSymbol
from src.risk.trades import TradeList


@dataclass
class SweepResult:
    """Resultado del sweep sobre una grilla de ``f``.

    Con ``simulate_f(..., compound=False)`` (lote sobre el tamano de cuenta, sin componer),
    ``equity``, ``mtm_min`` y ``blown_up`` describen la estrategia a lote fijo, no la cuenta: el
    replay acumula el PnL de esos lotes fijos, pero el lote no reacciona a esa equity (tampoco tras
    un ``blown_up``). Quien necesite la equity de una cuenta que reinvierte debe usar ``compound=True``.
    """

    f_grid: np.ndarray  # (n_f,) fracciones de riesgo
    equity: np.ndarray  # (n_f, n_trades+1) equity post-trade; equity[:,0]=E_0
    mtm_min: np.ndarray  # (n_f, n_trades) min equity DENTRO de cada trade (ruin MTM)
    lots: np.ndarray  # (n_f, n_trades) lotes ejecutados por trade
    pnl: np.ndarray  # (n_f, n_trades) PnL por trade a ese f
    n_sub_min_lot: np.ndarray  # (n_f,) cuantos trades se clampearon a min_lot
    n_margin_clipped: np.ndarray  # (n_f,) cuantos trades tocaron el clip de margen
    blown_up: np.ndarray  # (n_f,) bool: equity <= 0 en algun punto

    @property
    def n_f(self) -> int:
        return self.f_grid.shape[0]

    def terminal(self) -> np.ndarray:
        return self.equity[:, -1]

    def realized_max_dd_fraction(self, baseline: float | None = None) -> np.ndarray:
        """Max drawdown realizado (fraccion, negativo) por ``f`` sobre la equity path."""
        eq = self.equity
        base = self.equity[:, 0] if baseline is None else baseline
        running_max = np.maximum.accumulate(eq, axis=1)
        dd = (eq - running_max) / np.maximum(running_max, 1e-12)
        return dd.min(axis=1)  # (n_f,) el mas negativo


def _resolve_symbol(ticker: str, symbols: Mapping[str, BrokerSymbol]) -> BrokerSymbol:
    if ticker in symbols:
        return symbols[ticker]
    # tolerancia: casar por raiz, stripping sufijos de broker (.pa/.sa/_axi/_pro...).
    # Tambien por ticker_cascade (raiz del simbolo en datos SQX/Dukascopy: el CSV
    # puede decir DEUIDXEUR_axi aunque el ticker AXI sea DAX40.fs).
    root = re.split(r"[._]", str(ticker))[0]
    for key, sym in symbols.items():
        if re.split(r"[._]", key)[0] == root or sym.ticker_cascade == root:
            return sym
    raise KeyError(
        f"Symbol '{ticker}' (raiz '{root}') no esta en el map de specs "
        f"(disponibles: {list(symbols)}). Anadelo a la config del broker."
    )


def simulate_f(
    tl: TradeList,
    f_grid: np.ndarray | float,
    *,
    symbols: Mapping[str, BrokerSymbol],
    leverage: int = 100,
    max_lots: float = float("inf"),
    lots_if_no_mm: float | None = None,
    sl_in_money: np.ndarray | None = None,
    f_ref: float | None = None,
    use_margin_clip: bool = False,
    compound: bool = True,
) -> SweepResult:
    """Replaya la lista de trades sobre una grilla de ``f`` con re-floor.

    Args:
        tl: lista de trades (referencia).
        f_grid: fracciones de riesgo a barrer (p.ej. np.arange(0.0025, 0.0201, 0.0025)).
            Acepta escalar.
        symbols: map ``ticker_axi -> BrokerSymbol`` (debe cubrir todos los
            simbolos del trade list). Da lot_step, min_lot, contract_size.
        leverage: apalancamiento (solo si ``use_margin_clip``).
        max_lots: ``#MaxLots#`` del bloque ATRRiskBasedSizing (p.ej. 5.0).
        lots_if_no_mm: ``#LotsIfNoMM#`` (lote si el calculado es <=0). Default
            = ``min_lot`` del simbolo.
        sl_in_money: **modo EXACTO** — ``slInMoney`` por trade (USD/lote),
            tipicamente de :func:`src.risk.atr.sl_in_money_at_entries` (ATR
            real). Si se da, NO se back-deriva.
        f_ref: **modo rapido** — fraccion de riesgo del reference (>0) para
            back-derivar ``slInMoney = f_ref*E/Size``, con ``E = tl.equity_before`` (la equity
            del CSV, nunca la del replay, tambien con ``compound=False``). Se ignora si hay
            ``sl_in_money``.
        use_margin_clip: aplicar clip de margen (el MM de SQX NO lo hace; off).
        compound: base del lote continuo (defecto D1 del guardrail csv-d1-d2).
            ``True`` (por defecto, comportamiento historico, bit a bit): la equity del replay,
            ``lots_cont = f * E_n / sl`` con ``E_n`` avanzando con el PnL de cada orden.
            ``False``: el tamano de cuenta fijo, ``lots_cont = f * E_0 / sl`` con
            ``E_0 = tl.initial_equity``; el lote de cada orden no depende del PnL de las
            anteriores. El resto del sizing no cambia: margen opcional (``use_margin_clip`` sigue
            usando la equity del replay), re-floor a ``lot_step``, ``LotsIfNoMM``, ``MaxLots`` y lote
            minimo (bloque «Bounds fieles al MQL5 de SQX» del bucle), ``n_sub_min_lot`` y PnL.
            Con ``compound=False``, ``equity``, ``mtm_min`` y ``blown_up`` del resultado describen la
            estrategia a lote fijo, no la cuenta (ver :class:`SweepResult`).

    Se requiere ``sl_in_money`` O ``f_ref``.
    """
    f_grid = np.atleast_1d(np.asarray(f_grid, dtype=float))
    if (f_grid <= 0).any():
        raise ValueError("f_grid debe ser > 0")
    if sl_in_money is None and f_ref is None:
        raise ValueError("Hace falta sl_in_money (modo exacto) o f_ref (modo rapido)")
    if sl_in_money is None and f_ref <= 0:
        raise ValueError("f_ref debe ser > 0")

    n_f = f_grid.shape[0]
    n_t = tl.n_trades
    e0 = tl.initial_equity

    # Pre-resolver specs por trade (vectorizable pero n_t pequeno; loop claro).
    lot_step = np.empty(n_t)
    min_lot = np.empty(n_t)
    contract_size = np.empty(n_t)
    for i, ticker in enumerate(tl.symbol):
        sym = _resolve_symbol(str(ticker), symbols)
        lot_step[i] = sym.lot_step
        min_lot[i] = sym.min_lot
        contract_size[i] = sym.contract_size

    # slInMoney: modo exacto (ATR) o back-derivo (f_ref).
    sl_money = np.asarray(sl_in_money, dtype=float) if sl_in_money is not None else tl.sl_in_money(f_ref)
    pnl_per_lot = tl.pnl_per_lot()  # (n_t,)
    mae_per_lot = np.where(tl.size > 0, tl.mae / np.maximum(tl.size, 1e-12), 0.0)
    price = tl.open_price
    # LotsIfNoMM por trade (default = min_lot del simbolo).
    linm = np.full(n_t, lots_if_no_mm if lots_if_no_mm is not None else np.nan)
    linm = np.where(np.isnan(linm), min_lot, linm)

    # Estado vectorial sobre la grilla de f.
    equity = np.empty((n_f, n_t + 1))
    mtm_min = np.empty((n_f, n_t))
    lots_arr = np.empty((n_f, n_t))
    pnl_arr = np.empty((n_f, n_t))
    n_sub = np.zeros(n_f, dtype=int)
    n_margin = np.zeros(n_f, dtype=int)
    blown = np.zeros(n_f, dtype=bool)

    e = np.full(n_f, e0, dtype=float)
    equity[:, 0] = e
    active = np.ones(n_f, dtype=bool)  # paths no volados todavia

    for i in range(n_t):
        if sl_money[i] <= 0 or not np.isfinite(sl_money[i]):
            # trade sin riesgo definido -> no se puede sizizar; skip (PnL 0)
            lots = np.zeros(n_f)
            pnl = np.zeros(n_f)
        else:
            # Base del lote: equity del replay (compuesto, historico) o tamano de cuenta
            # fijo ``e0`` (D1: sin componer, el lote no depende del PnL previo).
            size_base = e if compound else e0
            lots_cont = f_grid * size_base / sl_money[i]  # (n_f,)
            # margin clip OPCIONAL (el MM ATRRiskBasedSizing de SQX NO lo hace).
            if use_margin_clip:
                with np.errstate(divide="ignore", invalid="ignore"):
                    lots_margin = np.where(
                        price[i] > 0,
                        e * leverage / (contract_size[i] * price[i]),
                        np.inf,
                    )
                lots_cont = np.minimum(lots_cont, lots_margin)
            # --- Bounds fieles al MQL5 de SQX (sqMMATRRiskBasedSizing) ---
            # 1) roundDown al lot_step (FLOOR)
            lots = np.floor(lots_cont / lot_step[i]) * lot_step[i]
            # 2) si <= 0 -> LotsIfNoMM
            no_open = lots <= 0
            lots = np.where(no_open, linm[i], lots)
            # 3) si > MaxLots -> MaxLots
            lots = np.minimum(lots, max_lots)
            # 4) si < Smallest_Lot (broker min) -> Smallest_Lot
            below_min = lots < min_lot[i]
            lots = np.where(below_min, min_lot[i], lots)
            n_sub += ((no_open | below_min) & active).astype(int)
            pnl = lots * pnl_per_lot[i]

        e_new = e + pnl
        # blow-up: equity <= 0
        now_blown = active & (e_new <= 0)
        blown |= now_blown
        # tras blow-up, congelar equity en su valor (no seguir)
        e_new = np.where(active, e_new, e)
        # MTM min durante el trade: E_n + mae_per_lot*lots (mae<=0)
        mtm = e + mae_per_lot[i] * lots
        mtm = np.minimum(mtm, e_new)  # el min entre el valle y el cierre

        equity[:, i + 1] = e_new
        mtm_min[:, i] = mtm
        lots_arr[:, i] = lots
        pnl_arr[:, i] = pnl
        # desactivar paths volados (no operan mas)
        active = active & ~now_blown
        e = e_new

    return SweepResult(
        f_grid=f_grid,
        equity=equity,
        mtm_min=mtm_min,
        lots=lots_arr,
        pnl=pnl_arr,
        n_sub_min_lot=n_sub,
        n_margin_clipped=n_margin,
        blown_up=blown,
    )


def default_f_grid(lo: float = 0.0025, hi: float = 0.02, step: float = 0.0025) -> np.ndarray:
    """Grilla por defecto: 0.25%..2% paso 0.25%."""
    return np.arange(lo, hi + step / 2, step)


def reference_clamping_fraction(
    tl: TradeList,
    symbols: Mapping[str, BrokerSymbol],
    atol: float = 1e-9,
) -> float:
    """Heuristica: fraccion de trades de referencia cuyo ``Size == min_lot``.

    Si la fraccion es alta, muchos trades de referencia pudieron estar
    clamped a min_lot por SQX -> el back-derivo de ``slInMoney`` sesga BAJO y el
    sweep SOBREDIMENSIONA a otro ``f`` (ver caveat en trades.py). No es prueba
    definitiva (imposible desde el CSV solo), pero una fraccion alta (p.ej.
    >20%) => VALIDA el sweep contra 2-3 re-runs de SQX antes de fiarte.
    """
    n = 0
    clamped = 0
    for ticker, size in zip(tl.symbol, tl.size):
        try:
            sym = _resolve_symbol(str(ticker), symbols)
        except KeyError:
            continue
        n += 1
        if abs(size - sym.min_lot) <= atol:
            clamped += 1
    return (clamped / n) if n else 0.0
