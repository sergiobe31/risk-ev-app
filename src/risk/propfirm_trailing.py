"""Constructor de datos del trailing intradía — plan Apex Fase 2 (adjudicado).

Modelo por minuto dentro de cada bloque flat-flat, sobre el calendario conjunto
(unión de los timestamps M1 de las series cubiertas **más TODOS los ct**,
cubiertos o no), con ffill físico dentro de ``[ot, ct)``:

    upper_t = realized_≤t + Σ_abiertos fav_t    (fav largos = max(high, prev_close) − open)
    lower_t = realized_≤t + Σ_abiertos adv_t    (adv = la envolvente adversa del joint-float)
    realized_≤t = cumsum del PnL cerrado por trade, inyectado EN el minuto ct

Cota puntual adjudicada: ``lower_t ≤ equity_real_t ≤ upper_t`` en todo instante
relevante (barras M1, intra-minuto vía OHLC sin supuesto de orden, cierres vía
realized exacto, bordes flat). Convenciones DIFERENTES del joint-float
(``propfirm._joint_float_columns``, que NO se toca — su máscara mantiene ct
activo, correcta para valles pero con realized produciría doble conteo en t=ct):

1. Envolvente activa en ``[ot, ct)`` — CERO EN ct — y realized inyectado EN ct
   (en t=ct: lower == upper == realized, igualdad exacta post-fill).
2. El calendario incluye TODOS los ct (cubiertos o no).
3. Fallback realized-only: bloque sin ninguna serie cubierta ⇒ lower == upper ==
   cumsum de closes en sus cts (nunca no-op silencioso: los escalares y
   agregados salen reales).

Escalares por bloque (relativos al saldo de inicio de bloque):

    block_low = min_t lower_t ;  block_high = max_t upper_t
    block_exc = max_t(cummax(upper)_t − lower_t)

Agregados por día(-sesión): ``rel_low[d]`` / ``rel_high[d]`` / ``rel_exc[d]``
con cummax **DE BLOQUE** (no reseteado por día — el suelo trailing no resetea).
El día se etiqueta con ``day_labeler`` si se pasa (sesión ET, ver
:mod:`src.risk.session_axis`); si no, día calendario. Los días del calendario
de salida sin minutos propios (p.ej. un día con solo un open de trade sin
cobertura) heredan el ESTADO al último minuto del día anterior con minutos
(equity plana = realized arrastrado; es la semántica ffill físico).

Camino solo-portfolio: el trailing exige alineación por minuto sobre el
calendario conjunto; el mono se obtiene como portfolio de 1 pata (paridad ya
testeada) — :func:`src.risk.propfirm.block_day_arrays` mono NO se toca.

Residual anti-conservador documentado (R2 del plan): el wander entre la última
barra M1 y el ct de cada trade no está acotado (el punto de cierre sí es exacto
vía realized).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrailingColumns:
    """Calendario + columnas de envolvente por trade (f-INDEPENDIENTE, precio).

    ``cal``: calendario conjunto (barras de series cubiertas ∪ TODOS los ct).
    ``day_ord``: ordinal de día(-sesión) por minuto de ``cal``.
    ``ct_pos``: posición en ``cal`` del minuto de cierre de CADA trade del
    bloque (cubiertos o no) — ahí se inyecta el realized.
    ``covered_j``: índices (en el bloque) de los trades con serie cubierta;
    ``adv_cols``/``fav_cols``: columnas alineadas a ``cal`` con ffill físico
    dentro de ``[ot, ct)`` y 0 fuera (EN ct incluido).
    """

    cal: np.ndarray
    day_ord: np.ndarray
    ct_pos: np.ndarray
    covered_j: np.ndarray
    adv_cols: list
    fav_cols: list


@dataclass
class TrailingBlockResult:
    """Escalares y agregados por día del trailing de un bloque a un ``f``.

    ``rel_low``/``rel_high``/``rel_exc``/``last_*`` están alineados a
    ``day_ord`` (días CON minutos en el calendario). Los ``last_*`` guardan el
    estado al último minuto del día para rellenar días sin minutos propios
    (ver :func:`align_trailing_days`).
    """

    block_low: float            # min_t lower_t
    block_high: float           # max_t upper_t
    block_exc: float            # max_t(cummax(upper)_t − lower_t)
    day_ord: np.ndarray         # (g,) ordinales de día con minutos
    rel_low: np.ndarray         # (g,) min lower_t del día
    rel_high: np.ndarray        # (g,) max upper_t del día
    rel_exc: np.ndarray         # (g,) max excursión del día (cummax de bloque)
    last_lower: np.ndarray      # (g,) lower_t al último minuto del día
    last_upper: np.ndarray      # (g,) upper_t al último minuto del día
    last_exc: np.ndarray        # (g,) excursión al último minuto del día


def _minute_day_ordinals(cal: np.ndarray, day_labeler) -> np.ndarray:
    """Ordinal de día por minuto del calendario (calendario o sesión ET)."""
    if day_labeler is None:
        return cal.astype("datetime64[D]").astype(np.int64) + 719163
    labels = day_labeler(pd.DatetimeIndex(cal))
    return np.array([pd.Timestamp(d).toordinal() for d in labels], dtype=np.int64)


def build_trailing_columns(b, minute_adv_by_sym: dict, minute_fav_by_sym: dict,
                           day_labeler=None) -> TrailingColumns:
    """Calendario conjunto + columnas adv/fav por trade de un bloque.

    ``b``: ``PortfolioDayBlock`` (usa ``open_t``/``close_t`` por trade).
    ``minute_adv_by_sym``/``minute_fav_by_sym``: ``{sym_id: minute_valle /
    minute_fav}`` con entradas ``{trade_local: (ts, vals, ot, ct)}``. Un trade
    con serie adversa pero sin favorable (firma trailing sin picos) es un
    error de setup: raise ruidoso, nunca degradación silenciosa.
    """
    ots = np.asarray(b.open_t)
    cts = np.asarray(b.close_t)
    if len(ots) != b.n_trades or len(cts) != b.n_trades:
        raise ValueError(
            "trailing requiere open_t/close_t por trade en el bloque "
            "(build_day_blocks_portfolio los adjunta desde Fase 2)"
        )
    covered, series = [], []
    for j in range(b.n_trades):
        si, tl = int(b.sym_id[j]), int(b.trade_local_id[j])
        adv = (minute_adv_by_sym.get(si) or {}).get(tl)
        fav = (minute_fav_by_sym.get(si) or {}).get(tl)
        if adv is None and fav is None:
            continue
        if adv is None or fav is None:
            raise ValueError(
                f"trailing: trade {tl} de la pata {si} tiene envolvente "
                f"{'adversa' if adv is not None else 'favorable'} pero no la "
                "otra — la firma trailing exige attach_minute_valleys(..., "
                "with_peaks=True) en TODAS las patas"
            )
        covered.append(j)
        series.append((pd.Timestamp(ots[j]), pd.Timestamp(cts[j]),
                       adv[0], adv[1], fav[1]))
    # Calendario = unión de timestamps cubiertos + TODOS los ct.
    parts = [cts] + [s[2] for s in series]
    cal = np.unique(np.concatenate(parts))
    m = len(cal)
    ct_pos = cal.searchsorted(cts)
    day_ord = _minute_day_ordinals(cal, day_labeler)
    adv_cols, fav_cols = [], []
    for ot, ct, ts, adv_vals, fav_vals in series:
        pos = cal.searchsorted(ts)
        out = []
        for vals in (adv_vals, fav_vals):
            col = np.full(m, np.nan)
            col[pos] = vals
            # ffill físico; fuera de [ot, ct) → 0 (EN ct incluido: la envolvente
            # muere en el minuto de cierre, donde realized toma el relevo).
            src = np.where(np.isnan(col), 0, np.arange(m))
            np.maximum.accumulate(src, out=src)
            col = col[src]
            col[np.isnan(col)] = 0.0
            col[(cal < ot.to_datetime64()) | (cal >= ct.to_datetime64())] = 0.0
            out.append(col)
        adv_cols.append(out[0])
        fav_cols.append(out[1])
    return TrailingColumns(
        cal=cal, day_ord=day_ord, ct_pos=ct_pos,
        covered_j=np.array(covered, dtype=int),
        adv_cols=adv_cols, fav_cols=fav_cols,
    )


def trailing_minute_bounds(b, lots: np.ndarray, pv_by_sym: dict[int, float],
                           tcols: TrailingColumns) -> tuple[np.ndarray, np.ndarray]:
    """``(lower_t, upper_t)`` del bloque a un ``f`` sobre el calendario conjunto.

    ``lots[j]`` = lotes del trade j del bloque a este f (re-floor discreto, no
    prefactorizable). Realized = cumsum de ``lots×pnl_per_lot`` inyectado EN el
    minuto ct de cada trade (todos, cubiertos o no; varios closes en el mismo
    minuto se suman). Bloque sin trades cubiertos ⇒ lower == upper == cumsum
    de closes (fallback realized-only).
    """
    m = len(tcols.cal)
    inj = np.zeros(m)
    pnl_trade = lots * b.pnl_per_lot
    np.add.at(inj, tcols.ct_pos, pnl_trade)
    realized = np.cumsum(inj)
    lower = realized.copy()
    upper = realized.copy()
    for k, j in enumerate(tcols.covered_j):
        w = lots[j] * pv_by_sym[int(b.sym_id[j])]
        lower += tcols.adv_cols[k] * w
        upper += tcols.fav_cols[k] * w
    return lower, upper


def trailing_block_result(b, lots: np.ndarray, pv_by_sym: dict[int, float],
                          minute_adv_by_sym: dict, minute_fav_by_sym: dict,
                          day_labeler=None,
                          tcols: TrailingColumns | None = None) -> TrailingBlockResult:
    """Escalares y agregados por día(-sesión) del trailing de un bloque a un f.

    ``tcols`` opcional: salida cacheada de :func:`build_trailing_columns`
    (f-independiente) para reusar calendario y columnas entre los f de una
    grilla — lo hace ``precompute_day_arrays_portfolio``.
    """
    if tcols is None:
        tcols = build_trailing_columns(b, minute_adv_by_sym, minute_fav_by_sym,
                                       day_labeler=day_labeler)
    lower, upper = trailing_minute_bounds(b, lots, pv_by_sym, tcols)
    run = np.maximum.accumulate(upper)
    exc = run - lower
    starts = np.flatnonzero(np.r_[True, np.diff(tcols.day_ord) != 0])
    ends = np.r_[starts[1:], len(tcols.cal)]
    days = tcols.day_ord[starts]
    return TrailingBlockResult(
        block_low=float(lower.min()),
        block_high=float(upper.max()),
        block_exc=float(exc.max()),
        day_ord=days,
        rel_low=np.minimum.reduceat(lower, starts),
        rel_high=np.maximum.reduceat(upper, starts),
        rel_exc=np.maximum.reduceat(exc, starts),
        last_lower=lower[ends - 1],
        last_upper=upper[ends - 1],
        last_exc=exc[ends - 1],
    )


def align_trailing_days(res: TrailingBlockResult,
                        day_ord_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alinea ``rel_low``/``rel_high``/``rel_exc`` al ``day_ord`` de salida.

    Día con minutos propios ⇒ sus agregados. Día sin minutos (p.ej. solo un
    open de trade sin cobertura) ⇒ estado al último minuto del día anterior
    con minutos (ffill físico: equity plana = realized arrastrado, excursión
    con cummax de bloque); si no hay día anterior con minutos ⇒ 0.0 (inicio
    de bloque flat).
    """
    n = len(day_ord_out)
    rel_low = np.zeros(n)
    rel_high = np.zeros(n)
    rel_exc = np.zeros(n)
    pos = np.searchsorted(res.day_ord, day_ord_out)
    for i, d in enumerate(day_ord_out):
        p = int(pos[i])
        if p < len(res.day_ord) and res.day_ord[p] == d:
            rel_low[i] = res.rel_low[p]
            rel_high[i] = res.rel_high[p]
            rel_exc[i] = res.rel_exc[p]
        elif p > 0:
            rel_low[i] = res.last_lower[p - 1]
            rel_high[i] = res.last_upper[p - 1]
            rel_exc[i] = res.last_exc[p - 1]
    return rel_low, rel_high, rel_exc
