"""Monte Carlo por bloques flat-flat para portfolios multi-símbolo (AXI Seed).

Estima P(pasar una etapa con barreras ±) para un portfolio de N estrategias que
comparten la cuenta. Modelo A: sizes del backtest individual, conservados al barajar
(no re-derivados sobre equity barajada) — válido mientras la etapa termina cerca del
capital inicial (cuantización del lote; ver ``reference_portfolio_mm_sizing_decision``).

Mecánica: barajea EPISODIOS flat-flat (instantes donde NINGUNA estrategia tiene
posición) del calendario conjunto — preserva la co-ocurrencia inter-símbolo y
RECUPERA tiempo calendario real por path (suma de ``duration_days`` de los bloques
barajados). Reconstruye el flotante worst merged día a día (= saldo cerrado acumulado +
Σ valle_flotante_d) y evalúa el primer cruce de ``pass_line`` / ``ruin_line``.

Reutiliza ``src.risk.mc.stationary_block_indices`` (agnóstico: n = nº de bloques) y
``src.risk.ruin.wilson_ci`` / ``RiskReport``. NO reutiliza
``ruin.compute_path_outcomes`` / ``mc.make_paths`` (acoplados a trade-level con
composición multiplicativa ``1+f·y`` y MTM ``a`` intra-trade — aquí la composición es
aditiva y el flotante worst ya viene por día).

El caller hace el I/O: cargar los CSV orderstocsv (``trades.load_trade_list``), el
OHLC M1, y los sizes por ``f`` (``mm.simulate_f``); construir ``StrategyMC`` y llamar
a ``run_portfolio_mc``.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.risk.mc import stationary_block_indices
from src.risk.ruin import wilson_ci, RiskReport


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

@dataclass
class StrategyMC:
    """Una estrategia del portfolio, precomputada para el MC flat-flat.

    ``valle`` y ``pnl_per_lot`` son f-independientes; ``lots`` (n_f, n_trades) viene
    de ``mm.simulate_f`` y encapsula el clamp por ``f`` (recalculado, no escalado).

    ``minute_valle`` (opcional, vía :func:`attach_minute_valleys`): serie M1 del
    flotante adverso por trade, ``{trade_idx: (ts, vals, ot, ct)}`` en unidades
    de precio — la materia prima del flotante conjunto intradía (``float_mode
    "joint"`` de ``src.risk.propfirm``, veredicto M2-1). None si no se pidió.

    ``minute_fav`` (opcional, vía ``attach_minute_valleys(..., with_peaks=True)``):
    dict PARALELO a ``minute_valle`` con la envolvente FAVORABLE de la misma
    pasada M1 (:func:`minute_float_series`), mismo formato y mismas claves.
    Materia prima del trailing intradía (plan Apex Fase 2,
    ``src.risk.propfirm_trailing``). None si no se pidió.
    """
    key: str
    open_t: np.ndarray
    close_t: np.ndarray
    close_d: np.ndarray            # date por trade
    valle: dict                    # {trade_idx: {date: valle_precio}} (f-indep, ≤0 adverse)
    pnl_per_lot: np.ndarray        # f-indep
    is_buy: np.ndarray
    pv: float                      # pointValue (USD/unidad de precio/lote)
    lots: np.ndarray               # (n_f, n_trades)
    n_trades: int
    minute_valle: dict | None = None
    minute_fav: dict | None = None


@dataclass
class BlockRecord:
    """Episodio flat-flat autocontenido (ambas estrategias sin posición en los bordes)."""
    idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float
    sym_id: np.ndarray             # (k,) índice de símbolo por trade
    trade_local_id: np.ndarray     # (k,) índice local del trade en su símbolo
    close_d: np.ndarray            # (k,) date de close por trade
    valle_precio: list             # (k,) dict {date: vp} f-indep
    pnl_per_lot: np.ndarray        # (k,) f-indep
    n_unique_trades: int


@dataclass
class PathOutcome:
    outcome: str          # 'pass' | 'ruin' | 'timeout' | 'neither'
    days: float           # días calendario acumulados hasta el stop
    trades: int           # n_unique_trades hasta el stop
    max_dd: float         # fracción (negativo) sobre el baseline
    terminal: float       # último merged del path


# ---------------------------------------------------------------------------
# Precompute (f-independiente)
# ---------------------------------------------------------------------------

def _ohlc_col(ohlc: pd.DataFrame, *names: str) -> str:
    """Devuelve la columna OHLC por alias habitual (o/h/l/c, Open/High/Low/Close)."""
    by_lower = {str(c).lower(): c for c in ohlc.columns}
    for name in names:
        col = by_lower.get(name.lower())
        if col is not None:
            return col
    raise KeyError(
        f"OHLC sin columnas requeridas {names}; columnas detectadas: {list(ohlc.columns)}"
    )


def _minute_float_core(ot, ct, op: float, is_buy: bool,
                       ohlc: pd.DataFrame) -> tuple | None:
    """Núcleo M1 compartido: (index, adv, fav) del tramo [ot, ct], o None.

    ``adv`` = envolvente adversa (convención de valle de
    :func:`precompute_strategy` sin agregar por día): ``signo·(worst − op)``
    con ``worst`` = min(low, prev_close) para largos / max(high, prev_close)
    para cortos, prev_close[0] = ``op``. ``fav`` = espejo simétrico favorable:
    ``signo·(best − op)`` con ``best`` = max(high, prev_close) para largos /
    min(low, prev_close) para cortos. None si el tramo no tiene cobertura OHLC
    (misma regla que el valle: ``ct <= ot`` o tramo vacío).
    """
    if pd.isna(ot) or pd.isna(ct) or ct <= ot:
        return None
    tramo = ohlc.loc[ot:ct]
    if tramo.empty:
        return None
    high_col = _ohlc_col(ohlc, "high", "h")
    low_col = _ohlc_col(ohlc, "low", "l")
    close_col = _ohlc_col(ohlc, "close", "c")
    signo = 1.0 if is_buy else -1.0
    prev_close = tramo[close_col].shift(1).to_numpy().copy()
    prev_close[0] = op
    if is_buy:
        precio_worst = np.minimum(tramo[low_col].to_numpy(), prev_close)
        precio_best = np.maximum(tramo[high_col].to_numpy(), prev_close)
    else:
        precio_worst = np.maximum(tramo[high_col].to_numpy(), prev_close)
        precio_best = np.minimum(tramo[low_col].to_numpy(), prev_close)
    adv = signo * (precio_worst - op)
    fav = signo * (precio_best - op)
    return tramo.index, adv, fav


def minute_valley_series(ot, ct, op: float, is_buy: bool,
                         ohlc: pd.DataFrame) -> pd.Series | None:
    """Serie M1 del flotante adverso de un trade, en unidades de precio.

    La convención de valle de :func:`precompute_strategy` SIN agregar por día:
    ``precio_worst`` = min(low, prev_close) para largos / max(high, prev_close)
    para cortos sobre las barras M1 del tramo [ot, ct] (ambos inclusive), con
    prev_close[0] = precio de apertura; ``vp = signo·(worst − op)`` (≤0 en lo
    adverso). Devuelve None si el tramo no tiene cobertura OHLC (trade sin
    valle). Es la materia prima del flotante conjunto intradía (``float_mode
    "joint"`` de ``src.risk.propfirm``, veredicto M2-1). Es la proyección
    adversa de :func:`minute_float_series` (mismo núcleo, bit-idéntica).
    """
    core = _minute_float_core(ot, ct, op, is_buy, ohlc)
    if core is None:
        return None
    idx, adv, _fav = core
    return pd.Series(adv, index=idx)


def minute_float_series(ot, ct, op: float, is_buy: bool,
                        ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Par (adversa, favorable) de envolventes M1 de un trade, en precio.

    Adversa = exactamente :func:`minute_valley_series` (mismo núcleo
    :func:`_minute_float_core`). Favorable = espejo simétrico: para largos
    ``max(high, prev_close) − open``; para cortos el espejo de ambas
    (``open − min(low, prev_close)``). ``prev_close[0] = op``. Devuelve None
    si el tramo no tiene cobertura OHLC (misma regla que el valle). Es la
    materia prima del trailing intradía (plan Apex Fase 2: ``upper_t =
    realized + Σ fav``, ``lower_t = realized + Σ adv``).
    """
    core = _minute_float_core(ot, ct, op, is_buy, ohlc)
    if core is None:
        return None
    idx, adv, fav = core
    return pd.Series(adv, index=idx), pd.Series(fav, index=idx)


def precompute_strategy(tl, ohlc: pd.DataFrame, pv: float, key: str = "",
                        day_labeler=None) -> StrategyMC:
    """valle_precio[i] = signo·(precio_worst−open) por día (f-indep, ≤0 adverse).

    Acepta columnas OHLC en formato corto (``o,h,l,c``) o largo
    (``Open,High,Low,Close`` / minúsculas). El valle diario es el mínimo de la
    serie M1 de :func:`minute_valley_series` agregada por día de calendario.

    ``day_labeler``: callable ts→date opcional (eje de sesiones ET, ver
    :mod:`src.risk.session_axis`) que sustituye al día calendario en el groupby
    de envolventes y en ``close_d``. None = día calendario (bit-idéntico).
    """
    df = tl.df
    open_t = pd.to_datetime(df["open_time"]).to_numpy()
    close_t = pd.to_datetime(df["close_time"]).to_numpy()
    is_buy = df["type"].astype(str).str.lower().str.startswith("b").to_numpy()
    valle: dict[int, dict] = {}
    for i in range(tl.n_trades):
        s = minute_valley_series(open_t[i], close_t[i], tl.open_price[i],
                                 bool(is_buy[i]), ohlc)
        if s is None:
            continue
        if day_labeler is None:
            valle[i] = dict(s.groupby(s.index.date).min())
        else:
            valle[i] = dict(s.groupby(day_labeler(s.index)).min())
    if day_labeler is None:
        close_d = pd.to_datetime(df["close_time"]).dt.date.to_numpy()
    else:
        close_d = np.asarray(day_labeler(pd.to_datetime(df["close_time"])))
    return StrategyMC(
        key=key, open_t=open_t, close_t=close_t, close_d=close_d,
        valle=valle, pnl_per_lot=tl.pnl_per_lot(), is_buy=is_buy, pv=pv,
        lots=np.empty((0, tl.n_trades)), n_trades=tl.n_trades,
    )


def attach_minute_valleys(strategy: StrategyMC, ohlc: pd.DataFrame,
                          open_prices: np.ndarray,
                          with_peaks: bool = False) -> StrategyMC:
    """Rellena ``strategy.minute_valle`` con la serie M1 de valle por trade.

    ``{trade_idx: (ts, vals, ot, ct)}``: timestamps y valores de
    :func:`minute_valley_series` más los extremos [ot, ct] (el flotante de una
    posición viva persiste entre su última barra y el cierre — la máscara del
    joint-float lo necesita). Trades sin cobertura OHLC: sin entrada (aportan 0
    al flotante, paridad BAJO-3). Por construcción, el mínimo diario de cada
    serie reproduce exactamente ``strategy.valle[i]``.

    ``with_peaks=True``: en la MISMA pasada M1 rellena además
    ``strategy.minute_fav``, dict paralelo (mismas claves y formato) con la
    envolvente favorable de :func:`minute_float_series` — requerido por la
    firma trailing (plan Apex Fase 2; sin picos el constructor trailing hace
    raise ruidoso). Con ``False`` (default) el comportamiento es bit-idéntico
    al histórico y ``minute_fav`` queda None.
    """
    mv: dict[int, tuple] = {}
    mf: dict[int, tuple] = {}
    for i in range(strategy.n_trades):
        ot, ct = strategy.open_t[i], strategy.close_t[i]
        op = float(open_prices[i])
        buy = bool(strategy.is_buy[i])
        if with_peaks:
            par = minute_float_series(ot, ct, op, buy, ohlc)
            if par is None:
                continue
            s_adv, s_fav = par
            mf[i] = (s_fav.index.values, s_fav.to_numpy(),
                     pd.Timestamp(ot), pd.Timestamp(ct))
        else:
            s_adv = minute_valley_series(ot, ct, op, buy, ohlc)
            if s_adv is None:
                continue
        mv[i] = (s_adv.index.values, s_adv.to_numpy(),
                 pd.Timestamp(ot), pd.Timestamp(ct))
    strategy.minute_valle = mv
    if with_peaks:
        strategy.minute_fav = mf
    return strategy


def attach_lots(strategy: StrategyMC, lots: np.ndarray) -> StrategyMC:
    """Asigna los sizes por f (shape (n_f, n_trades)) a la estrategia."""
    strategy.lots = lots
    return strategy


# ---------------------------------------------------------------------------
# Bloques flat-flat
# ---------------------------------------------------------------------------

def detect_flatflat_blocks(open_close_by_sym: dict) -> list[tuple]:
    """Bloques busy [(start, end)] donde alguna estrategia tiene posición.

    Cortes en instantes flat-flat (active_count==0). close(-1) se procesa antes que
    open(+1) en mismo timestamp (no romper close-then-open). El último busy sin cerrar
    (posición abierta al final del backtest) se descarta.
    """
    events = []
    for ot, ct in open_close_by_sym.values():
        for t in ot:
            if not pd.isna(t):
                events.append((pd.Timestamp(t), 1))
        for t in ct:
            if not pd.isna(t):
                events.append((pd.Timestamp(t), -1))
    events.sort(key=lambda e: (e[0], e[1]))
    blocks, active, busy_start = [], 0, None
    for t, delta in events:
        was_flat = (active == 0)
        active += delta
        if was_flat and active > 0:
            busy_start = t
        elif active == 0 and busy_start is not None:
            blocks.append((busy_start, t))
            busy_start = None
    return blocks


def build_block_records(strategies: dict, blocks: list[tuple],
                        oos_start: pd.Timestamp | None = None) -> list[BlockRecord]:
    """Filtra bloques enteramente en OOS (start ≥ oos_start) y adjunta info por trade."""
    syms = list(strategies.keys())  # orden estable → sym_id
    out: list[BlockRecord] = []
    for bstart, bend in blocks:
        if oos_start is not None and pd.Timestamp(bstart) < oos_start:
            continue
        sym_id, tloc, cd, vp, ppl = [], [], [], [], []
        for si, sym in enumerate(syms):
            sd = strategies[sym]
            ot, ct = sd.open_t, sd.close_t
            for i in range(len(ot)):
                if pd.isna(ot[i]) or pd.isna(ct[i]):
                    continue
                if ot[i] >= bstart and ct[i] <= bend and i in sd.valle:
                    sym_id.append(si)
                    tloc.append(i)
                    cd.append(sd.close_d[i])
                    vp.append(sd.valle[i])
                    ppl.append(sd.pnl_per_lot[i])
        if not sym_id:
            continue
        out.append(BlockRecord(
            idx=len(out), start=bstart, end=bend,
            duration_days=(pd.Timestamp(bend) - pd.Timestamp(bstart)).total_seconds() / 86400.0,
            sym_id=np.array(sym_id, dtype=int),
            trade_local_id=np.array(tloc, dtype=int),
            close_d=np.array(cd),
            valle_precio=vp, pnl_per_lot=np.array(ppl, dtype=float),
            n_unique_trades=len(sym_id),
        ))
    for j, b in enumerate(out):
        b.idx = j
    return out


def count_crossing(strategies: dict, blocks: list[tuple]) -> int:
    """Nº de trades no enteramente contenidos en un bloque (guardián V2; debe ser 0)."""
    starts = np.array([pd.Timestamp(b[0]).value for b in blocks])
    ends = np.array([pd.Timestamp(b[1]).value for b in blocks])
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    n_cross = 0
    for sd in strategies.values():
        for ot, ct in zip(sd.open_t, sd.close_t):
            if pd.isna(ot) or pd.isna(ct):
                continue
            ov, cv = pd.Timestamp(ot).value, pd.Timestamp(ct).value
            i = bisect.bisect_right(starts, ov) - 1
            if i < 0 or ends[i] < cv:
                n_cross += 1
    return n_cross


# ---------------------------------------------------------------------------
# Composición + evaluación del merged barajado
# ---------------------------------------------------------------------------

def _block_day_eval(b: BlockRecord, lots_by_f_sym, pv_by_sym, f_idx, saldo_start):
    """Por bloque: día → merged(d) = saldo_at_d + day_float(d); y total_pnl del bloque."""
    k = len(b.sym_id)
    pnl_trade = np.array([
        lots_by_f_sym[b.sym_id[j]][f_idx, b.trade_local_id[j]] * b.pnl_per_lot[j]
        for j in range(k)
    ])
    cd = b.close_d
    order = np.argsort(cd, kind="mergesort")
    cd_s, pt_s = cd[order], pnl_trade[order]
    uniq, inv = np.unique(cd_s, return_inverse=True)
    pnl_per_d = np.zeros(len(uniq))
    np.add.at(pnl_per_d, inv, pt_s)
    cs = np.cumsum(pnl_per_d)
    day_float = {}
    for j in range(k):
        lots = lots_by_f_sym[b.sym_id[j]][f_idx, b.trade_local_id[j]]
        pv = pv_by_sym[b.sym_id[j]]
        for d, vp in b.valle_precio[j].items():
            day_float[d] = day_float.get(d, 0.0) + vp * lots * pv
    all_d = sorted(set(uniq.tolist()) | set(day_float.keys()))
    uniq_list = uniq.tolist()
    merged = []
    for d in all_d:
        # saldo cerrado ANTES del día d: el flotante worst de d ocurre antes del close,
        # así que el PnL de los closes del propio d NO entra en el saldo de d (coherente
        # con mini_sim_merged / f_sweep, validado vs dailyEquity.bin).
        idx = bisect.bisect_left(uniq_list, d)
        prefix = saldo_start + (float(cs[idx - 1]) if idx > 0 else 0.0)
        merged.append(prefix + day_float.get(d, 0.0))
    total_pnl = float(cs[-1]) if len(cs) else 0.0
    return np.array(all_d), np.array(merged), total_pnl


def evaluate_path(block_order, blocks, lots_by_f_sym, pv_by_sym, f_idx, *,
                  pass_line: float, ruin_line: float, baseline: float,
                  horizon_days: float | None = None) -> PathOutcome:
    """Recorre los bloques en ``block_order``; primer cruce de pass/ruin decide."""
    saldo = 0.0
    days_seen = 0.0
    n_unique = 0
    peak = 0.0
    max_dd = 0.0
    outcome = "neither"
    last_merged = 0.0
    for bi in block_order:
        b = blocks[bi]
        n_unique += b.n_unique_trades
        days_seen += b.duration_days
        days, merged, total_pnl = _block_day_eval(b, lots_by_f_sym, pv_by_sym, f_idx, saldo)
        if len(merged):
            cumpeak = np.maximum.accumulate(np.maximum(merged, peak))
            max_dd = min(max_dd, float(np.min(merged - cumpeak)))
            peak = float(cumpeak[-1])
            last_merged = float(merged[-1])
            for m in merged:  # days en orden temporal
                if m <= ruin_line:
                    return PathOutcome("ruin", days_seen, n_unique, max_dd / baseline, float(m))
                if m >= pass_line:
                    return PathOutcome("pass", days_seen, n_unique, max_dd / baseline, float(m))
                if horizon_days is not None and days_seen > horizon_days:
                    return PathOutcome("timeout", days_seen, n_unique, max_dd / baseline, float(m))
        saldo += total_pnl
    return PathOutcome(outcome, days_seen, n_unique, max_dd / baseline, last_merged)


def merged_series_identity(blocks, lots_by_f_sym, pv_by_sym, f_idx) -> pd.Series:
    """Serie merged en orden original de bloques (identity path; oráculo V1)."""
    saldo = 0.0
    series = {}
    for b in blocks:
        days, merged, total_pnl = _block_day_eval(b, lots_by_f_sym, pv_by_sym, f_idx, saldo)
        for d, m in zip(days, merged):
            series[d] = m
        saldo += total_pnl
    return pd.Series(series)


def block_total_pnls(blocks, lots_by_f_sym, pv_by_sym, f_idx) -> np.ndarray:
    """PnL cerrado total de cada bloque a ``f_idx`` (unidad base del barajado).

    Es la serie cuya autocorrelacion (y la de sus cuadrados = clustering de
    rachas) quiere preservar el block bootstrap. El ACF es invariante a escala,
    asi que la fidelidad del barajado no depende del ``f_idx`` elegido.
    """
    return np.array([
        _block_day_eval(b, lots_by_f_sym, pv_by_sym, f_idx, 0.0)[2] for b in blocks
    ])


# ---------------------------------------------------------------------------
# Calibración del block_size — fidelidad ACF del barajado (criterio Alan)
# ---------------------------------------------------------------------------

@dataclass
class BlockSizeFidelity:
    """Fidelidad del barajado a un block_size dado (ACF del PnL por bloque).

    ``pearson``/``spearman`` comparan el ACF histórico (lags 1..nlags) con la
    mediana de los paths barajados, para el PnL y para sus cuadrados
    (clustering de rachas = el criterio relevante para ruina, como Alan).
    """
    block_size: float
    pearson: float
    spearman: float
    pearson_sq: float
    spearman_sq: float


@dataclass
class BlockSizeCalibration:
    """Resultado de calibrar block_size por fidelidad ACF (ver :func:`calibrate_block_size`)."""
    acf_hist: np.ndarray          # ACF (lags 1..nlags) del PnL por bloque histórico
    acf_sq_hist: np.ndarray       # idem de los cuadrados
    noise_floor: float            # 2/√n — suelo de significancia por lag
    sig_lags_sq: list             # lags ACF² que superan el suelo
    has_real_clustering: bool     # lags cortos (1-3) o >=3 lags sobre el suelo
    per_bs: list                  # list[BlockSizeFidelity]
    recommended: float            # bs mínimo con fidelidad ACF² Pearson > 0.8 (o el del campo `reason`)
    reason: str


def calibrate_block_size(
    pnl_hist: np.ndarray,
    bs_grid=(1.0, 2.0, 3.0, 5.0, 10.0),
    *,
    n_paths: int = 200,
    nlags: int = 20,
    seed: int = 42,
    threshold: float = 0.8,
) -> BlockSizeCalibration:
    """Calibra el block_size del bootstrap con fidelidad ACF (criterio Alan).

    La unidad del barajado es el episodio flat-flat; su serie de PnL
    (``pnl_hist``, orden cronológico) tiene una autocorrelación que el
    bootstrap preserva solo parcialmente según ``block_size`` (bs=1 =
    permutación pura → ACF ~0 en los paths). Esta función mide, por bs, la
    correlación Pearson (canon Alan, umbral 0.8) y Spearman entre el ACF
    histórico y la mediana de los paths barajados — para el PnL y sus
    cuadrados.

    Regla de decisión:
      - Si el ACF² histórico no tiene clustering real (ningún lag corto 1-3 ni
        >=3 lags totales sobre el suelo de ruido 2/√n — un lag suelto marginal
        entre 20 es lo que el azar produce ~64% de las veces), NO hay rachas
        inter-episodio que preservar: cualquier bs es defendible y se
        recomienda el mínimo de la grilla (el umbral no aplica: la fidelidad se
        mediría sobre ruido).
      - Si hay clustering real: bs mínimo con Pearson ACF² > ``threshold``; si
        ninguno lo alcanza, el mayor de la grilla con aviso.
    """
    from scipy import stats as scipy_stats
    from statsmodels.tsa.stattools import acf as _acf

    pnl_hist = np.asarray(pnl_hist, dtype=float)
    n = len(pnl_hist)
    if n < nlags + 5:
        raise ValueError(f"Serie de bloques demasiado corta ({n}) para ACF({nlags}).")

    def acf_vec(x):
        return _acf(x, nlags=nlags, fft=True)[1:]

    acf_hist = acf_vec(pnl_hist)
    acf_sq_hist = acf_vec(pnl_hist ** 2)
    noise_floor = 2.0 / np.sqrt(n)
    sig_lags_sq = [i + 1 for i, v in enumerate(acf_sq_hist) if abs(v) > noise_floor]
    has_clustering = any(l <= 3 for l in sig_lags_sq) or len(sig_lags_sq) >= 3

    rng = np.random.default_rng(seed)
    per_bs: list[BlockSizeFidelity] = []
    for bs in bs_grid:
        idx = stationary_block_indices(n, n_paths, float(bs), rng)
        paths = pnl_hist[idx]
        sim = np.array([acf_vec(p) for p in paths])
        sim_sq = np.array([acf_vec(p ** 2) for p in paths])
        med, med_sq = np.median(sim, axis=0), np.median(sim_sq, axis=0)
        per_bs.append(BlockSizeFidelity(
            block_size=float(bs),
            pearson=float(np.corrcoef(acf_hist, med)[0, 1]),
            spearman=float(scipy_stats.spearmanr(acf_hist, med).statistic),
            pearson_sq=float(np.corrcoef(acf_sq_hist, med_sq)[0, 1]),
            spearman_sq=float(scipy_stats.spearmanr(acf_sq_hist, med_sq).statistic),
        ))

    bs_values = [f.block_size for f in per_bs]
    if not has_clustering:
        recommended, reason = bs_values[0], (
            "ACF² histórico en el suelo de ruido: no hay rachas inter-episodio "
            "que preservar; cualquier block_size es defendible."
        )
    else:
        ok = [f.block_size for f in per_bs if f.pearson_sq > threshold]
        if ok:
            recommended, reason = min(ok), (
                f"Clustering real (lags ACF² {sig_lags_sq}): mínimo bs con "
                f"fidelidad ACF² Pearson>{threshold}."
            )
        else:
            recommended, reason = bs_values[-1], (
                f"Clustering real pero ningún bs alcanza Pearson>{threshold}: "
                "usa el mayor de la grilla (subestima rachas con bs menores)."
            )
    return BlockSizeCalibration(
        acf_hist=acf_hist, acf_sq_hist=acf_sq_hist, noise_floor=noise_floor,
        sig_lags_sq=sig_lags_sq, has_real_clustering=has_clustering,
        per_bs=per_bs, recommended=recommended, reason=reason,
    )


# ---------------------------------------------------------------------------
# Sweep + agregación
# ---------------------------------------------------------------------------

def aggregate_outcomes(outcomes, f, method, block_size, n_paths, horizon_days) -> RiskReport:
    outs = [o.outcome for o in outcomes]
    n = len(outs)
    n_pass = sum(1 for o in outs if o == "pass")
    n_ruin = sum(1 for o in outs if o == "ruin")
    n_time = sum(1 for o in outs if o == "timeout")
    days_pass = np.array([o.days for o in outcomes if o.outcome == "pass"], dtype=float)
    trades_pass = np.array([o.trades for o in outcomes if o.outcome == "pass"], dtype=float)
    dds = np.array([o.max_dd for o in outcomes], dtype=float)
    terms = np.array([o.terminal for o in outcomes], dtype=float)
    pct = lambda a, q: float(np.nanpercentile(a, q)) if len(a) else float("nan")
    return RiskReport(
        f=f, method=method, block_size=block_size, n_paths=n_paths,
        n_trades_horizon=int(horizon_days is not None),
        p_ruin=n_ruin / n, p_pass=n_pass / n,
        p_neither=(n - n_pass - n_ruin - n_time) / n, p_blown=0.0,
        p_ruin_ci=wilson_ci(n_ruin, n), p_pass_ci=wilson_ci(n_pass, n),
        median_trades_to_target=pct(trades_pass, 50), p90_trades_to_target=pct(trades_pass, 90),
        median_days_to_target=pct(days_pass, 50), p90_days_to_target=pct(days_pass, 90),
        median_max_dd=pct(dds, 50), p10_max_dd=pct(dds, 10),
        terminal_p05=pct(terms, 5), terminal_p50=pct(terms, 50), terminal_p95=pct(terms, 95),
    )


def sweep_f_portfolio(blocks, lots_by_f_sym, pv_by_sym, f_grid, *,
                      n_paths: int, block_size: float, seed: int,
                      pass_line: float, ruin_line: float, baseline: float,
                      horizon_days: float | None = None,
                      progress=None) -> tuple[list[RiskReport], np.ndarray]:
    """Baraja bloques (stationary_block_indices, compartido para toda la grilla f) y evalúa.

    ``progress`` opcional: callable(fi, f, rep) para reportar progreso (UI).
    """
    n_b = len(blocks)
    rng = np.random.default_rng(seed)
    idx_matrix = stationary_block_indices(n_b, n_paths, block_size, rng)
    reports: list[RiskReport] = []
    for fi, f in enumerate(f_grid):
        outs = [evaluate_path(idx_matrix[p], blocks, lots_by_f_sym, pv_by_sym, fi,
                              pass_line=pass_line, ruin_line=ruin_line, baseline=baseline,
                              horizon_days=horizon_days)
                for p in range(n_paths)]
        rep = aggregate_outcomes(outs, float(f), "block", block_size, n_paths, horizon_days)
        reports.append(rep)
        if progress is not None:
            progress(fi, float(f), rep)
    return reports, idx_matrix


# ---------------------------------------------------------------------------
# API de alto nivel
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMCResult:
    reports: list[RiskReport]
    idx_matrix: np.ndarray
    blocks: list[BlockRecord]
    n_crossing: int
    n_blocks_full: int


def run_portfolio_mc(strategies: dict, f_grid: np.ndarray, *,
                     n_paths: int, block_size: float = 1.0,
                     oos_start: pd.Timestamp | None = None, seed: int = 42,
                     pass_line: float = 350.0, ruin_line: float = -350.0,
                     baseline: float = 5000.0, horizon_days: float | None = None,
                     progress=None) -> PortfolioMCResult:
    """End-to-end: ``strategies`` (dict key→StrategyMC, orden = sym_id) → reports por f.

    Barreras por defecto = ±$350 sobre baseline $5k (AXI Seed). ``pass_line``/``ruin_line``
    son PnL absoluto desde 0 (merged flotante), ``baseline`` sólo escala el max_dd reportado.
    """
    lots_by_f_sym = {si: s.lots for si, s in enumerate(strategies.values())}
    pv_by_sym = {si: s.pv for si, s in enumerate(strategies.values())}
    oc = {k: (s.open_t, s.close_t) for k, s in strategies.items()}
    blocks_all = detect_flatflat_blocks(oc)
    n_cross = count_crossing(strategies, blocks_all)
    blocks = build_block_records(strategies, blocks_all, oos_start=oos_start)
    reports, idx_matrix = sweep_f_portfolio(
        blocks, lots_by_f_sym, pv_by_sym, f_grid,
        n_paths=n_paths, block_size=block_size, seed=seed,
        pass_line=pass_line, ruin_line=ruin_line, baseline=baseline,
        horizon_days=horizon_days, progress=progress,
    )
    return PortfolioMCResult(
        reports=reports, idx_matrix=idx_matrix, blocks=blocks,
        n_crossing=n_cross, n_blocks_full=len(blocks_all),
    )
