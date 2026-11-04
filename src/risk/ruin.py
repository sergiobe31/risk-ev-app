"""Barrera doble de primer paso + distribuciones de resultado.

Para cada path (resampleado en :mod:`mc`) y cada ``f``:

    E_after_k  = baseline * cumprod_k(1 + f * y_p)      # equity post-trade
    mtm_k      = E_before_k * (1 + f * a_p_k)           # valle intra-trade (a <= 0)

AXI (verificado, DevRev oficial): el max loss es drawdown ESTATICO desde la
baseline de Allocation inicial, evaluado en MTM tiempo real (intra-trade). El
target se evalua sobre equity cerrada. Por eso:

    ruin  si mtm toca ``ruin_line``  ANTES (o en el mismo trade) que el target.
    pass  si equity cerrada toca ``target_line`` con >= min_trades y >= min_days.

La barrera de ruin es MTM (mas estricta): un path que cierra verde pero cuyo
valle intra-trade cruza -7% es RUIN segun AXI. Sin MAE, a=0 y el MTM colapsa al
equity cerrado.

NOTA sobre min_days bajo resampling: el tiempo calendario NO es un estadistico
valido por-path cuando el MC permuta/scramblea la secuencia de trades (los
close_time de dos trades shufleados dan diferencias sin sentido, a menudo
negativas). Por eso ``min_days`` se evalua con una duracion SINTETICA
``(n_trades_a_target) * mean_inter_trade_days``, que es funcion del conteo de
trades (valido bajo resampling) y preserva el intento de la regla
("suficientes trades a tu cadencia media como para cubrir min_days").

Los agregados (terminal, max_dd, p_blown) se calculan sobre el prefijo valido
[0..stop_idx] donde stop_idx es la primera barrera (en AXI se deja de operar al
pasar o arruinarse), no sobre todo el horizonte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.risk.stages import StageBarriers

# Z para IC binomial (Wilson). 1.96 = 95%.
_Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Intervalo de confianza de Wilson (95% default) para una proporcion k/n.

    Mejor que el normal-asintotico en los extremos (k=0 -> lo=0; k=n -> hi=1).
    Devuelve (limite_inf, limite_sup) en [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    import math
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    # Fronteras exactas (evita epsilon float: k=0 -> lo=0, k=n -> hi=1).
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)





@dataclass
class RiskReport:
    """Resultado del MC para un ``f`` dado."""

    f: float
    method: str
    block_size: float
    n_paths: int
    n_trades_horizon: int
    p_ruin: float
    p_pass: float
    p_neither: float
    p_blown: float  # equity <= 0 antes de la primera barrera
    p_ruin_ci: tuple[float, float]  # Wilson 95% (lo, hi)
    p_pass_ci: tuple[float, float]
    median_trades_to_target: float  # entre los que pasan
    p90_trades_to_target: float
    median_days_to_target: float  # entre los que pasan (trades × cadencia)
    p90_days_to_target: float
    median_max_dd: float  # fraccion (negativo)
    p10_max_dd: float  # cola adversa (10% peor)
    terminal_p05: float
    terminal_p50: float
    terminal_p95: float

    def to_row(self) -> dict:
        return {
            "f": self.f,
            "method": self.method,
            "block_size": self.block_size,
            "n_paths": self.n_paths,
            "P(ruin)": self.p_ruin,
            "P(ruin) IC95": f"[{self.p_ruin_ci[0]:.1%},{self.p_ruin_ci[1]:.1%}]",
            "P(pass)": self.p_pass,
            "P(pass) IC95": f"[{self.p_pass_ci[0]:.1%},{self.p_pass_ci[1]:.1%}]",
            "P(neither)": self.p_neither,
            "P(blowup)": self.p_blown,
            "trades_a_target_p50": self.median_trades_to_target,
            "trades_a_target_p90": self.p90_trades_to_target,
            "dias_a_target_p50": self.median_days_to_target,
            "dias_a_target_p90": self.p90_days_to_target,
            "maxDD_p50": self.median_max_dd,
            "maxDD_p10": self.p10_max_dd,
            "terminal_p05": self.terminal_p05,
            "terminal_p50": self.terminal_p50,
            "terminal_p95": self.terminal_p95,
        }


@dataclass
class Recommendation:
    """`f` recomendado segun el veredicto del debate (max P(pass) s.a. ruin<=tol)."""

    f: float
    feasible: bool             # False si NINGUN f cumple ruin<=tol (entonces = menor ruin)
    p_pass: float
    p_ruin: float
    p_pass_ci: tuple[float, float]
    p_ruin_ci: tuple[float, float]
    median_days: float
    p90_days: float
    slow: bool                 # mediana dias-a-target > slow_cap (asesor, no restriccion)
    wide_ci: bool              # semi-ancho del IC > 50% del puntual (ruido de muestreo)
    reason: str                # explicacion humana


def recommend_f(
    reports: list["RiskReport"],
    tolerance: float,
    slow_cap_days: float = 365.0,
) -> Recommendation | None:
    """Elige el `f` recomendado.

    Objetivo (veredicto debate): **maximizar P(pass) incondicional** sujeto a
    **P(ruin) <= tolerance** (estimador puntual). Fallbacks:
      - ningun f factible -> el de menor P(ruin) (no se rechaza en silencio).
      - empate en P(pass) -> el f mas bajo.
      - la restriccion nunca ata -> max global de P(pass).
    Flags asores: `slow` (mediana dias > cap), `wide_ci` (IC ancho -> mas N).
    """
    if not reports:
        return None
    feasible = [r for r in reports if r.p_ruin <= tolerance]
    if feasible:
        best = max(feasible, key=lambda r: (r.p_pass, -r.f))  # max pass, empate -> f bajo
        reason = f"máx P(pass) con P(ruin) ≤ {tolerance:.0%}"
    else:
        best = min(reports, key=lambda r: (r.p_ruin, r.f))     # menor ruin, empate -> f bajo
        reason = f"ningún f cumple P(ruin) ≤ {tolerance:.0%}; se muestra el de menor ruin"

    def _wide(ci, p):
        # "ancho" si el semi-ancho del IC supera 50% del puntual O 2pp absolutos.
        # El floor de 2pp evita falso positivo en p≈0 (un ruin 0 preciso no es "ancho").
        return (ci[1] - ci[0]) / 2.0 > max(0.5 * abs(p), 0.02)

    mdays = best.median_days_to_target
    slow = bool(np.isfinite(mdays) and mdays > slow_cap_days)
    return Recommendation(
        f=best.f,
        feasible=bool(feasible),
        p_pass=best.p_pass,
        p_ruin=best.p_ruin,
        p_pass_ci=best.p_pass_ci,
        p_ruin_ci=best.p_ruin_ci,
        median_days=mdays,
        p90_days=best.p90_days_to_target,
        slow=slow,
        wide_ci=_wide(best.p_ruin_ci, best.p_ruin) or _wide(best.p_pass_ci, best.p_pass),
        reason=reason,
    )


def _first_true_mask(mask: np.ndarray) -> np.ndarray:
    """Indice (por fila) del primer True, o -1 si ninguno. mask: (N, T) bool."""
    N, T = mask.shape
    out = np.full(N, -1, dtype=np.int64)
    any_hit = mask.any(axis=1)
    out[any_hit] = np.argmax(mask, axis=1)[any_hit]
    return out


def mean_inter_trade_days(close_time: pd.Series) -> float | None:
    """MEDIA de dias entre trades consecutivos (cadencia realizada).

    Sirve para proyectar min_days / horizonte sobre paths resampleados sin usar
    timestamps shufleados (que no son un estadistico valido por-path). Se usa la
    MEDIA (no la mediana): el tiempo calendario esperado a N trades = N*mean
    (linealidad); la mediana subestimaria la cola (sesgo que el debate flaggeo).
    """
    ct = pd.to_datetime(close_time).dropna().sort_values()
    if len(ct) < 2:
        return None
    diffs = ct.diff().dropna().dt.total_seconds() / 86400.0
    mu = float(diffs.mean())
    return mu if mu > 0 else None


@dataclass
class PathOutcomes:
    """Outcomes por path (para construir distribuciones/CDF por f)."""

    outcome: np.ndarray            # (N,) 'ruin'|'pass'|'neither'
    trades_to_target: np.ndarray   # (N,) nan si no pasa
    days_to_target: np.ndarray     # (N,) nan si no pasa (trades*mitd)
    max_dd: np.ndarray             # (N,) fraccion (negativo)
    terminal: np.ndarray           # (N,) equity al stop


def compute_path_outcomes(
    idx_matrix: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    *,
    f: float,
    barriers: StageBarriers,
    mean_inter_trade_days: float | None = None,
    horizon_trades: float | None = None,
) -> PathOutcomes:
    """Computa los outcomes por path (sin agregar) para un ``f``.

    Lógica primer-paso: ruin si MTM toca ``ruin_line`` antes (o en el mismo
    trade) que el target; target sobre equity cerrada con gates min_trades/
    min_days (sintético). ``horizon_trades`` acota a los primeros N trades.
    """
    N, T = idx_matrix.shape
    baseline = barriers.baseline
    ruin_line = barriers.ruin_line
    target_line = barriers.target_line
    min_trades = barriers.min_unique_trades
    min_days = barriers.min_days
    mitd = mean_inter_trade_days

    y_p = y[idx_matrix]
    a_p = a[idx_matrix]
    factors = 1.0 + f * y_p
    equity_after = baseline * np.cumprod(factors, axis=1)
    equity_before = np.empty_like(equity_after)
    equity_before[:, 0] = baseline
    equity_before[:, 1:] = equity_after[:, :-1]
    mtm = equity_before * (1.0 + f * a_p)
    mtm = np.minimum(mtm, equity_after)

    ruin_mask = mtm <= ruin_line
    target_mask = equity_after >= target_line
    first_ruin = _first_true_mask(ruin_mask)
    first_target = _first_true_mask(target_mask)
    days_raw = (first_target + 1) * mitd if (mitd and mitd > 0) else np.full(N, np.nan)

    outcome = np.array(["neither"] * N, dtype=object)
    stop_idx = np.full(N, T - 1, dtype=np.int64)
    t_to_target = np.full(N, np.nan)
    hcap = (float(horizon_trades) - 1) if horizon_trades is not None else None
    for p in range(N):
        ri = int(first_ruin[p])
        ti = int(first_target[p])
        if hcap is not None:
            if ri > hcap:
                ri = -1
            if ti > hcap:
                ti = -1
        qual = -1
        if ti != -1:
            ok_trades = (min_trades is None) or (ti + 1 >= min_trades)
            ok_days = (min_days is None) or (days_raw[p] >= min_days)
            if ok_trades and ok_days:
                qual = ti
        if ri != -1 and (qual == -1 or ri <= qual):
            outcome[p] = "ruin"
            stop_idx[p] = ri
        elif qual != -1:
            outcome[p] = "pass"
            stop_idx[p] = qual
            t_to_target[p] = qual + 1
        elif hcap is not None:
            stop_idx[p] = max(0, min(int(hcap), T - 1))

    arange_T = np.arange(T)[None, :]
    valid = arange_T <= stop_idx[:, None]
    terminal = np.take_along_axis(equity_after, stop_idx[:, None], axis=1)[:, 0]
    running_max = np.maximum(np.maximum.accumulate(equity_after, axis=1), baseline)
    running_max_before = np.maximum.accumulate(equity_before, axis=1)
    dd_close = (equity_after - running_max) / np.maximum(running_max, 1e-12)
    dd_mtm = (mtm - running_max_before) / np.maximum(running_max_before, 1e-12)
    dd = np.where(valid, np.minimum(dd_close, dd_mtm), np.inf)
    max_dd = np.where(np.isfinite(dd.min(axis=1)), dd.min(axis=1), 0.0)

    days_to_target = t_to_target * mitd if (mitd and mitd > 0) else np.full(N, np.nan)
    return PathOutcomes(
        outcome=outcome, trades_to_target=t_to_target, days_to_target=days_to_target,
        max_dd=max_dd, terminal=terminal,
    )


def analyze_paths(
    idx_matrix: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    *,
    f: float,
    barriers: StageBarriers,
    mean_inter_trade_days: float | None = None,
    horizon_trades: float | None = None,
) -> RiskReport:
    """Agrega :func:`compute_path_outcomes` en un :class:`RiskReport` por ``f``."""
    N, T = idx_matrix.shape
    po = compute_path_outcomes(
        idx_matrix, y, a, f=f, barriers=barriers,
        mean_inter_trade_days=mean_inter_trade_days, horizon_trades=horizon_trades,
    )
    passing = po.outcome == "pass"
    n_ruin = int((po.outcome == "ruin").sum())
    n_pass = int(passing.sum())
    tt_days = po.days_to_target[passing & ~np.isnan(po.days_to_target)]
    tt_trades = po.trades_to_target[passing & ~np.isnan(po.trades_to_target)]

    def _pct(arr, q):
        return float(np.percentile(arr, q * 100)) if len(arr) else float("nan")

    return RiskReport(
        f=float(f),
        method="",
        block_size=0.0,
        n_paths=int(N),
        n_trades_horizon=int(T),
        p_ruin=float((po.outcome == "ruin").mean()),
        p_pass=float(passing.mean()),
        p_neither=float((po.outcome == "neither").mean()),
        p_blown=float((po.terminal <= 0).mean()),
        p_ruin_ci=wilson_ci(n_ruin, N),
        p_pass_ci=wilson_ci(n_pass, N),
        median_trades_to_target=_pct(tt_trades, 0.50),
        p90_trades_to_target=_pct(tt_trades, 0.90),
        median_days_to_target=_pct(tt_days, 0.50),
        p90_days_to_target=_pct(tt_days, 0.90),
        median_max_dd=_pct(po.max_dd, 0.50),
        p10_max_dd=_pct(po.max_dd, 0.10),
        terminal_p05=_pct(po.terminal, 0.05),
        terminal_p50=_pct(po.terminal, 0.50),
        terminal_p95=_pct(po.terminal, 0.95),
    )


def realized_rolling_mdd(
    equity_series: pd.Series,
    window_days: int = 60,
) -> pd.Series:
    """Max drawdown movil sobre ventanas de ``window_days`` (path realizado).

    Usa ``vectorbtpro`` ReturnsAccessor.rolling_max_drawdown sobre la serie de
    equity diaria (indexada por fecha, remuestreada a ``D`` con forward-fill).
    Devuelve la serie de MDD por ventana movil; el caller reporta ``.min()``
    (peor ventana de N dias) y la traza.

    NOTA: es una metrica DESCRIPTIVA del backtest historico unico (n=1
    trayectoria). Las ventanas moviles solapadas estan correlacionadas -> NO
    dan un IC honesto. El IC del drawdown-vs-limite esta en :class:`RiskReport`
    (``p_ruin`` + ``p_ruin_ci`` de Wilson), que cuenta caminos cuyo valle MTM
    toca ``ruin_line`` antes del target.
    """
    import vectorbtpro as vbt  # import perezoso: solo cuando se pide el rolling

    eq = equity_series.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    daily = eq.resample("D").last().ffill()
    rets = daily.vbt.returns
    return rets.rolling_max_drawdown(window=window_days)


def sweep_reports(
    idx_matrix: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    f_grid: np.ndarray,
    *,
    barriers: StageBarriers,
    method: str,
    block_size: float,
    close_time: pd.Series | None = None,
    horizon_days: float | None = None,
) -> list[RiskReport]:
    """Calcula :class:`RiskReport` para cada ``f`` de la grilla (mismos paths).

    La cadencia ``mean_inter_trade_days`` se calcula UNA vez de los close_time
    realizados (no por-path) y se proyecta a min_days de forma valida.
    ``horizon_days`` (si se da) -> ``horizon_trades = horizon_days / mitd`` (ruin
    dentro de N dias).
    """
    mitd = mean_inter_trade_days(close_time) if close_time is not None else None
    horizon_trades = (horizon_days / mitd) if (horizon_days and mitd and mitd > 0) else None
    out = []
    for f in f_grid:
        rep = analyze_paths(
            idx_matrix, y, a, f=f, barriers=barriers, mean_inter_trade_days=mitd,
            horizon_trades=horizon_trades,
        )
        rep.method = method
        rep.block_size = block_size
        out.append(rep)
    return out
