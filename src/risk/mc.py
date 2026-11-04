"""Generacion de paths de Monte Carlo (resampling de la secuencia de trades).

Dos motores, ambos sobre INDICES de trade (f-independientes), para luego
componerse a cualquier ``f`` en :mod:`ruin`:

* **IID** — permutacion de la secuencia (destruye autocorrelacion). Es el metodo
  que usa QuantAnalyzer -> sirve de cross-check apples-to-apples.
* **Block-bootstrap estacionario** (Politis-Romano) en **numpy** — preserva
  autocorrelacion y el timing path-dependiente de los drawdowns. Es el numero
  HEADLINE de P(ruin). Se implementa en numpy (no vbt.Splitter) porque el
  block-bootstrap a nivel trade no se beneficia de la API de CV de Splitter y
  deepseek cuestiono su block_size por defecto; aqui lo controlamos entero y
  pre-registramos ``block_size`` por ACF.

Por que resamplear INDICES: cada trade lleva su R-multiplo ``y_n`` y su
adverso ``a_n`` (flotante en el peor punto, en R). Resamplear indices preserva
el emparejamiento (y, a) y deja el ``f`` fuera -> un mismo set de paths sirve
para toda la grilla de ``f``.

NOTA: la composicion en :mod:`ruin` usa la forma continua ``E*(1+f*y)`` (sin
re-floor). El re-floor discreto del SWEEP (:mod:`mm`) es fiel a SQX; en el MC
su error (<=0.24%) no cambia materialmente la FORMA de la distribucion, que es
lo que el MC estima. Se documenta este cavehat en la UI.
"""
from __future__ import annotations

import numpy as np

from src.risk.trades import TradeList


def trade_R_pairs(
    tl: TradeList, f_ref: float | None = None, sl_in_money: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y, a): R-multiplo realizado y adverso por trade, f-independientes.

    Modo rapido (``f_ref``): ``y = r_n/f_ref``, ``a = mae/(f_ref*equity_before)``.
    Modo exacto (``sl_in_money``): ``y = pnl_per_lot/slInMoney``,
    ``a = mae_per_lot/slInMoney`` (slInMoney del ATR real). Ambos adimensionales.
    """
    if sl_in_money is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(sl_in_money > 0, tl.pnl_per_lot() / sl_in_money, 0.0)
            mae_per_lot = np.where(tl.size > 0, tl.mae / np.maximum(tl.size, 1e-12), 0.0)
            a = np.where(sl_in_money > 0, mae_per_lot / sl_in_money, 0.0)
        return np.nan_to_num(y, nan=0.0), np.nan_to_num(a, nan=0.0)
    if f_ref is None or f_ref <= 0:
        raise ValueError("Hace falta f_ref (rapido) o sl_in_money (exacto)")
    y = tl.r_multiples(f_ref)  # (n,)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(tl.equity_before > 0, tl.mae / (f_ref * tl.equity_before), 0.0)
    return y, np.nan_to_num(a, nan=0.0)


def iid_indices(n_trades: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """N permutaciones de arange(n_trades). Shape (n_paths, n_trades)."""
    idx = np.empty((n_paths, n_trades), dtype=np.int64)
    base = np.arange(n_trades, dtype=np.int64)
    for p in range(n_paths):
        rng.shuffle(base)
        idx[p] = base
    return idx


def stationary_block_indices(
    n_trades: int,
    n_paths: int,
    block_size: float,
    rng: np.random.Generator,
    length: int | None = None,
) -> np.ndarray:
    """Block-bootstrap estacionario (Politis-Romano 1994).

    Longitudes de bloque ~ Geometric(p=1/block_size); inicio de bloque uniforme.
    Concatena hasta cubrir ``length`` (por defecto ``n_trades``) y trunca.
    Preserva autocorrelacion intra-bloque. Shape (n_paths, length).

    ``length`` opcional (plan propfirm-carteras, fallback N4): permite paths mas
    largos que la ventana observada (horizonte extendido m·n_blocks) sin cambiar
    el universo de indices [0, n_trades). Por defecto (None) es identico a antes.
    """
    if block_size < 1:
        block_size = 1.0
    n_out = n_trades if length is None else int(length)
    p = 1.0 / block_size  # prob de cortar el bloque
    idx = np.empty((n_paths, n_out), dtype=np.int64)
    for path in range(n_paths):
        out = []
        while len(out) < n_out:
            start = int(rng.integers(0, n_trades))
            # longitud geometrica
            block_len = int(rng.geometric(p))
            for j in range(block_len):
                if len(out) >= n_out:
                    break
                out.append((start + j) % n_trades)  # wrap (estacionario)
        idx[path] = out[:n_out]
    return idx


def extend_stationary_rows(
    idx: np.ndarray,
    n_trades: int,
    block_size: float,
    extra: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Prolonga cada fila de un bootstrap estacionario en ``extra`` posiciones, desde su último índice.

    En :func:`stationary_block_indices`, la longitud de cada racha es Geometric(p = 1/bs), con soporte >= 1, y es
    geométrica pura (sin recuerdo): tras cada índice emitido, la racha sigue con probabilidad 1 - p, ya sea el primero
    o uno posterior de la racha. Por tanto, la cadena es de Markov sobre el índice actual::

        siguiente = (actual + 1) % n_trades  con prob. 1 - p
                    uniforme en [0, n_trades)  con prob. p

    Así, ``[fila, prolongación]`` se distribuye como una fila de longitud ``n + extra``, y la fila original (la F1) no
    cambia ni un bit. Con ``block_size <= 1`` (p = 1) es IID, como en :func:`stationary_block_indices`.

    Vectorizado sobre las filas. ORDEN DE CONSUMO del ``rng`` (contrato, para reproducibilidad): en cada uno de los
    ``extra`` pasos, primero ``u = rng.random(n_paths)`` y después ``starts = rng.integers(0, n_trades, n_paths)``
    (las dos llamadas en cada paso, aunque ``u`` decida cuáles filas usan ``starts``).

    Devuelve ``(n_paths, extra)`` con el mismo dtype que ``idx``.
    """
    idx = np.asarray(idx)
    if idx.ndim != 2:
        raise ValueError("idx debe ser 2D (n_paths, n_blocks)")
    n_paths = idx.shape[0]
    if block_size < 1:
        block_size = 1.0
    p = 1.0 / block_size
    out = np.empty((n_paths, int(extra)), dtype=idx.dtype)
    cur = idx[:, -1].astype(np.int64, copy=True)
    for k in range(int(extra)):
        u = rng.random(n_paths)
        starts = rng.integers(0, n_trades, n_paths)
        cur = np.where(u < 1.0 - p, (cur + 1) % n_trades, starts)
        out[:, k] = cur
    return out


def acf_block_size(y: np.ndarray) -> float:
    """Block size pre-registrable via tiempo de autocorrelacion integrado.

    tau_int = 1 + 2 * sum_{k>=1} rho_k, cortando cuando rho_k cae bajo el
    umbral de Bartlett 2/sqrt(T). Block size recomendado ~= tau_int (redondeado,
    min 1). Si la serie es corta o sin autocorrelacion signicativa, -> 1 (=> IID).
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    T = len(y)
    if T < 4 or y.var() == 0:
        return 1.0
    var = y.var()
    bartlett = 2.0 / np.sqrt(T)
    acf_sum = 0.0
    for k in range(1, min(T - 1, T // 2)):
        rho = np.dot(y[:-k], y[k:]) / ((T - k) * var)
        if abs(rho) < bartlett:
            break
        acf_sum += rho
    tau = 1.0 + 2.0 * acf_sum
    return float(max(1.0, round(tau)))


def default_block_size_fallback(n_trades: int) -> float:
    """Fallback clasico (Polititis-Romano): 3.15 * T^(1/3). Solo si el ACF da 1."""
    return 3.15 * n_trades ** (1.0 / 3.0)


def make_paths(
    tl: TradeList,
    n_paths: int,
    *,
    method: str = "block",
    block_size: float | None = None,
    seed: int = 0,
    f_ref: float | None = None,
    sl_in_money: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Genera paths de indices + (y, a).

    ``y, a`` vienen de ``f_ref`` (rapido) o ``sl_in_money`` (exacto).

    Returns:
        (idx_matrix (n_paths, n_trades), y (n_trades,), a (n_trades,),
         method, block_size_used)
    """
    n = tl.n_trades
    y, a = trade_R_pairs(tl, f_ref=f_ref, sl_in_money=sl_in_money)
    rng = np.random.default_rng(seed)
    if method == "iid":
        idx = iid_indices(n, n_paths, rng)
        bs = 1.0
    elif method == "block":
        bs = block_size if block_size is not None else acf_block_size(y)
        idx = stationary_block_indices(n, n_paths, bs, rng)
    else:
        raise ValueError(f"method desconocido: {method!r} (usa 'iid' o 'block')")
    return idx, y, a, method, float(bs)
