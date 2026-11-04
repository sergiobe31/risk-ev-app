"""Paso 2 de validación: la mitad MC del motor, contra forma cerrada (no SQX, no QA).

El paso 1 (``test_risk_sizing_parity``) validó el sizing contra SQX. Aquí
validamos el **Monte Carlo** — el contador de caminos y el IC — con un oracle
matemático: el **gambler's ruin** clásico. QuantAnalyzer no soporta ATR risk
sized, así que el oracle del MC pasa a ser la matemática de forma cerrada.

Construcción EXACTA (sin aproximación de paso chico):
  - Cada trade mueve la equity por un factor ``R`` (subida) o ``1/R`` (bajada).
    En log, pasos simétricos ±log(R) → random walk simétrico.
  - Barreras a **media distancia** entre niveles ``R^k`` (``baseline·R^7.5`` /
    ``baseline·R^-7.5``). Así el random walk las toca al primer nivel ENTERO que
    las cruza (cum=±8), con un margen de ~½ paso (~0.5% para R=1.01) que deja al
    épsilon de float fuera de juego. Si las barreras cayeran EXACTAMENTE en
    ``R^k``, la diferencia entre ``cumprod`` (equity_after) y ``**`` (línea)
    decidiría ``>=``/``<=`` por un épsilon → asimetría artificial ruin/target.
    Eso NO ocurre en la app real (y continuos, sin chocar la barrera exacta).
  - MAE ``a_n = 0`` → ruin sólo en equity cerrada (gambler's ruin puro).

Oracle (pasos efectivos = 8 a cada lado, ``A=B=8``):
  - ``P(ruin) = B/(A+B)`` (simétrico p_up=0.5); forma cerrada para p sesgado.
  - Test A: ``compute_path_outcomes`` reproduce la fórmula (contador correcto,
    primera barrera que se toca, sin confundir ruin/pass).
  - Test B: ``wilson_ci`` es un IC honesto (cubre el verdadero p al ~95%).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.ruin import compute_path_outcomes, wilson_ci
from src.risk.stages import StageBarriers

# --- Construcción sintética ------------------------------------------------
BASELINE = 5000.0
R = 1.01            # factor por paso (1% de equity)
F = 0.02            # riesgo por trade (irrelevante para el mapping; sólo escala y)
_HALF = 7.5         # barreras a media distancia entre R^7 y R^8 (float-robusto)
A_STEPS = 8         # pasos efectivos a ruin (cum = -8)
B_STEPS = 8         # pasos efectivos a target (cum = +8)
N_TRADES = 400      # >> absorción (media ~A·B=64) → neither residual <1%

# y tal que 1 + F·y = R  (subida)  ó  1/R  (bajada).
Y_UP = (R - 1.0) / F
Y_DOWN = (1.0 / R - 1.0) / F
_Y = np.array([Y_UP, Y_DOWN])
_A = np.array([0.0, 0.0])          # sin MAE → ruin sólo en equity cerrada

_BARRIERS = StageBarriers(
    phase_name="SYNTHETIC", display_name="synthetic",
    baseline=BASELINE,
    target_line=BASELINE * R ** _HALF,
    ruin_line=BASELINE * R ** (-_HALF),
    profit_target_pct=100.0 * (R ** _HALF - 1.0),
    max_loss_pct=100.0 * (R ** (-_HALF) - 1.0),
    min_unique_trades=None, min_days=None, leverage=100,
)


def _gamble_ruin_p_ruin(a: int, b: int, p_up: float) -> float:
    """P(tocar -a antes que +b), random walk ±1, prob ``p_up`` subir (forma cerrada).

    Simétrico (p_up=0.5): ``b/(a+b)``. Sesgado: ``1 - (1-(q/p)^a)/(1-(q/p)^(a+b))``.
    """
    if abs(p_up - 0.5) < 1e-12:
        return b / (a + b)
    ratio = (1.0 - p_up) / p_up
    p_target = (1.0 - ratio ** a) / (1.0 - ratio ** (a + b))
    return 1.0 - p_target


def _bernoulli_idx(n_paths: int, p_up: float, rng: np.random.Generator) -> np.ndarray:
    """idx_matrix (n_paths, N_TRADES): idx 0 (Y_UP) con prob ``p_up``, 1 (Y_DOWN) si no.

    Cada path es una secuencia Bernoulli libre (no permutación de conteo fijo),
    que es lo que el gambler's ruin clásico asume.
    """
    return (rng.random((n_paths, N_TRADES)) >= p_up).astype(np.int64)


# ---------------------------------------------------------------------------
# Test A — el MC reproduce la fórmula del gambler's ruin
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p_up", [0.50, 0.55])
def test_mc_reproduce_gambler_ruin(p_up):
    """compute_path_outcomes cuenta bien las barreras (caso simétrico y sesgado)."""
    rng = np.random.default_rng(20260717)
    idx = _bernoulli_idx(6000, p_up, rng)
    po = compute_path_outcomes(idx, _Y, _A, f=F, barriers=_BARRIERS)
    p_ruin = float((po.outcome == "ruin").mean())
    neither = float((po.outcome == "neither").mean())
    p_true = _gamble_ruin_p_ruin(A_STEPS, B_STEPS, p_up)
    # paths largos → absorción casi segura; el residual neither no sesga el gate.
    assert neither < 0.03, (
        f"p_up={p_up}: {neither:.1%} de paths sin absorber en {N_TRADES} trades "
        f"(sube N_TRADES o reduce A/B)"
    )
    # MC con 6000 paths → σ≈0.006 → 3σ≈0.018; holgura a 0.025.
    assert abs(p_ruin - p_true) < 0.025, (
        f"p_up={p_up}: P(ruin) MC={p_ruin:.4f} vs fórmula {p_true:.4f} "
        f"(diff {abs(p_ruin - p_true):.4f})"
    )


# ---------------------------------------------------------------------------
# Test B — el IC de Wilson está calibrado al 95%
# ---------------------------------------------------------------------------
def test_wilson_ic_calibrado_al_95():
    """K corridas con n_paths chico: el IC95 contiene el verdadero p ~95% de las veces."""
    p_up = 0.50
    p_true = _gamble_ruin_p_ruin(A_STEPS, B_STEPS, p_up)   # = 0.5 exacto
    K, n_paths = 400, 400
    rng = np.random.default_rng(11)
    cubre = 0
    for _ in range(K):
        idx = _bernoulli_idx(n_paths, p_up, rng)
        po = compute_path_outcomes(idx, _Y, _A, f=F, barriers=_BARRIERS)
        k = int((po.outcome == "ruin").sum())
        lo, hi = wilson_ci(k, n_paths)
        if lo <= p_true <= hi:
            cubre += 1
    cobertura = cubre / K
    # IC95 nominal → cobertura observada ~0.95 (σ≈0.011 con K=400 → margen [0.91, 0.99]).
    assert 0.91 <= cobertura <= 0.99, (
        f"IC95 mal calibrado: cobertura={cobertura:.1%} (esperado ~95%); "
        f"p_true={p_true:.4f}"
    )
