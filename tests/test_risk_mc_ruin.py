"""Tests del MC (resampling) + barrera doble de primer paso (ruin)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.mc import acf_block_size, iid_indices, stationary_block_indices, trade_R_pairs
from src.risk.ruin import analyze_paths
from src.risk.stages import StageBarriers

BARR = StageBarriers(
    phase_name="seed",
    display_name="Seed",
    baseline=10000.0,
    target_line=10700.0,  # +7%
    ruin_line=9300.0,     # -7% (estatico)
    profit_target_pct=7.0,
    max_loss_pct=-7.0,
    min_unique_trades=None,
    min_days=None,
    leverage=100,
)


def _report(y, a, f, idx=None):
    T = len(y)
    idx = np.arange(T)[None, :] if idx is None else idx  # path = secuencia original
    return analyze_paths(idx, np.asarray(y, float), np.asarray(a, float), f=f, barriers=BARR)


def test_path_que_cae_a_ruin_realizada():
    rep = _report(y=[-3, -3], a=[-3, -3], f=0.05)
    assert rep.p_ruin == 1.0
    assert rep.p_pass == 0.0


def test_path_que_alcanza_target_pasa():
    rep = _report(y=[3, 3], a=[0, 0], f=0.05)
    assert rep.p_pass == 1.0
    assert rep.p_ruin == 0.0


def test_path_verde_pero_mae_cruza_ruin_es_ruin_mtm():
    # Cierra verde (+1R -> +5% = 10500, sin target) pero intra-trade baja a -10% < -7%.
    rep = _report(y=[1], a=[-2], f=0.05)
    assert rep.p_ruin == 1.0  # el valle MTM toca ruin aunque cierre verde


def test_sin_mae_no_falsa_ruin_mtm():
    # a=0 -> MTM colapsa al equity cerrado; un cierre verde no es ruin.
    rep = _report(y=[1, 1], a=[0, 0], f=0.05)
    assert rep.p_ruin == 0.0


def test_block_bootstrap_preserva_media_marginal():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(5, 2000, block_size=2.0, rng=rng)
    means = y[idx].mean(axis=1)
    assert abs(means.mean() - y.mean()) < 0.05


def test_iid_es_permutacion():
    y = np.array([1.0, 2.0, 3.0])
    idx = iid_indices(3, 5, np.random.default_rng(1))
    for p in range(5):
        assert sorted(idx[p].tolist()) == [0, 1, 2]


def test_acf_block_size_minimo_1():
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # alterna, autocorr negativa
    bs = acf_block_size(y)
    assert bs >= 1.0


def test_sweep_reports_monotono_ruin_con_f(monkeypatch):
    """A mayor f, P(ruin) no decrece (monotonia debil) en un path adverso."""
    from src.risk.ruin import sweep_reports

    y = np.array([-1.0, -1.0, -1.0, -1.0])
    a = np.array([-1.0, -1.0, -1.0, -1.0])
    idx = np.tile(np.arange(4), (50, 1))  # 50 copias del mismo path
    reps = sweep_reports(
        idx, y, a, np.array([0.01, 0.03, 0.05, 0.08]),
        barriers=BARR, method="iid", block_size=1.0,
    )
    pruins = [r.p_ruin for r in reps]
    # No estrictamente monotono por el discretizado, pero 0.08 >= 0.01.
    assert pruins[-1] >= pruins[0]


def test_realized_rolling_mdd_smoke():
    """Smoke: rolling_max_drawdown (vbt) corre y devuelve valores <=0."""
    vbt = pytest.importorskip("vectorbtpro")
    from src.risk.ruin import realized_rolling_mdd

    idx = pd.date_range("2019-01-01", periods=400, freq="D")
    eq = pd.Series(np.linspace(10000, 11000, 400) + np.sin(np.arange(400)) * 300, index=idx)
    out = realized_rolling_mdd(eq, window_days=60)
    assert (out.dropna() <= 0).all()


# --- Regresiones de la revision adversarial --------------------------------

def _barr(min_days=None, min_trades=None, target_pct=7.0, loss_pct=-7.0):
    from src.risk.stages import StageBarriers
    return StageBarriers("seed", "Seed", 10000.0, 10000 * (1 + target_pct / 100),
                          10000 * (1 + loss_pct / 100), target_pct, loss_pct,
                          min_trades, min_days, 100)


def test_min_days_sintetico_no_corrompe_con_resampling():
    """min_days debe evaluarse con duracion sintetica (no timestamps shufleados)."""
    from src.risk.ruin import analyze_paths
    # Path que alcanza target en el trade 0 (+3R a f=0.05 = +15%).
    y = np.array([3.0, 3.0]); a = np.array([0.0, 0.0])
    idx = np.array([[0, 1]])
    # min_days=30, cadencia 100 dias/trade -> days=(0+1)*100=100 >= 30 -> PASS.
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr(min_days=30),
                        mean_inter_trade_days=100.0)
    assert rep.p_pass == 1.0  # antes: timestamps shufleados daban dias negativos -> neither


def test_min_days_sintetico_sigue_gateando():
    """Si la cadencia es baja, no se alcanza min_days -> neither (no se relaja)."""
    from src.risk.ruin import analyze_paths
    y = np.array([3.0, 3.0]); a = np.array([0.0, 0.0])
    idx = np.array([[0, 1]])
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr(min_days=30),
                        mean_inter_trade_days=10.0)  # days=10 < 30
    assert rep.p_pass == 0.0
    assert rep.p_neither == 1.0


def test_max_dd_incluye_valle_mtm():
    """Path verde-pero-MAE-ruin: max_dd debe reflejar el valle MTM, no 0."""
    from src.risk.ruin import analyze_paths
    y = np.array([1.0]); a = np.array([-2.0])  # cierra +5% (10500), valle -10% (9000)
    idx = np.array([[0]])
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr())
    assert rep.p_ruin == 1.0
    assert rep.median_max_dd <= -0.099  # antes era 0.0 (decoplado del MTM)


def test_max_dd_cuenta_drop_desde_baseline():
    """Una perdida en el trade 0 debe contar como DD desde baseline."""
    from src.risk.ruin import analyze_paths
    y = np.array([-3.0, 5.0, 2.5]); a = np.zeros(3)  # -6% luego recupera
    idx = np.array([[0, 1, 2]])
    rep = analyze_paths(idx, y, a, f=0.02, barriers=_barr())
    assert rep.median_max_dd <= -0.059  # antes 0.0 (running_max empezaba post-trade-0)


def test_terminal_se_detiene_en_la_barrera():
    """Al pasar, terminal = equity en el pase, no el horizonte completo."""
    from src.risk.ruin import analyze_paths
    y = np.array([3.0, -10.0]); a = np.zeros(2)  # pasa en trade0, trade1 crashea
    idx = np.array([[0, 1]])
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr())
    assert rep.p_pass == 1.0
    assert abs(rep.terminal_p50 - 11500.0) < 1.0  # no 5750 (post-pase)


def test_horizonte_descarta_ruin_tardia():
    """Ruin que ocurre tras el horizonte (trade 4, horizon 3) -> neither."""
    from src.risk.ruin import analyze_paths
    # E: 10000 -> 9950 -> 9900 -> 9850 -> 9800 -> 8820 (ruin en index 4)
    y = np.array([-0.1, -0.1, -0.1, -0.1, -2.0]); a = np.zeros(5)
    idx = np.array([[0, 1, 2, 3, 4]])
    rep_sin = analyze_paths(idx, y, a, f=0.05, barriers=_barr())
    rep_con = analyze_paths(idx, y, a, f=0.05, barriers=_barr(), horizon_trades=3)
    assert rep_sin.p_ruin == 1.0      # sin horizonte: ruin cuenta
    assert rep_con.p_ruin == 0.0      # ruin en trade 4 > horizon 3 -> neither
    assert rep_con.p_neither == 1.0


def test_horizonte_cuenta_ruin_temprana():
    """Ruin dentro del horizonte (trade 4, horizon 10) -> ruin."""
    from src.risk.ruin import analyze_paths
    y = np.array([-0.1, -0.1, -0.1, -0.1, -2.0]); a = np.zeros(5)
    idx = np.array([[0, 1, 2, 3, 4]])
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr(), horizon_trades=10)
    assert rep.p_ruin == 1.0


def test_barrier_override():
    """barriers_from_phase acepta overrides de ruin/target/min_days."""
    from src.risk.stages import barriers_from_phase
    b = barriers_from_phase("seed", 500.0, ruin_pct=-5.0, target_pct=10.0, min_days=45)
    assert b.max_loss_pct == -5.0
    assert b.profit_target_pct == 10.0
    assert b.min_days == 45
    # baseline = allocation_at(500) = 500*10 = 5000; lines escalan a los overrides
    assert b.ruin_line == pytest.approx(5000 * 0.95)
    assert b.target_line == pytest.approx(5000 * 1.10)


def test_wilson_ci_fronteras():
    from src.risk.ruin import wilson_ci
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 == 0.0 and 0 < hi0 < 0.05          # k=0 -> lo=0
    lon, hin = wilson_ci(100, 100)
    assert hin == 1.0 and 0.95 < lon < 1.0         # k=n -> hi=1
    lo5, hi5 = wilson_ci(5, 10)
    assert lo5 < 0.5 < hi5                          # contiene el puntual
    # más n -> IC más estrecho
    lo_big, hi_big = wilson_ci(500, 1000)
    assert (hi_big - lo_big) < (hi5 - lo5)


def test_report_trae_ci():
    """analyze_paths rellena los IC de ruin/pass."""
    from src.risk.ruin import analyze_paths
    y = np.array([3.0, -3.0]); a = np.zeros(2)
    idx = np.tile(np.arange(2), (50, 1))
    rep = analyze_paths(idx, y, a, f=0.05, barriers=_barr())
    assert 0.0 <= rep.p_ruin_ci[0] <= rep.p_ruin <= rep.p_ruin_ci[1] <= 1.0
    assert rep.p_pass_ci[0] <= rep.p_pass <= rep.p_pass_ci[1]


def _rep(f, p_pass, p_ruin, *, median_days=float("nan"),
         pass_ci=None, ruin_ci=None):
    """RiskReport minimal para tests de recommend_f."""
    from src.risk.ruin import RiskReport
    return RiskReport(
        f=f, method="", block_size=0.0, n_paths=1000, n_trades_horizon=100,
        p_ruin=p_ruin, p_pass=p_pass, p_neither=0.0, p_blown=0.0,
        p_ruin_ci=ruin_ci or (0.0, 1.0), p_pass_ci=pass_ci or (0.0, 1.0),
        median_trades_to_target=0, p90_trades_to_target=0,
        median_days_to_target=median_days, p90_days_to_target=0,
        median_max_dd=0.0, p10_max_dd=0.0,
        terminal_p05=0.0, terminal_p50=0.0, terminal_p95=0.0,
    )


def test_recommend_factible_maximiza_pass():
    from src.risk.ruin import recommend_f
    # factibles (ruin<=0.15): f=0.01 (pass 0.70) y f=0.02 (pass 0.55); f=0.03 no (ruin 0.20)
    reps = [_rep(0.01, 0.70, 0.04), _rep(0.02, 0.55, 0.08), _rep(0.03, 0.60, 0.20)]
    rec = recommend_f(reps, tolerance=0.15)
    assert rec.feasible
    assert rec.f == pytest.approx(0.01)       # max pass entre factibles


def test_recommend_infactible_menor_ruin():
    from src.risk.ruin import recommend_f
    reps = [_rep(0.01, 0.30, 0.25), _rep(0.02, 0.50, 0.35), _rep(0.03, 0.40, 0.30)]
    rec = recommend_f(reps, tolerance=0.15)   # ninguno <= 0.15
    assert not rec.feasible
    assert rec.f == pytest.approx(0.01)       # menor ruin (0.25)


def test_recommend_empate_f_bajo():
    from src.risk.ruin import recommend_f
    reps = [_rep(0.02, 0.60, 0.05), _rep(0.01, 0.60, 0.05)]  # mismo pass, f distintos
    rec = recommend_f(reps, tolerance=0.15)
    assert rec.f == pytest.approx(0.01)       # empate -> f mas bajo


def test_recommend_flags_slow_y_wide_ci():
    from src.risk.ruin import recommend_f
    reps = [_rep(0.01, 0.70, 0.05, median_days=400, ruin_ci=(0.02, 0.10))]
    rec = recommend_f(reps, tolerance=0.15)
    assert rec.slow                                   # 400 > 365
    assert rec.wide_ci                                # semi-ancho 0.04 > max(0.025, 0.02)=0.025


def test_recommend_no_falso_wide_en_ruin_cero():
    """ruin≈0 con CI preciso NO debe marcar wide_ci (era falso positivo)."""
    from src.risk.ruin import recommend_f
    reps = [_rep(0.01, 0.90, 0.0, ruin_ci=(0.0, 0.004), pass_ci=(0.85, 0.94))]
    rec = recommend_f(reps, tolerance=0.15)
    assert not rec.wide_ci                            # semi-ancho 0.002 < floor 0.02
