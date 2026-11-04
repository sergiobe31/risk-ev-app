"""Tests de la F2 y la fondeada en CONTINUACION de la secuencia (plan ftmo-fases-secuencias, 2026-10-16).

Antes del cambio, F1, F2 y fondeada de cada path se evaluaban sobre la MISMA fila ``idx_matrix[p]``
desde la posicion 0: con las mismas barreras y target F2 < F1, P(F2|F1) = 1 por construccion. Ahora la
F2 y la fondeada son ventanas posteriores de una fila prolongada (``PhaseTape``).

Cubre (todo determinista y sintetico):
  I2  reconstruccion legacy: con ``_next_start`` -> 0 el motor reproduce una implementacion de
      referencia del comportamiento anterior (1D y camino 2D; perfil de 2 fases y de 1 fase)
  I3  identidad de la F2; I4 identidad de la fondeada
  I5  determinismo; I7  la misma ``idx_ext`` en toda la grilla de f
  I8  equivalencia Markov de ``extend_stationary_rows`` (bs en {1, 3, 10}, 20 000 filas, 4 SE)
  I9  ``end_pos`` correcto en los 3 evaluadores (pass / fail / neither / expired)
  estructural: P(F2|F1) < 1 con el codigo nuevo y == 1 exacto con ``_next_start`` -> 0
  AC5 errores explicitos (ndarray, sin idx_ext, sweep_funded posicional, gaps_fixed sin gaps_ext, 422)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.risk import propfirm
from src.risk.mc import extend_stationary_rows, stationary_block_indices
from src.risk.propfirm import (
    BlockDayArrays,
    ContractCap,
    FundedRules,
    PhaseTape,
    PropFirmConfig,
    PropFirmPhase,
    TrailingDrawdown,
    _draw_gap_tape,
    build_phase_tape,
    evaluate_funded_path,
    evaluate_phase_path,
    evaluate_phase_path_trailing,
    simulate_challenge,
    simulate_firm_paths,
    sweep_funded,
)
from src.risk.propfirm_futures import evaluate_phase_path_eod_trail

ACC = 50_000.0
PHASE1 = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)
PHASE2 = PropFirmPhase("Verification", 5.0, 5.0, 10.0, 4)
FUNDED = FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0, max_payout_cycles=3)
GAP_DAYS = np.array([0.0, 1.0, 3.0, 7.0])

N_TRADES = 8          # tipos de bloque
N_BLOCKS = 12         # longitud de la fila (F1 y ventana de cada fase)
N_PATHS = 300
BS = 2.5


def _cfg(two_phases: bool = True) -> PropFirmConfig:
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=345.0, profit_split_pct=80.0,
        refund_fee_on_first_payout=True,
        phases=(PHASE1, PHASE2) if two_phases else (PHASE1,), funded=FUNDED,
    )


def _rand_blocks(seed: int = 5, k: int = N_TRADES):
    """Bloques sinteticos con 1-3 dias cada uno, esperanza positiva y colas (hay pass, fail y neither)."""
    rng = np.random.default_rng(seed)
    blocks, durations = [], []
    for _ in range(k):
        nd = int(rng.integers(1, 4))
        days = np.cumsum(rng.integers(1, 4, nd)) + int(rng.integers(0, 5))
        steps = rng.normal(420.0, 1100.0, nd)
        rel_closed = np.cumsum(steps)
        rel_before = np.concatenate([[0.0], rel_closed[:-1]])
        rel_float = np.minimum(rel_before, rel_closed) - rng.uniform(0.0, 250.0, nd)
        blocks.append(BlockDayArrays(
            day_ord=days.astype(int), rel_before=rel_before, rel_closed=rel_closed,
            rel_float=rel_float, opens_day=np.ones(nd, dtype=int), total_pnl=float(rel_closed[-1])))
        durations.append(float(days[-1] - days[0] + 1))
    return blocks, np.array(durations)


BLOCKS, DURATIONS = _rand_blocks()


def _rows(n_paths: int = N_PATHS, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = stationary_block_indices(N_TRADES, n_paths, BS, rng, length=N_BLOCKS)
    ext = extend_stationary_rows(idx, N_TRADES, BS, 2 * N_BLOCKS, np.random.default_rng([0, 2]))
    g1, g_ext = _draw_gap_tape(rng, GAP_DAYS, n_paths, N_BLOCKS)
    return idx, ext, g1, g_ext


IDX, EXT, G1, GEXT = _rows()
TAPE = build_phase_tape(IDX, EXT, G1, GEXT)


def _legacy(monkeypatch):
    """Reconstruye el comportamiento anterior: la fase siguiente arranca en 0 (misma fila que la F1)."""
    monkeypatch.setattr(propfirm, "_next_start", lambda e: 0)


# ---------------------------------------------------------------------------
# Implementacion de REFERENCIA del comportamiento anterior (escrita aqui, sin tocar el motor)
# ---------------------------------------------------------------------------

def _ref_legacy(idx, gaps, blocks, durations, cfg):
    n = idx.shape[0]
    acc = cfg.account_size
    ph1 = cfg.phases[0]
    ph2 = cfg.phases[1] if len(cfg.phases) > 1 else None
    out = {k: np.zeros(n, dtype=bool) for k in ("pass1", "neither1", "expired1", "neither2", "expired2")}
    out["pass2"] = np.ones(n, dtype=bool)
    out["days_to_pass1"] = np.full(n, np.nan)
    out["days_eval1"] = np.full(n, np.nan)
    for p in range(n):
        gp = gaps[p] if gaps is not None else None
        r1 = evaluate_phase_path(idx[p], blocks, durations, gp, account_size=acc, phase=ph1)
        out["pass1"][p] = r1.outcome == "pass"
        out["neither1"][p] = r1.outcome == "neither"
        out["expired1"][p] = r1.outcome == "expired"
        out["days_eval1"][p] = r1.days
        if out["pass1"][p]:
            out["days_to_pass1"][p] = r1.days
        if ph2 is not None:
            r2 = evaluate_phase_path(idx[p], blocks, durations, gp, account_size=acc, phase=ph2)
            out["pass2"][p] = r2.outcome == "pass"
            out["neither2"][p] = out["pass1"][p] and r2.outcome == "neither"
            out["expired2"][p] = out["pass1"][p] and r2.outcome == "expired"
    both = out["pass1"] & out["pass2"]
    out["funded_payouts"] = np.zeros((n, cfg.funded.max_payout_cycles))
    out["days_first_payout"] = np.full(n, np.nan)
    out["alive_at_cap"] = np.zeros(n, dtype=bool)
    for p in np.flatnonzero(both):
        gp = gaps[p] if gaps is not None else None
        rf = evaluate_funded_path(idx[p], blocks, durations, gp, account_size=acc, phase=ph1,
                                  funded=cfg.funded, profit_split=cfg.profit_split_pct / 100.0)
        out["funded_payouts"][p] = rf.cycle_payouts
        out["days_first_payout"][p] = rf.days_to_first_payout
        out["alive_at_cap"][p] = rf.alive_at_cap
    return out


_COMPARE_CH = ("pass1", "pass2", "neither1", "neither2", "expired1", "expired2", "days_to_pass1",
               "days_eval1")
_COMPARE = _COMPARE_CH + ("funded_payouts", "days_first_payout", "alive_at_cap")


def _assert_matches_ref(obj, ref, keys=_COMPARE):
    for k in keys:
        got = getattr(obj, k)
        np.testing.assert_array_equal(np.asarray(got), ref[k], err_msg=k)  # bit a bit (nan == nan)


# ---------------------------------------------------------------------------
# I2: reconstruccion legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("two_phases", [True, False])
def test_i2_simulate_firm_paths_legacy_igual_a_referencia(monkeypatch, two_phases):
    _legacy(monkeypatch)
    cfg = _cfg(two_phases)
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, cfg, gap_days=GAP_DAYS, seed=9, idx_ext=EXT)
    # Los gaps internos de simulate_firm_paths son la llamada de siempre: default_rng(seed).choice(...).
    g_old = np.random.default_rng(9).choice(GAP_DAYS, size=IDX.shape, replace=True)
    ref = _ref_legacy(IDX, g_old, BLOCKS, DURATIONS, cfg)
    _assert_matches_ref(out, ref)
    assert ref["pass1"].any() and (~ref["pass1"]).any()      # el escenario no es degenerado


@pytest.mark.parametrize("two_phases", [True, False])
def test_i2_camino_2d_challenge_mas_sweep_legacy_igual_a_referencia(monkeypatch, two_phases):
    _legacy(monkeypatch)
    cfg = _cfg(two_phases)
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    fu = sweep_funded(tape=TAPE, funded_start=ch.funded_start, day_arrays_f_grid=[BLOCKS, BLOCKS],
                      passers_mask=ch.both, durations=DURATIONS, cfg=cfg, f_grid=[0.01, 0.02])
    ref = _ref_legacy(IDX, G1, BLOCKS, DURATIONS, cfg)
    _assert_matches_ref(ch, ref, _COMPARE_CH)
    for f in (0.01, 0.02):
        np.testing.assert_array_equal(fu[f].funded_payouts, ref["funded_payouts"])
        np.testing.assert_array_equal(fu[f].days_first_payout, ref["days_first_payout"])
        np.testing.assert_array_equal(fu[f].alive_at_cap, ref["alive_at_cap"])
    assert (ch.funded_start == 0).all()


def test_i2_sin_gaps_legacy_igual_a_referencia(monkeypatch):
    _legacy(monkeypatch)
    cfg = _cfg(True)
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, cfg, gap_days=None, seed=0, idx_ext=EXT)
    _assert_matches_ref(out, _ref_legacy(IDX, None, BLOCKS, DURATIONS, cfg))


# ---------------------------------------------------------------------------
# I1: la F1 no cambia (arrays por path), con y sin continuacion
# ---------------------------------------------------------------------------

def test_i1_f1_identica_con_y_sin_continuacion(monkeypatch):
    cfg = _cfg(True)
    new = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    ref = _ref_legacy(IDX, G1, BLOCKS, DURATIONS, cfg)
    for k in ("pass1", "neither1", "expired1", "days_to_pass1", "days_eval1"):
        np.testing.assert_array_equal(getattr(new, k), ref[k], err_msg=k)
    np.testing.assert_array_equal(TAPE.idx[:, :N_BLOCKS], IDX)
    np.testing.assert_array_equal(TAPE.gaps[:, :N_BLOCKS], G1)


# ---------------------------------------------------------------------------
# I3, I4: identidades de la F2 y de la fondeada
# ---------------------------------------------------------------------------

def test_i3_pass2_es_el_evaluador_sobre_la_ventana_siguiente():
    cfg = _cfg(True)
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    n_seen = 0
    for p in range(N_PATHS):
        o1, g1 = TAPE.window(p, 0)
        r1 = evaluate_phase_path(o1, BLOCKS, DURATIONS, g1, account_size=ACC, phase=PHASE1)
        s2 = propfirm._next_start(r1.end_pos)
        assert s2 == r1.end_pos + 1
        o2, g2 = TAPE.window(p, s2)
        r2 = evaluate_phase_path(o2, BLOCKS, DURATIONS, g2, account_size=ACC, phase=PHASE2)
        assert ch.pass2[p] == (r2.outcome == "pass")
        assert ch.neither2[p] == (ch.pass1[p] and r2.outcome == "neither")
        assert ch.funded_start[p] == s2 + r2.end_pos + 1
        n_seen += int(ch.pass1[p])
    assert n_seen > 10


def test_i3_cotas_de_la_cinta():
    cfg = _cfg(True)
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    assert ch.funded_start.min() >= 1
    assert ch.funded_start.max() <= 2 * N_BLOCKS
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, _cfg(False))
    assert ch1.funded_start.max() <= N_BLOCKS
    # toda ventana cabe en la cinta (3n)
    assert TAPE.windows(ch.funded_start).shape == (N_PATHS, N_BLOCKS)


@pytest.mark.parametrize("two_phases", [True, False])
def test_i4_fondeada_es_el_evaluador_sobre_la_ventana_de_arranque(two_phases):
    cfg = _cfg(two_phases)
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    fu = sweep_funded(tape=TAPE, funded_start=ch.funded_start, day_arrays_f_grid=[BLOCKS],
                      passers_mask=ch.both, durations=DURATIONS, cfg=cfg, f_grid=[0.0])[0.0]
    passers = np.flatnonzero(ch.both)
    assert len(passers) > 5
    for p in passers:
        o, g = TAPE.window(p, ch.funded_start[p])
        rf = evaluate_funded_path(o, BLOCKS, DURATIONS, g, account_size=ACC, phase=PHASE1,
                                  funded=FUNDED, profit_split=0.8)
        np.testing.assert_array_equal(fu.funded_payouts[p], rf.cycle_payouts)
        np.testing.assert_array_equal(fu.days_first_payout[p], rf.days_to_first_payout)
        assert fu.alive_at_cap[p] == rf.alive_at_cap
    no_pass = np.flatnonzero(~ch.both)
    assert (fu.funded_payouts[no_pass] == 0.0).all()
    # Hay payouts: la fondeada se evaluo de verdad.
    assert (fu.funded_payouts[passers] > 0).any()


def test_windows_equivale_a_window_por_path():
    cfg = _cfg(True)
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    w = TAPE.windows(ch.funded_start)
    for p in (0, 7, 100, N_PATHS - 1):
        np.testing.assert_array_equal(w[p], TAPE.window(p, ch.funded_start[p])[0])


# ---------------------------------------------------------------------------
# I5, I7: determinismo y cinta comun
# ---------------------------------------------------------------------------

def test_i5_determinismo():
    cfg = _cfg(True)
    a = simulate_firm_paths(IDX, BLOCKS, DURATIONS, cfg, gap_days=GAP_DAYS, seed=3, idx_ext=EXT)
    b = simulate_firm_paths(IDX, BLOCKS, DURATIONS, cfg, gap_days=GAP_DAYS, seed=3, idx_ext=EXT)
    for k in _COMPARE + ("funded_start",):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    e1 = extend_stationary_rows(IDX, N_TRADES, BS, 5, np.random.default_rng([0, 2]))
    e2 = extend_stationary_rows(IDX, N_TRADES, BS, 5, np.random.default_rng([0, 2]))
    np.testing.assert_array_equal(e1, e2)
    g_a = _draw_gap_tape(np.random.default_rng(4), GAP_DAYS, 10, 6)
    g_b = _draw_gap_tape(np.random.default_rng(4), GAP_DAYS, 10, 6)
    np.testing.assert_array_equal(g_a[0], g_b[0])
    np.testing.assert_array_equal(g_a[1], g_b[1])


def test_draw_gap_tape_f1_es_la_llamada_de_siempre():
    g = np.random.default_rng(4).choice(GAP_DAYS, size=(10, 6), replace=True)
    g1, g_ext = _draw_gap_tape(np.random.default_rng(4), GAP_DAYS, 10, 6)
    np.testing.assert_array_equal(g1, g)
    assert g_ext.shape == (10, 12)


def test_extend_no_toca_la_fila_original_y_dtype():
    idx32 = IDX.astype(np.int32)
    before = idx32.copy()
    ext = extend_stationary_rows(idx32, N_TRADES, BS, 9, np.random.default_rng(0))
    np.testing.assert_array_equal(idx32, before)
    assert ext.dtype == idx32.dtype and ext.shape == (N_PATHS, 9)
    assert ext.min() >= 0 and ext.max() < N_TRADES
    assert extend_stationary_rows(IDX, N_TRADES, BS, 4, np.random.default_rng(0)).dtype == IDX.dtype


def test_i7_misma_idx_ext_en_toda_la_grilla_1d(monkeypatch):
    """El endpoint 1D pasa LA MISMA idx_ext (la del ctx) a cada f de la peticion."""
    api = pytest.importorskip("poc_web.api.main")
    leg = "risk_app/inputs/trades/nasdaq_2h/Strategy 1.30.184.csv"
    if not Path(leg).exists():
        pytest.skip("datos gitignorados de risk_app/inputs no disponibles")
    pf = api.portfolio_load(api.PortfolioLoadRequest(legs=[leg], account_size=100000.0))
    assert pf["ok"], pf
    # E7 de bloque-politis-white (opción 1): el test no trata del bloque automático; se fija el bloque que daba la regla
    # vieja aquí (1,0; la nueva da 1,10 y añadiría la evaluación de sensibilidad con bloque 1 en f*, una 4.ª llamada).
    req = api.PropfirmSimRequest(
        portfolio_id=pf["portfolio_id"], firm_key="ftmo_100k", f_lo=0.005, f_hi=0.015, f_step=0.005,
        n_paths=100, n_challenges=3, n_sequences=100, capital0_eur=3000.0, fx_eur_usd=1.1, block_size=1.0)
    ctx = api._propfirm_setup(req)
    # idx_ext = prolongacion con default_rng([0, 2]); no consume el rng(0) del setup.
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(ctx["n_blocks"], req.n_paths, ctx["bs"], rng)
    np.testing.assert_array_equal(ctx["idx_matrix"], idx)
    assert ctx["rng"].random() == rng.random()
    exp_ext = extend_stationary_rows(idx, ctx["n_blocks"], ctx["bs"], 2 * ctx["n_blocks"],
                                     np.random.default_rng([0, 2]))
    np.testing.assert_array_equal(ctx["idx_ext"], exp_ext)

    seen = []
    orig = api.simulate_firm_paths

    def spy(*a, **kw):
        seen.append(kw["idx_ext"])
        return orig(*a, **kw)

    monkeypatch.setattr(api, "simulate_firm_paths", spy)
    api.simulate_propfirm(req)
    assert len(seen) == 3
    for e in seen:
        np.testing.assert_array_equal(e, exp_ext)


def test_par_2d_gaps_de_la_f1_son_los_de_siempre(monkeypatch):
    """En el par 2D la cinta de gaps arranca con la llamada rng_pair.choice de hoy."""
    api = pytest.importorskip("poc_web.api.main")
    leg = "risk_app/inputs/trades/nasdaq_2h/Strategy 1.30.184.csv"
    if not Path(leg).exists():
        pytest.skip("datos gitignorados de risk_app/inputs no disponibles")
    pf = api.portfolio_load(api.PortfolioLoadRequest(legs=[leg], account_size=100000.0))
    req = api.PropfirmSimRequest(
        portfolio_id=pf["portfolio_id"], firm_key="ftmo_100k", f_lo=0.005, f_hi=0.01, f_step=0.005,
        n_paths=100, n_challenges=3, n_sequences=100, capital0_eur=3000.0, fx_eur_usd=1.1)
    ctx = api._propfirm_setup(req)
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(ctx["n_blocks"], req.n_paths, ctx["bs"], rng)
    g_old = rng.choice(ctx["gaps_emp"], size=idx.shape, replace=True)

    tapes = []
    orig = api.simulate_challenge

    def spy(tape, *a, **kw):
        tapes.append(tape)
        return orig(tape, *a, **kw)

    monkeypatch.setattr(api, "simulate_challenge", spy)
    api.simulate_propfirm_pair(req)
    assert tapes and all(t is tapes[0] for t in tapes)      # una sola cinta para toda la grilla
    np.testing.assert_array_equal(tapes[0].idx[:, :ctx["n_blocks"]], idx)
    np.testing.assert_array_equal(tapes[0].gaps[:, :ctx["n_blocks"]], g_old)


# ---------------------------------------------------------------------------
# Estructural: el fallo original (P(F2|F1) = 1)
# ---------------------------------------------------------------------------

def test_estructural_p_f2_dado_f1_menor_que_1_con_continuacion_e_igual_a_1_en_legacy(monkeypatch):
    cfg = _cfg(True)         # mismas barreras y min_days, target F2 (5%) < target F1 (10%)
    new = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    assert new.pass1.sum() > 20
    p_f2_f1_new = new.pass2[new.pass1].mean()
    assert p_f2_f1_new < 1.0
    _legacy(monkeypatch)
    old = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    np.testing.assert_array_equal(old.pass1, new.pass1)              # la F1 no cambia
    assert old.pass2[old.pass1].mean() == 1.0                        # el fallo original, exacto
    assert (old.both == old.pass1).all()


# ---------------------------------------------------------------------------
# I8: equivalencia Markov de extend_stationary_rows
# ---------------------------------------------------------------------------

N_ROWS_I8 = 20_000
N_I8 = 40            # n_trades
L0 = 6               # longitud de la fila original
EXTRA = 40


def _frames(bs, seed=0):
    rng = np.random.default_rng(seed)
    idx = stationary_block_indices(N_I8, N_ROWS_I8, bs, rng, length=L0)
    ext = extend_stationary_rows(idx, N_I8, bs, EXTRA, rng)
    ours = np.concatenate([idx, ext], axis=1)
    ref = stationary_block_indices(N_I8, N_ROWS_I8, bs, np.random.default_rng(seed + 1), length=L0 + EXTRA)
    return ours, ref


def _runs(frame, lo):
    """Longitudes de rachas COMPLETAS (inicio tras una ruptura, fin antes de otra) en columnas >= lo."""
    cont = (frame[:, lo + 1:] == (frame[:, lo:-1] + 1) % N_I8)         # (rows, W-1): transicion = continuar
    lens = []
    for row in cont:
        cur = None
        for c in row:
            if c:
                if cur is not None:
                    cur += 1
            else:
                if cur is not None:
                    lens.append(cur)
                cur = 1
    return np.array(lens)


@pytest.mark.parametrize("bs", [1.0, 3.0, 10.0])
def test_i8_equivalencia_markov_con_stationary_block_indices(bs):
    ours, ref = _frames(bs)
    p = 1.0 / bs
    q = (1.0 - p) + p / N_I8           # P(siguiente = actual + 1 mod n)
    sl = slice(L0, L0 + EXTRA)         # tramo prolongado (en la de referencia, el mismo tramo)
    # 1) frecuencia de "siguiente = actual + 1": Bernoulli(q) iid por transicion
    for name, fr in (("ext", ours), ("ref", ref)):
        c = (fr[:, sl][:, 1:] == (fr[:, sl][:, :-1] + 1) % N_I8)
        m = c.size
        se = np.sqrt(q * (1 - q) / m)
        assert abs(c.mean() - q) < 4 * se, (name, bs, c.mean(), q)
    # 2) la transicion de EMPALME fila -> prolongacion (la que crea el motor) tambien
    c0 = ours[:, L0] == (ours[:, L0 - 1] + 1) % N_I8
    assert abs(c0.mean() - q) < 4 * np.sqrt(q * (1 - q) / len(c0)), (bs, c0.mean(), q)
    # 3) marginal uniforme por columna (la cadena arranca en estacionario)
    for col in (L0, L0 + 1, L0 + 7, L0 + EXTRA - 1):
        cnt = np.bincount(ours[:, col], minlength=N_I8)
        se = np.sqrt((1 / N_I8) * (1 - 1 / N_I8) / N_ROWS_I8)
        assert (np.abs(cnt / N_ROWS_I8 - 1 / N_I8) < 4 * se).all(), (bs, col)
    # 4) racha media geometrica: 1/(1-q) = bs * n / (n - 1)  (la media del bootstrap es bs)
    if bs > 1:
        c = (ours[:, sl][:, 1:] == (ours[:, sl][:, :-1] + 1) % N_I8)
        qhat = c.mean()
        se_q = np.sqrt(q * (1 - q) / c.size)
        mean_run = 1.0 / (1.0 - qhat)
        se_mean = se_q / (1.0 - qhat) ** 2           # metodo delta
        assert abs(mean_run - bs * N_I8 / (N_I8 - 1)) < 4 * se_mean, (bs, mean_run)
        assert abs(mean_run - bs) < 0.1 * bs


@pytest.mark.parametrize("bs", [3.0, 10.0])
def test_i8_distribucion_de_longitudes_de_racha_geometrica(bs):
    ours, ref = _frames(bs, seed=11)
    p = 1.0 / bs
    q = (1.0 - p) + p / N_I8
    a, b = _runs(ours, L0), _runs(ref, L0)
    assert len(a) > 5000 and len(b) > 5000
    # a) mismas frecuencias de longitud que la referencia (mismo censurado por ventana) para l = 1..5
    for l in range(1, 6):
        pa, pb = (a == l).mean(), (b == l).mean()
        se = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
        assert abs(pa - pb) < 4 * se, (bs, l, pa, pb)
    # b) y frente a la teorica. Con T = EXTRA - 1 transiciones iid Bernoulli(q) (continuar), una racha
    #    completa de longitud l es el patron 0 1^(l-1) 0: hay T - l posiciones de inicio, asi que
    #    P(L = l | completa) = (T - l) q^(l-1) / sum_m (T - m) q^(m-1)   (media -> 1/(1-q) = bs n/(n-1)).
    T = EXTRA - 1
    ms = np.arange(1, T)
    w = (T - ms) * q ** (ms - 1.0)
    pmf = w / w.sum()
    for name, runs in (("ext", a), ("ref", b)):
        for l in range(1, 6):
            pl = (runs == l).mean()
            theo = pmf[l - 1]
            assert abs(pl - theo) < 4 * np.sqrt(theo * (1 - theo) / len(runs)), (name, bs, l, pl, theo)


# ---------------------------------------------------------------------------
# I9: end_pos en los 3 evaluadores
# ---------------------------------------------------------------------------

def _bda(days, rel_closed, rel_low=None, rel_high=None, rel_exc=None, trailing=False):
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_before = np.concatenate([[0.0], rel_closed[:-1]])
    kw = {}
    if rel_low is not None:
        kw["rel_low"] = np.asarray(rel_low, dtype=float)
    if trailing:
        rh = np.asarray(rel_high, dtype=float)
        re_ = np.asarray(rel_exc, dtype=float)
        kw.update(rel_high=rh, rel_exc=re_, block_low=float(np.min(kw["rel_low"])),
                  block_high=float(rh.max()), block_exc=float(re_.max()))
    return BlockDayArrays(
        day_ord=days, rel_before=rel_before, rel_closed=rel_closed, rel_float=rel_before.copy(),
        opens_day=np.ones(len(days), dtype=int), total_pnl=float(rel_closed[-1]), **kw)


def test_i9_end_pos_evaluate_phase_path():
    ph = PropFirmPhase("C", 10.0, 5.0, 10.0, 1)
    neutral = _bda([1], [100.0])
    win = _bda([1, 2], [3000.0, 6000.0])         # supera el 10% en el 2o dia del bloque
    loss = _bda([1], [-2800.0])                   # daily loss 5% -> fail
    blocks = [neutral, win, loss]
    dur = np.ones(3)
    kw = dict(account_size=ACC, phase=ph)
    r = evaluate_phase_path(np.array([0, 0, 1, 2]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("pass", 2)                    # posicion del BLOQUE, no del dia
    r = evaluate_phase_path(np.array([0, 2, 1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("fail", 1)
    r = evaluate_phase_path(np.array([0, 0, 0]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("neither", 2)                 # len(block_order) - 1
    r = evaluate_phase_path(np.array([2, 1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("fail", 0)
    r = evaluate_phase_path(np.array([1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("pass", 0)


TRAIL_PH = PropFirmPhase("Evaluation", 6.0, None, None, 0, drawdown=TrailingDrawdown(amount_usd=4000.0),
                         lifetime_days=30, on_horizon_exhausted="fail")


def test_i9_end_pos_evaluate_phase_path_trailing():
    acc = 150_000.0
    neutral = _bda([1], [100.0], rel_low=[0.0], rel_high=[100.0], rel_exc=[0.0], trailing=True)
    win = _bda([1], [9000.0], rel_low=[9000.0], rel_high=[9000.0], rel_exc=[0.0], trailing=True)
    breach = _bda([1], [5000.0], rel_low=[0.0], rel_high=[9000.0], rel_exc=[4000.0], trailing=True)
    late = _bda([1, 31], [1000.0, 5000.0], rel_low=[0.0, 4000.0], rel_high=[1000.0, 5000.0],
                rel_exc=[0.0, 1000.0], trailing=True)
    blocks = [neutral, win, breach, late]
    dur = np.ones(4)
    kw = dict(account_size=acc, phase=TRAIL_PH)
    r = evaluate_phase_path_trailing(np.array([0, 0, 1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("pass", 2)
    r = evaluate_phase_path_trailing(np.array([0, 2]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("fail", 1)
    r = evaluate_phase_path_trailing(np.array([0, 0]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("neither", 1)
    r = evaluate_phase_path_trailing(np.array([0, 3]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos, r.expired) == ("expired", 1, True)


def _eod_phase():
    return PropFirmPhase(
        name="Evaluation", profit_target_pct=6.0, max_daily_loss_pct=None, max_loss_pct=None,
        min_trading_days=0,
        drawdown=TrailingDrawdown(amount_usd=2000.0, lock_at_balance=50_000.0, type="eod_trailing",
                                  breach_on="equity"))


def test_i9_end_pos_evaluate_phase_path_eod_trail():
    neutral = _bda([1], [100.0], rel_low=[0.0])
    win = _bda([1], [3200.0], rel_low=[3200.0])            # objetivo 3.000 USD
    dead = _bda([1], [0.0], rel_low=[-2000.0])             # toca el umbral -> fail
    blocks = [neutral, win, dead]
    dur = np.ones(3)
    kw = dict(account_size=ACC, phase=_eod_phase())
    r = evaluate_phase_path_eod_trail(np.array([0, 1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("pass", 1)
    r = evaluate_phase_path_eod_trail(np.array([0, 0, 2]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("fail", 2)
    r = evaluate_phase_path_eod_trail(np.array([0, 0, 0]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("neither", 2)
    r = evaluate_phase_path_eod_trail(np.array([1]), blocks, dur, None, **kw)
    assert (r.outcome, r.end_pos) == ("pass", 0)


def test_end_pos_por_defecto_es_menos_uno():
    assert propfirm.PhasePathResult("pass", 1.0, 1, 1.0).end_pos == -1


# ---------------------------------------------------------------------------
# AC5: errores explicitos
# ---------------------------------------------------------------------------

def test_ac5_simulate_challenge_con_ndarray_valueerror():
    with pytest.raises(ValueError, match="PhaseTape"):
        simulate_challenge(IDX, BLOCKS, DURATIONS, _cfg())


def test_ac5_simulate_firm_paths_sin_idx_ext_valueerror():
    with pytest.raises(ValueError, match="idx_ext"):
        simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(), gap_days=GAP_DAYS, seed=1)


def test_ac5_sweep_funded_posicional_typeerror():
    cfg = _cfg()
    ch = simulate_challenge(TAPE, BLOCKS, DURATIONS, cfg)
    with pytest.raises(TypeError):
        sweep_funded(IDX, G1, [BLOCKS], ch.both, DURATIONS, cfg, [0.0])
    with pytest.raises(TypeError):
        sweep_funded(TAPE, ch.funded_start, [BLOCKS], ch.both, DURATIONS, cfg, [0.0])


def test_ac5_gaps_fixed_sin_gaps_ext_valueerror():
    with pytest.raises(ValueError, match="gaps_ext"):
        simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(), gaps_fixed=G1, idx_ext=EXT)


def test_build_phase_tape_valida_formas():
    with pytest.raises(ValueError):
        build_phase_tape(IDX, EXT[:, :-1], G1, GEXT)
    with pytest.raises(ValueError):
        build_phase_tape(IDX, EXT, G1, None)
    with pytest.raises(ValueError):
        build_phase_tape(IDX, EXT, G1[:, :-1], GEXT)
    t = build_phase_tape(IDX, EXT, None, None)
    assert isinstance(t, PhaseTape) and t.gaps is None and t.n_blocks == N_BLOCKS
    assert t.window(3, 2)[1] is None


def test_ac5_perfil_de_2_fases_con_caps_de_evaluacion_422(monkeypatch):
    from fastapi import HTTPException
    api = pytest.importorskip("poc_web.api.main")
    leg = "risk_app/inputs/trades/nasdaq_2h/Strategy 1.30.184.csv"
    if not Path(leg).exists():
        pytest.skip("datos gitignorados de risk_app/inputs no disponibles")
    base = api._firms["ftmo_100k"]
    assert len(base.phases) == 2
    ph1 = dataclasses.replace(base.phases[0], max_contracts=ContractCap(max_mini=12, max_micro=120))
    cfg_caps = dataclasses.replace(base, phases=(ph1, base.phases[1]))
    monkeypatch.setattr(api, "_firms", {**api._firms, "ftmo_100k": cfg_caps})
    pf = api.portfolio_load(api.PortfolioLoadRequest(legs=[leg], account_size=100000.0))
    req = api.PropfirmSimRequest(
        portfolio_id=pf["portfolio_id"], firm_key="ftmo_100k", f_lo=0.005, f_hi=0.01, f_step=0.005,
        n_paths=100, n_challenges=3, n_sequences=100, capital0_eur=3000.0, fx_eur_usd=1.1)
    with pytest.raises(HTTPException) as ei:
        api.simulate_propfirm(req)
    assert ei.value.status_code == 422
    assert "2 fases" in ei.value.detail
    with pytest.raises(HTTPException) as ei2:
        api.simulate_propfirm_pair(req)
    assert ei2.value.status_code == 422


# ---------------------------------------------------------------------------
# Correcciones de la revision de codigo (C3, C5, C6)
# ---------------------------------------------------------------------------

def test_c5_window_fuera_de_la_cinta_valueerror():
    n = TAPE.n_blocks
    TAPE.window(0, 0)
    TAPE.window(0, 2 * n)                       # ultima ventana valida: [2n, 3n)
    with pytest.raises(ValueError, match="fuera de la cinta"):
        TAPE.window(0, -1)
    with pytest.raises(ValueError, match="fuera de la cinta"):
        TAPE.window(0, 2 * n + 1)               # se truncaria en silencio


def test_c6_extend_stationary_rows_exige_2d():
    with pytest.raises(ValueError, match="2D"):
        extend_stationary_rows(IDX[0], N_TRADES, BS, 5, np.random.default_rng(0))


def test_c3_simulate_firm_paths_expone_funded_windows_no_la_cinta():
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, _cfg(True), gap_days=GAP_DAYS, seed=3, idx_ext=EXT)
    assert not hasattr(out, "phase_tape")
    assert out.funded_windows.dtype == np.int32
    assert out.funded_windows.shape == (N_PATHS, N_BLOCKS)
    tape = build_phase_tape(IDX, EXT, *_draw_gap_tape(np.random.default_rng(3), GAP_DAYS, *IDX.shape))
    np.testing.assert_array_equal(out.funded_windows, tape.windows(out.funded_start))
