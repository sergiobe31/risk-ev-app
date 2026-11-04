"""Motor con UNA fase (guardrail ``perfiles-v2``, fase 1, AC4).

La plantilla ``cfd_1f`` crea perfiles CFD de 1 fase: ``phases = phases[:1]`` de ``ftmo_50k``. El motor
ya lo admite (``parse_firm_spec`` acepta 1-2 fases; con 1 fase ``pass2 = 1`` y la fondeada empieza tras la
F1, ``_next_start(end_pos_F1)``). Estos tests lo DEMUESTRAN sin tocar ``src/risk/``:

  - misma cinta para ``ftmo_50k`` (2 fases) y su versión de 1 fase (``phases[:1]`` vía ``parse_firm_spec``);
  - ``pass1``, ``neither1`` y ``days_to_pass1`` idénticos por path;
  - ``pass2 == True`` y ``neither2 == False`` (no hay F2) y ``both == pass1``;
  - ``funded_start = _next_start(end_pos_F1)`` por path;
  - P(pasar) = media de ``pass1`` sobre los paths resueltos;
  - la fondeada de 1 fase es la evaluación sobre la ventana que arranca en ``funded_start``.

Todo sintético y determinista (mismo patrón que ``test_risk_propfirm_phase_tape.py``).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.risk import propfirm
from src.risk.mc import extend_stationary_rows, stationary_block_indices
from src.risk.propfirm import (
    BlockDayArrays,
    _draw_gap_tape,
    build_phase_tape,
    evaluate_funded_path,
    evaluate_phase_path,
    parse_firm_spec,
    simulate_challenge,
    simulate_firm_paths,
    sweep_funded,
)

OFFICIAL = Path(__file__).resolve().parents[1] / "config" / "propfirms.yaml"
ACC = 50_000.0
GAP_DAYS = np.array([0.0, 1.0, 3.0, 7.0])
N_TRADES = 8
N_BLOCKS = 12
N_PATHS = 300
BS = 2.5


def _raw() -> dict:
    return copy.deepcopy(yaml.safe_load(OFFICIAL.read_text(encoding="utf-8"))["firms"]["ftmo_50k"])


def _cfgs():
    raw2 = _raw()
    raw1 = copy.deepcopy(raw2)
    raw1["phases"] = raw1["phases"][:1]
    return parse_firm_spec("ftmo_50k", raw2), parse_firm_spec("ftmo_50k_1f", raw1)


CFG2, CFG1 = _cfgs()


def _rand_blocks(seed: int = 5, k: int = N_TRADES):
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


def _tape():
    rng = np.random.default_rng(1)
    idx = stationary_block_indices(N_TRADES, N_PATHS, BS, rng, length=N_BLOCKS)
    ext = extend_stationary_rows(idx, N_TRADES, BS, 2 * N_BLOCKS, np.random.default_rng([0, 2]))
    g1, g_ext = _draw_gap_tape(rng, GAP_DAYS, N_PATHS, N_BLOCKS)
    return idx, ext, build_phase_tape(idx, ext, g1, g_ext)


IDX, EXT, TAPE = _tape()


def test_la_version_de_1_fase_es_ftmo_50k_con_phases_recortadas():
    assert len(CFG2.phases) == 2 and len(CFG1.phases) == 1
    assert CFG1.phases[0] == CFG2.phases[0]
    assert CFG1.funded == CFG2.funded and CFG1.account_size == CFG2.account_size
    assert CFG1.fee == CFG2.fee and CFG1.profit_split_pct == CFG2.profit_split_pct


def test_ac4_pass1_neither1_y_dias_identicos_por_path_a_la_f1_de_ftmo_50k():
    ch2 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG2)
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    for k in ("pass1", "neither1", "days_to_pass1", "days_eval1", "expired1"):
        np.testing.assert_array_equal(getattr(ch1, k), getattr(ch2, k), err_msg=k)
    # el escenario no es degenerado: hay pass, fail y neither
    assert ch1.pass1.any() and (~ch1.pass1 & ~ch1.neither1).any() and ch1.neither1.any()


def test_ac4_sin_fase_2_pass2_es_true_y_both_es_pass1():
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    assert ch1.pass2.all() and not ch1.neither2.any()
    np.testing.assert_array_equal(ch1.both, ch1.pass1)
    np.testing.assert_array_equal(ch1.unresolved, ch1.neither1)


def test_ac4_funded_start_es_next_start_del_end_pos_de_la_f1():
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    phase1 = CFG1.phases[0]
    for p in range(N_PATHS):
        order, gaps = TAPE.window(p, 0)
        r1 = evaluate_phase_path(order, BLOCKS, DURATIONS, gaps, account_size=ACC, phase=phase1)
        assert ch1.funded_start[p] == propfirm._next_start(r1.end_pos) == r1.end_pos + 1
    assert ch1.funded_start.min() >= 1 and ch1.funded_start.max() <= N_BLOCKS


def test_ac4_con_1_fase_la_fondeada_arranca_antes_que_con_2():
    ch2 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG2)
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    both2 = ch2.pass1 & ch2.pass2
    assert both2.sum() > 5
    # Para quien pasa las dos fases, la F2 consume bloques: su fondeada empieza estrictamente después.
    assert (ch2.funded_start[both2] > ch1.funded_start[both2]).all()


def test_ac4_p_pasar_es_la_media_de_pass1_sobre_los_resueltos():
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    resolved = ~ch1.neither1
    assert 0 < resolved.sum() < N_PATHS
    p_pasar = float(ch1.both[resolved].mean())
    assert p_pasar == pytest.approx(float(ch1.pass1[resolved].mean()), abs=0)
    ref = sum(1 for p in range(N_PATHS) if ch1.pass1[p] and resolved[p]) / int(resolved.sum())
    assert p_pasar == pytest.approx(ref, abs=1e-15)
    assert 0.0 < p_pasar < 1.0


def test_ac4_fondeada_de_1_fase_es_el_evaluador_sobre_la_ventana_de_funded_start():
    ch1 = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    fu = sweep_funded(tape=TAPE, funded_start=ch1.funded_start, day_arrays_f_grid=[BLOCKS],
                      passers_mask=ch1.both, durations=DURATIONS, cfg=CFG1, f_grid=[0.0])[0.0]
    passers = np.flatnonzero(ch1.both)
    assert len(passers) > 5
    for p in passers:
        o, g = TAPE.window(p, ch1.funded_start[p])
        rf = evaluate_funded_path(o, BLOCKS, DURATIONS, g, account_size=ACC, phase=CFG1.phases[0],
                                  funded=CFG1.funded, profit_split=CFG1.profit_split_pct / 100.0)
        np.testing.assert_array_equal(fu.funded_payouts[p], rf.cycle_payouts)
    assert (fu.funded_payouts[np.flatnonzero(~ch1.both)] == 0.0).all()


def test_ac4_simulate_firm_paths_1d_equivale_al_camino_2d_con_1_fase():
    out = simulate_firm_paths(IDX, BLOCKS, DURATIONS, CFG1, gap_days=GAP_DAYS, seed=9, idx_ext=EXT)
    # mismos gaps que la cinta del 2D: _draw_gap_tape(default_rng(9), ...) en la F1
    g1, g_ext = _draw_gap_tape(np.random.default_rng(9), GAP_DAYS, N_PATHS, N_BLOCKS)
    tape = build_phase_tape(IDX, EXT, g1, g_ext)
    ch = simulate_challenge(tape, BLOCKS, DURATIONS, CFG1)
    for k in ("pass1", "pass2", "neither1", "neither2", "days_to_pass1", "funded_start"):
        np.testing.assert_array_equal(getattr(out, k), getattr(ch, k), err_msg=k)
    assert out.pass2.all()


def test_el_camino_de_1_fase_es_determinista():
    a = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    b = simulate_challenge(TAPE, BLOCKS, DURATIONS, CFG1)
    for k in ("pass1", "neither1", "days_to_pass1", "funded_start"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
