"""I6 del guardrail ``onboarding-b1``: ``cut_by_data`` (fondeada ``cycle`` viva cuando
se acaba la ventana de datos, sin llegar al cap de ciclos).

Todo sintético (sin CSV/OHLC). Cubre:
* ``evaluate_funded_path``: ventana corta y viva -> True; cap alcanzado -> False;
  muerta -> False; ``alive_at_cap`` no cambia y es disjunto de ``cut_by_data``.
* Propagación por ``simulate_firm_paths`` (``sweep_funded`` -> ``FundedOutcomes`` ->
  ``combine_challenge_funded`` -> ``FirmPathOutcomes``).
* ``FirmReport.funded_cut_by_data_frac`` con la MISMA máscara que ``alive_at_cap_frac``
  (``funded_reached = both & resolved``, ``sequence_ev``): enmienda 3.
* I1: el campo nuevo NO entra en ``to_row()``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.risk.propfirm import (
    BlockDayArrays,
    FirmPathOutcomes,
    FundedOutcomes,
    FundedRules,
    PropFirmConfig,
    PropFirmPhase,
    evaluate_funded_path,
    firm_report,
    sequence_ev,
    simulate_firm_paths,
)

ACC = 50_000.0
PHASE = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)


def _bda(days, rel_closed):
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_before = np.concatenate([[0.0], rel_closed[:-1]])
    return BlockDayArrays(
        day_ord=days, rel_before=rel_before, rel_closed=rel_closed,
        rel_float=rel_before.copy(), opens_day=np.ones(len(days), dtype=int),
        total_pnl=float(rel_closed[-1]))


def _funded_eval(ba, max_cycles, dur=30.0):
    funded = FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0,
                         max_payout_cycles=max_cycles)
    return evaluate_funded_path(
        np.array([0]), [ba], np.array([dur]),
        account_size=ACC, phase=PHASE, funded=funded, profit_split=0.8)


# --- evaluate_funded_path ---------------------------------------------------------
def test_ventana_corta_y_viva_es_cut_by_data():
    # 3 ciclos de tope (42 d) pero solo hay datos hasta el día 20: 1 ciclo cerrado.
    r = _funded_eval(_bda([1, 14, 20], [500.0, 1000.0, 1500.0]), max_cycles=3)
    assert r.cut_by_data is True
    assert r.alive_at_cap is False


def test_cap_alcanzado_no_es_cut_by_data():
    days = list(range(1, 61, 3))
    r = _funded_eval(_bda(days, [500.0 * (i + 1) for i in range(len(days))]),
                     max_cycles=3)
    assert r.alive_at_cap is True
    assert r.cut_by_data is False


def test_fondeada_muerta_no_es_cut_by_data():
    r = _funded_eval(_bda([1], [-6000.0]), max_cycles=3)  # 44000 <= 45000
    assert r.alive_at_cap is False
    assert r.cut_by_data is False


def test_alive_at_cap_y_cut_by_data_son_disjuntos():
    casos = [
        _bda([1, 14, 20], [500.0, 1000.0, 1500.0]),
        _bda(list(range(1, 61, 3)), [500.0 * (i + 1) for i in range(20)]),
        _bda([1], [-6000.0]),
    ]
    for ba in casos:
        r = _funded_eval(ba, max_cycles=3)
        assert not (r.alive_at_cap and r.cut_by_data)


def test_payouts_y_alive_at_cap_intactos():
    # Mismos números que test_risk_propfirm::test_funded_payout_y_reset_a_baseline.
    r = _funded_eval(_bda([1, 14, 28], [2000.0, 6000.0, 8000.0]), max_cycles=3)
    assert r.cycle_payouts[0] == pytest.approx(4800.0)
    assert r.cycle_payouts[1] == pytest.approx(1600.0)
    assert r.days_to_first_payout == pytest.approx(14.0)
    assert r.alive_at_cap is False
    assert r.cut_by_data is True   # 3er ciclo sin cerrar: se acabó la ventana


# --- Propagación por simulate_firm_paths ------------------------------------------
def _cfg(max_cycles, refund=False) -> PropFirmConfig:
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=345.0,
        profit_split_pct=80.0, refund_fee_on_first_payout=refund,
        phases=(PropFirmPhase("C", 1.0, 5.0, 10.0, 1),),
        funded=FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0,
                           max_payout_cycles=max_cycles))


def _pipeline(max_cycles, n_blocks=30, n_paths=20):
    """Todos los bloques ganan +300/día en 2 días: la F1 pasa en el 1er bloque y la
    fondeada sobrevive. Ventana de la fondeada ~ 59 bloques x 3 d (≈ 177 d)."""
    ba = _bda([1, 2], [300.0, 600.0])
    day_arrays = [ba] * n_blocks
    durations = np.full(n_blocks, 3.0)
    idx = np.tile(np.arange(n_blocks), (n_paths, 1))
    idx_ext = np.tile(np.arange(2 * n_blocks) % n_blocks, (n_paths, 1))
    return simulate_firm_paths(idx, day_arrays, durations, _cfg(max_cycles),
                               gap_days=None, seed=0, idx_ext=idx_ext)


def test_pipeline_ventana_menor_que_el_cap_marca_cut_by_data():
    out = _pipeline(max_cycles=50)               # 700 d de tope >> ventana
    assert out.both.all()
    assert out.cut_by_data is not None
    assert out.cut_by_data.all()
    assert not out.alive_at_cap.any()
    rep = firm_report(0.01, out, sequence_ev(
        out, _cfg(50), fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
        initial_capital_eur=5000.0, seed=1))
    assert rep.funded_cut_by_data_frac == pytest.approx(1.0)
    assert rep.alive_at_cap_frac == pytest.approx(0.0)


def test_pipeline_cap_alcanzado_no_marca_cut_by_data():
    out = _pipeline(max_cycles=2)                # 28 d de tope << ventana
    assert out.both.all()
    assert not out.cut_by_data.any()
    assert out.alive_at_cap.all()
    rep = firm_report(0.01, out, sequence_ev(
        out, _cfg(2), fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
        initial_capital_eur=5000.0, seed=1))
    assert rep.funded_cut_by_data_frac == pytest.approx(0.0)
    assert rep.alive_at_cap_frac == pytest.approx(1.0)


def test_sweep_funded_legacy_no_paga_cut_by_data_en_otros_modelos():
    # FundedOutcomes sin el campo (caminos ladder/futures) -> None por defecto.
    fu = FundedOutcomes(funded_payouts=np.zeros((2, 3)),
                        days_first_payout=np.full(2, np.nan),
                        alive_at_cap=np.zeros(2, dtype=bool))
    assert fu.cut_by_data is None


# --- Denominador (enmienda 3) ------------------------------------------------------
def _outcomes_mixtos():
    """8 paths: 0-3 llegan a fondeada y resuelven; 4 no pasa F1; 5 'neither' en F1;
    6-7 pasan F1 y F2 pero F2 neither (sin resolver)."""
    n = 8
    pass1 = np.array([1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    pass2 = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    neither1 = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    neither2 = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    funded = np.zeros((n, 3))
    funded[:4, 0] = 1000.0
    cut = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=bool)   # 4-7 NO deben contar
    alive = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    return FirmPathOutcomes(
        pass1=pass1, pass2=pass2, neither1=neither1, neither2=neither2,
        funded_payouts=funded, days_first_payout=np.where(pass1 & pass2, 14.0, np.nan),
        alive_at_cap=alive, days_to_pass1=np.where(pass1, 10.0, np.nan),
        cut_by_data=cut)


def test_denominador_es_both_y_resolved_igual_que_alive_at_cap_frac():
    oc = _outcomes_mixtos()
    cfg = _cfg(3)
    seq = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    rep = firm_report(0.01, oc, seq)
    funded_reached = oc.both & oc.resolved
    assert funded_reached.sum() == 4
    assert rep.funded_cut_by_data_frac == pytest.approx(
        oc.cut_by_data[funded_reached].mean())            # 2/4
    assert rep.funded_cut_by_data_frac == pytest.approx(0.5)
    # Misma máscara que alive_at_cap_frac del motor.
    assert rep.alive_at_cap_frac == pytest.approx(
        oc.alive_at_cap[funded_reached].mean())            # 1/4


def test_sin_cut_by_data_el_campo_es_none():
    oc = dataclasses.replace(_outcomes_mixtos(), cut_by_data=None)
    seq = sequence_ev(oc, _cfg(3), fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    assert firm_report(0.01, oc, seq).funded_cut_by_data_frac is None


def test_nadie_llega_a_fondeada_es_nan_como_alive_frac():
    base = _outcomes_mixtos()
    oc = dataclasses.replace(base, pass1=np.zeros(8, dtype=bool))
    seq = sequence_ev(oc, _cfg(3), fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    rep = firm_report(0.01, oc, seq)
    assert np.isnan(rep.funded_cut_by_data_frac)
    assert np.isnan(rep.alive_at_cap_frac)


def test_to_row_no_cambia_I1():
    oc = _outcomes_mixtos()
    seq = sequence_ev(oc, _cfg(3), fx_eur_usd=1.0, n_challenges=3, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    row = firm_report(0.01, oc, seq).to_row()
    assert "funded_cut_by_data_frac" not in row
    assert "vivo_al_cap" in row
