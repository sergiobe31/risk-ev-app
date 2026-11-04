"""Tests del diagnóstico de caps de contratos y multi-cuenta (plan Apex Fase 5).

Todo calculado a mano sobre bloques/outcomes sintéticos. Cubre: máximo de
contratos simultáneos por bloque (solapamientos, convenio de borde [ot, ct),
trade instantáneo ot==ct), equivalencia mini→micro cross-instrumento (tabla y
suma directa default), reducción path = max de bloques, violación estricta del
cap de evaluación, tier 1 (40 micros) vs tier alto (100) con max_observed
reportado, máscara de paths (PA solo passers), sizing_invalid no silenciable
(warning estructurado, strict > tol, FTMO intacto), config max_contracts
(parseo YAML real + validaciones duras), helpers de caps desde cfg, y
multi-cuenta (N=5: EV/fees/payouts ×N exactos, p_net_negative idéntica,
insolvencia con fee escalado, N=1 = no-op exacto, validación 1-20).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.risk.propfirm import (
    ChallengeOutcomes,
    ContractCap,
    ContractCapDiag,
    FirmPathOutcomes,
    FundedOutcomes,
    FundedRules,
    PortfolioDayBlock,
    PropFirmConfig,
    PropFirmPhase,
    block_max_simultaneous_micros,
    combine_challenge_funded,
    contract_cap_diagnostic,
    eval_contract_cap_micros,
    firm_report,
    load_propfirm_config,
    pa_min_tier_cap_micros,
    sequence_ev,
)

_T0 = pd.Timestamp("2026-01-05")  # lunes


def _pblock(ots, cts, sym_ids=None, local_ids=None, idx=0):
    """PortfolioDayBlock sintético con timestamps (lo demás, plausible)."""
    k = len(ots)
    ots64 = np.array(pd.to_datetime(list(ots)), dtype="datetime64[ns]")
    cts64 = np.array(pd.to_datetime(list(cts)), dtype="datetime64[ns]")
    if sym_ids is None:
        sym_ids = [0] * k
    if local_ids is None:
        local_ids = list(range(k))
    start, end = pd.Timestamp(ots64[0]), pd.Timestamp(cts64[-1])
    return PortfolioDayBlock(
        idx=idx, start=start, end=end,
        duration_days=(end - start).total_seconds() / 86400.0,
        sym_id=np.array(sym_ids, dtype=int),
        trade_local_id=np.array(local_ids, dtype=int),
        open_ord=ots64.astype("datetime64[D]").astype(int) + 719163,
        close_ord=cts64.astype("datetime64[D]").astype(int) + 719163,
        valle_precio=[{}] * k,
        pnl_per_lot=np.ones(k),
        n_trades=k,
        n_no_valley=k,
        open_t=ots64,
        close_t=cts64,
    )


def _h(hh, mm=0):
    return _T0 + pd.Timedelta(hours=hh, minutes=mm)


# Bloque A (todo micros, sym 0): t0 10→12, t1 11→13, t2 12→14, t3 15→15:30.
# Solapamiento a mano con el convenio [ot, ct): [10,11)=30, [11,12)=60,
# [12,13)=t1+t2=60 (t0 ya cerró EN las 12:00), [13,14)=30, [15,15:30)=50.
def _block_a():
    return _pblock([_h(10), _h(11), _h(12), _h(15)],
                   [_h(12), _h(13), _h(14), _h(15, 30)], idx=0)


LOTS_A = {0: np.array([[30.0, 30.0, 30.0, 50.0]])}


def test_max_simultaneo_a_mano_solapamientos_y_borde_exclusivo():
    b = _block_a()
    lots = np.array([30.0, 30.0, 30.0, 50.0])
    # Máximo 60: en el borde 12:00 t0 cierra y t2 abre — NO son simultáneos
    # (si el borde fuera inclusivo sería 90).
    assert block_max_simultaneous_micros(b, lots) == 60.0


def test_trade_instantaneo_ot_igual_ct_no_suma():
    b = _pblock([_h(10), _h(11), _h(12), _h(15), _h(16)],
                [_h(12), _h(13), _h(14), _h(15, 30), _h(16)], idx=0)
    lots = np.array([30.0, 30.0, 30.0, 50.0, 100.0])  # el de 100 es ot==ct
    assert block_max_simultaneous_micros(b, lots) == 60.0


def test_equivalencia_mini_a_micro_cross_instrumento():
    # s0: 40 micros 09:00→10:30; s1: 3 lotes de MINI 10:00→11:00 (=30 micro-eq).
    # Overlap [10:00,10:30): 40 + 30 = 70 micro-eq.
    b = _pblock([_h(9), _h(10)], [_h(10, 30), _h(11)],
                sym_ids=[0, 1], local_ids=[0, 0], idx=0)
    lots = np.array([40.0, 3.0])
    assert block_max_simultaneous_micros(b, lots, np.array([1.0, 10.0])) == 70.0
    # Sin tabla (default 1.0 = todo micros, suma directa): 40 + 3 = 43.
    assert block_max_simultaneous_micros(b, lots) == 43.0


def test_bloque_sin_timestamps_raise_ruidoso():
    b = _pblock([_h(9)], [_h(10)])
    b.open_t = None
    b.close_t = None
    with pytest.raises(ValueError, match="open_t/close_t"):
        block_max_simultaneous_micros(b, np.array([1.0]))


# ---------------------------------------------------------------------------
# contract_cap_diagnostic: reducción por path, violación estricta, máscara
# ---------------------------------------------------------------------------

def _blocks_ab():
    """Bloque A (máx 60 micro-eq) y bloque B multi-símbolo (máx 70)."""
    b_b = _pblock([_h(9), _h(10)], [_h(10, 30), _h(11)],
                  sym_ids=[0, 1], local_ids=[0, 0], idx=1)
    lots = {0: np.array([[30.0, 30.0, 30.0, 50.0]]), 1: np.array([[3.0]])}
    # El trade de sym 0 del bloque B usa local 0 → lots[0][0] = 30, no 40;
    # para fijar B a mano usamos su propia grilla:
    lots_b = {0: np.array([[40.0]]), 1: np.array([[3.0]])}
    # Unificamos: bloque A usa sym 0 local 0..3; bloque B sym 0 local 4 y
    # sym 1 local 0. Así una sola grilla sirve para ambos.
    b_b2 = _pblock([_h(9), _h(10)], [_h(10, 30), _h(11)],
                   sym_ids=[0, 1], local_ids=[4, 0], idx=1)
    lots_all = {0: np.array([[30.0, 30.0, 30.0, 50.0, 40.0]]),
                1: np.array([[3.0]])}
    return [_block_a(), b_b2], lots_all


def test_path_max_es_el_max_de_los_maximos_de_bloque():
    blocks, lots = _blocks_ab()
    idx = np.array([[0, 0], [1, 1], [0, 1]])  # repetir un bloque no mueve el máx
    diag = contract_cap_diagnostic(
        idx, blocks, lots, 0, cap_micros=1e9,
        micro_equiv_by_sym={0: 1.0, 1: 10.0},
    )
    np.testing.assert_allclose(diag.path_max, [60.0, 70.0, 70.0])
    assert diag.max_observed == 70.0
    assert diag.n_paths == 3
    assert diag.frac_violating == 0.0


def test_violacion_estricta_del_cap_de_evaluacion():
    blocks, lots = _blocks_ab()
    idx = np.array([[0, 0], [1, 1], [0, 1]])
    eq = {0: 1.0, 1: 10.0}
    # Cap 60: solo violan los paths con máx 70 (violación ESTRICTA: 60 no > 60).
    diag = contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=60.0,
                                   micro_equiv_by_sym=eq, scope="evaluation")
    assert diag.frac_violating == pytest.approx(2 / 3)
    assert diag.max_observed == 70.0
    assert diag.scope == "evaluation"
    # Cap 70: nadie viola (70 no > 70). Cap 120 (evaluación Apex): nadie.
    assert contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=70.0,
                                   micro_equiv_by_sym=eq).frac_violating == 0.0
    assert contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=120.0,
                                   micro_equiv_by_sym=eq).frac_violating == 0.0


def test_cap_evaluacion_apex_120_con_path_violador():
    # Un solo trade de 130 lotes micro: viola el cap de evaluación (120).
    b = _pblock([_h(9)], [_h(10)])
    lots = {0: np.array([[130.0]])}
    diag = contract_cap_diagnostic(np.array([[0]]), [b], lots, 0,
                                   cap_micros=120.0, scope="evaluation")
    assert diag.frac_violating == 1.0
    assert diag.max_observed == 130.0


def test_tier1_40_vs_tier_alto_y_max_observed_reportado():
    # PA: la cota conservadora v1 es el tier MÍNIMO (40 micros, del YAML real).
    cfg = load_propfirm_config()["apex_150k_intraday"]
    assert pa_min_tier_cap_micros(cfg) == 40.0
    blocks, lots = _blocks_ab()
    idx = np.array([[1]])
    eq = {0: 1.0, 1: 10.0}
    d40 = contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=40.0,
                                  micro_equiv_by_sym=eq, scope="funded")
    assert d40.frac_violating == 1.0      # 70 > 40 (tier 1)
    assert d40.max_observed == 70.0
    # Contra el cap del tier 4 (100) el mismo path sería legal — v1 no lo
    # resuelve solo (limitación documentada), pero max_observed lo permite.
    d100 = contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=100.0,
                                   micro_equiv_by_sym=eq, scope="funded")
    assert d100.frac_violating == 0.0
    assert d100.max_observed == 70.0      # 70 <= 100: legal en tier 4


def test_mascara_restringe_la_fraccion_y_cuenta_paths():
    blocks, lots = _blocks_ab()
    idx = np.array([[0, 0], [1, 1], [0, 1]])
    mask = np.array([True, False, True])
    diag = contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=60.0,
                                   micro_equiv_by_sym={0: 1.0, 1: 10.0},
                                   mask=mask)
    assert diag.n_paths == 2
    assert diag.frac_violating == 0.5     # de los paths 0 y 2, solo el 2 viola
    np.testing.assert_allclose(diag.path_max[[0, 2]], [60.0, 70.0])
    assert np.isnan(diag.path_max[1])


def test_diagnostic_validaciones_duras():
    blocks, lots = _blocks_ab()
    idx = np.array([[0]])
    with pytest.raises(ValueError, match="cap_micros"):
        contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=0.0)
    with pytest.raises(ValueError, match="scope"):
        contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=10.0,
                                scope="pa")
    with pytest.raises(ValueError, match="mask"):
        contract_cap_diagnostic(idx, blocks, lots, 0, cap_micros=10.0,
                                mask=np.array([True, False]))


# ---------------------------------------------------------------------------
# Config: max_contracts en el perfil Apex + helpers de caps
# ---------------------------------------------------------------------------

def test_yaml_apex_parsea_max_contracts_y_helpers():
    cfgs = load_propfirm_config()
    apex = cfgs["apex_150k_intraday"]
    assert apex.phases[0].max_contracts == ContractCap(max_mini=12, max_micro=120)
    assert eval_contract_cap_micros(apex) == 120.0
    assert pa_min_tier_cap_micros(apex) == 40.0
    ftmo = cfgs["ftmo_50k"]
    assert all(ph.max_contracts is None for ph in ftmo.phases)
    assert eval_contract_cap_micros(ftmo) is None
    assert pa_min_tier_cap_micros(ftmo) is None


def test_loader_max_contracts_validacion_dura(tmp_path):
    spec = {
        "display_name": "t", "account_size": 150000, "fee": 50,
        "profit_split_pct": 100,
        "phases": [{
            "name": "Evaluation", "profit_target_pct": 6.0,
            "min_trading_days": 0, "max_daily_loss_pct": None,
            "max_loss_pct": None,
            "max_contracts": {"max_mini": 0, "max_micro": 120},
        }],
        "funded": {"payout_cycle_days": 14, "min_payout_profit_pct": 0,
                   "max_payout_cycles": 50},
    }
    p = tmp_path / "firms.yaml"
    p.write_text(yaml.safe_dump({"firms": {"t": spec}}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_contracts"):
        load_propfirm_config(p)


# ---------------------------------------------------------------------------
# sizing_invalid: marca no silenciable en sequence_ev / firm_report
# ---------------------------------------------------------------------------

def _cfg_cycle():
    return PropFirmConfig(
        key="t", display_name="t", account_size=10_000.0, fee=100.0,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4),),
        funded=FundedRules(payout_cycle_days=3, min_payout_profit_pct=0.0,
                           max_payout_cycles=4),
    )


def _oc_basico(n=2):
    return FirmPathOutcomes(
        pass1=np.ones(n, dtype=bool), pass2=np.ones(n, dtype=bool),
        neither1=np.zeros(n, dtype=bool), neither2=np.zeros(n, dtype=bool),
        funded_payouts=np.zeros((n, 4)),
        days_first_payout=np.full(n, np.nan),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.full(n, 5.0),
    )


def _diag(scope, frac, cap=40.0, mx=70.0, n=2):
    return ContractCapDiag(scope=scope, cap_micros=cap, n_paths=n,
                           frac_violating=frac, max_observed=mx,
                           path_max=np.full(n, mx))


def test_sizing_invalid_cuando_la_fraccion_supera_el_umbral():
    oc = _oc_basico()
    oc.contract_cap_pa = _diag("funded", 0.5)
    cfg = _cfg_cycle()
    seq = sequence_ev(oc, cfg, fx_eur_usd=1.1, n_challenges=2, n_sequences=4,
                      initial_capital_eur=1000.0, seed=0)
    assert seq.sizing_invalid
    assert seq.frac_cap_violations_pa == pytest.approx(0.5)
    assert np.isnan(seq.frac_cap_violations_eval)
    assert seq.max_simultaneous_micros == pytest.approx(70.0)
    assert any(w.startswith("SIZING_INVALID[funded]") for w in seq.warnings)
    # Las métricas se reportan igualmente (se marca, no se aborta).
    assert seq.ev_per_attempt == pytest.approx(-100.0)
    rep = firm_report(0.01, oc, seq)
    assert rep.sizing_invalid
    assert rep.warnings == seq.warnings
    assert rep.frac_cap_violations_pa == pytest.approx(0.5)


def test_umbral_estricto_fraccion_igual_al_tol_no_invalida():
    oc = _oc_basico()
    oc.contract_cap_pa = _diag("funded", 0.5)
    seq = sequence_ev(oc, _cfg_cycle(), fx_eur_usd=1.1, n_challenges=2,
                      n_sequences=4, initial_capital_eur=1000.0, seed=0,
                      cap_violation_tol=0.5)
    assert not seq.sizing_invalid
    assert not any(w.startswith("SIZING_INVALID") for w in seq.warnings)
    # El diagnóstico sigue reportado aunque no invalide.
    assert seq.frac_cap_violations_pa == pytest.approx(0.5)


def test_sizing_invalid_tambien_por_cap_de_evaluacion():
    oc = _oc_basico()
    oc.contract_cap_eval = _diag("evaluation", 0.25, cap=120.0, mx=130.0)
    seq = sequence_ev(oc, _cfg_cycle(), fx_eur_usd=1.1, n_challenges=2,
                      n_sequences=4, initial_capital_eur=1000.0, seed=0)
    assert seq.sizing_invalid
    assert seq.frac_cap_violations_eval == pytest.approx(0.25)
    assert seq.max_simultaneous_micros == pytest.approx(130.0)
    assert any(w.startswith("SIZING_INVALID[evaluation]") for w in seq.warnings)


def test_ftmo_sin_diagnosticos_queda_intacto():
    oc = _oc_basico()
    seq = sequence_ev(oc, _cfg_cycle(), fx_eur_usd=1.1, n_challenges=2,
                      n_sequences=4, initial_capital_eur=1000.0, seed=0)
    assert not seq.sizing_invalid
    assert seq.warnings == ()
    assert np.isnan(seq.frac_cap_violations_eval)
    assert np.isnan(seq.frac_cap_violations_pa)
    assert np.isnan(seq.max_simultaneous_micros)
    assert seq.n_copied_accounts == 1


def test_combine_acopla_los_diagnosticos_de_cap():
    ch = ChallengeOutcomes(
        pass1=np.ones(2, dtype=bool), pass2=np.ones(2, dtype=bool),
        neither1=np.zeros(2, dtype=bool), neither2=np.zeros(2, dtype=bool),
        days_to_pass1=np.full(2, 5.0),
        contract_cap=_diag("evaluation", 0.0, cap=120.0, mx=60.0),
    )
    fu = FundedOutcomes(
        funded_payouts=np.zeros((2, 4)), days_first_payout=np.full(2, np.nan),
        alive_at_cap=np.zeros(2, dtype=bool),
        contract_cap=_diag("funded", 0.5),
    )
    oc = combine_challenge_funded(ch, fu)
    assert oc.contract_cap_eval is ch.contract_cap
    assert oc.contract_cap_pa is fu.contract_cap
    # FTMO (sin diagnósticos): ambos None.
    ch2 = ChallengeOutcomes(
        pass1=np.ones(2, dtype=bool), pass2=np.ones(2, dtype=bool),
        neither1=np.zeros(2, dtype=bool), neither2=np.zeros(2, dtype=bool),
        days_to_pass1=np.full(2, 5.0),
    )
    fu2 = FundedOutcomes(
        funded_payouts=np.zeros((2, 4)), days_first_payout=np.full(2, np.nan),
        alive_at_cap=np.zeros(2, dtype=bool),
    )
    oc2 = combine_challenge_funded(ch2, fu2)
    assert oc2.contract_cap_eval is None and oc2.contract_cap_pa is None


# ---------------------------------------------------------------------------
# Multi-cuenta (copy trading): N=5 escala todo ×N; N=1 = no-op exacto
# ---------------------------------------------------------------------------

def _cfg_multi():
    return PropFirmConfig(
        key="t5", display_name="t5", account_size=10_000.0, fee=100.0,
        profit_split_pct=100.0, refund_fee_on_first_payout=True,
        phases=(PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4),),
        funded=FundedRules(payout_cycle_days=3, min_payout_profit_pct=0.0,
                           max_payout_cycles=4),
    )


def _oc_multi():
    """4 paths resueltos: 0 y 1 pasan y cobran (3000/1500 USD), 2 y 3 fallan."""
    fp = np.zeros((4, 4))
    fp[0, 0], fp[1, 0] = 3000.0, 1500.0
    return FirmPathOutcomes(
        pass1=np.array([True, True, False, False]),
        pass2=np.ones(4, dtype=bool),
        neither1=np.zeros(4, dtype=bool), neither2=np.zeros(4, dtype=bool),
        funded_payouts=fp,
        days_first_payout=np.array([3.0, 3.0, np.nan, np.nan]),
        alive_at_cap=np.zeros(4, dtype=bool),
        days_to_pass1=np.array([5.0, 5.0, np.nan, np.nan]),
    )


def _seq(oc, cfg, n, seed=3, stop=False, capital=100_000.0, n_ch=10):
    return sequence_ev(oc, cfg, fx_eur_usd=1.25, n_challenges=n_ch,
                       n_sequences=300, initial_capital_eur=capital, seed=seed,
                       stop_on_insolvency=stop, n_copied_accounts=n)


def test_multicuenta_n5_escala_ev_fees_y_payouts_exactamente():
    oc, cfg = _oc_multi(), _cfg_multi()
    s1, s5 = _seq(oc, cfg, 1), _seq(oc, cfg, 5)
    # EV por path a mano (fx 1.25, refund 100 condicionado a payout):
    # p0: −100 + 3000/1.25 + 100 = 2400; p1: 1200; p2/p3: −100 → media 850.
    assert s1.ev_per_attempt == pytest.approx(850.0)
    assert s5.ev_per_attempt == pytest.approx(5 * 850.0)
    assert s5.ev_ci95 == (
        pytest.approx(5 * s1.ev_ci95[0]), pytest.approx(5 * s1.ev_ci95[1]))
    assert s5.net_mean == pytest.approx(5 * s1.net_mean)
    assert s5.net_p05 == pytest.approx(5 * s1.net_p05)
    assert s5.net_p95 == pytest.approx(5 * s1.net_p95)
    # E[payout|fond.] N=1: (2400+100 + 1200+100)/2 = 1900 → N=5: 9500.
    assert s1.mean_payout_eur == pytest.approx(1900.0)
    assert s5.mean_payout_eur == pytest.approx(9500.0)
    # Correlación perfecta: P(neto<0) NO cambia por copiar (mismo seed/draws).
    assert s5.p_net_negative == pytest.approx(s1.p_net_negative)
    # Trayectorias escaladas ×N alrededor del capital inicial.
    np.testing.assert_allclose(
        s5.trajectory_p50 - 100_000.0, 5 * (s1.trajectory_p50 - 100_000.0))
    assert s5.n_copied_accounts == 5
    assert any(w.startswith("MULTI_CUENTA") for w in s5.warnings)
    assert not s5.sizing_invalid
    rep = firm_report(0.01, oc, s5)
    assert rep.n_copied_accounts == 5
    assert rep.warnings == s5.warnings


def _oc_all_fail():
    return FirmPathOutcomes(
        pass1=np.zeros(4, dtype=bool), pass2=np.ones(4, dtype=bool),
        neither1=np.zeros(4, dtype=bool), neither2=np.zeros(4, dtype=bool),
        funded_payouts=np.zeros((4, 4)),
        days_first_payout=np.full(4, np.nan),
        alive_at_cap=np.zeros(4, dtype=bool),
        days_to_pass1=np.full(4, np.nan),
    )


def test_multicuenta_insolvencia_con_fee_escalado():
    # Todo falla: cada intento cuesta el fee. fee=100, capital=250, 5 challenges.
    # N=1: pre = 250,150,50 → ejecuta 2 intentos y para (ruina); neto −200.
    # N=5: fee efectivo 500 > 250 → 0 intentos, ruina inmediata, neto 0.
    oc, cfg = _oc_all_fail(), _cfg_multi()
    s1 = _seq(oc, cfg, 1, stop=True, capital=250.0, n_ch=5)
    assert s1.mean_attempts == pytest.approx(2.0)
    assert s1.p_ruin == pytest.approx(1.0)
    assert s1.net_mean == pytest.approx(-200.0)
    s5 = _seq(oc, cfg, 5, stop=True, capital=250.0, n_ch=5)
    assert s5.mean_attempts == pytest.approx(0.0)
    assert s5.p_ruin == pytest.approx(1.0)
    assert s5.net_mean == pytest.approx(0.0)
    # N=2, capital 600: fee 200; pre = 600,400,200,0 → 3 intentos; neto −600.
    s2 = _seq(oc, cfg, 2, stop=True, capital=600.0, n_ch=5)
    assert s2.mean_attempts == pytest.approx(3.0)
    assert s2.net_mean == pytest.approx(-600.0)
    # Sin stop el horizonte se ejecuta entero y el neto escala ×N exacto.
    s1n = _seq(oc, cfg, 1, stop=False, capital=250.0, n_ch=5)
    s5n = _seq(oc, cfg, 5, stop=False, capital=250.0, n_ch=5)
    assert s1n.net_mean == pytest.approx(-500.0)
    assert s5n.net_mean == pytest.approx(-2500.0)
    assert s1n.p_net_negative == pytest.approx(1.0)
    assert s5n.p_net_negative == pytest.approx(1.0)


def test_multicuenta_n1_es_noop_exacto():
    oc, cfg = _oc_multi(), _cfg_multi()
    s_def = sequence_ev(oc, cfg, fx_eur_usd=1.25, n_challenges=10,
                        n_sequences=300, initial_capital_eur=100_000.0, seed=3)
    s_n1 = _seq(oc, cfg, 1)
    assert s_n1.n_copied_accounts == 1
    assert s_n1.warnings == ()
    assert s_n1.ev_per_attempt == s_def.ev_per_attempt
    assert s_n1.ev_ci95 == s_def.ev_ci95
    assert s_n1.net_mean == s_def.net_mean
    assert s_n1.p_net_negative == s_def.p_net_negative
    np.testing.assert_array_equal(s_n1.trajectory_p50, s_def.trajectory_p50)


def test_multicuenta_validacion_rango_1_20():
    oc, cfg = _oc_multi(), _cfg_multi()
    for bad in (0, 21, -1, 2.5, True, "5"):
        with pytest.raises(ValueError, match="n_copied_accounts"):
            _seq(oc, cfg, bad)
    _seq(oc, cfg, 20)  # el tope es válido


def test_cap_violation_tol_validacion():
    oc, cfg = _oc_basico(), _cfg_cycle()
    for bad in (-0.1, 1.01):
        with pytest.raises(ValueError, match="cap_violation_tol"):
            sequence_ev(oc, cfg, fx_eur_usd=1.1, n_challenges=2, n_sequences=4,
                        initial_capital_eur=1000.0, cap_violation_tol=bad)
