"""Tests del simulador EV de prop firms (src/risk/propfirm.py).

Se testean las barreras a nivel de arrays diarios sinteticos (sin pipeline
CSV/OHLC): daily loss dinamico, max loss estatico, min trading days con
continuacion post-target, fondeada (payout/reset/breach/cap), nivel 2 con
distribuciones degeneradas, y la carga del YAML de perfiles.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.propfirm import (
    BlockDayArrays,
    FirmPathOutcomes,
    FundedRules,
    PropFirmPhase,
    evaluate_funded_path,
    evaluate_phase_path,
    firm_report,
    load_propfirm_config,
    recommend_f_ev,
    sequence_ev,
    FirmReport,
)

ACC = 50_000.0
# FTMO-like: target 10%, daily 5% (2500), max loss 10% (linea 45000), 4 dias.
PHASE = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)


def _bda(days, rel_closed, rel_float=None, opens=None):
    """BlockDayArrays sintetico. rel_float por defecto = rel_before (sin valle
    adverso). Precondicion: valle <= cierre del dia (min flotante <= cierre)."""
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_before = np.concatenate([[0.0], rel_closed[:-1]])
    if rel_float is None:
        rel_float = rel_before.copy()
    return BlockDayArrays(
        day_ord=days,
        rel_before=rel_before,
        rel_closed=rel_closed,
        rel_float=np.asarray(rel_float, dtype=float),
        opens_day=np.asarray(opens if opens is not None else [1] * len(days), dtype=int),
        total_pnl=float(rel_closed[-1]),
    )


def _phase_eval(ba, phase=PHASE, account=ACC):
    return evaluate_phase_path(
        np.array([0]), [ba], np.array([30.0]),
        account_size=account, phase=phase,
    )


# ---------------------------------------------------------------------------
# Daily loss dinamico
# ---------------------------------------------------------------------------

def test_daily_loss_breach_por_valle_intradia_no_por_cierre():
    # Cierra +1000 pero el valle del dia toca -2600 -> suelo 47500 (50000-2500).
    ba = _bda([1], [1000.0], rel_float=[-2600.0])
    assert _phase_eval(ba).outcome == "fail"


def test_daily_loss_no_breach_si_valle_por_encima_del_suelo():
    ba = _bda([1], [1000.0], rel_float=[-2000.0])
    r = _phase_eval(ba)
    assert r.outcome != "fail"  # neither (no llega a target)


def test_daily_floor_se_recalcula_desde_balance_inicio_de_dia():
    # Dia 1 cierra +4000 (54000). Dia 2: suelo = 54000-2500 = 51500 (NO 47500).
    # Cierra en 51000: por encima del suelo estatico 45000 y del suelo "desde
    # inicial" 47500, pero POR DEBAJO del suelo dinamico 51500 -> fail.
    ba = _bda([1, 2], [4000.0, 1000.0])
    assert _phase_eval(ba).outcome == "fail"


def test_max_loss_estatico_breach_aunque_daily_no_ate():
    # Tres dias de -2000 cerrados: dia1 48000 OK, dia2 46000 (suelo 45500) OK,
    # dia3: suelo dinamico 43500 (no ata) pero 44000 <= 45000 estatico -> fail.
    ba = _bda([1, 2, 3], [-2000.0, -4000.0, -6000.0])
    assert _phase_eval(ba).outcome == "fail"
    # Variante: mismo dia3 cerrando -500 (-6500 total)... 43500 < 43500 suelo -> tambien
    # falla por daily; comprobamos que el estatico ata ANTES con cierre -3000 (44000):
    ba2 = _bda([1, 2, 3], [-2000.0, -4000.0, -5500.0])
    assert _phase_eval(ba2).outcome == "fail"


# ---------------------------------------------------------------------------
# Min trading days + continuacion post-target (P3 de la revision)
# ---------------------------------------------------------------------------

def test_target_antes_de_min_days_no_es_pass_hasta_cubrir_dias():
    # +6000 (target +5000) en dia 2 con solo 2 dias operados; pass en dia 4.
    ba = _bda([1, 2, 3, 4], [3000.0, 6000.0, 6000.0, 6000.0])
    r = _phase_eval(ba)
    assert r.outcome == "pass"
    assert r.trading_days == 4
    assert r.days == 4.0


def test_breach_en_continuacion_post_target_es_fail():
    # Target en dia 2 (56000). Dia 3: suelo 56000-2500=53500; cierra 53000 -> fail.
    ba = _bda([1, 2, 3, 4], [3000.0, 6000.0, 3000.0, 3000.0])
    assert _phase_eval(ba).outcome == "fail"


def test_sin_target_ni_breach_es_neither():
    ba = _bda([1, 2, 3, 4, 5], [500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    assert _phase_eval(ba).outcome == "neither"


# ---------------------------------------------------------------------------
# Fondeada: payout, reset, breach, cap
# ---------------------------------------------------------------------------

FUNDED = FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0, max_payout_cycles=3)


def _funded_eval(ba, funded=FUNDED, split=0.8):
    return evaluate_funded_path(
        np.array([0]), [ba], np.array([30.0]),
        account_size=ACC, phase=PHASE, funded=funded, profit_split=split,
    )


def test_funded_payout_y_reset_a_baseline():
    # Dia 14: +6000 -> payout 0.8*6000 = 4800 y reset. Dia 28: +2000 desde el
    # reset -> payout 1600. Ciclo 3 sin mas dias -> 0.
    ba = _bda([1, 14, 28], [2000.0, 6000.0, 8000.0])
    r = _funded_eval(ba)
    assert r.cycle_payouts[0] == pytest.approx(4800.0)
    assert r.cycle_payouts[1] == pytest.approx(1600.0)
    assert r.cycle_payouts[2] == 0.0
    assert r.days_to_first_payout == pytest.approx(14.0)
    assert not r.alive_at_cap


def test_funded_ciclo_negativo_no_paga_ni_resetea():
    # Ciclo 1 cierra -1000: sin payout y la equity continua (no reset).
    # Ciclo 2 recupera +3000 neto -> payout sobre profit total al cierre del
    # ciclo 2: equity = 50000 -1000 +3000 = 52000 -> payout 0.8*2000 = 1600.
    ba = _bda([1, 14, 28], [-1000.0, -1000.0, 2000.0])
    r = _funded_eval(ba)
    assert r.cycle_payouts[0] == 0.0
    assert r.cycle_payouts[1] == pytest.approx(1600.0)


def test_funded_breach_para_la_cuenta():
    ba = _bda([1], [-6000.0])  # 44000 <= 45000 estatico
    r = _funded_eval(ba)
    assert r.cycle_payouts.sum() == 0.0
    assert np.isnan(r.days_to_first_payout)
    assert not r.alive_at_cap


def test_funded_alive_at_cap_si_sobrevive_todos_los_ciclos():
    # Dias 1..60 ganando poco pero sin breach: completa los 3 ciclos.
    days = list(range(1, 61, 3))
    ba = _bda(days, [500.0 * (i + 1) for i in range(len(days))])
    r = _funded_eval(ba)
    assert r.alive_at_cap
    assert r.cycle_payouts[0] > 0


# ---------------------------------------------------------------------------
# Nivel 2: EV con distribuciones degeneradas
# ---------------------------------------------------------------------------

def _outcomes(pass1, pass2, payouts, n=100, neither1=False):
    pass1 = np.full(n, pass1, dtype=bool)
    pass2 = np.full(n, pass2, dtype=bool)
    ne1 = np.zeros(n, dtype=bool)
    if neither1:
        ne1[n // 2:] = True
        pass1[n // 2:] = False
    funded = np.zeros((n, 3))
    funded[:, 0] = payouts
    return FirmPathOutcomes(
        pass1=pass1, pass2=pass2, neither1=ne1, neither2=np.zeros(n, dtype=bool),
        funded_payouts=funded,
        days_first_payout=np.where(np.asarray(payouts) > 0, np.full(n, 14.0), np.full(n, np.nan)),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.where(pass1, 10.0, np.nan),
    )


def _cfg(refund=True):
    from src.risk.propfirm import PropFirmConfig
    return PropFirmConfig(
        key="t", display_name="t", account_size=ACC, fee=345.0,
        profit_split_pct=80.0, refund_fee_on_first_payout=refund,
        phases=(PHASE,), funded=FUNDED,
    )


def test_sequence_ev_degenerado_todo_pasa():
    # P=1, payout fijo 1000 USD, fx=1, refund True -> EV = -345 + 1000 + 345 = 1000.
    cfg = _cfg(refund=True)
    res = sequence_ev(_outcomes(True, True, 1000.0), cfg, fx_eur_usd=1.0,
                      n_challenges=10, n_sequences=200, initial_capital_eur=5000.0, seed=1)
    assert res.ev_per_attempt == pytest.approx(1000.0)
    assert res.net_mean == pytest.approx(10_000.0)
    assert res.p_net_negative == 0.0


def test_sequence_ev_degenerado_nada_pasa():
    cfg = _cfg(refund=True)
    res = sequence_ev(_outcomes(False, True, 1000.0), cfg, fx_eur_usd=1.0,
                      n_challenges=10, n_sequences=200, initial_capital_eur=5000.0, seed=1)
    assert res.ev_per_attempt == pytest.approx(-345.0)
    assert res.p_net_negative == 1.0


def test_refund_solo_si_hay_primer_payout():
    # Pasa fases pero la fondeada no produce payout -> NO hay refund.
    cfg = _cfg(refund=True)
    res = sequence_ev(_outcomes(True, True, 0.0), cfg, fx_eur_usd=1.0,
                      n_challenges=10, n_sequences=200, initial_capital_eur=5000.0, seed=1)
    assert res.ev_per_attempt == pytest.approx(-345.0)
    # Con refund desactivado y payout 1000 -> EV = 655 (sin +345).
    cfg2 = _cfg(refund=False)
    res2 = sequence_ev(_outcomes(True, True, 1000.0), cfg2, fx_eur_usd=1.0,
                       n_challenges=10, n_sequences=200, initial_capital_eur=5000.0, seed=1)
    assert res2.ev_per_attempt == pytest.approx(655.0)


def test_sequence_ev_convierte_fx():
    cfg = _cfg(refund=False)
    res = sequence_ev(_outcomes(True, True, 1000.0), cfg, fx_eur_usd=2.0,
                      n_challenges=5, n_sequences=100, initial_capital_eur=5000.0, seed=1)
    # 1000 USD / 2.0 = 500 EUR - 345 fee = 155.
    assert res.ev_per_attempt == pytest.approx(155.0)


# ---------------------------------------------------------------------------
# YAML + recommend
# ---------------------------------------------------------------------------

def test_yaml_ftmo_carga_y_valida():
    cfgs = load_propfirm_config()
    ftmo = cfgs["ftmo_50k"]
    assert ftmo.fee == 345.0
    assert ftmo.account_size == 50_000.0
    assert [p.profit_target_pct for p in ftmo.phases] == [10.0, 5.0]
    assert all(p.max_daily_loss_pct == 5.0 and p.max_loss_pct == 10.0 for p in ftmo.phases)
    assert all(p.min_trading_days == 4 for p in ftmo.phases)
    assert ftmo.funded.max_payout_cycles == 50
    assert cfgs["ftmo_100k"].account_size == 100_000.0


def _report(f, ev, p_neg):
    return FirmReport(
        f=f, n_paths=100, p_pass1=0.5, p_pass1_ci=(0.4, 0.6),
        p_both=0.3, p_both_ci=(0.2, 0.4), p_unresolved=0.0,
        ev_per_attempt=ev, ev_ci95=(ev - 10, ev + 10), net_mean=ev * 10,
        p_net_negative=p_neg,
        net_p05=-1, net_p50=0, net_p95=1, mean_payout_eur=1000.0,
        p_payout_given_funded=0.8, challenges_per_payout=3.0,
        median_days_first_payout=30.0, median_days_pass1=15.0,
        alive_at_cap_frac=0.1, seq=None,
    )


def test_recommend_f_ev_respeta_tolerancia_p_negativo():
    reps = [_report(0.01, 100.0, 0.4), _report(0.02, 200.0, 0.6)]
    rec = recommend_f_ev(reps, tol_p_negative=0.5)
    assert rec.f == 0.01 and rec.feasible
    rec2 = recommend_f_ev(reps, tol_p_negative=0.7)
    assert rec2.f == 0.02 and rec2.feasible


def test_recommend_f_ev_fallback_sin_factible():
    reps = [_report(0.01, 100.0, 0.6), _report(0.02, 200.0, 0.8)]
    rec = recommend_f_ev(reps, tol_p_negative=0.5)
    assert not rec.feasible and rec.f == 0.01


# ---------------------------------------------------------------------------
# CRITICO-1 (revision de codigo): gaps empiricos en elapsed
# ---------------------------------------------------------------------------

PHASE_D1 = PropFirmPhase("T", 10.0, 5.0, 10.0, 1)


def test_gaps_empiricos_suman_al_elapsed_de_fase():
    # Dos bloques separados; el gap flat entre ellos debe contar en los dias.
    ba0 = _bda([1], [1000.0])
    ba1 = _bda([50], [5000.0])  # cierra 56000 >= target 55000 -> pass
    order = np.array([0, 1])
    dur = np.array([1.0, 1.0])
    sin = evaluate_phase_path(order, [ba0, ba1], dur,
                              account_size=ACC, phase=PHASE_D1)
    con = evaluate_phase_path(order, [ba0, ba1], dur, np.array([20.0, 0.0]),
                              account_size=ACC, phase=PHASE_D1)
    assert sin.outcome == "pass" and con.outcome == "pass"
    assert sin.days == 2.0
    assert con.days == 22.0  # +20 dias flat entre episodios


def test_gaps_hacen_tictac_los_ciclos_de_payout():
    # Ciclo de 14 dias calendario: sin gaps el ciclo no vence en 2 dias
    # comprimidos; con un gap de 20 dias si (los ciclos son CALENDARIO real).
    ba0 = _bda([1], [6000.0])
    ba1 = _bda([50], [6000.0])
    order = np.array([0, 1])
    dur = np.array([1.0, 1.0])
    kw = dict(account_size=ACC, phase=PHASE, funded=FUNDED, profit_split=0.8)
    sin = evaluate_funded_path(order, [ba0, ba1], dur, **kw)
    con = evaluate_funded_path(order, [ba0, ba1], dur, np.array([20.0, 0.0]), **kw)
    assert sin.cycle_payouts[0] == 0.0
    # elapsed=21 -> ciclo vence: equity 62000 -> profit 12000 -> payout 9600.
    assert con.cycle_payouts[0] == pytest.approx(9600.0)


def test_multi_ciclo_en_un_dia_solo_un_payout():
    # Un salto de elapsed que cruza 2 ciclos: el profit se cobra UNA vez;
    # el segundo ciclo vencido ese dia no ve profit (ya retirado).
    ba0 = _bda([1], [10000.0])
    ba1 = _bda([60], [0.0])
    order = np.array([0, 1])
    dur = np.array([1.0, 1.0])
    r = evaluate_funded_path(order, [ba0, ba1], dur, np.array([30.0, 0.0]),
                             account_size=ACC, phase=PHASE, funded=FUNDED,
                             profit_split=0.8)
    assert r.cycle_payouts[0] == pytest.approx(8000.0)  # 0.8 * 10000
    assert r.cycle_payouts[1] == 0.0


# ---------------------------------------------------------------------------
# CRITICO-2 (revision de codigo): neither excluido del EV + IC del EV
# ---------------------------------------------------------------------------

def test_neither_excluido_del_ev():
    # 50% paths resueltos (pass, payout 1000) + 50% neither: el EV se calcula
    # solo sobre resueltos = -345 + 1000 = 655 (no 155 como antes del fix).
    cfg = _cfg(refund=False)
    res = sequence_ev(_outcomes(True, True, 1000.0, neither1=True), cfg,
                      fx_eur_usd=1.0, n_challenges=10, n_sequences=200,
                      initial_capital_eur=5000.0, seed=1)
    assert res.ev_per_attempt == pytest.approx(655.0)
    assert res.p_unresolved == pytest.approx(0.5)


def test_todo_neither_lanza_error():
    cfg = _cfg(refund=False)
    oc = _outcomes(False, True, 0.0, neither1=True)
    oc.neither1[:] = True
    with pytest.raises(ValueError, match="[Nn]ingun path"):
        sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=5, n_sequences=10,
                    initial_capital_eur=1000.0, seed=1)


def test_ev_ci95_contiene_el_puntual():
    cfg = _cfg(refund=False)
    res = sequence_ev(_outcomes(True, True, 1000.0), cfg, fx_eur_usd=1.0,
                      n_challenges=5, n_sequences=50,
                      initial_capital_eur=5000.0, seed=1)
    lo, hi = res.ev_ci95
    assert lo <= res.ev_per_attempt <= hi



# ---------------------------------------------------------------------------
# Stop por insolvencia (bankroll < fee) — n_challenges acoplado al presupuesto
# ---------------------------------------------------------------------------

def test_stop_insolvencia_todo_falla_deteccion_exacta():
    # Todos los paths fallan: cada intento cuesta el fee y ninguno cobra.
    # capital0=1000, fee=345 -> se pagan 2 intentos (bankroll 1000, 655) y el
    # 3º (310 < 345) ya no se puede pagar: stop, neto = -690.
    cfg = _cfg(refund=False)
    oc = _outcomes(False, True, 0.0)
    res = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=200,
                      initial_capital_eur=1000.0, seed=1, stop_on_insolvency=True)
    assert res.mean_attempts == pytest.approx(2.0)
    assert res.p_ruin == 1.0
    assert res.net_mean == pytest.approx(-690.0)
    assert res.p_net_negative == 1.0
    # Trayectoria plana tras el stop (bankroll 310 congelado).
    assert res.trajectory_p50[3] == pytest.approx(310.0)
    assert res.trajectory_p50[-1] == pytest.approx(310.0)


def test_stop_insolvencia_los_payouts_alargan_el_bankroll():
    # 50% paths pasan con payout 1000, 50% fallan: las trayectorias que ganan
    # reponen bankroll y superan el minimo garantizado floor(1000/345)=2.
    cfg = _cfg(refund=False)
    n = 200
    pass1 = np.zeros(n, dtype=bool)
    pass1[: n // 2] = True
    funded = np.zeros((n, 3))
    funded[: n // 2, 0] = 1000.0
    oc = FirmPathOutcomes(
        pass1=pass1, pass2=np.ones(n, dtype=bool),
        neither1=np.zeros(n, dtype=bool), neither2=np.zeros(n, dtype=bool),
        funded_payouts=funded,
        days_first_payout=np.where(pass1, 14.0, np.nan),
        alive_at_cap=np.zeros(n, dtype=bool),
        days_to_pass1=np.where(pass1, 10.0, np.nan),
    )
    res = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=500,
                      initial_capital_eur=1000.0, seed=3, stop_on_insolvency=True)
    # Quiebra = los dos primeros intentos fallan (P ~ 0.25): algunas caen, no todas.
    assert 0.05 < res.p_ruin < 0.6
    assert res.mean_attempts > 2.0  # los payouts alargan mas alla del minimo
    assert res.mean_attempts <= 10.0


def test_stop_insolvencia_capital_menor_que_fee_no_hay_intentos():
    cfg = _cfg(refund=False)
    oc = _outcomes(False, True, 0.0)
    res = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=50,
                      initial_capital_eur=100.0, seed=1, stop_on_insolvency=True)
    assert res.mean_attempts == 0.0
    assert res.p_ruin == 1.0
    assert res.net_mean == 0.0


def test_stop_insolvencia_fee_cero_no_aplica():
    import dataclasses
    cfg = dataclasses.replace(_cfg(refund=False), fee=0.0)
    oc = _outcomes(False, True, 0.0)
    res = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=50,
                      initial_capital_eur=1000.0, seed=1, stop_on_insolvency=True)
    assert np.isnan(res.p_ruin) and np.isnan(res.mean_attempts)


def test_sin_stop_comportamiento_legacy_intacto():
    # stop_on_insolvency=False (default): mismo neto que siempre (sin barrera).
    cfg = _cfg(refund=False)
    oc = _outcomes(False, True, 0.0)
    res = sequence_ev(oc, cfg, fx_eur_usd=1.0, n_challenges=10, n_sequences=200,
                      initial_capital_eur=1000.0, seed=1)
    assert res.net_mean == pytest.approx(-3450.0)  # 10 intentos x -345
    assert np.isnan(res.p_ruin) and np.isnan(res.mean_attempts)
