"""Tests golden de la máquina de estados de la PA Apex (plan Apex Fase 4).

Todo calculado a mano sobre arrays sintéticos construidos directamente (sin
M1). Cubre: congelado del safety net (suelo fijo 150,100 para siempre y el
trailing deja de apretar), DLL pausa no mortal (clamp + sigue; precedencia del
trailing), invariante del check DLL con rel_low (subsume el solo-realizado),
downgrade de tier tras retiro, días cualificados (300 sí / 299.99 no / clamp
nunca), consistencia estricta (0.50 exacto denegado, 0.4999 pasa, deferral con
qual_days conservados, reset de ventana tras retiro), escalera de retiros
(topes, min $500, primer retiro ≥ 154,600, residual ≥ 154,100, cierre tras el
6º), re-anclaje del trailing post-payout vía saldo, inactividad rolling de 30
naturales (incl. gaps sin trades), integración multi-retiro con calendario de
payouts exacto, dispatch (cycle FTMO byte-idéntico; ladder nunca hereda
phase1) y end-to-end con sweep_funded + diagnósticos agregados.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.risk.propfirm import (
    BlockDayArrays,
    ChallengeOutcomes,
    ConsistencySpec,
    DailyLossSpec,
    FundedRules,
    InactivitySpec,
    LadderPayouts,
    PropFirmConfig,
    PropFirmPhase,
    ScalingTier,
    TrailingDrawdown,
    build_phase_tape,
    combine_challenge_funded,
    evaluate_funded_path,
    firm_report,
    load_propfirm_config,
    resolve_fee_eur,
    sequence_ev,
    sweep_funded,
)
from src.risk.propfirm_ladder import evaluate_funded_path_ladder

ACC = 150_000.0

APEX_FUNDED = FundedRules(
    payout_cycle_days=0,
    min_payout_profit_pct=0.0,
    max_payout_cycles=50,
    model="ladder",
    drawdown=TrailingDrawdown(
        amount_usd=4000.0, lock_at_balance=154100.0, frozen_floor=150100.0
    ),
    daily_loss=DailyLossSpec(
        type="pause_not_fatal",
        by_tier_usd=(2500.0, 2500.0, 2500.0, 3000.0, 4000.0),
        post_hit="clamp",
    ),
    scaling_tiers=(
        ScalingTier(1, 0.0, 1999.0, 4, 40),
        ScalingTier(2, 2000.0, 2999.0, 5, 50),
        ScalingTier(3, 3000.0, 4999.0, 7, 70),
        ScalingTier(4, 5000.0, 9999.0, 10, 100),
        ScalingTier(5, 10000.0, None, 10, 100),
    ),
    payouts=LadderPayouts(
        qualifying_day_min_usd=300.0,
        days_per_payout=5,
        ladder_usd=(2500.0, 3000.0, 3000.0, 4000.0, 4000.0, 5000.0),
        min_payout_usd=500.0,
        min_balance_first=154600.0,
        residual_floor=154100.0,
        max_payouts=6,
        consistency=ConsistencySpec(0.50, True, "since_last_payout"),
    ),
    inactivity=InactivitySpec(window_days=30, min_days=2, min_profit_usd=50.0),
)


def _lbda(days, rel_closed, rel_low=None, rel_high=None, rel_exc=None):
    """BlockDayArrays sintético con trailing, a mano.

    Default = día monótono alcista: rel_low = rel_before (sin dip), rel_high =
    rel_closed (sin pico extra), rel_exc = 0. Precondiciones del constructor
    real: rel_low[d] <= rel_closed[d], rel_high[d] >= rel_before[d] y
    rel_exc[d] con cummax de bloque.
    """
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_before = np.concatenate([[0.0], rel_closed[:-1]])
    n = len(days)
    rel_low = rel_before.copy() if rel_low is None else np.asarray(rel_low, dtype=float)
    rel_high = rel_closed.copy() if rel_high is None else np.asarray(rel_high, dtype=float)
    rel_exc = np.zeros(n) if rel_exc is None else np.asarray(rel_exc, dtype=float)
    return BlockDayArrays(
        day_ord=days,
        rel_before=rel_before,
        rel_closed=rel_closed,
        rel_float=rel_before.copy(),
        opens_day=np.ones(n, dtype=int),
        total_pnl=float(rel_closed[-1]),
        rel_low=rel_low,
        rel_high=rel_high,
        rel_exc=rel_exc,
        block_low=float(rel_low.min()),
        block_high=float(rel_high.max()),
        block_exc=float(rel_exc.max()),
    )


def _eval_l(bas, funded=APEX_FUNDED, durations=None, gaps=None, split=1.0):
    n = len(bas)
    return evaluate_funded_path_ladder(
        np.arange(n), bas,
        np.ones(n) if durations is None else np.asarray(durations, dtype=float),
        gaps,
        cfg_funded=funded, account_size=ACC, profit_split=split,
    )


def _assert_dead(r):
    assert not r.alive_at_cap and not r.closed_after_ladder


# ---------------------------------------------------------------------------
# Congelado del safety net
# ---------------------------------------------------------------------------

def _freeze_block():
    """d1 pico +4,500 (cruza 154,100 → congela); d2/d3 con excursiones >= 4,000
    que SIN congelado matarían; d4 toca 150,100 → muerte."""
    return _lbda(
        [1, 2, 3, 4],
        [4500.0, 1000.0, 300.0, 250.0],
        rel_low=[4500.0, 500.0, 200.0, 100.0],
        rel_high=[4500.0, 4600.0, 1100.0, 350.0],
        rel_exc=[0.0, 4100.0, 4400.0, 4500.0],
    )


def test_congelado_suelo_fijo_y_trailing_deja_de_apretar():
    funded = replace(APEX_FUNDED, daily_loss=None)  # aisla el trailing
    r = _eval_l([_freeze_block()], funded=funded)
    # d2: equity baja a 150,500 con excursión 4,100 (≥ amount) — congelado ⇒ el
    # trailing NO aprieta y sobrevive; d3 ídem (exc 4,400); d4 toca el suelo
    # fijo 150,100 → muerte. El suelo no resetea ni retrocede jamás.
    _assert_dead(r)
    assert r.days == 4.0
    assert r.dll_hits == 0
    assert not r.inactivity_death
    assert np.all(r.payouts == 0.0)
    assert np.all(np.isnan(r.payout_days))


def test_sin_congelado_la_excursion_de_d2_mata():
    # Contraejemplo: mismo path sin lock_at_balance → sin congelado; la
    # excursión de d2 (4,100 ≥ 4,000) liquida en el día 2.
    funded = replace(APEX_FUNDED, daily_loss=None,
                     drawdown=TrailingDrawdown(amount_usd=4000.0))
    r = _eval_l([_freeze_block()], funded=funded)
    _assert_dead(r)
    assert r.days == 2.0


# ---------------------------------------------------------------------------
# DLL: pausa no mortal, clamp, precedencia del trailing
# ---------------------------------------------------------------------------

def test_dll_hit_pausa_no_mortal_y_la_cuenta_sigue():
    # d1: valle −3,100 ≤ −2,500 (tier 1) → hit; cierra −3,000 (peor que −dll:
    # adj = −3,000 + 2,500 = −500 ≤ 0 ⇒ SIN clamp, se conserva el cierre real —
    # la liquidación puede ir con slippage). d2: +400 (cualifica) y sin hit.
    ba = _lbda(
        [1, 2],
        [-3000.0, -2600.0],
        rel_low=[-3100.0, -3050.0],
        rel_high=[0.0, -2600.0],
        rel_exc=[3100.0, 3050.0],
    )
    r = _eval_l([ba])
    assert r.alive_at_cap
    assert r.days == 2.0
    assert r.dll_hits == 1
    assert np.all(r.payouts == 0.0)
    assert np.isnan(r.days_to_first_payout)


def test_trailing_tiene_precedencia_sobre_dll():
    # Valle −4,500: el trailing liquida (4,500 ≥ 4,000) ANTES de evaluar el DLL
    # ⇒ dll_hits NO se incrementa aunque el día también rompiera el DLL.
    ba = _lbda([1], [-4500.0], rel_low=[-4500.0], rel_high=[0.0],
               rel_exc=[4500.0])
    r = _eval_l([ba])
    _assert_dead(r)
    assert r.days == 1.0
    assert r.dll_hits == 0
    assert not r.inactivity_death


def test_dll_invariante_rel_low_subsume_solo_realizado_y_clamp():
    # DLL pequeño (100) para aislar. Invariante: rel_low[d] <= rel_closed[d] ⇒
    # el check `rel_low − rel_before ≤ −dll` subsume el caso solo-realizado.
    # d1: sin valle intradía (rel_low == rel_closed == −100) → hit en el borde.
    # d2: valle −210 (hit) pero recupera a −50 → clamp: neto sesión = −100
    #     (adj = 50 + 100 = 150 > 0 ⇒ saldo −= 150).
    # d3: valle −70 de sesión (> −100) → NO hit.
    funded = replace(
        APEX_FUNDED,
        daily_loss=DailyLossSpec("pause_not_fatal", (100.0,) * 5, post_hit="clamp"),
    )
    ba = _lbda(
        [1, 2, 3],
        [-100.0, -50.0, -30.0],
        rel_low=[-100.0, -210.0, -120.0],
        rel_high=[0.0, -50.0, -30.0],
        rel_exc=[100.0, 210.0, 120.0],
    )
    r = _eval_l([ba], funded=funded)
    assert r.alive_at_cap
    assert r.dll_hits == 2
    # El día con clamp queda en −100 < 300: nunca cualifica (estructural).
    assert np.all(r.payouts == 0.0)


# ---------------------------------------------------------------------------
# Tier: downgrade tras retiro (el rango se lee del balance post-payout)
# ---------------------------------------------------------------------------

def test_downgrade_tier_tras_retiro_baja_el_dll():
    # d1–d5: +1,400/día → beneficio 7,000 (tier 4, DLL 3,000). d5 paga
    # min(2,500; 157,000−154,100) = 2,500 → beneficio residual 4,500 → tier 3
    # (DLL 2,500) para d6. d6 pierde 2,700: hit PORQUE el tier bajó (con tier 4
    # no sería hit: 2,700 < 3,000).
    ba = _lbda(
        [1, 2, 3, 4, 5, 6, 7],
        [1400.0, 2800.0, 4200.0, 5600.0, 7000.0, 4300.0, 4700.0],
        rel_low=[1400.0, 2800.0, 4200.0, 5600.0, 7000.0, 4300.0, 4300.0],
        rel_high=[1400.0, 2800.0, 4200.0, 5600.0, 7000.0, 7000.0, 4700.0],
        rel_exc=[0.0, 0.0, 0.0, 0.0, 0.0, 2700.0, 2700.0],
    )
    r = _eval_l([ba])
    assert r.alive_at_cap
    assert r.payouts[0] == pytest.approx(2500.0)
    assert r.days_to_first_payout == 5.0
    assert r.dll_hits == 1  # solo d6, y solo porque el tier degradó a 3


# ---------------------------------------------------------------------------
# Días cualificados: 300 sí, 299.99 no; día con clamp no cualifica
# ---------------------------------------------------------------------------

def test_cualificado_borde_300_si_299_99_no():
    # Gates bajadas para que el payout lo gobierne qual_days (válidas:
    # 150,600 ≥ 150,100 + 500). d1 = +299.99 NO cualifica; d2–d6 = +300 → el
    # 5º cualificado es d6 (si 299.99 contara, pagaría en d5 con 1,399.99).
    funded = replace(
        APEX_FUNDED,
        payouts=replace(APEX_FUNDED.payouts, min_balance_first=150600.0,
                        residual_floor=150100.0),
    )
    closes = np.cumsum([299.99] + [300.0] * 5)
    r = _eval_l([_lbda(np.arange(1, 7), closes)], funded=funded)
    assert r.days_to_first_payout == 6.0
    assert r.payout_days[0] == 6.0
    # min(2,500; 151,799.99 − 150,100) = 1,699.99; consistencia 300/1,799.99 ok.
    assert r.payouts[0] == pytest.approx(1699.99)
    assert r.consistency_deferrals == 0


def test_dia_con_clamp_no_cualifica_y_el_clamp_entra_en_saldo():
    # d1: valle −2,600 (hit DLL tier 1) pero recupera y cierra +1,000 → clamp:
    # neto sesión = −2,500 (adj = 1,000 + 2,500 = 3,500 ⇒ saldo −= 3,500).
    # Luego +700/día. Balance(d) = 146,500 + rel_closed[d]: cruza 154,600 en
    # d12 (rel_closed 8,700 → 155,200) ⇒ primer retiro min(2,500; 1,100) =
    # 1,100 en d12. SIN el clamp el balance cruzaría en d7 — el timing fija que
    # el ajuste entró en saldo; y el día clampleado (−2,500 < 300) no suma
    # cualificado (los 11 cualificados son d2–d12).
    ba = _lbda(
        np.arange(1, 13),
        np.concatenate([[1000.0], 1000.0 + 700.0 * np.arange(1, 12)]),
        rel_low=np.concatenate([[-2600.0], 1000.0 + 700.0 * np.arange(0, 11)]),
        rel_high=np.concatenate([[1000.0], 1000.0 + 700.0 * np.arange(1, 12)]),
        # cummax(upper) y lower suben a la par tras d1 ⇒ excursión constante 700.
        rel_exc=np.concatenate([[3600.0], np.full(11, 700.0)]),
    )
    r = _eval_l([ba])
    assert r.dll_hits == 1
    assert r.days_to_first_payout == 12.0
    assert r.payouts[0] == pytest.approx(1100.0)
    assert r.consistency_deferrals == 0


# ---------------------------------------------------------------------------
# Consistencia estricta: 0.50 exacto denegado, 0.4999 pasa, deferral, reset
# ---------------------------------------------------------------------------

def _consistency_block_a():
    """Sesiones [3,000, 750×9]: d5 acumula 6,000 con mejor día 3,000 (0.50
    EXACTO → denegado); d6 diluye a 0.444 → paga."""
    return _lbda(np.arange(1, 7), 3000.0 + 750.0 * np.arange(6))


def test_consistencia_share_exactamente_050_es_denegado():
    r = _eval_l([_consistency_block_a()])
    # d5: share = 3,000/6,000 = 0.5 exacto → strict ⇒ denegado (deferral).
    # d6: 3,000/6,750 = 0.444 < 0.5 → paga min(2,500; 156,750−154,100) = 2,500.
    assert r.consistency_deferrals == 1
    assert r.days_to_first_payout == 6.0
    assert r.payouts[0] == pytest.approx(2500.0)


def test_consistencia_04999_pasa():
    sessions = [2999.40, 750.15, 750.15, 750.15, 750.15]
    r = _eval_l([_lbda(np.arange(1, 6), np.cumsum(sessions))])
    # share = 2,999.40/6,000.00 = 0.4999 < 0.5 → paga en d5.
    assert r.consistency_deferrals == 0
    assert r.days_to_first_payout == 5.0
    assert r.payouts[0] == pytest.approx(1900.0)


def test_consistencia_deferral_conserva_quals_y_reset_tras_retiro():
    # Extiende A: tras pagar en d6, la ventana se RESETEA. d7 +2,000 y luego
    # +300/día: 5º cualificado en d11 pero share 2,000/3,200 = 0.625 → deferral
    # (y d12, d13); d14: 2,000/4,100 = 0.4878 → paga el 2º retiro. Si la
    # ventana NO se reseteara tras el retiro de d6, el share con los días 1–6
    # incluidos (3,000/9,950 = 0.30) pagaría en d11 sin deferrals extra.
    closes = np.concatenate([
        3000.0 + 750.0 * np.arange(6),          # d1–d6 (bloque A)
        [8750.0],                                # d7: +2,000
        8750.0 + 300.0 * np.arange(1, 8),        # d8–d14: +300/día
    ])
    r = _eval_l([_lbda(np.arange(1, 15), closes)])
    assert r.payouts[0] == pytest.approx(2500.0)
    assert r.payout_days[0] == 6.0
    assert r.consistency_deferrals == 4  # d5 (0.50 exacto) + d11, d12, d13
    # 2º retiro: min(ladder[1]=3,000; 158,350−154,100=4,250) = 3,000 en d14.
    assert r.payouts[1] == pytest.approx(3000.0)
    assert r.payout_days[1] == 14.0
    assert r.alive_at_cap
    assert not r.closed_after_ladder


# ---------------------------------------------------------------------------
# Escalera de retiros: topes, min $500, gates, residual, cierre tras el 6º
# ---------------------------------------------------------------------------

def _ladder_block_30d():
    """30 sesiones de +1,000/día (cualificadas, consistencia 0.2, sin dips)."""
    return _lbda(np.arange(1, 31), 1000.0 * np.arange(1, 31))


def test_escalera_completa_topes_y_cierre_tras_sexto():
    r = _eval_l([_ladder_block_30d()])
    # d5: balance 155,000 ≥ 154,600 → min(2,500; 900) = 900 (tope withdrawable,
    # residual queda EXACTAMENTE en 154,100). Luego +5,000 por ciclo:
    # d10: min(3,000; 5,000) = 3,000; d15: 3,000; d20: min(4,000; 9,000) =
    # 4,000; d25: 4,000; d30: min(5,000; 11,000) = 5,000 → cierre terminal.
    np.testing.assert_allclose(
        r.payouts[:6], [900.0, 3000.0, 3000.0, 4000.0, 4000.0, 5000.0])
    assert np.all(r.payouts[6:] == 0.0)
    np.testing.assert_allclose(
        r.payout_days, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    assert r.days_to_first_payout == 5.0
    assert r.closed_after_ladder
    assert not r.alive_at_cap
    assert r.days == 30.0
    assert r.dll_hits == 0
    assert r.consistency_deferrals == 0
    assert not r.inactivity_death
    assert r.payouts.sum() == pytest.approx(19_900.0)


def test_primer_retiro_exige_balance_154600():
    # +910/día: d5 balance 154,550 < 154,600 → NO paga aunque hay 5
    # cualificados; d6: 155,460 → min(2,500; 1,360) = 1,360.
    r = _eval_l([_lbda(np.arange(1, 7), 910.0 * np.arange(1, 7))])
    assert r.days_to_first_payout == 6.0
    assert r.payouts[0] == pytest.approx(1360.0)
    assert r.payout_days[0] == 6.0


def test_min_payout_500_gate():
    # Consistencia laxa (0.95 no estricta) para que no subsuma el min-gate.
    # Tras P1 (900, residual exacto 154,100), la ventana [300, 300, −1,101,
    # 300, 300, 300] llega a 5 cualificados en d11 con retirable 399 < 500 → NO
    # paga (y NO cuenta como deferral de consistencia: el share 0.752 pasa);
    # d12: retirable 699 → min(3,000; 699) = 699.
    funded = replace(
        APEX_FUNDED,
        payouts=replace(APEX_FUNDED.payouts,
                        consistency=ConsistencySpec(0.95, False, "since_last_payout")),
    )
    sessions = [1000.0] * 5 + [300.0, 300.0, -1101.0, 300.0, 300.0, 300.0, 300.0]
    closes = np.cumsum(sessions)
    rel_low = np.concatenate([[0.0], closes[:-1]])
    rel_low[7] = closes[7]  # d8 es monótona bajista: low == close
    ba = _lbda(np.arange(1, 13), closes, rel_low=rel_low)
    r = _eval_l([ba], funded=funded)
    assert r.payouts[0] == pytest.approx(900.0)
    assert r.payouts[1] == pytest.approx(699.0)
    assert r.payout_days[1] == 12.0
    assert r.consistency_deferrals == 0
    assert r.dll_hits == 0  # d8 (−1,101) nunca rompe el DLL de 2,500


# ---------------------------------------------------------------------------
# Re-anclaje post-payout: el trailing se evalúa con el saldo ajustado
# ---------------------------------------------------------------------------

def test_post_payout_trailing_reanclado_via_saldo():
    # d1–d5 +1,000 → P1 = 900 en d5 ⇒ saldo −= 900 (balance 154,100, congelado).
    # d8: rel_low 900 ⇒ equity_low = 150,000 − 900 + 900 = 150,000 ≤ 150,100 →
    # muerte. SIN el re-anclaje (saldo sin el payout) sería 150,900 y el modelo
    # se creería más rico de lo que está (anti-conservador que el mecanismo
    # saldo corrige). Además d8 rompería el DLL (valle de sesión −2,900) pero
    # el trailing mata primero ⇒ dll_hits se queda en 0 (precedencia).
    ba = _lbda(
        np.arange(1, 9),
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 4500.0, 3800.0, 950.0],
        rel_low=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 4100.0, 3700.0, 900.0],
        rel_high=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 5000.0, 4500.0, 3800.0],
        rel_exc=[0.0, 0.0, 0.0, 0.0, 0.0, 900.0, 1300.0, 4100.0],
    )
    r = _eval_l([ba])
    _assert_dead(r)
    assert r.days == 8.0
    assert r.payouts[0] == pytest.approx(900.0)
    assert r.days_to_first_payout == 5.0
    assert r.dll_hits == 0
    assert not r.inactivity_death


# ---------------------------------------------------------------------------
# Inactividad: ventana rolling de 30 naturales, ≥2 días con ≥ $50
# ---------------------------------------------------------------------------

def test_inactividad_muere_al_cumplirse_una_ventana_completa_insuficiente():
    # 31 sesiones diarias: d1 +60 (activo), d2–d31 +10 (inactivos). Al abrir la
    # sesión 31 (s = 30) la ventana (0, 30] tiene 1 día activo < 2 → muerte
    # ANTES de operar d31. Sin grace period (supuesto documentado).
    closes = np.cumsum([60.0] + [10.0] * 30)
    r = _eval_l([_lbda(np.arange(1, 32), closes)])
    _assert_dead(r)
    assert r.inactivity_death
    assert r.days == 30.0
    assert r.dll_hits == 0


def test_inactividad_sobrevive_con_actividad_minima():
    # +60 en d1, d16, d31 (activos); +10 el resto. Toda ventana completa
    # contiene ≥ 2 activos ⇒ vivo al cap.
    sessions = np.full(45, 10.0)
    sessions[[0, 15, 30]] = 60.0
    r = _eval_l([_lbda(np.arange(1, 46), np.cumsum(sessions))])
    assert r.alive_at_cap
    assert not r.inactivity_death
    assert r.days == 45.0


def test_inactividad_cuenta_dias_sin_trades_via_gaps():
    # b0: 1 día +100 (activo). Gap de 30 días naturales sin trades. b1 abre en
    # s = 31: la ventana (1, 31] ya no contiene el día activo (termina en 1.0)
    # ⇒ 0 < 2 → muerte al abrir b1, SIN operar su +5,000 (ni payout ni DLL).
    b0 = _lbda([1], [100.0])
    b1 = _lbda([1], [5000.0])
    r = _eval_l([b0, b1], durations=[1.0, 1.0], gaps=np.array([30.0, 0.0]))
    _assert_dead(r)
    assert r.inactivity_death
    assert r.days == 31.0
    assert np.all(r.payouts == 0.0)
    assert r.dll_hits == 0


# ---------------------------------------------------------------------------
# Integración: multi-retiro con calendario de payouts calculado a mano
# ---------------------------------------------------------------------------

def test_integracion_calendario_payouts_con_gaps_exacto():
    # 6 bloques de 5 sesiones de +1,000; gaps 4/0/2/0/1 días entre bloques.
    # Mismas cantidades que la escalera de 30 días (los gaps no mueven
    # balances), días de retiro desplazados por el calendario wall-clock:
    # 5, 14, 19, 26, 31, 37; cierre terminal en el día 37.
    blocks = [_lbda(np.arange(1, 6), 1000.0 * np.arange(1, 6)) for _ in range(6)]
    r = _eval_l(blocks, durations=[5.0] * 6,
                gaps=np.array([4.0, 0.0, 2.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(
        r.payouts[:6], [900.0, 3000.0, 3000.0, 4000.0, 4000.0, 5000.0])
    np.testing.assert_allclose(
        r.payout_days, [5.0, 14.0, 19.0, 26.0, 31.0, 37.0])
    assert r.days_to_first_payout == 5.0
    assert r.closed_after_ladder
    assert r.days == 37.0
    assert r.dll_hits == 0
    assert r.consistency_deferrals == 0
    assert not r.inactivity_death


# ---------------------------------------------------------------------------
# Dispatch en sweep_funded
# ---------------------------------------------------------------------------

def _cbda(days, rel_closed):
    """BlockDayArrays sin trailing (camino cycle FTMO)."""
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    return BlockDayArrays(
        day_ord=days,
        rel_before=np.concatenate([[0.0], rel_closed[:-1]]),
        rel_closed=rel_closed,
        rel_float=np.concatenate([[0.0], rel_closed[:-1]]),
        opens_day=np.ones(len(days), dtype=int),
        total_pnl=float(rel_closed[-1]),
    )


def _tape0(idx):
    """PhaseTape sintetica para llamar a sweep_funded con la fondeada en la ventana 0
    (``funded_start`` = 0): estos tests miden el DISPATCH del evaluador, no la continuacion."""
    idx = np.asarray(idx)
    n_paths, n = idx.shape
    return (build_phase_tape(idx, np.zeros((n_paths, 2 * n), dtype=int), None, None),
            np.zeros(n_paths, dtype=int))


def test_dispatch_cycle_ftmo_byte_identico():
    # Mini-golden FTMO: cuenta 10,000, ciclos de 3 días, split 100%. Bloque de
    # 6 días +1,000/día: ciclo 0 cierra d3 con profit 3,000 → payout 3,000 y
    # reset; ciclo 1 cierra d6 igual. cycle=2 < 4 ⇒ alive_at_cap False.
    phase = PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4)
    funded = FundedRules(payout_cycle_days=3, min_payout_profit_pct=0.0,
                         max_payout_cycles=4)
    cfg = PropFirmConfig(
        key="ftmo_t", display_name="ftmo_t", account_size=10_000.0, fee=100.0,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(phase,), funded=funded,
    )
    ba = _cbda(np.arange(1, 7), 1000.0 * np.arange(1, 7))
    idx = np.array([[0], [0]])
    passers = np.array([True, False])
    tape, fstart = _tape0(idx)
    out = sweep_funded(tape=tape, funded_start=fstart, day_arrays_f_grid=[[ba]],
                       passers_mask=passers, durations=np.ones(1), cfg=cfg,
                       f_grid=np.array([0.0]))[0.0]
    ref = evaluate_funded_path(np.array([0]), [ba], np.ones(1), None,
                               account_size=10_000.0, phase=phase,
                               funded=funded, profit_split=1.0)
    np.testing.assert_array_equal(out.funded_payouts[0], ref.cycle_payouts)
    np.testing.assert_allclose(out.funded_payouts[0], [3000.0, 3000.0, 0.0, 0.0])
    assert np.all(out.funded_payouts[1] == 0.0)  # no-passer a 0
    assert out.days_first_payout[0] == 3.0
    assert np.isnan(out.days_first_payout[1])
    assert not out.alive_at_cap.any()
    # Camino cycle: diagnósticos ladder a None (FTMO intacto).
    assert out.dll_hits is None and out.consistency_deferrals is None
    assert out.inactivity_deaths is None and out.closed_after_ladder is None


def test_dispatch_ladder_nunca_hereda_phase1():
    # Config Apex con una phase "trampa": max_loss 0.5% (línea 149,250) — el
    # dip de d1 (−1,000 → 149,000) la rompería bajo reglas FTMO. El ladder la
    # IGNORA: la PA solo obedece su trailing/DLL/payouts.
    trap = PropFirmPhase("Trampa", 6.0, 5.0, 0.5, 0)
    cfg = PropFirmConfig(
        key="apex_trap", display_name="apex_trap", account_size=ACC, fee=50.0,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(trap,), funded=APEX_FUNDED,
    )
    closes = np.concatenate([[-1000.0], -1000.0 + 1000.0 * np.arange(1, 8)])
    rel_low = np.concatenate([[0.0], closes[:-1]])
    rel_low[0] = -1000.0
    ba = _lbda(np.arange(1, 9), closes, rel_low=rel_low,
               rel_high=np.concatenate([[0.0], closes[1:]]))
    ref = evaluate_funded_path_ladder(np.array([0]), [ba], np.ones(1), None,
                                      cfg_funded=APEX_FUNDED, account_size=ACC,
                                      profit_split=1.0)
    tape, fstart = _tape0(np.array([[0]]))
    out = sweep_funded(tape=tape, funded_start=fstart, day_arrays_f_grid=[[ba]],
                       passers_mask=np.array([True]), durations=np.ones(1),
                       cfg=cfg, f_grid=np.array([0.0]))[0.0]
    # Sobrevive al dip y paga en d7 (balance 155,000): min(2,500; 900) = 900.
    assert ref.days_to_first_payout == 7.0
    np.testing.assert_array_equal(out.funded_payouts[0], ref.payouts)
    assert out.funded_payouts[0, 0] == pytest.approx(900.0)
    assert out.dll_hits[0] == 0
    assert out.days_first_payout[0] == 7.0


def test_ladder_sin_arrays_trailing_raise_ruidoso():
    with pytest.raises(ValueError, match="trailing=True"):
        _eval_l([_cbda([1], [1000.0])])


# ---------------------------------------------------------------------------
# End-to-end: sweep_funded ladder + diagnósticos agregados en sequence_ev
# ---------------------------------------------------------------------------

def _die_block_8d():
    """El bloque de test_post_payout_trailing_reanclado (paga 900 en d5, muere
    por trailing congelado en d8)."""
    return _lbda(
        np.arange(1, 9),
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 4500.0, 3800.0, 950.0],
        rel_low=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 4100.0, 3700.0, 900.0],
        rel_high=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 5000.0, 4500.0, 3800.0],
        rel_exc=[0.0, 0.0, 0.0, 0.0, 0.0, 900.0, 1300.0, 4100.0],
    )


def test_end_to_end_sweep_funded_y_diagnosticos_agregados():
    cfg = resolve_fee_eur(load_propfirm_config()["apex_150k_intraday"],
                          fx_eur_usd=1.10)
    blocks = [_ladder_block_30d(), _die_block_8d()]
    durations = np.array([30.0, 8.0])
    idx = np.array([[0], [1]])
    passers = np.array([True, True])
    tape, fstart = _tape0(idx)
    fu = sweep_funded(tape=tape, funded_start=fstart, day_arrays_f_grid=[blocks],
                      passers_mask=passers, durations=durations, cfg=cfg,
                      f_grid=np.array([0.0]))[0.0]
    np.testing.assert_allclose(
        fu.funded_payouts[0, :6],
        [900.0, 3000.0, 3000.0, 4000.0, 4000.0, 5000.0])
    assert np.all(fu.funded_payouts[0, 6:] == 0.0)
    assert fu.funded_payouts[1, 0] == pytest.approx(900.0)
    assert np.all(fu.funded_payouts[1, 1:] == 0.0)
    np.testing.assert_array_equal(fu.closed_after_ladder, [True, False])
    np.testing.assert_array_equal(fu.alive_at_cap, [False, False])
    np.testing.assert_array_equal(fu.dll_hits, [0, 0])
    np.testing.assert_array_equal(fu.consistency_deferrals, [0, 0])
    np.testing.assert_array_equal(fu.inactivity_deaths, [False, False])
    np.testing.assert_allclose(fu.days_first_payout, [5.0, 5.0])

    ch = ChallengeOutcomes(
        pass1=np.ones(2, dtype=bool), pass2=np.ones(2, dtype=bool),
        neither1=np.zeros(2, dtype=bool), neither2=np.zeros(2, dtype=bool),
        days_to_pass1=np.array([10.0, 10.0]),
    )
    oc = combine_challenge_funded(ch, fu)
    np.testing.assert_array_equal(oc.closed_after_ladder, [True, False])
    seq = sequence_ev(oc, cfg, fx_eur_usd=1.10, n_challenges=2, n_sequences=10,
                      initial_capital_eur=1000.0, seed=0)
    assert seq.mean_dll_hits == pytest.approx(0.0)
    assert seq.mean_consistency_deferrals == pytest.approx(0.0)
    assert seq.p_inactivity_death == pytest.approx(0.0)
    assert seq.p_closed_after_ladder == pytest.approx(0.5)
    # E[payouts | fondeada] = (19,900 + 900)/2 USD → EUR, sin refund.
    assert seq.mean_payout_eur == pytest.approx(10_400.0 / 1.10)
    rep = firm_report(0.01, oc, seq)
    assert rep.p_closed_after_ladder == pytest.approx(0.5)
    assert rep.mean_dll_hits == pytest.approx(0.0)


def test_diagnosticos_ladder_nan_en_camino_cycle():
    # FTMO: los arrays diagnóstico son None ⇒ medias nan (defaults intactos).
    n = 20
    tape, fstart = _tape0(np.zeros((n, 1), dtype=int))
    oc = combine_challenge_funded(
        ChallengeOutcomes(
            pass1=np.ones(n, dtype=bool), pass2=np.ones(n, dtype=bool),
            neither1=np.zeros(n, dtype=bool), neither2=np.zeros(n, dtype=bool),
            days_to_pass1=np.full(n, 10.0),
        ),
        fu=sweep_funded(
            tape=tape, funded_start=fstart,
            day_arrays_f_grid=[[_cbda(np.arange(1, 7), 1000.0 * np.arange(1, 7))]],
            passers_mask=np.ones(n, dtype=bool), durations=np.ones(1),
            cfg=PropFirmConfig(
                key="ftmo_t", display_name="ftmo_t", account_size=10_000.0,
                fee=100.0, profit_split_pct=100.0,
                refund_fee_on_first_payout=False,
                phases=(PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4),),
                funded=FundedRules(payout_cycle_days=3,
                                   min_payout_profit_pct=0.0,
                                   max_payout_cycles=4),
            ),
            f_grid=np.array([0.0]),
        )[0.0],
    )
    cfg = PropFirmConfig(
        key="ftmo_t", display_name="ftmo_t", account_size=10_000.0, fee=100.0,
        profit_split_pct=100.0, refund_fee_on_first_payout=False,
        phases=(PropFirmPhase("Challenge", 10.0, 5.0, 10.0, 4),),
        funded=FundedRules(payout_cycle_days=3, min_payout_profit_pct=0.0,
                           max_payout_cycles=4),
    )
    seq = sequence_ev(oc, cfg, fx_eur_usd=1.10, n_challenges=2, n_sequences=5,
                      initial_capital_eur=1000.0, seed=0)
    assert np.isnan(seq.mean_dll_hits)
    assert np.isnan(seq.mean_consistency_deferrals)
    assert np.isnan(seq.p_inactivity_death)
    assert np.isnan(seq.p_closed_after_ladder)
    rep = firm_report(0.01, oc, seq)
    assert np.isnan(rep.mean_dll_hits)
    assert np.isnan(rep.p_closed_after_ladder)
