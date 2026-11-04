"""Tests del walker de evaluación EOD-trail FTMO Futures (plan fase 2).

ACs cubiertos (ver plan ftmo-futures §4):
- AC-3 golden oficial del lock (umbrales [48000, 49000, 49000, 50000] con
  len(closed)+1) + breach al tocar el umbral a nivel walker (<=).
- AC-4 properties: umbral monótono no-decreciente; umbral del día d usa SOLO
  cierres estrictamente anteriores; cap = lock permanente; encaje
  umbral↔walker en secuencias largas multi-bloque (primer día de fail del
  walker == primer d con S_d + rel_low[d] <= T_d del helper dorado).
- AC-5 breach de equity intradía con recuperación de cierre → fail (la
  granularidad M1 manda); breach_on balance no se influye por el valle.
- AC-6 DLL hard de Pro mata al primer toque (rel_low − rel_before <= −dll,
  borde inclusivo); Growth eval (sin DLL) sobrevive el mismo día rojo.
- AC-7 golden oficial de consistencia (best day $2.000 al 40% → target
  efectivo $5.000; borde inclusivo con tolerancia 1e-9; best day subiendo a
  $2.500 retrasa el pase a $6.250) + la consistencia NUNCA produce fail.
- AC-8 days_eval1 relleno en pass/fail Y neither vía simulate_challenge, y
  p_unresolved sigue reportándose desde ChallengeOutcomes.

Los BlockDayArrays son sintéticos (patrón ``_tbda`` de
``test_risk_propfirm_trailing_eval.py``) con la precondición de plausibilidad
rel_low[d] <= rel_closed[d] (lower al último minuto = realized).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.mc import extend_stationary_rows
from src.risk.propfirm import (
    BlockDayArrays,
    ChallengeOutcomes,
    ConsistencySpec,
    FirmPathOutcomes,
    FundedOutcomes,
    FundedRules,
    PropFirmConfig,
    PropFirmPhase,
    TrailingDrawdown,
    build_phase_tape,
    combine_challenge_funded,
    simulate_challenge,
)
from src.risk.propfirm_futures import (
    eod_trailing_thresholds,
    evaluate_phase_path_eod_trail,
)

ACC = 50_000.0
AMOUNT = 2_000.0
CAP = 50_000.0
TARGET_USD = 3_000.0     # 6.0% de 50K (importes oficiales FTMO Futures)


def _dd(breach_on: str = "equity", amount: float = AMOUNT,
        lock: float | None = CAP) -> TrailingDrawdown:
    return TrailingDrawdown(amount_usd=amount, lock_at_balance=lock,
                            type="eod_trailing", breach_on=breach_on)


def _phase(target_pct: float = 6.0, *, daily_loss_usd=None, consistency=None,
           breach_on: str = "equity", amount: float = AMOUNT,
           lock: float | None = CAP) -> PropFirmPhase:
    return PropFirmPhase(
        name="Evaluation", profit_target_pct=target_pct,
        max_daily_loss_pct=None, max_loss_pct=None, min_trading_days=0,
        drawdown=_dd(breach_on=breach_on, amount=amount, lock=lock),
        daily_loss_usd=daily_loss_usd, consistency=consistency,
    )


G_PHASE = _phase()                                   # Growth eval (sin DLL)
P_PHASE = _phase(daily_loss_usd=1_000.0)             # Pro eval (hard 1.000)
CONS_PHASE = _phase(consistency=ConsistencySpec(
    max_best_day_share=0.40, strict=False, window="since_last_payout"))


def _fbda(days, rel_closed, rel_low, opens=None, rel_before=None):
    """BlockDayArrays sintético con rel_low (walker EOD; sin rel_high/exc)."""
    days = np.asarray(days, dtype=int)
    rel_closed = np.asarray(rel_closed, dtype=float)
    rel_low = np.asarray(rel_low, dtype=float)
    n = len(days)
    rb = (np.asarray(rel_before, dtype=float) if rel_before is not None
          else np.concatenate([[0.0], rel_closed[:-1]]))
    return BlockDayArrays(
        day_ord=days, rel_before=rb, rel_closed=rel_closed, rel_float=rb.copy(),
        opens_day=np.asarray(opens if opens is not None else [1] * n, dtype=int),
        total_pnl=float(rel_closed[-1]),
        rel_low=rel_low,
    )


def _eval(bas, phase=G_PHASE, account=ACC, order=None, durations=None,
          gaps=None):
    n = len(bas)
    return evaluate_phase_path_eod_trail(
        np.arange(n) if order is None else np.asarray(order),
        bas,
        np.ones(n) if durations is None else np.asarray(durations, dtype=float),
        gaps, account_size=account, phase=phase,
    )


# ---------------------------------------------------------------------------
# AC-3 GOLDEN OFICIAL (lock)
# ---------------------------------------------------------------------------

def test_golden_lock_oficial():
    # inicial $50.000, amount $2.000, cap $50.000 (lock permanente).
    t = eod_trailing_thresholds([51000, 50500, 52500], ACC, AMOUNT, CAP)
    # día 1 $48.000; cierre $51.000 → $49.000; cierre $50.500 → sigue
    # $49.000; cierre $52.500 → lock $50.000. len(closed)+1 umbrales.
    np.testing.assert_allclose(t, [48000.0, 49000.0, 49000.0, 50000.0])


def test_umbral_sin_cap_relativo_al_hwm():
    t = eod_trailing_thresholds([51000, 50500, 52500], ACC, AMOUNT, None)
    np.testing.assert_allclose(t, [48000.0, 49000.0, 49000.0, 50500.0])


def test_walker_breach_al_tocar_umbral():
    # A walker level: rel_low a T+1 no breachea, a T sí (<= = breach).
    # T_0 = 48000 → rel_low = −1999 (equity 48001) vivo; −2000 (48000) fail.
    vivo = _fbda([1], [0.0], [-1999.0])
    assert _eval([vivo]).outcome == "neither"
    toca = _fbda([1], [0.0], [-2000.0])
    r = _eval([toca])
    assert r.outcome == "fail"
    assert r.days == 1.0
    assert r.terminal == pytest.approx(ACC)


def test_walker_progresion_umbral_y_lock():
    # Tres días verdes suben el HWM de cierres; el umbral del día 4 queda
    # clavado en el lock (cap = inicial) aunque el HWM supere inicial+amount.
    ba = _fbda([1, 2, 3], [1000.0, 500.0, 2500.0],
               [1000.0, 500.0, 2500.0])
    r = _eval([ba])
    assert r.outcome == "neither"              # sin breach ni target (total 2500)
    # Umbral del día 4 (helper sobre los mismos cierres): lock 50000.
    t4 = eod_trailing_thresholds([ACC + 1000, ACC + 500, ACC + 2500],
                                 ACC, AMOUNT, CAP)[3]
    assert t4 == pytest.approx(CAP)
    # Día 4: el saldo (+2500) sube el ancla a 52500 → rel_low = −2500 toca
    # equity 50000 <= T → fail (el lock NO sube con el saldo: es umbral
    # absoluto clavado en el inicial).
    bloque_lock = _fbda([4], [-2500.0], [-2500.0],
                        rel_before=np.array([2500.0]))
    r2 = _eval([_fbda([1, 2, 3], [1000.0, 500.0, 2500.0],
                      [1000.0, 500.0, 2500.0]), bloque_lock],
               order=[0, 1], durations=np.array([3.0, 1.0]))
    assert r2.outcome == "fail"
    assert r2.days == 4.0


# ---------------------------------------------------------------------------
# AC-4 PROPERTIES
# ---------------------------------------------------------------------------

def test_umbral_monotono_no_decreciente():
    rng = np.random.default_rng(7)
    for _ in range(200):
        closed = rng.uniform(-5000.0, 8000.0, size=60)
        t = eod_trailing_thresholds(closed, ACC, AMOUNT, CAP)
        assert np.all(np.diff(t) >= 0.0)
        assert np.all(t <= CAP)


def test_umbral_usa_solo_cierres_anteriores():
    rng = np.random.default_rng(11)
    closed = rng.uniform(-3000.0, 6000.0, size=40)
    base = eod_trailing_thresholds(closed, ACC, AMOUNT, CAP)
    for d in (0, 7, 19, 39):
        for delta in (25000.0, -25000.0):
            mutated = closed.copy()
            mutated[d] += delta
            t = eod_trailing_thresholds(mutated, ACC, AMOUNT, CAP)
            # Mutar el cierre del día d no altera los umbrales 0..d (usa
            # cierres ESTRICTAMENTE anteriores, §2.9.2).
            np.testing.assert_array_equal(t[: d + 1], base[: d + 1])


def test_umbral_cap_en_lock():
    # HWM cruza inicial+amount: el umbral queda clavado en el cap para
    # siempre, aunque los cierres caigan después.
    t = eod_trailing_thresholds([53000.0, 40000.0, 30000.0], ACC, AMOUNT, CAP)
    np.testing.assert_allclose(t, [48000.0, 50000.0, 50000.0, 50000.0])
    t_nocap = eod_trailing_thresholds([53000.0, 40000.0, 30000.0],
                                      ACC, AMOUNT, None)
    np.testing.assert_allclose(t_nocap, [48000.0, 51000.0, 51000.0, 51000.0])


def test_umbral_walker_encaje_secuencia_larga():
    # ≥200 días multi-bloque: el primer día de fail del walker == primer d
    # con S_d + rel_low[d] <= T_d del helper dorado (o ausencia de fail).
    rng = np.random.default_rng(23)
    n_blocks, per = 5, 50
    blocks = []
    for b in range(n_blocks):
        rel_closed = rng.uniform(-4000.0, 5000.0, size=per)
        rel_low = rel_closed - rng.uniform(0.0, 3000.0, size=per)
        days = 100 * b + 1 + np.arange(per)
        blocks.append(_fbda(days, rel_closed, rel_low))
    durations = np.full(n_blocks, float(per))
    phase = _phase(target_pct=1e6)      # target inalcanzable: aísla breach
    r = _eval(blocks, phase=phase, durations=durations)

    # Referencia independiente: mismo accounting cross-bloque, umbral por el
    # helper dorado sobre el prefijo de cierres absolutos.
    hwm, saldo, elapsed_prev = ACC, 0.0, 0.0
    closed_prefix: list[float] = []
    expected_fail_days = None
    for pos, ba in enumerate(blocks):
        s_block = ACC + saldo
        day0 = int(ba.day_ord[0])
        for j in range(len(ba.day_ord)):
            t_day = eod_trailing_thresholds(closed_prefix, ACC, AMOUNT, CAP)
            t_day = t_day[len(closed_prefix)]
            if s_block + ba.rel_low[j] <= t_day:
                expected_fail_days = elapsed_prev + (int(ba.day_ord[j]) - day0) + 1.0
                break
            closed_prefix.append(s_block + ba.rel_closed[j])
        if expected_fail_days is not None:
            break
        saldo += ba.total_pnl
        elapsed_prev += durations[pos]

    if expected_fail_days is None:
        assert r.outcome == "neither"
    else:
        assert r.outcome == "fail"
        assert r.days == pytest.approx(expected_fail_days)


def test_s_d_incluye_saldo_cross_bloque():
    # S_d = ancla + saldo (NO balance de inicio de día): una pérdida del
    # bloque 1 reduce la equity absoluta del bloque 2 y hace breachear un
    # valle que con saldo 0 sobreviviría.
    b1 = _fbda([1], [-1000.0], [-1000.0])
    # Bloque 2: rel_low = −1000. Con saldo −1000 → equity 48000 <= T (48000)
    # → fail; con saldo 0 sería 49000 → vivo.
    b2 = _fbda([2], [0.0], [-1000.0])
    r = _eval([b1, b2], durations=np.array([1.0, 1.0]))
    assert r.outcome == "fail"
    assert r.days == 2.0
    # Y con el bloque 1 a cero, el mismo bloque 2 NO breachea.
    b1_flat = _fbda([1], [0.0], [0.0])
    r2 = _eval([b1_flat, b2], durations=np.array([1.0, 1.0]))
    assert r2.outcome == "neither"


# ---------------------------------------------------------------------------
# AC-5 breach equity intradía con recuperación de cierre
# ---------------------------------------------------------------------------

def test_breach_equity_intradia_con_recuperacion_de_cierre():
    # rel_low breachea T_0 (47500 <= 48000) pero el cierre (49900) queda por
    # encima → fail igual: la granularidad intradía M1 manda (un walker
    # EOD-only habría infravalorado este caso — adjudicación DeepSeek).
    ba = _fbda([1], [-100.0], [-2500.0])
    r = _eval([ba])
    assert r.outcome == "fail"
    assert r.days == 1.0


def test_breach_on_balance_ignora_el_valle():
    phase = _phase(breach_on="balance")
    ba = _fbda([1], [-100.0], [-2500.0])   # valle breachearía en equity
    assert _eval([ba], phase=phase).outcome == "neither"
    toca = _fbda([1], [-2000.0], [-2500.0])  # cierre 48000 <= T → fail
    assert _eval([toca], phase=phase).outcome == "fail"


def test_bloque_sin_rel_low_raise_ruidoso():
    ba = _fbda([1], [0.0], [0.0])
    ba = BlockDayArrays(day_ord=ba.day_ord, rel_before=ba.rel_before,
                        rel_closed=ba.rel_closed, rel_float=ba.rel_float,
                        opens_day=ba.opens_day, total_pnl=ba.total_pnl,
                        rel_low=None)
    with pytest.raises(ValueError, match="rel_low"):
        _eval([ba])
    with pytest.raises(ValueError, match="rel_low"):
        _eval([ba], phase=P_PHASE)


# ---------------------------------------------------------------------------
# AC-6 DLL hard de fase (Pro) / Growth sin DLL en eval
# ---------------------------------------------------------------------------

def test_dll_hard_pro_mata_al_primer_toque():
    # rel_low − rel_before = −1000 exacto (toque) → fail por DLL puro: la
    # equity (49000) NO breachea el Max DD (T_0 = 48000).
    ba = _fbda([1], [-500.0], [-1000.0])
    r = _eval([ba], phase=P_PHASE)
    assert r.outcome == "fail"
    assert r.days == 1.0
    # −999.9 no dispara (borde por debajo del trigger).
    casi = _fbda([1], [-500.0], [-999.9])
    assert _eval([casi], phase=P_PHASE).outcome == "neither"


def test_growth_eval_sin_dll():
    # El mismo día rojo del test anterior, con la fase Growth (sin DLL):
    # no hay fail por DLL (la equity tampoco toca el Max DD) → neither.
    ba = _fbda([1], [-500.0], [-1000.0])
    assert _eval([ba], phase=G_PHASE).outcome == "neither"


# ---------------------------------------------------------------------------
# AC-7 GOLDEN OFICIAL (consistencia 40%)
# ---------------------------------------------------------------------------

def test_golden_consistencia_40():
    # Target oficial $3.000, share 0.40, best day $2.000 → target efectivo
    # $5.000. Cadena de cierres en UN bloque (rel_before encadena solo);
    # nets diarios ≤ 2000 para que el best day se mantenga en 2000.
    # A) total 4999.99 → NO pass (sigue operando; backtest acaba → neither).
    ba_a = _fbda([1, 2, 3, 4, 5], [2000.0, 2500.0, 3000.0, 4000.0, 4999.99],
                 [2000.0, 2500.0, 3000.0, 4000.0, 4999.99])
    assert _eval([ba_a], phase=CONS_PHASE).outcome == "neither"
    # B) total 5000.00 exacto → pass (borde inclusivo, tolerancia 1e-9).
    ba_b = _fbda([1, 2, 3, 4, 5], [2000.0, 2500.0, 3000.0, 4000.0, 5000.0],
                 [2000.0, 2500.0, 3000.0, 4000.0, 5000.0])
    r = _eval([ba_b], phase=CONS_PHASE)
    assert r.outcome == "pass"
    assert r.days == 5.0
    # C) best day sube a 2500 (día 3) → deja de pasar a 5000, pasa a 6250
    #    exacto el día 4 (nets diarios: +2000, +500, +2500, +1250).
    ba_c3 = _fbda([1, 2, 3], [2000.0, 2500.0, 5000.0],
                  [2000.0, 2500.0, 5000.0])
    assert _eval([ba_c3], phase=CONS_PHASE).outcome == "neither"
    ba_c4 = _fbda([1, 2, 3, 4], [2000.0, 2500.0, 5000.0, 6250.0],
                  [2000.0, 2500.0, 5000.0, 6250.0])
    r_c = _eval([ba_c4], phase=CONS_PHASE)
    assert r_c.outcome == "pass"
    assert r_c.days == 4.0


def test_consistencia_nunca_produce_breach():
    # Path que incumple la consistencia de forma permanente (best day 2000,
    # total 4000 → 2000/4000 = 50% > 40%) sin tocar Max DD ni DLL: queda
    # neither (sigue operando), jamás fail por consistencia.
    dia1 = _fbda([1], [2000.0], [2000.0])
    dia2 = _fbda([2], [2000.0], [2000.0])
    r = _eval([dia1, dia2], phase=CONS_PHASE)
    assert r.outcome == "neither"
    # Y diluyendo después (best day 2000, total 10000 → 20% ≤ 40%) pasa
    # tarde, nunca breachea.
    diluye = [_fbda([3 + k], [1500.0], [1500.0]) for k in range(4)]
    r2 = _eval([dia1, dia2] + diluye, phase=CONS_PHASE)
    assert r2.outcome == "pass"


def test_consistencia_target_manda_cuando_best_day_bajo():
    # best_day/share (1000/0.40 = 2500) < target (3000) → manda el target:
    # pasa en cuanto total >= 3000 aunque el mejor día no llegue al 40%
    # del total (nets diarios: +1000, +500, +500, +1000).
    ba4 = _fbda([1, 2, 3, 4], [1000.0, 1500.0, 2000.0, 3000.0],
                [1000.0, 1500.0, 2000.0, 3000.0])
    r = _eval([ba4], phase=CONS_PHASE)
    assert r.outcome == "pass"
    assert r.days == 4.0
    # El mismo path truncado a total 2000 (< target) no pasa.
    ba3 = _fbda([1, 2, 3], [1000.0, 1500.0, 2000.0],
                [1000.0, 1500.0, 2000.0])
    assert _eval([ba3], phase=CONS_PHASE).outcome == "neither"


# ---------------------------------------------------------------------------
# AC-8 days_eval1 relleno en TODOS los outcomes (vía simulate_challenge)
# ---------------------------------------------------------------------------

def _cfg(phase=G_PHASE):
    return PropFirmConfig(
        key="ft", display_name="ft", account_size=ACC, fee=0.0,
        profit_split_pct=90.0, refund_fee_on_first_payout=False,
        phases=(phase,),
        funded=FundedRules(payout_cycle_days=14, min_payout_profit_pct=0.0,
                           max_payout_cycles=3),
    )


def _tape(idx, n_trades):
    """PhaseTape sintetica (sin gaps): la F1 es ``idx``."""
    ext = extend_stationary_rows(idx, n_trades, 1.0, 2 * idx.shape[1],
                                 np.random.default_rng(0))
    return build_phase_tape(idx, ext, None, None)


def test_days_eval1_relleno_en_pass_fail_y_neither():
    b_pass = _fbda([1, 2], [TARGET_USD + 200.0, TARGET_USD + 200.0],
                   [TARGET_USD + 200.0, TARGET_USD + 200.0])
    b_fail = _fbda([1], [0.0], [-AMOUNT - 500.0])
    b_neither = _fbda([1, 2, 3], [100.0, 200.0, 300.0],
                      [100.0, 200.0, 300.0])
    idx = np.array([[0], [1], [2]])
    durations = np.array([2.0, 1.0, 3.0])
    ch = simulate_challenge(_tape(idx, 3), [b_pass, b_fail, b_neither], durations,
                            _cfg(), portfolio=True)

    assert list(ch.pass1) == [True, False, False]
    assert list(ch.neither1) == [False, False, True]
    # days_eval1: pass a los 1 día; fail a los 1; neither = duración completa.
    np.testing.assert_allclose(ch.days_eval1, [1.0, 1.0, 3.0])
    np.testing.assert_allclose(ch.days_to_pass1, [1.0, np.nan, np.nan])
    # p_unresolved a nivel path sigue reportándose desde ChallengeOutcomes.
    assert list(ch.unresolved) == [False, False, True]

    # Propagación a FirmPathOutcomes (campo aditivo; los legacy lo traen None).
    fu = FundedOutcomes(funded_payouts=np.zeros((3, 3)),
                        days_first_payout=np.full(3, np.nan),
                        alive_at_cap=np.zeros(3, dtype=bool))
    oc = combine_challenge_funded(ch, fu)
    assert isinstance(oc, FirmPathOutcomes)
    np.testing.assert_allclose(oc.days_eval1, [1.0, 1.0, 3.0])
    legacy = combine_challenge_funded(
        ChallengeOutcomes(pass1=np.ones(1, dtype=bool),
                          pass2=np.ones(1, dtype=bool),
                          neither1=np.zeros(1, dtype=bool),
                          neither2=np.zeros(1, dtype=bool),
                          days_to_pass1=np.array([5.0])),
        FundedOutcomes(funded_payouts=np.zeros((1, 3)),
                       days_first_payout=np.full(1, np.nan),
                       alive_at_cap=np.zeros(1, dtype=bool)))
    assert legacy.days_eval1 is None


def test_dispatch_eod_trailing_exige_portfolio_para_equity():
    b = _fbda([1], [0.0], [0.0])
    idx = np.array([[0]])
    with pytest.raises(ValueError, match="camino portfolio"):
        simulate_challenge(_tape(idx, 1), [b], np.array([1.0]), _cfg(),
                           portfolio=False)
    # breach_on balance sí admite mono (no usa rel_low).
    cfg_bal = _cfg(phase=_phase(breach_on="balance"))
    ch = simulate_challenge(_tape(idx, 1), [b], np.array([1.0]), cfg_bal,
                            portfolio=False)
    assert list(ch.neither1) == [True]
    np.testing.assert_allclose(ch.days_eval1, [1.0])


def test_walker_rechaza_drawdown_no_eod():
    from src.risk.propfirm import TrailingDrawdown as _TD
    fase_apex = PropFirmPhase(
        name="Evaluation", profit_target_pct=6.0, max_daily_loss_pct=None,
        max_loss_pct=None, min_trading_days=0,
        drawdown=_TD(amount_usd=AMOUNT),
    )
    b = _fbda([1], [0.0], [0.0])
    with pytest.raises(ValueError, match="eod_trailing"):
        evaluate_phase_path_eod_trail(np.array([0]), [b], np.array([1.0]),
                                      account_size=ACC, phase=fase_apex)
