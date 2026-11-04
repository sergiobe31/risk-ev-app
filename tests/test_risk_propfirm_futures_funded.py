"""Tests de la máquina funded FTMO Futures (plan fase 3, riesgo ALTO:
mecánica de dinero). ACs: AC-9 (golden qualifying), AC-10 (golden
min-new-profit + monotonía/cotas), AC-11 (HWM pre-retiro y colchón),
AC-12 (gemelos eff_low + tercer caso rel_before≠0), AC-13 (soft-DLL nunca
mata, clamp signo max, caso no-op), AC-14 (hard DLL Pro), AC-15 (constraints
de payout + conservación del dinero), AC-16 (políticas greedy/buffer a nivel
máquina; la parte de compare_policies_futures es Fase 4 — la función aún no
existe y se reporta en el entregable).

Convenciones pineadas por los fixtures: ancla del día = account + saldo
ACTUAL (los arrays rel_* son relativos al ancla del BLOQUE; los ajustes
mid-bloque entran en saldo, patrón cross-bloque verificado en Apex);
eff_low = max(rel_low, rel_before − dll) en unidades de bloque; trigger DLL
con rel_low CRUDO. Por eso los fixtures de días encadenados usan UN bloque
(rel_before encadena solo) salvo que se quiera pinchar explícitamente una
frontera de bloque.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.propfirm import (
    BlockDayArrays,
    DailyLossSpec,
    FundedRules,
    FuturesPayouts,
    TrailingDrawdown,
)
from src.risk.propfirm_futures import (
    dll_eff_low,
    evaluate_funded_path_futures,
    payout_request_max,
    soft_dll_clamp,
)

ACC = 50_000.0
HORIZON = 365.0


def _funded(amount=2_000.0, lock=ACC, dll_type="pause_not_fatal",
            dll=1_000.0, qual_min=150.0, days_per=4, wshare=0.50,
            cap=2_500.0, min_payout=20.0, max_cycles=50,
            buffer=None) -> FundedRules:
    return FundedRules(
        payout_cycle_days=0, min_payout_profit_pct=0.0,
        max_payout_cycles=max_cycles, model="futures",
        drawdown=TrailingDrawdown(amount_usd=amount, lock_at_balance=lock,
                                  type="eod_trailing", breach_on="equity"),
        daily_loss=DailyLossSpec(type=dll_type, by_tier_usd=(float(dll),),
                                 post_hit="clamp"),
        payouts_futures=FuturesPayouts(
            qualifying_day_min_usd=qual_min, days_per_payout=days_per,
            withdraw_share=wshare, request_cap_usd=cap,
            min_new_profit_frac=0.50, min_payout_usd=min_payout,
            withdrawal_policy="greedy",
            buffer_usd=(float(buffer) if buffer is not None else amount),
        ),
    )


def _fbda(days, rel_closed, rel_low, opens=None, rel_before=None):
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


def _run(bas, cfg=None, order=None, durations=None, gaps=None,
         account=ACC, policy="greedy", horizon=HORIZON):
    cfg = cfg or _funded()
    n = len(bas)
    return evaluate_funded_path_futures(
        np.arange(n) if order is None else np.asarray(order),
        bas,
        np.ones(n) if durations is None else np.asarray(durations, dtype=float),
        gaps,
        cfg_funded=cfg, account_size=account, profit_split=0.9,
        horizon_days=horizon, withdrawal_policy=policy,
    )


# ---------------------------------------------------------------------------
# AC-9 GOLDEN OFICIAL (qualifying 3 de 4)
# ---------------------------------------------------------------------------

def test_golden_qualifying_3_de_4():
    # Growth 50K (4 × $150): [+200, +300, +100, +400] → 3 de 4 cualifican
    # (+100 < 150) → NO elegible; un 5º día +150 → elegible y paga.
    ba4 = _fbda([1, 2, 3, 4], [200.0, 500.0, 600.0, 1000.0],
                [200.0, 500.0, 600.0, 1000.0])
    r = _run([ba4])
    assert r.days == 4.0 and not r.alive_at_cap and r.payouts.sum() == 0.0
    assert r.payout_deferrals == 0          # aún no hay 4 qualifying days
    ba5 = _fbda([1, 2, 3, 4, 5],
                [200.0, 500.0, 600.0, 1000.0, 1150.0],
                [200.0, 500.0, 600.0, 1000.0, 1150.0])
    r5 = _run([ba5])
    # balance 501150 → poi 1150; request = min(2500, 0.5×1150, 1150/0.5)
    # = 575 → registrado 0.9 × 575.
    assert r5.payouts[0] == pytest.approx(575.0 * 0.9)
    assert r5.days_to_first_payout == 5.0
    assert r5.payout_deferrals == 0


# ---------------------------------------------------------------------------
# AC-10 GOLDEN OFICIAL (min-new-profit) + properties
# ---------------------------------------------------------------------------

def test_golden_min_new_profit():
    # Oficial: profit viejo $2.000 + nuevo del ciclo $400, share 1.0, cap
    # alto → request_max = min(1e9, 2400, 400/0.5) = $800 exactos.
    req = payout_request_max(
        profit_over_initial=2400.0, new_profit_cycle=400.0, share=1.0,
        cap=1e9, min_new_frac=0.5, balance=52_400.0, threshold=48_000.0,
        policy="greedy", buffer_usd=2_000.0)
    assert req == pytest.approx(800.0)


def test_payout_request_max_monotonia_y_cotas():
    rng = np.random.default_rng(5)
    for _ in range(500):
        poi = rng.uniform(-500.0, 20_000.0)
        new = rng.uniform(-500.0, 10_000.0)
        share = rng.uniform(0.05, 1.0)
        cap = rng.uniform(100.0, 10_000.0)
        balance = rng.uniform(40_000.0, 80_000.0)
        thr = rng.uniform(44_000.0, 50_000.0)
        req = payout_request_max(
            profit_over_initial=poi, new_profit_cycle=new, share=share,
            cap=cap, min_new_frac=0.5, balance=balance, threshold=thr,
            policy="greedy", buffer_usd=2_000.0)
        # El helper clampa a 0: req == max(0, min(cap, share·poi, new/frac))
        assert req == pytest.approx(max(0.0, min(cap, share * poi,
                                                 new / 0.5)))
        # no-decreciente en profit_over_initial (ceteris paribus)
        req_up = payout_request_max(
            profit_over_initial=poi + 100.0, new_profit_cycle=new, share=share,
            cap=cap, min_new_frac=0.5, balance=balance, threshold=thr,
            policy="greedy", buffer_usd=2_000.0)
        assert req_up >= req - 1e-12
        # no-decreciente en new_profit_cycle
        req_new = payout_request_max(
            profit_over_initial=poi, new_profit_cycle=new + 100.0, share=share,
            cap=cap, min_new_frac=0.5, balance=balance, threshold=thr,
            policy="greedy", buffer_usd=2_000.0)
        assert req_new >= req - 1e-12
        # buffer: además <= balance − T − buffer (clamp a 0 si la cota < 0)
        req_b = payout_request_max(
            profit_over_initial=poi, new_profit_cycle=new, share=share,
            cap=cap, min_new_frac=0.5, balance=balance, threshold=thr,
            policy="buffer_preserving", buffer_usd=2_000.0)
        assert req_b <= max(0.0, balance - thr - 2_000.0) + 1e-9
        assert req_b <= req + 1e-12


def test_payout_request_max_no_retirable_con_ciclo_no_positivo():
    # new_profit_cycle <= 0 → no retirable (request 0 → deferral).
    assert payout_request_max(
        profit_over_initial=5_000.0, new_profit_cycle=0.0, share=1.0,
        cap=10_000.0, min_new_frac=0.5, balance=55_000.0, threshold=48_000.0,
        policy="greedy", buffer_usd=2_000.0) == 0.0
    assert payout_request_max(
        profit_over_initial=5_000.0, new_profit_cycle=-100.0, share=1.0,
        cap=10_000.0, min_new_frac=0.5, balance=55_000.0, threshold=48_000.0,
        policy="greedy", buffer_usd=2_000.0) == 0.0


# ---------------------------------------------------------------------------
# AC-11 HWM pre-retiro y colchón
# ---------------------------------------------------------------------------

def test_hwm_pre_retiro_el_umbral_no_baja():
    # SIN cap (lock None) y dll grande (sin hits): 4 días de +1000 → payout
    # de 2000 el día 4 (balance 54000, poi 4000, share 0.5 → 2000). HWM =
    # 54000 (pre-retiro) → T_5 = 52000. Si el HWM usara el cierre
    # POST-retiro (52000), T_5 sería 50000 y el día 5 SOBREVIVIRÍA: con
    # rel_low = −3900 la equity es 50100 → con HWM correcto muere
    # (50100 <= 52000); con el bug viviría (50100 > 50000).
    cfg = _funded(amount=2_000.0, lock=None, dll=5_000.0)
    verdes = _fbda([1, 2, 3, 4], [1000.0, 2000.0, 3000.0, 4000.0],
                   [1000.0, 2000.0, 3000.0, 4000.0])
    dia5 = _fbda([5], [0.0], [-3900.0])
    r = _run([verdes, dia5], cfg=cfg, durations=np.array([4.0, 1.0]))
    assert not r.alive_at_cap
    assert r.days == 5.0                    # muerte el día 5 por Max DD
    assert r.dll_hits == 0                  # dll 5000: −3900 no dispara
    assert r.payouts[0] == pytest.approx(2000.0 * 0.9)


def test_hwm_pre_retiro_y_colchon_con_lock():
    # CON lock (cap = inicial, perfil por defecto): 4 días de +1500 → payout
    # 2500 (cap del request) el día 4. Colchón del día 4 pre-retiro:
    # 56000 − 50000 = 6000; tras retirar 2500 el balance es 53500 y el
    # colchón 3500 = 6000 − 2500 EXACTAMENTE (el umbral no baja: lock).
    verdes = _fbda([1, 2, 3, 4], [1500.0, 3000.0, 4500.0, 6000.0],
                   [1500.0, 3000.0, 4500.0, 6000.0])
    r = _run([verdes])
    assert r.payouts[0] == pytest.approx(2500.0 * 0.9)
    assert r.final_balance == pytest.approx(53_500.0)
    assert r.final_balance - ACC == pytest.approx(3_500.0)


# ---------------------------------------------------------------------------
# AC-12 gemelos eff_low (buffer ≷ dll) + tercer caso rel_before = +500
# ---------------------------------------------------------------------------

def test_eff_low_helper_dimensionalidad():
    # Tercer caso del plan (N1): rel_before = +500, dll = 1000,
    # rel_low = −4000 → el trigger dispara (−4500 <= −1000) y
    # eff_low = max(−4000, 500 − 1000) = −500 (NO −1000: la forma
    # dimensionalmente errónea max(rel_low, −dll) daría −1000).
    eff, hit = dll_eff_low(-4000.0, 500.0, 1000.0)
    assert hit and eff == -500.0
    # rel_before = 0 reduce a la forma simple.
    eff0, hit0 = dll_eff_low(-3000.0, 0.0, 1000.0)
    assert hit0 and eff0 == -1000.0
    # Sin hit: eff_low = rel_low.
    eff1, hit1 = dll_eff_low(-600.0, 0.0, 1000.0)
    assert not hit1 and eff1 == -600.0


def test_eff_low_buffer_mayor_que_dll_vive():
    # B = 2500 > dll = 1000; rel_before = 0; rel_low = −3000 (∈ [−B−dll, −B]):
    # el crudo breachearía (47000 <= 47500) PERO el DLL dispara y la
    # liquidación fija el peor punto en −1000 → 49000 > 47500 → VIVO.
    cfg = _funded(amount=2_500.0)
    dia = _fbda([1], [-2500.0], [-3000.0])
    r = _run([dia], cfg=cfg)
    assert r.days == 1.0 and not r.alive_at_cap      # vivo al fin del backtest
    assert r.dll_hits == 1
    # Clamp: cierre peor que −dll → neto exacto en −dll → balance 49000.
    assert r.final_balance == pytest.approx(49_000.0)


def test_eff_low_buffer_menor_que_dll_muere():
    # Gemelo: B = 500 < dll = 1000 (amount 500); mismo rel_low = −3000:
    # ambos disparan, eff_low = −1000 → 49000 <= 49500 → MUERTE por Max DD
    # (no por DLL: la muerte llega en el paso del breach, antes del efecto).
    cfg = _funded(amount=500.0)
    dia = _fbda([1], [-2500.0], [-3000.0])
    r = _run([dia], cfg=cfg)
    assert r.days == 1.0 and not r.alive_at_cap
    assert r.dll_hits == 1                  # el trigger sí disparó
    assert r.payouts.sum() == 0.0


def test_eff_low_tercer_caso_rel_before_positivo():
    # rel_before = +500, dll = 1000, rel_low = −4000, B = 2000 (amount 1500,
    # T_0 = 48500). El +500 es el día 1 del MISMO bloque (rel_before[0] = 0
    # en bloques reales). Día 2: trigger con el crudo (−4500 <= −1000),
    # eff_low = max(−4000, 500 − 1000) = −500 → equity 49500 > 48500 → VIVO.
    cfg = _funded(amount=1_500.0)
    ba = _fbda([1, 2], [500.0, -4000.0], [500.0, -4000.0])
    r = _run([ba], cfg=cfg)
    assert r.dll_hits == 1 and r.days == 2.0 and not r.alive_at_cap
    # observed día 2 = −4500 → neto −1000; clamp_adj = 3500; balance =
    # 50000 + 500 (día 1) − 1000 (neto clamped) = 49500 (conservación).
    assert r.final_balance == pytest.approx(49_500.0)


def test_trigger_dll_usa_rel_low_crudo():
    # §2.9.13: rel_low = −600 con rel_before = 0 NO llega a −dll = −1000 →
    # NO hay hit; el breach con eff_low = rel_low (−600 → 49400 <= 49500) es
    # por Max DD y dll_hits queda en 0.
    cfg = _funded(amount=500.0)
    dia = _fbda([1], [-600.0], [-600.0])
    r = _run([dia], cfg=cfg)
    assert r.days == 1.0 and not r.alive_at_cap
    assert r.dll_hits == 0


# ---------------------------------------------------------------------------
# AC-13 soft-DLL nunca mata / clamp signo max / caso no-op
# ---------------------------------------------------------------------------

def test_soft_dll_nunca_mata():
    # Colchón enorme (amount 40000 → T_0 = 10000): 10 días forzados con
    # rel_low = rel_closed = −5000·k acumulado (hit; cierre peor que −dll)
    # → la cuenta NUNCA muere por DLL; dll_hits = 10; cada día cuesta
    # exactamente −1000 en balance.
    cfg = _funded(amount=40_000.0)
    cum = -5000.0 * np.arange(1, 11)
    ba = _fbda(np.arange(1, 11), cum, cum.copy())
    r = _run([ba], cfg=cfg)
    assert r.days == 10.0 and not r.alive_at_cap
    assert r.dll_hits == 10
    assert r.final_balance == pytest.approx(ACC - 10_000.0)


def test_clamp_signo_max():
    # Día con cierre peor que −dll: el neto queda EXACTO en −dll (max), no
    # por debajo (min del ladder Apex, divergencia deliberada documentada).
    observed, adj = soft_dll_clamp(-2500.0, 1000.0)
    assert observed == -1000.0 and adj == 1500.0
    cfg = _funded(amount=40_000.0)
    dia = _fbda([1], [-2500.0], [-2500.0])
    r = _run([dia], cfg=cfg)
    assert r.dll_hits == 1
    assert r.final_balance == pytest.approx(ACC - 1000.0)


def test_clamp_noop_dia_recuperado():
    # Caso no-op (v2 #15): hit intradía (rel_low = −1900) pero cierre mejor
    # que −dll (+200 neto) → session_net NO cambia y, siendo >= umbral, el
    # día CUALIFICA: con 4 qualifying days el payout llega el día 4. Si el
    # clamp (erróneo) hubiera puesto el neto en −1000, el ciclo sería
    # 600 − 1000 < 0 → request 0 → deferral.
    dias = _fbda([1, 2, 3, 4], [200.0, 400.0, 600.0, 800.0],
                 [200.0, 400.0, 600.0, -1900.0])
    r = _run([dias])
    assert r.dll_hits == 1
    assert r.payout_deferrals == 0
    assert r.days_to_first_payout == 4.0
    # request = min(2500, 0.5×800, 800/0.5) = 400 → registrado 360.
    assert r.payouts[0] == pytest.approx(400.0 * 0.9)


# ---------------------------------------------------------------------------
# AC-14 hard DLL funded Pro
# ---------------------------------------------------------------------------

def test_hard_dll_funded_pro_mata():
    cfg = _funded(dll_type="hard_fatal", dll=1_000.0, amount=3_000.0)
    ba = _fbda([1, 2], [200.0, -1500.0], [200.0, -1500.0])
    r = _run([ba], cfg=cfg)
    assert r.days == 2.0 and not r.alive_at_cap
    assert r.dll_hits == 1                  # el trigger disparó (hard)
    assert r.payouts.sum() == 0.0           # payouts posteriores a cero


# ---------------------------------------------------------------------------
# AC-15 constraints de payout + conservación del dinero
# ---------------------------------------------------------------------------

def test_payout_request_nunca_supera_el_cap():
    cfg = _funded()
    ba = _fbda([1, 2, 3, 4], [3000.0, 6000.0, 9000.0, 12000.0],
               [3000.0, 6000.0, 9000.0, 12000.0])
    r = _run([ba], cfg=cfg)
    assert r.payouts[0] == pytest.approx(2500.0 * 0.9)   # cap 2500


def test_payout_request_nunca_supera_share_por_profit():
    # 4 días de +500 → poi 2000 → share 0.5 × 2000 = 1000 (manda el share).
    cfg = _funded()
    ba = _fbda([1, 2, 3, 4], [500.0, 1000.0, 1500.0, 2000.0],
               [500.0, 1000.0, 1500.0, 2000.0])
    r = _run([ba], cfg=cfg)
    assert r.payouts[0] == pytest.approx(1000.0 * 0.9)


def test_payout_menor_20_es_deferral_y_luego_paga():
    # qual_min pequeño y +2/día: poi = 2k → request = min(2500, k, 4k) = k.
    # Días 4..19: k = 4..19 < 20 → deferral (16); el día 20 request = 20 →
    # paga. Los qual_days se conservan tras cada deferral.
    cfg = _funded(qual_min=1.0)
    cum = 2.0 * np.arange(1, 21)
    ba = _fbda(np.arange(1, 21), cum, cum.copy())
    r = _run([ba], cfg=cfg)
    assert r.payout_deferrals == 16
    assert r.days_to_first_payout == 20.0
    assert r.payouts[0] == pytest.approx(20.0 * 0.9)


def test_reset_de_qual_days_y_ciclo_tras_pago():
    # +5/día, qual_min 1, days_per 4: primer payout día 8 (poi 40 → request
    # min(2500, 20, 80) = 20). Tras pagar, el ciclo se resetea: 4 días más a
    # +5 → ciclo nuevo 20 → poi del día 12 = 40 → request min(2500, 20, 40)
    # = 20 → segundo payout el día 12.
    cfg = _funded(qual_min=1.0)
    cum = 5.0 * np.arange(1, 13)
    ba = _fbda(np.arange(1, 13), cum, cum.copy())
    r = _run([ba], cfg=cfg)
    assert r.days_to_first_payout == 8.0
    n_payouts = int((r.payouts > 0).sum())
    assert n_payouts == 2
    assert r.payouts[1] == pytest.approx(20.0 * 0.9)


def test_alive_at_cap_al_llegar_a_funded_cap_days():
    cfg = _funded()
    cum = 100.0 * np.arange(1, 5)
    ba = _fbda(np.arange(1, 5), cum, cum.copy())
    r = _run([ba], cfg=cfg, horizon=2.0)
    assert r.alive_at_cap
    assert r.days == 3.0    # día 3 = primero con total_days > 2


def test_conservacion_del_dinero():
    # Fixture aleatoria multi-bloque con clamps, payouts y SIN muerte:
    # balance_final == account + Σ session_net (post-clamp) − Σ request.
    # El lado esperado replica SOLO las reglas adjudicadas (clamp max y
    # fórmula del request), no la máquina.
    rng = np.random.default_rng(17)
    blocks = []
    nets_flat = []
    for b in range(3):
        nets = rng.uniform(-1500.0, 2500.0, size=20)
        cum = np.cumsum(nets)
        lows = cum - rng.uniform(0.0, 2500.0, size=20)
        blocks.append(_fbda(100 * b + 1 + np.arange(20), cum, lows))
        nets_flat.extend(nets.tolist())
    cfg = _funded(amount=40_000.0, qual_min=1.0, days_per=3)
    r = _run(blocks, cfg=cfg, durations=np.full(3, 20.0))
    assert r.days == 60.0 and not r.alive_at_cap   # sin muerte ni cap
    assert r.dll_hits > 0

    dll = 1_000.0
    post_clamp = [max(n, -dll) for n in nets_flat]
    saldo = 0.0
    qual = 0
    cycle = 0.0
    req_total = 0.0
    for ba in blocks:
        for j in range(len(ba.day_ord)):
            observed = ba.rel_closed[j] - ba.rel_before[j]
            hit = ba.rel_low[j] - ba.rel_before[j] <= -dll
            if hit:
                net, adj = soft_dll_clamp(observed, dll)
                saldo += adj
            else:
                net = observed
            if net >= cfg.payouts_futures.qualifying_day_min_usd:
                qual += 1
            cycle += net
            if qual >= cfg.payouts_futures.days_per_payout:
                bal = ACC + saldo + ba.rel_closed[j]
                req = payout_request_max(
                    profit_over_initial=bal - ACC, new_profit_cycle=cycle,
                    share=cfg.payouts_futures.withdraw_share,
                    cap=cfg.payouts_futures.request_cap_usd,
                    min_new_frac=cfg.payouts_futures.min_new_profit_frac,
                    balance=bal, threshold=10_000.0, policy="greedy",
                    buffer_usd=cfg.payouts_futures.buffer_usd)
                if req >= cfg.payouts_futures.min_payout_usd:
                    saldo -= req
                    req_total += req
                    qual = 0
                    cycle = 0.0
        saldo += ba.total_pnl
    # post_clamp[k] == observed + clamp_adj por día → Σ clamps ya incluido.
    expected_balance = ACC + sum(post_clamp) - req_total
    assert r.final_balance == pytest.approx(expected_balance, abs=1e-6)
    # Cross-check: lo registrado es 0.9 × Σ request.
    assert r.payouts.sum() == pytest.approx(0.9 * req_total, abs=1e-6)


# ---------------------------------------------------------------------------
# AC-16 políticas greedy vs buffer (a nivel máquina)
# ---------------------------------------------------------------------------

def test_politicas_greedy_vs_buffer():
    # Pro 50K: share 1.0, cap 5000, amount 3000 (buffer default 3000),
    # dll hard 1000 (días positivos: sin hits). 4 días de +1000:
    # balance_pre día 4 = 54000; T_4 = 50000 (lock, cap). greedy request =
    # min(5000, 4000, 8000) = 4000 → balance 50000. buffer: además <=
    # 54000 − 50000 − 3000 = 1000 → request 1000 → balance 53000.
    cfg = _funded(amount=3_000.0, dll_type="hard_fatal", wshare=1.00,
                  cap=5_000.0)
    verdes = _fbda([1, 2, 3, 4], [1000.0, 2000.0, 3000.0, 4000.0],
                   [1000.0, 2000.0, 3000.0, 4000.0])
    r_greedy = _run([verdes], cfg=cfg, policy="greedy")
    r_buffer = _run([verdes], cfg=cfg, policy="buffer_preserving")
    # Ambas producen payouts válidos (registrado = 0.9 × request).
    assert r_greedy.payouts[0] == pytest.approx(4000.0 * 0.9)
    assert r_buffer.payouts[0] == pytest.approx(1000.0 * 0.9)
    # Buffer nunca deja balance_post < T + buffer (53000 >= 50000 + 3000);
    # greedy sí puede quedar por debajo (50000 < 53000).
    assert r_buffer.final_balance == pytest.approx(53_000.0)
    assert r_greedy.final_balance == pytest.approx(50_000.0)
    # Determinismo bit-exact (misma entrada, misma política, sin rng).
    r_greedy2 = _run([verdes], cfg=cfg, policy="greedy")
    assert r_greedy2.dll_hits == r_greedy.dll_hits
    assert r_greedy2.payout_deferrals == r_greedy.payout_deferrals
    assert r_greedy2.final_balance == r_greedy.final_balance
    np.testing.assert_array_equal(r_greedy2.payouts, r_greedy.payouts)
    # La parte de compare_policies_futures (ΔEV + IC pareado sobre el índice
    # común) es Fase 4: exige sequence_ev_futures, que aún no existe.


def test_payout_overflows_cuenta_y_trunca():
    # max_payout_cycles = 1 con dos payouts elegibles: el segundo desborda →
    # payout_overflows = 1 y alive_at_cap (convención de truncado).
    cfg = _funded(qual_min=1.0, days_per=4, max_cycles=1)
    cum = 500.0 * np.arange(1, 9)
    ba = _fbda(np.arange(1, 9), cum, cum.copy())
    r = _run([ba], cfg=cfg)
    assert r.payout_overflows == 1
    assert r.alive_at_cap
    assert (r.payouts > 0).sum() == 1
