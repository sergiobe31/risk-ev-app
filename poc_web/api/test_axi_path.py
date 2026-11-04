"""Tests unitarios del walker F4 del lente AXI (``axi_path.py``).

Criterios cubiertos (ANALISIS_CASOS_PASS_F4.md §10.5):
- A1: look-ahead — la decisión de suelo usa T(d) por APERTURA, no el T
  block-level ni T por cierre (fixtures B y B2 son discriminantes exactos).
- V1: ``min_unique_trades=None`` ⇒ bit-idéntico a ``evaluate_path``.
- V2: paths fuera del cubo degradado ⇒ bit-idénticos al motor.
- V4 (por path): ``raw_outcome(walker)`` ≡ ``evaluate_path`` en TODOS los
  fixtures, incluidos los del cubo degradado.
- Casos 2a/2b/2c de §4, orden ruin→pass→timeout (§10.3-6), fórmula
  ``days_cal`` (§10.3-3) y reparto ``post_target_split`` (V3).

F5b (PROPUESTA_F5_PRECISION_MOTOR.md §8.6): convención dual — V1/V2/V4 anclan
la variante BOUND (``mtm="bound"``, default) y la rama JOINT
(``mtm="joint"``) se ejercita con fixtures multi-símbolo desincronizados
(extremos no simultáneos y ffill cross-sesión, §8.2: la dirección NO es
monótona), equivalencia joint≡bound en mono-símbolo y equivalencia contra la
construcción adjudicada de ``propfirm``. Los fixtures históricos con valle
fuera de [open, close] (conveniencia sintética del bound, que no mira el
tramo) NO son ejercitables en joint: la máscara física [ot, ct] del joint lo
descarta — en datos reales el valle siempre cae dentro del tramo del trade.

Convención merged del motor (la hereda el walker): ``merged(d)`` = saldo
cerrado ANTES del día d + peor flotante de d ⇒ el close de un trade solo es
visible en la equity evaluada a partir del día SIGUIENTE a su cierre.

Uso:  PYTHONPATH=. python -m pytest poc_web/api/test_axi_path.py
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import BlockRecord, evaluate_path

from poc_web.api.axi_path import (evaluate_path_axi, joint_columns_cache,
                                  mean_inter_block_gap, post_target_split,
                                  precompute_joint, precompute_open_days,
                                  raw_outcome)

D0 = dt.date(2024, 1, 1)
PASS, RUIN, BASE = 100.0, -100.0, 1000.0


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n - 1)


def mk_block(idx: int, start_n: int, end_n: int, trades: list[dict]) -> BlockRecord:
    """trades: dict(local_id, open_n, close_n, pnl, valle={n: vp})."""
    start, end = pd.Timestamp(day(start_n)), pd.Timestamp(day(end_n))
    return BlockRecord(
        idx=idx, start=start, end=end,
        duration_days=(end - start).total_seconds() / 86400.0,
        sym_id=np.zeros(len(trades), dtype=int),
        trade_local_id=np.array([t["local_id"] for t in trades], dtype=int),
        close_d=np.array([day(t["close_n"]) for t in trades]),
        valle_precio=[{day(n): vp for n, vp in t.get("valle", {}).items()}
                      for t in trades],
        pnl_per_lot=np.array([t["pnl"] for t in trades], dtype=float),
        n_unique_trades=len(trades),
    )


def mk_env(blocks, n_trades: int, open_ns: list[int], gap_mean: float = 5.0):
    """Piezas para los walkers: lots=1, pv=1, open_days por apertura real."""
    lots_by = {0: np.ones((1, n_trades))}
    pv_by = {0: 1.0}
    strats = {0: SimpleNamespace(
        open_t=pd.to_datetime([day(n) for n in open_ns]).to_numpy())}
    open_days = precompute_open_days(blocks, strats)
    return lots_by, pv_by, open_days, gap_mean


def run_engine(blocks, lots_by, pv_by, horizon=None):
    return evaluate_path(np.arange(len(blocks)), blocks, lots_by, pv_by, 0,
                         pass_line=PASS, ruin_line=RUIN, baseline=BASE,
                         horizon_days=horizon)


def run_f4(blocks, env, min_trades, horizon=None,
           pass_line=PASS, ruin_line=RUIN):
    """Walker en modo BOUND (default) — ancla de fidelidad V1/V2/V4 (§8.6-2)."""
    lots_by, pv_by, open_days, gap_mean = env
    return evaluate_path_axi(np.arange(len(blocks)), blocks, open_days,
                             lots_by, pv_by, 0, pass_line=pass_line,
                             ruin_line=ruin_line, baseline=BASE,
                             min_unique_trades=min_trades, gap_mean=gap_mean,
                             horizon_days=horizon, mtm="bound")


def run_f4_joint(blocks, env, min_trades, minute_by_sym, horizon=None,
                 pass_line=PASS, ruin_line=RUIN):
    """Walker en modo JOINT (walker expuesto F5b) con minute_valle sintético."""
    lots_by, pv_by, open_days, gap_mean = env
    cols = joint_columns_cache(blocks, minute_by_sym)
    pre = precompute_joint(blocks, lots_by, pv_by, 0, minute_by_sym, cols)
    return evaluate_path_axi(np.arange(len(blocks)), blocks, open_days,
                             lots_by, pv_by, 0, pass_line=pass_line,
                             ruin_line=ruin_line, baseline=BASE,
                             min_unique_trades=min_trades, gap_mean=gap_mean,
                             horizon_days=horizon, mtm="joint", joint_pre=pre)


def assert_same(a_out, a_days, a_trades_block, a_max_dd, a_terminal, eng):
    """Bit-identidad contra el motor (campos con semántica engine)."""
    assert a_out == eng.outcome
    assert a_days == eng.days
    assert a_trades_block == eng.trades
    assert a_max_dd == eng.max_dd
    assert a_terminal == eng.terminal


# --- FIX A: caso 1 — pass cualificado en el primer toque --------------------
def fix_a():
    blocks = [mk_block(0, 1, 5, [
        {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 150.0},
        {"local_id": 1, "open_n": 2, "close_n": 4, "pnl": 10.0,
         "valle": {2: -1.0, 3: -1.0, 4: -1.0}},
    ])]
    return blocks, mk_env(blocks, 2, [1, 2])


# --- FIX B: A1 — T block-level habría pasado; T(d)-apertura NO --------------
def fix_b():
    blocks = [mk_block(0, 1, 9, [
        {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 150.0},
        {"local_id": 1, "open_n": 3, "close_n": 9, "pnl": -400.0,
         "valle": {2: -0.5, 3: -400.0}},
    ])]
    return blocks, mk_env(blocks, 2, [1, 3])


# --- FIX B2: A1 — apertura vs cierre: con T por CIERRE sería ruin -----------
def fix_b2():
    blocks = [mk_block(0, 1, 9, [
        {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 150.0},
        {"local_id": 1, "open_n": 2, "close_n": 9, "pnl": 1.0,
         "valle": {2: -0.5, 5: -400.0}},
    ])]
    return blocks, mk_env(blocks, 2, [1, 2])


# --- FIX C: 2a — toque con T bajo, continuación muere -----------------------
def fix_c():
    blocks = [
        mk_block(0, 1, 3, [
            {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 120.0,
             "valle": {2: -0.5}}]),
        mk_block(1, 4, 7, [
            {"local_id": 1, "open_n": 4, "close_n": 6, "pnl": -400.0,
             "valle": {4: -260.0, 5: -260.0, 6: -260.0}}]),
    ]
    return blocks, mk_env(blocks, 2, [1, 4])


# --- FIX D: 2b — toque con T bajo, continuación cualifica -------------------
def fix_d():
    blocks = [
        mk_block(0, 1, 3, [
            {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 120.0,
             "valle": {2: -0.5}}]),
        mk_block(1, 4, 7, [
            {"local_id": 1, "open_n": 4, "close_n": 6, "pnl": 5.0,
             "valle": {4: -1.0}}]),
    ]
    return blocks, mk_env(blocks, 2, [1, 4])


# --- FIX E: 2c — toque con T bajo, nunca re-cualifica, fin de datos ---------
def fix_e():
    blocks = [
        mk_block(0, 1, 3, [
            {"local_id": 0, "open_n": 1, "close_n": 1, "pnl": 120.0,
             "valle": {2: -0.5}}]),
        mk_block(1, 4, 8, [
            {"local_id": 1, "open_n": 4, "close_n": 6, "pnl": -60.0,
             "valle": {4: -30.0, 6: -30.0, 7: -0.5}}]),
    ]
    return blocks, mk_env(blocks, 2, [1, 4])


# --- FIX G: caso 6 — ruin primero -------------------------------------------
def fix_g():
    blocks = [mk_block(0, 1, 3, [
        {"local_id": 0, "open_n": 1, "close_n": 2, "pnl": -150.0,
         "valle": {1: -120.0, 2: -120.0}},
    ])]
    return blocks, mk_env(blocks, 1, [1])


# --- FIX H: sin cruce alguno (para horizon / neither) ------------------------
def fix_h():
    blocks = [mk_block(0, 1, 4, [
        {"local_id": 0, "open_n": 1, "close_n": 3, "pnl": 10.0,
         "valle": {1: -1.0, 3: -1.0}},
    ])]
    return blocks, mk_env(blocks, 1, [1])


ALL_FIXES = [fix_a, fix_b, fix_b2, fix_c, fix_d, fix_e, fix_g, fix_h]


# ---------------------------------------------------------------------------
# V4 por path: raw_outcome(walker F4) ≡ motor, en TODOS los fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fix", ALL_FIXES, ids=[f.__name__ for f in ALL_FIXES])
def test_v4_raw_derivado_del_first_cross_es_el_motor(fix):
    blocks, env = fix()
    eng = run_engine(blocks, env[0], env[1])
    o = run_f4(blocks, env, min_trades=2)
    raw = raw_outcome(o)
    assert (raw.outcome, raw.days, raw.trades, raw.max_dd, raw.terminal) == \
           (eng.outcome, eng.days, eng.trades, eng.max_dd, eng.terminal)


# ---------------------------------------------------------------------------
# V1: min_unique_trades=None ⇒ degenera bit-idéntico al motor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fix", ALL_FIXES, ids=[f.__name__ for f in ALL_FIXES])
def test_v1_sin_suelo_degenera_al_motor(fix):
    blocks, env = fix()
    eng = run_engine(blocks, env[0], env[1])
    o = run_f4(blocks, env, min_trades=None)
    assert_same(o.outcome, o.days, o.trades_block, o.max_dd, o.terminal, eng)


# ---------------------------------------------------------------------------
# V2: fuera del cubo degradado, el walker F4 es bit-idéntico al motor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fix", ALL_FIXES, ids=[f.__name__ for f in ALL_FIXES])
def test_v2_fuera_del_cubo_degradado_bit_identico(fix):
    blocks, env = fix()
    eng = run_engine(blocks, env[0], env[1])
    o = run_f4(blocks, env, min_trades=2)
    degradado = o.first_cross == "pass" and o.fc_trades_day < 2
    if degradado:
        pytest.skip("path del cubo caso-2: cambia por diseño")
    assert_same(o.outcome, o.days, o.trades_block, o.max_dd, o.terminal, eng)


# ---------------------------------------------------------------------------
# Casos concretos
# ---------------------------------------------------------------------------
def test_caso1_pass_cualificado_al_primer_toque():
    blocks, env = fix_a()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "pass"
    assert o.trades == 2 and o.trades_block == 2
    assert o.first_cross == "pass" and o.fc_trades_day == 2
    assert o.n_blocks == 1
    # days_seen = duración del bloque completo (semántica motor): 4.0
    assert o.days == 4.0
    assert o.days_cal == 4.0 + (1 - 1) * 5.0  # k = n_bloques − 1


def test_a1_lookahead_block_level_habria_pasado_per_day_no():
    """FIX B: d2 toca target con T(d)=1 (T block-level=2); d3 el bloque se
    hunde. Con T block-level el walker se habría detenido en pass en d2; con
    T(d) por día continúa y muere: outcome ruin con first_cross pass."""
    blocks, env = fix_b()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "ruin"               # 2a destapado
    assert o.first_cross == "pass"           # el motor lo cantó pass
    assert o.fc_trades_day == 1              # T(d) real en el toque
    assert o.fc_trades_block == 2            # T block-level (lo que veía F2b)
    assert raw_outcome(o).outcome == "pass"  # raw = el motor


def test_a1_apertura_vs_cierre_conteo_por_apertura():
    """FIX B2: el trade 2 ABRE d2 y CIERRA d9; el bloque se hunde en d5.
    Contando por apertura, T(d2)=2 ⇒ pass en d2. Contando por cierre,
    T(d2)=1 ⇒ continuaría y moriría en d5. Outcome pass fija la apertura."""
    blocks, env = fix_b2()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "pass"
    assert o.trades == 2


def test_caso2a_ruin_encubierta():
    blocks, env = fix_c()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "ruin"
    assert o.first_cross == "pass" and o.fc_trades_day == 1
    assert o.n_blocks == 2
    # max_dd des-censurado: el motor se quedó en el toque (max_dd 0.0)
    eng = run_engine(blocks, env[0], env[1])
    assert eng.outcome == "pass" and eng.max_dd == 0.0
    assert o.max_dd == pytest.approx(-259.5 / BASE)


def test_caso2b_pass_recuperado():
    blocks, env = fix_d()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "pass"
    assert o.trades == 2                       # T(d)-apertura en el stop
    assert o.first_cross == "pass" and o.fc_trades_day == 1
    assert o.days == 2.0 + 3.0                 # dur(b1)+dur(b2), semántica motor
    assert o.days_cal == o.days + (2 - 1) * 5.0


def test_caso2c_sin_resolver_al_agotarse_datos():
    blocks, env = fix_e()
    o = run_f4(blocks, env, min_trades=2)
    assert o.outcome == "neither"
    assert o.first_cross == "pass" and o.fc_trades_day == 1
    assert o.trades == 2                       # juntó T pero E quedó < target


def test_horizon_con_t_bajo_es_timeout_y_raw_es_pass():
    """Orden ruin→pass→timeout (§10.3-6): agotar horizonte con T<min es
    "timeout" (sin resolver); el motor (raw) ya había cantado pass antes."""
    blocks, env = fix_e()
    horizon = 2.5                              # dentro del bloque 2 (acumulado)
    o = run_f4(blocks, env, min_trades=2, horizon=horizon)
    assert o.outcome == "timeout"
    assert o.first_cross == "pass"
    eng = run_engine(blocks, env[0], env[1], horizon=horizon)
    assert eng.outcome == "pass"               # pass se chequea antes que horizon
    assert raw_outcome(o).outcome == "pass"


def test_horizon_sin_cruce_bit_identico_al_motor():
    blocks, env = fix_h()
    eng = run_engine(blocks, env[0], env[1], horizon=1.0)
    o = run_f4(blocks, env, min_trades=2, horizon=1.0)
    assert eng.outcome == "timeout"
    assert_same(o.outcome, o.days, o.trades_block, o.max_dd, o.terminal, eng)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_mean_inter_block_gap_runtime():
    b1 = mk_block(0, 1, 3, [{"local_id": 0, "open_n": 1, "close_n": 2,
                             "pnl": 1.0}])
    b2 = mk_block(1, 5, 9, [{"local_id": 0, "open_n": 5, "close_n": 8,
                             "pnl": 1.0}])
    # span = 8d; Σdur = 2+4 = 6 ⇒ ḡ = (8−6)/(2−1) = 2.0
    assert mean_inter_block_gap([b1, b2]) == pytest.approx(2.0)
    assert mean_inter_block_gap([b1]) == 0.0


def test_post_target_split_reparte_el_cubo():
    outs = []
    for fix in (fix_c, fix_d, fix_e):        # 2a, 2b, 2c respectivamente
        blocks, env = fix()
        outs.append(run_f4(blocks, env, min_trades=2))
    blocks, env = fix_a()                    # caso 1: NO entra en el cubo
    outs.append(run_f4(blocks, env, min_trades=2))
    rep = post_target_split(outs, 2)
    assert rep == {"n_deg": 3, "n_ruin": 1, "n_pass": 1, "n_unresolved": 1}
    # Sin suelo (Pro M): cubo vacío por definición
    assert post_target_split(outs, None) == {"n_deg": 0, "n_ruin": 0,
                                             "n_pass": 0, "n_unresolved": 0}


def test_pureza_no_muta_inputs():
    blocks, env = fix_c()
    snap_dur = [b.duration_days for b in blocks]
    snap_n = [b.n_unique_trades for b in blocks]
    snap_od = [od.copy() for od in env[2]]
    run_f4(blocks, env, min_trades=2)
    assert [b.duration_days for b in blocks] == snap_dur
    assert [b.n_unique_trades for b in blocks] == snap_n
    for before, after in zip(snap_od, env[2]):
        assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# F5b — rama JOINT (§8.6): helpers, fixtures multi-símbolo y tests
# ---------------------------------------------------------------------------
def mk_minute(points: dict, ot: pd.Timestamp, ct: pd.Timestamp) -> tuple:
    """Entrada ``minute_valle`` sintética (misma forma que
    ``attach_minute_valleys``): ``points`` = {timestamp: flotante en precio}.
    """
    keys = sorted(points, key=pd.Timestamp)
    ts = pd.to_datetime(keys).to_numpy()
    vals = np.array([points[k] for k in keys], dtype=float)
    return (ts, vals, ot, ct)


def mk_block_multi(idx: int, start_n: int, end_n: int,
                   trades: list[dict]) -> BlockRecord:
    """trades: dict(sym, local_id, open_n, close_n, pnl, valle={n: vp})."""
    start, end = pd.Timestamp(day(start_n)), pd.Timestamp(day(end_n))
    return BlockRecord(
        idx=idx, start=start, end=end,
        duration_days=(end - start).total_seconds() / 86400.0,
        sym_id=np.array([t["sym"] for t in trades], dtype=int),
        trade_local_id=np.array([t["local_id"] for t in trades], dtype=int),
        close_d=np.array([day(t["close_n"]) for t in trades]),
        valle_precio=[{day(n): vp for n, vp in t.get("valle", {}).items()}
                      for t in trades],
        pnl_per_lot=np.array([t["pnl"] for t in trades], dtype=float),
        n_unique_trades=len(trades),
    )


def mk_env_multi(blocks, opens_by_sym: dict, gap_mean: float = 5.0):
    """opens_by_sym: {sym: [open_n por trade local]} (orden de inserción =
    sym_id, como en la construcción real de bloques)."""
    lots_by = {s: np.ones((1, len(v))) for s, v in opens_by_sym.items()}
    pv_by = {s: 1.0 for s in opens_by_sym}
    strats = {s: SimpleNamespace(
        open_t=pd.to_datetime([day(n) for n in v]).to_numpy())
        for s, v in opens_by_sym.items()}
    open_days = precompute_open_days(blocks, strats)
    return lots_by, pv_by, open_days, gap_mean


def minute_map_from_blocks(blocks) -> dict:
    """``minute_by_sym`` sintético que reproduce EXACTAMENTE el valle diario
    de cada trade (los sin valle no tienen entrada, paridad BAJO-3). Cada día
    con valle: 00:00→0, 12:00→vp, 23:59→0 (el último punto evita que el ffill
    arrastre el valle a días sin declarar — en la serie real hay barras M1
    cada día de la posición)."""
    out: dict[int, dict] = {}
    for b in blocks:
        for j in range(len(b.sym_id)):
            vp = b.valle_precio[j]
            if not vp:
                continue
            s, lid = int(b.sym_id[j]), int(b.trade_local_id[j])
            points = {}
            for d, v in vp.items():
                t0 = pd.Timestamp(d)
                points[t0] = 0.0
                points[t0 + pd.Timedelta(hours=12)] = v
                points[t0 + pd.Timedelta(hours=23, minutes=59)] = 0.0
            ds = sorted(vp)
            out.setdefault(s, {})[lid] = mk_minute(
                points, ot=pd.Timestamp(ds[0]),
                ct=pd.Timestamp(ds[-1]) + pd.Timedelta(hours=23, minutes=59))
    return out


# --- FIX J1: dos patas vivas con extremos NO simultáneos (joint < bound) ----
def fix_j1():
    """§8.2 dirección típica: el bound suma ambos valles (corr=1: −200); el
    joint solo ve −100 (los mínimos no coinciden en el minuto)."""
    trades = [
        {"sym": 0, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": 0.0,
         "valle": {2: -100.0}},
        {"sym": 1, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": 0.0,
         "valle": {2: -100.0}},
    ]
    blocks = [mk_block_multi(0, 1, 3, trades)]
    env = mk_env_multi(blocks, {0: [1], 1: [1]})
    d2 = pd.Timestamp(day(2))
    minute = {
        0: {0: mk_minute({d2.replace(hour=10): -100.0,
                          d2.replace(hour=14): 0.0},
                         ot=d2.replace(hour=9), ct=d2.replace(hour=17))},
        1: {0: mk_minute({d2.replace(hour=10): 0.0,
                          d2.replace(hour=14): -100.0},
                         ot=d2.replace(hour=9), ct=d2.replace(hour=17))},
    }
    return blocks, env, minute


# --- FIX J2: ffill cross-sesión (joint > bound — §8.2 NO es monótona) -------
def fix_j2():
    """Pata A viva hasta d3 pero SIN barras M1 en d3: el ffill físico arrastra
    su flotante adverso (−60) a d3; el bound no le aplica valle en d3 (no hay
    valle sin barras) ⇒ joint (−110) MÁS adverso que bound (−50)."""
    trades = [
        {"sym": 0, "local_id": 0, "open_n": 1, "close_n": 3, "pnl": 0.0,
         "valle": {1: 0.0, 2: -60.0}},
        {"sym": 1, "local_id": 0, "open_n": 3, "close_n": 3, "pnl": 0.0,
         "valle": {3: -50.0}},
    ]
    blocks = [mk_block_multi(0, 1, 4, trades)]
    env = mk_env_multi(blocks, {0: [1], 1: [3]})
    d2, d3 = pd.Timestamp(day(2)), pd.Timestamp(day(3))
    minute = {
        0: {0: mk_minute({d2.replace(hour=16): -60.0},
                         ot=pd.Timestamp(day(1)).replace(hour=9),
                         ct=d3.replace(hour=18))},  # viva d1-d3, sin barras d3
        1: {0: mk_minute({d3.replace(hour=10): -50.0},
                         ot=d3.replace(hour=9), ct=d3.replace(hour=18))},
    }
    return blocks, env, minute


def test_joint_desincronizado_menos_adverso_que_bound():
    """J1: bound toca ruina (−200 ≤ −150) y joint sobrevive (−100); el raw del
    modo joint sigue siendo el motor (ruin)."""
    blocks, env, minute = fix_j1()
    o_b = run_f4(blocks, env, min_trades=2, ruin_line=-150.0)
    o_j = run_f4_joint(blocks, env, min_trades=2, minute_by_sym=minute,
                       ruin_line=-150.0)
    assert o_b.outcome == "ruin"
    assert o_j.outcome == "neither"          # joint MENOS adverso aquí
    assert o_j.first_cross == "ruin"         # snapshot SIEMPRE bound
    assert raw_outcome(o_j).outcome == "ruin"


def test_joint_ffill_mas_adverso_que_bound():
    """J2: bound sobrevive (−50 > −80) y joint toca ruina (−110 ≤ −80): la
    dirección NO es monótona (§8.2). Raw (motor) = neither."""
    blocks, env, minute = fix_j2()
    o_b = run_f4(blocks, env, min_trades=2, ruin_line=-80.0)
    o_j = run_f4_joint(blocks, env, min_trades=2, minute_by_sym=minute,
                       ruin_line=-80.0)
    assert o_b.outcome == "neither"
    assert o_j.outcome == "ruin"             # joint MÁS adverso aquí
    assert o_j.trades == 2                   # T(d3) por apertura: A abrió d1, B d3
    assert raw_outcome(o_j).outcome == "neither"


# Mono-símbolo/un-vivo-por-minuto con valle dentro del tramo del trade (los
# demás fixtures usan valle fuera de [open, close] como conveniencia del
# bound; la máscara física del joint lo descarta y joint≢bound por diseño).
MONO_EQUIV_FIXES = [fix_a, fix_b2, fix_g, fix_h]


@pytest.mark.parametrize("fix", MONO_EQUIV_FIXES,
                         ids=[f.__name__ for f in MONO_EQUIV_FIXES])
def test_joint_equivale_bound_un_solo_vivo(fix):
    """Un solo trade vivo por minuto ⇒ min de la suma = valle del trade:
    joint ≡ bound EXACTAMENTE (oráculo del espejo ``_block_day_eval_joint``)
    y el walker joint reproduce el outcome bound."""
    from poc_web.api.axi_path import _block_day_eval_joint
    from src.risk.mc_portfolio import _block_day_eval
    blocks, env = fix()
    minute = minute_map_from_blocks(blocks)
    lots_by, pv_by, _, _ = env
    cols = joint_columns_cache(blocks, minute)
    for bi, b in enumerate(blocks):
        days_b, merged_b, tot_b = _block_day_eval(b, lots_by, pv_by, 0, 0.0)
        days_j, merged_j, tot_j = _block_day_eval_joint(
            b, np.ones(len(b.sym_id)), pv_by, minute, cols[bi])
        assert list(days_b) == list(days_j)
        np.testing.assert_array_equal(merged_b, merged_j)
        assert tot_b == tot_j
    o_b = run_f4(blocks, env, min_trades=2)
    o_j = run_f4_joint(blocks, env, min_trades=2, minute_by_sym=minute)
    assert (o_j.outcome, o_j.days, o_j.trades, o_j.max_dd, o_j.terminal,
            o_j.days_cal) == (o_b.outcome, o_b.days, o_b.trades, o_b.max_dd,
                              o_b.terminal, o_b.days_cal)


def test_joint_equivale_construccion_propfirm():
    """Equivalencia (§8.6-2): la construcción joint del walker coincide con la
    adjudicada de ``propfirm.block_day_arrays_portfolio(float_mode="joint")``
    sobre la misma materia prima (rel_float = merged0, convención de prefijo
    saldo-antes-del-día)."""
    from src.risk.propfirm import (PortfolioDayBlock,
                                   block_day_arrays_portfolio)
    blocks, env, minute = fix_j1()
    b = blocks[0]
    lots_by, pv_by, _, _ = env
    cols = joint_columns_cache(blocks, minute)
    days_w, merged0_w, tot_w = precompute_joint(
        blocks, lots_by, pv_by, 0, minute, cols)[0]
    pdb = PortfolioDayBlock(
        idx=0, start=b.start, end=b.end, duration_days=b.duration_days,
        sym_id=b.sym_id, trade_local_id=b.trade_local_id,
        open_ord=np.array([pd.Timestamp(day(1)).toordinal()] * 2),
        close_ord=np.array([pd.Timestamp(day(2)).toordinal()] * 2),
        valle_precio=b.valle_precio, pnl_per_lot=b.pnl_per_lot,
        n_trades=2, n_no_valley=0)
    arr = block_day_arrays_portfolio(pdb, lots_by, pv_by, 0,
                                     float_mode="joint", minute_by_sym=minute)
    pf = {dt.date.fromordinal(int(o)): v
          for o, v in zip(arr.day_ord, arr.rel_float)}
    assert len(days_w) > 0
    for d, m in zip(days_w, merged0_w):
        assert d in pf, f"día {d} ausente en la construcción propfirm"
        assert m == pytest.approx(pf[d], abs=1e-12)


def test_joint_minute_valle_vacio_y_trade_sin_cobertura():
    """``minute_valle={}`` (dict vacío ≠ None, §8.5): sin series, flotante 0 —
    merged0 = prefijo cerrado. Trade sin entrada (BAJO-3): aporta 0 al
    flotante pero su PnL cerrado cuenta."""
    from poc_web.api.axi_path import _block_day_eval_joint
    blocks0 = [mk_block_multi(0, 1, 3, [
        {"sym": 0, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": 50.0},
        {"sym": 1, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": -20.0},
    ])]
    pv_by = {0: 1.0, 1: 1.0}
    lots = np.ones(2)
    # (a) todo vacío: cols None → sin flotante
    cols = joint_columns_cache(blocks0, {0: {}, 1: {}})
    days0, m0, tot0 = _block_day_eval_joint(
        blocks0[0], lots, pv_by, {0: {}, 1: {}}, cols[0])
    assert tot0 == pytest.approx(30.0)          # el PnL cerrado cuenta
    np.testing.assert_array_equal(m0, [0.0])    # prefijo ANTES de d2 = 0
    # (b) solo la pata 0 con serie (la 1 aporta 0 al flotante)
    blocks1 = [mk_block_multi(0, 1, 3, [
        {"sym": 0, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": 50.0,
         "valle": {2: -40.0}},
        {"sym": 1, "local_id": 0, "open_n": 1, "close_n": 2, "pnl": -20.0},
    ])]
    d2 = pd.Timestamp(day(2))
    minute = {0: {0: mk_minute({d2.replace(hour=10): -40.0,
                                d2.replace(hour=14): 0.0},
                               ot=d2.replace(hour=9), ct=d2.replace(hour=17))},
              1: {}}
    cols1 = joint_columns_cache(blocks1, minute)
    days1, m1, tot1 = _block_day_eval_joint(
        blocks1[0], lots, pv_by, minute, cols1[0])
    assert tot1 == pytest.approx(30.0)
    np.testing.assert_array_equal(m1, [-40.0])  # solo aporta la pata con serie


@pytest.mark.parametrize("fix", ALL_FIXES + [fix_j1, fix_j2],
                         ids=[f.__name__ for f in ALL_FIXES + [fix_j1, fix_j2]])
def test_v4_joint_raw_es_motor(fix):
    """V4 en modo joint: ``raw_outcome(walker joint)`` ≡ ``evaluate_path`` en
    TODOS los fixtures (el sub-pase bound es motor-exacto, §8.6-1)."""
    if fix in (fix_j1, fix_j2):
        blocks, env, minute = fix()
    else:
        blocks, env = fix()
        minute = minute_map_from_blocks(blocks)
    eng = run_engine(blocks, env[0], env[1])
    o = run_f4_joint(blocks, env, min_trades=2, minute_by_sym=minute)
    raw = raw_outcome(o)
    assert (raw.outcome, raw.days, raw.trades, raw.max_dd, raw.terminal) == \
           (eng.outcome, eng.days, eng.trades, eng.max_dd, eng.terminal)


def test_mtm_desconocido_y_joint_sin_pre():
    blocks, env = fix_a()
    lots_by, pv_by, open_days, gap_mean = env
    with pytest.raises(ValueError, match="mtm"):
        evaluate_path_axi(np.arange(1), blocks, open_days, lots_by, pv_by, 0,
                          pass_line=PASS, ruin_line=RUIN, baseline=BASE,
                          min_unique_trades=2, gap_mean=gap_mean, mtm="x")
    with pytest.raises(ValueError, match="joint_pre"):
        evaluate_path_axi(np.arange(1), blocks, open_days, lots_by, pv_by, 0,
                          pass_line=PASS, ruin_line=RUIN, baseline=BASE,
                          min_unique_trades=2, gap_mean=gap_mean, mtm="joint")
