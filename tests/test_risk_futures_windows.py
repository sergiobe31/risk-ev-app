"""Fase B del plan float-valleys-ventanas: sizing Opción A + float_mode="windows".

Plan float-valleys-ventanas (W2, W4, B1-B8,
enmiendas V2-V5, V9). Los goldens numéricos están escritos a mano en cada test.

Cobertura de aceptación:
  B1  test_b1_*            golden de ventanas (1 y 2 lotes, con/sin comisión)
  B2  test_b2_*            trade que cruza sesión (labeler 16:10)
  B3  test_b3_*            empate de instante: ambos vivos
  B4  test_b4_*            walker eod_trailing e2e (FTMO Futures Growth 50K)
  B5  test_b5_*            fixed_risk_micros + diag (V5)
  B6  test_b6_*            fail-loud
  V2  test_v2_*            delta de cierre en la banda negativa de tolerancia
  B8  test_b8_*            default bound intacto (la regresión bit-idéntica la
                           cubre el resto de la suite sin tocar)
Extra: oráculo independiente por estados (property test aleatorio con empates).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.risk.futures_legs import (
    FuturesLegData, fixed_risk_micros, strategy_from_trade_table,
)
from src.risk.propfirm import (
    BlockDayArrays, PortfolioDayBlock, _windows_day_float,
    block_day_arrays_portfolio, build_day_blocks_portfolio,
    load_propfirm_config, precompute_day_arrays_portfolio,
)
from src.risk.propfirm_futures import evaluate_phase_path_eod_trail
from src.risk.session_axis import make_session_day_labeler

D0 = pd.Timestamp("2026-03-02")      # lunes
D1 = pd.Timestamp("2026-03-03")
D2 = pd.Timestamp("2026-03-04")


def ts(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    h, m = hhmm.split(":")
    return day + pd.Timedelta(hours=int(h), minutes=int(m))


def table(rows, comm: float = 0.0) -> pd.DataFrame:
    """rows: (open_ts, close_ts, lots, pnl_net, mae_usd). Precios dummy: SL 1 punto."""
    return pd.DataFrame({
        "open_ts": pd.to_datetime([r[0] for r in rows]),
        "close_ts": pd.to_datetime([r[1] for r in rows]),
        "lots": [float(r[2]) for r in rows],
        "pnl_net": [float(r[3]) for r in rows],
        "mae_usd": [float(r[4]) for r in rows],
        "open_px": 100.0, "close_px": 100.0, "sl_px": 99.0,
        "is_buy": True,
        "comm_usd": [comm * float(r[2]) for r in rows],
    })


META = dict(symbol="@micro_nasdaq", pv=2.0, tick_size=0.25,
            backtest_precision="Selected TF", strategy_name="t")


def build(legs: dict, lots_by_leg: dict, day_labeler=None):
    """legs {key: DataFrame}; lots_by_leg {key: [lots por trade]} (un f).

    Devuelve (blocks, lots_by_sym, pv_by_sym, adverse_by_sym).
    """
    strategies, adv, lots_by_sym, pv = {}, {}, {}, {}
    for si, (key, df) in enumerate(legs.items()):
        st, leg = strategy_from_trade_table(df, META, key, day_labeler=day_labeler)
        strategies[key] = st
        adv[si] = leg.adverse_per_lot
        lots_by_sym[si] = np.array([lots_by_leg[key]], dtype=float)
        pv[si] = st.pv
    blocks = build_day_blocks_portfolio(strategies, day_labeler=day_labeler)
    return blocks, lots_by_sym, pv, adv


def direct_block(trades, day_labeler=None) -> PortfolioDayBlock:
    """Bloque a mano. trades: (sym, local_id, open_ts, close_ts, pnl_per_lot)."""
    k = len(trades)
    ot = np.array([t[2] for t in trades], dtype="datetime64[ns]")
    ct = np.array([t[3] for t in trades], dtype="datetime64[ns]")
    return PortfolioDayBlock(
        idx=0, start=pd.Timestamp(ot.min()), end=pd.Timestamp(ct.max()),
        duration_days=1.0,
        sym_id=np.array([t[0] for t in trades]),
        trade_local_id=np.array([t[1] for t in trades]),
        open_ord=np.array([pd.Timestamp(t[2]).toordinal() for t in trades]),
        close_ord=np.array([pd.Timestamp(t[3]).toordinal() for t in trades]),
        valle_precio=[{} for _ in range(k)],
        pnl_per_lot=np.array([t[4] for t in trades], dtype=float),
        n_trades=k, n_no_valley=k, open_t=ot, close_t=ct,
    )


def windows_bda(b, lots, adv_by_sym):
    return block_day_arrays_portfolio(
        b, {s: np.array([lots[s]]) for s in lots}, {s: 2.0 for s in lots}, 0,
        float_mode="windows", adverse_by_sym=adv_by_sym)


# ---------------------------------------------------------------------------
# strategy_from_trade_table
# ---------------------------------------------------------------------------

def test_strategy_from_trade_table_campos_y_contrato():
    df = table([(ts(D0, "09:00"), ts(D0, "11:00"), 2, 100.0, 150.0)], comm=1.12)
    df["open_px"], df["sl_px"] = 21000.0, 20972.0
    st, leg = strategy_from_trade_table(df, META, "k")
    assert st.key == "k" and st.n_trades == 1 and st.valle == {}
    assert st.lots.shape == (0, 1) and st.pv == 2.0 and bool(st.is_buy[0])
    assert st.open_t.dtype == np.dtype("datetime64[ns]")
    assert st.close_t.dtype == np.dtype("datetime64[ns]")
    assert pd.Timestamp(st.close_d[0]).toordinal() == D0.toordinal()
    np.testing.assert_allclose(st.pnl_per_lot, [50.0])               # 100/2
    assert isinstance(leg, FuturesLegData)
    np.testing.assert_allclose(leg.adverse_per_lot, [(150 + 2.24) / 2])
    np.testing.assert_allclose(leg.comm_per_lot, [1.12])
    np.testing.assert_allclose(leg.risk_per_micro, [28.0 * 2.0])     # |Δpx|·pv
    np.testing.assert_allclose(leg.pnl_per_lot, [50.0])
    assert leg.symbol == "@micro_nasdaq" and leg.pv == 2.0
    assert leg.backtest_precision == "Selected TF"


def test_strategy_from_trade_table_labeler_etiqueta_close_d():
    lab = make_session_day_labeler("America/New_York", cut_hhmm="16:10")
    df = table([(ts(D0, "15:00"), ts(D1, "10:00"), 1, 50.0, 200.0),
                (ts(D0, "10:00"), ts(D0, "16:30"), 1, 5.0, 10.0)])
    st, _ = strategy_from_trade_table(df, META, "k", day_labeler=lab)
    # cierre D1 10:00 → sesión D1; cierre D0 16:30 (≥16:10) → sesión D1.
    assert [pd.Timestamp(d).toordinal() for d in st.close_d] == [
        D1.toordinal(), D1.toordinal()]
    st2, _ = strategy_from_trade_table(df, META, "k")
    assert [pd.Timestamp(d).toordinal() for d in st2.close_d] == [
        D1.toordinal(), D0.toordinal()]


def test_strategy_from_trade_table_fail_loud():
    df = table([(ts(D0, "09:00"), ts(D0, "11:00"), 1, 1.0, 1.0)])
    with pytest.raises(ValueError, match="columnas"):
        strategy_from_trade_table(df.drop(columns=["mae_usd"]), META, "k")
    bad = df.copy()
    bad["lots"] = 0.0
    with pytest.raises(ValueError, match="lots"):
        strategy_from_trade_table(bad, META, "k")
    bad["lots"] = -1.0
    with pytest.raises(ValueError, match="lots"):
        strategy_from_trade_table(bad, META, "k")
    with pytest.raises(ValueError, match="pv"):
        strategy_from_trade_table(df, {"symbol": "x"}, "k")
    # C2: SL == open → riesgo inicial nulo.
    nosl = df.copy()
    nosl["sl_px"] = nosl["open_px"]
    with pytest.raises(ValueError, match="sl_px == open_px"):
        strategy_from_trade_table(nosl, META, "k")


def test_c4_windows_nat_en_open_t_o_close_t_es_error():
    for col in ("open_t", "close_t"):
        b = direct_block([(0, 0, ts(D1, "09:00"), ts(D1, "10:00"), 10.0)])
        arr = getattr(b, col).copy()
        arr[0] = np.datetime64("NaT")
        setattr(b, col, arr)
        with pytest.raises(ValueError, match="NaT"):
            _windows_day_float(b, np.ones(1), {0: np.array([1.0])})


# ---------------------------------------------------------------------------
# B1 golden de ventanas
# ---------------------------------------------------------------------------

def _abc_rows():
    return [
        (ts(D1, "09:00"), ts(D1, "11:00"), 1, 100.0, 150.0),   # A
        (ts(D1, "13:00"), ts(D1, "15:00"), 1, -120.0, 120.0),  # B
        (ts(D1, "14:00"), ts(D1, "15:30"), 1, 40.0, 80.0),     # C
    ]


def _b1_legs(comm: float):
    """Pata 'main' (E el día 0 + A, B, C el día 1) + pata 'cover' (trade de PnL
    0 y flotante 0 que une D0 08:00 → D1 17:00 en UN bloque flat-flat; sin él A
    y B∪C serían bloques separados)."""
    main = table([(ts(D0, "09:00"), ts(D0, "11:00"), 1, 30.0, 10.0)] + _abc_rows(),
                 comm=comm)
    cover = table([(ts(D0, "08:00"), ts(D1, "17:00"), 1, 0.0, 0.0)])
    return {"main": main, "cover": cover}


@pytest.mark.parametrize("n_lots", [1, 2])
def test_b1_golden_sin_comision(n_lots):
    blocks, lots, pv, adv = build(_b1_legs(0.0),
                                  {"main": [n_lots] * 4, "cover": [1]})
    assert len(blocks) == 1
    bda = block_day_arrays_portfolio(blocks[0], lots, pv, 0, float_mode="windows",
                                     adverse_by_sym=adv)
    assert list(bda.day_ord) == [D0.toordinal(), D1.toordinal()]
    # Día D1: la suma de MAE daría −350·n; la ventana da −150·n (A cierra +100
    # antes de abrir B/C). Día D0: E abre (−10n), cierra → −10n.
    np.testing.assert_allclose(bda.rel_before, [0.0, 30.0 * n_lots])
    np.testing.assert_allclose(bda.rel_closed, [30.0 * n_lots, 50.0 * n_lots])
    np.testing.assert_allclose(bda.rel_float, [-10.0 * n_lots,
                                               30.0 * n_lots - 150.0 * n_lots])
    # rel_low del día = rel_before + day_float.
    np.testing.assert_allclose(bda.rel_low, bda.rel_before + np.array(
        [-10.0 * n_lots, -150.0 * n_lots]))
    assert bda.rel_low is not bda.rel_float          # copia, no alias
    assert bda.rel_high is None and bda.rel_exc is None
    assert bda.block_low is None and bda.block_high is None and bda.block_exc is None
    assert (bda.rel_low <= bda.rel_closed + 1e-12).all()


def test_b1_day_float_directo_1_y_2_lotes():
    # Bloque de un solo día (sin cover): −150 con 1 lote, −300 con 2 lotes.
    trades = [(0, 0, ts(D1, "09:00"), ts(D1, "11:00"), 100.0),
              (0, 1, ts(D1, "13:00"), ts(D1, "15:00"), -120.0),
              (0, 2, ts(D1, "14:00"), ts(D1, "15:30"), 40.0)]
    b = direct_block(trades)
    adv = {0: np.array([150.0, 120.0, 80.0])}
    for n, esperado in ((1, -150.0), (2, -300.0)):
        df = _windows_day_float(b, np.full(3, float(n)), adv)
        assert df == {D1.toordinal(): pytest.approx(esperado)}
        # Contra-ejemplo documentado: la suma de MAE da −350·n.
        assert -sum(adv[0]) * n == pytest.approx(-350.0 * n)


def test_b1_golden_con_comision_niveles_y_minimo():
    # comisión RT 1,12/lote sumada al adverso; mismos pnl_net.
    a = np.array([150.0, 120.0, 80.0]) + 1.12
    trades = [(0, 0, ts(D1, "09:00"), ts(D1, "11:00"), 100.0),
              (0, 1, ts(D1, "13:00"), ts(D1, "15:00"), -120.0),
              (0, 2, ts(D1, "14:00"), ts(D1, "15:30"), 40.0)]
    b = direct_block(trades)
    day_float = _windows_day_float(b, np.ones(3), {0: a})
    assert day_float == {D1.toordinal(): pytest.approx(-151.12)}
    # Secuencia de eventos en orden (A open, A close, B open, C open, B close,
    # C close) → niveles −151,12 · +100 · −21,12 · −102,24 · −101,12 · +20.
    deltas = np.array([-a[0], 100.0 + a[0], -a[1], -a[2], -120.0 + a[1], 40.0 + a[2]])
    levels = np.cumsum(deltas)
    np.testing.assert_allclose(levels, [-151.12, 100.0, -21.12, -102.24, -101.12, 20.0],
                               atol=1e-9)
    assert levels.min() == pytest.approx(-151.12)


def test_b1_golden_con_comision_pipeline():
    blocks, lots, pv, adv = build(_b1_legs(1.12),
                                  {"main": [1] * 4, "cover": [1]})
    bda = block_day_arrays_portfolio(blocks[0], lots, pv, 0, float_mode="windows",
                                     adverse_by_sym=adv)
    np.testing.assert_allclose(bda.rel_before, [0.0, 30.0])
    np.testing.assert_allclose(bda.rel_low, [-11.12 + 0.0, 30.0 - 151.12])


def test_b1_precompute_windows_dos_f():
    blocks, lots, pv, adv = build(_b1_legs(0.0), {"main": [1] * 4, "cover": [1]})
    lots2 = {0: np.array([[1.0] * 4, [2.0] * 4]), 1: np.array([[1.0], [1.0]])}
    out = precompute_day_arrays_portfolio(blocks, lots2, pv, 2,
                                          float_mode="windows", adverse_by_sym=adv)
    assert len(out) == 2 and all(len(x) == 1 for x in out)
    np.testing.assert_allclose(out[0][0].rel_low, [-10.0, 30.0 - 150.0])
    np.testing.assert_allclose(out[1][0].rel_low, [-20.0, 60.0 - 300.0])
    # Equivale al camino por bloque.
    ref = block_day_arrays_portfolio(blocks[0], lots2, pv, 1, float_mode="windows",
                                     adverse_by_sym=adv)
    np.testing.assert_allclose(out[1][0].rel_low, ref.rel_low)


# ---------------------------------------------------------------------------
# B2 trade que cruza sesión
# ---------------------------------------------------------------------------

def test_b2_cruce_de_sesion_ambas_sesiones_llevan_w():
    lab = make_session_day_labeler("America/New_York", cut_hhmm="16:10")
    legs = {
        # T abre D0 15:00 (sesión D0), cierra D1 10:00 (sesión D1).
        "t": table([(ts(D0, "15:00"), ts(D1, "10:00"), 1, 50.0, 200.0)], comm=1.12),
        # W abre D1 09:00, cierra D1 11:00: solapa a T → mismo bloque.
        "u": table([(ts(D1, "09:00"), ts(D1, "11:00"), 1, -40.0, 60.0)], comm=1.12),
    }
    blocks, lots, pv, adv = build(legs, {"t": [1], "u": [1]}, day_labeler=lab)
    assert len(blocks) == 1
    b = blocks[0]
    assert list(b.open_ord) == [D0.toordinal(), D1.toordinal()]
    assert list(b.close_ord) == [D1.toordinal(), D1.toordinal()]
    bda = block_day_arrays_portfolio(b, lots, pv, 0, float_mode="windows",
                                     adverse_by_sym=adv, day_labeler=lab)
    assert list(bda.day_ord) == [D0.toordinal(), D1.toordinal()]
    w_t, w_u = -201.12, -61.12
    # Sesión D0: solo T abierto → −201,12. Sesión D1: arrastre T (−201,12),
    # W abre (−262,24), T cierra (+251,12 → −11,12), W cierra (+21,12 → +10).
    np.testing.assert_allclose(bda.rel_float, [w_t, w_t + w_u])
    # Sesión de apertura: nada cerró aún → rel_closed == rel_before.
    assert bda.rel_closed[0] == bda.rel_before[0] == 0.0
    np.testing.assert_allclose(bda.rel_closed, [0.0, 10.0])      # 50 − 40
    assert (bda.rel_low <= bda.rel_closed + 1e-12).all()


def test_b2_cruce_multisesion_incluye_sesiones_intermedias():
    # Trade de 3 sesiones: D0 → D2; la sesión D1 (sin eventos propios) aparece
    # con el arrastre w y el walker la verá como día con rel_low = w.
    b = direct_block([(0, 0, ts(D0, "10:00"), ts(D2, "10:00"), 30.0)])
    f = _windows_day_float(b, np.array([1.0]), {0: np.array([100.0])})
    assert f == {D0.toordinal(): pytest.approx(-100.0),
                 D1.toordinal(): pytest.approx(-100.0),
                 D2.toordinal(): pytest.approx(-100.0)}   # cierra +30: min = arrastre


# ---------------------------------------------------------------------------
# B3 empate de instante
# ---------------------------------------------------------------------------

def test_b3_empate_instante_ambos_vivos():
    t10 = ts(D1, "10:00")
    trades = [(0, 0, ts(D1, "09:00"), t10, 50.0),     # A cierra a las 10:00
              (0, 1, t10, ts(D1, "11:00"), 10.0)]     # B abre a las 10:00 (mismo ns)
    b = direct_block(trades)
    f = _windows_day_float(b, np.ones(2), {0: np.array([100.0, 80.0])})
    # A open −100 · B open −80 (negativos antes que el cierre) → −180 ·
    # A close +150 → −30 · B close +90 → +60. Secuencial daría −100.
    assert f == {D1.toordinal(): pytest.approx(-180.0)}


# ---------------------------------------------------------------------------
# V2 delta de cierre en la banda negativa de tolerancia
# ---------------------------------------------------------------------------

def test_v2_delta_cierre_en_banda_negativa():
    b = direct_block([(0, 0, ts(D1, "09:00"), ts(D1, "10:00"), -10.005)])
    # adverse 10,00 y pnl −10,005 → delta de cierre −0,005 (< 0): no rompe y el
    # mínimo es el del cierre (−10,005), no el del estado vivo (−10,00).
    f = _windows_day_float(b, np.ones(1), {0: np.array([10.0])})
    assert f == {D1.toordinal(): pytest.approx(-10.005)}
    bda = windows_bda(b, {0: np.ones(1)}, {0: np.array([10.0])})
    assert bda.rel_low[0] == pytest.approx(-10.005)
    assert bda.rel_closed[0] == pytest.approx(-10.005)
    assert bda.rel_low[0] <= bda.rel_closed[0] + 1e-12


# ---------------------------------------------------------------------------
# Oráculo independiente por ESTADOS (property test con empates y cruces)
# ---------------------------------------------------------------------------

def _oracle(b, lots, adv_by_sym):
    """min por sesión de Σ_j L_j(t) − baseline, muestreando los candidatos
    (inicio de sesión y cada instante de evento) con la regla inclusiva
    'vivo si open <= t <= close'. Formulación distinta de la de eventos."""
    k = b.n_trades
    w = np.array([-adv_by_sym[int(b.sym_id[j])][int(b.trade_local_id[j])] * lots[j]
                  for j in range(k)])
    pnl = b.pnl_per_lot * lots
    ot = [pd.Timestamp(x) for x in b.open_t]
    ct = [pd.Timestamp(x) for x in b.close_t]
    # NOTA: asume pnl - w >= 0 en todo trade (el generador aleatorio lo
    # garantiza). El caso de la banda negativa de tolerancia (V2) lo cubre
    # test_v2_delta_cierre_en_banda_negativa.
    sessions = set()
    for j in range(k):
        sessions |= set(range(int(b.open_ord[j]), int(b.close_ord[j]) + 1))
    out = {}
    for s in sessions:
        base = sum(pnl[j] for j in range(k) if b.close_ord[j] < s)
        start = pd.Timestamp.fromordinal(s) - pd.Timedelta(1, "ns")
        cands = [start]
        cands += [ot[j] for j in range(k) if b.open_ord[j] == s]
        cands += [ct[j] for j in range(k) if b.close_ord[j] == s]
        best = np.inf
        for t in cands:
            tot = 0.0
            for j in range(k):
                if t < ot[j]:
                    continue
                tot += w[j] if t <= ct[j] else pnl[j]
            best = min(best, tot - base)
        out[s] = best
    return out


@pytest.mark.parametrize("seed", range(40))
def test_windows_vs_oraculo_por_estados_aleatorio(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 9))
    base = pd.Timestamp("2026-03-02 06:00")
    trades, adv = [], {0: [], 1: []}
    cnt = {0: 0, 1: 0}
    lots = []
    for _ in range(k):
        sym = int(rng.integers(0, 2))
        # rejilla de 30 min → muchos empates; duraciones hasta ~2,5 días.
        o = base + pd.Timedelta(minutes=30 * int(rng.integers(0, 6 * 48)))
        c = o + pd.Timedelta(minutes=30 * int(rng.integers(1, 120)))
        n = float(rng.integers(1, 6))
        a = float(rng.integers(0, 300))          # adverso por lote (USD)
        ppl = float(rng.integers(-int(a), 300)) if a > 0 else float(rng.integers(0, 300))
        trades.append((sym, cnt[sym], o, c, ppl))
        adv[sym].append(a)
        cnt[sym] += 1
        lots.append(n)
    b = direct_block(trades)
    adv = {s: np.array(v) for s, v in adv.items() if v}
    lots = np.array(lots)
    got = _windows_day_float(b, lots, adv)
    exp = _oracle(b, lots, adv)
    # El oráculo muestrea también sesiones intermedias (arrastre): mismas claves.
    assert set(got) == set(exp)
    for s in exp:
        assert got[s] == pytest.approx(exp[s], abs=1e-9)


# ---------------------------------------------------------------------------
# B4 walker e2e (FTMO Futures Growth 50K: eod_trailing, 2.000 USD, equity)
# ---------------------------------------------------------------------------
# Config verificada: amount 2.000, lock_at_balance 50.000, breach_on equity, target
# 6% = 3.000, consistencia 40% (best day ≤ 40% del total).
#
# Bloque a mano (pata 'main' + cover D1 08:00 → D2 17:00 de PnL 0):
#   D1: X abre 10:00 / cierra 15:00, pnl +500, MAE 100.
#   D2: Y abre 10:00 / cierra 15:00, pnl +300, MAE 3.100  (+ Z solo en el gemelo).
# Números a mano (S = 50.000 al inicio del bloque):
#   D1: T_1 = 50.000 − 2.000 = 48.000. Eventos: X open −100, X close +600 → min
#       −100 ⇒ rel_low = −100 ⇒ equity 49.900 > 48.000 (vivo). Cierre 50.500 ⇒
#       HWM 50.500 ⇒ T_2 = min(50.500 − 2.000, 50.000) = 48.500.
#   D2 (i): Y open −3.100 · Y close +3.400 ⇒ min −3.100. rel_before = +500 ⇒
#       rel_low = −2.600 ⇒ equity 47.400 ≤ 48.500 ⇒ FAIL en D2 (days = 2,0) aunque
#       el balance EOD de D2 (50.000 + 500 + 300 = 50.800) esté > T_2 (V9).
#   D2 (ii) gemelo: Z abre 09:00 / cierra 09:30, pnl +1.200, MAE 50:
#       Z open −50 · Z close +1.250 (→ +1.200) · Y open −3.100 (→ −1.900) · Y close
#       +3.400 (→ +1.500) ⇒ min −1.900 ⇒ rel_low = 500 − 1.900 = −1.400 ⇒ equity
#       48.600 > 48.500 ⇒ NO fail. La SUMA de MAE (−50 − 3.100 = −3.150) habría
#       dado rel_low = −2.650 ⇒ equity 47.350 ≤ 48.500 ⇒ fail (falso breach).
#       Cierre D2: 52.000 (total +2.000 < 3.000) ⇒ 'neither'.
#
# Cobertura complementaria ya existente del walker (no se duplica aquí):
#   - lock_at_balance: tests/test_risk_propfirm_futures_eval.py::test_golden_lock_oficial,
#     ::test_walker_progresion_umbral_y_lock, ::test_umbral_cap_en_lock.
#   - días cualificados (funded): tests/test_risk_propfirm_futures_funded.py::
#     test_golden_qualifying_3_de_4, ::test_reset_de_qual_days_y_ciclo_tras_pago.

ACC = 50_000.0


def _phase_50k():
    cfg = load_propfirm_config()["ftmo_futures_growth_50k"]
    ph = cfg.phases[0]
    assert ph.drawdown.type == "eod_trailing"
    assert ph.drawdown.amount_usd == 2000.0 and ph.drawdown.breach_on == "equity"
    assert ph.drawdown.lock_at_balance == 50_000.0
    assert cfg.account_size == ACC
    return ph


def _b4_bda(y_mae: float, with_z: bool):
    main_rows = [(ts(D1, "10:00"), ts(D1, "15:00"), 1, 500.0, 100.0),   # X
                 (ts(D2, "10:00"), ts(D2, "15:00"), 1, 300.0, y_mae)]  # Y
    if with_z:
        main_rows.append((ts(D2, "09:00"), ts(D2, "09:30"), 1, 1200.0, 50.0))  # Z
    cover = table([(ts(D1, "08:00"), ts(D2, "17:00"), 1, 0.0, 0.0)])
    blocks, lots, pv, adv = build(
        {"main": table(main_rows), "cover": cover},
        {"main": [1] * len(main_rows), "cover": [1]})
    assert len(blocks) == 1
    bda = block_day_arrays_portfolio(blocks[0], lots, pv, 0, float_mode="windows",
                                     adverse_by_sym=adv)
    return blocks[0], bda


def _walk(b, bda):
    return evaluate_phase_path_eod_trail(
        np.array([0]), [bda], np.array([b.duration_days]), None,
        account_size=ACC, phase=_phase_50k())


def test_b4_i_breach_intradia_con_cierre_recuperado_fail_en_la_sesion():
    b, bda = _b4_bda(3100.0, with_z=False)
    assert list(bda.day_ord) == [D1.toordinal(), D2.toordinal()]
    np.testing.assert_allclose(bda.rel_before, [0.0, 500.0])
    np.testing.assert_allclose(bda.rel_closed, [500.0, 800.0])
    np.testing.assert_allclose(bda.rel_low, [-100.0, -2600.0])
    # Cierre de D2 (50.800) por encima de T_2 = 48.500, pero equity 47.400 ≤ T_2.
    assert ACC + bda.rel_closed[1] == pytest.approx(50_800.0)
    assert ACC + bda.rel_closed[1] > 48_500.0
    assert ACC + bda.rel_low[1] == pytest.approx(47_400.0)
    assert ACC + bda.rel_low[1] <= 48_500.0
    r = _walk(b, bda)
    assert r.outcome == "fail"
    assert r.days == 2.0                       # exactamente la sesión D2
    assert r.terminal == pytest.approx(50_800.0)


def test_b4_ii_ganadora_previa_misma_sesion_evita_el_fail():
    b, bda = _b4_bda(3100.0, with_z=True)
    np.testing.assert_allclose(bda.rel_before, [0.0, 500.0])
    np.testing.assert_allclose(bda.rel_low, [-100.0, 500.0 - 1900.0])
    np.testing.assert_allclose(bda.rel_closed, [500.0, 2000.0])
    assert ACC + bda.rel_low[1] == pytest.approx(48_600.0)
    assert ACC + bda.rel_low[1] > 48_500.0
    r = _walk(b, bda)
    assert r.outcome == "neither"
    assert r.terminal == pytest.approx(52_000.0)
    # La suma de MAE (−50 − 3.100 = −3.150) en la misma sesión habría dado fail.
    sum_mae = BlockDayArrays(
        day_ord=bda.day_ord, rel_before=bda.rel_before, rel_closed=bda.rel_closed,
        rel_float=bda.rel_float, opens_day=bda.opens_day, total_pnl=bda.total_pnl,
        rel_low=np.array([-100.0, 500.0 - 3150.0]))
    assert ACC + sum_mae.rel_low[1] == pytest.approx(47_350.0)
    assert ACC + sum_mae.rel_low[1] <= 48_500.0
    assert _walk(b, sum_mae).outcome == "fail"


@pytest.mark.parametrize("y_mae,esperado", [(2000.0, "fail"), (1999.99, "neither")])
def test_b4_borde_inclusivo_equity_igual_a_T(y_mae, esperado):
    # rel_low D2 = 500 − y_mae ⇒ equity = 50.500 − y_mae; T_2 = 48.500 (≤ inclusivo).
    b, bda = _b4_bda(y_mae, with_z=False)
    assert _walk(b, bda).outcome == esperado


# ---------------------------------------------------------------------------
# B5 sizing Opción A
# ---------------------------------------------------------------------------

def test_b5_fixed_risk_micros_golden():
    risk = [54.0, 617.0, 5.5]
    lots, diag = fixed_risk_micros(risk, [0.01], 50_000.0, cap=50, comm_per_micro=1.12)
    assert lots.shape == (1, 3)
    # X = 500: floor(500/54) = 9; 617 > X → clamp a 1; floor(500/5.5) = 90 → cap 50.
    np.testing.assert_array_equal(lots[0], [9.0, 1.0, 50.0])
    assert lots.dtype == float
    d = diag[0]
    assert d["X"] == pytest.approx(500.0)
    assert d["frac_clamp_up"] == pytest.approx(1 / 3)
    assert d["frac_capped"] == pytest.approx(1 / 3)
    # n·risk/X = [0,972 ; 1,234 ; 0,55] → p95 (interp. lineal) = 0,972 + 0,9·0,262.
    assert d["p95_risk_eff_over_X"] == pytest.approx(0.972 + 0.9 * (1.234 - 0.972))
    # n·(risk+1,12)/X = [0,99216 ; 1,23624 ; 0,662].
    assert d["p95_total_adverse_over_X"] == pytest.approx(
        0.99216 + 0.9 * (1.23624 - 0.99216))


def test_b5_diag_por_f_y_sin_cap_ni_comision():
    risk = [54.0, 617.0, 5.5]
    lots, diag = fixed_risk_micros(risk, [0.005, 0.01], 50_000.0)
    assert len(diag) == 2 and lots.shape == (2, 3)
    # f = 0,5 %: X = 250 → [4, 1, 45] (sin cap).
    np.testing.assert_array_equal(lots[0], [4.0, 1.0, 45.0])
    # f = 1 % sin cap: [9, 1, 90].
    np.testing.assert_array_equal(lots[1], [9.0, 1.0, 90.0])
    assert diag[0]["X"] == pytest.approx(250.0)
    assert diag[0]["frac_clamp_up"] == pytest.approx(1 / 3)
    assert diag[0]["frac_capped"] == 0.0 and diag[1]["frac_capped"] == 0.0
    assert "p95_total_adverse_over_X" not in diag[0]
    # n·risk/X (f=0,5 %) = [216/250 ; 617/250 ; 247,5/250] = [0,864 ; 2,468 ; 0,99].
    assert diag[0]["p95_risk_eff_over_X"] == pytest.approx(0.99 + 0.9 * (2.468 - 0.99))


def test_b5_cap_none_sin_tope_y_cap_no_recorta_si_no_excede():
    lots, diag = fixed_risk_micros([5.5], [0.01], 50_000.0, cap=None)
    assert lots[0, 0] == 90.0 and diag[0]["frac_capped"] == 0.0
    lots, diag = fixed_risk_micros([5.5], [0.01], 50_000.0, cap=90)
    assert lots[0, 0] == 90.0 and diag[0]["frac_capped"] == 0.0   # no excede
    lots, diag = fixed_risk_micros([5.5], [0.01], 50_000.0, cap=89)
    assert lots[0, 0] == 89.0 and diag[0]["frac_capped"] == 1.0


def test_b5_fixed_risk_micros_entradas_invalidas():
    with pytest.raises(ValueError):
        fixed_risk_micros([54.0, 0.0], [0.01], 50_000.0)
    with pytest.raises(ValueError):
        fixed_risk_micros([54.0], [0.0], 50_000.0)
    with pytest.raises(ValueError):
        fixed_risk_micros([54.0], [0.01], 0.0)


def test_b5_comm_array_por_trade():
    _, diag = fixed_risk_micros([54.0, 54.0], [0.01], 50_000.0,
                                comm_per_micro=np.array([0.0, 2.0]))
    # n = 9: [9·54/500 ; 9·56/500] = [0,972 ; 1,008] → p95 = 0,972 + 0,95·0,036.
    assert diag[0]["p95_total_adverse_over_X"] == pytest.approx(0.972 + 0.95 * 0.036)


# ---------------------------------------------------------------------------
# B6 fail-loud
# ---------------------------------------------------------------------------

def _simple_block():
    return direct_block([(0, 0, ts(D1, "09:00"), ts(D1, "10:00"), 10.0),
                         (1, 0, ts(D1, "09:30"), ts(D1, "10:30"), 5.0)])


def test_b6_windows_sin_adverse_by_sym():
    b = _simple_block()
    lots = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    pv = {0: 2.0, 1: 2.0}
    with pytest.raises(ValueError, match="adverse_by_sym"):
        block_day_arrays_portfolio(b, lots, pv, 0, float_mode="windows")
    with pytest.raises(ValueError, match="adverse_by_sym"):
        precompute_day_arrays_portfolio([b], lots, pv, 1, float_mode="windows")
    with pytest.raises(ValueError, match="adverse_by_sym"):
        precompute_day_arrays_portfolio([], lots, pv, 1, float_mode="windows")


def test_b6_windows_pata_o_id_local_faltante():
    b = _simple_block()
    lots = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    pv = {0: 2.0, 1: 2.0}
    with pytest.raises(ValueError, match="pata 1"):
        block_day_arrays_portfolio(b, lots, pv, 0, float_mode="windows",
                                   adverse_by_sym={0: np.array([1.0])})
    with pytest.raises(ValueError, match="id local"):
        block_day_arrays_portfolio(
            b, lots, pv, 0, float_mode="windows",
            adverse_by_sym={0: np.array([1.0]), 1: np.array([])})
    with pytest.raises(ValueError, match="pata 1"):
        precompute_day_arrays_portfolio([b], lots, pv, 1, float_mode="windows",
                                        adverse_by_sym={0: np.array([1.0])})


def test_b6_windows_con_trailing_es_error():
    b = _simple_block()
    lots = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    pv = {0: 2.0, 1: 2.0}
    adv = {0: np.array([1.0]), 1: np.array([1.0])}
    with pytest.raises(ValueError, match="trailing"):
        block_day_arrays_portfolio(b, lots, pv, 0, float_mode="windows",
                                   adverse_by_sym=adv, trailing=True)
    with pytest.raises(ValueError, match="trailing"):
        precompute_day_arrays_portfolio([b], lots, pv, 1, float_mode="windows",
                                        adverse_by_sym=adv, trailing=True)


def test_b6_float_mode_desconocido_sigue_siendo_error():
    b = _simple_block()
    lots = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    with pytest.raises(ValueError, match="desconocido"):
        block_day_arrays_portfolio(b, lots, {0: 2.0, 1: 2.0}, 0, float_mode="nope")


# ---------------------------------------------------------------------------
# B8 regresión: los demás modos no cambian
# ---------------------------------------------------------------------------

def test_b8_defaults_siguen_siendo_bound_y_sin_rel_low():
    for fn in (block_day_arrays_portfolio, precompute_day_arrays_portfolio):
        assert inspect.signature(fn).parameters["float_mode"].default == "bound"
        assert inspect.signature(fn).parameters["adverse_by_sym"].default is None
    b = direct_block([(0, 0, ts(D1, "09:00"), ts(D1, "10:00"), 10.0)])
    # _day_float_bound lee claves de fecha (pd.Timestamp como en producción).
    b.valle_precio = [{D1: -5.0}]
    bda = block_day_arrays_portfolio(b, {0: np.array([[1.0]])}, {0: 2.0}, 0)
    assert bda.rel_low is None
    assert bda.rel_float[0] == pytest.approx(-10.0)      # valle·lotes·pv
    out = precompute_day_arrays_portfolio([b], {0: np.array([[1.0]])}, {0: 2.0}, 1)
    assert out[0][0].rel_low is None
