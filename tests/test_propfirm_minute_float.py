"""Fase 1 del plan cfd-sqx-minuto-par: ``float_mode="minute"`` del motor de prop firms.

Plan cfd-sqx-minuto-par (§2.1 y §4 "Fase 1",
A1-A9) y su revisión dual. Los goldens están escritos a mano en cada test.

Cobertura de aceptación:
  A1  test_a1_*   golden a mano (realizado + flotante el mismo día), minute vs joint
  A2  test_a2_*   trade que cruza medianoche (día 2 presente, realizado en ct)
  A3  test_a3_*   linealidad exacta al duplicar lotes
  A4  test_a4_*   propiedad minute >= windows (200 bloques aleatorios)
  A5  test_a5_*   series a 0: mínimo del realizado acumulado del día
  A6  test_a6_*   fail-loud (3 casos)
  A7  test_a7_*   precompute == block_day_arrays por f y bloque
  A8  regresión   la cubre el resto de la suite (bound/joint/windows intactos)
  A9  test_a9_*   fin de semana con evaluate_phase_path
Extra: etiquetador de sesión (day_labeler) y bound/joint sin cambios de firma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import StrategyMC
from src.risk.propfirm import (
    BlockDayArrays, PortfolioDayBlock, PropFirmPhase,
    block_day_arrays_portfolio, build_day_blocks_portfolio,
    evaluate_phase_path, precompute_day_arrays_portfolio,
)

PV = 100.0
D0 = pd.Timestamp("2024-03-04")        # lunes


def at(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    h, m = hhmm.split(":")
    return day + pd.Timedelta(hours=int(h), minutes=int(m))


def serie(ot, ct, pts):
    """Entrada de ``minute_valle``: (ts, vals, ot, ct). ``pts`` = [(ts, valor_precio)]."""
    ts = np.array([pd.Timestamp(p[0]) for p in pts], dtype="datetime64[ns]")
    vals = np.array([p[1] for p in pts], dtype=float)
    return ts, vals, pd.Timestamp(ot), pd.Timestamp(ct)


def block(trades) -> PortfolioDayBlock:
    """Bloque a mano. trades: (sym, local_id, open_ts, close_ts, pnl_per_lot)."""
    k = len(trades)
    ot = np.array([t[2] for t in trades], dtype="datetime64[ns]")
    ct = np.array([t[3] for t in trades], dtype="datetime64[ns]")
    return PortfolioDayBlock(
        idx=0, start=pd.Timestamp(ot.min()), end=pd.Timestamp(ct.max()),
        duration_days=(pd.Timestamp(ct.max()) - pd.Timestamp(ot.min())).total_seconds() / 86400.0,
        sym_id=np.array([t[0] for t in trades]),
        trade_local_id=np.array([t[1] for t in trades]),
        open_ord=np.array([pd.Timestamp(t[2]).toordinal() for t in trades]),
        close_ord=np.array([pd.Timestamp(t[3]).toordinal() for t in trades]),
        valle_precio=[{} for _ in range(k)],
        pnl_per_lot=np.array([t[4] for t in trades], dtype=float),
        n_trades=k, n_no_valley=k, open_t=ot, close_t=ct,
    )


def mbs(*entries):
    """minute_by_sym desde (sym, local, serie)."""
    out: dict = {}
    for s, i, ser in entries:
        out.setdefault(s, {})[i] = ser
    return out


def run(b, lots, minute_by_sym, mode="minute", pv=PV, **kw):
    """Un f. ``lots``: {sym: array por trade local} (una fila)."""
    return block_day_arrays_portfolio(
        b, {s: np.array([v], dtype=float) for s, v in lots.items()},
        {s: pv for s in lots}, 0, float_mode=mode,
        minute_by_sym=minute_by_sym, **kw)


def idx(bda, day: pd.Timestamp) -> int:
    return int(np.flatnonzero(bda.day_ord == day.toordinal())[0])


# ---------------------------------------------------------------------------
# Escenario A1 (reutilizado)
# ---------------------------------------------------------------------------

def _a1_block_y_series():
    d = D0
    b = block([
        (0, 0, at(d, "09:00"), at(d, "10:00"), -100.0),
        (1, 0, at(d, "09:30"), at(d, "15:00"), +50.0),
    ])
    m = mbs(
        (0, 0, serie(at(d, "09:00"), at(d, "10:00"),
                     [(at(d, "09:00"), -0.3), (at(d, "09:30"), -0.8)])),
        (1, 0, serie(at(d, "09:30"), at(d, "15:00"),
                     [(at(d, "09:30"), -0.5), (at(d, "12:00"), -1.5),
                      (at(d, "14:00"), -0.2)])),
    )
    return b, m


def test_a1_golden_realizado_mas_flotante_el_mismo_dia():
    b, m = _a1_block_y_series()
    bda = run(b, {0: [1.0], 1: [1.0]}, m)
    # Niveles por minuto (cal = 09:00, 09:30, 10:00, 12:00, 14:00, 15:00):
    #   09:00 -30 ; 09:30 -80-50 = -130 ; 10:00 realizado A -100 + B(-0,5) -50 = -150 ;
    #   12:00 -100 + (-150) = -250 ; 14:00 -100-20 = -120 ; 15:00 -100+50 = -50.
    assert list(bda.day_ord) == [D0.toordinal()]
    assert bda.rel_float[0] == pytest.approx(-250.0)
    assert bda.rel_closed[0] == pytest.approx(-50.0)
    assert bda.rel_before[0] == 0.0
    assert bda.total_pnl == pytest.approx(-50.0)
    # Solo perfiles estáticos: sin campos trailing.
    assert bda.rel_low is None and bda.rel_high is None and bda.rel_exc is None
    assert bda.block_low is None and bda.block_high is None and bda.block_exc is None


def test_a1_joint_del_mismo_bloque_da_menos_pesimista():
    """Documenta la diferencia: joint NO incluye el realizado intradía (-100 de A)
    en el mínimo, solo la suma de flotantes vivos (A -80 + B -50 a las 09:30/10:00;
    B -150 a las 12:00) -> -150; minute da -250."""
    b, m = _a1_block_y_series()
    joint = run(b, {0: [1.0], 1: [1.0]}, m, mode="joint")
    minute = run(b, {0: [1.0], 1: [1.0]}, m, mode="minute")
    assert joint.rel_float[0] == pytest.approx(-150.0)
    assert minute.rel_float[0] == pytest.approx(-250.0)
    assert minute.rel_float[0] < joint.rel_float[0]


def test_a1_via_build_day_blocks_portfolio_con_strategymc():
    """Mismo golden pasando por ``StrategyMC`` + ``build_day_blocks_portfolio``."""
    d = D0

    def strat(key, ot, ct, pnl):
        return StrategyMC(
            key=key,
            open_t=np.array([ot], dtype="datetime64[ns]"),
            close_t=np.array([ct], dtype="datetime64[ns]"),
            close_d=pd.to_datetime([ct]).normalize().to_numpy(),
            valle={}, pnl_per_lot=np.array([pnl]), is_buy=np.array([True]),
            pv=PV, lots=np.ones((1, 1)), n_trades=1)

    sa = strat("A", at(d, "09:00"), at(d, "10:00"), -100.0)
    sb = strat("B", at(d, "09:30"), at(d, "15:00"), +50.0)
    blocks = build_day_blocks_portfolio({"A": sa, "B": sb})
    assert len(blocks) == 1
    _, m = _a1_block_y_series()
    out = precompute_day_arrays_portfolio(
        blocks, {0: sa.lots, 1: sb.lots}, {0: PV, 1: PV}, 1,
        float_mode="minute", minute_by_sym=m)
    assert out[0][0].rel_float[0] == pytest.approx(-250.0)
    assert out[0][0].rel_closed[0] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# A2: cruce de medianoche
# ---------------------------------------------------------------------------

def test_a2_cruce_de_medianoche_realizado_en_ct_en_el_dia_2():
    d1, d2 = D0, D0 + pd.Timedelta(days=1)
    b = block([(0, 0, at(d1, "22:00"), at(d2, "10:00"), -30.0)])
    m = mbs((0, 0, serie(at(d1, "22:00"), at(d2, "10:00"),
                         [(at(d1, "22:00"), -0.2), (at(d1, "23:00"), -0.6),
                          (at(d2, "09:00"), -0.1)])))
    bda = run(b, {0: [1.0]}, m)
    assert list(bda.day_ord) == [d1.toordinal(), d2.toordinal()]
    # Día 1: mínimo del flotante del día 1 (-60 a las 23:00); nada cerrado todavía.
    assert bda.rel_float[0] == pytest.approx(-60.0)
    assert bda.rel_before[0] == 0.0 and bda.rel_closed[0] == 0.0
    # Día 2: 09:00 -> -10 ; en ct (10:00) el realizado -30 toma el relevo (la
    # envolvente muere EN ct): el mínimo del día 2 incluye el realizado.
    assert bda.rel_float[1] == pytest.approx(-30.0)
    assert bda.rel_before[1] == 0.0
    assert bda.rel_closed[1] == pytest.approx(-30.0)


def test_a2_dia_intermedio_sin_barras_ni_eventos_esta_presente_y_arrastra_estado():
    d1 = D0
    d2 = D0 + pd.Timedelta(days=1)
    d3 = D0 + pd.Timedelta(days=2)
    b = block([(0, 0, at(d1, "22:00"), at(d3, "10:00"), +20.0)])
    m = mbs((0, 0, serie(at(d1, "22:00"), at(d3, "10:00"),
                         [(at(d1, "22:00"), -0.2), (at(d1, "23:00"), -0.6),
                          (at(d3, "08:00"), -0.4)])))
    bda = run(b, {0: [1.0]}, m)
    assert list(bda.day_ord) == [d1.toordinal(), d2.toordinal(), d3.toordinal()]
    # El día 2 no abre ni cierra nada ni tiene barras: hereda el estado del
    # último minuto del día 1 (-60), no 0.
    assert bda.rel_float[1] == pytest.approx(-60.0)
    assert bda.rel_before[1] == 0.0 and bda.rel_closed[1] == 0.0
    assert bda.rel_float[0] == pytest.approx(-60.0)
    # Día 3: min(08:00 -> -40, ct -> +20) = -40 ; cierra +20.
    assert bda.rel_float[2] == pytest.approx(-40.0)
    assert bda.rel_closed[2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# A3: linealidad
# ---------------------------------------------------------------------------

def test_a3_linealidad_exacta_al_duplicar_lotes():
    b, m = _a1_block_y_series()
    # Segundo trade por pata para que haya días y estructura.
    d2 = D0 + pd.Timedelta(days=1)
    b = block([
        (0, 0, at(D0, "09:00"), at(D0, "10:00"), -100.0),
        (1, 0, at(D0, "09:30"), at(d2, "15:00"), +50.0),
        (0, 1, at(D0, "11:00"), at(D0, "12:30"), +80.0),
    ])
    m = mbs(
        (0, 0, serie(at(D0, "09:00"), at(D0, "10:00"),
                     [(at(D0, "09:00"), -0.3), (at(D0, "09:30"), -0.8)])),
        (1, 0, serie(at(D0, "09:30"), at(d2, "15:00"),
                     [(at(D0, "09:30"), -0.5), (at(D0, "20:00"), -1.5),
                      (at(d2, "14:00"), -0.2)])),
        (0, 1, serie(at(D0, "11:00"), at(D0, "12:30"),
                     [(at(D0, "11:00"), -0.1), (at(D0, "12:00"), -0.9)])),
    )
    x1 = run(b, {0: [1.0, 1.0], 1: [1.0]}, m)
    x2 = run(b, {0: [2.0, 2.0], 1: [2.0]}, m)
    np.testing.assert_array_equal(x1.day_ord, x2.day_ord)
    np.testing.assert_array_equal(2.0 * x1.rel_float, x2.rel_float)
    np.testing.assert_array_equal(2.0 * x1.rel_before, x2.rel_before)
    np.testing.assert_array_equal(2.0 * x1.rel_closed, x2.rel_closed)
    assert 2.0 * x1.total_pnl == x2.total_pnl
    # Y con un factor no potencia de 2 (tolerancia de redondeo).
    x3 = run(b, {0: [3.0, 3.0], 1: [3.0]}, m)
    np.testing.assert_allclose(3.0 * x1.rel_float, x3.rel_float, rtol=1e-12, atol=1e-9)


# ---------------------------------------------------------------------------
# A4: propiedad minute >= windows con series coherentes
# ---------------------------------------------------------------------------

def _bloque_aleatorio(rng):
    """Bloque sintético + series YA ajustadas + adverse_by_sym coherente."""
    k = int(rng.integers(1, 6))
    base = D0 + pd.Timedelta(minutes=int(rng.integers(0, 3 * 24 * 60)))
    trades, ent, adv = [], [], {0: [], 1: []}
    n_loc = {0: 0, 1: 0}
    pnl_l = {0: [], 1: []}
    for j in range(k):
        s = int(rng.integers(0, 2))
        li = n_loc[s]
        n_loc[s] += 1
        ot = (base + pd.Timedelta(minutes=int(rng.integers(0, 2 * 24 * 60)))).floor("min")
        ct = ot + pd.Timedelta(minutes=int(rng.integers(1, 4 * 24 * 60)))
        n_bars = int(rng.integers(1, 8))
        span = int((ct - ot).total_seconds() // 60)
        offs = np.unique(np.r_[0, rng.integers(0, span, size=n_bars)])
        pts = [(ot + pd.Timedelta(minutes=int(o)), float(rng.normal(-0.4, 0.8)))
               for o in offs]
        vals = np.array([p[1] for p in pts])
        # Coherencia por construcción (A4): adverse = max(0, -min(serie)·pv);
        # pnl = último valor de la serie · pv (>= -adverse).
        adv[s].append(max(0.0, -float(vals.min()) * PV))
        pnl = float(vals[-1]) * PV
        pnl_l[s].append(pnl)
        trades.append((s, li, ot, ct, pnl))
        ent.append((s, li, serie(ot, ct, pts)))
    return block(trades), mbs(*ent), {s: np.array(v) for s, v in adv.items() if v}


def test_a4_minute_cota_a_ventanas_en_200_bloques_aleatorios():
    rng = np.random.default_rng(20261014)
    n_dias_comunes = 0
    for _ in range(200):
        b, m, adv = _bloque_aleatorio(rng)
        lots = {s: rng.integers(1, 4, size=len(adv[s])).astype(float) for s in adv}
        mi = run(b, lots, m, mode="minute")
        wi = run(b, lots, m, mode="windows", adverse_by_sym=adv)
        sm, sw = set(mi.day_ord.tolist()), set(wi.day_ord.tolist())
        assert sw <= sm, "el conjunto de días de minute debe contener el de windows"
        pm = dict(zip(mi.day_ord.tolist(), mi.rel_float))
        pw = dict(zip(wi.day_ord.tolist(), wi.rel_float))
        for d in sw:
            n_dias_comunes += 1
            assert pm[d] >= pw[d] - 1e-9, (d, pm[d], pw[d])
        # Mismas columnas de cerrado/antes en días comunes.
        cm = dict(zip(mi.day_ord.tolist(), mi.rel_closed))
        cw = dict(zip(wi.day_ord.tolist(), wi.rel_closed))
        for d in sw:
            assert cm[d] == pytest.approx(cw[d])
    assert n_dias_comunes > 200


# ---------------------------------------------------------------------------
# A5: sin flotante
# ---------------------------------------------------------------------------

def test_a5_series_a_cero_minimo_del_realizado_acumulado_del_dia():
    d = D0
    b = block([
        (0, 0, at(d, "09:00"), at(d, "10:00"), -100.0),
        (1, 0, at(d, "09:30"), at(d, "11:00"), +300.0),
    ])
    m = mbs(
        (0, 0, serie(at(d, "09:00"), at(d, "10:00"), [(at(d, "09:00"), 0.0)])),
        (1, 0, serie(at(d, "09:30"), at(d, "11:00"), [(at(d, "09:30"), 0.0)])),
    )
    bda = run(b, {0: [1.0], 1: [1.0]}, m)
    # Realizado acumulado por minuto (09:00, 09:30, 10:00, 11:00): 0, 0, -100, +200.
    assert bda.rel_float[0] == pytest.approx(-100.0)
    assert bda.rel_closed[0] == pytest.approx(200.0)


def test_a5_un_unico_trade_perdedor_rel_float_igual_rel_closed():
    d = D0
    b = block([(0, 0, at(d, "09:00"), at(d, "10:00"), -40.0)])
    m = mbs((0, 0, serie(at(d, "09:00"), at(d, "10:00"), [(at(d, "09:00"), 0.0)])))
    bda = run(b, {0: [1.0]}, m)
    assert bda.rel_float[0] == pytest.approx(-40.0)
    assert bda.rel_float[0] == pytest.approx(bda.rel_closed[0])


def test_a5_un_unico_trade_ganador_el_minimo_es_el_estado_previo_al_cierre():
    """Matiz de A5: con un trade ganador el mínimo del día es 0 (minuto de
    apertura, antes del cierre), no rel_closed; la igualdad rel_float == rel_closed
    vale para el trade perdedor (el mínimo cae en ct)."""
    d = D0
    b = block([(0, 0, at(d, "09:00"), at(d, "10:00"), +40.0)])
    m = mbs((0, 0, serie(at(d, "09:00"), at(d, "10:00"), [(at(d, "09:00"), 0.0)])))
    bda = run(b, {0: [1.0]}, m)
    assert bda.rel_float[0] == pytest.approx(0.0)
    assert bda.rel_closed[0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# A6: fail-loud
# ---------------------------------------------------------------------------

def test_a6_minute_sin_minute_by_sym_es_valueerror():
    b, m = _a1_block_y_series()
    with pytest.raises(ValueError, match="minute_by_sym"):
        run(b, {0: [1.0], 1: [1.0]}, None)
    with pytest.raises(ValueError, match="minute_by_sym"):
        precompute_day_arrays_portfolio(
            [b], {0: np.ones((1, 1)), 1: np.ones((1, 1))}, {0: PV, 1: PV}, 1,
            float_mode="minute")
    # Sin bloques tampoco se degrada en silencio.
    with pytest.raises(ValueError, match="minute_by_sym"):
        precompute_day_arrays_portfolio([], {}, {}, 1, float_mode="minute")


def test_a6_minute_con_trailing_es_valueerror():
    b, m = _a1_block_y_series()
    with pytest.raises(ValueError, match="trailing"):
        run(b, {0: [1.0], 1: [1.0]}, m, trailing=True, minute_fav_by_sym=m)
    with pytest.raises(ValueError, match="trailing"):
        precompute_day_arrays_portfolio(
            [b], {0: np.ones((1, 1)), 1: np.ones((1, 1))}, {0: PV, 1: PV}, 1,
            float_mode="minute", minute_by_sym=m, trailing=True,
            minute_fav_by_sym=m)


def test_a6_trade_del_bloque_sin_serie_es_valueerror():
    b, m = _a1_block_y_series()
    sin_pata_1 = {0: m[0]}                       # falta la pata entera
    with pytest.raises(ValueError, match="serie"):
        run(b, {0: [1.0], 1: [1.0]}, sin_pata_1)
    sin_trade = {0: m[0], 1: {}}                 # pata presente, trade ausente
    with pytest.raises(ValueError, match="serie"):
        run(b, {0: [1.0], 1: [1.0]}, sin_trade)
    with pytest.raises(ValueError, match="serie"):
        precompute_day_arrays_portfolio(
            [b], {0: np.ones((1, 1)), 1: np.ones((1, 1))}, {0: PV, 1: PV}, 1,
            float_mode="minute", minute_by_sym=sin_trade)


# ---------------------------------------------------------------------------
# A7: caché por bloque == llamada directa
# ---------------------------------------------------------------------------

def _strategy(key, trades, pv, lots_rows):
    ot = np.array([t[0] for t in trades], dtype="datetime64[ns]")
    ct = np.array([t[1] for t in trades], dtype="datetime64[ns]")
    return StrategyMC(
        key=key, open_t=ot, close_t=ct,
        close_d=pd.to_datetime(ct).normalize().to_numpy(),
        valle={}, pnl_per_lot=np.array([t[2] for t in trades], dtype=float),
        is_buy=np.ones(len(trades), dtype=bool), pv=pv,
        lots=np.array(lots_rows, dtype=float), n_trades=len(trades))


def test_a7_precompute_igual_a_block_day_arrays_por_f_y_bloque():
    rng = np.random.default_rng(7)

    def gen(n, key_off):
        trades, ent = [], {}
        t = D0 + pd.Timedelta(hours=key_off)
        for i in range(n):
            ot = (t + pd.Timedelta(minutes=int(rng.integers(30, 3000)))).floor("min")
            ct = ot + pd.Timedelta(minutes=int(rng.integers(5, 2500)))
            span = int((ct - ot).total_seconds() // 60)
            offs = np.unique(np.r_[0, rng.integers(0, span, size=4)])
            pts = [(ot + pd.Timedelta(minutes=int(o)), float(rng.normal(-0.3, 0.6)))
                   for o in offs]
            pnl = float(rng.normal(0, 150))
            trades.append((ot, ct, pnl))
            ent[i] = serie(ot, ct, pts)
            t = ct + pd.Timedelta(minutes=int(rng.integers(30, 4000)))
        return trades, ent

    t0, e0 = gen(8, 0)
    t1, e1 = gen(8, 3)
    n_f = 3
    # Lotes distintos por f y por trade.
    l0 = rng.integers(1, 6, size=(n_f, 8)) * np.array([[1.0], [2.0], [0.5]])
    l1 = rng.integers(1, 6, size=(n_f, 8)) * np.array([[1.0], [3.0], [0.5]])
    s0, s1 = _strategy("A", t0, 100.0, l0), _strategy("B", t1, 50.0, l1)
    blocks = build_day_blocks_portfolio({"A": s0, "B": s1})
    assert len(blocks) >= 2
    m = {0: e0, 1: e1}
    lots_by_sym = {0: s0.lots, 1: s1.lots}
    pv = {0: 100.0, 1: 50.0}
    out = precompute_day_arrays_portfolio(
        blocks, lots_by_sym, pv, n_f, float_mode="minute", minute_by_sym=m)
    assert len(out) == n_f and all(len(r) == len(blocks) for r in out)
    for fi in range(n_f):
        for bi, b in enumerate(blocks):
            ref = block_day_arrays_portfolio(
                b, lots_by_sym, pv, fi, float_mode="minute", minute_by_sym=m)
            got = out[fi][bi]
            np.testing.assert_array_equal(got.day_ord, ref.day_ord)
            np.testing.assert_array_equal(got.rel_before, ref.rel_before)
            np.testing.assert_array_equal(got.rel_closed, ref.rel_closed)
            np.testing.assert_array_equal(got.rel_float, ref.rel_float)
            np.testing.assert_array_equal(got.opens_day, ref.opens_day)
            assert got.total_pnl == ref.total_pnl
            assert got.rel_low is None and got.block_low is None
    # Los lotes distintos por f cambian realmente el resultado.
    assert not np.array_equal(out[0][0].rel_float, out[1][0].rel_float)


# ---------------------------------------------------------------------------
# A9: fin de semana
# ---------------------------------------------------------------------------

FRI = pd.Timestamp("2024-03-08")
SAT = pd.Timestamp("2024-03-09")
SUN = pd.Timestamp("2024-03-10")
MON = pd.Timestamp("2024-03-11")


def _a9_bda():
    b = block([
        (0, 0, at(FRI, "09:00"), at(FRI, "15:00"), +300.0),    # X
        (1, 0, at(FRI, "10:00"), at(MON, "10:00"), -20.0),     # Y
    ])
    m = mbs(
        (0, 0, serie(at(FRI, "09:00"), at(FRI, "15:00"), [(at(FRI, "09:00"), 0.0)])),
        (1, 0, serie(at(FRI, "10:00"), at(MON, "10:00"),
                     [(at(FRI, "10:00"), -0.5), (at(FRI, "13:00"), -1.0),
                      (at(FRI, "16:00"), -4.0), (at(MON, "09:00"), -0.2)])),
    )
    return run(b, {0: [1.0], 1: [1.0]}, m)


PHASE = PropFirmPhase(name="Challenge", profit_target_pct=1000.0,
                      max_daily_loss_pct=5.0, max_loss_pct=10.0,
                      min_trading_days=1)


def test_a9_sabado_y_domingo_en_day_ord_con_estado_del_ultimo_minuto_del_viernes():
    bda = _a9_bda()
    assert list(bda.day_ord) == [FRI.toordinal(), SAT.toordinal(),
                                 SUN.toordinal(), MON.toordinal()]
    # Viernes: mínimo = -100 (X cerrado a las 15:00 cae del lado de Y: 13:00 -100;
    # 16:00 +300-400 = -100).
    assert bda.rel_float[idx(bda, FRI)] == pytest.approx(-100.0)
    for d in (SAT, SUN):
        i = idx(bda, d)
        assert bda.rel_before[i] == pytest.approx(300.0)
        assert bda.rel_closed[i] == pytest.approx(300.0)
        assert bda.rel_float[i] == pytest.approx(-100.0)
    # Lunes: 09:00 -> 300 - 20 = 280 ; ct realizado 280.
    assert bda.rel_float[idx(bda, MON)] == pytest.approx(280.0)
    assert bda.rel_closed[idx(bda, MON)] == pytest.approx(280.0)


def test_a9_el_path_falla_el_sabado_y_no_falla_sin_esas_filas():
    bda = _a9_bda()
    # Cuenta 7.000: 5% diario = 350, 10% = 700. Suelo del sábado = 7000+300-350 = 6950
    # y la equity arrastrada = 7000-100 = 6900 < 6950 -> fail el sábado (día nº 2).
    res = evaluate_phase_path(np.array([0]), [bda], np.array([4.0]),
                              account_size=7000.0, phase=PHASE)
    assert res.outcome == "fail"
    assert res.days == pytest.approx(2.0)
    # Si el sábado y el domingo no existieran (comportamiento antiguo), no rompe.
    keep = ~np.isin(bda.day_ord, [SAT.toordinal(), SUN.toordinal()])
    recortado = BlockDayArrays(
        day_ord=bda.day_ord[keep], rel_before=bda.rel_before[keep],
        rel_closed=bda.rel_closed[keep], rel_float=bda.rel_float[keep],
        opens_day=bda.opens_day[keep], total_pnl=bda.total_pnl)
    res2 = evaluate_phase_path(np.array([0]), [recortado], np.array([4.0]),
                               account_size=7000.0, phase=PHASE)
    assert res2.outcome != "fail"


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------

def test_extra_day_labeler_sesion_desplaza_el_dia_de_los_minutos():
    """Con un etiquetador de sesión (día = hora + 7 h), los minutos de 17:00-24:00
    pertenecen al día siguiente; open_ord/close_ord del bloque deben venir con el
    mismo etiquetador (así los construye build_day_blocks_portfolio)."""
    def labeler(ix):
        return [(pd.Timestamp(t) + pd.Timedelta(hours=7)).normalize() for t in ix]

    ot, ct = at(D0, "16:00"), at(D0, "20:00")
    b = block([(0, 0, ot, ct, -30.0)])
    b.open_ord = np.array([labeler([ot])[0].toordinal()])
    b.close_ord = np.array([labeler([ct])[0].toordinal()])
    m = mbs((0, 0, serie(ot, ct, [(ot, -0.1), (at(D0, "18:00"), -0.5)])))
    bda = run(b, {0: [1.0]}, m, day_labeler=labeler)
    # 16:00 -> sesión D0 (nivel -10); 18:00 y ct (20:00) -> sesión D0+1.
    assert list(bda.day_ord) == [D0.toordinal(), D0.toordinal() + 1]
    assert bda.rel_float[0] == pytest.approx(-10.0)
    assert bda.rel_float[1] == pytest.approx(-50.0)
    assert bda.rel_closed[1] == pytest.approx(-30.0)


def test_extra_bound_y_joint_no_cambian_con_el_parametro_nuevo():
    """minute_cols es opcional y solo lo consume 'minute'; joint sigue igual."""
    b, m = _a1_block_y_series()
    a = run(b, {0: [1.0], 1: [1.0]}, m, mode="joint")
    c = run(b, {0: [1.0], 1: [1.0]}, m, mode="joint", minute_cols=None)
    np.testing.assert_array_equal(a.rel_float, c.rel_float)
    with pytest.raises(ValueError, match="desconocido"):
        run(b, {0: [1.0], 1: [1.0]}, m, mode="otro")
