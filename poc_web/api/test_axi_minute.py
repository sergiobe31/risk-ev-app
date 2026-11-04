"""Lente AXI con el flotante por MINUTO y el realizado intradía (guardrail ``csv-d1-d2``, fase 2, defecto D2).

Plan csv-d1-d2 (§2.3 y Enmiendas v2 E4, E6, E7, E8, E10, E12). Los goldens están escritos a mano. Todo sintético:
no necesita ``risk_app/inputs/``.

Cobertura de aceptación:
  AC2.2  test_ac2_2_*           golden AXI: ``precompute_minute`` -700 frente a ``precompute_joint`` -500 (E4)
  E6     test_e6_*              días ordenados y sin filas de relleno entre la apertura y el cierre
  E7     test_e7_*              ``strategies`` es una LISTA indexada por ``sym_id``
  E8     test_e8_*              contadores ``n_mae_bounded``/``n_mae_zero`` y su peso en los bloques AXI
  AC2.6  test_ac2_6_*           ``meta.float_convention`` y ``meta.policy_assumption`` hablan del realizado
  Flags  test_flag_*            ``AXI_FLOAT_MODE = "joint"`` reproduce el camino anterior (bit a bit: textos, funciones
                                usadas y cifras fijadas de la corrida de ANTES de la fase 2)
  Extra  walker, formato, linealidad, independencia entre bloques y validaciones; propiedad con bloques aleatorios
         multi-pata y multi-día contra una referencia por fuerza bruta minuto a minuto; equivalencia con el modo
         ``minute`` del motor de prop firm; dos patas de extremo a extremo (orden de patas y realizado en el endpoint)

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_axi_minute.py -q
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poc_web.api import main as main_mod
from poc_web.api.axi_path import (
    evaluate_path_axi,
    joint_columns_cache,
    minute_columns_cache,
    precompute_joint,
    precompute_minute,
    precompute_open_days,
)
from poc_web.api.main import _csv_minute_series, _leg_cache, _mae_usage
from src.risk.mc_portfolio import BlockRecord, StrategyMC
from src.risk.trades import load_trade_list

D0 = pd.Timestamp("2024-03-04")        # lunes
FRI = pd.Timestamp("2024-03-08")
MON = pd.Timestamp("2024-03-11")


def at(hhmm: str, day: pd.Timestamp = D0) -> pd.Timestamp:
    h, m = hhmm.split(":")
    return day + pd.Timedelta(hours=int(h), minutes=int(m))


def serie(ot, ct, pts):
    """Entrada de ``minute_valle``: (ts, vals, ot, ct). ``pts`` = [(ts, valor)]."""
    ts = np.array([pd.Timestamp(p[0]) for p in pts], dtype="datetime64[ns]")
    vals = np.array([p[1] for p in pts], dtype=float)
    return ts, vals, pd.Timestamp(ot), pd.Timestamp(ct)


def mbs(*entries):
    """``minute_by_sym`` desde (sym, local, serie)."""
    out: dict = {}
    for s, i, ser in entries:
        out.setdefault(s, {})[i] = ser
    return out


def env_blocks(groups, n_f: int = 1, lots=(1.0,)):
    """Bloques AXI a mano, uno por grupo de trades: (sym, local_id, open_ts, close_ts, pnl_por_lote).

    Devuelve ``(blocks, strats, strategies, lots_by, pv_by)``: ``strats`` es el dict que consume
    ``precompute_open_days``; ``strategies`` la LISTA en el orden de ``sym_id`` que consume
    ``minute_columns_cache`` (E7). ``lots``: un valor por fila de f (se repite al ancho de la pata)."""
    all_t = [t for g in groups for t in g]
    syms = sorted({t[0] for t in all_t})
    n_local = {s: 1 + max(t[1] for t in all_t if t[0] == s) for s in syms}
    open_t = {s: np.full(n_local[s], np.datetime64("NaT"), dtype="datetime64[ns]") for s in syms}
    close_t = {s: np.full(n_local[s], np.datetime64("NaT"), dtype="datetime64[ns]") for s in syms}
    for s, loc, ot, ct, _ in all_t:
        open_t[s][loc] = np.datetime64(pd.Timestamp(ot))
        close_t[s][loc] = np.datetime64(pd.Timestamp(ct))
    blocks = []
    for k, trades in enumerate(groups):
        start = min(pd.Timestamp(t[2]) for t in trades)
        end = max(pd.Timestamp(t[3]) for t in trades)
        blocks.append(BlockRecord(
            idx=k, start=start, end=end, duration_days=(end - start).total_seconds() / 86400.0,
            sym_id=np.array([t[0] for t in trades], dtype=int),
            trade_local_id=np.array([t[1] for t in trades], dtype=int),
            close_d=np.array([pd.Timestamp(t[3]).date() for t in trades]),
            valle_precio=[{pd.Timestamp(t[3]).date(): -1.0} for t in trades],
            pnl_per_lot=np.array([t[4] for t in trades], dtype=float),
            n_unique_trades=len(trades)))
    strats = {s: SimpleNamespace(open_t=open_t[s], close_t=close_t[s]) for s in syms}
    lots_by = {s: np.array([[lots[min(r, len(lots) - 1)]] * n_local[s] for r in range(n_f)], dtype=float)
               for s in syms}
    pv_by = {s: 1.0 for s in syms}
    return blocks, strats, list(strats.values()), lots_by, pv_by


def env(trades, n_f: int = 1, lots=(1.0,)):
    """Un solo bloque (ver :func:`env_blocks`)."""
    return env_blocks([trades], n_f, lots)


# ---------------------------------------------------------------------------
# Escenario E4 (a mano): Enmienda v2 E4 del plan, un bloque y un día, lotes 1, pv 1, series en $
#   Pata A: abre 09:00, cierra 10:00 con realizado -400; serie {09:00: 0, 09:30: -200, 10:00: -400}.
#   Pata B: abre 09:00, cierra 16:00 con realizado -50; serie {09:00: 0, 10:00: -100, 14:00: -300, 16:00: -50}.
# ---------------------------------------------------------------------------
def _e4_trades(day: pd.Timestamp = D0, k: int = 0):
    return [(0, k, at("09:00", day), at("10:00", day), -400.0), (1, k, at("09:00", day), at("16:00", day), -50.0)]


def _e4_series(day: pd.Timestamp = D0, k: int = 0):
    return [
        (0, k, serie(at("09:00", day), at("10:00", day),
                     [(at("09:00", day), 0.0), (at("09:30", day), -200.0), (at("10:00", day), -400.0)])),
        (1, k, serie(at("09:00", day), at("16:00", day),
                     [(at("09:00", day), 0.0), (at("10:00", day), -100.0), (at("14:00", day), -300.0),
                      (at("16:00", day), -50.0)])),
    ]


def _e4():
    return env(_e4_trades()), mbs(*_e4_series())


def test_ac2_2_golden_axi_minute_menos_700_y_joint_menos_500():
    """E4. Joint (columnas vivas en [ot, ct]): mínimos por minuto 0 / -200 / -500 / -300 / -50 → -500: no suma el
    realizado de A (-400) con el flotante de B a las 14:00. Minute (vivas en [ot, ct), realizado EN ct): 0 / -200 /
    (-400 - 100) = -500 / (-400 - 300) = -700 / (-400 - 50) = -450 → -700."""
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    cache = minute_columns_cache(blocks, strategies, m)
    (days, merged0, total), = precompute_minute(blocks, lots_by, pv_by, 0, cache)
    assert list(days) == [D0.date()]
    np.testing.assert_array_equal(merged0, [-700.0])
    assert total == pytest.approx(-450.0)
    jcols = joint_columns_cache(blocks, m)
    (days_j, merged_j, total_j), = precompute_joint(blocks, lots_by, pv_by, 0, m, jcols)
    assert list(days_j) == [D0.date()]
    np.testing.assert_array_equal(merged_j, [-500.0])
    assert total_j == pytest.approx(-450.0)


def test_precompute_minute_tiene_el_mismo_formato_que_precompute_joint():
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    cache = minute_columns_cache(blocks, strategies, m)
    out_m = precompute_minute(blocks, lots_by, pv_by, 0, cache)
    out_j = precompute_joint(blocks, lots_by, pv_by, 0, m, joint_columns_cache(blocks, m))
    assert isinstance(out_m, list) and len(out_m) == len(out_j) == len(blocks)
    days, merged0, total = out_m[0]
    assert isinstance(days, np.ndarray) and isinstance(merged0, np.ndarray) and isinstance(total, float)
    assert all(type(d) is dt.date for d in days)                     # `date`, no Timestamp ni ordinal
    assert merged0.dtype == np.float64 and merged0.shape == days.shape
    # mismas claves de día en el caso de un solo día de calendario
    assert list(days) == list(out_j[0][0])


def test_precompute_minute_es_lineal_en_los_lotes_y_no_muta_la_cache():
    """Dos filas de f (lotes 1 y 2): el mínimo diario escala exactamente ×2; llamar de nuevo no cambia nada."""
    blocks, strats, strategies, lots_by, pv_by = env(_e4_trades(), n_f=2, lots=(1.0, 2.0))
    m = mbs(*_e4_series())
    cache = minute_columns_cache(blocks, strategies, m)
    snap = [c.adv_cols[0].copy() for c in cache]
    f0 = precompute_minute(blocks, lots_by, pv_by, 0, cache)[0]
    f1 = precompute_minute(blocks, lots_by, pv_by, 1, cache)[0]
    np.testing.assert_array_equal(f0[1], [-700.0])
    np.testing.assert_array_equal(f1[1], 2.0 * f0[1])
    assert f1[2] == 2.0 * f0[2]
    for c, s in zip(cache, snap):
        np.testing.assert_array_equal(c.adv_cols[0], s)
    again = precompute_minute(blocks, lots_by, pv_by, 0, cache)[0]
    np.testing.assert_array_equal(again[1], f0[1])


def test_la_cache_descarta_la_favorable_como_alias_de_la_adversa():
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    cache = minute_columns_cache(blocks, strategies, m)
    assert len(cache) == 1 and cache[0].fav_cols is cache[0].adv_cols


def test_los_bloques_son_independientes_y_el_walker_suma_el_saldo():
    """``merged0`` es relativo al inicio de CADA bloque (saldo_start = 0): el escenario E4 repetido el martes da el
    mismo ``merged0`` (-700). El realizado del primer bloque (-450) lo suma el walker como ``saldo``: con línea de
    ruina -1.100 el martes toca en minute (-450 - 700 = -1.150) y no en joint (-450 - 500 = -950)."""
    tue = D0 + pd.Timedelta(days=1)
    blocks, strats, strategies, lots_by, pv_by = env_blocks([_e4_trades(D0, 0), _e4_trades(tue, 1)])
    m = mbs(*_e4_series(D0, 0), *_e4_series(tue, 1))
    pre_m = precompute_minute(blocks, lots_by, pv_by, 0, minute_columns_cache(blocks, strategies, m))
    np.testing.assert_array_equal(pre_m[0][1], [-700.0])
    np.testing.assert_array_equal(pre_m[1][1], [-700.0])
    assert pre_m[0][2] == pre_m[1][2] == pytest.approx(-450.0)
    assert [pre_m[0][0][0], pre_m[1][0][0]] == [D0.date(), tue.date()]
    pre_j = precompute_joint(blocks, lots_by, pv_by, 0, m, joint_columns_cache(blocks, m))
    o_m = _walk(pre_m, blocks, strats, lots_by, pv_by, ruin_line=-1100.0)
    o_j = _walk(pre_j, blocks, strats, lots_by, pv_by, ruin_line=-1100.0)
    assert o_m.outcome == "ruin" and o_m.terminal == pytest.approx(-1150.0) and o_m.n_blocks == 2
    assert o_j.outcome == "neither"


# ---------------------------------------------------------------------------
# Propiedad: precompute_minute == referencia por fuerza bruta (minuto a minuto), bloques aleatorios multi-pata y multi-día
# ---------------------------------------------------------------------------
def _bloque_aleatorio(rng):
    """1-5 trades en 2 patas, abiertos entre 1 min y 3 días, con series de 1-7 puntos (el primero en la apertura)."""
    k = int(rng.integers(1, 6))
    base = D0 + pd.Timedelta(minutes=int(rng.integers(0, 3 * 24 * 60)))
    trades, series, n_loc, lots = [], {}, {0: 0, 1: 0}, {0: [], 1: []}
    for q in range(k):
        s = 0 if q == 0 else int(rng.integers(0, 2))                    # la pata 0 siempre está (la lista `strategies`)
        li = n_loc[s]
        n_loc[s] += 1
        ot = (base + pd.Timedelta(minutes=int(rng.integers(0, 2 * 24 * 60)))).floor("min")
        ct = ot + pd.Timedelta(minutes=int(rng.integers(1, 3 * 24 * 60)))
        span = int((ct - ot).total_seconds() // 60)
        offs = np.unique(np.r_[0, rng.integers(0, span, size=int(rng.integers(1, 7)))])
        pts = [(ot + pd.Timedelta(minutes=int(o)), float(rng.normal(-0.4, 0.8))) for o in offs]
        trades.append((s, li, ot, ct, float(rng.normal(0.0, 3.0))))
        series.setdefault(s, {})[li] = serie(ot, ct, pts)
        lots[s].append(float(rng.choice([0.5, 1.0, 1.5, 2.0])))
    return trades, series, lots


def _referencia(trades, series, lots, pv):
    """Mínimo diario de ``realizado_≤t + Σ flotante de las vivas en [ot, ct)`` por fuerza bruta: el realizado de un trade
    entra EN su ``ct``; el flotante es el último punto de su serie con ``ts ≤ t`` (0 antes del primero)."""
    mom = {t[2] for t in trades} | {t[3] for t in trades}
    for s, d in series.items():
        for _, (ts, _v, _o, _c) in d.items():
            mom |= {pd.Timestamp(x) for x in ts}
    por_dia: dict = {}
    for t in sorted(mom):
        real = sum(lots[s][li] * ppl for s, li, ot, ct, ppl in trades if ct <= t)
        flot = 0.0
        for s, li, ot, ct, ppl in trades:
            if ot <= t < ct:
                ts, vals, _o, _c = series[s][li]
                prev = [v for x, v in zip(ts, vals) if pd.Timestamp(x) <= t]
                flot += lots[s][li] * pv[s] * (prev[-1] if prev else 0.0)
        d = t.date()
        por_dia[d] = min(por_dia.get(d, np.inf), real + flot)
    return por_dia


@pytest.mark.parametrize("semilla", range(40))
def test_precompute_minute_coincide_con_la_referencia_por_fuerza_bruta(semilla):
    rng = np.random.default_rng(1000 + semilla)
    trades, series, lots = _bloque_aleatorio(rng)
    pv = {0: 2.0, 1: 0.5}
    blocks, strats, strategies, _, _ = env_blocks([trades])
    lots_by = {s: np.array([lots[s]], dtype=float) for s in lots if lots[s]}
    cache = minute_columns_cache(blocks, strategies, series)
    (days, merged0, total), = precompute_minute(blocks, lots_by, pv, 0, cache)
    ref = _referencia(trades, series, lots, pv)
    assert list(days) == sorted(ref)                                   # mismos días, ordenados
    np.testing.assert_allclose(merged0, [ref[d] for d in days], rtol=1e-9, atol=1e-9)
    assert total == pytest.approx(sum(lots[s][li] * ppl for s, li, _o, _c, ppl in trades))


# ---------------------------------------------------------------------------
# E6: conjunto de días
# ---------------------------------------------------------------------------
def _weekend_block():
    """X abre el viernes 10:00 y cierra el lunes 10:00 (realizado +20); Y abre y cierra el viernes (realizado -30)."""
    trades = [(0, 0, at("10:00", FRI), at("10:00", MON), 20.0), (1, 0, at("11:00", FRI), at("15:00", FRI), -30.0)]
    m = mbs(
        (0, 0, serie(at("10:00", FRI), at("10:00", MON),
                     [(at("10:00", FRI), -10.0), (at("16:00", FRI), -40.0), (at("09:00", MON), -5.0)])),
        (1, 0, serie(at("11:00", FRI), at("15:00", FRI),
                     [(at("11:00", FRI), -5.0), (at("14:00", FRI), -35.0), (at("15:00", FRI), -30.0)])))
    return env(trades), m


def test_e6_sin_filas_de_relleno_entre_la_apertura_y_el_cierre_y_dias_ordenados():
    """Un fin de semana con una posición abierta NO añade filas (a diferencia del modo minute del lente prop firm):
    los días son los de los minutos propios ∪ los de cierre, igual que el joint de hoy."""
    (blocks, strats, strategies, lots_by, pv_by), m = _weekend_block()
    cache = minute_columns_cache(blocks, strategies, m)
    (days, merged0, total), = precompute_minute(blocks, lots_by, pv_by, 0, cache)
    assert list(days) == [FRI.date(), MON.date()]                    # sin sábado ni domingo
    assert [d.weekday() for d in days] == [4, 0]
    assert all(days[i] < days[i + 1] for i in range(len(days) - 1))
    # Viernes: 10:00 -10 · 11:00 -15 · 14:00 -45 · 15:00 (Y cerrada: -30; X -10) -40 · 16:00 (-30 + -40) = -70.
    # Lunes: 09:00 (-30 + -5) = -35 · 10:00 (X cerrada: -30 + 20) = -10.
    np.testing.assert_array_equal(merged0, [-70.0, -35.0])
    assert total == pytest.approx(-10.0)


def test_precompute_minute_equivale_al_modo_minute_del_motor_en_los_dias_comunes():
    """La construcción de AXI coincide con ``block_day_arrays_portfolio(float_mode="minute")`` del lente prop firm sobre
    la misma materia prima (``merged0`` = ``rel_float``). Dif. de diseño (E6): el motor añade las filas del sábado y del
    domingo con la posición abierta; AXI no."""
    from src.risk.propfirm import PortfolioDayBlock, block_day_arrays_portfolio
    (blocks, strats, strategies, lots_by, pv_by), m = _weekend_block()
    (days, merged0, total), = precompute_minute(blocks, lots_by, pv_by, 0,
                                                minute_columns_cache(blocks, strategies, m))
    b = blocks[0]
    ot = np.array([strats[int(s)].open_t[int(i)] for s, i in zip(b.sym_id, b.trade_local_id)], dtype="datetime64[ns]")
    ct = np.array([strats[int(s)].close_t[int(i)] for s, i in zip(b.sym_id, b.trade_local_id)], dtype="datetime64[ns]")
    pdb = PortfolioDayBlock(
        idx=0, start=b.start, end=b.end, duration_days=b.duration_days, sym_id=b.sym_id, trade_local_id=b.trade_local_id,
        open_ord=np.array([pd.Timestamp(t).toordinal() for t in ot]),
        close_ord=np.array([pd.Timestamp(t).toordinal() for t in ct]), valle_precio=b.valle_precio,
        pnl_per_lot=b.pnl_per_lot, n_trades=2, n_no_valley=0, open_t=ot, close_t=ct)
    arr = block_day_arrays_portfolio(pdb, lots_by, pv_by, 0, float_mode="minute", minute_by_sym=m)
    motor = {dt.date.fromordinal(int(o)): v for o, v in zip(arr.day_ord, arr.rel_float)}
    assert len(motor) == 4 and len(days) == 2                          # el motor rellena sábado y domingo; AXI no
    for d, v in zip(days, merged0):
        assert motor[d] == pytest.approx(v, abs=1e-12)
    assert total == pytest.approx(arr.total_pnl)


def test_e6_el_joint_de_hoy_usa_el_mismo_conjunto_de_dias():
    (blocks, strats, strategies, lots_by, pv_by), m = _weekend_block()
    (days_j, _, _), = precompute_joint(blocks, lots_by, pv_by, 0, m, joint_columns_cache(blocks, m))
    cache = minute_columns_cache(blocks, strategies, m)
    (days_m, _, _), = precompute_minute(blocks, lots_by, pv_by, 0, cache)
    assert list(days_m) == list(days_j) == [FRI.date(), MON.date()]


def test_e6_un_trade_sin_serie_aporta_su_dia_de_cierre_y_solo_el_realizado():
    """Sin serie en ``minute_by_sym`` el trade no aporta flotante, pero su minuto de cierre sí entra en el calendario
    (el realizado cuenta): el día de cierre está en ``days``."""
    trades = [(0, 0, at("09:00"), at("10:00"), -80.0)]
    blocks, strats, strategies, lots_by, pv_by = env(trades)
    cache = minute_columns_cache(blocks, strategies, {})
    (days, merged0, total), = precompute_minute(blocks, lots_by, pv_by, 0, cache)
    assert list(days) == [D0.date()]
    np.testing.assert_array_equal(merged0, [-80.0])
    assert total == pytest.approx(-80.0)


# ---------------------------------------------------------------------------
# E7: ``strategies`` es una lista indexada por sym_id
# ---------------------------------------------------------------------------
def test_e7_strategies_se_indexa_por_sym_id_el_orden_importa():
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    assert isinstance(strategies, list) and strategies == list(strats.values())
    ok = precompute_minute(blocks, lots_by, pv_by, 0, minute_columns_cache(blocks, strategies, m))
    assert ok[0][1][0] == -700.0
    # con las patas permutadas, a cada trade se le asignan los ot/ct de la otra: el resultado cambia
    # (A queda viva hasta las 16:00 y B muere a las 10:00): 09:30 -200 · 10:00 (B cerrada -50; A -400) -450 · 16:00 -450.
    bad = precompute_minute(blocks, lots_by, pv_by, 0,
                            minute_columns_cache(blocks, strategies[::-1], m))
    assert bad[0][1][0] == pytest.approx(-450.0) and bad[0][1][0] != ok[0][1][0]


def test_e7_un_dict_de_strategies_por_nombre_falla_en_voz_alta():
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    with pytest.raises(TypeError, match="LISTA"):
        minute_columns_cache(blocks, {"a.csv": strategies[0], "b.csv": strategies[1]}, m)
    assert minute_columns_cache(blocks, tuple(strategies), m)             # una tupla en orden también vale


def test_cache_y_bloques_deben_coincidir():
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    cache = minute_columns_cache(blocks, strategies, m)
    with pytest.raises(ValueError, match="no coinciden"):
        precompute_minute(blocks, lots_by, pv_by, 0, cache + cache)
    with pytest.raises(ValueError, match="no coinciden"):
        precompute_minute(blocks, lots_by, pv_by, 0, [])


# ---------------------------------------------------------------------------
# El walker no cambia: decide sobre ``merged0 + saldo`` de la serie precomputada
# ---------------------------------------------------------------------------
def _walk(pre, blocks, strats, lots_by, pv_by, ruin_line):
    open_days = precompute_open_days(blocks, strats)
    return evaluate_path_axi(np.arange(len(blocks)), blocks, open_days, lots_by, pv_by, 0,
                             pass_line=1000.0, ruin_line=ruin_line, baseline=1000.0,
                             min_unique_trades=None, gap_mean=0.0, mtm="joint", joint_pre=pre)


def test_el_walker_ve_la_ruina_con_el_realizado_en_minute_y_no_en_joint():
    """Línea de ruina -600: el mínimo del día en joint (-500) no la toca; en minute (-700, con el realizado de A) sí."""
    (blocks, strats, strategies, lots_by, pv_by), m = _e4()
    pre_m = precompute_minute(blocks, lots_by, pv_by, 0, minute_columns_cache(blocks, strategies, m))
    pre_j = precompute_joint(blocks, lots_by, pv_by, 0, m, joint_columns_cache(blocks, m))
    o_m = _walk(pre_m, blocks, strats, lots_by, pv_by, ruin_line=-600.0)
    o_j = _walk(pre_j, blocks, strats, lots_by, pv_by, ruin_line=-600.0)
    assert o_m.outcome == "ruin" and o_m.terminal == -700.0
    assert o_j.outcome == "neither"
    # y con la línea en -750 ninguna convención la toca
    assert _walk(pre_m, blocks, strats, lots_by, pv_by, ruin_line=-750.0).outcome == "neither"


# ---------------------------------------------------------------------------
# E8: contadores de los trades sin M1 en los bloques AXI (BlockRecord)
# ---------------------------------------------------------------------------
_HEADER = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n")


def _csv_leg(rows, *, covered=()):
    """Pata CSV mínima: ``rows`` = (open, close, size, pnl, mae); ``covered`` = índices con serie M1 propia."""
    lines, bal = [], 10_000.0
    for i, (o, c, size, pnl, mae) in enumerate(rows):
        bal = round(bal + pnl, 2)
        lines.append(f"{i + 1};XAUUSD_axi;Buy;{o};2000.00;{size:.2f};{c};2001.00;{pnl:.2f};{bal:.2f};IST;"
                     f"Exit After 1 Bars;{mae:.2f};10;1;-\n")
    tl = load_trade_list((_HEADER + "".join(lines)).encode())
    ot = pd.to_datetime(tl.df["open_time"]).to_numpy()
    ct = pd.to_datetime(tl.df["close_time"]).to_numpy()
    mv = {i: serie(ot[i], ct[i], [(ot[i], -0.1)]) for i in covered}
    s = StrategyMC(key="leg", open_t=ot, close_t=ct,
                   close_d=pd.to_datetime(tl.df["close_time"]).dt.date.to_numpy(), valle={},
                   pnl_per_lot=tl.pnl_per_lot(), is_buy=np.ones(tl.n_trades, dtype=bool), pv=100.0,
                   lots=np.ones((1, tl.n_trades)), n_trades=tl.n_trades, minute_valle=mv)
    return {"name": "leg.csv", "tl": tl, "s": s}


def test_e8_contadores_en_bloques_axi_con_un_trade_sin_m1():
    """Los bloques reales de AXI (``build_block_records``) solo incluyen trades con valle, así que hoy el contador sale
    a 0; pero si un bloque lleva un trade sin serie propia, ``_mae_usage`` lo cuenta (acotado por su MAE o a 0)."""
    leg = _csv_leg([("2024.03.04 09:00", "2024.03.04 10:00", 0.10, 30.0, -10.0),    # 0: con serie M1
                    ("2024.03.05 09:00", "2024.03.05 15:00", 0.50, -120.0, -200.0)],  # 1: sin M1, con MAE
                   covered=(0,))
    block = BlockRecord(
        idx=0, start=D0, end=D0, duration_days=1.0, sym_id=np.array([0, 0]), trade_local_id=np.array([0, 1]),
        close_d=np.array([dt.date(2024, 3, 4), dt.date(2024, 3, 5)]), valle_precio=[{}, {}],
        pnl_per_lot=np.array([300.0, -240.0]), n_unique_trades=2)
    lots_by = {0: np.array([[1.0, 2.0]])}
    u = _mae_usage([block], [leg], lots_by, 0)
    assert (u["n_mae_bounded"], u["n_mae_zero"]) == (1, 0)
    # pesos |lotes · PnL por lote|: trade 0 = 300; trade 1 = 2 · 240 = 480 → 480 / 780
    assert u["pnl_w_mae_bounded"] == pytest.approx(480.0 / 780.0)
    assert u["pnl_w_mae_zero"] == 0.0
    # la serie sintética del trade 1: MAE / (size · pv) = -200 / (0,5 · 100) = -4 por unidad de precio
    series, nb, nz = _csv_minute_series(leg)
    assert (nb, nz) == (1, 0) and series[1][1][0] == pytest.approx(-4.0)
    # y el walker puede consumir la serie: en el cálculo por minuto el trade 1 (2 lotes) aporta -4 · 2 · 100 = -800
    # desde su apertura (martes 05, 09:00) hasta el cierre exclusive (15:00), donde entra el realizado -240 · 2 = -480.
    strategies = [leg["s"]]
    cache = minute_columns_cache([block], strategies, {0: series})
    (days, merged0, total), = precompute_minute([block], lots_by, {0: 100.0}, 0, cache)
    assert list(days) == [dt.date(2024, 3, 4), dt.date(2024, 3, 5)]
    # lunes 04: serie con un solo punto de -0.1 · 1 · 100 = -10 en la apertura (ffill hasta el cierre); cierre +300
    # martes 05: realizado lunes +300 → 300 - 800 = -500 a las 09:00; al cierre 300 - 480 = -180 → mínimo -500
    # (el pv se pasa 100: las series están en precio y el motor reescala por lotes · pv)
    np.testing.assert_allclose(merged0, [-10.0, -500.0])
    assert total == pytest.approx(300.0 - 480.0)


# ---------------------------------------------------------------------------
# Cartera sintética de extremo a extremo (portfolio_load + simulate_axi)
# ---------------------------------------------------------------------------
_STEP = 0.01
_CSV_CAPITAL = 50_000.0
_ACCOUNT = 100_000.0
_F_REF = 0.01
_POLICY_JOINT_SHA256 = "6a1a10c58babb1e926150735e130a67bc09f31c3e4aa80135309feedd6b23e17"
_FLOAT_CONVENTION_JOINT = ("joint-m1 en las decisiones (min diario de la suma de flotantes M1 de las posiciones "
                           "vivas); first_cross/raw en bound (peor-punto-por-día, referencia motor)")
_FLOAT_CONVENTION_MINUTE = "flotante por minuto con el realizado intradía (decisiones); raw y first_cross en bound"


def _synth_files(tmp_path, *, n: int = 160, gap_every: int = 10):
    """CSV orderstocsv + OHLC M1 de XAUUSD_axi: un trade por día hábil (09:00-11:00), backtest que compone, 60 % de
    aciertos, y cada ``gap_every``-ésimo trade SIN barras M1 (sin valle)."""
    days = pd.bdate_range("2023-01-02", periods=n)
    e = _CSV_CAPITAL
    rows, frames, covered = [], [], []
    for i, d in enumerate(days):
        sl = 1500.0 + 10.0 * (i % 5)
        sz = round(float(np.floor(_F_REF * e / sl / _STEP) * _STEP), 2)
        win = (i * 7 + 3) % 10 < 6
        ppl = (1.2 if win else -1.0) * sl
        p = round(sz * ppl, 2)
        e = round(e + p, 2)
        ot, ct = d + pd.Timedelta(hours=9), d + pd.Timedelta(hours=11)
        exit_px = 2000.0 + ppl / 100.0
        rows.append(f"{i + 1};XAUUSD_axi;Buy;{ot:%Y.%m.%d %H:%M};2000.00;{sz:.2f};{ct:%Y.%m.%d %H:%M};{exit_px:.2f};"
                    f"{p:.2f};{e:.2f};IST;Exit After 1 Bars;{-abs(ppl) * sz * 0.5:.2f};{abs(ppl) * sz:.2f};1;-\n")
        has_m1 = not (gap_every and i % gap_every == gap_every - 1)
        covered.append(has_m1)
        if has_m1:
            idx = pd.date_range(ot, ct, freq="1min")
            path = 2000.0 + (exit_px - 2000.0) * np.linspace(0.0, 1.0, len(idx))
            frames.append(pd.DataFrame({"o": path, "h": path + 0.3, "l": path - 0.3 - (0.0 if win else 1.0),
                                        "c": path}, index=idx))
    csv = tmp_path / "axi_minute_synth_leg.csv"
    csv.write_text(_HEADER + "".join(rows))
    return SimpleNamespace(csv=csv, ohlc=pd.concat(frames), covered=np.array(covered))


@pytest.fixture
def synth(tmp_path, monkeypatch):
    """Cartera sintética de una pata CSV cargada con ``portfolio_load`` (modo rápido: sin .sqx)."""
    s = _synth_files(tmp_path)
    monkeypatch.setattr(main_mod, "_load_ohlc", lambda sym: s.ohlc)
    before = set(_leg_cache)
    res = main_mod.portfolio_load(main_mod.PortfolioLoadRequest(
        legs=[str(s.csv)], account_size=_ACCOUNT, f_ref=_F_REF))
    assert res["ok"], res
    s.pid = res["portfolio_id"]
    yield s
    main_mod._portfolios.pop(s.pid, None)
    for k in set(_leg_cache) - before:
        _leg_cache.pop(k, None)


def _axi(synth, **kw):
    return main_mod.simulate_axi(main_mod.AxiSimRequest(
        portfolio_id=synth.pid, stage="seed", f_lo=0.01, f_hi=0.02, f_step=0.01, n_paths=100, **kw))


def test_ac2_6_axi_float_convention_y_policy_hablan_del_realizado(synth):
    assert main_mod.AXI_FLOAT_MODE == "minute"
    meta = _axi(synth)["meta"]
    assert meta["float_convention"] == _FLOAT_CONVENTION_MINUTE
    assert "realizado" in meta["float_convention"] and "realizado intradía" in meta["policy_assumption"]
    assert "JOINT" not in meta["policy_assumption"]
    assert "minuto" in meta["policy_assumption"]
    # los campos raw siguen en bound (sub-pase del motor): lo dicen los dos textos
    assert "bound" in meta["float_convention"] and "bound" in meta["policy_assumption"]


def test_e8_la_lente_axi_devuelve_los_contadores_y_los_trades_sin_m1_quedan_fuera_de_los_bloques(synth):
    """Hoy ``build_block_records`` (motor) solo incluye trades con valle: los 16 trades sin M1 de la cartera sintética
    NO entran en los bloques AXI (``meta.coverage`` = 144/160) y, por tanto, ninguno se acota con su MAE: los
    contadores existen y valen 0, y no hay aviso."""
    out = _axi(synth)
    meta = out["meta"]
    assert meta["coverage"] == pytest.approx(144 / 160)
    assert (meta["n_mae_bounded"], meta["n_mae_zero"]) == (0, 0)
    assert meta["pnl_w_mae_bounded"] == 0.0 and meta["pnl_w_mae_zero"] == 0.0
    assert not [w for w in out["warnings"] if "MAE" in w or "OHLC M1" in w]


def test_axi_minute_usa_las_funciones_nuevas_con_strategies_en_lista_y_series_sinteticas(synth, monkeypatch):
    seen = {}
    real_cols, real_pre = main_mod.minute_columns_cache, main_mod.precompute_minute

    def spy_cols(blocks, strategies, minute_by_sym):
        seen["strategies"], seen["mbs"] = strategies, minute_by_sym
        return real_cols(blocks, strategies, minute_by_sym)

    pre_calls = []

    def spy_pre(blocks, lots_by, pv_by, fi, cache):
        pre_calls.append(fi)
        return real_pre(blocks, lots_by, pv_by, fi, cache)

    monkeypatch.setattr(main_mod, "minute_columns_cache", spy_cols)
    monkeypatch.setattr(main_mod, "precompute_minute", spy_pre)
    monkeypatch.setattr(main_mod, "joint_columns_cache", lambda *a, **k: pytest.fail("joint no debe usarse"))
    monkeypatch.setattr(main_mod, "precompute_joint", lambda *a, **k: pytest.fail("joint no debe usarse"))
    # bloque-politis-white (E7, opción 1): el bloque que daba la regla vieja en este escenario (10,0), fijado; y SIN la
    # sensibilidad con bloque 1, que con un bloque > 1 añade una evaluación más en f* (este test cuenta las del barrido).
    monkeypatch.setattr(main_mod, "_axi_bs1", lambda *a, **k: (None, []))
    _axi(synth, block_size=10.0)
    leg_s = main_mod._portfolios[synth.pid]["legs"][0]["s"]
    assert isinstance(seen["strategies"], list) and seen["strategies"] == [leg_s]      # E7
    assert pre_calls == [0, 1]                                                          # una vez por f
    mv = seen["mbs"][0]
    assert len(mv) == 160 and len(leg_s.minute_valle) == 144                            # 16 sintéticas añadidas...
    assert all(i in mv for i in leg_s.minute_valle) and all(mv[i] is leg_s.minute_valle[i]
                                                            for i in leg_s.minute_valle)
    assert len(leg_s.minute_valle) == 144                                               # ...sin tocar `s.minute_valle`


def test_flag_axi_joint_reproduce_el_camino_anterior(synth, monkeypatch):
    """Con ``AXI_FLOAT_MODE = "joint"`` se usan ``joint_columns_cache`` + ``precompute_joint`` con el ``minute_valle``
    original, y el ``meta`` es el de antes (textos literales de la fase 1; sin claves nuevas)."""
    monkeypatch.setattr(main_mod, "AXI_FLOAT_MODE", "joint")
    seen = {}
    real_cols = main_mod.joint_columns_cache

    def spy_cols(blocks, minute_by_sym):
        seen["mbs"] = minute_by_sym
        return real_cols(blocks, minute_by_sym)

    monkeypatch.setattr(main_mod, "joint_columns_cache", spy_cols)
    monkeypatch.setattr(main_mod, "minute_columns_cache", lambda *a, **k: pytest.fail("minute no debe usarse"))
    monkeypatch.setattr(main_mod, "precompute_minute", lambda *a, **k: pytest.fail("minute no debe usarse"))
    out = _axi(synth)
    leg_s = main_mod._portfolios[synth.pid]["legs"][0]["s"]
    assert seen["mbs"][0] is leg_s.minute_valle
    meta = out["meta"]
    assert meta["float_convention"] == _FLOAT_CONVENTION_JOINT
    assert hashlib.sha256(meta["policy_assumption"].encode("utf-8")).hexdigest() == _POLICY_JOINT_SHA256
    assert not {"n_mae_bounded", "n_mae_zero", "pnl_w_mae_bounded", "pnl_w_mae_zero"} & set(meta)


def _numeros(out, *claves) -> str:
    """Huella (sha256, 16 hex) de las partes NUMÉRICAS de una respuesta (las ``claves`` de primer nivel), con los
    floats redondeados a 6 decimales (estable entre plataformas)."""
    def r(o):
        if isinstance(o, float):
            return "nan" if o != o else round(o, 6)
        if isinstance(o, dict):
            return {k: r(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [r(v) for v in o]
        return o
    blob = json.dumps({k: r(out.get(k)) for k in claves}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# Huella de ``reports``, ``recommendation`` y ``f_star`` del escenario ``synth`` en el árbol de ANTES de la fase 2 (base
# e0cbf9a con la fase 1 aplicada), con el mismo generador sintético ejecutado en ese árbol.
_PIN_AXI_ANTES_DE_LA_FASE_2 = "aee193c6f31e13e6"


def test_flag_axi_joint_las_cifras_son_las_de_antes_de_la_fase_2(synth, monkeypatch):
    monkeypatch.setattr(main_mod, "AXI_FLOAT_MODE", "joint")
    # bloque-politis-white (E7, opción 1): la cifra fijada es la de la corrida con el bloque que daba la regla vieja en este
    # escenario (10,0); se fija para conservarla bit a bit (la sensibilidad con bloque 1 no toca estas claves, E19).
    assert _numeros(_axi(synth, block_size=10.0), "reports", "recommendation", "f_star") == _PIN_AXI_ANTES_DE_LA_FASE_2


def test_axi_flag_invalido_falla_en_voz_alta(synth, monkeypatch):
    monkeypatch.setattr(main_mod, "AXI_FLOAT_MODE", "minuto")
    with pytest.raises(ValueError, match="AXI_FLOAT_MODE"):
        _axi(synth)


# ---------------------------------------------------------------------------
# Dos patas de extremo a extremo (el orden de las patas y el realizado llegan al endpoint)
# ---------------------------------------------------------------------------
# Precio de XAUUSD (100 $ por punto y lote), igual cada día, en barras de 1 min de 09:00 a 16:00:
#   09:00 → 10:00: 2000 → 2010 · 10:00 → 13:00: 2010 → 2000 · 13:00 → 14:00: 2000 → 1990 · 14:00 → 16:00: 1990 → 2000.
# Pata A: Venta 09:00 → 10:00 a 2000 → 2010 (pierde 10 puntos = 1.000 $ por lote). Pata B: Compra 09:00 → 16:00 a 2000 →
# 2000 (realizado 0, pero a las 14:00 va a -10 puntos = -1.000 $ por lote). A y B forman un bloque cada día.
_KNOT_MIN = [0, 60, 240, 300, 420]
_KNOT_PX = [2000.0, 2010.0, 2000.0, 1990.0, 2000.0]


def _two_leg_files(tmp_path, *, n: int = 40):
    days = pd.bdate_range("2023-01-02", periods=n)
    frames, rows_a, rows_b, bal = [], [], [], 100_000.0
    for i, d in enumerate(days):
        t0 = d + pd.Timedelta(hours=9)
        idx = pd.date_range(t0, t0 + pd.Timedelta(hours=7), freq="1min")
        px = np.interp((idx - t0).total_seconds() / 60.0, _KNOT_MIN, _KNOT_PX)
        frames.append(pd.DataFrame({"o": px, "h": px, "l": px, "c": px}, index=idx))
        bal = round(bal - 1000.0, 2)
        rows_a.append(f"{i + 1};XAUUSD_axi;Sell;{t0:%Y.%m.%d %H:%M};2000.00;1.00;"
                      f"{t0 + pd.Timedelta(hours=1):%Y.%m.%d %H:%M};2010.00;-1000.00;{bal:.2f};IST;Exit After 1 Bars;"
                      "-1500.00;0;1;-\n")
        rows_b.append(f"{i + 1};XAUUSD_axi;Buy;{t0:%Y.%m.%d %H:%M};2000.00;1.00;"
                      f"{t0 + pd.Timedelta(hours=7):%Y.%m.%d %H:%M};2000.00;0.00;100000.00;IST;Exit After 1 Bars;"
                      "-1000.00;0;1;-\n")
    a, b = tmp_path / "axi_leg_a.csv", tmp_path / "axi_leg_b.csv"
    a.write_text(_HEADER + "".join(rows_a))
    b.write_text(_HEADER + "".join(rows_b))
    return SimpleNamespace(csvs=[a, b], ohlc=pd.concat(frames), n=n)


@pytest.fixture
def two_leg(tmp_path, monkeypatch):
    s = _two_leg_files(tmp_path)
    monkeypatch.setattr(main_mod, "_load_ohlc", lambda sym: s.ohlc)
    before = set(_leg_cache)
    res = main_mod.portfolio_load(main_mod.PortfolioLoadRequest(
        legs=[str(p) for p in s.csvs], account_size=_ACCOUNT, f_ref=_F_REF))
    assert res["ok"] and [x["status"] for x in res["legs"]] == ["ok", "ok"], res
    s.pid = res["portfolio_id"]
    yield s
    main_mod._portfolios.pop(s.pid, None)
    for k in set(_leg_cache) - before:
        _leg_cache.pop(k, None)


def test_axi_dos_patas_el_endpoint_pasa_las_patas_en_orden_y_suma_el_realizado(two_leg, monkeypatch):
    """Día 0 (A y B con M1): a las 14:00 A ya cerró (-1.000 · lotes_A realizados) y B va a -1.000 · lotes_B, así que el
    peor momento del día es -1.000 · (lotes_A + lotes_B) y el total del bloque -1.000 · lotes_A. Si el endpoint pasara las
    estrategias en otro orden, a cada trade se le asignarían los ot/ct de la otra pata y el valle de B (14:00) quedaría
    fuera de su máscara [ot, ct)."""
    seen, captured = {}, []
    real_cols, real_pre = main_mod.minute_columns_cache, main_mod.precompute_minute

    def spy_cols(blocks, strategies, minute_by_sym):
        seen["strategies"] = strategies
        return real_cols(blocks, strategies, minute_by_sym)

    def spy_pre(blocks, lots_by, pv_by, fi, cache):
        out = real_pre(blocks, lots_by, pv_by, fi, cache)
        captured.append((fi, lots_by, out))
        return out

    monkeypatch.setattr(main_mod, "minute_columns_cache", spy_cols)
    monkeypatch.setattr(main_mod, "precompute_minute", spy_pre)
    # bloque-politis-white (E7, opción 1): el bloque que daba la regla vieja en este escenario (10,0), fijado; y SIN la
    # sensibilidad con bloque 1, que con un bloque > 1 añade una evaluación más en f* (este test cuenta las del barrido).
    monkeypatch.setattr(main_mod, "_axi_bs1", lambda *a, **k: (None, []))
    _axi(two_leg, block_size=10.0)
    legs = main_mod._portfolios[two_leg.pid]["legs"]
    assert seen["strategies"] == [legs[0]["s"], legs[1]["s"]]
    assert [c[0] for c in captured] == [0, 1] and len(captured[0][2]) == two_leg.n
    for fi, lots_by, out in captured:
        la, lb = lots_by[0][fi, 0], lots_by[1][fi, 0]
        days, merged0, total = out[0]
        assert list(days) == [pd.Timestamp("2023-01-02").date()]
        assert merged0[0] == pytest.approx(-1000.0 * (la + lb), rel=1e-9)
        assert total == pytest.approx(-1000.0 * la, rel=1e-9)
