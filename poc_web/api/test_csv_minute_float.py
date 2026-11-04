"""Flotante por MINUTO con el realizado en las patas CSV de la lente prop firm (guardrail ``csv-d1-d2``, fase 2, D2).

Plan csv-d1-d2 (§2.2 y Enmiendas v2 E3, E4, E8, E9, E10, E11, E16). Todo sintético (no necesita ``risk_app/inputs/``).
Los goldens están escritos a mano en cada test.

Cobertura de aceptación:
  AC2.1  test_ac2_1_*        golden a mano (E4): joint -500, minute -700, cerrado -450; y la variante con la pata B sin M1
  AC2.3  test_ac2_3_*        serie sintética de MAE: valor, constancia hasta el cierre, mínimo del día, contadores,
                             ot == ct y MAE en la barra de cierre (E10: ahí minute es MENOS adverso que joint)
  AC2.4  test_ac2_4_*        perfil con trailing (Apex): bit a bit igual con cualquier ``CSV_FLOAT_MODE``
  AC2.6  test_ac2_6_*        ``meta.float_mode``, ``float_note`` y avisos de la lente
  AC2.7  test_ac2_7_*        par 2D con cartera CSV
  AC2.9  test_ac2_9_*        fin de semana con posición abierta: ``trading_days`` igual en joint y en minute (E11)
  E3     test_e3_*           predicado: minute si y solo si el perfil es estático
  E9     test_e9_*           ``_csv_minute_series`` solo sirve a patas CSV
  E16    test_e16_*          aviso de capital de la carga informativo
  Flags  test_flag_*         ``CSV_FLOAT_MODE = "joint"`` reproduce el camino anterior
  Extra  la cartera de dos patas (Venta que pierde por la mañana + Compra con valle por la tarde) demuestra D2 de
         extremo a extremo (``portfolio_load`` + ``_propfirm_setup``); propiedad con patas CSV aleatorias (parte sin M1)
         contra una referencia por fuerza bruta minuto a minuto; cifras fijadas de la corrida de ANTES de la fase 2
         (``CSV_FLOAT_MODE = "joint"`` y Apex)

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_csv_minute_float.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poc_web.api import main as main_mod
from poc_web.api.main import (
    KIND_SQX,
    _check_float_flag,
    _csv_mae_masks,
    _csv_minute_meta,
    _csv_minute_series,
    _leg_cache,
    _legs_in_strats_order,
    _mae_usage,
    _mae_warnings,
)
from src.risk.input_checks import initial_capital_mismatch
from src.risk.mc_portfolio import StrategyMC
from src.risk.propfirm import (
    BlockDayArrays,
    PortfolioDayBlock,
    PropFirmPhase,
    block_day_arrays_portfolio,
    build_day_blocks_portfolio,
    evaluate_phase_path,
    precompute_day_arrays_portfolio,
)
from src.risk.propfirm_trailing import build_trailing_columns
from src.risk.session_axis import make_session_day_labeler
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


def block(trades) -> PortfolioDayBlock:
    """Bloque a mano. ``trades``: (sym, local_id, open_ts, close_ts, pnl_por_lote)."""
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


def run(b, lots, minute_by_sym, mode="minute", pv=1.0):
    """Un f. ``lots``: {sym: [lote por trade local]}."""
    return block_day_arrays_portfolio(
        b, {s: np.array([v], dtype=float) for s, v in lots.items()}, {s: pv for s in lots}, 0,
        float_mode=mode, minute_by_sym=minute_by_sym)


# ===========================================================================
# AC2.1 — golden a mano (Enmienda v2 E4): un bloque y un día, lotes 1, pv 1, series en $
#   Pata A: abre 09:00, cierra 10:00 con realizado -400; serie {09:00: 0, 09:30: -200, 10:00: -400}.
#   Pata B: abre 09:00, cierra 16:00 con realizado -50; serie {09:00: 0, 10:00: -100, 14:00: -300, 16:00: -50}.
# ===========================================================================
def _e4_block_y_series():
    b = block([(0, 0, at("09:00"), at("10:00"), -400.0), (1, 0, at("09:00"), at("16:00"), -50.0)])
    m = mbs(
        (0, 0, serie(at("09:00"), at("10:00"),
                     [(at("09:00"), 0.0), (at("09:30"), -200.0), (at("10:00"), -400.0)])),
        (1, 0, serie(at("09:00"), at("16:00"),
                     [(at("09:00"), 0.0), (at("10:00"), -100.0), (at("14:00"), -300.0), (at("16:00"), -50.0)])),
    )
    return b, m


def test_ac2_1_golden_e4_joint_menos_500_minute_menos_700_cerrado_menos_450():
    """Joint (columnas vivas en [ot, ct]): mínimos por minuto 0 / -200 / -500 / -300 / -50 → -500; no suma el realizado
    de A (-400) con el flotante de B a las 14:00 (el defecto D2). Minute (vivas en [ot, ct), realizado inyectado EN ct):
    0 / -200 / (-400 - 100) = -500 / (-400 - 300) = -700 / (-400 - 50) = -450 → -700."""
    b, m = _e4_block_y_series()
    joint = run(b, {0: [1.0], 1: [1.0]}, m, mode="joint")
    minute = run(b, {0: [1.0], 1: [1.0]}, m, mode="minute")
    assert list(joint.day_ord) == list(minute.day_ord) == [D0.toordinal()]
    assert joint.rel_float[0] == pytest.approx(-500.0)
    assert minute.rel_float[0] == pytest.approx(-700.0)
    for bda in (joint, minute):
        assert bda.rel_closed[0] == pytest.approx(-450.0)
        assert bda.rel_before[0] == 0.0
        assert bda.total_pnl == pytest.approx(-450.0)
    assert minute.rel_float[0] < joint.rel_float[0] < minute.rel_closed[0]


def test_ac2_1_golden_e4_por_strategymc_y_build_day_blocks_portfolio():
    """Mismo golden pasando por ``StrategyMC`` + ``build_day_blocks_portfolio`` + ``precompute_day_arrays_portfolio``
    (el camino que usa ``_propfirm_setup``)."""
    def strat(key, ot, ct, pnl):
        return StrategyMC(
            key=key, open_t=np.array([ot], dtype="datetime64[ns]"), close_t=np.array([ct], dtype="datetime64[ns]"),
            close_d=pd.to_datetime([ct]).normalize().to_numpy(), valle={}, pnl_per_lot=np.array([pnl]),
            is_buy=np.array([True]), pv=1.0, lots=np.ones((1, 1)), n_trades=1)

    sa = strat("A", at("09:00"), at("10:00"), -400.0)
    sb = strat("B", at("09:00"), at("16:00"), -50.0)
    blocks = build_day_blocks_portfolio({"A": sa, "B": sb})
    assert len(blocks) == 1
    _, m = _e4_block_y_series()
    kw = dict(lots_by_sym={0: sa.lots, 1: sb.lots}, pv_by_sym={0: 1.0, 1: 1.0}, n_f=1, minute_by_sym=m)
    minute = precompute_day_arrays_portfolio(blocks, float_mode="minute", **kw)[0][0]
    joint = precompute_day_arrays_portfolio(blocks, float_mode="joint", **kw)[0][0]
    assert minute.rel_float[0] == pytest.approx(-700.0) and joint.rel_float[0] == pytest.approx(-500.0)
    assert minute.rel_closed[0] == joint.rel_closed[0] == pytest.approx(-450.0)


# ---------------------------------------------------------------------------
# Pata CSV mínima para los tests de ``_csv_minute_series``
# ---------------------------------------------------------------------------
_HEADER = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n")
_HEADER_SIN_MAE = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MFE ($);Time in trade;Comment\n")


def _csv_leg(rows, *, series=None, pv=100.0, with_mae=True, name="leg.csv"):
    """Pata CSV mínima. ``rows``: (tipo, apertura, cierre, size, pnl, mae) EN ORDEN DE CIERRE (``mae`` ``None`` = campo
    vacío). ``series``: {índice: [(ts, valor)]} = trades con serie M1 propia (``minute_valle``/``valle``); el resto no
    tiene cobertura OHLC. ``pv`` = valor del punto de la estrategia (``s.pv``)."""
    lines, bal = [], 10_000.0
    for i, (typ, o, c, size, pnl, mae) in enumerate(rows):
        bal = round(bal + pnl, 2)
        mae_txt = "" if mae is None else f"{mae:.2f}"
        mae_col = f"{mae_txt};" if with_mae else ""
        lines.append(f"{i + 1};XAUUSD_axi;{typ};{o};2000.00;{size:.2f};{c};2001.00;{pnl:.2f};{bal:.2f};IST;"
                     f"Exit After 1 Bars;{mae_col}10;1;-\n")
    tl = load_trade_list(((_HEADER if with_mae else _HEADER_SIN_MAE) + "".join(lines)).encode())
    ot = pd.to_datetime(tl.df["open_time"]).to_numpy()
    ct = pd.to_datetime(tl.df["close_time"]).to_numpy()
    cd = pd.to_datetime(tl.df["close_time"]).dt.date.to_numpy()
    mv, valle = {}, {}
    for i, pts in (series or {}).items():
        mv[i] = serie(ot[i], ct[i], pts)
        valle[i] = {cd[i]: min(p[1] for p in pts)}
    s = StrategyMC(key=name, open_t=ot, close_t=ct, close_d=cd, valle=valle, pnl_per_lot=tl.pnl_per_lot(),
                   is_buy=tl.df["type"].astype(str).str.lower().str.startswith("b").to_numpy(), pv=pv,
                   lots=np.ones((1, tl.n_trades)), n_trades=tl.n_trades, minute_valle=mv)
    return {"name": name, "tl": tl, "s": s}


# Cinco trades en orden de cierre: 0 y 4 con M1 propio; 1, 2 (venta) y 3 (ot == ct) sin cobertura OHLC.
_ROWS = [
    ("Buy", "2024.03.04 09:00", "2024.03.04 10:00", 0.10, 30.0, -10.0),     # 0 con M1
    ("Buy", "2024.03.05 09:00", "2024.03.05 15:00", 0.50, -120.0, -200.0),  # 1 sin M1: v = -200/(0,5·100) = -4
    ("Sell", "2024.03.06 09:00", "2024.03.06 12:00", 0.25, -60.0, -75.0),   # 2 sin M1: v = -75/(0,25·100) = -3
    ("Buy", "2024.03.07 10:00", "2024.03.07 10:00", 0.20, -5.0, -30.0),     # 3 sin M1, ot == ct: v = -30/(0,2·100) = -1,5
    ("Buy", "2024.03.08 09:00", "2024.03.08 11:00", 0.10, 12.0, -4.0),      # 4 con M1
]
_SERIES = {0: [("2024.03.04 09:00", -0.1)], 4: [("2024.03.08 09:00", -0.2)]}


def _leg5(**kw):
    kw.setdefault("series", {i: [(pd.Timestamp(t), v) for t, v in pts] for i, pts in _SERIES.items()})
    return _csv_leg(_ROWS, **kw)


# ===========================================================================
# AC2.3 — serie sintética de MAE para los trades sin cobertura OHLC M1
# ===========================================================================
def test_ac2_3_serie_sintetica_formato_y_valor_de_cada_trade_sin_m1():
    leg = _leg5()
    s = leg["s"]
    before_keys, before_objs = set(s.minute_valle), {i: s.minute_valle[i] for i in s.minute_valle}
    series, n_bounded, n_zero = _csv_minute_series(leg)
    assert set(series) == {0, 1, 2, 3, 4} and (n_bounded, n_zero) == (3, 0)
    for i in (0, 4):                                              # los reales, tal cual (mismos objetos)
        assert series[i] is s.minute_valle[i]
    esperado = {1: -4.0, 2: -3.0, 3: -1.5}                        # MAE / (size · pv), también en la venta
    for i, v in esperado.items():
        ts, vals, ot, ct = series[i]
        assert isinstance(ts, np.ndarray) and ts.dtype == np.dtype("datetime64[ns]") and ts.shape == (1,)
        assert ts[0] == s.open_t[i] and isinstance(ot, pd.Timestamp) and isinstance(ct, pd.Timestamp)
        assert ot == pd.Timestamp(s.open_t[i]) and ct == pd.Timestamp(s.close_t[i])
        assert vals.shape == (1,) and vals.dtype == np.float64
        assert vals[0] == pytest.approx(v, rel=1e-12)
    # no modifica `s.minute_valle` (el camino trailing de Apex sigue viendo las series originales)
    assert set(s.minute_valle) == before_keys and all(s.minute_valle[i] is o for i, o in before_objs.items())
    assert series is not s.minute_valle


def test_ac2_3_el_signo_del_mae_del_csv_da_igual():
    """``load_trade_list`` normaliza el MAE a no-positivo con ``-abs``: un CSV que lo trae en magnitud (positivo) da la
    misma serie."""
    rows = [(*r[:5], abs(r[5])) for r in _ROWS]
    a = _csv_minute_series(_leg5())[0]
    b = _csv_minute_series(_csv_leg(rows, series={i: [(pd.Timestamp(t), v) for t, v in pts]
                                                  for i, pts in _SERIES.items()}))[0]
    for i in (1, 2, 3):
        assert a[i][1][0] == b[i][1][0] and a[i][1][0] < 0


def test_ac2_3_valor_constante_entre_la_apertura_y_el_cierre_y_cero_en_el_cierre():
    """El constructor del trailing mantiene el valor desde ``ot`` hasta justo antes de ``ct`` (envolvente activa en
    ``[ot, ct)``) y lo anula EN ``ct``, donde entra el realizado. Otra pata con barras a las 09:30/10:00/12:00 aporta
    minutos intermedios al calendario."""
    leg = _leg5()
    series = _csv_minute_series(leg)[0]
    d5 = pd.Timestamp("2024-03-05")
    otro = serie(at("09:30", d5), at("12:00", d5),
                 [(at("09:30", d5), -1.0), (at("10:00", d5), -2.0), (at("12:00", d5), -2.0)])
    b = block([(0, 1, at("09:00", d5), at("15:00", d5), -240.0), (1, 0, at("09:30", d5), at("12:00", d5), 5.0)])
    tcols = build_trailing_columns(b, {0: series, 1: {0: otro}}, {0: series, 1: {0: otro}})
    cal = [pd.Timestamp(t) for t in tcols.cal]
    assert cal == [at("09:00", d5), at("09:30", d5), at("10:00", d5), at("12:00", d5), at("15:00", d5)]
    col = tcols.adv_cols[0]                                            # la columna de la pata sintética
    np.testing.assert_array_equal(col, [-4.0, -4.0, -4.0, -4.0, 0.0])


def test_ac2_3_el_minimo_del_dia_incluye_la_serie_sintetica_y_el_joint_de_antes_no():
    """Trade 1 (sin M1): flotante -4 · lotes · pv = -400 a 1 lote desde las 09:00 hasta el cierre (15:00), donde el
    realizado es -240 por lote. El mínimo del día es el MAE (-400), por debajo del realizado (-240); el joint de antes
    contaba 0 (sin serie). Escala con los lotes."""
    leg = _leg5()
    s = leg["s"]
    series = _csv_minute_series(leg)[0]
    blocks = build_day_blocks_portfolio({"leg": s})
    assert len(blocks) == 4                                            # el trade 3 (ot == ct) no forma bloque
    pos = [int(b.trade_local_id[0]) for b in blocks]
    assert pos == [0, 1, 2, 4]
    b1 = blocks[pos.index(1)]
    kw = dict(lots_by_sym={0: s.lots}, pv_by_sym={0: 100.0}, n_f=1)
    minute = precompute_day_arrays_portfolio(blocks, float_mode="minute", minute_by_sym={0: series}, **kw)[0][pos.index(1)]
    joint = precompute_day_arrays_portfolio(blocks, float_mode="joint", minute_by_sym={0: s.minute_valle}, **kw)[0][pos.index(1)]
    assert minute.rel_float[0] == pytest.approx(-400.0)
    assert minute.rel_closed[0] == pytest.approx(-240.0)
    assert joint.rel_float[0] == 0.0 and b1.n_no_valley == 1
    s.lots = np.full((1, s.n_trades), 2.0)
    kw2 = dict(lots_by_sym={0: s.lots}, pv_by_sym={0: 100.0}, n_f=1)
    minute2 = precompute_day_arrays_portfolio(blocks, float_mode="minute", minute_by_sym={0: series}, **kw2)[0][pos.index(1)]
    assert minute2.rel_float[0] == pytest.approx(-800.0)
    # trade 2 (venta, 0,25 lotes originales): -3 · 100 = -300 a 1 lote; realizado -240
    s.lots = np.ones((1, s.n_trades))
    minute3 = precompute_day_arrays_portfolio(blocks, float_mode="minute", minute_by_sym={0: series}, **kw)[0][pos.index(2)]
    assert minute3.rel_float[0] == pytest.approx(-300.0) and minute3.rel_closed[0] == pytest.approx(-240.0)


def test_ac2_3_un_trade_con_ot_igual_ct_solo_aporta_el_realizado():
    """Trade 3 (abre y cierra a las 10:00): la envolvente está activa en ``[ot, ct)`` = vacío, así que su MAE (-150 por
    lote) NO cuenta; queda el realizado (-25 por lote) inyectado en ``ct``."""
    leg = _leg5()
    series = _csv_minute_series(leg)[0]
    ot, ct = pd.Timestamp(leg["s"].open_t[3]), pd.Timestamp(leg["s"].close_t[3])
    assert ot == ct and series[3][1][0] == pytest.approx(-1.5)
    b = block([(0, 3, ot, ct, float(leg["s"].pnl_per_lot[3]))])
    lotes = {0: [1.0] * leg["s"].n_trades}                             # un lote por trade local de la pata
    bda = run(b, lotes, {0: series}, pv=100.0)
    assert bda.rel_float[0] == pytest.approx(-25.0) and bda.rel_closed[0] == pytest.approx(-25.0)
    joint = run(b, lotes, {0: leg["s"].minute_valle}, mode="joint", pv=100.0)
    assert joint.rel_float[0] == 0.0                                   # el joint de antes: sin serie → 0


def test_e10_todo_el_adverso_en_la_barra_de_cierre_minute_es_menos_adverso_que_joint():
    """La envolvente de cada trade muere EN ``ct`` (convención adjudicada del trailing) y el realizado toma el relevo:
    si todo el adverso ocurre en la barra de cierre, el mínimo por minuto no lo ve. Un trade con serie {09:00: 0,
    15:00: -300}, cierre a las 15:00 con +50 por lote: joint = -300 (vivo en [ot, ct]); minute = 0 (09:00; luego +50).
    Es la limitación que dice el `float_note` («lo que ocurriera dentro de la barra de cierre no se cuenta»)."""
    b = block([(0, 0, at("09:00"), at("15:00"), 50.0)])
    m = mbs((0, 0, serie(at("09:00"), at("15:00"), [(at("09:00"), 0.0), (at("15:00"), -300.0)])))
    joint = run(b, {0: [1.0]}, m, mode="joint")
    minute = run(b, {0: [1.0]}, m, mode="minute")
    assert joint.rel_float[0] == pytest.approx(-300.0)
    assert minute.rel_float[0] == pytest.approx(0.0)
    assert minute.rel_float[0] > joint.rel_float[0]                    # minute MENOS adverso (menos pesimista) aquí
    assert "barra de cierre" in main_mod.CSV_MINUTE_FLOAT_NOTE


def test_ac2_3_contadores_csv_sin_columna_mae_todo_cuenta_como_cero():
    leg = _leg5(with_mae=False)
    assert leg["tl"].has_mae is False
    series, n_bounded, n_zero = _csv_minute_series(leg)
    assert (n_bounded, n_zero) == (0, 3)
    for i in (1, 2, 3):
        assert series[i][1][0] == 0.0
    synth, zero = _csv_mae_masks(leg)
    assert synth.tolist() == [False, True, True, True, False] and zero.tolist() == synth.tolist()


def test_ac2_3_contadores_size_no_positivo_o_mae_no_finito_cuentan_como_cero_aparte():
    leg = _leg5()
    tl = leg["tl"]
    size = tl.size.copy()
    size[1] = 0.0                                                      # `load_trade_list` lo rechaza; defensa del helper
    mae = tl.mae.copy()
    mae[2] = np.nan
    leg2 = {**leg, "tl": dataclasses.replace(tl, size=size, mae=mae)}
    series, n_bounded, n_zero = _csv_minute_series(leg2)
    assert (n_bounded, n_zero) == (1, 2)
    assert series[1][1][0] == 0.0 and series[2][1][0] == 0.0
    assert series[3][1][0] == pytest.approx(-1.5)
    mae[2] = -np.inf                                                   # ±inf tampoco es un MAE utilizable
    leg3 = {**leg, "tl": dataclasses.replace(tl, mae=mae)}
    assert _csv_minute_series(leg3)[1:] == (2, 1)


def test_ac2_3_mae_vacio_en_una_fila_del_csv_cuenta_como_cero():
    """Con la columna MAE presente pero vacía en una fila, ``load_trade_list`` la normaliza a 0; el helper la detecta
    en la columna cruda y no la trata como «acotada con MAE 0»."""
    rows = list(_ROWS)
    rows[2] = (*rows[2][:5], None)
    leg = _csv_leg(rows, series={i: [(pd.Timestamp(t), v) for t, v in pts] for i, pts in _SERIES.items()})
    assert leg["tl"].has_mae and leg["tl"].mae[2] == 0.0
    series, n_bounded, n_zero = _csv_minute_series(leg)
    assert (n_bounded, n_zero) == (2, 1) and series[2][1][0] == 0.0


def test_ac2_3_punto_de_valor_no_positivo_deja_todo_a_cero():
    leg = _leg5(pv=0.0)
    assert _csv_minute_series(leg)[1:] == (0, 3)


def test_ac2_3_los_trades_con_fecha_nat_no_se_sintetizan():
    leg = _leg5()
    leg["s"].open_t = leg["s"].open_t.copy()
    leg["s"].open_t[2] = np.datetime64("NaT")
    series, n_bounded, n_zero = _csv_minute_series(leg)
    assert 2 not in series and (n_bounded, n_zero) == (2, 0)


def test_ac2_3_sin_trades_sin_m1_devuelve_las_originales_en_un_dict_nuevo():
    rows = _ROWS[:1]
    leg = _csv_leg(rows, series={0: [(pd.Timestamp("2024.03.04 09:00"), -0.1)]})
    series, n_bounded, n_zero = _csv_minute_series(leg)
    mv = leg["s"].minute_valle
    assert (n_bounded, n_zero) == (0, 0) and set(series) == set(mv) == {0}
    assert series is not mv and series[0] is mv[0]


# ===========================================================================
# Propiedad: patas CSV aleatorias (parte sin M1, con serie sintética) == referencia por fuerza bruta minuto a minuto
# ===========================================================================
def _patas_aleatorias(rng):
    """Dos patas CSV (pv 100) con 3-6 trades cada una, solapados entre sí, ~la mitad sin cobertura M1."""
    legs = []
    for p in range(2):
        rows = []
        for _ in range(int(rng.integers(3, 7))):
            ot = (D0 + pd.Timedelta(minutes=int(rng.integers(0, 4 * 24 * 60)))).floor("min")
            ct = ot + pd.Timedelta(minutes=int(rng.integers(30, 2 * 24 * 60)))
            size = float(rng.choice([0.1, 0.25, 0.5, 1.0]))
            rows.append(("Buy" if rng.random() < 0.6 else "Sell", ot, ct, size,
                         float(np.round(rng.normal(0.0, 60.0), 2)), float(-np.round(rng.uniform(5.0, 150.0), 2))))
        rows.sort(key=lambda r: r[2])                                  # orden de cierre = índice local de la pata
        series = {}
        for i, (_t, ot, ct, _s, _p, _m) in enumerate(rows):
            con_m1 = (p == 1) if i == 0 else rng.random() < 0.5        # el 1.º trade: sin M1 en la pata 0, con M1 en la 1
            if con_m1:                                                 # con M1: 1-5 puntos, el primero en la apertura
                span = int((ct - ot).total_seconds() // 60)
                offs = np.unique(np.r_[0, rng.integers(0, span, size=int(rng.integers(0, 5)))])
                series[i] = [(ot + pd.Timedelta(minutes=int(o)), float(rng.normal(-0.5, 0.9))) for o in offs]
        fmt = [(t, f"{o:%Y.%m.%d %H:%M}", f"{c:%Y.%m.%d %H:%M}", s, p_, m) for t, o, c, s, p_, m in rows]
        legs.append(_csv_leg(fmt, series=series, pv=100.0, name=f"p{p}.csv"))
    return legs


def _referencia_fb(blocks, lots_by, pv_by, minute_by_sym):
    """Por bloque, el mínimo diario de ``realizado_≤t + Σ flotante de las vivas en [ot, ct)`` por fuerza bruta (el realizado
    de un trade entra EN su ``ct``; el flotante es el último punto de su serie con ``ts ≤ t``, 0 antes del primero)."""
    out = []
    for b in blocks:
        tr = [(int(b.sym_id[j]), int(b.trade_local_id[j]), pd.Timestamp(b.open_t[j]), pd.Timestamp(b.close_t[j]),
               float(b.pnl_per_lot[j])) for j in range(b.n_trades)]
        mom = {t[2] for t in tr} | {t[3] for t in tr}
        for s, li, _o, _c, _p in tr:
            mom |= {pd.Timestamp(x) for x in minute_by_sym[s][li][0]}
        por_dia: dict = {}
        for t in sorted(mom):
            real = sum(lots_by[s][0, li] * ppl for s, li, ot, ct, ppl in tr if ct <= t)
            flot = 0.0
            for s, li, ot, ct, _p in tr:
                if ot <= t < ct:
                    ts, vals = minute_by_sym[s][li][:2]
                    prev = [v for x, v in zip(ts, vals) if pd.Timestamp(x) <= t]
                    flot += lots_by[s][0, li] * pv_by[s] * (prev[-1] if prev else 0.0)
            por_dia[t.date()] = min(por_dia.get(t.date(), np.inf), real + flot)
        out.append(por_dia)
    return out


@pytest.mark.parametrize("semilla", range(30))
def test_patas_csv_aleatorias_con_series_sinteticas_coinciden_con_la_referencia_por_fuerza_bruta(semilla):
    rng = np.random.default_rng(2000 + semilla)
    legs = _patas_aleatorias(rng)
    strats = {l["name"]: l["s"] for l in legs}
    for l in legs:                                                     # lotes distintos por trade (no solo 1)
        l["s"].lots = rng.choice([0.5, 1.0, 1.5, 2.0], size=(1, l["s"].n_trades))
    blocks = build_day_blocks_portfolio(strats)
    legs_ord = _legs_in_strats_order(legs, strats)
    minute_by_sym = {i: _csv_minute_series(l)[0] for i, l in enumerate(legs_ord)}
    lots_by = {i: s.lots for i, s in enumerate(strats.values())}
    pv_by = {i: s.pv for i, s in enumerate(strats.values())}
    got = precompute_day_arrays_portfolio(blocks, lots_by, pv_by, 1, float_mode="minute",
                                          minute_by_sym=minute_by_sym)[0]
    ref = _referencia_fb(blocks, lots_by, pv_by, minute_by_sym)
    for bda, r in zip(got, ref):
        por_dia = {pd.Timestamp.fromordinal(int(o)).date(): v for o, v in zip(bda.day_ord, bda.rel_float)}
        assert set(r) <= set(por_dia)                                  # el motor añade además los días de relleno
        for d, v in r.items():
            assert por_dia[d] == pytest.approx(v, rel=1e-9, abs=1e-9), (semilla, d)
    assert len(blocks) >= 1 and sum(int(_csv_mae_masks(l)[0].sum()) for l in legs) >= 1   # hubo series sintéticas


# ===========================================================================
# AC2.1 (variante) — la pata B sin M1 se acota con su MAE y reproduce el golden E4
# ===========================================================================
def test_ac2_1_variante_con_la_pata_b_sin_m1_acotada_por_su_mae():
    """Pata A como en E4 (serie propia, pv 1, a 1 lote). Pata B sin cobertura OHLC, MAE del CSV -300 $ a su tamaño de
    1 lote: serie sintética -300 constante desde las 09:00 hasta las 16:00 (exclusive). Minute: 09:30 -200-300 = -500;
    10:00 (realizado A -400) -400-300 = -700; 16:00 realizado -450 → -700 (= el golden E4). El joint de antes ve solo
    a A (B no tiene entrada → 0): -400."""
    a = _csv_leg([("Buy", "2024.03.04 09:00", "2024.03.04 10:00", 1.0, -400.0, -400.0)], pv=1.0, name="A.csv",
                 series={0: [(at("09:00"), 0.0), (at("09:30"), -200.0), (at("10:00"), -400.0)]})
    bb = _csv_leg([("Buy", "2024.03.04 09:00", "2024.03.04 16:00", 1.0, -50.0, -300.0)], pv=1.0, name="B.csv")
    sa, sb = a["s"], bb["s"]
    blocks = build_day_blocks_portfolio({"A": sa, "B": sb})
    assert len(blocks) == 1 and blocks[0].n_no_valley == 1
    kw = dict(lots_by_sym={0: sa.lots, 1: sb.lots}, pv_by_sym={0: 1.0, 1: 1.0}, n_f=1)
    m = {0: _csv_minute_series(a)[0], 1: _csv_minute_series(bb)[0]}
    assert m[1][0][1][0] == pytest.approx(-300.0)
    minute = precompute_day_arrays_portfolio(blocks, float_mode="minute", minute_by_sym=m, **kw)[0][0]
    joint = precompute_day_arrays_portfolio(
        blocks, float_mode="joint", minute_by_sym={0: sa.minute_valle, 1: sb.minute_valle}, **kw)[0][0]
    assert minute.rel_float[0] == pytest.approx(-700.0) and minute.rel_closed[0] == pytest.approx(-450.0)
    assert joint.rel_float[0] == pytest.approx(-400.0)


# ===========================================================================
# E9, helpers de orden y de flags
# ===========================================================================
def test_e9_los_helpers_de_flotante_con_mae_rechazan_una_pata_sqx():
    leg = {"kind": KIND_SQX, "tl": None, "s": SimpleNamespace(n_trades=0)}
    for fn in (_csv_minute_series, _csv_mae_masks):
        with pytest.raises(AssertionError, match="solo sirve a patas CSV"):
            fn(leg)
    assert "kind" not in _leg5()                                       # las CSV no llevan `kind` y pasan la guarda


def test_las_longitudes_incoherentes_de_la_pata_fallan_en_voz_alta():
    leg = _leg5()
    leg["s"].n_trades = 4
    with pytest.raises(ValueError, match="trades en la estrategia"):
        _csv_minute_series(leg)


def test_legs_in_strats_order_sigue_el_orden_de_strats_incluso_con_nombres_repetidos():
    a, b, c = {"name": "a.csv"}, {"name": "b.csv"}, {"name": "a.csv", "otra": True}
    assert _legs_in_strats_order([a, b], {"a.csv": 1, "b.csv": 2}) == [a, b]
    assert _legs_in_strats_order([b, a], {"b.csv": 2, "a.csv": 1}) == [b, a]
    # nombre repetido: `strats` colapsa en la clave (gana la última pata) y el índice de pata sigue su orden
    assert _legs_in_strats_order([a, b, c], {"a.csv": 3, "b.csv": 2}) == [c, b]


def test_check_float_flag_acepta_minute_y_joint_y_rechaza_lo_demas():
    assert _check_float_flag("X", "minute") == "minute" and _check_float_flag("X", "joint") == "joint"
    for malo in ("minuto", "Minute", "bound", "", None):
        with pytest.raises(ValueError, match="'minute' o 'joint'"):
            _check_float_flag("CSV_FLOAT_MODE", malo)


# ===========================================================================
# Contadores y avisos de los trades sin M1 en los bloques (_mae_usage / _mae_warnings)
# ===========================================================================
def test_mae_usage_cuenta_en_los_bloques_con_peso_pnl():
    """Bloques reales de la lente prop firm (``build_day_blocks_portfolio``): trades 0, 1, 2 y 4 (el 3 no forma bloque).
    Sin M1: 1 y 2, con MAE → acotados. Pesos |lotes · PnL por lote| a 1 lote: 300 / 240 / 240 / 120 → acotados
    480/900."""
    leg = _leg5()
    s = leg["s"]
    blocks = build_day_blocks_portfolio({"leg": s})
    u = _mae_usage(blocks, [leg], {0: s.lots}, 0)
    assert (u["n_mae_bounded"], u["n_mae_zero"]) == (2, 0)
    assert u["pnl_w_mae_bounded"] == pytest.approx(480.0 / 900.0) and u["pnl_w_mae_zero"] == 0.0
    assert [w for w in _mae_warnings(u)] == [
        f"2 trades sin OHLC M1: su flotante se acota con el MAE del CSV desde la apertura "
        f"(peso PnL f central {480.0 / 900.0:.1%})."]


def test_mae_usage_mezcla_acotados_y_ceros_con_sus_dos_avisos():
    rows = list(_ROWS)
    rows[2] = (*rows[2][:5], None)                                     # el trade 2 sin MAE → cero
    leg = _csv_leg(rows, series={i: [(pd.Timestamp(t), v) for t, v in pts] for i, pts in _SERIES.items()})
    s = leg["s"]
    blocks = build_day_blocks_portfolio({"leg": s})
    u = _mae_usage(blocks, [leg], {0: s.lots}, 0)
    assert (u["n_mae_bounded"], u["n_mae_zero"]) == (1, 1)
    assert u["pnl_w_mae_bounded"] == pytest.approx(240.0 / 900.0) and u["pnl_w_mae_zero"] == pytest.approx(240.0 / 900.0)
    w = _mae_warnings(u)
    assert len(w) == 2
    assert w[0].startswith("1 trades sin OHLC M1: su flotante se acota con el MAE del CSV desde la apertura (peso PnL")
    assert w[1] == "1 trades sin MAE en el CSV: su flotante intradía cuenta como 0 (optimista)."


def test_mae_usage_sin_trades_sin_m1_no_avisa_y_sin_peso_da_nan():
    leg = _csv_leg(_ROWS[:1], series={0: [(pd.Timestamp("2024.03.04 09:00"), -0.1)]})
    blocks = build_day_blocks_portfolio({"leg": leg["s"]})
    u = _mae_usage(blocks, [leg], {0: leg["s"].lots}, 0)
    assert (u["n_mae_bounded"], u["n_mae_zero"]) == (0, 0) and _mae_warnings(u) == []
    assert _mae_usage([], [leg], {0: leg["s"].lots}, 0)["pnl_w_mae_bounded"] != 0.0  # NaN: no hay peso


def test_csv_minute_meta_solo_con_uso_de_mae():
    assert _csv_minute_meta({"csv_minute": None}) == {} and _csv_minute_meta({}) == {}
    usage = {"n_mae_bounded": 1, "n_mae_zero": 0, "pnl_w_mae_bounded": 0.1, "pnl_w_mae_zero": 0.0}
    out = _csv_minute_meta({"csv_minute": usage})
    assert out["float_note"] == main_mod.CSV_MINUTE_FLOAT_NOTE and {k: out[k] for k in usage} == usage


# ===========================================================================
# AC2.9 (E11) — fin de semana con una posición abierta
# ===========================================================================
PHASE = PropFirmPhase(name="Challenge", profit_target_pct=1000.0, max_daily_loss_pct=5.0, max_loss_pct=10.0,
                      min_trading_days=1)


def _weekend():
    """X abre el viernes 10:00 y cierra el lunes 10:00 (+20); Y abre y cierra el viernes (-30), pv 1."""
    b = block([(0, 0, at("10:00", FRI), at("10:00", MON), 20.0), (1, 0, at("11:00", FRI), at("15:00", FRI), -30.0)])
    m = mbs(
        (0, 0, serie(at("10:00", FRI), at("10:00", MON),
                     [(at("10:00", FRI), -10.0), (at("16:00", FRI), -40.0), (at("09:00", MON), -5.0)])),
        (1, 0, serie(at("11:00", FRI), at("15:00", FRI),
                     [(at("11:00", FRI), -5.0), (at("14:00", FRI), -35.0), (at("15:00", FRI), -30.0)])))
    return b, m


def test_ac2_9_fin_de_semana_con_posicion_abierta_trading_days_igual_en_joint_y_en_minute():
    b, m = _weekend()
    joint = run(b, {0: [1.0], 1: [1.0]}, m, mode="joint")
    minute = run(b, {0: [1.0], 1: [1.0]}, m, mode="minute")
    # minute añade las filas del sábado y del domingo (FTMO recalcula el suelo cada medianoche); joint no
    assert list(joint.day_ord) == [FRI.toordinal(), MON.toordinal()]
    assert list(minute.day_ord) == [FRI.toordinal(), FRI.toordinal() + 1, FRI.toordinal() + 2, MON.toordinal()]
    np.testing.assert_array_equal(minute.opens_day, [2, 0, 0, 0])
    durations = np.array([b.duration_days])
    res_j = evaluate_phase_path(np.array([0]), [joint], durations, account_size=7000.0, phase=PHASE)
    res_m = evaluate_phase_path(np.array([0]), [minute], durations, account_size=7000.0, phase=PHASE)
    assert res_j.outcome == res_m.outcome == "neither"
    assert res_j.trading_days == res_m.trading_days == 1               # las filas de fin de semana no abren nada
    assert res_j.days == res_m.days == pytest.approx(4.0)              # los días transcurridos son diferencias de day_ord


# ===========================================================================
# Cartera sintética de extremo a extremo (portfolio_load + los endpoints)
# ===========================================================================
_STEP = 0.01
_CSV_CAPITAL = 50_000.0
_ACCOUNT = 100_000.0
_F_REF = 0.01


def _synth_files(tmp_path, *, n: int = 160, gap_every: int = 10, with_mae: bool = True):
    """CSV orderstocsv + OHLC M1 de XAUUSD_axi: un trade por día hábil (09:00-11:00), backtest que COMPONE, 60 % de
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
        mae = f"{-abs(ppl) * sz * 0.5:.2f};" if with_mae else ""
        rows.append(f"{i + 1};XAUUSD_axi;Buy;{ot:%Y.%m.%d %H:%M};2000.00;{sz:.2f};{ct:%Y.%m.%d %H:%M};{exit_px:.2f};"
                    f"{p:.2f};{e:.2f};IST;Exit After 1 Bars;{mae}{abs(ppl) * sz:.2f};1;-\n")
        has_m1 = not (gap_every and i % gap_every == gap_every - 1)
        covered.append(has_m1)
        if has_m1:
            idx = pd.date_range(ot, ct, freq="1min")
            path = 2000.0 + (exit_px - 2000.0) * np.linspace(0.0, 1.0, len(idx))
            frames.append(pd.DataFrame({"o": path, "h": path + 0.3, "l": path - 0.3 - (0.0 if win else 1.0),
                                        "c": path}, index=idx))
    csv = tmp_path / ("d2_synth_leg.csv" if with_mae else "d2_synth_leg_sin_mae.csv")
    csv.write_text((_HEADER if with_mae else _HEADER_SIN_MAE) + "".join(rows))
    return SimpleNamespace(csv=csv, ohlc=pd.concat(frames), covered=np.array(covered))


@pytest.fixture
def load_portfolio(monkeypatch):
    """Fábrica ``load(files, ohlc, account_size)`` → respuesta de ``portfolio_load``. Limpia ``_portfolios`` y
    ``_leg_cache`` al terminar."""
    created: list[str] = []
    before = set(_leg_cache)

    def load(files, ohlc, account_size: float = _ACCOUNT):
        monkeypatch.setattr(main_mod, "_load_ohlc", lambda sym: ohlc)
        res = main_mod.portfolio_load(main_mod.PortfolioLoadRequest(
            legs=[str(p) for p in files], account_size=account_size, f_ref=_F_REF))
        created.append(res["portfolio_id"])
        return res

    yield load
    for pid in created:
        main_mod._portfolios.pop(pid, None)
    for k in set(_leg_cache) - before:
        _leg_cache.pop(k, None)


@pytest.fixture
def synth(tmp_path, load_portfolio):
    s = _synth_files(tmp_path)
    res = load_portfolio([s.csv], s.ohlc)
    assert res["ok"], res
    s.pid, s.load = res["portfolio_id"], res
    return s


def _req(pid, firm="ftmo_100k", **kw):
    base = dict(portfolio_id=pid, firm_key=firm, f_lo=0.0025, f_hi=0.005, f_step=0.0025, n_paths=100,
                n_challenges=5, n_sequences=100, capital0_eur=3000.0, fx_eur_usd=1.25)
    base.update(kw)
    return main_mod.PropfirmSimRequest(**base)


def _stable(out) -> str:
    """Salida sin lo que varía entre corridas (tiempos), serializada para comparar bit a bit (NaN incluido)."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "elapsed_s"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(out), sort_keys=True, default=str, ensure_ascii=False)


def _numeros(out, *claves) -> str:
    """Huella (sha256, 16 hex) de las partes NUMÉRICAS de una respuesta (las ``claves`` de primer nivel), con los
    floats redondeados a 6 decimales (estable entre plataformas). Sirve para fijar cifras de un escenario sintético."""
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


def test_ac2_6_meta_1d_float_mode_minute_nota_contadores_y_aviso(synth):
    out = main_mod.simulate_propfirm(_req(synth.pid))
    meta = out["meta"]
    assert meta["float_mode"] == "minute"
    assert "realizado" in meta["float_note"] and "realizado intradía" in meta["float_note"]
    # E10: lo dice el float_note (la envolvente muere en el minuto de cierre)
    assert "minuto de cierre" in meta["float_note"] and "barra de cierre" in meta["float_note"]
    assert meta["float_note"] == main_mod.CSV_MINUTE_FLOAT_NOTE
    # 16 de los 160 trades no tienen M1; todos con MAE → acotados; ninguno a 0
    assert (meta["n_no_valley"], meta["n_mae_bounded"], meta["n_mae_zero"]) == (16, 16, 0)
    assert meta["n_no_valley"] == meta["n_mae_bounded"] + meta["n_mae_zero"]
    assert meta["pnl_w_no_valley"] == pytest.approx(meta["pnl_w_mae_bounded"] + meta["pnl_w_mae_zero"])
    esperado = (f"16 trades sin OHLC M1: su flotante se acota con el MAE del CSV desde la apertura "
                f"(peso PnL f central {meta['pnl_w_mae_bounded']:.1%}).")
    assert esperado in out["warnings"]
    assert not [w for w in out["warnings"] if "flotante intradía cuenta como 0" in w or "sin valle OHLC" in w]
    assert not [w for w in out["warnings"] if "sin MAE" in w]
    assert meta["sizing_base"] == "tamaño de cuenta (sin componer)"


def test_ac2_6_el_peso_pnl_de_la_meta_es_el_del_f_central(synth):
    """Cálculo independiente (sin ``_mae_usage``): Σ|lotes · PnL por lote| de los trades sin valle de los bloques entre el
    total, con los lotes de la fila CENTRAL de la grilla (``n_f // 2``), la misma que usa ``pnl_w_no_valley``."""
    req = _req(synth.pid, f_lo=0.0025, f_hi=0.0075, f_step=0.0025)        # 3 puntos: el central es el índice 1
    ctx = main_mod._propfirm_setup(req)

    def peso(fi):
        sin_valle = total = 0.0
        for b in ctx["blocks"]:
            for j in range(b.n_trades):
                w = abs(ctx["lots_by"][int(b.sym_id[j])][fi, int(b.trade_local_id[j])] * b.pnl_per_lot[j])
                total += w
                if not b.valle_precio[j]:
                    sin_valle += w
        return sin_valle / total

    assert len(ctx["f_grid"]) == 3
    assert ctx["csv_minute"]["pnl_w_mae_bounded"] == pytest.approx(peso(1), rel=1e-12)
    assert abs(peso(0) - peso(1)) > 1e-6 and abs(peso(2) - peso(1)) > 1e-6     # el test distingue las filas
    assert ctx["pnl_w_nv"] == pytest.approx(peso(1), rel=1e-12)


def test_ac2_7_el_par_2d_corre_y_usa_minute(synth):
    out = main_mod.simulate_propfirm_pair(_req(synth.pid))
    meta = out["meta"]
    assert meta["float_mode"] == "minute" and "realizado" in meta["float_note"]
    assert (meta["n_mae_bounded"], meta["n_mae_zero"]) == (16, 0)
    assert out["cells"] and meta["cells_resolved"] == len(out["cells"])
    assert any("trades sin OHLC M1" in w for w in out["warnings"])


def test_d2_csv_sin_columna_mae_los_trades_sin_m1_cuentan_como_cero_con_su_aviso(tmp_path, load_portfolio):
    s = _synth_files(tmp_path, with_mae=False)
    res = load_portfolio([s.csv], s.ohlc)
    assert res["ok"], res
    out = main_mod.simulate_propfirm(_req(res["portfolio_id"]))
    meta = out["meta"]
    assert meta["float_mode"] == "minute"
    assert (meta["n_no_valley"], meta["n_mae_bounded"], meta["n_mae_zero"]) == (16, 0, 16)
    assert "16 trades sin MAE en el CSV: su flotante intradía cuenta como 0 (optimista)." in out["warnings"]
    assert not [w for w in out["warnings"] if "se acota con el MAE" in w]
    assert meta["pnl_w_mae_zero"] == pytest.approx(meta["pnl_w_no_valley"]) and meta["pnl_w_mae_bounded"] == 0.0


def test_e3_predicado_minute_si_y_solo_si_el_perfil_es_estatico(synth):
    """Perfiles estáticos (cycle, sin trailing ni et_session) → minute; Apex y los perfiles de futuros → joint +
    trailing (ya incluyen el realizado), sin contadores."""
    for firm in ("ftmo_50k", "ftmo_50k_swing", "ftmo_100k"):
        ctx = main_mod._propfirm_setup(_req(synth.pid, firm))
        assert not main_mod._profile_extended(ctx["cfg"]) and ctx["float_mode"] == "minute", firm
        assert ctx["csv_minute"] is not None and ctx["trailing"] is False
        assert all(b.rel_low is None for ba in ctx["day_arrays_grid"] for b in ba)
    for firm in ("apex_150k_intraday", "ftmo_futures_growth_50k", "ftmo_futures_pro_100k"):
        ctx = main_mod._propfirm_setup(_req(synth.pid, firm))
        assert main_mod._profile_extended(ctx["cfg"]) and ctx["float_mode"] == "joint", firm
        assert ctx["csv_minute"] is None and ctx["trailing"] is True
        assert all(b.rel_low is not None for ba in ctx["day_arrays_grid"] for b in ba)


def test_ac2_4_apex_bit_a_bit_no_depende_de_csv_float_mode_ni_de_la_fase_2(synth, monkeypatch):
    """Perfil con trailing (``apex_150k_intraday``) con patas CSV: el resultado completo es idéntico con cualquier
    ``CSV_FLOAT_MODE`` y los day-arrays son los de la receta de antes (``float_mode="joint"`` + ``trailing=True`` con el
    ``minute_valle`` original y las envolventes favorables), campo a campo."""
    outs = {}
    for mode in ("minute", "joint"):
        monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", mode)
        outs[mode] = _stable(main_mod.simulate_propfirm(_req(synth.pid, "apex_150k_intraday")))
    assert outs["minute"] == outs["joint"]
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "minute")
    ctx = main_mod._propfirm_setup(_req(synth.pid, "apex_150k_intraday"))
    assert ctx["float_mode"] == "joint" and ctx["csv_minute"] is None
    labeler = make_session_day_labeler(ctx["source_tz"], cut_hhmm=ctx["cfg"].day_axis_cut_hhmm)
    strats = list(ctx["strats"].values())
    ref = precompute_day_arrays_portfolio(
        ctx["blocks"], ctx["lots_by"], {i: s.pv for i, s in enumerate(strats)}, len(ctx["f_grid"]),
        float_mode=main_mod.FLOAT_MODE, minute_by_sym={i: s.minute_valle for i, s in enumerate(strats)},
        trailing=True, minute_fav_by_sym={i: s.minute_fav for i, s in enumerate(strats)}, day_labeler=labeler)
    assert len(ref) == len(ctx["day_arrays_grid"])
    for ref_f, got_f in zip(ref, ctx["day_arrays_grid"]):
        for r, g in zip(ref_f, got_f):
            for fld in dataclasses.fields(BlockDayArrays):
                a, b = getattr(r, fld.name), getattr(g, fld.name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b, err_msg=fld.name)
                else:
                    assert a == b, fld.name


def test_flag_csv_joint_reproduce_el_camino_anterior(synth, monkeypatch):
    """Con ``CSV_FLOAT_MODE = "joint"``: ``float_mode`` joint, day-arrays de la receta de antes, aviso de antes y
    ninguna clave nueva en ``meta``."""
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "joint")
    req = _req(synth.pid)
    ctx = main_mod._propfirm_setup(req)
    assert ctx["float_mode"] == "joint" and ctx["csv_minute"] is None
    strats = list(ctx["strats"].values())
    ref = precompute_day_arrays_portfolio(
        ctx["blocks"], ctx["lots_by"], {i: s.pv for i, s in enumerate(strats)}, len(ctx["f_grid"]),
        float_mode="joint", minute_by_sym={i: s.minute_valle for i, s in enumerate(strats)}, trailing=False)
    for ref_f, got_f in zip(ref, ctx["day_arrays_grid"]):
        for r, g in zip(ref_f, got_f):
            for fld in dataclasses.fields(BlockDayArrays):
                a, b = getattr(r, fld.name), getattr(g, fld.name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b, err_msg=fld.name)
                else:
                    assert a == b, fld.name
    w = [x for x in ctx["warnings"] if "valle" in x]
    assert len(w) == 1 and re.fullmatch(
        r"16 trades sin valle OHLC \(peso PnL f central \d+\.\d%\): flotante intradía cuenta como 0\.", w[0])
    meta = main_mod.simulate_propfirm(req)["meta"]
    assert meta["float_mode"] == "joint"
    assert not {"float_note", "n_mae_bounded", "n_mae_zero", "pnl_w_mae_bounded", "pnl_w_mae_zero"} & set(meta)


# Cifras (huella de reports, recomendación, f*, trayectoria, celdas…) del escenario ``synth`` en el árbol de ANTES de la fase 2
# (base e0cbf9a con la fase 1 aplicada), obtenidas con el mismo código de este fichero ejecutado en ese árbol. Fijan que
# `CSV_FLOAT_MODE = "joint"` y el perfil Apex devuelven exactamente las cifras de entonces.
_PIN_ANTES_DE_LA_FASE_2 = {"ftmo_1d": "951eb17f6ad53242", "ftmo_par": "79b3788a9b3c13e3",
                           "apex_1d": "317d2fb23dd1439b"}
_CLAVES_1D = ("reports", "recommendation", "f_star", "trajectory", "intentos_garantizados")


def test_flag_csv_joint_las_cifras_son_las_de_antes_de_la_fase_2(synth, monkeypatch):
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "joint")
    # bloque-politis-white (E7, opción 1): las cifras fijadas son las de la corrida con el bloque que daba la regla vieja en
    # este escenario (10,0); se fija para conservarlas bit a bit (la sensibilidad con bloque 1 no toca estas claves, E19).
    assert _numeros(main_mod.simulate_propfirm(_req(synth.pid, block_size=10.0)), *_CLAVES_1D) \
        == _PIN_ANTES_DE_LA_FASE_2["ftmo_1d"]
    assert _numeros(main_mod.simulate_propfirm_pair(_req(synth.pid, block_size=10.0)), "cells", "recommendation",
                    "cap_sensitivity") == _PIN_ANTES_DE_LA_FASE_2["ftmo_par"]


@pytest.mark.parametrize("modo", ["minute", "joint"])
def test_ac2_4_apex_las_cifras_son_las_de_antes_de_la_fase_2(synth, monkeypatch, modo):
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", modo)
    out = main_mod.simulate_propfirm(_req(synth.pid, "apex_150k_intraday"))
    assert _numeros(out, *_CLAVES_1D) == _PIN_ANTES_DE_LA_FASE_2["apex_1d"]


def test_flag_csv_invalido_falla_en_voz_alta(synth, monkeypatch):
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "minuto")
    with pytest.raises(ValueError, match="CSV_FLOAT_MODE"):
        main_mod._propfirm_setup(_req(synth.pid))


# ---------------------------------------------------------------------------
# E16 — aviso de capital de la carga
# ---------------------------------------------------------------------------
def test_e16_el_aviso_de_capital_de_la_carga_es_informativo(synth):
    legs = synth.load["legs"]
    assert legs[0]["warnings"][0] == (
        "capital del backtest $50,000 ≠ tamaño de cuenta $100,000: informativo, la app dimensiona cada orden sobre el "
        "tamaño de cuenta.")
    texto = " ".join(legs[0]["warnings"] + synth.load["warnings"])
    assert "no es fiable" not in texto and "escalados a otra equity" not in texto
    assert f"{synth.csv.name}: capital del backtest $50,000" in " ".join(synth.load["warnings"])
    assert synth.load["guards"]["capital"]["status"] == "warn"         # la guarda sigue recogiendo el aviso
    # la función del motor NO se ha tocado: sigue diciendo lo de antes (otras herramientas siguen componiendo)
    tl = load_trade_list(synth.csv.read_bytes())
    assert "no es fiable" in initial_capital_mismatch(tl, _ACCOUNT)


@pytest.mark.parametrize("account, avisa", [(50_000.0, False), (52_000.0, False), (53_000.0, True)])
def test_e16_la_tolerancia_del_5_por_ciento_es_la_del_motor(tmp_path, load_portfolio, account, avisa):
    s = _synth_files(tmp_path, n=60)
    res = load_portfolio([s.csv], s.ohlc, account_size=account)
    textos = [w for w in res["legs"][0]["warnings"] if "capital" in w]
    assert bool(textos) is avisa
    assert (res["guards"]["capital"]["status"] == "warn") is avisa


def test_la_guarda_valles_de_la_carga_dice_que_se_acota_con_el_mae(synth):
    d = synth.load["guards"]["valles"]
    assert d["status"] == "warn" and re.fullmatch(
        # Texto ajustado por el orquestador tras la revisión de código de la fase 2 (C4, C7; 2026-10-19): dice también qué
        # pasa con trailing y en la lente AXI, y avisaría de las patas sin columna MAE (aquí no hay ninguna).
        r"16 trades sin valle OHLC \(peso PnL a f_ref \d+\.\d%\): en retos estáticos su flotante se acota con su MAE "
        r"desde la apertura; con trailing \(Apex o Futures\) cuenta como 0; en la lente AXI esos trades quedan fuera "
        r"de la simulación\.", d["detail"])


# ===========================================================================
# Extra — D2 de extremo a extremo con dos patas: una Venta que pierde por la mañana y una Compra con valle por la tarde
# ===========================================================================
# Precio de XAUUSD (100 $ por punto y lote), igual cada día, en barras de 1 min de 09:00 a 16:00:
#   09:00 → 10:00: 2000 → 2010 · 10:00 → 13:00: 2010 → 2000 · 13:00 → 14:00: 2000 → 1990 · 14:00 → 16:00: 1990 → 2000.
# Pata A: Venta 09:00 → 10:00 a 2000 → 2010: pierde 10 puntos (-1.000 $/lote). Pata B: Compra 09:00 → 16:00 a 2000 → 2000:
# realizado 0, pero a las 14:00 va -10 puntos (-1.000 $/lote). A y B son un único bloque cada día.
_KNOT_MIN = [0, 60, 240, 300, 420]
_KNOT_PX = [2000.0, 2010.0, 2000.0, 1990.0, 2000.0]


def _two_leg_files(tmp_path, *, n: int = 40, uncovered_every: int = 4, a_mae: float = -3000.0):
    days = pd.bdate_range("2023-01-02", periods=n)
    frames, rows_a, rows_b, bal = [], [], [], 100_000.0
    for i, d in enumerate(days):
        t0 = d + pd.Timedelta(hours=9)
        idx = pd.date_range(t0, t0 + pd.Timedelta(hours=7), freq="1min")
        px = np.interp((idx - t0).total_seconds() / 60.0, _KNOT_MIN, _KNOT_PX)
        df = pd.DataFrame({"o": px, "h": px, "l": px, "c": px}, index=idx)
        if uncovered_every and i % uncovered_every == uncovered_every - 1:
            df = df[(df.index < t0) | (df.index > t0 + pd.Timedelta(hours=1))]     # sin barras 09:00-10:00 (inclusive)
        frames.append(df)
        bal = round(bal - 1000.0, 2)
        rows_a.append(f"{i + 1};XAUUSD_axi;Sell;{t0:%Y.%m.%d %H:%M};2000.00;1.00;{t0 + pd.Timedelta(hours=1):%Y.%m.%d %H:%M};"
                      f"2010.00;-1000.00;{bal:.2f};IST;Exit After 1 Bars;{a_mae:.2f};0;1;-\n")
        rows_b.append(f"{i + 1};XAUUSD_axi;Buy;{t0:%Y.%m.%d %H:%M};2000.00;1.00;{t0 + pd.Timedelta(hours=7):%Y.%m.%d %H:%M};"
                      f"2000.00;0.00;100000.00;IST;Exit After 1 Bars;-1000.00;0;1;-\n")
    a, b = tmp_path / "d2_leg_a.csv", tmp_path / "d2_leg_b.csv"
    a.write_text(_HEADER + "".join(rows_a))
    b.write_text(_HEADER + "".join(rows_b))
    return SimpleNamespace(csvs=[a, b], ohlc=pd.concat(frames), n=n, uncovered_every=uncovered_every)


@pytest.fixture
def two_leg(tmp_path, load_portfolio):
    s = _two_leg_files(tmp_path)
    res = load_portfolio(s.csvs, s.ohlc)
    assert res["ok"], res
    assert [x["status"] for x in res["legs"]] == ["ok", "ok"]
    s.pid, s.load = res["portfolio_id"], res
    return s


def test_d2_de_extremo_a_extremo_el_realizado_de_una_pata_se_suma_al_flotante_de_la_otra(two_leg, monkeypatch):
    """Día 0 (ambas patas con M1): a 14:00, A ya cerró (-1.000 · lotes_A realizados) y B va a -1.000 · lotes_B →
    el peor momento del día es -1.000 · (lotes_A + lotes_B) en minute; el joint de antes (solo flotantes de las posiciones
    vivas) lo deja en -1.000 · lotes_B. Con lotes iguales, el doble."""
    req = _req(two_leg.pid, f_lo=0.01, f_hi=0.02, f_step=0.01, n_paths=100)
    ctx = main_mod._propfirm_setup(req)
    assert ctx["float_mode"] == "minute" and list(ctx["strats"]) == ["d2_leg_a.csv", "d2_leg_b.csv"]
    assert ctx["n_blocks"] == two_leg.n
    for fi in range(2):
        la, lb = ctx["lots_by"][0][fi, 0], ctx["lots_by"][1][fi, 0]
        assert la == lb == pytest.approx(1.0 * (fi + 1))               # f / f_ref · tamaño 1 · (cuenta / equity) = 1
        bda = ctx["day_arrays_grid"][fi][0]
        assert list(bda.day_ord) == [pd.Timestamp("2023-01-02").toordinal()]
        assert bda.rel_float[0] == pytest.approx(-1000.0 * (la + lb), rel=1e-9)
        assert bda.rel_closed[0] == pytest.approx(-1000.0 * la) and bda.rel_before[0] == 0.0
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "joint")
    ctxj = main_mod._propfirm_setup(req)
    assert ctxj["float_mode"] == "joint"
    for fi in range(2):
        lb = ctxj["lots_by"][1][fi, 0]
        assert ctxj["day_arrays_grid"][fi][0].rel_float[0] == pytest.approx(-1000.0 * lb, rel=1e-9)
        assert ctxj["day_arrays_grid"][fi][0].rel_float[0] > ctx["day_arrays_grid"][fi][0].rel_float[0]


def test_d2_de_extremo_a_extremo_la_pata_sin_m1_se_acota_con_su_mae_alineada_con_su_pata(two_leg, monkeypatch):
    """Día 3 (la pata A sin barras de 09:00 a 10:00): A queda con la serie sintética MAE / (size · pv) = -3.000 / (1 ·
    100) = -30 puntos, constante hasta las 10:00 → -3.000 · lotes_A. Es el mínimo del día (el realizado + el valle de B,
    -1.000 · (lotes_A + lotes_B), queda por encima). El joint de antes ignora a A ese día: -1.000 · lotes_B. Si el índice de
    pata se desalineara con los lotes, estos valores no saldrían."""
    req = _req(two_leg.pid, f_lo=0.01, f_hi=0.02, f_step=0.01, n_paths=100)
    ctx = main_mod._propfirm_setup(req)
    k = 3
    for fi in range(2):
        la, lb = ctx["lots_by"][0][fi, k], ctx["lots_by"][1][fi, k]
        assert la > 1.0 * (fi + 1) and lb == pytest.approx(1.0 * (fi + 1))   # A pierde cada día: su equity baja
        bda = ctx["day_arrays_grid"][fi][k]
        assert bda.rel_float[0] == pytest.approx(-3000.0 * la, rel=1e-9)
        assert bda.rel_float[0] < -1000.0 * (la + lb)                   # el MAE manda sobre «realizado + valle de B»
    assert (ctx["csv_minute"]["n_mae_bounded"], ctx["csv_minute"]["n_mae_zero"]) == (two_leg.n // 4, 0)
    assert ctx["n_no_valley"] == two_leg.n // 4
    monkeypatch.setattr(main_mod, "CSV_FLOAT_MODE", "joint")
    ctxj = main_mod._propfirm_setup(req)
    for fi in range(2):
        assert ctxj["day_arrays_grid"][fi][k].rel_float[0] == pytest.approx(-1000.0 * ctxj["lots_by"][1][fi, k],
                                                                            rel=1e-9)


def test_d2_con_el_eje_de_dias_ftmo_las_series_sinteticas_van_en_el_eje_pedido(two_leg):
    """Con ``day_axis="ftmo"`` (−1 h) las patas se recargan desplazadas: ``s.open_t``/``s.close_t`` (de los que salen las
    series sintéticas) y las series reales viven en ese eje. Todo el día queda dentro del mismo día de calendario, así que
    los mínimos son los de antes; si la serie sintética usara los tiempos del eje de servidor, la máscara [ot, ct)
    no coincidiría con las barras y no saldría el mismo valor."""
    kw = dict(f_lo=0.01, f_hi=0.02, f_step=0.01, n_paths=100)
    servidor = main_mod._propfirm_setup(_req(two_leg.pid, **kw))
    ftmo = main_mod._propfirm_setup(_req(two_leg.pid, day_axis="ftmo", **kw))
    assert ftmo["float_mode"] == "minute" and ftmo["legs_used"][0]["s"] is not servidor["legs_used"][0]["s"]
    assert pd.Timestamp(ftmo["legs_used"][0]["s"].open_t[0]) == pd.Timestamp("2023-01-02 08:00")   # eje desplazado
    for fi in range(2):
        for k in (0, 3):                       # día 0 (ambas con M1) y día 3 (la pata A sin M1: serie sintética)
            a, b = servidor["day_arrays_grid"][fi][k], ftmo["day_arrays_grid"][fi][k]
            np.testing.assert_allclose(b.rel_float, a.rel_float, rtol=1e-9)
            np.testing.assert_allclose(b.rel_closed, a.rel_closed, rtol=1e-9)
    la = ftmo["lots_by"][0][0, 3]                                       # lotes de A el día 3 a f = 1 %
    assert ftmo["day_arrays_grid"][0][3].rel_float[0] == pytest.approx(-3000.0 * la, rel=1e-9)
    assert ftmo["csv_minute"]["n_mae_bounded"] == two_leg.n // 4


def test_d2_los_contadores_son_los_de_la_ventana_oos(two_leg):
    """``n_mae_bounded`` cuenta los trades de los bloques de la ventana: con un corte OOS solo quedan los bloques
    posteriores (la pata A sin M1 los días 3, 7, 11…)."""
    dias = pd.bdate_range("2023-01-02", periods=two_leg.n)
    corte = dias[20]
    esperado = sum(1 for i in range(two_leg.n) if i % 4 == 3 and dias[i] >= corte)
    ctx = main_mod._propfirm_setup(_req(two_leg.pid, f_lo=0.01, f_hi=0.02, f_step=0.01, oos_start=str(corte.date())))
    assert ctx["n_blocks"] == two_leg.n - 20 and 0 < esperado < two_leg.n // 4
    assert ctx["csv_minute"]["n_mae_bounded"] == esperado == ctx["n_no_valley"]


def test_d2_de_extremo_a_extremo_el_endpoint_devuelve_los_contadores_de_la_cartera(two_leg):
    out = main_mod.simulate_propfirm(_req(two_leg.pid, f_lo=0.01, f_hi=0.02, f_step=0.01))
    meta = out["meta"]
    assert meta["float_mode"] == "minute"
    assert (meta["n_mae_bounded"], meta["n_mae_zero"]) == (two_leg.n // 4, 0)
    assert any(w.startswith(f"{two_leg.n // 4} trades sin OHLC M1: su flotante se acota con el MAE del CSV") for w in
               out["warnings"])
