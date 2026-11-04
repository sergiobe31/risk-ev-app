"""Fase 2 del plan ``cfd-sqx-minuto-par`` (2026-10-14): flotante por MINUTO (OHLC M1)
y barrido 2D para carteras de patas ``.sqx`` de CFD FTMO (criterios B1-B12 del plan).

Tests a nivel de FUNCIÓN (sin levantar FastAPI), como ``test_cfd_sqx_legs.py``:
``.sqx`` sintéticos escritos con el writer de ``tests/test_sqx_trades.py`` y un OHLC M1
sintético en formato MT5 sin cabecera (``yyyy.MM.dd,HH:mm,o,h,l,c,vol``, hora del
servidor) escrito a ``tmp_path`` y apuntado con ``main.SQX_OHLC_DIR``. El OHLC es
COHERENTE con los trades: contiene cada minuto entre apertura y cierre, ``open_px`` sale
de la barra de ejecución y el ``mae_usd`` de cada trade se fija como
``−min(serie ajustada sin costes)·pv·lots`` (barras + spread en cortos), nunca por debajo
de la pérdida bruta (``read_trade_table`` lo rechazaría) ni de 0, con un margen
diminuto por el redondeo a float32 del ``.sqx``.

Mapa criterio → test (plan §4, Fase 2):
  B1  test_b1_carga_y_1d_en_minuto
  B2  test_b2_windows_identico_a_una_corrida_sin_ohlc
  B3  test_b3_sin_ohlc_cae_a_ventanas_con_la_pata_en_el_warning
  B4  test_b4_ohlc_desplazado_60_min_no_cuadra_y_cae_a_ventanas
  B5  test_b5_monotonia_minuto_frente_a_ventanas_con_crn
  B6  test_b6_costes_en_precio_plano_corto_y_largo (+ nivel del motor en ct)
  B7  test_b7_eje_ftmo_en_la_semana_de_desfase_dst
  B8  test_b8_par_2d_cfd_en_minuto_y_difiere_de_ventanas
  B9  (test existente) test_futures_sqx_legs.py::test_c3_axi_par_y_diag_de_pata_son_409
      + test_b12_futuros_sqx_sin_flotante_m1_y_par_sigue_en_409 aquí
  B10 regresión: test_cfd_sqx_legs.py, test_futures_sqx_legs.py, test_propfirm_pair.py y
      la suite completa (se ejecutan aparte)
  B13 test_b13_recorte_por_el_mae_de_sqx_largo_y_corto (+ cota en cartera sintética)
  B11 test_b11_trade_que_cruza_la_medianoche_de_praga_cuenta_en_los_dos_dias
  B12 test_b12_axi_y_diagnostico_por_pata_con_cartera_cfd_son_409
Extras: serie sintética sin barras, ``with_minutes=False`` sin claves nuevas.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_cfd_minute_float.py -q
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from poc_web.api import main
from poc_web.api.main import (
    AxiSimRequest,
    LegDiagRequest,
    PortfolioLoadRequest,
    PropfirmSimRequest,
    _load_sqx_leg,
    _propfirm_setup,
    portfolio_leg_diagnostics,
    portfolio_load,
    simulate_axi,
    simulate_propfirm,
    simulate_propfirm_pair,
)
from src.risk.atr import load_ohlc_m1
from src.risk.mc_portfolio import attach_lots, minute_valley_series
from src.risk.propfirm import (
    _minute_columns,
    build_day_blocks_portfolio,
)
from src.risk.propfirm_trailing import trailing_minute_bounds

_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer_minute",
    Path(__file__).resolve().parents[2] / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx

SWING = "ftmo_50k_swing"
ACCOUNT = 50000.0
PV = 100000.0
TICK = 0.0001
SPREAD = 0.3 * TICK       # typical_spread_points 0,3 × tick del YAML (EURUSD_ftmo)
COMM = 5.0                # USD por lote ida y vuelta
LOTS = 0.5
OHLC_NAME = "EURUSD_ftmo.csv"


def _f32(x: float) -> float:
    return float(np.float32(x))


# ---------------------------------------------------------------------------
# Mundo sintético: OHLC M1 + .sqx coherentes
# ---------------------------------------------------------------------------
class World:
    def __init__(self, legs, ohlc_path, ohlc, trades):
        self.legs = legs              # rutas .sqx (absolutas)
        self.ohlc_path = ohlc_path    # CSV escrito (o None)
        self.ohlc = ohlc              # DataFrame (o,h,l,c) SIN desplazar
        self.trades = trades          # por pata: lista de dicts del writer


def make_world(tmp_path: Path, specs_per_leg: list[list[dict]], *, seed: int = 0,
               flat: float | None = None, write_ohlc: bool = True,
               ohlc_shift_min: int = 0, drop_ranges: list[tuple] | None = None,
               overrides: dict | None = None, p_win: float = 0.65) -> World:
    """``specs_per_leg``: una lista de specs por pata; spec = ``open``/``close``
    (datetime del servidor), ``side`` (``"buy"``/``"sell"``/None = elegir mirando el
    futuro con prob. ``p_win`` de acertar), ``swap`` (USD totales, def. 0), ``dist``
    (distancia al SL, def. 15 pips), ``mae`` (USD totales; si falta, el de la serie). ``flat`` = precio constante (barras planas).
    ``drop_ranges`` = [(ini, fin)] de barras que NO se escriben en el CSV (el trade
    queda sin cobertura); el MAE se calcula antes de borrar."""
    rng = np.random.default_rng(seed)
    minutes: set = set()
    for specs in specs_per_leg:
        for sp in specs:
            minutes.update(pd.date_range(pd.Timestamp(sp["open"]).floor("min"),
                                         pd.Timestamp(sp["close"]).floor("min"),
                                         freq="min"))
    idx = pd.DatetimeIndex(sorted(minutes))
    rows, price = [], (flat if flat is not None else 1.10)
    for ts in idx:
        o = round(price, 5)
        if flat is not None:
            c = h = l = o
        else:
            c = round(o + float(rng.normal(0.0, 1e-4)), 5)
            u = float(rng.uniform(0.0, 5e-5))
            h = round(max(o, c) + u, 5)
            l = round(min(o, c) - float(rng.uniform(0.0, 5e-5)), 5)
        rows.append((ts, o, h, l, c))
        price = c
    ohlc = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).set_index("ts")
    for ts, (o, h, l, c) in (overrides or {}).items():
        ohlc.loc[pd.Timestamp(ts)] = (o, h, l, c)

    legs, all_trades = [], []
    root = tmp_path / "extra" / "bank"
    root.mkdir(parents=True, exist_ok=True)
    for k, specs in enumerate(specs_per_leg):
        trades = []
        for sp in specs:
            ot, ct_raw = pd.Timestamp(sp["open"]), pd.Timestamp(sp["close"])
            o_px = _f32(ohlc.loc[ot.floor("min"), "o"])
            c_px = _f32(ohlc.loc[ct_raw.floor("min"), "c"])
            side = sp.get("side")
            if side is None:
                up = c_px > o_px
                win = rng.random() < p_win
                side = "buy" if up == win else "sell"
            is_buy = side == "buy"
            sign = 1.0 if is_buy else -1.0
            lots = sp.get("lots", LOTS)
            dist = sp.get("dist", 0.0015)
            sl_px = _f32(o_px - sign * dist)
            gross = (c_px - o_px) * sign * PV * lots
            swap = float(sp.get("swap", 0.0))
            pnl = gross - COMM * lots + swap
            ct = max(ct_raw, ot + pd.Timedelta(1, "ns"))
            s = minute_valley_series(ot.to_datetime64(), ct.to_datetime64(), o_px,
                                     is_buy, ohlc)
            mae_series = (-(float(s.min()) - (0.0 if is_buy else SPREAD)) * PV * lots
                          if s is not None else 0.0)
            mae = max(mae_series, -gross, 0.0) * (1.0 + 1e-6) + 1e-3
            if "mae" in sp:                       # MAE fijado a mano (B13: menor que la serie)
                mae = max(float(sp["mae"]), -gross, 0.0)
            trades.append(dict(
                open=ot.to_pydatetime(), close=ct_raw.to_pydatetime(), lots=lots,
                open_px=o_px, close_px=c_px, sl_px=sl_px, pnl=pnl, mae=mae, flag=1))
        legs.append(str(write_sqx(root / f"Strategy 8.{k}.1.sqx", trades,
                                  symbol="EURUSD_ftmo", instrument="EURUSD_cfd_ftmo",
                                  pv=PV, tick=TICK, capital=ACCOUNT)))
        all_trades.append(trades)

    out = ohlc.copy()
    for a, b in (drop_ranges or []):
        out = out[(out.index < pd.Timestamp(a)) | (out.index > pd.Timestamp(b))]
    ohlc_path = None
    if write_ohlc:
        d = main.SQX_OHLC_DIR
        d.mkdir(parents=True, exist_ok=True)
        ohlc_path = d / OHLC_NAME
        w = out.copy()
        w.index = w.index + pd.Timedelta(minutes=ohlc_shift_min)
        with open(ohlc_path, "w", encoding="utf-8") as fh:
            for ts, r in w.iterrows():
                fh.write(f"{ts:%Y.%m.%d},{ts:%H:%M},{r.o:.5f},{r.h:.5f},{r.l:.5f},"
                         f"{r.c:.5f},100\n")
    return World(legs, ohlc_path, ohlc, all_trades)


def big_specs(n_days: int = 100, n_legs: int = 3) -> list[list[dict]]:
    """3 patas solapadas: un trade por día hábil, abren a las 10:00/10:30/11:00 del
    servidor y duran 2 h (bloques flat-flat de 3 trades; todos del mismo día)."""
    days = pd.bdate_range("2023-01-02", periods=n_days)
    return [[dict(open=datetime(d.year, d.month, d.day, 10) + timedelta(minutes=30 * k),
                  close=datetime(d.year, d.month, d.day, 12) + timedelta(minutes=30 * k))
             for d in days] for k in range(n_legs)]


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "ohlc")


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path, big_specs())


def _load(legs, firm_key=SWING):
    return portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT,
                                               firm_key=firm_key))


def _req(res, **kw):
    base = dict(portfolio_id=res["portfolio_id"], firm_key=SWING,
                f_lo=0.01, f_hi=0.03, f_step=0.01, n_paths=300,
                n_challenges=5, n_sequences=100, capital0_eur=3000.0,
                fx_eur_usd=1.08, day_axis="ftmo")
    base.update(kw)
    return PropfirmSimRequest(**base)


def _worst_loss(ctx, fi: int) -> float:
    """Peor pérdida intradía respecto al inicio del día (USD) sobre bloques y días."""
    w = 0.0
    for bda in ctx["day_arrays_grid"][fi]:
        w = max(w, float(np.max(bda.rel_before - bda.rel_float)))
    return w


# ---------------------------------------------------------------------------
# B1 — carga + 1D con OHLC, eje FTMO
# ---------------------------------------------------------------------------
def test_b1_carga_y_1d_en_minuto(world):
    res = _load(world.legs)
    assert res["ok"], res
    g = res["guards"]["flotante_m1"]
    assert g["status"] == "ok" and "minuto" in g["detail"]
    assert "flotante_m1" in res["guards"] and res["guards"]["m1"]["status"] == "ok"
    out = simulate_propfirm(_req(res))
    meta = out["meta"]
    assert meta["float_mode"] == "minute"
    assert meta["day_axis"] == "ftmo"
    assert meta["leg_kind"] == "sqx_futures"
    assert "minuto" in meta["float_note"] and "ANTI-conservadores" in meta["float_note"]
    assert "MAE de SQX" in meta["float_note"] and "simultaneidad" in meta["float_note"]
    assert set(meta["minute_diag"]) == {"Strategy 8.0.1.sqx", "Strategy 8.1.1.sqx",
                                        "Strategy 8.2.1.sqx"}
    for d in meta["minute_diag"].values():
        assert d["coherencia"] == pytest.approx(1.0)
        assert d["sinteticas"] == 0 and d["n_trades"] == 100
        # enmienda v3: claves explícitas previas al recorte; aquí el MAE ya cubre la serie
        assert "frac_serie_peor_que_mae" not in d and "max_exceso_usd_lote" not in d
        assert d["frac_serie_peor_que_mae_antes_del_recorte"] == 0.0
        assert d["max_exceso_usd_lote_antes_del_recorte"] <= 0.01
        assert d["frac_recortadas"] == 0.0
    # el warning "Cota a precisión del backtest" es solo del modo ventanas
    assert not any("Cota a precisión del backtest" in w for w in out["warnings"])
    assert len(out["reports"]) == 3


def test_with_minutes_false_no_trae_claves_de_minuto(world):
    leg = _load_sqx_leg(world.legs[0], ACCOUNT, None, axis_hours=-1.0)
    assert "minute_valle" not in leg and "minute_diag" not in leg
    leg_m = _load_sqx_leg(world.legs[0], ACCOUNT, None, axis_hours=-1.0, with_minutes=True)
    assert leg_m is not leg                      # clave de caché distinta
    assert leg_m["minute_reason"] is None and len(leg_m["minute_valle"]) == 100


# ---------------------------------------------------------------------------
# B2 — sqx_float="windows" = camino anterior, bit a bit
# ---------------------------------------------------------------------------
def test_b2_windows_identico_a_una_corrida_sin_ohlc(world, tmp_path, monkeypatch):
    a = simulate_propfirm(_req(_load(world.legs), sqx_float="windows"))
    # corrida sin OHLC (auto cae a ventanas): cachés limpias y directorio vacío
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "vacio")
    b = simulate_propfirm(_req(_load(world.legs)))
    for out in (a, b):
        assert out["meta"]["float_mode"] == "windows"
        assert "minute_diag" not in out["meta"]
        out["meta"].pop("elapsed_s")
    assert a["reports"] == b["reports"]
    assert a["recommendation"] == b["recommendation"]
    assert a["meta"] == b["meta"]
    assert a["trajectory"] == b["trajectory"]
    # con ventanas explícitas sigue el warning de precisión del backtest
    assert any("Cota a precisión del backtest" in w for w in a["warnings"])


# ---------------------------------------------------------------------------
# B3 / B4 — sin OHLC y OHLC que no cuadra: ventanas + warning
# ---------------------------------------------------------------------------
def test_b3_sin_ohlc_cae_a_ventanas_con_la_pata_en_el_warning(tmp_path):
    w = make_world(tmp_path, big_specs(20), write_ohlc=False)
    res = _load(w.legs)
    assert res["ok"]                              # nunca es error de carga
    g = res["guards"]["flotante_m1"]
    assert g["status"] == "warn" and "Strategy 8.0.1.sqx" in g["detail"]
    out = simulate_propfirm(_req(res, f_lo=0.0025, f_hi=0.0075, f_step=0.0025))
    assert out["meta"]["float_mode"] == "windows"
    assert any("Strategy 8.0.1.sqx" in x and "ventanas" in x for x in out["warnings"])
    assert any("no disponible" in x for x in out["warnings"])


def test_b4_ohlc_desplazado_60_min_no_cuadra_y_cae_a_ventanas(tmp_path):
    w = make_world(tmp_path, big_specs(20), ohlc_shift_min=60)
    res = _load(w.legs)
    assert res["guards"]["flotante_m1"]["status"] == "warn"
    assert "no cuadra" in res["guards"]["flotante_m1"]["detail"]
    out = simulate_propfirm(_req(res, f_lo=0.0025, f_hi=0.0075, f_step=0.0025))
    assert out["meta"]["float_mode"] == "windows" and "minute_diag" not in out["meta"]
    assert any("no cuadra" in x and "ventanas" in x for x in out["warnings"])
    leg = _load_sqx_leg(w.legs[0], ACCOUNT, None, with_minutes=True)
    assert leg["minute_valle"] is None and "coherencia" in leg["minute_reason"]


# ---------------------------------------------------------------------------
# B5 — monotonía con CRN (solo sintético con MAE = −min(serie ajustada)·pv·lots)
# ---------------------------------------------------------------------------
def test_b5_monotonia_minuto_frente_a_ventanas_con_crn(world):
    res = _load(world.legs)
    req_m = _req(res)
    req_w = _req(res, sqx_float="windows")
    ctx_m, ctx_w = _propfirm_setup(req_m), _propfirm_setup(req_w)
    assert ctx_m["float_mode"] == "minute" and ctx_w["float_mode"] == "windows"
    out_m, out_w = simulate_propfirm(req_m), simulate_propfirm(req_w)
    lost = []
    for fi, (rm, rw) in enumerate(zip(out_m["reports"], out_w["reports"])):
        assert rm["f"] == rw["f"]
        assert rm["p_pass1"] >= rw["p_pass1"] - 1e-12, (fi, rm["p_pass1"], rw["p_pass1"])
        wm, ww = _worst_loss(ctx_m, fi), _worst_loss(ctx_w, fi)
        assert wm <= ww + 1e-6, (fi, wm, ww)
        lost.append((wm, ww))
    # el escenario discrimina: a algún f ventanas es estrictamente más pesimista
    assert any(ww > wm + 1.0 for wm, ww in lost), lost


# ---------------------------------------------------------------------------
# B6 — costes en precio con precio plano; y en ct el nivel del motor = PnL neto
# ---------------------------------------------------------------------------
def test_b6_costes_en_precio_plano_corto_y_largo(tmp_path):
    flat = 1.125                                   # exacto en float32
    days = pd.bdate_range("2023-02-06", periods=6)
    specs = [dict(open=datetime(d.year, d.month, d.day, 10, 0),
                  close=datetime(d.year, d.month, d.day, 11, 0),
                  side="buy" if k % 2 else "sell") for k, d in enumerate(days)]
    # un corto y un largo que cruzan el rollover 23:00 con CARGO de swap (−0,5 / −0,8 USD)
    specs.append(dict(open=datetime(2023, 2, 14, 22, 0), close=datetime(2023, 2, 15, 2, 0),
                      side="sell", swap=-0.5))
    specs.append(dict(open=datetime(2023, 2, 16, 22, 0), close=datetime(2023, 2, 17, 2, 0),
                      side="buy", swap=-0.8))
    w = make_world(tmp_path, [specs], flat=flat)
    leg = _load_sqx_leg(w.legs[0], ACCOUNT, None, axis_hours=0.0, with_minutes=True)
    assert leg["minute_reason"] is None
    # la fila i de la tabla (orden por close_ts) es el trade local i
    swap_by_close = {datetime(2023, 2, 15, 2, 0): 0.5, datetime(2023, 2, 17, 2, 0): 0.8}
    n_checked = 0
    for i, t in enumerate(sorted(w.trades[0], key=lambda t: t["close"])):
        ts, vals, _ot, _ct = leg["minute_valle"][i]
        is_buy = t["sl_px"] < t["open_px"]
        swap_cost = swap_by_close.get(t["close"], 0.0)          # cargos (−0,5 / −0,8 USD)
        cost = (COMM * t["lots"] + swap_cost) / t["lots"] / PV   # comm + swap, en precio
        expected = -(0.0 if is_buy else SPREAD) - cost
        np.testing.assert_allclose(vals, expected, atol=1e-7)
        n_checked += 1
    assert n_checked == 8
    # nivel del motor: en ct el nivel es EXACTAMENTE el PnL neto realizado (envolvente
    # cero en ct, realizado inyectado en ct)
    s = attach_lots(leg["s"], np.full((1, 8), 1.0))
    blocks = build_day_blocks_portfolio({"p": s})
    cross = [b for b in blocks if b.n_trades == 1 and b.close_ord[0] > b.open_ord[0]]
    assert len(cross) == 2
    for b in cross:
        tcols = _minute_columns(b, {0: leg["minute_valle"]})
        lots = np.ones(b.n_trades)
        lower, upper = trailing_minute_bounds(b, lots, {0: PV}, tcols)
        assert lower[tcols.ct_pos[0]] == pytest.approx(b.pnl_per_lot[0], abs=1e-9)
        assert upper[tcols.ct_pos[0]] == pytest.approx(b.pnl_per_lot[0], abs=1e-9)
        # y un minuto antes sigue siendo el nivel flotante (coste + spread si corto)
        assert lower[tcols.ct_pos[0] - 1] < b.pnl_per_lot[0] + 1e-9
    assert leg["fleg"].pv == PV


# ---------------------------------------------------------------------------
# B7 — eje FTMO en la semana de desfase DST (20-mar-2023)
# ---------------------------------------------------------------------------
def test_b7_eje_ftmo_en_la_semana_de_desfase_dst(tmp_path):
    days = pd.bdate_range("2023-03-06", periods=5)
    specs = [dict(open=datetime(d.year, d.month, d.day, 10, 0),
                  close=datetime(d.year, d.month, d.day, 11, 30), side="buy") for d in days]
    specs.append(dict(open=datetime(2023, 3, 20, 22, 0), close=datetime(2023, 3, 21, 1, 30),
                      side="buy"))
    w = make_world(tmp_path, [specs])
    srv = _load_sqx_leg(w.legs[0], ACCOUNT, None, axis_hours=0.0, with_minutes=True)
    ftmo = _load_sqx_leg(w.legs[0], ACCOUNT, None, axis_hours=-1.0, with_minutes=True)
    assert srv["minute_reason"] is None and ftmo["minute_reason"] is None
    i = int(np.where(pd.to_datetime(srv["s"].open_t) == pd.Timestamp("2023-03-20 22:00"))[0][0])
    for j in range(len(specs)):
        ts_s, v_s, _, _ = srv["minute_valle"][j]
        ts_f, v_f, ot_f, ct_f = ftmo["minute_valle"][j]
        shift = (pd.Timestamp(ftmo["s"].open_t[j]) - pd.Timestamp(srv["s"].open_t[j]))
        assert shift == pd.Timedelta(hours=-2 if j == i else -1)
        # los ts de la serie se desplazan EXACTAMENTE igual que open_ts
        assert (pd.to_datetime(ts_f) - pd.to_datetime(ts_s) == shift).all()
        np.testing.assert_array_equal(v_f, v_s)
        assert ot_f == pd.Timestamp(ftmo["s"].open_t[j])
        assert ct_f == pd.Timestamp(ftmo["s"].close_t[j])
    ts_f = pd.to_datetime(ftmo["minute_valle"][i][0])
    assert ts_f[0] == pd.Timestamp("2023-03-20 20:00")
    assert ts_f[-1] == pd.Timestamp("2023-03-20 23:30")


# ---------------------------------------------------------------------------
# B8 — par 2D con cartera .sqx CFD
# ---------------------------------------------------------------------------
def test_b8_par_2d_cfd_en_minuto_y_difiere_de_ventanas(world):
    res = _load(world.legs)
    out = simulate_propfirm_pair(_req(res))                # 200: no levanta HTTPException
    meta = out["meta"]
    assert meta["float_mode"] == "minute" and meta["day_axis"] == "ftmo"
    assert meta["leg_kind"] == "sqx_futures" and "minuto" in meta["float_note"]
    assert set(meta["minute_diag"]) == {l["pata"] for l in res["legs"]}
    assert meta["cells_resolved"] == len(out["cells"]) == 9
    assert out["recommendation"] is not None
    # la diagonal queda cerca del 1D (mismo f; solo cambian gaps y seed de secuencia)
    one_d = simulate_propfirm(_req(res))
    diag = {(c["f_challenge"], c["f_funded"]): c for c in out["cells"]}
    for r in one_d["reports"]:
        assert abs(diag[(r["f"], r["f"])]["p_pass1"] - r["p_pass1"]) < 0.25
    # prueba de cableado: el 2D en minuto DIFIERE del 2D en ventanas (aquí ventanas es
    # claramente más pesimista: B5)
    out_w = simulate_propfirm_pair(_req(res, sqx_float="windows"))
    assert out_w["meta"]["float_mode"] == "windows" and "minute_diag" not in out_w["meta"]
    cells_w = {(c["f_challenge"], c["f_funded"]): c for c in out_w["cells"]}
    diffs = [abs(diag[k]["p_pass1"] - cells_w[k]["p_pass1"])
             for k in diag if k in cells_w]
    assert diffs and max(diffs) > 0.0
    assert any(diag[k]["p_pass1"] > cells_w[k]["p_pass1"] for k in diag if k in cells_w)


# ---------------------------------------------------------------------------
# B11 — trade que cruza la medianoche de Praga: cuenta en los dos días
# ---------------------------------------------------------------------------
def test_b11_trade_que_cruza_la_medianoche_de_praga_cuenta_en_los_dos_dias(tmp_path):
    days = pd.bdate_range("2023-01-02", periods=8)
    specs = [dict(open=datetime(d.year, d.month, d.day, 10, 0),
                  close=datetime(d.year, d.month, d.day, 11, 0), side="buy")
             for d in days if d != pd.Timestamp("2023-01-09")]
    # largo: abre 23:30 del lunes 9-ene (servidor) y cierra 02:30 del martes. Praga =
    # servidor − 1 h: abre 22:30 lun, cierra 01:30 mar; la medianoche de Praga es 01:00
    # del servidor. Caída de 10 pips antes (23:50 srv) y de 25 pips después (01:50 srv).
    specs.append(dict(open=datetime(2023, 1, 9, 23, 30), close=datetime(2023, 1, 10, 2, 30),
                      side="buy"))
    base = 1.10
    # barras planas salvo dos mínimos (las dos caídas)
    ov = {pd.Timestamp("2023-01-09 23:50"): (base, base, base - 0.0010, base),
          pd.Timestamp("2023-01-10 01:50"): (base, base, base - 0.0025, base)}
    w = make_world(tmp_path, [specs], flat=base, overrides=ov)
    res = _load(w.legs)
    assert res["guards"]["flotante_m1"]["status"] == "ok", res["guards"]["flotante_m1"]
    ctx = _propfirm_setup(_req(res, f_lo=0.0025, f_hi=0.0075, f_step=0.0025))
    assert ctx["float_mode"] == "minute"
    leg = ctx["legs_used"][0]
    i = int(np.where(pd.to_datetime(leg["s"].open_t) == pd.Timestamp("2023-01-09 22:30"))[0][0])
    ts, vals, _ot, ct = leg["minute_valle"][i]
    ts = pd.to_datetime(ts)
    d_mon, d_tue = pd.Timestamp("2023-01-09").toordinal(), pd.Timestamp("2023-01-10").toordinal()
    bi = next(k for k, b in enumerate(ctx["blocks"])
              if b.open_ord[0] == d_mon and b.close_ord[0] == d_tue)
    fi = 1
    lots = float(ctx["lots_by"][0][fi][i])
    bda = ctx["day_arrays_grid"][fi][bi]
    assert list(bda.day_ord) == [d_mon, d_tue]
    flot = vals * lots * PV                         # nivel flotante del trade por barra
    mon = flot[ts.normalize() == pd.Timestamp("2023-01-09")]
    tue = flot[(ts.normalize() == pd.Timestamp("2023-01-10")) & (ts < ct)]
    realized = lots * float(leg["fleg"].pnl_per_lot[i])
    # lunes: el valle de 10 pips; martes: el de 25 pips (el realizado en ct es mayor)
    assert float(mon.min()) < -0.0010 * lots * PV + 1e-6 < 0
    assert bda.rel_float[0] == pytest.approx(float(mon.min()), abs=1e-6)
    assert bda.rel_float[1] == pytest.approx(min(float(tue.min()), realized), abs=1e-6)
    assert bda.rel_float[1] < bda.rel_float[0] < 0
    assert bda.rel_before[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B13 (enmienda v3) — recorte de la serie por el MAE de SQX (+ costes, + spread en cortos)
# ---------------------------------------------------------------------------
def _b13_world(tmp_path: Path) -> World:
    """Precio plano 1,10; en la barra de EJECUCIÓN de cada trade el low (largos) o el high
    (cortos) está 50 pips más allá del precio de entrada (precios previos a la ejecución);
    el resto de barras es plano. El MAE de SQX (20-40 USD en total) es mucho menor que esos
    50 pips (250 USD con 0,5 lotes), así que la serie sin recortar sería muy peor que él."""
    base = 1.10
    days = pd.bdate_range("2023-02-06", periods=10)
    specs, ov = [], {}
    for k, d in enumerate(days):
        ot = datetime(d.year, d.month, d.day, 10, 0)
        buy = k % 2 == 0
        specs.append(dict(open=ot, close=datetime(d.year, d.month, d.day, 11, 0),
                          side="buy" if buy else "sell", mae=20.0 + 5.0 * k))
        ov[pd.Timestamp(ot)] = ((base, base, base - 0.0050, base) if buy
                                else (base, base + 0.0050, base, base))
    return make_world(tmp_path, [specs], flat=base, overrides=ov)


def test_b13_recorte_por_el_mae_de_sqx_largo_y_corto(tmp_path):
    w = _b13_world(tmp_path)
    leg = _load_sqx_leg(w.legs[0], ACCOUNT, None, axis_hours=0.0, with_minutes=True)
    assert leg["minute_reason"] is None
    fl, s = leg["fleg"], leg["s"]
    n_long = n_short = 0
    for i in range(s.n_trades):
        _ts, vals, _ot, _ct = leg["minute_valle"][i]
        adv = float(fl.adverse_per_lot[i])                 # (mae + comm + swap-cargo)/lots
        if bool(s.is_buy[i]):
            floor = -adv / PV
            n_long += 1
        else:
            floor = -adv / PV - SPREAD
            n_short += 1
        # recortada EXACTAMENTE en el suelo (el low/high de 50 pips era mucho peor) ...
        assert float(np.min(vals)) == pytest.approx(floor, abs=1e-9)
        assert float(np.min(vals)) > -0.0050 + 1e-4        # ... y no queda a 50 pips
        # ... y nunca por encima del suelo en el resto de barras (planas: coste/spread)
        assert float(np.max(vals)) <= 1e-12
    assert (n_long, n_short) == (5, 5)
    # el diagnóstico conserva lo previo al recorte y cuenta los recortados
    d = leg["minute_diag"]
    assert d["frac_recortadas"] == pytest.approx(1.0)
    assert d["frac_serie_peor_que_mae_antes_del_recorte"] == pytest.approx(1.0)
    assert d["max_exceso_usd_lote_antes_del_recorte"] > 200.0   # ~ (50 pips − MAE)·1e5/lote


def test_b13_serie_acotada_por_el_mae_en_toda_la_cartera_sintetica(tmp_path):
    """Tras el recorte, −min(serie)·pv <= adverse_per_lot (+ spread·pv en cortos) + 1e-9 en
    TODOS los trades (cartera con trades recortados, otros no, y una serie sintética)."""
    gap = (datetime(2023, 1, 9, 10, 0) - timedelta(minutes=1), datetime(2023, 1, 9, 12, 30))
    # cada mundo reescribe el mismo CSV de OHLC: se construye y se mide de uno en uno
    builders = (lambda: _b13_world(tmp_path / "a"),
                lambda: make_world(tmp_path / "b", big_specs(30, 2)),
                lambda: make_world(tmp_path / "c", [big_specs(40, 1)[0]], drop_ranges=[gap]))
    n_tot = n_clip = n_synth = 0
    for build in builders:
        wd = build()
        main._sqx_ohlc_cache.clear()
        for lp in wd.legs:
            leg = _load_sqx_leg(lp, ACCOUNT, None, axis_hours=0.0, with_minutes=True)
            assert leg["minute_reason"] is None, leg["minute_reason"]
            fl, s = leg["fleg"], leg["s"]
            n_synth += leg["minute_diag"]["sinteticas"]
            for i in range(s.n_trades):
                vals = leg["minute_valle"][i][1]
                cota = float(fl.adverse_per_lot[i]) + (0.0 if bool(s.is_buy[i]) else SPREAD * PV)
                assert -float(np.min(vals)) * PV <= cota + 1e-9, (lp, i)
                n_tot += 1
            n_clip += round(leg["minute_diag"]["frac_recortadas"] * s.n_trades)
    assert n_tot == 10 + 60 + 40 and n_clip >= 10 and n_synth == 1


# ---------------------------------------------------------------------------
# Serie sintética (trade sin barras)
# ---------------------------------------------------------------------------
def test_trade_sin_barras_usa_serie_sintetica_y_se_cuenta(tmp_path):
    specs = big_specs(40, 1)[0]
    gap = (datetime(2023, 1, 2 + 7, 10, 0) - timedelta(minutes=1), datetime(2023, 1, 2 + 7, 12, 30))
    w = make_world(tmp_path, [specs], drop_ranges=[gap])
    leg = _load_sqx_leg(w.legs[0], ACCOUNT, None, axis_hours=-1.0, with_minutes=True)
    assert leg["minute_reason"] is None            # 39/40 = 97,5% ≥ 95%
    d = leg["minute_diag"]
    assert d["sinteticas"] == 1 and d["coherencia"] == pytest.approx(39 / 40)
    sizes = [len(v[0]) for v in leg["minute_valle"].values()]
    assert sizes.count(1) == 1                     # la sintética: ts=[ot]
    k = sizes.index(1)
    ts, vals, ot, ct = leg["minute_valle"][k]
    fl = leg["fleg"]
    is_buy = bool(leg["s"].is_buy[k])
    # serie sintética = −mae/lots/pv, y con el ajuste de costes (spread solo en cortos,
    # comisión y swap-cargo) queda en −adverse_per_lot/pv − spread (corto)
    expected = -fl.adverse_per_lot[k] / PV - (0.0 if is_buy else SPREAD)
    assert vals[0] == pytest.approx(expected, abs=1e-9)
    assert pd.Timestamp(ts[0]) == ot


# ---------------------------------------------------------------------------
# B12 — AXI y diagnóstico por pata siguen en 409; futuros sin flotante_m1
# ---------------------------------------------------------------------------
def test_b12_axi_y_diagnostico_por_pata_con_cartera_cfd_son_409(world):
    res = _load(world.legs)
    pid = res["portfolio_id"]
    with pytest.raises(HTTPException) as e1:
        simulate_axi(AxiSimRequest(portfolio_id=pid))
    assert e1.value.status_code == 409
    with pytest.raises(HTTPException) as e2:
        portfolio_leg_diagnostics(pid, LegDiagRequest(f=0.01))
    assert e2.value.status_code == 409


def test_b12_futuros_sqx_sin_flotante_m1_y_par_sigue_en_409(tmp_path):
    root = tmp_path / "extra" / "fut"
    paths = []
    for k in range(3):
        tr = [_W.long_win(open=datetime(2023, 1, 2 + d, 10 + k, 0),
                          close=datetime(2023, 1, 2 + d, 11 + k, 0)) for d in range(8)]
        paths.append(str(write_sqx(root / f"Strategy 1.{k}.1.sqx", tr)))
    res = portfolio_load(PortfolioLoadRequest(legs=paths, account_size=ACCOUNT,
                                              firm_key="ftmo_futures_growth_50k"))
    assert "flotante_m1" not in res["guards"]
    assert res["guards"]["m1"]["status"] == "ok"
    with pytest.raises(HTTPException) as e:
        simulate_propfirm_pair(PropfirmSimRequest(
            portfolio_id=res["portfolio_id"], firm_key="ftmo_futures_growth_50k"))
    assert e.value.status_code == 409 and "futuros" in e.value.detail
