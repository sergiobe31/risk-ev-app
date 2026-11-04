"""Guardrail ``entrada-4-bloques``, fase F3: lente AXI con carteras de patas ``.sqx`` (plan §2.7, enmiendas E1, E19-E21).

Todo sintético y hermético, como ``test_sqx_fichas.py``: dos patas ``.sqx`` de AXI escritas con el writer de
``tests/test_sqx_trades.py`` (``XAUUSD_axi`` y ``EURUSD_axi``, con sus fichas OFICIALES de ``broker_axi.yaml``), un OHLC M1
sintético en ``tmp`` apuntado con ``main.SQX_OHLC_DIR_OVERRIDES = {"axi": tmp}`` (formato MT5 sin cabecera,
``yyyy.MM.dd,HH:mm,o,h,l,c,vol``, hora del servidor; el nombre de fichero es el ``ohlc_m1`` de la ficha,
``<sqx_symbol>.csv``), el fichero de fichas de usuario en un temporal (I4) y las cachés vacías.

Mapa criterio -> test (el id del criterio está en el nombre):

- AC3.1 200, ``p_pass + p_ruin + p_sin_resolver = 1`` y campos *raw* == ``run_portfolio_mc`` (tolerancia cero) ... ``test_ac3_1_*``
- AC3.2 valle diario = mínimo por fecha; operación sin barras = puntos ``daily``; pata sin OHLC = toda ``daily`` y
  guarda ``flotante_m1`` en ``warn``; cobertura 100 % ................................................. ``test_ac3_2_*``
- AC3.2bis (E1 + E21) forma del valle (fechas, floats = mínimo de la serie del día, que puede ser > 0, claves ⊆ fechas de
  [ot, ct], una entrada por operación) ......................................................................... ``test_ac3_2bis_*``
- E20 el spread de la serie por minuto (coherencia, cortos y cota diaria) es el del backtest: ``defaultSpread · tickSize``
  del .sqx; la ficha solo si el .sqx no lo trae o no es > 0 ................................................... ``test_e20_*``
- AC3.3 lotes == ``fixed_risk_micros(..., cap=None, lot_step=0.01, min_lot=0.01)`` ..................... ``test_ac3_3_*``
- AC3.4 I6 (la ``StrategyMC`` cacheada sigue con ``valle == {}`` y la prop firm no cambia) y
  ``synth_mode="point"`` == las series de hoy (hash tomado ANTES de la F3 sobre los fixtures de
  ``test_cfd_minute_float``) ......................................................................... ``test_ac3_4_*``
- AC3.5 cartera ``.sqx`` de FTMO (CFD o futuros) en la lente AXI = 409 con ".sqx" y "AXI"; la vía CSV no cambia
  (``test_axi_minute.py`` y ``test_axi_path.py`` pasan sin tocarlos, incluido el hash ``aee193c6f31e13e6``) .. ``test_ac3_5_*``
- AC3.6 meta (``sizing``, ``sizing_base``, ``float_mode``, ``float_note``, ``broker``, ``legs[].trades``) ... ``test_ac3_6_*``
- ``portfolio_load`` con una cartera ``.sqx`` de AXI: guardas coherentes ................................ ``test_carga_*``

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_axi_sqx.py -q
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from poc_web.api import main, symbols
from sqx.sqx_trades import DEFAULT_CFD_COSTS_PATH, load_costs_policy
from src.risk.futures_legs import fixed_risk_micros
from src.risk.mc_portfolio import attach_lots, minute_valley_series, run_portfolio_mc

ROOT = Path(__file__).resolve().parents[2]
REAL_USER_FILES = (ROOT / "config" / "symbols.user.yaml", ROOT / "config" / "propfirms.user.yaml")
ACCOUNT = 5000.0                 # el tamaño de cuenta de la Seed de AXI
F_REF = 0.005

_SPEC = importlib.util.spec_from_file_location("_tst_sqx_trades_writer_axi", ROOT / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx
patch_instrument_info = _W.patch_instrument_info

# Dos patas de AXI. ``hour``: hora del servidor a la que abre su operación diaria (90 min); no se solapan, así que cada
# operación es un bloque flat-flat. ``step``: desviación típica de la barra M1 (precio); ``dists``: distancias al SL.
SPECS = {
    "XAUUSD_axi": dict(file="AXI XAU.sqx", instrument="XAUUSD_mt5_axi", pv=100.0, tick=0.01, px0=2000.0, step=0.35,
                       nd=2, dists=(3.0, 4.5, 6.0), hour=10),
    "EURUSD_axi": dict(file="AXI EUR.sqx", instrument="EURUSD_axi", pv=100000.0, tick=0.0001, px0=1.10, step=6e-5,
                       nd=5, dists=(0.0006, 0.0009, 0.0012), hour=12),
}
LOTS = 0.1
# Operaciones de varios días SIN barras M1 en la pata de oro (un largo de 4 fechas y un corto de 3), DESPUÉS del periodo
# con operaciones diarias: ``start`` = días naturales tras la última operación diaria (a las 10:00), ``span`` = días
# naturales hasta el cierre (a las 15:00).
MULTIDAY = (
    dict(start=10, span=3, side="buy", open_px=2000.0, sl_px=1995.0, close_px=1990.0, mae=120.0),
    dict(start=24, span=2, side="sell", open_px=2000.0, sl_px=2005.0, close_px=2008.0, mae=90.0),
)


def _f32(x: float) -> float:
    return float(np.float32(x))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _ficheros_reales_de_usuario_intactos():
    """I4: ningún test de este módulo crea ni toca ``config/symbols.user.yaml`` ni ``config/propfirms.user.yaml``."""
    def state():
        return {str(p): (p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in REAL_USER_FILES}

    before = state()
    yield
    assert state() == before, "los tests tocaron un fichero de usuario REAL (I4)"


@pytest.fixture(autouse=True)
def _aislado(monkeypatch, tmp_path):
    """Cachés vacías, catálogo y OHLC aislados del disco real (nunca el OHLC de AXI real), fichas de usuario en tmp."""
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")
    monkeypatch.setattr(main, "SQX_OHLC_DIR_OVERRIDES", {"axi": tmp_path / "ohlc_axi"})
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "symbols.user.yaml")
    symbols._reload()


@pytest.fixture()
def api():
    return TestClient(main.app)


@pytest.fixture()
def spy(monkeypatch):
    """Captura las ``StrategyMC`` y los bloques que ``simulate_axi`` pasa a ``build_block_records`` (lo que el endpoint
    USA de verdad, no un helper)."""
    seen: dict = {}
    real = main.build_block_records

    def wrapped(strategies, blocks, oos_start=None):
        seen["strats"] = strategies
        seen["blocks"] = real(strategies, blocks, oos_start=oos_start)
        return seen["blocks"]

    monkeypatch.setattr(main, "build_block_records", wrapped)
    return seen


# ---------------------------------------------------------------------------
# Mundo sintético: OHLC M1 + .sqx de AXI coherentes
# ---------------------------------------------------------------------------
class World:
    def __init__(self, legs, ohlc_dir, ohlc, trades):
        self.legs = legs              # {símbolo SQX: ruta del .sqx}
        self.paths = list(legs.values())
        self.ohlc_dir = ohlc_dir
        self.ohlc = ohlc              # {símbolo: DataFrame o,h,l,c con TODAS las barras}
        self.trades = trades          # {símbolo: lista de dicts del writer}


N_DAYS = 80                      # operaciones diarias por pata (2 patas -> 160 bloques flat-flat)


def make_axi_world(tmp_path: Path, *, n_days: int = N_DAYS, seed: int = 11,
                   write_ohlc: tuple[str, ...] = ("XAUUSD_axi", "EURUSD_axi"), multiday: bool = False,
                   drop_bars_of_trade: int | None = None) -> World:
    """Dos patas de AXI con ``n_days`` operaciones diarias de 90 min (y, si ``multiday``, las dos operaciones de varios
    días SIN barras de la pata de oro). ``write_ohlc``: de qué símbolos se escribe el CSV de OHLC (el de los demás no
    existe). ``drop_bars_of_trade``: índice de la operación diaria de ORO a la que se quitan las barras del CSV (queda sin
    cobertura; el MAE se calcula antes de quitarlas)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    odir = tmp_path / "ohlc_axi"
    odir.mkdir(parents=True, exist_ok=True)
    legs, ohlcs, all_trades = {}, {}, {}
    for sym, sp in SPECS.items():
        sheet = symbols.resolve(sym)                      # la ficha OFICIAL de AXI: spread y comisión coherentes
        spread, comm = sheet.spread_px, sheet.commission["per_lot_rt"]
        windows = [(pd.Timestamp(d) + pd.Timedelta(hours=sp["hour"]),
                    pd.Timestamp(d) + pd.Timedelta(hours=sp["hour"], minutes=90)) for d in days]
        rows, price = [], sp["px0"]
        for ot, ct in windows:
            for ts in pd.date_range(ot, ct, freq="min"):
                o = round(price, sp["nd"])
                c = round(o + float(rng.normal(0.0, sp["step"])), sp["nd"])
                hi = round(max(o, c) + float(rng.uniform(0.0, sp["step"])), sp["nd"])
                lo = round(min(o, c) - float(rng.uniform(0.0, sp["step"])), sp["nd"])
                rows.append((ts, o, hi, lo, c))
                price = c
        ohlc = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).set_index("ts")
        trades = []
        for ot, ct in windows:
            o_px, c_px = _f32(ohlc.loc[ot, "o"]), _f32(ohlc.loc[ct, "c"])
            buy = (c_px > o_px) == (rng.random() < 0.6)
            sign = 1.0 if buy else -1.0
            sl_px = _f32(o_px - sign * float(rng.choice(sp["dists"])))
            gross = (c_px - o_px) * sign * sp["pv"] * LOTS
            s = minute_valley_series(ot.to_datetime64(), ct.to_datetime64(), o_px, buy, ohlc)
            mae_series = -(float(s.min()) - (0.0 if buy else spread)) * sp["pv"] * LOTS
            mae = max(mae_series, -gross, 0.0) * (1.0 + 1e-6) + 1e-3
            trades.append(dict(open=ot.to_pydatetime(), close=ct.to_pydatetime(), lots=LOTS, open_px=o_px,
                               close_px=c_px, sl_px=sl_px, pnl=gross - comm * LOTS, mae=mae, flag=1))
        if sym == "XAUUSD_axi" and multiday:
            for m in MULTIDAY:
                m_open = pd.Timestamp(days[-1]) + pd.Timedelta(days=m["start"], hours=10)
                m_close = m_open + pd.Timedelta(days=m["span"], hours=5)
                sign = 1.0 if m["side"] == "buy" else -1.0
                gross = (m["close_px"] - m["open_px"]) * sign * sp["pv"] * LOTS
                trades.append(dict(open=m_open.to_pydatetime(), close=m_close.to_pydatetime(), lots=LOTS,
                                   open_px=m["open_px"], close_px=m["close_px"], sl_px=m["sl_px"],
                                   pnl=gross - comm * LOTS, mae=m["mae"], flag=1))
        legs[sym] = str(write_sqx(tmp_path / "extra" / "bank" / sp["file"], trades, symbol=sym,
                                  instrument=sp["instrument"], pv=sp["pv"], tick=sp["tick"], capital=ACCOUNT))
        ohlcs[sym], all_trades[sym] = ohlc, trades
        if sym in write_ohlc:
            out = ohlc
            if sym == "XAUUSD_axi" and drop_bars_of_trade is not None:
                t = trades[drop_bars_of_trade]
                out = ohlc[(ohlc.index < pd.Timestamp(t["open"]).floor("min"))
                           | (ohlc.index > pd.Timestamp(t["close"]).floor("min"))]
            with open(odir / f"{sym}.csv", "w", encoding="utf-8") as fh:
                nd = sp["nd"]
                for ts, r in out.iterrows():
                    fh.write(f"{ts:%Y.%m.%d},{ts:%H:%M},{r.o:.{nd}f},{r.h:.{nd}f},{r.l:.{nd}f},{r.c:.{nd}f},100\n")
    return World(legs, odir, ohlcs, all_trades)


FLAT_PX = 2000.0


def _profit_trade() -> dict:
    """Una operación larga de oro (E21) con barras: el día 1 entra a 2000 y a las 10:30 la barra sube a 2010; el día 2 el precio
    se queda en 2010 toda la sesión. Su valle del día 2 es POSITIVO (10 − comisión/lote/pv)."""
    d1, d2 = pd.Timestamp("2023-03-06"), pd.Timestamp("2023-03-07")
    bars = [(ts, FLAT_PX, FLAT_PX, FLAT_PX, FLAT_PX)
            for ts in pd.date_range(d1 + pd.Timedelta(hours=10), d1 + pd.Timedelta(hours=10, minutes=29), freq="min")]
    bars.append((d1 + pd.Timedelta(hours=10, minutes=30), FLAT_PX, FLAT_PX + 10.0, FLAT_PX, FLAT_PX + 10.0))
    bars += [(ts, FLAT_PX + 10.0, FLAT_PX + 10.0, FLAT_PX + 10.0, FLAT_PX + 10.0)
             for ts in pd.date_range(d2 + pd.Timedelta(hours=10), d2 + pd.Timedelta(hours=15), freq="min")]
    trade = dict(open=datetime(2023, 3, 6, 10, 0), close=datetime(2023, 3, 7, 15, 0), lots=LOTS, open_px=FLAT_PX,
                 close_px=FLAT_PX + 10.0, sl_px=FLAT_PX - 5.0, pnl=100.0 - 0.45, mae=1.0, flag=1)
    return dict(trade=trade, bars=bars)


PROFIT_TRADE = _profit_trade()


def make_flat_leg(tmp_path: Path, *, side: str, ask_offset: float = 0.0, default_spread: str | None = None,
                  n_days: int = 12, write_ohlc: bool = True, name: str = "AXI XAU.sqx",
                  extra: list[dict] | None = None) -> str:
    """Una pata ``XAUUSD_axi`` con OHLC PLANO a 2000,00 y ``n_days`` operaciones diarias (10:00-11:30, 0,1 lotes), TODAS largas
    (``side="buy"``: entran a 2000,00 + ``ask_offset``, es decir, al ask, por encima del high de la barra, y cierran a 2000,00)
    o TODAS cortas (``side="sell"``: entran y cierran a 2000,00). Con ``default_spread`` (texto, en unidades del tickSize, 0,01)
    se añade ese ``defaultSpread`` al ``InstrumentInfo`` del .sqx, como lo escribe SQX. ``extra``: operaciones y barras
    adicionales (``{"trade": dict del writer, "bars": [(ts, o, h, l, c)]}``). El OHLC va a la carpeta de AXI de los tests."""
    days = pd.bdate_range("2023-01-02", periods=n_days)
    odir = tmp_path / "ohlc_axi"
    odir.mkdir(parents=True, exist_ok=True)
    comm = symbols.resolve("XAUUSD_axi").commission["per_lot_rt"]
    rows, trades = [], []
    for d in days:
        ot = pd.Timestamp(d) + pd.Timedelta(hours=10)
        ct = ot + pd.Timedelta(minutes=90)
        rows += [(ts, FLAT_PX, FLAT_PX, FLAT_PX, FLAT_PX) for ts in pd.date_range(ot, ct, freq="min")]
        buy = side == "buy"
        open_px = _f32(FLAT_PX + ask_offset) if buy else _f32(FLAT_PX)
        close_px = _f32(FLAT_PX)
        sl_px = _f32(open_px - 5.0) if buy else _f32(open_px + 5.0)
        gross = (close_px - open_px) * (1.0 if buy else -1.0) * 100.0 * LOTS
        trades.append(dict(open=ot.to_pydatetime(), close=ct.to_pydatetime(), lots=LOTS, open_px=open_px,
                           close_px=close_px, sl_px=sl_px, pnl=gross - comm * LOTS, mae=5.0, flag=1))
    for ex in extra or ():
        trades.append(ex["trade"])
        rows += list(ex["bars"])
    if write_ohlc:
        with open(odir / "XAUUSD_axi.csv", "w", encoding="utf-8") as fh:
            for ts, o, h, l, c in sorted(rows, key=lambda r: r[0]):
                fh.write(f"{ts:%Y.%m.%d},{ts:%H:%M},{o:.2f},{h:.2f},{l:.2f},{c:.2f},100\n")
    p = write_sqx(tmp_path / "extra" / "bank" / name, trades, symbol="XAUUSD_axi", instrument="XAUUSD_mt5_axi",
                  pv=100.0, tick=0.01, capital=ACCOUNT)
    if default_spread is not None:
        patch_instrument_info(p, defaultSpread=default_spread)
    return str(p)


def _load(paths, account_size: float = ACCOUNT, **kw) -> dict:
    return main.portfolio_load(main.PortfolioLoadRequest(legs=[str(p) for p in paths], account_size=account_size, **kw))


F_KW = dict(f_lo=0.01, f_hi=0.04, f_step=0.01)


def _sim(pid: str, **kw) -> dict:
    base = dict(portfolio_id=pid, stage="seed", n_paths=200, **F_KW)
    base.update(kw)
    return main.simulate_axi(main.AxiSimRequest(**base))


def _f_grid(f_lo=F_KW["f_lo"], f_hi=F_KW["f_hi"], f_step=F_KW["f_step"]) -> np.ndarray:
    return np.arange(f_lo, f_hi + f_step / 2, f_step)       # la misma expresión que el endpoint


def _daily_leg(path: str, **kw) -> dict:
    """La pata como la recarga el endpoint (cota diaria por el MAE; caché compartida con la carga)."""
    return main._load_leg(path, ACCOUNT, F_REF, with_minutes=True, synth_mode="daily", **kw)


def _dates(ot, ct) -> list[dt.date]:
    return [d.date() for d in pd.date_range(pd.Timestamp(ot).normalize(), pd.Timestamp(ct).normalize(), freq="D")]


def _floor_px(leg: dict, i: int) -> float:
    """El suelo de la operación ``i`` (E19): ``−adverse_per_lot/pv − spread si es corto``, con el spread que la serie USÓ
    (E20: ``minute_diag["spread_px_usado"]``; en estos sintéticos, el de la ficha)."""
    short = not bool(leg["s"].is_buy[i])
    spread = float(leg["minute_diag"]["spread_px_usado"])
    return -float(leg["fleg"].adverse_per_lot[i]) / float(leg["s"].pv) - (spread if short else 0.0)


# ---------------------------------------------------------------------------
# AC3.1: 200, cierre de probabilidad y campos raw == run_portfolio_mc con tolerancia cero
# ---------------------------------------------------------------------------
RAW_FIELDS = [
    "p_pass_raw", "p_pass_raw_ci", "p_ruin_raw", "p_ruin_raw_ci",
    "median_days_to_target", "p90_days_to_target", "median_trades_to_target", "p90_trades_to_target",
    "median_max_dd", "p10_max_dd", "terminal_p05", "terminal_p50", "terminal_p95",
]


def _eq(a, b) -> bool:
    """Igualdad exacta con NaN/None equivalentes (``_jsonable``: NaN -> null), como ``check_axi.py``."""
    if a is None and b is None:
        return True
    if a is None:
        return isinstance(b, float) and math.isnan(b)
    if b is None:
        return isinstance(a, float) and math.isnan(a)
    return a == b


def _independent_strats(paths, f_grid: np.ndarray) -> dict:
    """Las ``StrategyMC`` que el motor recibiría, construidas AQUÍ sin los helpers del endpoint: valle diario con un
    bucle de Python plano sobre la serie de cada operación y lotes con ``fixed_risk_micros`` (paso y mínimo de la ficha)."""
    strats = {}
    for path in paths:
        leg = _daily_leg(path)
        s = leg["s"]
        valle = {}
        for i in range(s.n_trades):
            ts, vals, _ot, _ct = leg["minute_valle"][i]
            d: dict = {}
            for t, v in zip(ts, vals):
                day = pd.Timestamp(t).date()
                d[day] = min(d.get(day, math.inf), float(v))
            valle[i] = d
        lots, _ = fixed_risk_micros(leg["fleg"].risk_per_micro, f_grid, ACCOUNT, cap=None, lot_step=0.01, min_lot=0.01)
        strats[leg["name"]] = attach_lots(dataclasses.replace(s, valle=valle), lots)
    return strats


def test_ac3_1_simulate_axi_200_cierre_de_probabilidad_y_raw_igual_al_motor_con_tolerancia_cero(api, tmp_path):
    w = make_axi_world(tmp_path)
    load = api.post("/api/portfolio/load", json={"legs": w.paths, "account_size": ACCOUNT}).json()
    assert load["ok"], load
    assert load["guards"]["bloques"]["status"] == "ok" and load["guards"]["flotante_m1"]["status"] == "ok"
    body = dict(portfolio_id=load["portfolio_id"], stage="seed", n_paths=200, **F_KW)
    r = api.post("/api/simulate/axi", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["meta"]["n_blocks"] >= 30 and out["meta"]["crossing"] == 0 and len(out["reports"]) == 4
    # cierre de probabilidad por f
    for rep in out["reports"]:
        assert abs(rep["p_pass"] + rep["p_ruin"] + rep["p_sin_resolver"] - 1.0) <= 1e-9, rep["f"]
    # fidelidad raw con tolerancia cero frente a run_portfolio_mc (la construcción de check_axi.py): mismos bloques,
    # lotes, tamaño de bloque, semilla (la de siempre, 42) y barreras de etapa
    f_grid = _f_grid()
    _, baseline, pass_line, ruin_line = main._stage_barriers("seed")
    direct = run_portfolio_mc(_independent_strats(w.paths, f_grid), f_grid, n_paths=200,
                              block_size=out["meta"]["block_size"], oos_start=None, pass_line=pass_line,
                              ruin_line=ruin_line, baseline=baseline, horizon_days=None)
    by_f = {rep["f"]: rep for rep in out["reports"]}
    n_cmp = 0
    for rd in direct.reports:
        api_rep = by_f[float(rd.f)]
        mapped = {"p_pass_raw": rd.p_pass, "p_pass_raw_ci": list(rd.p_pass_ci), "p_ruin_raw": rd.p_ruin,
                  "p_ruin_raw_ci": list(rd.p_ruin_ci), "median_days_to_target": rd.median_days_to_target,
                  "p90_days_to_target": rd.p90_days_to_target, "median_trades_to_target": rd.median_trades_to_target,
                  "p90_trades_to_target": rd.p90_trades_to_target, "median_max_dd": rd.median_max_dd,
                  "p10_max_dd": rd.p10_max_dd, "terminal_p05": rd.terminal_p05, "terminal_p50": rd.terminal_p50,
                  "terminal_p95": rd.terminal_p95}
        for k in RAW_FIELDS:
            assert _eq(api_rep[k], mapped[k]), f"f={rd.f:g} {k}: motor={mapped[k]!r} api={api_rep[k]!r}"
            n_cmp += 1
    assert n_cmp == 4 * len(RAW_FIELDS)
    # el escenario no es trivial: hay caminos que pasan o se arruinan (si todo fuera 0, la igualdad no probaría nada)
    assert any(rep["p_pass_raw"] > 0 or rep["p_ruin_raw"] > 0 for rep in out["reports"])
    assert any(rd.median_max_dd != 0 for rd in direct.reports)


def test_ac3_1_con_corte_oos_tambien_cuadra_con_el_motor(tmp_path):
    w = make_axi_world(tmp_path)
    pid = _load(w.paths)["portfolio_id"]
    out = _sim(pid, oos_start="2023-02-01")
    f_grid = _f_grid()
    _, baseline, pass_line, ruin_line = main._stage_barriers("seed")
    direct = run_portfolio_mc(_independent_strats(w.paths, f_grid), f_grid, n_paths=200,
                              block_size=out["meta"]["block_size"], oos_start=pd.Timestamp("2023-02-01"),
                              pass_line=pass_line, ruin_line=ruin_line, baseline=baseline, horizon_days=None)
    assert out["meta"]["oos_start"] == "2023-02-01" and out["meta"]["n_blocks"] < out["meta"]["n_blocks_full"]
    for rd, rep in zip(direct.reports, out["reports"]):
        assert _eq(rep["p_pass_raw"], rd.p_pass) and _eq(rep["p_ruin_raw"], rd.p_ruin)
        assert _eq(rep["terminal_p50"], rd.terminal_p50) and _eq(rep["median_max_dd"], rd.median_max_dd)


# ---------------------------------------------------------------------------
# AC3.2: valle diario, operaciones sin barras, pata sin OHLC, cobertura
# ---------------------------------------------------------------------------
def test_ac3_2_el_valle_diario_es_el_minimo_por_fecha_del_servidor_de_la_serie(tmp_path, spy):
    """Propiedad: para TODA operación de TODA pata, ``valle[i]`` es ``{fecha: min(vals de esa fecha)}`` de su serie por
    minuto (calculada aquí con un bucle plano, no con la agrupación de pandas del endpoint)."""
    w = make_axi_world(tmp_path, multiday=True, drop_bars_of_trade=7)
    pid = _load(w.paths)["portfolio_id"]
    _sim(pid)
    assert list(spy["strats"]) == ["AXI XAU.sqx", "AXI EUR.sqx"]
    n_checked = n_multi_date = 0
    for path in w.paths:
        leg = _daily_leg(path)
        s = spy["strats"][leg["name"]]
        assert s is not leg["s"] and s.valle != {}                       # una COPIA con valle
        assert set(s.valle) == set(range(s.n_trades))
        for i in range(s.n_trades):
            ts, vals, _ot, _ct = leg["minute_valle"][i]
            expected: dict = {}
            for t, v in zip(ts, vals):
                d = pd.Timestamp(t).date()
                expected[d] = min(expected.get(d, math.inf), float(v))
            assert s.valle[i] == expected
            n_checked += 1
            n_multi_date += len(expected) > 1
    assert n_checked == 2 * N_DAYS + 2 and n_multi_date == 2           # solo las dos operaciones de varios días


def test_ac3_2_una_operacion_sin_barras_tiene_puntos_daily_con_el_suelo_en_cada_fecha_de_ot_a_ct(tmp_path):
    w = make_axi_world(tmp_path, multiday=True, drop_bars_of_trade=7)
    leg = _daily_leg(w.legs["XAUUSD_axi"])
    assert leg["minute_reason"] is None and leg["minute_diag"]["coherencia"] >= main.SQX_MINUTE_COHERENCE_MIN
    assert leg["minute_diag"]["sinteticas"] == 3 and leg["minute_diag"]["n_trades"] == N_DAYS + 2   # 2 multidía + 1
    s = leg["s"]
    spans = {i: (pd.Timestamp(s.open_t[i]), pd.Timestamp(s.close_t[i])) for i in range(s.n_trades)
             if pd.Timestamp(s.close_t[i]).date() != pd.Timestamp(s.open_t[i]).date()}
    assert len(spans) == 2
    n_points = 0
    for i, (ot, ct) in spans.items():
        ts, vals, ot_s, ct_s = leg["minute_valle"][i]
        stamps = [pd.Timestamp(t) for t in ts]
        assert [t.date() for t in stamps] == _dates(ot, ct)                             # un punto por fecha de [ot, ct]
        assert stamps[0] == ot and all(t == t.normalize() for t in stamps[1:])          # max(ot, inicio de ese día)
        assert (ot_s, ct_s) == (ot, ct)
        np.testing.assert_allclose(vals, _floor_px(leg, i), rtol=0, atol=1e-12)         # el suelo, en TODAS las fechas
        n_points += len(vals)
    assert n_points == 4 + 3
    # la operación diaria a la que se quitaron las barras: una sola fecha, el suelo (E19: incluye comisión)
    j = next(i for i in range(s.n_trades) if pd.Timestamp(s.open_t[i]) == pd.Timestamp(w.trades["XAUUSD_axi"][7]["open"]))
    ts, vals, *_ = leg["minute_valle"][j]
    assert len(ts) == 1 and pd.Timestamp(ts[0]) == pd.Timestamp(s.open_t[j])
    np.testing.assert_allclose(vals, _floor_px(leg, j), rtol=0, atol=1e-12)


def test_ac3_2_golden_a_mano_de_la_serie_daily_sin_barras():
    """Ejemplo a mano de ``_sqx_minute_series(..., synth_mode="daily")`` sin OHLC (E19: el suelo incluye comisión y swap).

    pv = 100, spread = 0,08. Largo de 0,1 lotes del 2023-01-02 22:30 al 2023-01-04 01:00 (3 fechas): MAE 12,00 y comisión
    0,45 -> adverse/lote = 124,5 -> suelo = −124,5/100 = −1,245. Corto de 0,1 lotes del 2023-01-09 09:00 al 2023-01-09 17:00
    (1 fecha): MAE 9,00, comisión 0,45 y swap-cargo 0,30 -> adverse/lote = 97,5 -> suelo = −0,975 − 0,08 = −1,055."""
    df = pd.DataFrame({
        "open_ts": pd.to_datetime(["2023-01-02 22:30", "2023-01-09 09:00"]),
        "close_ts": pd.to_datetime(["2023-01-04 01:00", "2023-01-09 17:00"]),
        "lots": [0.1, 0.1], "open_px": [2000.0, 2000.0], "is_buy": [True, False],
        "mae_usd": [12.0, 9.0], "comm_usd": [0.45, 0.45], "swap_usd": [0.0, -0.30]})
    fleg = types.SimpleNamespace(adverse_per_lot=np.array([(12.0 + 0.45) / 0.1, (9.0 + 0.45 + 0.30) / 0.1]))
    series, diag = main._sqx_minute_series(df, {"pv": 100.0, "tick_size": 0.01}, fleg, None, 0.08, lambda s: s,
                                           synth_mode="daily")
    ts0, v0, ot0, ct0 = series[0]
    assert [str(t) for t in ts0] == ["2023-01-02T22:30:00.000000000", "2023-01-03T00:00:00.000000000",
                                     "2023-01-04T00:00:00.000000000"]
    np.testing.assert_allclose(v0, [-1.245] * 3, rtol=0, atol=1e-12)
    assert (ot0, ct0) == (pd.Timestamp("2023-01-02 22:30"), pd.Timestamp("2023-01-04 01:00"))
    ts1, v1, *_ = series[1]
    assert [str(t) for t in ts1] == ["2023-01-09T09:00:00.000000000"]
    np.testing.assert_allclose(v1, [-1.055], rtol=0, atol=1e-12)
    assert diag["n_trades"] == 2 and diag["sinteticas"] == 2 and diag["frac_largos_bid"] is None
    # con "point" (lo de hoy) es UN punto en ot con el mismo valor
    p_series, p_diag = main._sqx_minute_series(df, {"pv": 100.0, "tick_size": 0.01}, fleg, None, 0.08, lambda s: s)
    assert [len(p_series[i][0]) for i in (0, 1)] == [1, 1] and p_diag["sinteticas"] == 2
    np.testing.assert_allclose([p_series[0][1][0], p_series[1][1][0]], [-1.245, -1.055], rtol=0, atol=1e-12)
    # y el valle diario de la operación de 3 fechas: una entrada por fecha con el suelo
    assert main._daily_valley(ts0, v0) == pytest.approx({dt.date(2023, 1, 2): -1.245, dt.date(2023, 1, 3): -1.245,
                                                          dt.date(2023, 1, 4): -1.245}, abs=1e-12)


def test_ac3_2_golden_a_mano_del_valle_diario():
    ts = np.array(["2023-01-02T10:00", "2023-01-02T10:01", "2023-01-03T00:00", "2023-01-03T09:00"],
                  dtype="datetime64[ns]")
    got = main._daily_valley(ts, np.array([-1.0, -3.0, -2.0, -0.5]))
    assert got == {dt.date(2023, 1, 2): -3.0, dt.date(2023, 1, 3): -2.0}
    assert all(type(k) is dt.date and type(v) is float for k, v in got.items())


def test_ac3_2_una_pata_sin_ohlc_es_entera_daily_y_la_guarda_flotante_m1_da_warn(tmp_path):
    w = make_axi_world(tmp_path, write_ohlc=("XAUUSD_axi",), multiday=False)         # el OHLC de EURUSD_axi no existe
    res = _load(w.paths)
    assert res["ok"]                                                                    # nunca es error de carga
    g = res["guards"]["flotante_m1"]
    assert g["status"] == "warn" and "AXI EUR.sqx" in g["detail"] and "AXI XAU.sqx" not in g["detail"]
    assert g["detail"].startswith("1/2 pata(s) sin OHLC M1 coherente")
    assert "la lente AXI usará la cota diaria por el MAE (más pesimista)" in g["detail"]
    eur = _daily_leg(w.legs["EURUSD_axi"])
    assert eur["minute_valle"] is not None and "cota diaria por el MAE" in eur["minute_reason"]
    assert "EURUSD_axi.csv no disponible o ilegible" in eur["minute_reason"]
    d = eur["minute_diag"]
    assert d["cota_diaria"] is True and d["sinteticas"] == d["n_trades"] == N_DAYS and d["coherencia"] is None
    s = eur["s"]
    for i in range(s.n_trades):                                                         # ENTERAMENTE sintética diaria
        ts, vals, *_ = eur["minute_valle"][i]
        assert [pd.Timestamp(t).date() for t in ts] == _dates(s.open_t[i], s.close_t[i])
        np.testing.assert_allclose(vals, _floor_px(eur, i), rtol=0, atol=1e-12)
    assert _daily_leg(w.legs["XAUUSD_axi"])["minute_reason"] is None                    # la de oro, con su OHLC
    # con "point" (la lente prop firm) es lo de siempre: sin serie y con motivo (sin "cota diaria")
    p = main._load_leg(w.legs["EURUSD_axi"], ACCOUNT, F_REF, with_minutes=True)
    assert p["minute_valle"] is None and "cota diaria" not in p["minute_reason"]
    assert p["minute_reason"] == "OHLC EURUSD_axi.csv no disponible o ilegible en ohlc_axi/"
    # y la simulación la usa: meta/avisos cuentan la pata con cota diaria
    out = _sim(res["portfolio_id"])
    assert "1 de 2 pata(s) usan la cota diaria por el MAE" in out["meta"]["float_note"]
    assert any(x.startswith("Flotante por la cota diaria por el MAE") and "AXI EUR.sqx" in x for x in out["warnings"])


def test_ac3_2_ficha_sin_ohlc_m1_o_sin_spread_px_tambien_cae_a_la_cota_diaria(tmp_path):
    w = make_axi_world(tmp_path, n_days=12)
    leg = _daily_leg(w.legs["XAUUSD_axi"])
    sheet = symbols.resolve("XAUUSD_axi")
    broker = symbols.load_registry().brokers["axi"]
    df, meta = main.read_trade_table_spec(w.legs["XAUUSD_axi"], symbols.reader_spec(sheet))
    to_axis = main._sqx_axis_transform(0.0)
    # ficha OFICIAL sin ohlc_m1: el motivo de siempre + la cota diaria
    s1, d1, r1 = main._sqx_leg_minutes_sheet(df, meta, leg["fleg"], dataclasses.replace(sheet, ohlc_m1=None), broker,
                                             to_axis, synth_mode="daily")
    assert r1 == "sin ohlc_m1 en el YAML de costes para XAUUSD_axi: se usa la cota diaria por el MAE"
    assert len(s1) == 12 and d1["cota_diaria"] is True and d1["sinteticas"] == 12
    # ficha sin spread_px: el OHLC existe, pero sin spread no hay coherencia; la cota diaria no lleva spread en cortos
    s2, d2, r2 = main._sqx_leg_minutes_sheet(df, meta, leg["fleg"], dataclasses.replace(sheet, spread_px=None), broker,
                                             to_axis, synth_mode="daily")
    assert r2 == ("YAML de costes sin typical_spread_points/tick_size para XAUUSD_axi: se usa la cota diaria por el MAE "
                  "(sin spread en los cortos)")
    assert d2["cota_diaria"] is True and d2["ohlc"] == "XAUUSD_axi.csv"
    for i in range(len(s2)):
        np.testing.assert_allclose(s2[i][1], -float(leg["fleg"].adverse_per_lot[i]) / 100.0, rtol=0, atol=1e-12)
    # con "point" las mismas fichas dan (None, None, motivo) como siempre
    assert main._sqx_leg_minutes_sheet(df, meta, leg["fleg"], dataclasses.replace(sheet, ohlc_m1=None), broker,
                                       to_axis) == (None, None, "sin ohlc_m1 en el YAML de costes para XAUUSD_axi")


def test_ac3_2_todas_las_operaciones_de_la_ventana_estan_en_algun_bloque_cobertura_100(tmp_path, spy):
    w = make_axi_world(tmp_path, multiday=True, drop_bars_of_trade=7)
    pid = _load(w.paths)["portfolio_id"]
    out = _sim(pid)
    meta = out["meta"]
    n_total = 2 * N_DAYS + 2
    assert meta["trades_in_window"] == n_total and meta["coverage"] == 1.0 and meta["crossing"] == 0
    assert sum(b.n_unique_trades for b in spy["blocks"]) == n_total
    assert {(int(si), int(loc)) for b in spy["blocks"] for si, loc in zip(b.sym_id, b.trade_local_id)} == {
        (si, i) for si, s in enumerate(spy["strats"].values()) for i in range(s.n_trades)}
    # con un corte OOS la ventana es la posterior al corte y tampoco se pierde ninguna operación
    out2 = _sim(pid, oos_start="2023-02-01")
    assert out2["meta"]["coverage"] == 1.0 and 0 < out2["meta"]["trades_in_window"] < n_total
    assert sum(b.n_unique_trades for b in spy["blocks"]) == out2["meta"]["trades_in_window"]


# ---------------------------------------------------------------------------
# AC3.2bis (E1 + E21): forma del valle
# ---------------------------------------------------------------------------
def test_ac3_2bis_forma_del_valle_claves_date_valores_float_igual_al_minimo_del_dia_y_una_entrada_por_operacion(
        tmp_path, spy):
    """E21: el valor de cada fecha es el MÍNIMO de la serie de ese día (no se exige "≤ 0": puede ser > 0 si la operación
    estuvo en beneficio todo el día; el sintético de este test da <= 0, pero no es un invariante del código)."""
    w = make_axi_world(tmp_path, multiday=True, drop_bars_of_trade=7)
    _sim(_load(w.paths)["portfolio_id"])
    for path in w.paths:
        leg = _daily_leg(path)
        s = spy["strats"][leg["name"]]
        assert len(s.valle) == s.n_trades and set(s.valle) == set(range(s.n_trades))      # una entrada por operación
        key_type = type(s.close_d[0])
        assert key_type is dt.date                                                           # el tipo de ``close_d``
        for i, v in s.valle.items():
            assert v and all(type(d) is dt.date and type(x) is float for d, x in v.items())  # fechas y floats
            ts, vals, *_ = leg["minute_valle"][i]
            by_day: dict = {}
            for t, x in zip(ts, vals):
                by_day.setdefault(pd.Timestamp(t).date(), []).append(float(x))
            assert v == {d: min(xs) for d, xs in by_day.items()}, (leg["name"], i)           # = mínimo de la serie del día
            allowed = set(_dates(s.open_t[i], s.close_t[i]))
            assert set(v) <= allowed, (leg["name"], i)                                       # ⊆ fechas de [ot, ct]
            assert all(type(d) is key_type for d in v)
    # la copia no toca las claves/valores de la cacheada
    assert all(_daily_leg(p)["s"].valle == {} for p in w.paths)


def test_ac3_2bis_e21_el_valle_de_un_dia_en_beneficio_es_positivo_igual_que_el_valle_csv(tmp_path, spy):
    """E21: una operación larga que a las 10:30 del día 1 salta de 2000 a 2010 y se queda ahí todo el día 2 tiene, el día 2,
    un valle POSITIVO (+10 menos la comisión por lote/pv): es el mínimo de la excursión del día, sin recortar a 0, como el
    valle CSV de ``precompute_strategy``. El día 1 (donde entra a 2000) sigue siendo <= 0."""
    p = make_flat_leg(tmp_path, side="sell", n_days=8, extra=[PROFIT_TRADE])
    pid = _load([p])["portfolio_id"]
    _sim(pid, n_paths=100)
    leg = _daily_leg(p)
    s = spy["strats"][leg["name"]]
    j = int(np.argmax(np.array(s.close_t, dtype="datetime64[ns]") - np.array(s.open_t, dtype="datetime64[ns]")))
    day1, day2 = dt.date(2023, 3, 6), dt.date(2023, 3, 7)
    cost = 4.5 / 100.0                                                    # comisión por lote / valor del punto
    assert set(s.valle[j]) == {day1, day2}
    assert s.valle[j][day2] > 0 and s.valle[j][day2] == pytest.approx(10.0 - cost, abs=1e-6)
    assert s.valle[j][day1] <= 0 and s.valle[j][day1] == pytest.approx(-cost, abs=1e-6)
    ts, vals, *_ = leg["minute_valle"][j]
    assert s.valle[j][day2] == min(float(x) for t, x in zip(ts, vals) if pd.Timestamp(t).date() == day2)   # = mínimo del día
    assert float(np.max(vals)) > 9.0                                                    # la serie sí llega arriba
    assert leg["minute_reason"] is None and leg["minute_diag"]["sinteticas"] == 0       # con barras: no es la cota


# ---------------------------------------------------------------------------
# AC3.3: lotes
# ---------------------------------------------------------------------------
def test_ac3_3_los_lotes_son_fixed_risk_micros_con_el_paso_y_el_minimo_de_la_ficha_y_a_igual_a_account_size(tmp_path, spy):
    w = make_axi_world(tmp_path, multiday=True)
    pid = _load(w.paths)["portfolio_id"]
    _sim(pid)
    f_grid = _f_grid()
    want_by = {}
    for path in w.paths:
        leg = _daily_leg(path)
        sheet = symbols.resolve(leg["meta"]["symbol"])
        assert (sheet.lot_step, sheet.min_lot) == (0.01, 0.01)
        want, _ = fixed_risk_micros(leg["fleg"].risk_per_micro, f_grid, ACCOUNT, cap=None, lot_step=0.01, min_lot=0.01)
        want_by[leg["name"]] = want
        got = spy["strats"][leg["name"]].lots
        assert got.shape == (len(f_grid), leg["s"].n_trades)
        np.testing.assert_array_equal(got, want)                                             # exacto, no aproximado
        assert np.all(got >= 0.01) and np.allclose(got / 0.01, np.round(got / 0.01))         # en pasos de 0,01
    # A es el tamaño de cuenta de la CARGA: otra cuenta, otros lotes (X = f·A)
    pid2 = _load(w.paths, account_size=2 * ACCOUNT)["portfolio_id"]
    _sim(pid2)
    leg = main._load_leg(w.paths[0], 2 * ACCOUNT, F_REF, with_minutes=True, synth_mode="daily")
    want2, _ = fixed_risk_micros(leg["fleg"].risk_per_micro, f_grid, 2 * ACCOUNT, cap=None, lot_step=0.01, min_lot=0.01)
    np.testing.assert_array_equal(spy["strats"][leg["name"]].lots, want2)
    assert not np.array_equal(want2, want_by[leg["name"]])


# ---------------------------------------------------------------------------
# AC3.4: I6 y synth_mode="point" == las series de hoy
# ---------------------------------------------------------------------------
def test_ac3_4_i6_la_strategymc_cacheada_sigue_con_valle_vacio_y_la_prop_firm_da_lo_mismo_antes_y_despues(
        tmp_path, monkeypatch):
    """I6: la ``StrategyMC`` cacheada de cada pata la comparte la lente prop firm. Tras correr la lente AXI sigue con
    ``valle == {}`` y la lente prop firm (con las patas de la carga, sin recargar: ``sqx_float="windows"``, eje del
    servidor) da EXACTAMENTE el mismo resultado antes y después. Para poder correr la prop firm sobre una cartera de
    AXI se asigna el bróker de la cartera al perfil ``ftmo_50k`` (``broker_for_profile``): es lo único simulado."""
    real = symbols.broker_for_profile
    monkeypatch.setattr(symbols, "broker_for_profile", lambda key: "axi" if key == "ftmo_50k" else real(key))
    w = make_axi_world(tmp_path, multiday=True)
    res = _load(w.paths)
    pid = res["portfolio_id"]
    pf = main._portfolios[pid]
    assert pf["broker"] == "axi" and all(l["s"].valle == {} for l in pf["legs"])

    def prop():
        out = main.simulate_propfirm(main.PropfirmSimRequest(
            portfolio_id=pid, firm_key="ftmo_50k", f_lo=0.01, f_hi=0.03, f_step=0.01, n_paths=100, n_challenges=3,
            n_sequences=100, sqx_float="windows", day_axis="servidor"))
        return {k: out[k] for k in ("reports", "recommendation")}

    before = json.dumps(prop(), sort_keys=True)
    axi_1 = _sim(pid)
    # el endpoint de AXI recargó con "daily": las patas de la carga SON esas mismas entradas de la caché (acierto)
    assert all(_daily_leg(p) is l for p, l in zip(w.paths, pf["legs"]))
    assert all(l["s"].valle == {} for l in pf["legs"])
    assert all(v["s"].valle == {} for v in main._leg_cache.values() if v.get("kind") == main.KIND_SQX)
    after = json.dumps(prop(), sort_keys=True)
    assert before == after
    # y al revés: la lente AXI no depende de que la prop firm haya puesto sus lotes en la StrategyMC cacheada
    axi_2 = _sim(pid)
    for a, b in ((axi_1, axi_2),):
        a["meta"].pop("elapsed_s"), b["meta"].pop("elapsed_s")
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# hash de las series por minuto de HOY (sin synth_mode: "point" implícito), tomado en el paso 0 de la F3 sobre los
# fixtures de test_cfd_minute_float con el código de la F2 (numpy 2.4.4). Los de big_specs y sin_barras dependen del RNG
# sembrado de numpy: si el flujo del RNG cambia en otra versión, el test se omite en vez de dar un falso fallo.
PASO0_HASH = {
    "big_specs_eje_ftmo": "efa9b81d76064230ad5f4539da3b0e5f23ac3404074baed0b9625ddf155742c8",
    "big_specs_eje_servidor": "59ba40bed0ff149a4ade06aed6c3a2eae2e673afa2b5daca40e9b295caa0f520",
    "b13_recorte": "c4d7402e693ccfe546d651da886a98eb741b32b113b9198c70e2d397ea3a5405",
    "sin_barras": "62f989e4234f6701f2aef9e82b44ce69f94b6232304d9157d00cf8d4712fbb0d",
}
RNG_CANARY = [0.1257302210933933, -0.1321048632913019]        # np.random.default_rng(0).normal(size=2)


E20_DIAG_KEYS = ("spread_px_usado", "spread_fuente")        # E20 las añade al ``minute_diag``; el hash del paso 0 no las tiene


def _digest_legs(legs) -> str:
    """sha256 de las series por minuto (ts, vals, ot, ct) y el diagnóstico de una lista de patas cargadas. El diagnóstico
    se hashea SIN las dos claves que añadió la enmienda E20 (``spread_px_usado``, ``spread_fuente``): el hash del paso 0 se
    tomó antes de ellas y lo que demuestra es que las series y el resto del diagnóstico no cambian."""
    h = hashlib.sha256()
    for leg in legs:
        mv = leg["minute_valle"]
        for i in sorted(mv):
            ts, vals, ot, ct = mv[i]
            h.update(np.asarray(ts, dtype="datetime64[ns]").view("int64").tobytes())
            h.update(np.asarray(vals, dtype="float64").tobytes())
            h.update(np.int64(ot.value).tobytes())
            h.update(np.int64(ct.value).tobytes())
        diag = {k: v for k, v in leg["minute_diag"].items() if k not in E20_DIAG_KEYS}
        h.update(json.dumps(diag, sort_keys=True).encode())
    return h.hexdigest()


def _cfd_minute_float_fixtures():
    spec = importlib.util.spec_from_file_location("_tst_cfd_minute_float_fixtures",
                                                  Path(__file__).with_name("test_cfd_minute_float.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("explicit", [False, True], ids=["por_defecto", "synth_mode_point_explicito"])
def test_ac3_4_synth_mode_point_da_las_series_de_hoy_hash_del_paso_0(tmp_path, explicit):
    if np.random.default_rng(0).normal(size=2).tolist() != RNG_CANARY:
        pytest.skip("el flujo del RNG de numpy es otro: el hash del paso 0 no es comparable")
    t = _cfd_minute_float_fixtures()
    kw = dict(synth_mode="point") if explicit else {}
    got = {}

    def fresh():
        main._leg_cache.clear()
        main._sqx_ohlc_cache.clear()
        main._sqx_symbol_cache.clear()

    fresh()
    wa = t.make_world(tmp_path / "a", t.big_specs())
    legs_a = [main._load_sqx_leg(p, t.ACCOUNT, None, axis_hours=-1.0, with_minutes=True, **kw) for p in wa.legs]
    got["big_specs_eje_ftmo"] = _digest_legs(legs_a)
    main._leg_cache.clear()
    legs_a2 = [main._load_sqx_leg(p, t.ACCOUNT, None, axis_hours=0.0, with_minutes=True, **kw) for p in wa.legs]
    got["big_specs_eje_servidor"] = _digest_legs(legs_a2)
    fresh()
    wb = t._b13_world(tmp_path / "b")
    leg_b = main._load_sqx_leg(wb.legs[0], t.ACCOUNT, None, axis_hours=0.0, with_minutes=True, **kw)
    got["b13_recorte"] = _digest_legs([leg_b])
    fresh()
    gap = (datetime(2023, 1, 9, 10, 0) - timedelta(minutes=1), datetime(2023, 1, 9, 12, 30))
    wc = t.make_world(tmp_path / "c", [t.big_specs(40, 1)[0]], drop_ranges=[gap])
    leg = main._load_sqx_leg(wc.legs[0], t.ACCOUNT, None, axis_hours=-1.0, with_minutes=True, **kw)
    assert leg["minute_diag"]["sinteticas"] == 1                       # la operación sin barras: UN punto en ot
    assert sum(len(v[0]) == 1 for v in leg["minute_valle"].values()) == 1
    got["sin_barras"] = _digest_legs([leg])
    assert got == PASO0_HASH
    # E20: estos .sqx sintéticos no traen ``defaultSpread`` (el writer no lo escribe): el spread es el de la ficha, el MISMO
    # número que antes (la ruta de FTMO no cambia), y el diagnóstico lo dice
    ficha = symbols.resolve("EURUSD_ftmo").spread_px
    for lg in [*legs_a, *legs_a2, leg_b, leg]:
        assert lg["minute_diag"]["spread_fuente"] == "ficha" and lg["minute_diag"]["spread_px_usado"] == ficha


def test_ac3_4_en_un_mundo_con_barras_en_todas_las_operaciones_daily_es_igual_que_point(tmp_path):
    """Con OHLC en todas las operaciones no hay nada sintético que cambiar: las series de ``"daily"`` y de ``"point"``
    son idénticas bit a bit (``"daily"`` solo toca lo que no tiene barras)."""
    w = make_axi_world(tmp_path, n_days=30)
    for path in w.paths:
        p = main._load_leg(path, ACCOUNT, F_REF, with_minutes=True, synth_mode="point")
        d = main._load_leg(path, ACCOUNT, F_REF, with_minutes=True, synth_mode="daily")
        assert p is not d and p["minute_reason"] is None and d["minute_reason"] is None
        assert _digest_legs([p]) == _digest_legs([d]) and p["minute_diag"]["sinteticas"] == 0


def test_ac3_4_synth_mode_entra_en_la_clave_de_cache_y_uno_invalido_falla_en_voz_alta(tmp_path):
    w = make_axi_world(tmp_path, n_days=12)
    p1 = main._load_leg(w.paths[0], ACCOUNT, F_REF, with_minutes=True)
    assert main._load_leg(w.paths[0], ACCOUNT, F_REF, with_minutes=True, synth_mode="point") is p1
    d1 = main._load_leg(w.paths[0], ACCOUNT, F_REF, with_minutes=True, synth_mode="daily")
    assert d1 is not p1 and main._load_leg(w.paths[0], ACCOUNT, F_REF, with_minutes=True, synth_mode="daily") is d1
    # sin series por minuto el modo no cuenta (no duplica entradas de la caché)
    n = main._load_leg(w.paths[0], ACCOUNT, F_REF)
    assert main._load_leg(w.paths[0], ACCOUNT, F_REF, synth_mode="daily") is n and "minute_valle" not in n
    with pytest.raises(ValueError, match="synth_mode"):
        main._load_leg(w.paths[0], ACCOUNT, F_REF, with_minutes=True, synth_mode="semanal")
    with pytest.raises(ValueError, match="synth_mode"):
        main._sqx_minute_series(pd.DataFrame(), {"pv": 1.0}, None, None, 0.0, lambda s: s, synth_mode="x")


# ---------------------------------------------------------------------------
# AC3.5: la lente AXI y las carteras .sqx de otro bróker
# ---------------------------------------------------------------------------
def _ftmo_cfd_legs(tmp_path: Path, n: int = 2) -> list[str]:
    out = []
    for k in range(n):
        tr = [dict(open=datetime(2023, 1, 2 + d, 10 + k, 0), close=datetime(2023, 1, 2 + d, 11 + k, 0), lots=0.5,
                   open_px=1.125, close_px=1.1265, sl_px=1.1235, pnl=75.0 - 2.5, mae=10.0, flag=1) for d in range(8)]
        out.append(str(write_sqx(tmp_path / "extra" / "ftmo" / f"EUR F{k}.sqx", tr, symbol="EURUSD_ftmo",
                                 instrument="EURUSD_cfd_ftmo", pv=100000.0, tick=0.0001, capital=50000.0)))
    return out


def _ftmo_futures_legs(tmp_path: Path, n: int = 2) -> list[str]:
    out = []
    for k in range(n):
        tr = [_W.long_win(open=datetime(2023, 1, 2 + d, 10 + k, 0), close=datetime(2023, 1, 2 + d, 11 + k, 0))
              for d in range(8)]
        out.append(str(write_sqx(tmp_path / "extra" / "fut" / f"MNQ F{k}.sqx", tr)))
    return out


@pytest.mark.parametrize("kind, legs_fn, label", [
    ("cfd", _ftmo_cfd_legs, "FTMO CFD (ftmo_cfd)"), ("futuros", _ftmo_futures_legs, "FTMO Futures (ftmo_futures)")])
def test_ac3_5_cartera_sqx_de_ftmo_en_la_lente_axi_es_409_con_sqx_axi_y_el_broker_de_las_patas(
        tmp_path, api, kind, legs_fn, label):
    res = _load(legs_fn(tmp_path))
    pf = main._portfolios[res["portfolio_id"]]
    assert pf["kind"] == main.KIND_SQX and pf["broker"] in ("ftmo_cfd", "ftmo_futures")
    with pytest.raises(HTTPException) as e:
        _sim(res["portfolio_id"])
    assert e.value.status_code == 409
    d = e.value.detail
    assert ".sqx" in d and "AXI" in d and label in d, d
    assert "Axi CFD (axi)" in d                                    # y el bróker que SÍ admite la lente
    r = api.post("/api/simulate/axi", json={"portfolio_id": res["portfolio_id"]})
    assert r.status_code == 409 and ".sqx" in r.json()["detail"] and "AXI" in r.json()["detail"]


def test_ac3_5_una_cartera_sqx_de_axi_se_admite_y_el_par_2d_y_el_diagnostico_por_pata_siguen_en_409(tmp_path):
    w = make_axi_world(tmp_path, n_days=40)
    res = _load(w.paths)
    pid = res["portfolio_id"]
    assert _sim(pid, n_paths=100)["meta"]["broker"] == "axi"                               # la lente AXI sí
    with pytest.raises(HTTPException) as e1:
        main.portfolio_leg_diagnostics(pid, main.LegDiagRequest(f=0.01))
    assert e1.value.status_code == 409 and ".sqx" in e1.value.detail
    with pytest.raises(HTTPException) as e2:       # el par 2D de las patas CFD pasa por el setup prop firm: bróker distinto
        main.simulate_propfirm_pair(main.PropfirmSimRequest(portfolio_id=pid, firm_key="ftmo_50k"))
    assert e2.value.status_code == 409 and "Las patas son de Axi CFD (axi)" in e2.value.detail
    with pytest.raises(HTTPException) as e3:       # y la lente prop firm 1D igual: una cartera de AXI nunca encaja con FTMO
        main.simulate_propfirm(main.PropfirmSimRequest(portfolio_id=pid, firm_key="ftmo_50k"))
    assert e3.value.status_code == 409 and "Las patas son de Axi CFD (axi)" in e3.value.detail


def test_ac3_5_sin_ninguna_pata_cargada_el_409_es_el_de_sin_patas_validas(tmp_path):
    raro = write_sqx(tmp_path / "extra" / "bank" / "RARO.sqx", [_W.long_win()], symbol="SIMBOLO_RARO", instrument="X",
                     pv=2.0, tick=0.25)
    res = _load([raro])
    assert not res["ok"] and main._portfolios[res["portfolio_id"]]["broker"] is None
    with pytest.raises(HTTPException) as e:
        _sim(res["portfolio_id"])
    assert e.value.status_code == 409 and "no tiene patas válidas" in e.value.detail


def test_ac3_5_portfolio_desconocido_sigue_siendo_404():
    with pytest.raises(HTTPException) as e:
        _sim("nope")
    assert e.value.status_code == 404


def test_ac3_5_el_broker_de_la_lente_sale_de_brokers_yaml_no_del_id_axi(tmp_path, monkeypatch):
    """La admisión mira ``fichas.lens_broker("axi")``: si ``lenses.axi`` fuera otro bróker, la cartera de ese bróker
    entraría y la de ``axi`` sería la rechazada."""
    w = make_axi_world(tmp_path, n_days=12)
    pid = _load(w.paths)["portfolio_id"]
    monkeypatch.setattr(symbols, "lens_broker", lambda lens: "ftmo_cfd")
    with pytest.raises(HTTPException) as e:
        _sim(pid, n_paths=100)
    assert e.value.status_code == 409 and "Axi CFD (axi)" in e.value.detail and "FTMO CFD (ftmo_cfd)" in e.value.detail
    monkeypatch.setattr(symbols, "lens_broker", lambda lens: None)
    with pytest.raises(HTTPException) as e:
        _sim(pid, n_paths=100)
    assert e.value.status_code == 409 and "no tiene bróker asignado" in e.value.detail


# ---------------------------------------------------------------------------
# AC3.6: meta
# ---------------------------------------------------------------------------
def test_ac3_6_meta_de_axi_con_sqx(tmp_path, monkeypatch):
    # ni ``_csv_minute_series`` ni ``_mae_usage`` (solo CSV) se llaman con .sqx
    def boom(*a, **k):
        raise AssertionError("solo CSV: no debe llamarse con patas .sqx")

    monkeypatch.setattr(main, "_csv_minute_series", boom)
    monkeypatch.setattr(main, "_mae_usage", boom)
    w = make_axi_world(tmp_path, multiday=True, drop_bars_of_trade=7)
    res = _load(w.paths)
    # la guarda ``s.minute_valle`` de las patas CSV no aplica: la StrategyMC de una pata .sqx nunca la lleva
    assert all(not l["s"].minute_valle for l in main._portfolios[res["portfolio_id"]]["legs"])
    out = _sim(res["portfolio_id"])
    meta = out["meta"]
    assert meta["sizing"] == main.SQX_SIZING_MODE == "riesgo fijo (Opción A)"
    assert meta["sizing_base"] == main._sizing_base(main.KIND_SQX) == "tamaño de cuenta (sin componer)"
    assert meta["float_mode"] == "minute" and meta["broker"] == "axi"
    assert [(l["pata"], l["símbolo"], l["modo"], l["trades"]) for l in meta["legs"]] == [
        ("AXI XAU.sqx", "XAUUSD_axi", "riesgo fijo (Opción A)", N_DAYS + 2),
        ("AXI EUR.sqx", "EURUSD_axi", "riesgo fijo (Opción A)", N_DAYS)]
    assert meta["trades_in_window"] == 2 * N_DAYS + 2 and meta["coverage"] == 1.0
    note = meta["float_note"]
    assert "Las 2 pata(s) tienen OHLC M1 coherente" in note and "NETO de comisión" in note
    assert "3 operación(es) sin barras M1 dentro de patas con OHLC usan también la cota diaria por el MAE" in note
    # E20: estos .sqx sintéticos no traen defaultSpread, así que el spread de la serie es el de la ficha, y la nota lo dice
    assert "Spread de la serie (coherencia con el OHLC, cortos y cota diaria): 2 de 2 el de la ficha." in note
    assert "n_mae_bounded" not in meta and "n_mae_zero" not in meta                 # los contadores son de las CSV
    assert meta["float_convention"] == main._AXI_FLOAT_CONVENTION_MINUTE
    # la suposición de política de la interfaz no dice que las operaciones sin M1 quedan fuera de los bloques (con .sqx
    # entran todas): solo cambia esa cláusula del texto de siempre
    assert main._AXI_POLICY_MINUTE_SQX != main._AXI_POLICY_MINUTE
    assert "los trades sin cobertura M1 no entran en los bloques" in main._AXI_POLICY_MINUTE
    assert "los trades sin cobertura M1 no entran en los bloques" not in meta["policy_assumption"]
    assert "cota diaria por el MAE y entran en los bloques, ver float_note" in meta["policy_assumption"]
    assert meta["policy_assumption"].replace("las operaciones sin barras M1 usan su cota diaria por el MAE y entran en "
                                             "los bloques, ver float_note", "los trades sin cobertura M1 no entran en "
                                             "los bloques, ver coverage") == main._AXI_POLICY_MINUTE


def test_ac3_6_la_nota_cuenta_las_patas_con_cota_diaria_y_el_porque(tmp_path):
    w = make_axi_world(tmp_path, write_ohlc=(), n_days=40)                         # ninguna pata tiene OHLC
    out = _sim(_load(w.paths)["portfolio_id"], n_paths=100)
    note = out["meta"]["float_note"]
    assert "2 de 2 pata(s) usan la cota diaria por el MAE porque no tienen OHLC M1 coherente" in note
    assert "AXI XAU.sqx (OHLC XAUUSD_axi.csv no disponible o ilegible en ohlc_axi/)" in note
    assert "AXI EUR.sqx (OHLC EURUSD_axi.csv no disponible o ilegible en ohlc_axi/)" in note
    assert "más pesimista" in note and out["meta"]["coverage"] == 1.0
    assert sum("cota diaria por el MAE" in x for x in out["warnings"]) == 1


def test_ac3_6_el_flag_joint_de_investigacion_tambien_corre_con_sqx(tmp_path, monkeypatch):
    w = make_axi_world(tmp_path, n_days=40)
    pid = _load(w.paths)["portfolio_id"]
    monkeypatch.setattr(main, "AXI_FLOAT_MODE", "joint")
    out = _sim(pid, n_paths=100)
    assert out["meta"]["float_mode"] == "joint" and "modo joint" in out["meta"]["float_note"]
    assert out["meta"]["float_convention"].startswith("joint-m1")
    monkeypatch.setattr(main, "AXI_FLOAT_MODE", "minuto")                          # un valor mal escrito falla en voz alta
    with pytest.raises(ValueError, match="AXI_FLOAT_MODE"):
        _sim(pid, n_paths=100)


# ---------------------------------------------------------------------------
# E20: el spread de la serie por minuto es el del backtest (defaultSpread · tickSize del propio .sqx)
# ---------------------------------------------------------------------------
COST_PX = 4.5 / 100.0            # comisión por lote / valor del punto del oro de AXI: lo que la serie resta en cada minuto


def test_e20_a_largos_al_ask_la_coherencia_usa_el_spread_del_sqx_y_la_ficha_solo_si_el_sqx_no_lo_trae(tmp_path):
    """Largos que entran al ASK, 0,15 $ por encima del high de barras planas (lo que pasa con el oro de AXI): con el
    spread de la ficha (0,08; tolerancia 0,08 + 2·0,01 = 0,10) ninguna entrada cuadra y la pata cae a la cota diaria; con
    ``defaultSpread = 18`` en el .sqx (18 · 0,01 = 0,18; tolerancia 0,20) cuadra todo y hay serie de verdad."""
    ficha = symbols.resolve("XAUUSD_axi")
    assert (ficha.spread_px, ficha.tick_size) == (0.08, None)                       # la ficha oficial de AXI
    sin = make_flat_leg(tmp_path, side="buy", ask_offset=0.15, name="sin.sqx")                       # sin defaultSpread
    con = make_flat_leg(tmp_path, side="buy", ask_offset=0.15, default_spread="18", name="con.sqx")
    d = _daily_leg(sin)
    assert d["minute_diag"]["coherencia"] == 0.0
    assert "no cuadra" in d["minute_reason"] and "cota diaria por el MAE" in d["minute_reason"]
    assert (d["minute_diag"]["spread_fuente"], d["minute_diag"]["spread_px_usado"]) == ("ficha", 0.08)
    assert main._load_leg(sin, ACCOUNT, F_REF, with_minutes=True)["minute_valle"] is None            # "point": lo de hoy
    d = _daily_leg(con)
    assert d["minute_reason"] is None and d["minute_diag"]["coherencia"] == 1.0 and d["minute_diag"]["sinteticas"] == 0
    assert d["minute_diag"]["spread_fuente"] == "sqx" and d["minute_diag"]["spread_px_usado"] == 18.0 * 0.01
    assert main._load_leg(con, ACCOUNT, F_REF, with_minutes=True)["minute_valle"] is not None
    # la guarda de la carga y la nota de la simulación lo reflejan
    g_sin, g_con = _load([sin]), _load([con])
    assert g_sin["guards"]["flotante_m1"]["status"] == "warn"
    assert "cota diaria por el MAE" in g_sin["guards"]["flotante_m1"]["detail"]
    assert g_con["guards"]["flotante_m1"]["status"] == "ok"
    n_con = _sim(g_con["portfolio_id"], n_paths=100)["meta"]["float_note"]
    assert "Las 1 pata(s) tienen OHLC M1 coherente" in n_con
    assert "1 de 1 pata(s) usan el del backtest (defaultSpread del .sqx)" in n_con
    n_sin = _sim(g_sin["portfolio_id"], n_paths=100)["meta"]["float_note"]
    assert "1 de 1 pata(s) usan la cota diaria por el MAE" in n_sin and "1 de 1 el de la ficha" in n_sin


def test_e20_a_el_ajuste_de_los_cortos_de_la_serie_usa_el_spread_del_sqx(tmp_path):
    """Cortos con barras planas: cada minuto vale −(spread + comisión/lote/pv). Con la ficha, 0,08; con el .sqx, 0,18."""
    for name, ds, spread, fuente in (("sin.sqx", None, 0.08, "ficha"), ("con.sqx", "18", 18.0 * 0.01, "sqx")):
        leg = _daily_leg(make_flat_leg(tmp_path, side="sell", default_spread=ds, name=name))
        d = leg["minute_diag"]
        assert leg["minute_reason"] is None and d["coherencia"] == 1.0 and d["sinteticas"] == 0
        assert (d["spread_fuente"], d["spread_px_usado"]) == (fuente, spread)
        for i in range(leg["s"].n_trades):
            np.testing.assert_allclose(leg["minute_valle"][i][1], -(spread + COST_PX), rtol=0, atol=1e-6)


@pytest.mark.parametrize("ds", [None, "18"], ids=["ficha", "sqx"])
def test_e20_a_el_suelo_de_la_cota_diaria_de_una_pata_sin_ohlc_usa_el_mismo_spread(tmp_path, ds):
    leg = _daily_leg(make_flat_leg(tmp_path, side="sell", default_spread=ds, write_ohlc=False))
    d = leg["minute_diag"]
    spread = 0.18 if ds else 0.08
    assert d["cota_diaria"] is True and d["sinteticas"] == d["n_trades"] and d["coherencia"] is None
    assert d["spread_fuente"] == ("sqx" if ds else "ficha") and d["spread_px_usado"] == (18.0 * 0.01 if ds else 0.08)
    for i in range(leg["s"].n_trades):                                                 # cortos: el spread entra en el suelo
        floor = -float(leg["fleg"].adverse_per_lot[i]) / 100.0 - spread
        np.testing.assert_allclose(leg["minute_valle"][i][1], floor, rtol=0, atol=1e-12)
        assert _floor_px(leg, i) == pytest.approx(floor, abs=1e-12)


@pytest.mark.parametrize("ds", [None, "0", "-3", "abc", "nan"])
def test_e20_b_sin_defaultspread_valido_en_el_sqx_manda_la_ficha(tmp_path, ds):
    """Sin ``defaultSpread`` (los sintéticos de los tests: el writer no lo escribe), o con uno no positivo o ilegible, el spread
    es el de la ficha, como antes de E20."""
    p = make_flat_leg(tmp_path, side="sell", default_spread=ds)
    assert main._sqx_backtest_spread(Path(p)) is None
    d = _daily_leg(p)["minute_diag"]
    assert (d["spread_fuente"], d["spread_px_usado"]) == ("ficha", 0.08)


def test_e20_el_spread_del_sqx_es_default_spread_por_tick_size_sin_redondeo_y_en_ftmo_es_el_mismo_float_que_la_ficha(tmp_path):
    con = make_flat_leg(tmp_path, side="sell", default_spread="18")
    assert main._sqx_backtest_spread(Path(con)) == 18.0 * 0.01
    # FTMO CFD: la ficha oficial es ``typical_spread_points · tick_size``; un .sqx con ese ``defaultSpread`` y ese ``tickSize``
    # da EXACTAMENTE el mismo float (misma expresión, sin redondeo): la ruta de FTMO no cambia
    policy = load_costs_policy(DEFAULT_CFD_COSTS_PATH)["sqx_policy"]
    n = 0
    for sym in ("EURUSD_ftmo", "XAUUSD_ftmo(2)", "USATECHIDXUSD_ftmo(2)"):
        sheet = symbols.resolve(sym)
        pts = policy[sym]["typical_spread_points"]
        p = write_sqx(tmp_path / f"f{n}.sqx", [_W.long_win()], symbol=sym, instrument="I", pv=sheet.point_value_usd,
                      tick=sheet.tick_size)
        patch_instrument_info(p, defaultSpread=repr(pts))
        assert main._sqx_backtest_spread(p) == sheet.spread_px, sym
        n += 1
    assert n == 3


def test_e20_semantica_del_argumento_sqx_spread_px_de_la_serie_por_fichas(tmp_path):
    p = make_flat_leg(tmp_path, side="sell", n_days=8)                              # OHLC plano coherente, cortos
    leg = _daily_leg(p)
    sheet = symbols.resolve("XAUUSD_axi")
    broker = symbols.load_registry().brokers["axi"]
    df, meta = main.read_trade_table_spec(p, symbols.reader_spec(sheet))
    to_axis = main._sqx_axis_transform(0.0)
    f = main._sqx_leg_minutes_sheet

    def run(sh=sheet, mode="point", **kw):
        return f(df, meta, leg["fleg"], sh, broker, to_axis, mode, **kw)

    # sin el argumento (los llamadores de antes): spread de la ficha y diagnóstico de siempre, SIN las claves de E20
    series0, d0, r0 = run()
    assert r0 is None and not (set(E20_DIAG_KEYS) & set(d0))
    # con None (el .sqx no trae spread): la ficha, y ahora con las claves
    series1, d1, _ = run(sqx_spread_px=None)
    assert (d1["spread_fuente"], d1["spread_px_usado"]) == ("ficha", 0.08)
    assert all(np.array_equal(series0[i][1], series1[i][1]) for i in series0)
    # un spread del .sqx > 0 manda; 0, negativo o no finito: la ficha
    series2, d2, _ = run(sqx_spread_px=0.18)
    assert (d2["spread_fuente"], d2["spread_px_usado"]) == ("sqx", 0.18)
    assert not np.array_equal(series0[0][1], series2[0][1])                           # y la serie de los cortos cambia
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert run(sqx_spread_px=bad)[1]["spread_fuente"] == "ficha"
    # ficha SIN spread: sin el del .sqx es el motivo de siempre; con él hay serie (el .sqx pone el spread)
    sin_spread = dataclasses.replace(sheet, spread_px=None)
    legacy = "YAML de costes sin typical_spread_points/tick_size para XAUUSD_axi"
    assert run(sin_spread) == (None, None, legacy) and run(sin_spread, sqx_spread_px=None) == (None, None, legacy)
    s3, d3, r3 = run(sin_spread, sqx_spread_px=0.18)
    assert r3 is None and (d3["spread_fuente"], d3["spread_px_usado"]) == ("sqx", 0.18) and len(s3) == 8
    # cota diaria (sin OHLC) y sin ningún spread: 0 en el suelo de los cortos y el motivo lo dice; con el del .sqx, el suyo
    sin_todo = dataclasses.replace(sheet, spread_px=None, ohlc_m1=None)
    _, d4, r4 = run(sin_todo, "daily", sqx_spread_px=None)
    assert r4.endswith("se usa la cota diaria por el MAE (sin spread en los cortos)")
    assert (d4["spread_fuente"], d4["spread_px_usado"]) == (None, 0.0)
    _, d5, r5 = run(sin_todo, "daily", sqx_spread_px=0.18)
    assert r5.endswith("se usa la cota diaria por el MAE") and (d5["spread_fuente"], d5["spread_px_usado"]) == ("sqx", 0.18)


# ---------------------------------------------------------------------------
# portfolio_load con una cartera .sqx de AXI: guardas coherentes
# ---------------------------------------------------------------------------
def test_carga_las_guardas_de_una_cartera_sqx_de_axi_hablan_de_la_lente_axi_y_no_de_ftmo(tmp_path):
    w = make_axi_world(tmp_path)
    res = _load(w.paths)
    g = res["guards"]
    assert res["ok"] and main._portfolios[res["portfolio_id"]]["broker"] == "axi"
    assert g["tipo_patas"]["status"] == "ok" and "AXI Select (MT5)" in g["tipo_patas"]["detail"]
    assert "lente AXI" in g["tipo_patas"]["detail"] and "FTMO" not in g["tipo_patas"]["detail"]
    assert g["flotante_m1"] == {"status": "ok", "detail": "OHLC M1 coherente en las 2 pata(s): flotante por minuto "
                                                          "disponible (cota por minuto con el realizado intradía)."}
    assert "lente AXI" in g["corte_1610"]["detail"] and "FTMO" not in g["corte_1610"]["detail"]
    assert "lente AXI" in g["valles"]["detail"] and "ventanas" not in g["valles"]["detail"]
    assert "lente AXI" in g["ventanas"]["detail"] and g["ventanas"]["status"] == "ok"
    # las patas de la carga se leyeron con la cota diaria: son las mismas entradas de caché que recarga el endpoint
    pf = main._portfolios[res["portfolio_id"]]
    assert all(_daily_leg(p) is l for p, l in zip(w.paths, pf["legs"]))
    assert all(l["minute_valle"] is not None for l in pf["legs"])


def test_carga_una_cartera_sqx_de_ftmo_conserva_las_guardas_de_siempre(tmp_path):
    res = _load(_ftmo_cfd_legs(tmp_path))
    g = res["guards"]
    assert g["tipo_patas"]["detail"] == ("patas .sqx nativas de CFD FTMO (valle por ventanas con swap, sizing Opción A "
                                         "con lotes de 0,01)")
    assert g["corte_1610"]["detail"].startswith("no aplica: CFD FTMO Swing")
    assert g["valles"]["detail"] == "no aplica: patas .sqx usan ventanas"
    assert "valle por ventanas = cota a precisión del backtest" in g["ventanas"]["detail"]
    assert g["flotante_m1"]["status"] == "warn" and g["flotante_m1"]["detail"].endswith(
        "la simulación usará la cota por ventanas (más pesimista).")


# ---------------------------------------------------------------------------
# Fase F4: correcciones de la revisión de código de la F3 (H2, H3 y H4; H1 vive en test_sqx_fichas.py)
# ---------------------------------------------------------------------------
class _Boom:
    """Un valor que no se deja convertir a float (``TypeError``)."""


@pytest.mark.parametrize("ds, tick", [
    ("abc", 0.01), ("", 0.01), (_Boom(), 0.01), ([18.0], 0.01),                     # default_spread no numérico
    (18.0, "x"), (18.0, ""), (18.0, _Boom()), (18.0, [0.01]),                       # tick_size no numérico
], ids=["ds_texto", "ds_vacio", "ds_objeto", "ds_lista", "tick_texto", "tick_vacio", "tick_objeto", "tick_lista"])
def test_cr_f3_h2_un_default_spread_o_tick_size_no_numerico_cae_a_la_ficha_y_nunca_veta_la_pata(
        tmp_path, monkeypatch, ds, tick):
    """H2: la fuente del spread es informativa. Si ``read_instrument_info`` devolviera un ``default_spread`` (o un
    ``tick_size``) que no se deja convertir a float, ``_sqx_backtest_spread`` da ``None`` y manda el ``spread_px`` de la ficha
    (E20), en vez de tumbar la pata con ``ValueError``/``TypeError`` (el ``float(...)`` estaba fuera del ``try``)."""
    p = make_flat_leg(tmp_path, side="sell", n_days=8)
    monkeypatch.setattr(main, "read_instrument_info", lambda path: {"default_spread": ds, "tick_size": tick})
    assert main._sqx_backtest_spread(Path(p)) is None
    leg = _daily_leg(p)                                              # la pata CARGA, con el spread de la ficha
    assert leg["minute_reason"] is None and leg["minute_diag"]["coherencia"] == 1.0
    assert (leg["minute_diag"]["spread_fuente"], leg["minute_diag"]["spread_px_usado"]) == ("ficha", 0.08)
    res = _load([p])
    assert res["legs"][0]["status"] == "ok" and res["ok"]
    assert _sim(res["portfolio_id"], n_paths=100)["meta"]["float_note"].count("el de la ficha") == 1


@pytest.mark.parametrize("exc", [ValueError("formato futuro"), TypeError("valor raro")], ids=["ValueError", "TypeError"])
def test_cr_f3_h2_si_el_lector_lanza_value_error_o_type_error_tambien_cae_a_la_ficha(tmp_path, monkeypatch, exc):
    p = make_flat_leg(tmp_path, side="sell", n_days=8)

    def boom(path):
        raise exc

    monkeypatch.setattr(main, "read_instrument_info", boom)
    assert main._sqx_backtest_spread(Path(p)) is None
    assert _daily_leg(p)["minute_diag"]["spread_fuente"] == "ficha"
    # y lo de siempre sigue valiendo: SQXFormatError y OSError, y un dato válido
    from sqx.d1.extract_sqx import SQXFormatError
    for e in (SQXFormatError("x"), OSError("y")):
        monkeypatch.setattr(main, "read_instrument_info", lambda path, e=e: (_ for _ in ()).throw(e))
        assert main._sqx_backtest_spread(Path(p)) is None
    monkeypatch.setattr(main, "read_instrument_info", lambda path: {"default_spread": "18", "tick_size": 0.01})
    assert main._sqx_backtest_spread(Path(p)) == 18.0 * 0.01                 # un texto numérico SÍ se convierte


def test_cr_f3_h3_sqx_leg_minutes_sheet_valida_synth_mode_al_principio_incluso_en_las_ramas_sin_serie(tmp_path):
    """H3: un ``synth_mode`` inválido falla en voz alta (``ValueError``) desde la primera línea, como ``_load_sqx_leg`` y
    ``_sqx_minute_series``; antes, en las ramas SIN serie (futuros, ficha sin ``ohlc_m1``, sin spread) devolvía un motivo
    silencioso y solo fallaba al llegar a construir la serie."""
    w = make_axi_world(tmp_path, n_days=12)
    leg = _daily_leg(w.legs["XAUUSD_axi"])
    sheet = symbols.resolve("XAUUSD_axi")
    reg = symbols.load_registry()
    df, meta = main.read_trade_table_spec(w.legs["XAUUSD_axi"], symbols.reader_spec(sheet))
    to_axis = main._sqx_axis_transform(0.0)
    f = main._sqx_leg_minutes_sheet
    casos = {
        "con serie": (sheet, reg.brokers["axi"]),
        "ficha sin ohlc_m1": (dataclasses.replace(sheet, ohlc_m1=None), reg.brokers["axi"]),
        "ficha sin spread": (dataclasses.replace(sheet, spread_px=None), reg.brokers["axi"]),
        "bróker de futuros": (sheet, reg.brokers["ftmo_futures"]),
    }
    for nombre, (sh, br) in casos.items():
        for bad in ("semanal", "", "DAILY", None):
            try:
                got = f(df, meta, leg["fleg"], sh, br, to_axis, synth_mode=bad)
            except ValueError as exc:
                assert "synth_mode" in str(exc), (nombre, bad, str(exc))
            else:
                pytest.fail(f"{nombre}, synth_mode={bad!r}: no falló, devolvió {got[2]!r}")
    # el mensaje nombra los modos válidos y el recibido, igual que los otros dos puntos de entrada
    with pytest.raises(ValueError, match=r"synth_mode debe ser uno de \('point', 'daily'\) \(recibido 'semanal'\)"):
        f(df, meta, leg["fleg"], sheet, reg.brokers["axi"], to_axis, synth_mode="semanal")
    # los dos modos válidos siguen dando lo de siempre en esas mismas ramas
    assert f(df, meta, leg["fleg"], dataclasses.replace(sheet, ohlc_m1=None), reg.brokers["axi"], to_axis) == (
        None, None, "sin ohlc_m1 en el YAML de costes para XAUUSD_axi")
    assert f(df, meta, leg["fleg"], sheet, reg.brokers["ftmo_futures"], to_axis, "daily") == (
        None, None, "no aplica: pata de futuros (sin OHLC M1)")
    s, d, why = f(df, meta, leg["fleg"], dataclasses.replace(sheet, ohlc_m1=None), reg.brokers["axi"], to_axis, "daily")
    assert d["cota_diaria"] is True and why.endswith("se usa la cota diaria por el MAE") and len(s) == 12


def test_cr_f3_h4_la_meta_de_axi_con_sqx_describe_las_patas_recargadas_no_las_de_la_carga(tmp_path, spy):
    """H4: si el ``.sqx`` cambia entre ``portfolio_load`` y ``simulate_axi``, el endpoint RECARGA las patas y simula con las
    nuevas; ``meta.legs`` (y el resto de ecos por pata) tienen que describir lo SIMULADO, no ``pf["legs"]`` (que sigue con la
    lectura de la carga)."""
    w = make_axi_world(tmp_path, n_days=40)
    pid = _load(w.paths)["portfolio_id"]
    cargadas = main._portfolios[pid]["legs"]
    assert [main._leg_n_trades(l) for l in cargadas] == [40, 40]
    # se reescriben los dos .sqx (mismos ficheros y nombres) con 60 operaciones y se les da otro mtime
    w2 = make_axi_world(tmp_path, n_days=60)
    assert w2.paths == w.paths
    for p in w.paths:
        ns = Path(p).stat().st_mtime_ns + 5_000_000_000
        os.utime(p, ns=(ns, ns))
    out = _sim(pid, n_paths=100)
    meta = out["meta"]
    simuladas = {n: s.n_trades for n, s in spy["strats"].items()}
    assert simuladas == {"AXI XAU.sqx": 60, "AXI EUR.sqx": 60}                    # el motor simuló las recargadas
    assert [(l["pata"], l["símbolo"], l["modo"], l["trades"]) for l in meta["legs"]] == [
        ("AXI XAU.sqx", "XAUUSD_axi", "riesgo fijo (Opción A)", 60),
        ("AXI EUR.sqx", "EURUSD_axi", "riesgo fijo (Opción A)", 60)]                 # y la meta lo dice (antes: 40 y 40)
    assert meta["trades_in_window"] == 120 and meta["sizing"] == main.SQX_SIZING_MODE
    assert [main._leg_n_trades(l) for l in main._portfolios[pid]["legs"]] == [40, 40]     # la carga no se toca
    # sin cambios entre la carga y la simulación, la meta sigue siendo la de siempre
    pid2 = _load(w.paths)["portfolio_id"]
    assert [l["trades"] for l in _sim(pid2, n_paths=100)["meta"]["legs"]] == [60, 60]
