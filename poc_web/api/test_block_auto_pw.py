"""Guardrail ``bloque-politis-white``, fase A: bloque automático de Politis-White sobre la serie en R y sensibilidad con bloque 1
(plan §2.2-2.3 y enmiendas E1, E2, E4, E5, E6, E8, E10, E12-E16, E18, E19).

Todo sintético y hermético (no necesita ``risk_app/inputs``): los ``.sqx`` se escriben con el writer de
``tests/test_sqx_trades.py``, el OHLC y las fichas de usuario van a carpetas temporales (nunca los reales). Los resultados
de cada operación se generan con una cadena de regímenes persistentes (rachas de aciertos y de fallos), de modo que la serie en R
tiene dependencia y el bloque de Politis-White sale claramente > 1 (con una serie iid saldría 1 y varios tests serían vacíos).

Mapa criterio -> test (el id del criterio está en el nombre):

- AC-A4 (E6) serie R a mano, episodio a episodio: pata de futuros (micros), patas de CFD de la lente AXI, episodios con
  operaciones solapadas de 2 patas, riesgos distintos por operación, pata CSV con ``sl_exact`` y pata CSV en modo rápido,
  operaciones con riesgo no válido (E8), suma vectorizada igual a un bucle plano (E14) ......................... ``test_ac_a4_*``
- AC-A5 (E5, E13) independencia de la rejilla y de ``f_ref``: mismo ``meta.block_size`` con 5 rejillas y 2 ``f_ref`` en las dos
  lentes (``.sqx`` y CSV en modo rápido), y con ``oos_start`` ................................................ ``test_ac_a5_*``
- AC-A6 (E8) respaldo: < 25 episodios o más del 10 % de términos excluidos -> bloque 1, ``fallback_iid`` y aviso
  .............................................................................................. ``test_ac_a6_*``
- AC-A7 (E4, E15, E16, E18, E19) sensibilidad con bloque 1 solo en f*: coincide con una petición manual de bloque 1, es ``null``
  con bloque 1 o ``fallback_iid``, sale también en modo manual > 1, aviso si y solo si se supera la tolerancia, y f*, la
  recomendación y la receta no cambian por calcularla .............................................................. ``test_ac_a7_*``
- AC-A8 / E2 / E10 / E12 textos: ``_bs_explain`` por modo, el render de ``tamano_bloque`` con un decimal y el método, y los
  campos nuevos de ``meta.model_context.block_size`` ........................................................ ``test_ac_a8_*``
- E13 ``check_parity`` envía el ``meta.block_size`` del baseline ............................................. ``test_e13_*``

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_block_auto_pw.py -q
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poc_web.api import main, symbols
from poc_web.api.main import (
    AxiSimRequest,
    PortfolioLoadRequest,
    PropfirmSimRequest,
    _auto_block_size,
    _bs_explain,
    _episode_r_series,
    _model_context,
    portfolio_load,
    simulate_axi,
    simulate_propfirm,
    simulate_propfirm_pair,
)
from poc_web.api.model_notes import NOTES, render_run
from src.risk.block_length import politis_white_info

ROOT = Path(__file__).resolve().parents[2]
REAL_USER_FILES = (ROOT / "config" / "symbols.user.yaml", ROOT / "config" / "propfirms.user.yaml")


def _load_sibling(name: str, rel: str):
    """Carga un módulo hermano por ruta (el patrón del repo: ``poc_web/api`` no es un paquete)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_W = _load_sibling("_tst_block_auto_sqx_writer", "tests/test_sqx_trades.py")
write_sqx = _W.write_sqx
_SIB: dict = {}


def _csv_mod():
    """El módulo ``test_csv_minute_float.py`` (constantes y cabecera del CSV sintético), cargado la primera vez que hace falta."""
    if "csv" not in _SIB:
        _SIB["csv"] = _load_sibling("_tst_block_auto_csv_world", "poc_web/api/test_csv_minute_float.py")
    return _SIB["csv"]


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
    """Cachés vacías, catálogo y OHLC aislados del disco real (nunca el OHLC real), fichas de usuario en tmp."""
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    (tmp_path / "ohlc_axi").mkdir()
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")
    monkeypatch.setattr(main, "SQX_OHLC_DIR_OVERRIDES", {"axi": tmp_path / "ohlc_axi"})
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "symbols.user.yaml")
    symbols._reload()


@pytest.fixture()
def spy(monkeypatch):
    """Captura lo que el endpoint USA de verdad: cada llamada a ``_episode_r_series`` (bloques, patas, serie y excluidos) y a
    ``stationary_block_indices`` (para comprobar que la serie tiene un valor por episodio que se remuestrea, E1)."""
    seen = {"r": [], "idx": []}
    real_r, real_idx = main._episode_r_series, main.stationary_block_indices

    def r_wrapped(blocks, legs_ord, pf):
        out = real_r(blocks, legs_ord, pf)
        seen["r"].append(SimpleNamespace(blocks=blocks, legs=legs_ord, pf=pf, r=out[0], n_excluded=out[1]))
        return out

    def idx_wrapped(n_trades, n_paths, block_size, rng, length=None):
        seen["idx"].append(SimpleNamespace(n=n_trades, bs=block_size))
        return real_idx(n_trades, n_paths, block_size, rng, length=length)

    monkeypatch.setattr(main, "_episode_r_series", r_wrapped)
    monkeypatch.setattr(main, "stationary_block_indices", idx_wrapped)
    return seen


# ---------------------------------------------------------------------------
# Resultados con dependencia (regímenes persistentes) y oráculo independiente de la serie R
# ---------------------------------------------------------------------------
# Los mundos sintéticos salen de un RNG sembrado: si el flujo de numpy es otro (otra versión), los umbrales de abajo (bloque > 1,5)
# no son comparables y los tests que construyen un mundo se omiten en vez de dar un falso fallo (mismo criterio que
# ``test_recipe.py``). Los tests sin mundo (series a mano, textos) corren siempre.
RNG_CANARY = ([0.8050029237453802, 0.8079407897364937], [-3, 1, 10], [15.0, 25.0, 10.0],
              [0.3066951046284146, 0.32677856433599894])


def _rng_or_skip() -> None:
    r = np.random.default_rng(5)
    got = (r.random(2).tolist(), r.integers(-40, 40, 3).tolist(), r.choice([10.0, 15.0, 20.0, 25.0], 3).tolist(),
           r.uniform(0.0, 0.8, 2).tolist())
    if got != RNG_CANARY:
        pytest.skip("el flujo del RNG de numpy es otro: los mundos sintéticos de este módulo no son comparables")


def _regime(n: int, seed: int, persist: float = 0.93) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reg, s = np.empty(n, dtype=int), 1
    for t in range(n):
        if t and rng.random() >= persist:
            s = 1 - s
        reg[t] = s
    return reg


def _wins(reg: np.ndarray, seed: int, p_good: float = 0.85, p_bad: float = 0.15) -> np.ndarray:
    return np.random.default_rng(seed).random(len(reg)) < np.where(reg == 1, p_good, p_bad)


def _f32(x: float) -> float:
    return float(np.float32(x))


def _r_of(t: dict, pv: float) -> float:
    """R de UNA operación como lo ve el motor: ``(pnl/lotes) / (|apertura − SL| · pv)`` con los valores tal como quedan en el
    ``.sqx`` (float32). Es independiente del código de la app."""
    return (_f32(t["pnl"]) / _f32(t["lots"])) / (abs(_f32(t["open_px"]) - _f32(t["sl_px"])) * pv)


def _episodes(legs_trades: list[list[dict]], pvs: list[float]) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """(R por episodio, inicio de cada episodio) con un fusionado de intervalos plano e independiente de
    ``detect_flatflat_blocks``: dos operaciones están en el mismo episodio si una abre ESTRICTAMENTE antes de que cierre la
    otra (los extremos que se tocan son episodios distintos)."""
    items = sorted(((pd.Timestamp(t["open"]), pd.Timestamp(t["close"]), _r_of(t, pv))
                    for trades, pv in zip(legs_trades, pvs) for t in trades), key=lambda x: x[0])
    eps, cur = [], None
    for o, c, r in items:
        if cur is not None and o < cur[1]:
            cur = [cur[0], max(cur[1], c), cur[2] + r, cur[3] + 1]
        else:
            if cur is not None:
                eps.append(cur)
            cur = [o, c, r, 1]
    eps.append(cur)
    return np.array([e[2] for e in eps]), [e[0] for e in eps]


# ---- futuros (.sqx de micros; lente prop firm) ----
FUT, FUT_ACCOUNT, FUT_PV, FUT_COMM = "ftmo_futures_growth_50k", 50000.0, 2.0, 1.12


def _fut_trades(days, wins, seed, *, hour: int, minute: int, dists, base_px: float, skip_every: int = 0) -> list[dict]:
    """Una operación larga por día (90 min) con SL a una distancia que varía por operación: gana 1,5 R o pierde en el SL. Los
    precios son múltiplos de 0,25 (exactos en float32)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, d in enumerate(days):
        dist = float(rng.choice(dists))
        open_px = base_px + 0.25 * float(rng.integers(-40, 40))
        if skip_every and i % skip_every == skip_every - 1:
            continue
        sl_px = open_px - dist
        close_px = open_px + 1.5 * dist if wins[i] else sl_px
        gross = (close_px - open_px) * FUT_PV
        mae = float(rng.uniform(0.0, 0.8)) * dist * FUT_PV if wins[i] else -gross
        ot = datetime(d.year, d.month, d.day, hour, minute)
        out.append(dict(open=ot, close=ot + timedelta(minutes=90), lots=1.0, open_px=open_px, close_px=close_px,
                        sl_px=sl_px, pnl=gross - FUT_COMM, mae=max(mae, 0.0), flag=1))
    return out


def make_fut_world(tmp_path: Path, *, n_days: int = 240, seed: int = 5):
    """Dos patas de micros (símbolo por defecto del writer) que se SOLAPAN: la A opera 10:00-11:30 y la B 11:00-12:30; la B
    se salta uno de cada cinco días (esos días el episodio es de una sola operación). Riesgos distintos por operación."""
    _rng_or_skip()
    days = pd.bdate_range("2023-01-02", periods=n_days)
    reg = _regime(n_days, seed)
    trades = [
        _fut_trades(days, _wins(reg, seed + 1), seed + 10, hour=10, minute=0, dists=(10.0, 15.0, 20.0, 25.0), base_px=18000.0),
        _fut_trades(days, _wins(reg, seed + 2), seed + 20, hour=11, minute=0, dists=(8.0, 12.0, 30.0, 40.0), base_px=17000.0,
                    skip_every=5),
    ]
    paths = [str(write_sqx(tmp_path / "extra" / "bank" / f"Strategy 7.{k}.1.sqx", tr, capital=FUT_ACCOUNT))
             for k, tr in enumerate(trades)]
    return SimpleNamespace(paths=paths, trades=trades, pvs=[FUT_PV, FUT_PV])


# ---- CFD de AXI (.sqx; lente AXI, sin OHLC: cota diaria por el MAE) ----
AXI_ACCOUNT, AXI_LOTS = 5000.0, 0.1
AXI_SPECS = {
    "XAUUSD_axi": dict(file="AXI XAU.sqx", instrument="XAUUSD_mt5_axi", pv=100.0, tick=0.01, px0=2000.0,
                       dists=(3.0, 4.5, 6.0), hour=10),
    "EURUSD_axi": dict(file="AXI EUR.sqx", instrument="EURUSD_axi", pv=100000.0, tick=0.0001, px0=1.10,
                       dists=(0.0006, 0.0009, 0.0012), hour=11),
}


def make_axi_world(tmp_path: Path, *, n_days: int = 240, seed: int = 5):
    """Dos patas de CFD de AXI que se SOLAPAN (10:00-11:30 y 11:00-12:30), con resultados de régimen persistente y sin OHLC
    (cada pata usa su cota diaria por el MAE). La segunda se salta uno de cada cinco días."""
    _rng_or_skip()
    days = pd.bdate_range("2023-01-02", periods=n_days)
    reg = _regime(n_days, seed)
    legs, trades_by_leg, pvs = {}, [], []
    for k, (sym, sp) in enumerate(AXI_SPECS.items()):
        sheet = symbols.resolve(sym)                       # la ficha OFICIAL de AXI: la comisión por lote
        comm = sheet.commission["per_lot_rt"]
        wins, rng = _wins(reg, seed + 1 + k), np.random.default_rng(seed + 10 * (k + 1))
        trades = []
        for i, d in enumerate(days):
            dist = float(rng.choice(sp["dists"]))
            o_px = _f32(sp["px0"] + sp["tick"] * float(rng.integers(-30, 30)))
            if k == 1 and i % 5 == 4:
                continue
            sl_px = _f32(o_px - dist)
            c_px = _f32(o_px + 1.5 * dist) if wins[i] else sl_px
            gross = (c_px - o_px) * sp["pv"] * AXI_LOTS
            mae = float(rng.uniform(0.0, 0.8)) * dist * sp["pv"] * AXI_LOTS if wins[i] else -gross + 1e-3
            ot = datetime(d.year, d.month, d.day, sp["hour"], 0)
            trades.append(dict(open=ot, close=ot + timedelta(minutes=90), lots=AXI_LOTS, open_px=o_px, close_px=c_px,
                               sl_px=sl_px, pnl=gross - comm * AXI_LOTS, mae=max(mae, 0.0), flag=1))
        legs[sym] = str(write_sqx(tmp_path / "extra" / "bank" / sp["file"], trades, symbol=sym,
                                  instrument=sp["instrument"], pv=sp["pv"], tick=sp["tick"], capital=AXI_ACCOUNT))
        trades_by_leg.append(trades)
        pvs.append(sp["pv"])
    return SimpleNamespace(paths=list(legs.values()), trades=trades_by_leg, pvs=pvs)


# ---- CSV (una pata en modo rápido: sin .sqx emparejado) ----
CSV_ACCOUNT = 100_000.0


def make_csv_world(tmp_path: Path, *, n: int = 200, seed: int = 5):
    """CSV orderstocsv + OHLC M1 de ``XAUUSD_axi``: un trade por día hábil (09:00-11:00), backtest que compone y resultados de
    régimen persistente (la misma construcción que ``test_csv_minute_float._synth_files``, con los aciertos dados)."""
    _rng_or_skip()
    wins = _wins(_regime(n, seed), seed + 1)
    days = pd.bdate_range("2023-01-02", periods=n)
    C = _csv_mod()
    e, rows, frames = C._CSV_CAPITAL, [], []
    for i, d in enumerate(days):
        sl = 1500.0 + 10.0 * (i % 5)
        sz = round(float(np.floor(C._F_REF * e / sl / C._STEP) * C._STEP), 2)
        ppl = (1.2 if wins[i] else -1.0) * sl
        p = round(sz * ppl, 2)
        e = round(e + p, 2)
        ot, ct = d + pd.Timedelta(hours=9), d + pd.Timedelta(hours=11)
        exit_px = 2000.0 + ppl / 100.0
        rows.append(f"{i + 1};XAUUSD_axi;Buy;{ot:%Y.%m.%d %H:%M};2000.00;{sz:.2f};{ct:%Y.%m.%d %H:%M};{exit_px:.2f};"
                    f"{p:.2f};{e:.2f};IST;Exit After 1 Bars;{-abs(ppl) * sz * 0.5:.2f};{abs(ppl) * sz:.2f};1;-\n")
        idx = pd.date_range(ot, ct, freq="1min")
        path = 2000.0 + (exit_px - 2000.0) * np.linspace(0.0, 1.0, len(idx))
        frames.append(pd.DataFrame({"o": path, "h": path + 0.3, "l": path - 0.3 - (0.0 if wins[i] else 1.0), "c": path},
                                   index=idx))
    csv = tmp_path / "bloque_pw_synth_leg.csv"
    csv.write_text(C._HEADER + "".join(rows), encoding="utf-8")
    return SimpleNamespace(csv=csv, ohlc=pd.concat(frames), n=n)


def load_csv(monkeypatch, w, f_ref: float) -> str:
    monkeypatch.setattr(main, "_load_ohlc", lambda sym: w.ohlc)
    res = portfolio_load(PortfolioLoadRequest(legs=[str(w.csv)], account_size=CSV_ACCOUNT, f_ref=f_ref))
    assert res["ok"], res
    return res["portfolio_id"]


# ---- peticiones ----
def load_fut(w) -> str:
    res = portfolio_load(PortfolioLoadRequest(legs=w.paths, account_size=FUT_ACCOUNT, firm_key=FUT))
    assert res["ok"], res
    return res["portfolio_id"]


def load_axi(w) -> str:
    res = portfolio_load(PortfolioLoadRequest(legs=w.paths, account_size=AXI_ACCOUNT))
    assert res["ok"], res
    return res["portfolio_id"]


PF_GRID = dict(f_lo=0.0025, f_hi=0.0075, f_step=0.0025)


def preq(pid, firm=FUT, **kw) -> PropfirmSimRequest:
    base = dict(portfolio_id=pid, firm_key=firm, n_paths=200, n_challenges=5, n_sequences=100, capital0_eur=3000.0,
                fx_eur_usd=1.08, **PF_GRID)
    base.update(kw)
    return PropfirmSimRequest(**base)


AXI_GRID = dict(f_lo=0.01, f_hi=0.04, f_step=0.01)


def areq(pid, **kw) -> AxiSimRequest:
    base = dict(portfolio_id=pid, stage="seed", n_paths=200, **AXI_GRID)
    base.update(kw)
    return AxiSimRequest(**base)


def row_at(out: dict, f: float, kpis) -> dict:
    """La fila de ``reports`` de la f dada (por cercanía) con las claves pedidas."""
    r = min(out["reports"], key=lambda r: abs(r["f"] - f))
    assert abs(r["f"] - f) < 1e-9, (r["f"], f)
    return {k: r[k] for k in kpis}


PROP_KPIS = ("p_both", "p_net_negative", "ev_per_attempt")
AXI_KPIS = ("p_pass", "p_ruin")


def canon(out: dict, *, drop_meta=("elapsed_s", "bs1")) -> str:
    """Respuesta serializada estable, sin lo que varía entre corridas (tiempos) y sin las claves dadas de ``meta``."""
    o = json.loads(json.dumps(out, default=str))
    for k in drop_meta:
        o["meta"].pop(k, None)
    o["warnings"] = [w for w in o["warnings"] if not w.startswith("Con bloque 1")]
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


# ===========================================================================
# AC-A4 (E6): la serie R, a mano
# ===========================================================================
def test_ac_a4_futuros_serie_r_a_mano_con_solapes_y_riesgos_distintos(tmp_path, spy):
    w = make_fut_world(tmp_path)
    pid = load_fut(w)
    out = simulate_propfirm(preq(pid))
    expected, _starts = _episodes(w.trades, w.pvs)
    got = spy["r"][0]
    # E1: exactamente los episodios que remuestrea la lente, en el mismo orden
    assert len(got.r) == len(got.blocks) == out["meta"]["n_blocks"] == len(expected)
    assert spy["idx"][0].n == len(got.r)
    np.testing.assert_allclose(got.r, expected, rtol=1e-12, atol=0)
    assert got.n_excluded == 0
    # el escenario ejerce lo que dice: episodios de 2 operaciones solapadas de 2 patas y de 1, y riesgos distintos
    sizes = np.array([len(b.sym_id) for b in got.blocks])
    assert set(sizes) == {1, 2} and (sizes == 2).sum() > 100
    two_legs = [b for b in got.blocks if len(b.sym_id) == 2]
    assert all(set(b.sym_id) == {0, 1} for b in two_legs)
    risks = {round(abs(_f32(t["open_px"]) - _f32(t["sl_px"])) * FUT_PV, 6) for tr in w.trades for t in tr}
    assert len(risks) >= 6
    # lote entero de micros: lot_step = min_lot = 1 (el riesgo por lote es el de un micro)
    leg0 = got.legs[0]
    assert float(leg0["meta"]["pv"]) == FUT_PV and out["meta"]["leg_kind"] == "sqx_futures"
    assert float(leg0["meta"].get("lot_step") or 1.0) == float(leg0["meta"].get("min_lot") or 1.0) == 1.0
    np.testing.assert_allclose(leg0["fleg"].risk_per_micro,
                               [abs(_f32(t["open_px"]) - _f32(t["sl_px"])) * FUT_PV for t in w.trades[0]], rtol=1e-12)


def test_ac_a4_cfd_axi_serie_r_a_mano_lente_axi(tmp_path, spy):
    w = make_axi_world(tmp_path)
    pid = load_axi(w)
    out = simulate_axi(areq(pid))
    expected, _starts = _episodes(w.trades, w.pvs)
    got = spy["r"][0]
    assert len(got.r) == len(got.blocks) == out["meta"]["n_blocks"] == len(expected)
    assert spy["idx"][0].n == len(got.r)
    np.testing.assert_allclose(got.r, expected, rtol=1e-12, atol=0)
    assert got.n_excluded == 0
    assert {len(b.sym_id) for b in got.blocks} == {1, 2}             # solapes de 2 patas y episodios de una sola operación
    # el R de una operación depende de su distancia al SL (la comisión pesa distinto en R): el oráculo no es trivial
    assert len({round(_r_of(t, pv), 6) for tr, pv in zip(w.trades, w.pvs) for t in tr}) >= 10


def test_ac_a4_con_corte_oos_la_serie_es_la_cola_de_la_serie_completa(tmp_path, spy):
    """E1/E13: la serie sale de los bloques de la ventana (``oos_start``): es el SUFIJO de la serie completa."""
    w = make_axi_world(tmp_path)
    pid = load_axi(w)
    simulate_axi(areq(pid))
    full = spy["r"][-1].r
    cut = "2023-08-01"
    out = simulate_axi(areq(pid, oos_start=cut))
    win = spy["r"][-1]
    expected, starts = _episodes(w.trades, w.pvs)
    keep = np.array([s >= pd.Timestamp(cut) for s in starts])
    assert 0 < keep.sum() < len(keep)
    assert len(win.r) == out["meta"]["n_blocks"] == int(keep.sum())
    np.testing.assert_allclose(win.r, expected[keep], rtol=1e-12, atol=0)
    np.testing.assert_allclose(win.r, full[-len(win.r):], rtol=1e-12, atol=0)
    assert spy["idx"][-1].n == len(win.r)


def _blk(sym, tid, ppl):
    return SimpleNamespace(sym_id=np.array(sym, dtype=int), trade_local_id=np.array(tid, dtype=int),
                           pnl_per_lot=np.array(ppl, dtype=float))


def _sqx_leg(risk):
    return {"name": "sqx", "kind": main.KIND_SQX, "fleg": SimpleNamespace(risk_per_micro=np.array(risk, dtype=float))}


def _csv_exact_leg(sl):
    return {"name": "csv_exacta", "sl_exact": np.array(sl, dtype=float), "tl": None}


def _csv_fast_leg(equity_before, size):
    eq, sz = np.array(equity_before, dtype=float), np.array(size, dtype=float)

    def sl_in_money(f_ref):                                   # lo mismo que ``TradeList.sl_in_money``
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sz > 0, f_ref * eq / sz, np.nan)
    return {"name": "csv_rapida", "sl_exact": None, "tl": SimpleNamespace(sl_in_money=sl_in_money)}


def test_ac_a4_ejemplo_numerico_a_mano_tres_clases_de_pata():
    """Golden a mano. Tres patas: ``.sqx`` (riesgo 50, 80), CSV exacta (``sl_exact`` 40, 100) y CSV rápida
    (``f_ref · equity / size``: 0,01 · 10.000 / 2 = 50 y 0,01 · 10.000 / 4 = 25). Cuatro episodios:
    #0 una operación de la ``.sqx``: 100/50 = 2;  #1 solape de las tres patas: 100/80 + (-20)/40 + 30/50 = 1,25 - 0,5 + 0,6 = 1,35;
    #2 CSV rápida: -50/25 = -2;  #3 CSV exacta: 150/100 = 1,5."""
    legs = [_sqx_leg([50.0, 80.0]), _csv_exact_leg([40.0, 100.0]), _csv_fast_leg([10_000.0, 10_000.0], [2.0, 4.0])]
    pf = {"f_ref": 0.01}
    blocks = [_blk([0], [0], [100.0]),
              _blk([0, 1, 2], [1, 0, 0], [100.0, -20.0, 30.0]),
              _blk([2], [1], [-50.0]),
              _blk([1], [1], [150.0])]
    r, n_excl = _episode_r_series(blocks, legs, pf)
    np.testing.assert_allclose(r, [2.0, 1.35, -2.0, 1.5], rtol=1e-12, atol=0)
    assert n_excl == 0


def test_ac_a4_la_pata_csv_rapida_escala_con_f_ref_y_el_bloque_no_cambia():
    """E5: en modo rápido el riesgo es ``f_ref · equity / size`` (lineal en ``f_ref``, mismo factor en todas las operaciones):
    la serie se escala en 1/f_ref y el b* de Politis-White (invariante de escala) no cambia."""
    rng = np.random.default_rng(1)
    n = 300
    eq = 50_000.0 * (1 + 0.001 * np.arange(n))
    size = np.round(rng.uniform(0.5, 3.0, n), 2)
    reg = _regime(n, 3)
    ppl = np.where(_wins(reg, 4), 1.2, -1.0) * rng.uniform(900.0, 1700.0, n)
    leg = _csv_fast_leg(eq, size)
    blocks = [_blk([0], [i], [ppl[i]]) for i in range(n)]
    r1, e1 = _episode_r_series(blocks, [leg], {"f_ref": 0.005})
    r2, e2 = _episode_r_series(blocks, [leg], {"f_ref": 0.02})
    assert e1 == e2 == 0
    np.testing.assert_allclose(r1, 4.0 * r2, rtol=1e-12, atol=0)
    b1, b2 = politis_white_info(r1), politis_white_info(r2)
    assert b1.b == pytest.approx(b2.b, abs=1e-9) and b1.b > 1.5
    assert (b1.opt_m, b1.m) == (b2.opt_m, b2.m)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf"), -float("inf")])
def test_ac_a4_operacion_con_riesgo_no_valido_no_suma_y_se_cuenta(bad):
    """E8: riesgo ≤ 0 o no finito -> la operación NO suma a su episodio y se cuenta; los episodios nunca se descartan."""
    legs = [_csv_exact_leg([10.0, bad, 20.0])]
    blocks = [_blk([0], [0], [30.0]), _blk([0], [1], [99.0]), _blk([0], [2], [40.0])]
    r, n_excl = _episode_r_series(blocks, legs, {"f_ref": 0.01})
    np.testing.assert_allclose(r, [3.0, 0.0, 2.0], rtol=1e-12, atol=0)      # el episodio #1 sigue ahí, con R = 0
    assert n_excl == 1 and len(r) == len(blocks)


def test_ac_a4_cociente_no_finito_tambien_se_excluye_y_el_resto_del_episodio_si_suma():
    """Un pnl por lote no finito (tamaño 0 en un CSV) invalida solo ESA operación; las demás del episodio suman."""
    legs = [_sqx_leg([10.0, 10.0]), _csv_exact_leg([5.0])]
    blocks = [_blk([0, 1, 0], [0, 0, 1], [20.0, float("nan"), -10.0])]
    r, n_excl = _episode_r_series(blocks, legs, {"f_ref": 0.01})
    np.testing.assert_allclose(r, [2.0 - 1.0], rtol=1e-12, atol=0)
    assert n_excl == 1


def test_ac_a4_sin_bloques_serie_vacia():
    r, n = _episode_r_series([], [_sqx_leg([1.0])], {"f_ref": 0.01})
    assert len(r) == 0 and n == 0


def test_ac_a4_sym_id_fuera_de_las_patas_es_error_y_no_una_serie_desalineada():
    with pytest.raises(ValueError, match="desalineada"):
        _episode_r_series([_blk([2], [0], [1.0])], [_sqx_leg([1.0])], {"f_ref": 0.01})


def test_ac_a4_suma_vectorizada_igual_a_un_bucle_plano_sobre_datos_aleatorios():
    """E14: la suma por episodio (``np.add.at``) da lo mismo que el bucle plano de la fórmula, con riesgos válidos e inválidos
    mezclados y patas de las tres clases."""
    rng = np.random.default_rng(11)
    n_ep = 400
    r_sqx = rng.uniform(5.0, 60.0, 500)
    sl = rng.uniform(5.0, 60.0, 500)
    mask = rng.random(500) < 0.06
    sl[mask] = rng.choice([0.0, np.nan, np.inf, -3.0], size=int(mask.sum()))          # riesgos no válidos de las cuatro clases
    eq, size = np.full(500, 20_000.0), np.round(rng.uniform(0.0, 3.0, 500), 1)       # algún size = 0 -> riesgo NaN
    legs = [_sqx_leg(r_sqx), _csv_exact_leg(sl), _csv_fast_leg(eq, size)]
    pf = {"f_ref": 0.01}
    risks = [r_sqx, sl, np.where(size > 0, 0.01 * eq / np.where(size > 0, size, 1), np.nan)]
    blocks = []
    for _ in range(n_ep):
        k = int(rng.integers(1, 5))
        blocks.append(_blk(rng.integers(0, 3, k), rng.integers(0, 500, k), rng.normal(0.0, 50.0, k)))
    r, n_excl = _episode_r_series(blocks, legs, pf)
    want, bad = np.zeros(n_ep), 0
    for i, b in enumerate(blocks):
        for s, t, p in zip(b.sym_id, b.trade_local_id, b.pnl_per_lot):
            risk = risks[s][t]
            if np.isfinite(risk) and risk > 0 and np.isfinite(p / risk):
                want[i] += p / risk
            else:
                bad += 1
    np.testing.assert_allclose(r, want, rtol=1e-12, atol=1e-12)
    assert n_excl == bad and bad > 0


# ===========================================================================
# AC-A5 (E5, E13): independencia de la rejilla y de f_ref
# ===========================================================================
GRIDS_PF = [dict(f_lo=0.0025, f_hi=0.0075, f_step=0.0025), dict(f_lo=0.005, f_hi=0.0125, f_step=0.0025),
            dict(f_lo=0.0025, f_hi=0.005, f_step=0.0025), dict(f_lo=0.0075, f_hi=0.01, f_step=0.0025),
            dict(f_lo=0.0025, f_hi=0.0125, f_step=0.005)]
GRIDS_AXI = [dict(f_lo=0.01, f_hi=0.04, f_step=0.01), dict(f_lo=0.005, f_hi=0.02, f_step=0.0025),
             dict(f_lo=0.0075, f_hi=0.015, f_step=0.0025), dict(f_lo=0.0115, f_hi=0.0125, f_step=0.001),
             dict(f_lo=0.015, f_hi=0.025, f_step=0.0025)]


def test_ac_a5_futuros_prop_firm_mismo_bloque_con_cinco_rejillas_y_dos_f_ref(tmp_path):
    w = make_fut_world(tmp_path)
    blocks = set()
    for f_ref in (0.005, 0.02):
        res = portfolio_load(PortfolioLoadRequest(legs=w.paths, account_size=FUT_ACCOUNT, firm_key=FUT, f_ref=f_ref))
        assert res["ok"], res
        for g in GRIDS_PF:
            out = simulate_propfirm(preq(res["portfolio_id"], n_paths=100, **g))
            assert out["meta"]["model_context"]["block_size"]["mode"] == "auto"
            blocks.add(out["meta"]["block_size"])
    assert len(blocks) == 1 and next(iter(blocks)) > 1.5, blocks      # EXACTAMENTE el mismo, y no es el trivial 1


def test_ac_a5_cfd_axi_mismo_bloque_con_cinco_rejillas_y_dos_f_ref(tmp_path):
    w = make_axi_world(tmp_path)
    blocks = set()
    for f_ref in (0.005, 0.02):
        res = portfolio_load(PortfolioLoadRequest(legs=w.paths, account_size=AXI_ACCOUNT, f_ref=f_ref))
        assert res["ok"], res
        for g in GRIDS_AXI:
            out = simulate_axi(areq(res["portfolio_id"], n_paths=100, **g))
            assert out["meta"]["model_context"]["block_size"]["mode"] == "auto"
            blocks.add(out["meta"]["block_size"])
    assert len(blocks) == 1 and next(iter(blocks)) > 1.5, blocks


def test_ac_a5_csv_en_modo_rapido_mismo_bloque_en_las_dos_lentes_con_cinco_rejillas_y_dos_f_ref(tmp_path, monkeypatch, spy):
    """E5: pata CSV en modo rápido (sin ``sl_exact``): el riesgo por lote depende de ``f_ref``, la serie se escala y el bloque no."""
    w = make_csv_world(tmp_path)
    by_lens = {"propfirm": set(), "axi": set()}
    series = {}
    for f_ref in (0.005, 0.02):
        pid = load_csv(monkeypatch, w, f_ref)
        assert all(l["sizing_mode"] == "rápido" for l in main._portfolios[pid]["legs"])
        for g_pf, g_axi in zip(GRIDS_PF, GRIDS_AXI):
            out = simulate_propfirm(preq(pid, firm="ftmo_100k", n_paths=100, n_challenges=3, n_sequences=100, **g_pf))
            by_lens["propfirm"].add(out["meta"]["block_size"])
            series[("propfirm", f_ref)] = spy["r"][-1].r.copy()
            out = simulate_axi(areq(pid, n_paths=100, **g_axi))
            by_lens["axi"].add(out["meta"]["block_size"])
            series[("axi", f_ref)] = spy["r"][-1].r.copy()
    for lens, bs in by_lens.items():
        assert len(bs) == 1 and next(iter(bs)) > 1.5, (lens, bs)
    for lens in by_lens:                                              # la serie se escala 1/f_ref: 0,02/0,005 = 4
        np.testing.assert_allclose(series[(lens, 0.005)], 4.0 * series[(lens, 0.02)], rtol=1e-9, atol=0)


def test_ac_a5_con_oos_start_mismo_bloque_con_cualquier_rejilla_dentro_de_la_ventana(tmp_path):
    """E13: el bloque sale de los bloques de la ventana ``oos_start``; con cualquier rejilla dentro de ella es el mismo (y no es
    el de la historia completa)."""
    w = make_fut_world(tmp_path)
    pid = load_fut(w)
    cut = "2023-06-01"
    win = {simulate_propfirm(preq(pid, n_paths=100, oos_start=cut, **g))["meta"]["block_size"] for g in GRIDS_PF}
    full = simulate_propfirm(preq(pid, n_paths=100))["meta"]["block_size"]
    assert len(win) == 1 and next(iter(win)) > 1.0
    assert next(iter(win)) != full                                    # la ventana tiene otra serie, otro b*
    w2 = make_axi_world(tmp_path / "axi")
    pid2 = load_axi(w2)
    win2 = {simulate_axi(areq(pid2, n_paths=100, oos_start=cut, **g))["meta"]["block_size"] for g in GRIDS_AXI}
    full2 = simulate_axi(areq(pid2, n_paths=100))["meta"]["block_size"]
    assert len(win2) == 1 and next(iter(win2)) > 1.5 and next(iter(win2)) != full2


# ===========================================================================
# AC-A6 (E8): respaldo
# ===========================================================================
def test_ac_a6_menos_de_25_episodios_prop_firm_bs_1_fallback_y_aviso(tmp_path):
    w = make_fut_world(tmp_path, n_days=20)
    out = simulate_propfirm(preq(load_fut(w), n_paths=100))
    m, bsc = out["meta"], out["meta"]["model_context"]["block_size"]
    assert m["n_blocks"] == 20 < 25
    assert m["block_size"] == 1.0 and bsc["mode"] == bsc["method"] == "fallback_iid"
    assert bsc["reason"] == main._BS_REASONS["fallback_iid"] and bsc["opt_m"] is None and bsc["m"] is None
    assert any("Solo 20 bloques (<25)" in x and "block_size = 1" in x for x in out["warnings"])
    assert m["bs1"] is None                                           # con bloque 1 la cifra es la misma: no hay sensibilidad


def test_ac_a6_menos_de_25_episodios_axi_bs_1_fallback_y_aviso(tmp_path):
    w = make_axi_world(tmp_path, n_days=20)
    out = simulate_axi(areq(load_axi(w), n_paths=100))
    m, bsc = out["meta"], out["meta"]["model_context"]["block_size"]
    assert m["n_blocks"] == 20 < 25
    assert m["block_size"] == 1.0 and bsc["mode"] == bsc["method"] == "fallback_iid"
    assert any("Solo 20 bloques (<25)" in x for x in out["warnings"])
    assert not any("Pocos bloques para calibrar" in x for x in out["warnings"])      # el aviso viejo ya no existe
    assert m["bs1"] is None


def _fake_risk(monkeypatch, frac_bad: float):
    """Pone riesgo 0 en ``frac_bad`` de las operaciones de TODAS las patas (el resto, el riesgo real)."""
    real = main._leg_risk_per_lot

    def fake(leg, pf):
        r = np.array(real(leg, pf), dtype=float)
        r[: int(round(frac_bad * len(r)))] = 0.0
        return r
    monkeypatch.setattr(main, "_leg_risk_per_lot", fake)


def test_ac_a6_mas_del_10_por_ciento_de_terminos_excluidos_prop_firm_bs_1_fallback_y_aviso(tmp_path, monkeypatch):
    w = make_fut_world(tmp_path)
    pid = load_fut(w)
    _fake_risk(monkeypatch, 0.2)
    out = simulate_propfirm(preq(pid, n_paths=100))
    bsc = out["meta"]["model_context"]["block_size"]
    assert out["meta"]["block_size"] == 1.0 and bsc["mode"] == "fallback_iid"
    assert bsc["series"] == "R" and bsc["n_excluded_terms"] > 0
    assert any("sin riesgo por lote válido (más del 10 %)" in x for x in out["warnings"])
    assert out["meta"]["bs1"] is None


def test_ac_a6_mas_del_10_por_ciento_de_terminos_excluidos_axi_bs_1_fallback_y_aviso(tmp_path, monkeypatch):
    w = make_axi_world(tmp_path)
    pid = load_axi(w)
    _fake_risk(monkeypatch, 0.2)
    out = simulate_axi(areq(pid, n_paths=100))
    bsc = out["meta"]["model_context"]["block_size"]
    assert out["meta"]["block_size"] == 1.0 and bsc["mode"] == "fallback_iid" and bsc["n_excluded_terms"] > 0
    assert any("sin riesgo por lote válido (más del 10 %)" in x for x in out["warnings"])
    assert out["meta"]["bs1"] is None


def test_ac_a6_el_umbral_del_10_por_ciento_es_estricto_sobre_el_total_de_operaciones():
    """E8: el umbral es sobre el total de operaciones de los bloques: 10 de 100 -> sigue siendo automático; 11 de 100 -> respaldo."""
    n = 100
    reg = _regime(n, 7, persist=0.97)
    ppl = np.where(_wins(reg, 8), 1.5, -1.0) * 10.0
    blocks = [_blk([0], [i], [ppl[i]]) for i in range(n)]
    for n_bad, esperado in ((10, "auto"), (11, "fallback_iid"), (0, "auto")):
        risk = np.full(n, 10.0)
        risk[:n_bad] = 0.0
        bs, mode, info, warns = _auto_block_size(blocks, [_csv_exact_leg(risk)], {"f_ref": 0.01})
        assert mode == esperado and info["n_excluded_terms"] == n_bad, (n_bad, mode)
        assert (warns == []) == (esperado == "auto")
        if esperado == "fallback_iid":
            assert bs == 1.0 and "11 de 100 operaciones" in warns[0]
        else:
            assert bs >= 1.0 and info["m"] is not None and info["series"] == "R"


def test_ac_a6_el_umbral_de_25_episodios_24_respaldo_25_automatico():
    for n, esperado in ((24, "fallback_iid"), (25, "auto")):
        ppl = np.where(_wins(_regime(n, 9), 10), 1.5, -1.0)
        blocks = [_blk([0], [i], [ppl[i]]) for i in range(n)]
        _, mode, _, warns = _auto_block_size(blocks, [_sqx_leg(np.full(n, 1.0))], {"f_ref": 0.01})
        assert mode == esperado and bool(warns) == (esperado == "fallback_iid"), n


# ===========================================================================
# AC-A7 (E4, E15, E16, E18, E19): sensibilidad con bloque 1 solo en f*
# ===========================================================================
def test_ac_a7_prop_firm_futuros_bs1_coincide_con_una_peticion_manual_de_bloque_1(tmp_path):
    w = make_fut_world(tmp_path)
    pid = load_fut(w)
    auto = simulate_propfirm(preq(pid))
    assert auto["meta"]["block_size"] > 1.5 and auto["meta"]["model_context"]["block_size"]["mode"] == "auto"
    bs1 = auto["meta"]["bs1"]
    assert isinstance(bs1, list) and len(bs1) == 1 and set(bs1[0]) == {"f", *PROP_KPIS}      # E16: S = {f*}
    assert bs1[0]["f"] == auto["f_star"]
    manual = simulate_propfirm(preq(pid, block_size=1.0))
    assert manual["meta"]["bs1"] is None and manual["meta"]["block_size"] == 1.0
    want = row_at(manual, auto["f_star"], PROP_KPIS)
    assert {k: bs1[0][k] for k in PROP_KPIS} == want                       # EXACTAMENTE igual (misma semilla y caminos)
    # no es una cifra trivial: con el bloque automático el resultado de f* es otro
    assert row_at(auto, auto["f_star"], PROP_KPIS) != want


def test_ac_a7_prop_firm_cycle_csv_bs1_coincide_con_una_peticion_manual_de_bloque_1(tmp_path, monkeypatch):
    """La otra rama de ``_propfirm_eval_f`` (perfil ``cycle``, ``sequence_ev``): P(F1+F2), P(neto<0) y EV."""
    w = make_csv_world(tmp_path)
    pid = load_csv(monkeypatch, w, 0.01)
    kw = dict(firm="ftmo_100k", n_paths=150, n_challenges=3, n_sequences=100)
    auto = simulate_propfirm(preq(pid, **kw))
    assert auto["meta"]["block_size"] > 1.5
    bs1 = auto["meta"]["bs1"]
    assert len(bs1) == 1 and bs1[0]["f"] == auto["f_star"]
    manual = simulate_propfirm(preq(pid, block_size=1.0, **kw))
    assert {k: bs1[0][k] for k in PROP_KPIS} == row_at(manual, auto["f_star"], PROP_KPIS)


def test_ac_a7_axi_bs1_coincide_con_una_peticion_manual_de_bloque_1(tmp_path):
    w = make_axi_world(tmp_path)
    pid = load_axi(w)
    auto = simulate_axi(areq(pid))
    assert auto["meta"]["block_size"] > 1.5 and auto["meta"]["model_context"]["block_size"]["mode"] == "auto"
    bs1 = auto["meta"]["bs1"]
    assert isinstance(bs1, list) and len(bs1) == 1 and set(bs1[0]) == {"f", *AXI_KPIS}
    assert bs1[0]["f"] == auto["f_star"]
    manual = simulate_axi(areq(pid, block_size=1.0))
    assert manual["meta"]["bs1"] is None
    want = row_at(manual, auto["f_star"], AXI_KPIS)
    assert {k: bs1[0][k] for k in AXI_KPIS} == want
    assert row_at(auto, auto["f_star"], AXI_KPIS) != want


def test_ac_a7_bs1_es_null_con_bloque_1_manual_menor_que_1_o_fallback(tmp_path):
    wf, wa = make_fut_world(tmp_path / "f"), make_axi_world(tmp_path / "a")
    pf_id, pa_id = load_fut(wf), load_axi(wa)
    for bs in (1.0, 0.5, 1.0 + 5e-10):                                # ≤ 1 + 1e-9 (E12)
        assert simulate_propfirm(preq(pf_id, n_paths=100, block_size=bs))["meta"]["bs1"] is None, bs
        assert simulate_axi(areq(pa_id, n_paths=100, block_size=bs))["meta"]["bs1"] is None, bs
    short = load_fut(make_fut_world(tmp_path / "f2", n_days=20))
    assert simulate_propfirm(preq(short, n_paths=100))["meta"]["bs1"] is None          # fallback_iid
    for bs in (1.0 + 1e-6, 2.0):                                      # y por encima de 1 + 1e-9, sí
        assert len(simulate_propfirm(preq(pf_id, n_paths=100, block_size=bs))["meta"]["bs1"]) == 1


def test_ac_a7_modo_manual_mayor_que_1_tambien_calcula_la_sensibilidad_e15(tmp_path):
    wa = make_axi_world(tmp_path)
    pid = load_axi(wa)
    out = simulate_axi(areq(pid, block_size=3.0))
    assert out["meta"]["model_context"]["block_size"]["mode"] == "manual" and out["meta"]["block_size"] == 3.0
    assert len(out["meta"]["bs1"]) == 1
    manual1 = simulate_axi(areq(pid, block_size=1.0))
    assert {k: out["meta"]["bs1"][0][k] for k in AXI_KPIS} == row_at(manual1, out["meta"]["bs1"][0]["f"], AXI_KPIS)


def test_ac_a7_la_sensibilidad_reconstruye_las_dos_matrices_de_prop_firm_e4(tmp_path, monkeypatch):
    """E4: con bloque 1, ``idx_matrix`` e ``idx_ext`` (prolongación F2/fondeada) se rehacen con generadores NUEVOS. Si la
    sensibilidad reutilizara la ``idx_ext`` del bloque principal, P(F1+F2) saldría distinto del de una petición manual de
    bloque 1 (perfil ``cycle`` de dos fases: la F2 usa ``idx_ext``)."""
    w = make_csv_world(tmp_path)
    pid = load_csv(monkeypatch, w, 0.01)
    kw = dict(firm="ftmo_100k", n_paths=150, n_challenges=3, n_sequences=100)
    ctx = main._propfirm_setup(preq(pid, **kw))
    n, paths = ctx["n_blocks"], ctx["idx_matrix"].shape[0]
    assert ctx["bs"] > 1.5 and len(ctx["cfg"].phases) == 2
    _, idx1, ext1 = main._propfirm_index_matrices(n, paths, 1.0)
    _, idx2, ext2 = main._propfirm_index_matrices(n, paths, 1.0)
    np.testing.assert_array_equal(idx1, idx2)                          # generadores nuevos y deterministas
    np.testing.assert_array_equal(ext1, ext2)
    assert not np.array_equal(ext1, ctx["idx_ext"]) and not np.array_equal(idx1, ctx["idx_matrix"])
    # la matriz principal es la del bloque principal, con la misma construcción
    _, idx_m, ext_m = main._propfirm_index_matrices(n, paths, ctx["bs"])
    np.testing.assert_array_equal(idx_m, ctx["idx_matrix"])
    np.testing.assert_array_equal(ext_m, ctx["idx_ext"])
    cfg, req, f_grid = ctx["cfg"], preq(pid, **kw), ctx["f_grid"]
    manual = simulate_propfirm(preq(pid, block_size=1.0, **kw))
    for fi in range(len(f_grid)):
        want = row_at(manual, float(f_grid[fi]), PROP_KPIS)
        rep_ok, _, _ = main._propfirm_eval_f(ctx, req, cfg, False, fi, float(f_grid[fi]), idx1, ext1)
        assert {k: getattr(rep_ok, k) for k in PROP_KPIS} == want
    # la prueba tiene dientes: con la prolongación del bloque principal en vez de la de bloque 1, alguna f cambia
    cambia = False
    for fi in range(len(f_grid)):
        want = row_at(manual, float(f_grid[fi]), PROP_KPIS)
        rep_mixto, _, _ = main._propfirm_eval_f(ctx, req, cfg, False, fi, float(f_grid[fi]), idx1, ctx["idx_ext"])
        cambia |= {k: getattr(rep_mixto, k) for k in PROP_KPIS} != want
    assert cambia


def test_ac_a7_aviso_si_y_solo_si_se_supera_la_tolerancia_axi_de_extremo_a_extremo(tmp_path):
    w = make_axi_world(tmp_path)
    pid = load_axi(w)
    vistos = set()
    for tol in (1e-6, 0.02, 0.05, 0.1, 0.2, 0.5, 0.99):
        out = simulate_axi(areq(pid, n_paths=150, tol_ruin=tol))
        bs1 = out["meta"]["bs1"]
        assert len(bs1) == 1
        avisos = [x for x in out["warnings"] if x.startswith("Con bloque 1 (sin la dependencia entre bloques")]
        assert bool(avisos) == (bs1[0]["p_ruin"] > tol), (tol, bs1, avisos)
        assert len(avisos) <= 1
        vistos.add(bool(avisos))
        if avisos:
            assert "P(ruina)" in avisos[0] and f"{100 * bs1[0]['f']:.2f} %" in avisos[0]
            assert f"{100 * bs1[0]['p_ruin']:.1f} %" in avisos[0] and "por encima de tu tolerancia" in avisos[0]
    assert vistos == {True, False}                                    # el barrido de tolerancias ejerce los dos lados


def test_ac_a7_aviso_si_y_solo_si_se_supera_la_tolerancia_prop_firm_de_extremo_a_extremo(tmp_path):
    w = make_fut_world(tmp_path)
    pid = load_fut(w)
    vistos = set()
    for tol in (1e-6, 0.05, 0.2, 0.4, 0.5, 0.6, 0.99):
        out = simulate_propfirm(preq(pid, n_paths=150, tol_neg=tol))
        bs1 = out["meta"]["bs1"]
        assert len(bs1) == 1
        avisos = [x for x in out["warnings"] if x.startswith("Con bloque 1 (sin la dependencia entre bloques")]
        assert bool(avisos) == (bs1[0]["p_net_negative"] > tol), (tol, bs1, avisos)
        vistos.add(bool(avisos))
        if avisos:
            assert "P(neto<0)" in avisos[0] and f"{100 * bs1[0]['f']:.2f} %" in avisos[0]
    assert vistos == {True, False}


def test_ac_a7_el_umbral_del_aviso_es_estricto_y_el_texto_es_el_del_plan():
    """Unidad: ``_axi_bs1`` con caminos fabricados: 10 ruinas de 100 con tolerancia 10 % -> sin aviso (no estricto), 11 -> aviso."""
    def outs_at_factory(n_ruin):
        def outs_at(fi, idx):
            return [SimpleNamespace(outcome="ruin" if p < n_ruin else "pass", days=1.0, trades=1, max_dd=0.0, terminal=0.0)
                    for p in range(100)]
        return outs_at

    req = SimpleNamespace(n_paths=100, horizon_days=None, tol_ruin=0.10)
    f_grid = np.array([0.01, 0.02, 0.03])
    for n_ruin, hay_aviso in ((10, False), (11, True), (0, False)):
        rows, warns = main._axi_bs1(outs_at_factory(n_ruin), f_grid, 0.02, 120, req)
        assert rows == [{"f": 0.02, "p_pass": 1.0 - n_ruin / 100, "p_ruin": n_ruin / 100}]
        assert bool(warns) == hay_aviso
    w = main._bs1_warning(0.0115, "P(ruina)", 0.1234, 0.10)
    assert w == ("Con bloque 1 (sin la dependencia entre bloques de tu backtest), en f* = 1.15 % la P(ruina) sería 12.3 %, "
                 "por encima de tu tolerancia (10 %).")


@pytest.mark.parametrize("lens", ["propfirm", "axi"])
def test_ac_a7_e19_fstar_recomendacion_y_receta_bit_a_bit_iguales_con_y_sin_sensibilidad(tmp_path, monkeypatch, lens):
    if lens == "propfirm":
        pid = load_fut(make_fut_world(tmp_path))
        run, hook = (lambda: simulate_propfirm(preq(pid, n_paths=150))), "_propfirm_bs1"
    else:
        pid = load_axi(make_axi_world(tmp_path))
        run, hook = (lambda: simulate_axi(areq(pid, n_paths=150))), "_axi_bs1"
    con = run()
    assert con["meta"]["bs1"] is not None and "recipe" in con["meta"]
    with monkeypatch.context() as mp:
        mp.setattr(main, hook, lambda *a, **k: (None, []))             # SIN sensibilidad
        sin = run()
    assert sin["meta"]["bs1"] is None
    assert con["f_star"] == sin["f_star"]
    assert json.dumps(con["recommendation"], sort_keys=True) == json.dumps(sin["recommendation"], sort_keys=True)
    assert json.dumps(con["meta"]["recipe"], sort_keys=True) == json.dumps(sin["meta"]["recipe"], sort_keys=True)
    assert canon(con) == canon(sin)                                    # y todo lo demás, salvo bs1 y su aviso


def test_ac_a7_la_sensibilidad_no_consume_el_generador_de_la_peticion_principal(tmp_path, monkeypatch):
    """E4: el ``rng`` de la petición (el que el par 2D sigue usando para los gaps fijos) queda tal como lo dejó dibujar la
    ``idx_matrix``: la sensibilidad, que corre DESPUÉS, usa generadores nuevos y no lo toca."""
    w = make_csv_world(tmp_path)
    pid = load_csv(monkeypatch, w, 0.01)
    captured, real = [], main._propfirm_setup

    def wrapped(req):
        ctx = real(req)
        captured.append(ctx)
        return ctx

    monkeypatch.setattr(main, "_propfirm_setup", wrapped)
    out = simulate_propfirm(preq(pid, firm="ftmo_100k", n_paths=100, n_challenges=3, n_sequences=100))
    assert out["meta"]["bs1"] is not None                                  # la sensibilidad SÍ corrió
    ctx = captured[0]
    fresh, idx, _ = main._propfirm_index_matrices(ctx["n_blocks"], ctx["idx_matrix"].shape[0], ctx["bs"])
    np.testing.assert_array_equal(idx, ctx["idx_matrix"])
    assert ctx["rng"].bit_generator.state == fresh.bit_generator.state     # y el rng de la petición no avanzó más


def test_ac_a7_el_par_2d_hereda_el_bloque_automatico_y_no_lleva_sensibilidad(tmp_path, monkeypatch):
    w = make_csv_world(tmp_path)
    pid = load_csv(monkeypatch, w, 0.01)
    out = simulate_propfirm_pair(preq(pid, firm="ftmo_100k", n_paths=100, n_challenges=3, n_sequences=100))
    one = simulate_propfirm(preq(pid, firm="ftmo_100k", n_paths=100, n_challenges=3, n_sequences=100))
    assert out["meta"]["block_size"] == one["meta"]["block_size"] > 1.5
    assert "bs1" not in out["meta"]
    bsc = out["meta"]["model_context"]["block_size"]
    assert bsc["mode"] == "auto" and bsc["method"] == "politis_white" and bsc["series"] == "R"


# ===========================================================================
# AC-A8, E2, E10, E12: textos y campos del contexto de modelo
# ===========================================================================
def test_ac_a8_bs_explain_decide_solo_por_el_modo_e2():
    assert _bs_explain("auto") == ("auto", main._BS_REASONS["auto"], None)
    assert _bs_explain("manual") == ("manual", "manual: lo fijaste tú", None)
    mode, reason, cl = _bs_explain("fallback_iid")
    assert mode == "fallback_iid" and cl is None
    assert "menos de 25 bloques" in reason and "más del 10 %" in reason and reason.endswith("bloques sueltos")
    assert "Politis-White" in main._BS_REASONS["auto"] and "no depende de la f ni de la rejilla" in main._BS_REASONS["auto"]
    assert "(R)" in main._BS_REASONS["auto"]
    assert len({r for r in main._BS_REASONS.values()}) == 3
    with pytest.raises(ValueError):
        _bs_explain("calibrado")


def _blocks_syn(n=40):
    t0 = pd.Timestamp("2021-01-01")
    return [SimpleNamespace(start=t0 + pd.Timedelta(days=10 * i), end=t0 + pd.Timedelta(days=10 * i + 2))
            for i in range(n)]


def _note_text(mc, nid="tamano_bloque"):
    return {n["id"]: n["texto"] for n in mc["notes"]}[nid]


def test_ac_a8_el_modo_auto_nunca_se_etiqueta_fallback_aunque_no_haya_calibracion_e2():
    """El bug que E2 cierra: antes ``cal is None`` convertía una corrida automática en ``fallback_iid``."""
    mc = _model_context(lente="axi", blocks=_blocks_syn(), n_blocks=40, bs=3.06, bs_mode="auto",
                        bs_info={"series": "R", "n_excluded_terms": 0, "opt_m": 2, "m": 4},
                        extra={"min_dias": 5, "min_trades": 20})
    assert mc["block_size"]["mode"] == "auto" and mc["block_size"]["method"] == "politis_white"
    assert mc["block_size"]["reason"] == main._BS_REASONS["auto"]
    # con la firma vieja (cal/f_cal) el resultado es el mismo: se aceptan y se ignoran
    mc2 = _model_context(lente="axi", blocks=_blocks_syn(), n_blocks=40, bs=3.06, bs_mode="auto", cal=None, f_cal=0.01,
                         bs_info={"series": "R", "n_excluded_terms": 0, "opt_m": 2, "m": 4},
                         extra={"min_dias": 5, "min_trades": 20})
    assert mc2 == mc


def test_ac_a8_render_de_tamano_bloque_en_los_tres_modos_con_un_decimal_y_el_metodo():
    base = dict(lente="axi", blocks=_blocks_syn(), n_blocks=40, extra={"min_dias": 5, "min_trades": 20})
    auto = _model_context(bs=3.06, bs_mode="auto", bs_info={"series": "R", "n_excluded_terms": 2, "opt_m": 3, "m": 6}, **base)
    man = _model_context(bs=3.0, bs_mode="manual", **base)
    fb = _model_context(bs=1.0, bs_mode="fallback_iid", bs_info={"series": "R", "n_excluded_terms": 9, "opt_m": None, "m": None},
                        **base)
    assert _note_text(auto) == f"Grupos de 3.1 bloques de media (auto): {main._BS_REASONS['auto']}."
    assert _note_text(man) == "Grupos de 3.0 bloques de media (manual): manual: lo fijaste tú."
    assert _note_text(fb) == f"Grupos de 1.0 bloques de media (fallback_iid): {main._BS_REASONS['fallback_iid']}."
    assert "Politis-White" in _note_text(auto) and "{" not in _note_text(auto)
    for mc in (auto, man, fb):                                         # siempre exactamente un decimal
        assert re.match(r"Grupos de \d+\.\d bloques de media \(", _note_text(mc))
    assert _note_text(_model_context(bs=2.675, bs_mode="manual", **base)).startswith("Grupos de 2.7 bloques")   # y no '3'


def test_ac_a8_campos_nuevos_de_model_context_e10():
    base = dict(lente="propfirm", blocks=_blocks_syn(), n_blocks=40, cfg=main._firms["ftmo_50k"])
    auto = _model_context(bs=3.06, bs_mode="auto", bs_info={"series": "R", "n_excluded_terms": 2, "opt_m": 3, "m": 6}, **base)
    assert auto["block_size"] == {
        "value": 3.06, "mode": "auto", "reason": main._BS_REASONS["auto"], "has_real_clustering": None, "n_blocks": 40,
        "calibrated_at_f": None, "method": "politis_white", "series": "R", "n_excluded_terms": 2, "opt_m": 3, "m": 6}
    man = _model_context(bs=2.0, bs_mode="manual", **base)
    assert man["block_size"]["method"] == "manual"
    assert (man["block_size"]["series"], man["block_size"]["n_excluded_terms"], man["block_size"]["opt_m"],
            man["block_size"]["m"]) == (None, None, None, None)
    fb = _model_context(bs=1.0, bs_mode="fallback_iid", bs_info={"series": "R", "n_excluded_terms": 0, "opt_m": None, "m": None},
                        **base)
    assert fb["block_size"]["method"] == "fallback_iid" and fb["block_size"]["series"] == "R"
    assert fb["block_size"]["has_real_clustering"] is None and fb["block_size"]["calibrated_at_f"] is None


def test_ac_a8_la_nota_tamano_bloque_dice_el_metodo_y_los_dos_umbrales_e8():
    n = next(x for x in NOTES if x.id == "tamano_bloque")
    t = n.texto
    assert "Politis y White (2004, con la corrección de 2009)" in t and "riesgos (R)" in t
    assert "No depende de la f ni de la rejilla" in t
    assert "menos de 25 bloques" in t and "más del 10 %" in t           # los dos umbrales (E8)
    assert "vale 1 cuando no hay dependencia" in t
    assert n.en_esta_corrida == "Grupos de {bs} bloques de media ({bs_modo}): {bs_razon}."
    assert n.fuente == "src/risk/block_length.py; poc_web/api/main.py (_episode_r_series y bloque automático)"
    assert render_run(n, {"bs": "3.1", "bs_modo": "auto", "bs_razon": "x"}) == "Grupos de 3.1 bloques de media (auto): x."


def test_ac_a8_de_extremo_a_extremo_el_render_y_el_contexto_en_las_dos_lentes(tmp_path):
    """El bloque de la app es el b* de Politis-White de la serie R calculada a mano (oráculo independiente: fusión de
    intervalos y R por operación desde la tabla de trades), con ``opt_m`` y ``m`` del estimador."""
    wf = make_fut_world(tmp_path / "f")
    out = simulate_propfirm(preq(load_fut(wf), n_paths=100))
    bsc = out["meta"]["model_context"]["block_size"]
    assert bsc["value"] == out["meta"]["block_size"] and bsc["method"] == "politis_white" and bsc["series"] == "R"
    assert bsc["n_excluded_terms"] == 0 and isinstance(bsc["opt_m"], int) and isinstance(bsc["m"], int)
    assert bsc["has_real_clustering"] is None and bsc["calibrated_at_f"] is None
    txt = _note_text(out["meta"]["model_context"])
    assert txt.startswith(f"Grupos de {out['meta']['block_size']:.1f} bloques de media (auto): tamaño medio elegido con el método")
    pw = politis_white_info(_episodes(wf.trades, wf.pvs)[0])
    assert out["meta"]["block_size"] == pytest.approx(pw.b, rel=1e-12) and (bsc["opt_m"], bsc["m"]) == (pw.opt_m, pw.m)
    wa = make_axi_world(tmp_path / "a")
    outa = simulate_axi(areq(load_axi(wa), n_paths=100))
    mca = outa["meta"]["model_context"]
    assert mca["block_size"]["method"] == "politis_white" and "Grupos de" in _note_text(mca)
    pwa = politis_white_info(_episodes(wa.trades, wa.pvs)[0])
    assert outa["meta"]["block_size"] == pytest.approx(pwa.b, rel=1e-12)
    assert (mca["block_size"]["opt_m"], mca["block_size"]["m"]) == (pwa.opt_m, pwa.m)


# ===========================================================================
# E13: check_parity manda el bloque del baseline
# ===========================================================================
def test_e13_check_parity_envia_el_block_size_del_baseline(tmp_path, monkeypatch):
    here = ROOT / "poc_web" / "api"
    if not all((here / f).is_file() for f in ("check_parity.py", "baseline_f1.json", "baseline_f1_request.json")):
        pytest.skip("check_parity y su baseline no viajan en el export compartible")
    mod = _load_sibling("_tst_check_parity", "poc_web/api/check_parity.py")
    base = json.loads((here / "baseline_f1.json").read_text(encoding="utf-8"))
    req = json.loads((here / "baseline_f1_request.json").read_text(encoding="utf-8"))
    assert req["block_size"] == 0.0 and base["meta"]["block_size"] == 1.0          # el baseline se hizo con bloque 1
    tracked_before = (here / "parity_f1_new.json").read_bytes() if (here / "parity_f1_new.json").is_file() else None
    sent = []
    for k_it, baseline_meta in enumerate((base["meta"], {k: v for k, v in base["meta"].items() if k != "block_size"})):
        b = dict(base, meta=baseline_meta)
        d = tmp_path / f"here_{k_it}"
        d.mkdir()
        (d / "baseline_f1_request.json").write_text(json.dumps(req), encoding="utf-8")
        (d / "baseline_f1.json").write_text(json.dumps(b), encoding="utf-8")
        monkeypatch.setattr(mod, "HERE", d)
        monkeypatch.setattr(sys, "argv", ["check_parity"])

        class _Resp:
            def __init__(self, data):
                self.data = data

            def read(self):
                return json.dumps(self.data).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda url, timeout=0: _Resp(
            {"firms": [{"key": req["firm_key"], "account_size": 50000.0}]}))

        def post(url, payload):
            sent.append(dict(payload))
            if url.endswith("/api/portfolio/load"):
                return {"ok": True, "portfolio_id": "p", "guards": {}}
            return b                                                              # la simulación devuelve el baseline

        monkeypatch.setattr(mod, "_post", post)
        assert mod.main() == 0
    sim_payloads = [p for p in sent if "portfolio_id" in p]
    assert [p["block_size"] for p in sim_payloads] == [1.0, 1.0]                  # el del baseline; 1,0 si falta la clave
    assert all(p["block_size"] != 0.0 for p in sim_payloads)                      # nunca el 0 (automático) del request viejo
    # el test escribió el resultado en su carpeta temporal y no tocó el fichero versionado
    after = (here / "parity_f1_new.json").read_bytes() if (here / "parity_f1_new.json").is_file() else None
    assert after == tracked_before and (tmp_path / "here_0" / "parity_f1_new.json").is_file()
