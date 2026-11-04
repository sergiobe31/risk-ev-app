"""Guardrail ``entrada-4-bloques``, fase F2: lectura de ``.sqx`` vía fichas de símbolo, catálogo de patas,
compatibilidad bróker ↔ perfil y rechazos de salidas parciales.

Todo sintético y hermético: los ``.sqx`` se escriben con el writer de ``tests/test_sqx_trades.py`` (mismo patrón que
``test_cfd_sqx_legs.py``), el fichero de fichas del usuario y el de perfiles son temporales (I4: nunca los reales) y
los OHLC van a carpetas temporales (``main.SQX_OHLC_DIR_OVERRIDES``). Los tests que necesitan la demo
(``scripts/export_shareable/make_demo.py``) o el golden se omiten si no viajan (export compartible).

Mapa criterio -> test (el id del criterio está en el nombre):

- AC2.1 lectura contra ficha == lectura contra YAML ... ``test_ac2_1_*`` (demo, sintéticos y golden I2)
- AC2.2 catálogo ...................................... ``test_ac2_2_*``
- AC2.3 carga (sin ficha, dos brókers, ATM, órdenes compartidas, paridad) ... ``test_ac2_3_*``
- AC2.3bis (E11) desacuerdo con una ficha de usuario ..... ``test_ac2_3bis_*``
- AC2.4 compatibilidad perfil ↔ bróker ................ ``test_ac2_4_*``
- AC2.5 (E4) caché ..................................... ``test_ac2_5_*``
- AC2.6 ``draft-from-sqx`` ............................. ``test_ac2_6_*``
- AC2.7 los tests existentes pasan sin editar sus aserciones: lo comprueba la suite completa (ver el informe de la fase)

Fase F3 (correcciones de la revisión de código de la F2; H1/H2 viven en ``tests/test_sqx_trades.py`` y H3/H5 en
``test_symbols.py``): ``test_h1_*`` (``draft-from-sqx`` nunca da 500), ``test_h6_*`` (la paridad nunca tumba la carga),
``test_h7_*`` (motivo de "sin serie por minuto"), ``test_h8_*`` (paridad de motivos YAML ↔ ficha), ``test_h9_*`` (centinela
de la clave de caché).

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_sqx_fichas.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import types
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from poc_web.api import main, profiles, symbols
from sqx.d1.extract_sqx import SQXFormatError
from sqx.sqx_trades import (
    DEFAULT_CFD_COSTS_PATH,
    costs_path_for_symbol,
    read_trade_table,
    read_trade_table_spec,
)
from src.risk.futures_legs import strategy_from_trade_table

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("golden_sqx_fichas_I2.json")
REAL_USER_FILES = (ROOT / "config" / "symbols.user.yaml", ROOT / "config" / "propfirms.user.yaml")
ACCOUNT = 50000.0

_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer_fichas", ROOT / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx
patch_instrument_info = _W.patch_instrument_info

# sha256 de los 6 .sqx de la demo calculados ANTES de la fase F2 (con el ``write_sqx`` de la F1): el escritor con los
# valores por defecto (``portfolio_xml=None``, ``last_settings=None``) produce el MISMO ZIP.
DEMO_SHA256_ANTES_DE_F2 = {
    "demo_cfd/DEMO_sintetica_1_GBPUSD_ftmo.sqx": "024cd8556f27589896f1c3420d90c0d782d453a3224ef92edf982cba583242c5",
    "demo_cfd/DEMO_sintetica_2_GBPUSD_ftmo.sqx": "55fc51b10513cb3a6d955b8bbb3d8137b2fb8f3a3275505b262aee4f1fed81a3",
    "demo_cfd/DEMO_sintetica_3_AUDUSD_ftmo.sqx": "ac912c5a3884bd5b697c816333e5ad7a7562338103b3c8d3fa752edf800746b0",
    "demo_futuros/DEMO_sintetica_4_micro_nasdaq.sqx": "2fdf5d1cc6af47cd49677d67ad0b8e10ef256885b9195c8aa1959fb87553ddf2",
    "demo_futuros/DEMO_sintetica_5_micro_oro.sqx": "42a2809854ef2864fb63f21de87d127bf48c553d65afe92ec4bf2df8f6b5bb55",
    "demo_futuros/DEMO_sintetica_6_micro_eurusd.sqx": "c609ad40d4112139539c6767d74e8365899b991d30b6c5589efd9768e54378cf",
}
# Claves de ``meta`` que la fase F2 añade a las respuestas de ``simulate_propfirm`` (ninguna: la F2 no toca la meta).
NEW_META_KEYS: frozenset[str] = frozenset({"recipe", "bs1"})   # F4: la receta para el EA (adjudicado por el orquestador);
# bloque-politis-white (2026-10-21): ``bs1`` = la sensibilidad con bloque 1 (``null`` con bloque 1, como en el golden)


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
    """Cachés vacías, catálogo y OHLC aislados del disco real, fichas de usuario en un temporal."""
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")
    monkeypatch.setattr(main, "SQX_OHLC_DIR_OVERRIDES", {"axi": tmp_path / "ohlc_axi"})   # nunca el OHLC real de AXI
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")
    monkeypatch.setattr(main, "TRADES_DIR", tmp_path / "no_trades")      # las patas CSV reales del árbol principal
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "symbols.user.yaml")
    symbols._reload()


@pytest.fixture()
def user_file(tmp_path):
    return tmp_path / "symbols.user.yaml"


@pytest.fixture()
def profiles_file(tmp_path, monkeypatch):
    """Perfiles de usuario en un temporal (mismo fixture que ``user_file`` de ``test_profiles.py``)."""
    path = tmp_path / "propfirms.user.yaml"
    monkeypatch.setattr(profiles, "USER_PROFILES_PATH", path)
    main.reload_firms()
    yield path
    monkeypatch.undo()
    main.reload_firms()


@pytest.fixture()
def api():
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# .sqx sintéticos
# ---------------------------------------------------------------------------
def _f32(x: float) -> float:
    return float(np.float32(x))


def _trades(slot: int, seed: int, *, px0: float, unit: float, dists: tuple, pv: float, lots: float, comm: float,
            short: bool = True, n_days: int = 100, fixed_px: float | None = None) -> list[dict]:
    """Un trade por día hábil en [slot:00, slot+1:30] (hora del servidor), SL a una de ``dists``, 60 % ganadoras a
    1,5R (resto cierran en el SL con MAE = pérdida bruta). Precios redondeados a float32 para que PnL y MAE cuadren con
    lo que relee el extractor. Intradía: sin swap."""
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.bdate_range("2023-01-02", periods=n_days):
        dist = float(rng.choice(dists))
        open_px = _f32(px0 + unit * float(rng.integers(-40, 40))) if fixed_px is None else _f32(fixed_px)
        win = rng.random() < 0.6
        sign = -1.0 if short else 1.0
        sl_px = _f32(open_px - sign * dist)
        close_px = _f32(open_px + sign * 1.5 * dist) if win else sl_px
        gross = (close_px - open_px) * sign * pv * lots
        mae = (float(rng.uniform(0.0, 0.8)) * dist * pv * lots) if win else -gross
        out.append(dict(open=datetime(d.year, d.month, d.day, slot, 0), close=datetime(d.year, d.month, d.day, slot + 1, 30),
                        lots=lots, open_px=open_px, close_px=close_px, sl_px=sl_px, pnl=gross - comm * lots,
                        mae=max(mae, 0.0), flag=1))
    return out


def eur_ftmo(slot: int = 10, seed: int = 1, **kw) -> list[dict]:
    return _trades(slot, seed, px0=1.10, unit=0.0001, dists=(0.0010, 0.0015, 0.0020, 0.0025), pv=100000.0,
                   lots=0.5, comm=5.0, **kw)


def xau_axi(slot: int = 10, seed: int = 1, **kw) -> list[dict]:
    return _trades(slot, seed, px0=2000.0, unit=0.25, dists=(5.0, 7.5, 10.0, 12.5), pv=100.0, lots=0.1, comm=4.5, **kw)


def eur_axi(slot: int = 10, seed: int = 1, **kw) -> list[dict]:
    return _trades(slot, seed, px0=1.10, unit=0.0001, dists=(0.0010, 0.0015, 0.0020, 0.0025), pv=100000.0,
                   lots=0.1, comm=4.5, **kw)


def mnq(slot: int = 10, seed: int = 1, **kw) -> list[dict]:
    return _trades(slot, seed, px0=18000.0, unit=0.25, dists=(10.0, 15.0, 20.0, 25.0), pv=2.0, lots=1.0, comm=1.12,
                   short=False, **kw)


def w_eur_ftmo(path: Path, trades: list[dict], **kw) -> str:
    return str(write_sqx(path, trades, symbol="EURUSD_ftmo", instrument="EURUSD_cfd_ftmo", pv=100000.0, tick=0.0001,
                         capital=ACCOUNT, **kw))


def w_xau_axi(path: Path, trades: list[dict], **kw) -> str:
    return str(write_sqx(path, trades, symbol="XAUUSD_axi", instrument="XAUUSD_mt5_axi", pv=100.0, tick=0.01,
                         capital=ACCOUNT, **kw))


def w_eur_axi(path: Path, trades: list[dict], **kw) -> str:
    return str(write_sqx(path, trades, symbol="EURUSD_axi", instrument="EURUSD_axi", pv=100000.0, tick=0.0001,
                         capital=ACCOUNT, **kw))


def w_mnq(path: Path, trades: list[dict], **kw) -> str:
    return str(write_sqx(path, trades, capital=ACCOUNT, **kw))             # @micro_nasdaq por defecto


def _root(tmp_path: Path) -> Path:
    return tmp_path / "extra" / "bank"


def _eur_ftmo_legs(tmp_path: Path, n: int = 3) -> list[str]:
    return [w_eur_ftmo(_root(tmp_path) / f"EUR F{k}.sqx", eur_ftmo(10 + 2 * k, 100 + k)) for k in range(n)]


def _xau_axi_legs(tmp_path: Path, n: int = 3) -> list[str]:
    return [w_xau_axi(_root(tmp_path) / f"XAU A{k}.sqx", xau_axi(10 + 2 * k, 200 + k)) for k in range(n)]


def _mnq_legs(tmp_path: Path, n: int = 3) -> list[str]:
    return [w_mnq(_root(tmp_path) / f"MNQ M{k}.sqx", mnq(10 + 2 * k, 300 + k)) for k in range(n)]


def _send(c, method, url, obj=None):
    return c.request(method, url, content=json.dumps(obj), headers={"content-type": "application/json"})


def _load(legs, firm_key="", **kw) -> dict:
    return main.portfolio_load(main.PortfolioLoadRequest(
        legs=[str(p) for p in legs], account_size=ACCOUNT, firm_key=firm_key, **kw))


def _load_http(api, legs, firm_key=""):
    r = api.post("/api/portfolio/load", json={"legs": legs, "account_size": ACCOUNT, "firm_key": firm_key})
    assert r.status_code == 200, r.text
    return r.json()


def _sim_http(api, pid, firm_key, axis="servidor"):
    return api.post("/api/simulate/propfirm", json={
        "portfolio_id": pid, "firm_key": firm_key, "f_lo": 0.0025, "f_hi": 0.005, "f_step": 0.0025, "n_paths": 100,
        "n_challenges": 3, "n_sequences": 100, "day_axis": axis})


def _make_demo(out_dir: Path):
    script = ROOT / "scripts" / "export_shareable" / "make_demo.py"
    if not script.is_file():
        pytest.skip("make_demo.py ausente (export compartible)")
    sys.path.insert(0, str(script.parent))
    try:
        import make_demo
        return make_demo.generate(ROOT, out_dir)
    finally:
        sys.path.pop(0)


def _mk_user_cfd(api, *, sym="ZZZ_mi", pv=100000.0, tick=0.0001, **kw):
    """Bróker CFD del usuario ``mi_cfd`` (con carpeta de OHLC) y una ficha de usuario ``sym`` en él."""
    r = _send(api, "POST", "/api/symbols/brokers", {
        "id": "mi_cfd", "display_name": "Mi CFD", "company": "Mi Compañía", "asset_class": "cfd",
        "clock": "ny_plus_7", "ohlc_dir": "risk_app/inputs/ohlc_mi"})
    assert r.status_code == 200, r.text
    body = dict(broker="mi_cfd", sqx_symbol=sym, ticker=sym, point_value_usd=pv, lot_step=0.01, min_lot=0.01,
                commission={"per_lot_rt": 5.0}, spread_px=3 * tick, tick_size=tick)
    body.update(kw)
    r = _send(api, "POST", "/api/symbols", body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# AC2.1: lectura contra ficha == lectura contra el YAML
# ---------------------------------------------------------------------------
def _assert_parity(path: Path) -> None:
    sym = main._sqx_symbol(Path(path))
    sheet = symbols.resolve(sym)
    assert sheet is not None and sheet.official, sym
    costs = costs_path_for_symbol(sym)
    df_y, meta_y = read_trade_table(path, costs_path=costs)
    df_s, meta_s = read_trade_table_spec(path, symbols.reader_spec(sheet))
    pd.testing.assert_frame_equal(df_y, df_s)
    assert meta_y == meta_s, path


def test_ac2_1_las_6_sqx_de_la_demo_leen_igual_contra_la_ficha_que_contra_el_yaml(tmp_path):
    _make_demo(tmp_path / "demo")
    # las 6 de FTMO (las de AXI, desde 2026-10-24, no están en los YAML de costes: no hay camino YAML que comparar)
    files = sorted(p for d in ("demo_cfd", "demo_futuros") for p in (tmp_path / "demo" / d).glob("*.sqx"))
    assert len(files) == 6
    for f in files:
        _assert_parity(f)


def test_ac2_1_los_sqx_de_la_demo_son_los_mismos_bytes_que_antes_de_la_f2(tmp_path):
    """``write_sqx`` con ``portfolio_xml=None`` y ``last_settings=None`` produce el MISMO ZIP que antes de la F2:
    los 6 ``.sqx`` de la demo (que escribe ese helper) conservan su sha256."""
    _make_demo(tmp_path / "demo")
    got = {str(p.relative_to(tmp_path / "demo")): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((tmp_path / "demo").rglob("*.sqx")) if p.parent.name != "demo_axi"}   # AXI: 2026-10-24
    assert got == DEMO_SHA256_ANTES_DE_F2


def _eur_swap_trades(slot: int, seed: int, n_days: int = 100) -> list[dict]:
    """EURUSD_ftmo corto con noches de swap (abono de 0,10 USD/lote/noche cada 10.º trade): como el sintético de
    ``test_cfd_sqx_legs.py``."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    out = []
    for k, d in enumerate(days):
        dist = float(rng.choice([0.0010, 0.0015, 0.0020, 0.0025]))
        open_px = _f32(1.10 + 0.0001 * float(rng.integers(-40, 40)))
        win = rng.random() < 0.6
        sl_px = _f32(open_px + dist)
        close_px = _f32(open_px - 1.5 * dist) if win else sl_px
        gross = (open_px - close_px) * 100000.0 * 0.5
        mae = (float(rng.uniform(0.0, 0.8)) * dist * 100000.0 * 0.5) if win else -gross
        overnight = k % 10 == 9 and k + 2 < n_days
        swap = 0.10 * 2 * 0.5 if overnight else 0.0
        cd = days[k + 2] if overnight else d
        out.append(dict(open=datetime(d.year, d.month, d.day, slot, 0), close=datetime(cd.year, cd.month, cd.day, slot + 1, 30),
                        lots=0.5, open_px=open_px, close_px=close_px, sl_px=sl_px, pnl=gross - 5.0 * 0.5 + swap,
                        mae=max(mae, 0.0), flag=1))
    return out


def test_ac2_1_sinteticos_cfd_futuros_y_oro_leen_igual_contra_la_ficha_que_contra_el_yaml(tmp_path):
    cfd = w_eur_ftmo(tmp_path / "c.sqx", _eur_swap_trades(10, 7))                    # comisión fija y swap por noche
    fut = w_mnq(tmp_path / "f.sqx", mnq(10, 8))                                       # futuros: pasos de lote 1 y 1
    gold = str(_W.write_gold(tmp_path / "g.sqx", [                                   # comisión proporcional (PercentageBased)
        _W.gold_long(d, px, win=(d % 2 == 0))
        for d, px in zip((4, 5, 6, 7, 8), (1200.0, 1800.0, 2400.0, 3000.0, 4096.0))]))
    us100 = str(write_sqx(                                                            # índice sin comisión, pv = 1
        tmp_path / "u.sqx", _trades(10, 9, px0=18000.0, unit=0.25, dists=(10.0, 15.0), pv=1.0, lots=0.1, comm=0.0,
                                    short=False, n_days=30),
        symbol="USATECHIDXUSD_ftmo(2)", instrument="US100_cfd_ftmo", pv=1.0, tick=1.0))
    for p in (cfd, fut, gold, us100):
        _assert_parity(Path(p))
    assert symbols.resolve("USATECHIDXUSD_ftmo(2)").commission == {"per_lot_rt": 0.0}


def test_ac2_1_serie_por_minuto_por_ficha_identica_a_la_del_yaml_en_ftmo(tmp_path, monkeypatch):
    """Bit-identidad de la serie por minuto: ``_sqx_leg_minutes_sheet`` (ficha oficial de ``ftmo_cfd``) da exactamente
    lo mismo que ``_sqx_leg_minutes`` (YAML de costes) sobre el mismo ``.sqx`` y el mismo OHLC."""
    tr = _trades(10, 41, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=25,
                 fixed_px=1.125)
    ohlc_dir = tmp_path / "ohlc_ftmo"
    _write_flat_ohlc(ohlc_dir / "EURUSD_ftmo.csv", tr, 1.125)           # el nombre sale del YAML (ohlc_m1)
    monkeypatch.setattr(main, "SQX_OHLC_DIR", ohlc_dir)                    # la constante que sustituyen los tests de FTMO
    p = Path(w_eur_ftmo(_root(tmp_path) / "E.sqx", tr))
    df, meta = read_trade_table(p, costs_path=DEFAULT_CFD_COSTS_PATH)
    _, fleg = strategy_from_trade_table(df, meta, key=p.name)
    sheet = symbols.resolve("EURUSD_ftmo")
    broker = symbols.load_registry().brokers[sheet.broker]
    to_axis = main._sqx_axis_transform(-1.0)                              # eje FTMO: la misma transformación en ambos
    old_series, old_diag, old_reason = main._sqx_leg_minutes(df, meta, fleg, "cfd", DEFAULT_CFD_COSTS_PATH, to_axis)
    new_series, new_diag, new_reason = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, broker, to_axis)
    assert old_reason is None and new_reason is None and len(old_series) == len(new_series) == 25
    assert old_diag == new_diag and old_diag["coherencia"] == pytest.approx(1.0)
    for i in old_series:
        for a, b in zip(old_series[i], new_series[i]):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b
    # y con los motivos de "sin serie" de siempre para una ficha oficial
    gbp = symbols.resolve("GBPUSD_ftmo")                                  # sin ohlc_m1 en el YAML
    r_old = main._sqx_leg_minutes(df, {**meta, "symbol": "GBPUSD_ftmo"}, fleg, "cfd", DEFAULT_CFD_COSTS_PATH, to_axis)
    r_new = main._sqx_leg_minutes_sheet(df, {**meta, "symbol": "GBPUSD_ftmo"}, fleg, gbp, broker, to_axis)
    assert r_old == r_new == (None, None, "sin ohlc_m1 en el YAML de costes para GBPUSD_ftmo")


def _first_diff(a, b, path: str = "") -> str | None:
    """Primera diferencia entre dos estructuras JSON (para un mensaje de fallo legible)."""
    if type(a) is not type(b):
        return f"{path}: tipo {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: claves {sorted(set(a) ^ set(b))}"
        for k in sorted(a):
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: longitud {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


GOLDEN_CASES = {
    "demo_cfd__ftmo_50k": ("demo_cfd", "ftmo_50k", "ftmo"),
    "demo_futuros__ftmo_futures_growth_50k": ("demo_futuros", "ftmo_futures_growth_50k", None),
}


def test_ac2_1_golden_i2_las_respuestas_de_la_demo_no_cambian(tmp_path):
    """I2: las dos respuestas completas de ``simulate_propfirm`` (demo_cfd + ``ftmo_50k`` y demo_futuros +
    ``ftmo_futures_growth_50k``, 300 caminos) son IGUALES al golden tomado ANTES de la F2, salvo las claves nuevas de
    ``meta`` (``NEW_META_KEYS``, que el test lista explícitamente; hoy ninguna)."""
    if not GOLDEN.is_file():
        pytest.skip("golden I2 ausente (export compartible)")
    _make_demo(tmp_path / "demo")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(golden) == set(GOLDEN_CASES)
    for key, (folder, firm, axis) in GOLDEN_CASES.items():
        legs = sorted(str(p) for p in (tmp_path / "demo" / folder).glob("*.sqx"))
        assert len(legs) == 3
        load = _load(legs, firm_key=firm)
        assert load["ok"], load
        # bloque-politis-white (E7, opción 1): el golden se tomó con el bloque automático de la regla vieja (``meta.block_size``
        # del propio golden: 1,0 en las dos demos); se fija ese bloque para conservar TODAS las cifras bit a bit.
        kw = dict(portfolio_id=load["portfolio_id"], firm_key=firm, f_lo=0.0025, f_hi=0.02, f_step=0.0025, n_paths=300,
                  block_size=golden[key]["meta"]["block_size"])
        if axis is not None:
            kw["day_axis"] = axis
        got = main.simulate_propfirm(main.PropfirmSimRequest(**kw))
        got["meta"].pop("elapsed_s")
        nuevas = set(got["meta"]) - set(golden[key]["meta"])
        assert nuevas == NEW_META_KEYS, f"{key}: claves nuevas de meta no listadas: {sorted(nuevas - NEW_META_KEYS)}"
        assert got["meta"].pop("bs1") is None                     # con bloque 1 la sensibilidad no existe (la cifra sería la misma)
        got["meta"].pop("recipe")
        got = json.loads(json.dumps(got))                         # JSON puro (claves str, listas), como el golden
        # Lo único que el guardrail cambia POR DISEÑO en la respuesta es la descripción del bloque (``model_context.block_size``:
        # modo, razón y campos nuevos) y su nota ``tamano_bloque``: se comprueba aparte y se quita de los dos lados; el resto
        # de ``model_context`` (días de backtest, horizonte, demás notas) y TODA la respuesta siguen comparándose entera.
        gold = json.loads(json.dumps(golden[key]))
        gb, nb = gold["meta"]["model_context"].pop("block_size"), got["meta"]["model_context"].pop("block_size")
        assert nb["value"] == gb["value"] == kw["block_size"] and nb["mode"] == "manual" and nb["n_blocks"] == gb["n_blocks"]
        for resp in (gold, got):
            resp["meta"]["model_context"]["notes"] = [n for n in resp["meta"]["model_context"]["notes"]
                                                      if n["id"] != "tamano_bloque"]
        assert _first_diff(gold, got) is None, f"{key}: {_first_diff(gold, got)}"


# ---------------------------------------------------------------------------
# AC2.2: catálogo
# ---------------------------------------------------------------------------
def _by_name() -> dict[str, dict]:
    return {l["name"]: l for l in main._scan_legs()}


def test_ac2_2_un_sqx_ftmo_da_ficha_ok_y_su_broker(tmp_path):
    w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    w_mnq(_root(tmp_path) / "MNQ.sqx", mnq())
    by = _by_name()
    assert (by["EUR"]["ficha"], by["EUR"]["broker"], by["EUR"]["asset"], by["EUR"]["motivo"]) == \
        ("ok", "ftmo_cfd", "cfd", None)
    assert (by["MNQ"]["ficha"], by["MNQ"]["broker"], by["MNQ"]["asset"], by["MNQ"]["motivo"]) == \
        ("ok", "ftmo_futures", "futures", None)
    assert by["EUR"]["kind"] == by["MNQ"]["kind"] == main.KIND_SQX == "sqx_futures"
    assert set(by["EUR"]) == {"path", "folder", "name", "kind", "asset", "broker", "ficha", "motivo"}


def test_ac2_2_un_sqx_sintetico_de_axi_da_broker_axi_y_asset_cfd(tmp_path):
    path = write_sqx(_root(tmp_path) / "XAU.sqx", xau_axi(), symbol="XAUUSD_axi", instrument="XAUUSD_mt5_axi",
                     pv=100.0, tick=0.01)
    e = _by_name()["XAU"]
    assert (e["broker"], e["asset"], e["ficha"], e["motivo"]) == ("axi", "cfd", "ok", None)
    assert e["path"] == str(path)


def test_ac2_2_un_simbolo_desconocido_aparece_con_sin_ficha_i7(tmp_path):
    write_sqx(_root(tmp_path) / "RARO.sqx", eur_ftmo(), symbol="SIMBOLO_RARO", instrument="X", pv=100000.0, tick=0.0001)
    e = _by_name()["RARO"]
    assert (e["ficha"], e["broker"], e["asset"], e["motivo"]) == ("sin_ficha", None, None, None)


def test_ac2_2_un_sqx_ilegible_no_aparece(tmp_path):
    root = _root(tmp_path)
    root.mkdir(parents=True)
    (root / "roto.sqx").write_bytes(b"esto no es un zip")
    with zipfile.ZipFile(root / "sin_settings.sqx", "w") as zf:                       # ZIP válido sin settings.xml
        zf.writestr("otra_cosa.xml", "<a/>")
    w_eur_ftmo(root / "EUR.sqx", eur_ftmo())
    names = set(_by_name())
    assert names == {"EUR"}


def test_ac2_2_ficha_no_usable_y_ficha_oculta(tmp_path, api):
    write_sqx(_root(tmp_path) / "DAX.sqx", eur_axi(), symbol="DEUIDXEUR_axi", instrument="DAX_axi", pv=100000.0,
              tick=0.0001)
    e = _by_name()["DAX"]
    assert (e["ficha"], e["broker"], e["asset"], e["motivo"]) == ("no_usable", "axi", "cfd", "divisa no USD (v1)")
    # ocultar la ficha de EURUSD_ftmo: el .sqx sigue en el catálogo, ahora sin ficha, y el motivo lo dice
    w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    r = _send(api, "POST", "/api/symbols/visibility", {"keys": ["ftmo_cfd/EURUSD_ftmo"], "hidden": True})
    assert r.status_code == 200
    e = _by_name()["EUR"]
    assert (e["ficha"], e["broker"], e["asset"]) == ("sin_ficha", None, None)
    assert "oculta" in e["motivo"] and "EURUSD_ftmo" in e["motivo"]
    _send(api, "POST", "/api/symbols/visibility", {"keys": ["ftmo_cfd/EURUSD_ftmo"], "hidden": False})
    assert _by_name()["EUR"]["ficha"] == "ok"


def test_ac2_2_catalog_entry_de_symbols_y_las_patas_csv_no_cambian(tmp_path, monkeypatch, api):
    reg = symbols.load_registry()
    assert symbols.catalog_entry("EURUSD_ftmo", reg) == {"ficha": "ok", "broker": "ftmo_cfd", "asset": "cfd", "motivo": None}
    assert symbols.catalog_entry("nada_a")["ficha"] == "sin_ficha"
    trades = tmp_path / "trades"
    (trades / "dir").mkdir(parents=True)
    (trades / "dir" / "a.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(main, "TRADES_DIR", trades)
    w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    legs = api.get("/api/catalog").json()["legs"]
    csv_leg = next(l for l in legs if l["kind"] == "csv")
    assert csv_leg == {"path": str(trades / "dir" / "a.csv"), "folder": "dir", "name": "a", "kind": "csv"}
    assert [l["broker"] for l in legs if l["kind"] == "sqx_futures"] == ["ftmo_cfd"]


# ---------------------------------------------------------------------------
# AC2.3: carga
# ---------------------------------------------------------------------------
def test_ac2_3_sin_ficha_es_error_de_pata_con_el_texto(tmp_path):
    p = write_sqx(_root(tmp_path) / "RARO.sqx", eur_ftmo(), symbol="SIMBOLO_RARO", instrument="X", pv=100000.0,
                  tick=0.0001)
    msg = ("RARO.sqx: sin ficha para el símbolo 'SIMBOLO_RARO': da de alta el símbolo (pestaña Símbolos); "
           "fuera de la allowlist de fichas")
    with pytest.raises(ValueError) as ei:
        main._load_leg(str(p), ACCOUNT, 0.005)
    assert str(ei.value) == msg
    res = _load([str(p)])
    assert not res["ok"] and res["legs"][0]["status"] == "error" and res["legs"][0]["error"] == msg
    assert main._portfolios[res["portfolio_id"]]["broker"] is None


def test_ac2_3_con_ficha_no_usable_el_error_es_el_motivo(tmp_path):
    p = write_sqx(_root(tmp_path) / "DAX.sqx", eur_axi(), symbol="DEUIDXEUR_axi", instrument="DAX_axi", pv=100000.0,
                  tick=0.0001)
    with pytest.raises(ValueError, match=r"DAX\.sqx: ficha no usable para el símbolo 'DEUIDXEUR_axi': divisa no USD \(v1\)"):
        main._load_leg(str(p), ACCOUNT, 0.005)


def test_ac2_3_con_la_ficha_oculta_el_error_lo_dice(tmp_path, api):
    p = w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    _send(api, "POST", "/api/symbols/visibility", {"keys": ["ftmo_cfd/EURUSD_ftmo"], "hidden": True})
    with pytest.raises(ValueError, match=r"sin ficha para el símbolo 'EURUSD_ftmo'.*está oculta"):
        main._load_leg(p, ACCOUNT, 0.005)


def test_ac2_3_un_sqx_ilegible_conserva_el_error_concreto_del_lector(tmp_path):
    root = _root(tmp_path)
    root.mkdir(parents=True)
    (root / "roto.sqx").write_bytes(b"esto no es un zip")
    with pytest.raises(ValueError, match=r"roto\.sqx: no es un ZIP válido"):
        main._load_leg(str(root / "roto.sqx"), ACCOUNT, 0.005)


def test_ac2_3_patas_de_dos_brokers_dan_tipo_patas_error_y_ninguna_se_carga(tmp_path):
    cfd = w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    axi = w_xau_axi(_root(tmp_path) / "XAU.sqx", xau_axi())
    res = _load([cfd, axi])
    g = res["guards"]["tipo_patas"]
    assert g["status"] == "error"
    assert g["detail"] == "patas de brókers distintos (axi, ftmo_cfd): carga una cartera de un solo bróker"
    assert not res["ok"] and [l["status"] for l in res["legs"]] == ["error", "error"]
    assert all(l["error"] == g["detail"] for l in res["legs"])
    assert main._portfolios[res["portfolio_id"]]["legs"] == [] and main._portfolios[res["portfolio_id"]]["kind"] == "mixed"


def test_ac2_3_futuros_con_cfd_conserva_la_frase_de_siempre_ademas_de_los_brokers(tmp_path):
    res = _load([w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo()), w_mnq(_root(tmp_path) / "MNQ.sqx", mnq())])
    d = res["guards"]["tipo_patas"]["detail"]
    assert res["guards"]["tipo_patas"]["status"] == "error"
    assert d.startswith("patas de brókers distintos (ftmo_cfd, ftmo_futures): carga una cartera de un solo bróker")
    assert "futuros y de CFD" in d                       # test_cartera_mixta_futuros_y_cfd_es_error (sin editar)


def test_ac2_3_las_patas_sin_ficha_no_cuentan_para_la_guarda_de_brokers(tmp_path):
    ok = w_eur_ftmo(_root(tmp_path) / "EUR.sqx", eur_ftmo())
    raro = write_sqx(_root(tmp_path) / "RARO.sqx", eur_ftmo(), symbol="SIMBOLO_RARO", instrument="X", pv=100000.0,
                     tick=0.0001)
    res = _load([ok, raro])
    assert res["guards"]["tipo_patas"]["status"] == "ok"
    st = {l["pata"]: l["status"] for l in res["legs"]}
    assert st == {"EUR.sqx": "ok", "RARO.sqx": "error"} and not res["ok"]
    assert main._portfolios[res["portfolio_id"]]["broker"] == "ftmo_cfd"           # el de la pata que sí tiene ficha


def test_ac2_3_el_detalle_ok_de_tipo_patas_no_cambia_en_ftmo_y_es_generico_en_otros_brokers(tmp_path):
    assert _load(_eur_ftmo_legs(tmp_path))["guards"]["tipo_patas"] == {
        "status": "ok", "detail": "patas .sqx nativas de CFD FTMO (valle por ventanas con swap, sizing Opción A con lotes de 0,01)"}
    assert _load(_mnq_legs(tmp_path))["guards"]["tipo_patas"] == {
        "status": "ok", "detail": "patas .sqx nativas de futuros (valle por ventanas, sizing Opción A)"}
    g = _load(_xau_axi_legs(tmp_path))["guards"]["tipo_patas"]
    assert g["status"] == "ok" and "AXI Select (MT5)" in g["detail"] and "CFD" in g["detail"]
    assert "FTMO" not in g["detail"]


def test_ac2_3_atm_activado_es_error_de_pata(tmp_path):
    atm = '<Strategy><ATMs source="x" enable="true"/></Strategy>'
    p = w_eur_ftmo(_root(tmp_path) / "ATM.sqx", eur_ftmo(), portfolio_xml=atm)
    with pytest.raises(ValueError, match=r"ATM\.sqx: salidas parciales \(ATM\) activadas en la estrategia: no soportadas en la v1"):
        main._load_leg(p, ACCOUNT, 0.005)
    res = _load([p, w_eur_ftmo(_root(tmp_path) / "OK.sqx", eur_ftmo(12, 5))])
    st = {l["pata"]: l["status"] for l in res["legs"]}
    assert st == {"ATM.sqx": "error", "OK.sqx": "ok"} and not res["ok"]
    # con el ATM apagado (la etiqueta real de SQX) se carga
    off = w_eur_ftmo(_root(tmp_path) / "OFF.sqx", eur_ftmo(), portfolio_xml='<Strategy><ATMs source="none" enable="false" /></Strategy>')
    assert main._load_leg(off, ACCOUNT, 0.005)["name"] == "OFF.sqx"


def test_ac2_3_dos_ordenes_con_la_misma_colocacion_y_ejecucion_son_error_de_pata(tmp_path):
    tr = eur_ftmo()
    tr.append(dict(tr[0], close=datetime(2023, 1, 3, 15, 0)))        # otra salida de la misma entrada (colocación y ejecución iguales)
    p = w_eur_ftmo(_root(tmp_path) / "DUP.sqx", tr)
    with pytest.raises(ValueError, match=r"DUP\.sqx: 2 órdenes comparten hora de colocación y de ejecución \(¿salidas parciales\?\): no soportado en la v1"):
        main._load_leg(p, ACCOUNT, 0.005)
    assert _load([p])["legs"][0]["status"] == "error"


# --- paridad con la regla de riesgo fijo (E16) --------------------------------------------------------------------
RISKED = 57.5
LAST_FIXED_AMOUNT = f"""<root><Methods>
  <Method type="FixedAmount" use="true"><Params>
    <Param key="#RiskedMoney#" value="{RISKED}">{RISKED}</Param><Param key="#Decimals#" value="2">2</Param>
    <Param key="#LotsIfNoMM#" value="0.01">0.01</Param><Param key="#MaxLots#" value="10">10</Param>
  </Params></Method></Methods></root>"""


def _fixed_amount_trades(n: int = 60, *, huge: int = 4) -> list[dict]:
    """Cortos EURUSD_ftmo cuyo lote sigue la regla del EA: ``floor(R / (|open − SL| · pv · 0,01)) · 0,01`` y, si el
    ``floor`` da 0, ``LotsIfNoMM`` (0,01). Cada ``huge``-ésimo trade tiene un SL tan lejano que va a ``LotsIfNoMM``."""
    rng = np.random.default_rng(5)
    dists = (0.0010, 0.0015, 0.0020, 0.0035, 0.0012)          # R / (riesgo·0,01) = 57,5; 38,3; 28,8; 16,4; 47,9: lejos de un entero
    days = pd.bdate_range("2023-01-02", periods=n)
    out = []
    for k, d in enumerate(days):
        dist = 0.5 if k % huge == huge - 1 else float(rng.choice(dists))
        open_px = _f32(1.10 + 0.0001 * float(rng.integers(-40, 40)))
        sl_px = _f32(open_px + dist)
        risk = abs(open_px - sl_px) * 100000.0
        steps = int(np.floor(RISKED / (risk * 0.01) + 1e-9))
        lots = round(steps * 0.01, 10) if steps > 0 else 0.01
        win = rng.random() < 0.5
        close_px = _f32(open_px - 1.5 * dist) if win else sl_px
        gross = (open_px - close_px) * 100000.0 * lots
        mae = (float(rng.uniform(0.0, 0.8)) * dist * 100000.0 * lots) if win else -gross
        out.append(dict(open=datetime(d.year, d.month, d.day, 10, 0), close=datetime(d.year, d.month, d.day, 11, 30),
                        lots=lots, open_px=open_px, close_px=close_px, sl_px=sl_px, pnl=gross - 5.0 * lots,
                        mae=max(mae, 0.0), flag=1))
    return out


def test_ac2_3_paridad_fixed_amount_lotes_que_siguen_la_regla_dan_1(tmp_path):
    tr = _fixed_amount_trades()
    p = w_eur_ftmo(_root(tmp_path) / "FA.sqx", tr, last_settings=LAST_FIXED_AMOUNT)
    leg = main._load_leg(p, ACCOUNT, 0.005)
    assert leg["mm_backtest"] == {"method": "FixedAmount", "params": {
        "RiskedMoney": RISKED, "Decimals": 2.0, "LotsIfNoMM": 0.01, "MaxLots": 10.0}}
    par = leg["paridad"]
    assert par["risked_money"] == RISKED and par["n"] == len(tr)
    assert par["frac_ok"] == 1.0 and par["frac_ok_sin_lifm"] == 1.0
    assert par["n_lifm"] == 15                                   # los 15 trades (k % 4 == 3 de 60) con el SL lejano
    # A se cancela: con otro tamaño de cuenta la paridad es la misma (la recarga lee de la caché de otra clave)
    other = main._load_leg(p, 100000.0, 0.005)
    assert other["paridad"] == par
    # se expone también en el dict de la ficha y el bróker
    assert leg["broker"] == "ftmo_cfd" and leg["ficha"]["official"] is True and leg["ficha"]["sqx_symbol"] == "EURUSD_ftmo"


def test_ac2_3_paridad_con_un_lote_alterado_baja_de_1(tmp_path):
    tr = _fixed_amount_trades()
    alterado = [dict(t) for t in tr]
    # cambia el lote del trade 1 (no es de LotsIfNoMM) conservando coherencia de PnL/MAE con el nuevo lote
    i = 1
    t = alterado[i]
    factor = 2.0
    t["lots"] = round(t["lots"] * factor, 10)
    t["pnl"] = (t["pnl"] + 5.0 * tr[i]["lots"]) * factor - 5.0 * t["lots"]
    t["mae"] = t["mae"] * factor
    p = w_eur_ftmo(_root(tmp_path) / "FA2.sqx", alterado, last_settings=LAST_FIXED_AMOUNT)
    par = main._load_leg(p, ACCOUNT, 0.005)["paridad"]
    assert par["frac_ok"] == pytest.approx((len(tr) - 1) / len(tr)) and par["frac_ok"] < 1.0
    assert par["frac_ok_sin_lifm"] < 1.0 and par["n_lifm"] == 15


def test_ac2_3_paridad_solo_con_fixed_amount_y_riesgo_positivo(tmp_path):
    base = _fixed_amount_trades(20)
    atr = LAST_FIXED_AMOUNT.replace('type="FixedAmount"', 'type="ATRRiskBasedSizing"')
    leg = main._load_leg(w_eur_ftmo(_root(tmp_path) / "ATR.sqx", base, last_settings=atr), ACCOUNT, 0.005)
    assert leg["mm_backtest"]["method"] == "ATRRiskBasedSizing" and leg["paridad"] is None
    leg = main._load_leg(w_eur_ftmo(_root(tmp_path) / "NOMM.sqx", base), ACCOUNT, 0.005)
    assert leg["mm_backtest"] is None and leg["paridad"] is None
    cero = LAST_FIXED_AMOUNT.replace(f'value="{RISKED}">{RISKED}', 'value="0">0')
    leg = main._load_leg(w_eur_ftmo(_root(tmp_path) / "R0.sqx", base, last_settings=cero), ACCOUNT, 0.005)
    assert leg["mm_backtest"]["params"]["RiskedMoney"] == 0.0 and leg["paridad"] is None


def test_ac2_3_paridad_si_todas_las_ordenes_van_a_lots_if_no_mm_la_fraccion_sin_ellas_es_none(tmp_path):
    base = _fixed_amount_trades(20, huge=1)                     # k % 1 == 0: TODAS con el SL lejano
    leg = main._load_leg(w_eur_ftmo(_root(tmp_path) / "ALL.sqx", base, last_settings=LAST_FIXED_AMOUNT), ACCOUNT, 0.005)
    assert leg["paridad"]["n_lifm"] == leg["paridad"]["n"] == 20 and leg["paridad"]["frac_ok_sin_lifm"] is None
    assert leg["paridad"]["frac_ok"] == 1.0


def test_ac2_3_la_paridad_es_informativa_no_veta_la_carga(tmp_path):
    tr = _fixed_amount_trades()
    # el lastSettings declara otro RiskedMoney (9999) que los lotes escritos: la regla da otros lotes y la paridad baja
    p = w_eur_ftmo(_root(tmp_path) / "FA3.sqx", tr,
                   last_settings=LAST_FIXED_AMOUNT.replace(f'value="{RISKED}">{RISKED}', 'value="9999">9999'))
    res = _load([p])
    assert res["legs"][0]["status"] == "ok"
    assert main._load_leg(p, ACCOUNT, 0.005)["paridad"]["frac_ok"] < 1.0


# --- AC2.3bis (E11) -------------------------------------------------------------------------------------------------------
def test_ac2_3bis_point_value_distinto_de_una_ficha_de_usuario_nombra_los_dos_valores_y_la_fuente(api, tmp_path):
    _mk_user_cfd(api, sym="ZZZ_mi", pv=50000.0)                                  # la ficha dice 50.000
    p = write_sqx(_root(tmp_path) / "Z.sqx", eur_ftmo(), symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0,
                  tick=0.0001)                                                    # el .sqx trae 100.000
    with pytest.raises(ValueError) as ei:
        main._load_leg(p, ACCOUNT, 0.005)
    msg = str(ei.value)
    assert msg == "Z.sqx: pointValue del .sqx (100000.0) ≠ point_value_usd de la ficha (50000.0, fuente: manual)"
    assert "100000.0" in msg and "50000.0" in msg and "fuente" in msg
    res = _load([p])
    assert res["legs"][0]["status"] == "error" and "fuente: manual" in res["legs"][0]["error"]


def test_ac2_3bis_el_aviso_de_tick_y_comision_de_una_ficha_de_usuario_tambien_nombran_valores_y_fuente(api, tmp_path):
    _mk_user_cfd(api, sym="ZZZ_mi", pv=100000.0, tick=0.0005, commission={"per_lot_rt": 9.0})
    p = write_sqx(_root(tmp_path) / "Z.sqx", eur_ftmo(), symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0,
                  tick=0.0001)
    leg = main._load_leg(p, ACCOUNT, 0.005)
    warns = leg["exact_warns"]
    assert any("tickSize del .sqx (0.0001) ≠ tick_size de la ficha (0.0005, fuente: manual)" in w for w in warns)
    w_com = next(w for w in warns if "per_lot_rt de la ficha" in w)
    assert "5.0000" in w_com and "9.0000" in w_com and "fuente: manual" in w_com


# ---------------------------------------------------------------------------
# AC2.4: compatibilidad perfil <-> bróker
# ---------------------------------------------------------------------------
def test_ac2_4_cfd_ftmo_con_ftmo_50k_da_200(api, tmp_path):
    load = _load_http(api, _eur_ftmo_legs(tmp_path), "ftmo_50k")
    assert load["ok"], load
    assert main._portfolios[load["portfolio_id"]]["broker"] == "ftmo_cfd"
    assert not any("devolverá 409" in w for w in load["warnings"])
    r = _sim_http(api, load["portfolio_id"], "ftmo_50k", axis="ftmo")
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["leg_kind"] == "sqx_futures"


def test_ac2_4_cfd_ftmo_de_la_demo_con_ftmo_50k_da_200(api, tmp_path):
    _make_demo(tmp_path / "demo")
    legs = sorted(str(p) for p in (tmp_path / "demo" / "demo_cfd").glob("*.sqx"))
    load = _load_http(api, legs, "ftmo_50k")
    assert load["ok"] and _sim_http(api, load["portfolio_id"], "ftmo_50k", axis="ftmo").status_code == 200


def test_ac2_4_sintetico_axi_con_ftmo_50k_da_409_con_el_texto_del_broker(api, tmp_path):
    load = _load_http(api, _xau_axi_legs(tmp_path), "ftmo_50k")
    assert any("La simulación devolverá 409." in w and "Las patas son de Axi CFD (axi)" in w for w in load["warnings"])
    r = _sim_http(api, load["portfolio_id"], "ftmo_50k")
    assert r.status_code == 409
    assert r.json()["detail"] == ("Las patas son de Axi CFD (axi); el perfil 'ftmo_50k' opera en FTMO CFD (ftmo_cfd). "
                                  "Re-testea las estrategias en SQX con los datos de ese bróker.")


def test_ac2_4_futuros_con_ftmo_futures_growth_50k_da_200(api, tmp_path):
    load = _load_http(api, _mnq_legs(tmp_path), "ftmo_futures_growth_50k")
    assert load["ok"], load
    r = _sim_http(api, load["portfolio_id"], "ftmo_futures_growth_50k")
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["funded_model"] == "futures"


def test_ac2_4_futuros_de_la_demo_con_ftmo_futures_growth_50k_da_200(api, tmp_path):
    _make_demo(tmp_path / "demo")
    legs = sorted(str(p) for p in (tmp_path / "demo" / "demo_futuros").glob("*.sqx"))
    load = _load_http(api, legs, "ftmo_futures_growth_50k")
    assert load["ok"] and _sim_http(api, load["portfolio_id"], "ftmo_futures_growth_50k").status_code == 200


def test_ac2_4_futuros_con_apex_da_409_aun_no_admite_patas_sqx(api, tmp_path):
    load = _load_http(api, _mnq_legs(tmp_path), "apex_150k_intraday")
    assert any("El perfil 'apex_150k_intraday' aún no admite patas .sqx. La simulación devolverá 409." in w
               for w in load["warnings"])
    r = _sim_http(api, load["portfolio_id"], "apex_150k_intraday")
    assert r.status_code == 409 and r.json()["detail"] == "El perfil 'apex_150k_intraday' aún no admite patas .sqx."
    assert "aún no admite patas .sqx" in r.json()["detail"]


def test_ac2_4_el_409_de_broker_va_antes_que_el_de_capacidad_y_conserva_las_palabras_de_siempre(api, tmp_path):
    """CFD con un perfil de futuros y futuros con un perfil CFD: primero se comprueba el bróker (mensaje nuevo), que
    nombra "FTMO CFD" / "FTMO Futures" como los mensajes de capacidad de siempre (tests existentes sin editar)."""
    cfd = _load_http(api, _eur_ftmo_legs(tmp_path), "ftmo_futures_growth_50k")
    r = _sim_http(api, cfd["portfolio_id"], "ftmo_futures_growth_50k")
    assert r.status_code == 409 and r.json()["detail"].startswith("Las patas son de FTMO CFD (ftmo_cfd); el perfil ")
    assert "opera en FTMO Futures (ftmo_futures)" in r.json()["detail"]
    fut = _load_http(api, _mnq_legs(tmp_path), "ftmo_100k")
    r = _sim_http(api, fut["portfolio_id"], "ftmo_100k")
    assert r.status_code == 409 and r.json()["detail"].startswith("Las patas son de FTMO Futures (ftmo_futures)")
    assert "opera en FTMO CFD (ftmo_cfd)" in r.json()["detail"]


def test_ac2_4_un_perfil_de_usuario_sobre_ftmo_50k_se_comporta_como_ftmo_50k(api, profiles_file, tmp_path):
    r = _send(api, "POST", "/api/profiles", {"base": "ftmo_50k", "key": "mi_perfil", "display_name": "Mi perfil"})
    assert r.status_code == 200, r.text
    assert symbols.broker_for_profile("mi_perfil") == "ftmo_cfd"
    legs = _eur_ftmo_legs(tmp_path)
    lo = _load_http(api, legs, "ftmo_50k")
    lu = _load_http(api, legs, "mi_perfil")
    assert lu["ok"] and not any("devolverá 409" in w for w in lu["warnings"])
    ro = _sim_http(api, lo["portfolio_id"], "ftmo_50k", axis="ftmo")
    ru = _sim_http(api, lu["portfolio_id"], "mi_perfil", axis="ftmo")
    assert ro.status_code == ru.status_code == 200
    assert ru.json()["reports"] == ro.json()["reports"] and ru.json()["recommendation"] == ro.json()["recommendation"]
    # y un perfil de usuario sobre Apex hereda el "sin bróker"
    assert _send(api, "POST", "/api/profiles", {"base": "apex_150k_intraday", "key": "mi_apex",
                                                "display_name": "Mi Apex"}).status_code == 200
    la = _load_http(api, _mnq_legs(tmp_path), "mi_apex")
    rr = _sim_http(api, la["portfolio_id"], "mi_apex")
    assert rr.status_code == 409 and "aún no admite patas .sqx" in rr.json()["detail"]


def test_ac2_4_el_reloj_del_broker_decide_el_eje_de_dias(api, tmp_path):
    """§2.6: ``ny_wall`` (reloj de pared de Nueva York) → labeler de sesiones NY con el corte del perfil;
    ``ny_plus_7`` (hora del servidor) → sin labeler, el desplazamiento de eje de los CFD. Lo decide el bróker, no la clase
    de activo: aquí dos brókers CFD del usuario con el mismo ``.sqx`` (un trade que cierra a las 16:30)."""
    for bid, clock in (("cfd_wall", "ny_wall"), ("cfd_plus7", "ny_plus_7")):
        r = _send(api, "POST", "/api/symbols/brokers", {"id": bid, "display_name": bid, "company": "Co",
                                                        "asset_class": "cfd", "clock": clock})
        assert r.status_code == 200, r.text
    tr = eur_ftmo(n_days=10)
    tr[0] = dict(tr[0], open=datetime(2023, 1, 2, 15, 0), close=datetime(2023, 1, 2, 16, 30))
    legs = {}
    for bid, sym in (("cfd_wall", "WALL_x"), ("cfd_plus7", "PLUS_x")):
        r = _send(api, "POST", "/api/symbols", dict(broker=bid, sqx_symbol=sym, point_value_usd=100000.0, lot_step=0.01,
                                                    min_lot=0.01, commission={"per_lot_rt": 5.0}))
        assert r.status_code == 200, r.text
        legs[sym] = str(write_sqx(_root(tmp_path) / f"{sym}.sqx", tr, symbol=sym, instrument="I", pv=100000.0,
                                  tick=0.0001))
    first = lambda leg: min(range(len(leg["s"].close_t)), key=lambda i: leg["s"].close_t[i])      # noqa: E731
    wall = main._load_leg(legs["WALL_x"], ACCOUNT, 0.005, day_labeler_tz="Europe/Athens", day_labeler_cut="16:10")
    plus7 = main._load_leg(legs["PLUS_x"], ACCOUNT, 0.005, day_labeler_tz="Europe/Athens", day_labeler_cut="16:10")
    i = first(wall)
    assert str(wall["s"].close_d[i]) == "2023-01-03"      # sesión NY: 16:30 > 16:10 → ya es la sesión del día siguiente
    assert str(plus7["s"].close_d[first(plus7)]) == "2023-01-02"        # sin labeler: día calendario
    assert wall["asset"] == plus7["asset"] == "cfd"
    # y el eje de desplazamiento solo se aplica con "ny_plus_7"
    shifted = main._load_leg(legs["PLUS_x"], ACCOUNT, 0.005, axis_hours=-1.0)
    assert pd.Timestamp(shifted["s"].open_t[first(shifted)]) != pd.Timestamp(plus7["s"].open_t[first(plus7)])
    unshifted = main._load_leg(legs["WALL_x"], ACCOUNT, 0.005, axis_hours=-1.0)
    assert pd.Timestamp(unshifted["s"].open_t[first(unshifted)]) == pd.Timestamp(wall["s"].open_t[first(wall)])


def test_ac2_4_firm_key_vacio_usa_el_primer_perfil_oficial_y_se_comprueba_igual(api, tmp_path):
    load = _load_http(api, _mnq_legs(tmp_path), "")                      # sin perfil en la carga
    assert not any("devolverá 409" in w for w in load["warnings"])
    r = api.post("/api/simulate/propfirm", json={"portfolio_id": load["portfolio_id"], "firm_key": "", "n_paths": 100,
                                                 "n_challenges": 2, "n_sequences": 100})
    assert r.status_code == 409 and "Las patas son de FTMO Futures (ftmo_futures)" in r.json()["detail"]


def test_ac2_4_la_lente_axi_y_el_diagnostico_por_pata_siguen_en_409_para_cualquier_sqx(api, tmp_path):
    """Fase F3: la lente AXI ADMITE una cartera .sqx del bróker de AXI (test_axi_sqx.py) y rechaza con 409 la de otro
    bróker (aquí FTMO CFD); el diagnóstico por pata no cambia."""
    load = _load_http(api, _xau_axi_legs(tmp_path), "")
    ftmo = _load_http(api, _eur_ftmo_legs(tmp_path), "")
    with pytest.raises(HTTPException) as e1:
        main.simulate_axi(main.AxiSimRequest(portfolio_id=ftmo["portfolio_id"]))
    assert e1.value.status_code == 409 and ".sqx" in e1.value.detail
    with pytest.raises(HTTPException) as e2:
        main.portfolio_leg_diagnostics(load["portfolio_id"], main.LegDiagRequest(f=0.01))
    assert e2.value.status_code == 409


# ---------------------------------------------------------------------------
# AC2.5 (E4): editar una ficha invalida la caché
# ---------------------------------------------------------------------------
def test_ac2_5_cambiar_point_value_usd_por_la_api_hace_fallar_la_pata_por_el_point_value(api, tmp_path):
    _mk_user_cfd(api, sym="ZZZ_mi", pv=100000.0)
    p = write_sqx(_root(tmp_path) / "Z.sqx", eur_ftmo(), symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0,
                  tick=0.0001)
    leg1 = main._load_leg(p, ACCOUNT, 0.005)
    assert main._load_leg(p, ACCOUNT, 0.005) is leg1                              # sin cambios: misma pata cacheada
    r = _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_mi", {"point_value_usd": 50000.0})
    assert r.status_code == 200, r.text
    with pytest.raises(ValueError, match=r"pointValue del \.sqx \(100000\.0\) ≠ point_value_usd de la ficha \(50000\.0, fuente: manual\)"):
        main._load_leg(p, ACCOUNT, 0.005)
    # vuelve a su valor: carga de nuevo
    assert _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_mi", {"point_value_usd": 100000.0}).status_code == 200
    assert main._load_leg(p, ACCOUNT, 0.005)["name"] == "Z.sqx"


def test_ac2_5_la_firma_de_la_ficha_protege_la_cache_aunque_el_gancho_no_se_dispare(api, tmp_path, user_file):
    """Segunda capa de E4: editar el fichero de usuario A MANO (sin la API, así que ``set_reload_hook`` no vacía la
    caché de patas) cambia la firma de la ficha y por tanto la clave: la pata vieja no se sirve."""
    _mk_user_cfd(api, sym="ZZZ_mi", pv=100000.0)
    p = write_sqx(_root(tmp_path) / "Z.sqx", eur_ftmo(), symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0,
                  tick=0.0001)
    leg1 = main._load_leg(p, ACCOUNT, 0.005)
    n_keys = len(main._leg_cache)
    doc = yaml.safe_load(user_file.read_text(encoding="utf-8"))
    doc["symbols"]["mi_cfd"]["ZZZ_mi"]["point_value_usd"] = 50000.0
    user_file.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    ns = user_file.stat().st_mtime_ns + 5_000_000_000
    os.utime(user_file, ns=(ns, ns))                                 # mtime distinto sin esperar
    assert len(main._leg_cache) == n_keys                           # el gancho NO se disparó
    with pytest.raises(ValueError, match="pointValue"):
        main._load_leg(p, ACCOUNT, 0.005)
    # y la pata vieja sigue en la caché bajo su clave antigua (no se sirve, pero no se pierde)
    assert any(v is leg1 for v in main._leg_cache.values())


def _write_flat_ohlc(path: Path, trades: list[dict], price: float) -> None:
    """OHLC M1 sintético PLANO (formato MT5 sin cabecera, hora del servidor) con todos los minutos de cada trade."""
    rows = []
    for t in trades:
        for ts in pd.date_range(pd.Timestamp(t["open"]).floor("min"), pd.Timestamp(t["close"]).floor("min"), freq="min"):
            rows.append(f"{ts:%Y.%m.%d},{ts:%H:%M},{price:.5f},{price:.5f},{price:.5f},{price:.5f},100")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _series_min(leg: dict) -> np.ndarray:
    """Mínimo de la serie por minuto de cada trade (en el orden de la pata)."""
    return np.array([leg["minute_valle"][i][1].min() for i in range(len(leg["minute_valle"]))])


def test_ac2_5_cambiar_spread_px_cambia_la_serie_por_minuto_de_una_pata_cfd(api, tmp_path):
    tr = _trades(10, 31, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=30,
                 fixed_px=1.125)                       # 1,125 y 2^-8 son exactos en float32: barras planas sin ruido
    _write_flat_ohlc(tmp_path / "ohlc_mi" / "ZZZ_mi.csv", tr, 1.125)
    main.SQX_OHLC_DIR_OVERRIDES["mi_cfd"] = tmp_path / "ohlc_mi"
    _mk_user_cfd(api, sym="ZZZ_mi", spread_px=0.00003, ohlc_m1="ZZZ_mi.csv")
    p = write_sqx(_root(tmp_path) / "Z.sqx", tr, symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0, tick=0.0001)
    leg1 = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert leg1["minute_reason"] is None and len(leg1["minute_valle"]) == 30
    assert leg1["minute_diag"]["coherencia"] == pytest.approx(1.0) and leg1["minute_diag"]["ohlc"] == "ZZZ_mi.csv"
    base = _series_min(leg1)
    # cortos con barras planas: la serie es −spread − comisión/lote/pv en cada minuto
    np.testing.assert_allclose(base, -0.00003 - 5.0 / 100000.0, rtol=0, atol=1e-12)
    r = _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_mi", {"spread_px": 0.00009})
    assert r.status_code == 200, r.text
    leg2 = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert leg2 is not leg1
    np.testing.assert_allclose(_series_min(leg2) - base, -(0.00009 - 0.00003), rtol=0, atol=1e-12)


def test_ac2_5_el_mtime_del_ohlc_entra_en_la_clave(api, tmp_path):
    tr = _trades(10, 32, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=10,
                 fixed_px=1.125)
    f = tmp_path / "ohlc_mi" / "ZZZ_mi.csv"
    _write_flat_ohlc(f, tr, 1.125)
    main.SQX_OHLC_DIR_OVERRIDES["mi_cfd"] = tmp_path / "ohlc_mi"
    _mk_user_cfd(api, sym="ZZZ_mi", ohlc_m1="ZZZ_mi.csv")
    p = write_sqx(_root(tmp_path) / "Z.sqx", tr, symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0, tick=0.0001)
    leg1 = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert main._load_leg(p, ACCOUNT, 0.005, with_minutes=True) is leg1
    ns = f.stat().st_mtime_ns + 5_000_000_000
    os.utime(f, ns=(ns, ns))
    assert main._load_leg(p, ACCOUNT, 0.005, with_minutes=True) is not leg1


def test_ac2_5_los_motivos_de_sin_serie_por_minuto_de_una_ficha_de_usuario_citan_la_ficha(api, tmp_path):
    tr = _trades(10, 33, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=10,
                 fixed_px=1.125)
    _mk_user_cfd(api, sym="ZZZ_mi")                                      # sin ohlc_m1
    p = write_sqx(_root(tmp_path) / "Z.sqx", tr, symbol="ZZZ_mi", instrument="ZZZ_cfd_mi", pv=100000.0, tick=0.0001)
    leg = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert leg["minute_valle"] is None and leg["minute_reason"] == "sin ohlc_m1 en la ficha de ZZZ_mi"
    # con ohlc_m1 pero sin fichero
    assert _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_mi", {"ohlc_m1": "ZZZ_mi.csv"}).status_code == 200
    main.SQX_OHLC_DIR_OVERRIDES["mi_cfd"] = tmp_path / "ohlc_mi"
    leg = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert leg["minute_reason"] == "OHLC ZZZ_mi.csv no disponible o ilegible en ohlc_mi/"
    # con el fichero pero sin spread_px
    _write_flat_ohlc(tmp_path / "ohlc_mi" / "ZZZ_mi.csv", tr, 1.125)
    assert _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_mi", {"spread_px": None}).status_code == 200
    leg = main._load_leg(p, ACCOUNT, 0.005, with_minutes=True)
    assert leg["minute_reason"] == "la ficha de ZZZ_mi no trae spread_px"
    # una ficha de futuros nunca tiene serie por minuto
    fut = main._load_leg(w_mnq(_root(tmp_path) / "M.sqx", mnq(n_days=20)), ACCOUNT, 0.005, with_minutes=True)
    assert fut["minute_valle"] is None and fut["minute_reason"] == "no aplica: pata de futuros (sin OHLC M1)"


def test_ac2_5_ohlc_dir_por_broker(tmp_path):
    reg = symbols.load_registry()
    assert main._ohlc_dir(reg.brokers["ftmo_cfd"]) == main.SQX_OHLC_DIR == tmp_path / "sin_ohlc"      # la constante de los tests
    assert main._ohlc_dir(reg.brokers["axi"]) == tmp_path / "ohlc_axi"                               # el override del fixture
    del main.SQX_OHLC_DIR_OVERRIDES["axi"]
    assert main._ohlc_dir(reg.brokers["axi"]) == ROOT / "risk_app" / "inputs" / "ohlc_axi"
    with pytest.raises(ValueError, match="no tiene carpeta de OHLC"):
        main._ohlc_dir(reg.brokers["ftmo_futures"])
    main.SQX_OHLC_DIR_OVERRIDES["ftmo_futures"] = tmp_path / "x"
    assert main._ohlc_dir(reg.brokers["ftmo_futures"]) == tmp_path / "x"                             # el override manda


# ---------------------------------------------------------------------------
# AC2.6: GET /api/symbols/draft-from-sqx
# ---------------------------------------------------------------------------
COMM_SIZE_BASED = ('<Method type="None" use="false"><Params /></Method><Method type="SizeBased" use="true"><Params>'
                   '<Param key="Commission" className="SizeBased">4.5</Param></Params></Method>')


def _sqx_with_instrument(tmp_path: Path, name="NUEVO.sqx", *, symbol="ZZZ_nuevo", pv=100000.0, tick=0.0001, **attrs) -> str:
    p = write_sqx(_root(tmp_path) / name, eur_ftmo(n_days=10), symbol=symbol, instrument=f"{symbol}_instr", pv=pv, tick=tick)
    return str(patch_instrument_info(p, **attrs))


def _draft(api, path):
    return api.get("/api/symbols/draft-from-sqx", params={"path": path})


def test_ac2_6_draft_from_sqx_da_los_campos_en_needs_review_y_min_lot_en_faltan(api, tmp_path):
    p = _sqx_with_instrument(tmp_path, orderSizeStep="0.01", defaultSpread="0.5", commissions=COMM_SIZE_BASED)
    r = _draft(api, p)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["sqx_symbol"] == "ZZZ_nuevo" and d["broker"] is None
    assert d["point_value_usd"] == 100000.0 and d["lot_step"] == 0.01 and d["tick_size"] == 0.0001
    assert d["commission"] == {"per_lot_rt": 4.5} and d["spread_px"] == pytest.approx(0.5 * 0.0001, rel=1e-12)
    assert d["min_lot"] is None and "min_lot" in d["faltan"] and d["faltan"][0] == "broker"
    fields = ["point_value_usd", "lot_step", "commission", "spread_px", "tick_size"]
    assert d["needs_review"] == fields
    assert d["sources"] == {f: {"from": "sqx", "date": d["sources"][f]["date"]} for f in fields}
    assert all(s["from"] == "sqx" for s in d["sources"].values())
    assert (d["guardable"], d["editable"], d["motivo"], d["existing"], d["conflicts"]) == (True, True, None, None, [])
    json.dumps(d, allow_nan=False)
    assert not (tmp_path / "symbols.user.yaml").exists()                         # el borrador no guarda nada


def test_ac2_6_sin_los_atributos_opcionales_no_inventa_valores(api, tmp_path):
    d = _draft(api, _sqx_with_instrument(tmp_path)).json()
    assert d["lot_step"] is None and d["spread_px"] is None and d["commission"] == {}
    assert d["needs_review"] == ["point_value_usd", "tick_size"]
    assert d["faltan"] == ["broker", "lot_step", "min_lot", "commission"]
    # una comisión que no es SizeBased se avisa y no se propone
    pct = '<Method type="PercentageBased" use="true"><Params><Param key="Commission">0.0014</Param></Params></Method>'
    d = _draft(api, _sqx_with_instrument(tmp_path, "P.sqx", commissions=pct)).json()
    assert d["commission"] == {} and any("PercentageBased" in w for w in d["warnings"])
    assert "commission" in d["faltan"] and "commission" not in d["needs_review"]


def test_ac2_6_una_ruta_fuera_del_catalog_es_400(api, tmp_path):
    fuera = write_sqx(tmp_path / "fuera" / "F.sqx", eur_ftmo(n_days=5), symbol="ZZZ_f", instrument="X", pv=1.0, tick=1.0)
    r = _draft(api, str(fuera))
    assert r.status_code == 400 and "no está en el catálogo" in r.json()["detail"]["errors"][0]
    dentro = _sqx_with_instrument(tmp_path)
    assert _draft(api, dentro).status_code == 200
    traversal = str(_root(tmp_path) / ".." / ".." / "fuera" / "F.sqx")
    assert _draft(api, traversal).status_code == 400
    assert _draft(api, "").status_code == 400
    assert _draft(api, "/etc/passwd").status_code == 400
    assert api.get("/api/symbols/draft-from-sqx").status_code == 422                 # falta ?path=
    # una ruta con ``..`` que resuelve a un .sqx del catálogo sí vale (se normaliza)
    assert _draft(api, str(_root(tmp_path) / ".." / "bank" / "NUEVO.sqx")).status_code == 200


def test_ac2_6_sin_proveedor_de_catalog_todo_es_400(api, tmp_path, monkeypatch):
    p = _sqx_with_instrument(tmp_path)
    provider = symbols._catalog_provider
    assert provider is not None and p in provider()                    # main lo registra al importarse
    monkeypatch.setattr(symbols, "_catalog_provider", None)
    assert _draft(api, p).status_code == 400
    monkeypatch.setattr(symbols, "_catalog_provider", provider)
    assert _draft(api, p).status_code == 200


def test_ac2_6_un_sqx_ilegible_no_esta_en_el_catalog_y_da_400(api, tmp_path):
    root = _root(tmp_path)
    root.mkdir(parents=True)
    (root / "roto.sqx").write_bytes(b"esto no es un zip")
    assert _draft(api, str(root / "roto.sqx")).status_code == 400


def test_ac2_6_si_el_sqx_no_se_puede_leer_da_400_y_no_500(api, tmp_path, monkeypatch):
    p = _sqx_with_instrument(tmp_path)

    def desaparece(path):
        raise FileNotFoundError("ya no está")

    def ilegible(path):
        raise SQXFormatError("settings.xml ausente")

    for boom, texto in ((desaparece, "ya no está"), (ilegible, "settings.xml ausente")):
        monkeypatch.setattr(symbols, "read_instrument_info", boom)
        r = _draft(api, p)
        assert r.status_code == 400 and texto in r.json()["detail"]["errors"][0], r.text


def test_ac2_6_conflictos_con_una_ficha_oficial_son_e11_y_el_borrador_no_es_guardable(api, tmp_path):
    p = _sqx_with_instrument(tmp_path, symbol="EURUSD_ftmo", orderSizeStep="0.05", defaultSpread="0.5",
                             commissions=COMM_SIZE_BASED)
    d = _draft(api, p).json()
    by = {c["field"]: c for c in d["conflicts"]}
    assert set(by) == {"lot_step", "commission", "spread_px"}                       # pv y tick coinciden con la oficial
    assert (by["lot_step"]["actual"], by["lot_step"]["propuesto"]) == (0.01, 0.05)
    assert (by["lot_step"]["fuente_actual"], by["lot_step"]["fuente_propuesta"]) == ("official", "sqx")
    assert by["commission"]["actual"] == {"per_lot_rt": 5.0} and by["commission"]["propuesto"] == {"per_lot_rt": 4.5}
    assert "la ficha oficial dice 0.01, tu .sqx dice 0.05; si tu cuenta es distinta, crea un bróker propio" in by["lot_step"]["texto"]
    assert d["existing"] == {"broker": "ftmo_cfd", "sqx_symbol": "EURUSD_ftmo", "official": True}
    assert (d["guardable"], d["editable"]) == (False, False)                        # H2
    assert d["motivo"].startswith("el símbolo SQX 'EURUSD_ftmo' ya tiene ficha en el bróker 'ftmo_cfd'")


def test_ac2_6_conflictos_con_una_ficha_de_usuario_son_aceptables_campo_a_campo(api, tmp_path):
    _mk_user_cfd(api, sym="ZZZ_nuevo", pv=100000.0, tick=0.0001, spread_px=0.0002, commission={"per_lot_rt": 7.0})
    p = _sqx_with_instrument(tmp_path, orderSizeStep="0.01", defaultSpread="0.5", commissions=COMM_SIZE_BASED)
    d = _draft(api, p).json()
    by = {c["field"]: c for c in d["conflicts"]}
    assert set(by) == {"commission", "spread_px"}
    assert by["spread_px"]["fuente_actual"] == "manual" and by["spread_px"]["fuente_propuesta"] == "sqx"
    assert "tu ficha dice" in by["spread_px"]["texto"] and "puedes quedarte con el valor del .sqx, campo a campo" in by["spread_px"]["texto"]
    assert (d["guardable"], d["editable"], d["motivo"]) == (True, True, None)
    # se acepta solo la comisión del .sqx (PUT con la procedencia); el spread de la ficha se queda
    r = _send(api, "PUT", "/api/symbols/mi_cfd/ZZZ_nuevo", {"commission": d["commission"],
                                                            "sources": {"commission": d["sources"]["commission"]}})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["commission"] == {"per_lot_rt": 4.5} and saved["sources"]["commission"]["from"] == "sqx"
    assert saved["spread_px"] == 0.0002 and saved["sources"]["spread_px"]["from"] == "manual"
    again = _draft(api, p).json()
    assert {c["field"] for c in again["conflicts"]} == {"spread_px"}


def test_ac2_6_alta_guiada_de_punta_a_punta_sin_ficha_a_pata_cargada(api, tmp_path):
    p = _sqx_with_instrument(tmp_path, orderSizeStep="0.01", defaultSpread="0.5", commissions=COMM_SIZE_BASED)
    assert _by_name()["NUEVO"]["ficha"] == "sin_ficha"
    assert _load([p])["legs"][0]["status"] == "error"
    # bróker de usuario + borrador completado por el usuario (bróker y lote mínimo) -> POST
    _send(api, "POST", "/api/symbols/brokers", {"id": "mi_cfd", "display_name": "Mi CFD", "company": "Mi Compañía",
                                                "asset_class": "cfd", "clock": "ny_plus_7"})
    d = _draft(api, p).json()
    d["broker"], d["min_lot"] = "mi_cfd", 0.01
    r = _send(api, "POST", "/api/symbols", d)                                       # los campos de solo lectura se ignoran
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["needs_review"] == d["needs_review"] and saved["sources"]["point_value_usd"]["from"] == "sqx"
    assert saved["sources"]["min_lot"]["from"] == "manual"
    e = _by_name()["NUEVO"]
    assert (e["ficha"], e["broker"], e["asset"]) == ("ok", "mi_cfd", "cfd")
    res = _load([p])
    assert res["legs"][0]["status"] == "ok", res
    leg = main._portfolios[res["portfolio_id"]]["legs"][0]
    assert leg["broker"] == "mi_cfd" and leg["ficha"]["official"] is False
    assert leg["ficha"]["needs_review"] == d["needs_review"]                        # el resumen avisa de lo sin verificar


# ---------------------------------------------------------------------------
# F3: correcciones de la revisión de código de la F2 (H1 mitad endpoint, H6, H7, H8, H9)
# ---------------------------------------------------------------------------
def test_h1_draft_from_sqx_sin_symbolinfo_o_sin_instrumentinfo_es_400_y_nunca_500(api, tmp_path, monkeypatch):
    """H1 (mitad endpoint; la del lector está en ``tests/test_sqx_trades.py``): ``draft-from-sqx`` devuelve 400 con un texto
    claro cuando el ``settings.xml`` no trae el ``SymbolInfo`` del símbolo del resultado Main o su ``InstrumentInfo``, nunca
    un 500 (``StopIteration``/``AttributeError``). Un ``settings.xml`` sin ``InstrumentInfo`` de verdad ni siquiera entra en el
    catálogo (la lectura barata del símbolo ya lo trata como ilegible): también 400."""
    import sqx.sqx_trades as st
    p = _sqx_with_instrument(tmp_path)
    assert _draft(api, p).status_code == 200
    real = st._parse_settings_xml(zipfile.ZipFile(p).read("settings.xml"))
    # un .sqx cuyo settings.xml no trae InstrumentInfo de verdad: ni siquiera entra en el catálogo (ilegible) -> 400
    with zipfile.ZipFile(p) as zf:
        items = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]
    q = tmp_path / "extra" / "bank" / "SIN_INSTR.sqx"
    with zipfile.ZipFile(q, "w") as zf:
        for name, data in items:
            if name == "settings.xml":
                head, tail = data.decode().split("<InstrumentInfo", 1)
                data = (head + tail.split("/>", 1)[1]).encode()
            zf.writestr(name, data)
    assert "SIN_INSTR" not in _by_name() and _draft(api, str(q)).status_code == 400
    # 1) el símbolo del resultado Main no figura en ningún SymbolInfo (se salta la validación previa del parser)
    monkeypatch.setattr(st, "_parse_settings_xml", lambda xml: {**real, "symbol": "NO_ESTA_EN_NINGUN_SYMBOLINFO"})
    r = _draft(api, p)
    assert r.status_code == 400, r.text
    assert "sin SymbolInfo" in r.json()["detail"]["errors"][0] and "NO_ESTA_EN_NINGUN_SYMBOLINFO" in r.json()["detail"]["errors"][0]
    # 2) el SymbolInfo existe pero sin InstrumentInfo, y el .sqx SÍ está en el catálogo (se fuerza su símbolo)
    monkeypatch.setattr(st, "_parse_settings_xml", lambda xml: real)
    monkeypatch.setattr(main, "_sqx_symbol", lambda path: "ZZZ_nuevo")
    assert "SIN_INSTR" in _by_name()
    r = _draft(api, str(q))
    assert r.status_code == 400 and "no trae InstrumentInfo" in r.json()["detail"]["errors"][0], r.text


def test_h6_la_paridad_nunca_tumba_la_carga_longitudes_distintas_y_cualquier_excepcion(tmp_path, monkeypatch):
    """H6: ``_fixed_amount_parity`` es informativa. Comprueba que las longitudes de los lotes del .sqx y del riesgo por lote
    coinciden y captura CUALQUIER excepción: devuelve ``None`` en vez de tumbar ``_load_sqx_leg``."""
    mm = {"method": "FixedAmount", "params": {"RiskedMoney": 57.5}}
    meta = {"lot_step": 0.01, "min_lot": 0.01}
    fleg = types.SimpleNamespace(risk_per_micro=np.full(5, 50.0), comm_per_lot=np.zeros(5))
    ok = main._fixed_amount_parity(mm, np.full(5, 0.01), fleg, meta, ACCOUNT)
    assert ok is not None and ok["n"] == 5                                         # la base: con longitudes iguales, calcula
    # longitudes distintas (antes: ValueError de broadcast FUERA del try, que tumbaba la carga de la pata)
    assert main._fixed_amount_parity(mm, np.full(3, 0.01), fleg, meta, ACCOUNT) is None
    assert main._fixed_amount_parity(mm, np.full(7, 0.01), fleg, meta, ACCOUNT) is None
    assert main._fixed_amount_parity(mm, np.full((5, 1), 0.01), fleg, meta, ACCOUNT) is None
    # cualquier otra excepción (antes: solo se capturaba ValueError de fixed_risk_micros)
    assert main._fixed_amount_parity(mm, np.full(5, 0.01), fleg, meta, 0.0) is None            # ZeroDivisionError
    assert main._fixed_amount_parity({"method": "FixedAmount", "params": {"RiskedMoney": "x"}}, np.full(5, 0.01), fleg,
                                     meta, ACCOUNT) is None                       # TypeError al comparar
    assert main._fixed_amount_parity({"method": "FixedAmount", "params": None}, np.full(5, 0.01), fleg, meta,
                                     ACCOUNT) is None                              # AttributeError
    bad = types.SimpleNamespace(risk_per_micro=None, comm_per_lot=None)
    assert main._fixed_amount_parity(mm, np.full(5, 0.01), bad, meta, ACCOUNT) is None
    # y de punta a punta: aunque fixed_risk_micros reviente con una excepción inesperada, la pata carga (sin paridad)
    p = w_eur_ftmo(_root(tmp_path) / "FA.sqx", _fixed_amount_trades(), last_settings=LAST_FIXED_AMOUNT)

    def boom(*a, **k):
        raise RuntimeError("fallo inesperado del cálculo de lotes")

    monkeypatch.setattr(main, "fixed_risk_micros", boom)
    leg = main._load_leg(p, ACCOUNT, 0.005)
    assert leg["name"] == "FA.sqx" and leg["paridad"] is None and leg["mm_backtest"]["method"] == "FixedAmount"
    assert _load([p])["legs"][0]["status"] == "ok"


def _ficha_de_eur_ftmo(tmp_path: Path, n_days: int = 10):
    """``(df, meta, fleg, ficha, bróker)`` de un EURUSD_ftmo sintético con barras planas (para llamar a las series por
    minuto directamente)."""
    tr = _trades(10, 41, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=n_days,
                 fixed_px=1.125)
    p = Path(w_eur_ftmo(_root(tmp_path) / "E.sqx", tr))
    df, meta = read_trade_table(p, costs_path=DEFAULT_CFD_COSTS_PATH)
    _, fleg = strategy_from_trade_table(df, meta, key=p.name)
    sheet = symbols.resolve("EURUSD_ftmo")
    return tr, df, meta, fleg, sheet, symbols.load_registry().brokers[sheet.broker]


def test_h7_la_ficha_sin_ohlc_m1_y_el_broker_sin_carpeta_de_ohlc_son_motivos_distintos(tmp_path, monkeypatch):
    """H7: antes los dos casos daban «sin ohlc_m1 …», aunque con una ficha que SÍ trae ``ohlc_m1`` el problema es del
    bróker (no declara ``ohlc_dir``)."""
    tr, df, meta, fleg, sheet, broker = _ficha_de_eur_ftmo(tmp_path)
    to_axis = main._sqx_axis_transform(-1.0)
    sin_dir = dataclasses.replace(broker, ohlc_dir=None)              # un bróker CFD sin carpeta de OHLC
    assert sheet.ohlc_m1 == "EURUSD_ftmo.csv" and "ftmo_cfd" not in main.SQX_OHLC_DIR_OVERRIDES
    # la ficha trae ohlc_m1 pero el bróker no tiene carpeta
    r = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, sin_dir, to_axis)
    assert r == (None, None, "el bróker 'ftmo_cfd' no tiene carpeta de OHLC (ohlc_dir): no se puede leer "
                             "EURUSD_ftmo.csv de la ficha de EURUSD_ftmo")
    # la ficha NO trae ohlc_m1: el motivo de siempre (oficial: cita el YAML; de usuario: cita la ficha)
    assert main._sqx_leg_minutes_sheet(df, meta, fleg, dataclasses.replace(sheet, ohlc_m1=None), sin_dir, to_axis) == (
        None, None, "sin ohlc_m1 en el YAML de costes para EURUSD_ftmo")
    assert main._sqx_leg_minutes_sheet(df, meta, fleg, dataclasses.replace(sheet, ohlc_m1=None, official=False), broker,
                                       to_axis) == (None, None, "sin ohlc_m1 en la ficha de EURUSD_ftmo")
    # con la carpeta (por el override de los tests) el bróker sí tiene dónde buscar: pasa al siguiente motivo (fichero ausente)
    monkeypatch.setitem(main.SQX_OHLC_DIR_OVERRIDES, "ftmo_cfd", tmp_path / "ohlc_x")
    r2 = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, sin_dir, to_axis)
    assert r2 == (None, None, "OHLC EURUSD_ftmo.csv no disponible o ilegible en ohlc_x/")
    # y con "daily" el motivo de cada caso se conserva y se añade la cota diaria
    monkeypatch.delitem(main.SQX_OHLC_DIR_OVERRIDES, "ftmo_cfd")
    _, d, why = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, sin_dir, to_axis, synth_mode="daily")
    assert why == ("el bróker 'ftmo_cfd' no tiene carpeta de OHLC (ohlc_dir): no se puede leer EURUSD_ftmo.csv de la "
                   "ficha de EURUSD_ftmo: se usa la cota diaria por el MAE") and d["motivo_sin_serie"] in why


def test_h8_paridad_del_motivo_entre_el_yaml_y_la_ficha_sin_spread_y_con_ohlc_ilegible_en_ftmo(tmp_path, monkeypatch):
    """H8: ``_sqx_leg_minutes`` (YAML) y ``_sqx_leg_minutes_sheet`` (ficha oficial de FTMO CFD) dan el MISMO resultado en
    los motivos «OHLC ilegible» y «sin spread» (la serie feliz y «sin ohlc_m1» ya los cubre
    ``test_ac2_1_serie_por_minuto_por_ficha_identica_a_la_del_yaml_en_ftmo``)."""
    tr, df, meta, fleg, sheet, broker = _ficha_de_eur_ftmo(tmp_path)
    to_axis = main._sqx_axis_transform(-1.0)
    ohlc_dir = tmp_path / "ohlc_ftmo"
    monkeypatch.setattr(main, "SQX_OHLC_DIR", ohlc_dir)
    ohlc_dir.mkdir(parents=True)
    csv = ohlc_dir / "EURUSD_ftmo.csv"
    # 1) OHLC ilegible: el fichero existe pero no se puede leer como OHLC
    csv.write_text("esto no es un ohlc\n", encoding="utf-8")
    old = main._sqx_leg_minutes(df, meta, fleg, "cfd", DEFAULT_CFD_COSTS_PATH, to_axis)
    new = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, broker, to_axis)
    assert old == new == (None, None, "OHLC EURUSD_ftmo.csv no disponible o ilegible en ohlc_ftmo/")
    # 2) fichero ausente: el mismo motivo
    csv.unlink()
    main._sqx_ohlc_cache.clear()
    old = main._sqx_leg_minutes(df, meta, fleg, "cfd", DEFAULT_CFD_COSTS_PATH, to_axis)
    new = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, broker, to_axis)
    assert old == new == (None, None, "OHLC EURUSD_ftmo.csv no disponible o ilegible en ohlc_ftmo/")
    # 3) sin spread: un YAML de costes sin typical_spread_points para el símbolo y la ficha oficial equivalente sin spread_px
    _write_flat_ohlc(csv, tr, 1.125)
    main._sqx_ohlc_cache.clear()
    costs = yaml.safe_load(DEFAULT_CFD_COSTS_PATH.read_text(encoding="utf-8"))
    costs["sqx_policy"]["EURUSD_ftmo"] = {}
    sin_spread = tmp_path / "ftmo_cfd_costs_sin_spread.yaml"
    sin_spread.write_text(yaml.safe_dump(costs, sort_keys=False, allow_unicode=True), encoding="utf-8")
    old = main._sqx_leg_minutes(df, meta, fleg, "cfd", sin_spread, to_axis)
    new = main._sqx_leg_minutes_sheet(df, meta, fleg, dataclasses.replace(sheet, spread_px=None), broker, to_axis)
    assert old == new == (None, None, "YAML de costes sin typical_spread_points/tick_size para EURUSD_ftmo")
    # y con todo en orden las dos dan la serie (el camino feliz sigue igual con este mismo OHLC)
    ok_old = main._sqx_leg_minutes(df, meta, fleg, "cfd", DEFAULT_CFD_COSTS_PATH, to_axis)
    ok_new = main._sqx_leg_minutes_sheet(df, meta, fleg, sheet, broker, to_axis)
    assert ok_old[2] is None and ok_new[2] is None and ok_old[1] == ok_new[1] and len(ok_old[0]) == len(ok_new[0]) == 10


def test_h9_el_centinela_de_la_clave_de_cache_sin_ruta_de_ohlc_es_vacio_y_no_none(tmp_path):
    """H9: con ``with_minutes`` y sin ruta de OHLC (ficha sin ``ohlc_m1``) la clave lleva ``("minutes", "", None)``; el
    texto ``"None"`` no figura nunca como centinela."""
    p = write_sqx(_root(tmp_path) / "G.sqx", eur_ftmo(n_days=10), symbol="GBPUSD_ftmo", instrument="GBPUSD_cfd_ftmo",
                  pv=100000.0, tick=0.0001, capital=ACCOUNT)         # GBPUSD_ftmo no trae ohlc_m1 en el YAML
    leg = main._load_leg(str(p), ACCOUNT, 0.005, with_minutes=True)
    assert leg["minute_valle"] is None and leg["minute_reason"] == "sin ohlc_m1 en el YAML de costes para GBPUSD_ftmo"
    keys = [k for k, v in main._leg_cache.items() if v is leg]
    assert len(keys) == 1
    k = keys[0]
    i = k.index("minutes")
    assert k[i + 1] == "" and k[i + 2] is None
    assert "None" not in k
    # con ruta de OHLC (EURUSD_ftmo trae ohlc_m1) el centinela no interviene: la ruta va como texto
    q = w_eur_ftmo(_root(tmp_path) / "E.sqx", eur_ftmo(n_days=10))
    leg2 = main._load_leg(q, ACCOUNT, 0.005, with_minutes=True)
    k2 = next(k for k, v in main._leg_cache.items() if v is leg2)
    assert k2[k2.index("minutes") + 1].endswith("EURUSD_ftmo.csv")


# ---------------------------------------------------------------------------
# F4: corrección H1 de la revisión de código de la F3 (la enmienda E20 también vale en la lente prop firm)
# ---------------------------------------------------------------------------
def test_cr_f3_h1_la_serie_por_minuto_de_la_prop_firm_usa_el_spread_del_sqx_cuando_difiere_de_la_ficha(
        tmp_path, monkeypatch):
    """H1 (E20 es un cambio de comportamiento de TODA la vía ``.sqx``, no solo de la lente AXI): la serie por minuto de la
    lente prop firm de una cartera CFD de FTMO usa el spread del BACKTEST (``defaultSpread · tickSize`` del propio ``.sqx``) y
    solo cae al ``spread_px`` de la ficha si el ``.sqx`` no lo trae. La bit-identidad de las carteras reales de FTMO se
    cumple por COINCIDENCIA DE DATOS (en sus ``.sqx``, ``defaultSpread`` es el ``typical_spread_points`` del YAML), no como
    invariante del código: con un ``defaultSpread`` distinto de la ficha la serie de la prop firm cambia, y es lo querido."""
    tr = _trades(10, 41, px0=1.125, unit=0.0001, dists=(0.00390625,), pv=100000.0, lots=0.5, comm=5.0, n_days=30,
                 fixed_px=1.125)                                   # cortos con barras planas: cada minuto = −(spread + coste)
    ohlc_dir = tmp_path / "ohlc_ftmo"
    _write_flat_ohlc(ohlc_dir / "EURUSD_ftmo.csv", tr, 1.125)
    monkeypatch.setattr(main, "SQX_OHLC_DIR", ohlc_dir)
    ficha = symbols.resolve("EURUSD_ftmo").spread_px
    assert ficha == 3e-05                                           # la ficha oficial: 0,3 pips
    sin = Path(w_eur_ftmo(_root(tmp_path) / "E_sin.sqx", tr))       # sin defaultSpread (el writer no lo escribe)
    con = Path(w_eur_ftmo(_root(tmp_path) / "E_con.sqx", tr))
    patch_instrument_info(con, defaultSpread="5")                   # 5 · 0,0001 = 0,0005 ≠ 0,00003
    assert main._sqx_backtest_spread(sin) is None and main._sqx_backtest_spread(con) == 5.0 * 0.0001
    cost = 5.0 / 100000.0                                           # comisión por lote / valor del punto
    seen = {}
    for nombre, p, fuente, spread in (("E_sin.sqx", sin, "ficha", ficha), ("E_con.sqx", con, "sqx", 5.0 * 0.0001)):
        load = _load([p], firm_key="ftmo_50k")
        assert load["ok"], load
        out = main.simulate_propfirm(main.PropfirmSimRequest(
            portfolio_id=load["portfolio_id"], firm_key="ftmo_50k", f_lo=0.0025, f_hi=0.005, f_step=0.0025, n_paths=100,
            n_challenges=3, n_sequences=100, day_axis="ftmo"))
        meta = out["meta"]
        assert meta["float_mode"] == "minute" and meta["leg_kind"] == main.KIND_SQX
        d = meta["minute_diag"][nombre]                             # lo que usó la LENTE PROP FIRM (synth_mode "point")
        assert (d["spread_fuente"], d["spread_px_usado"]) == (fuente, spread), nombre
        assert d["coherencia"] == pytest.approx(1.0) and d["sinteticas"] == 0
        leg = main._load_leg(str(p), ACCOUNT, 0.005, axis_hours=-1.0, with_minutes=True)
        assert leg["minute_diag"]["spread_fuente"] == fuente
        seen[nombre] = _series_min(leg)
        np.testing.assert_allclose(seen[nombre], -(spread + cost), rtol=0, atol=1e-12)     # los cortos restan el spread usado
    assert not np.allclose(seen["E_con.sqx"], seen["E_sin.sqx"], atol=1e-6)               # y la serie cambia con él
