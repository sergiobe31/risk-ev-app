"""Fichas de símbolo por bróker (guardrail ``entrada-4-bloques``, fase F1: modelo y API, sin cambiar la carga).

Todo sintético: el fichero de usuario es un temporal (``symbols.USER_SYMBOLS_PATH`` en ``tmp_path``), la hoja de
MT5 es el fixture ``tests/fixtures/mt5_sheet_ftmo_sample.csv`` (se omiten los tests que lo usan si no viaja,
p. ej. en el export compartible) y los ``.sqx`` de paridad se escriben con el writer de
``tests/test_sqx_trades.py`` (mismo patrón que ``test_cfd_sqx_legs.py``).

Mapa criterio -> test (cada criterio lleva su id en el nombre del test):

- AC1.1 paridad de adaptadores ......... ``test_ac1_1_*``
- AC1.2 unicidad de ``sqx_symbol`` ....... ``test_ac1_2_*``
- AC1.3 CRUD, escritura atómica, 409/400 . ``test_ac1_3_*``
- AC1.4 validación ..................... ``test_ac1_4_*``
- AC1.5 / AC1.5bis hoja de MT5 + E6 + E11 ``test_ac1_5_*``, ``test_ac1_5bis_*``, ``test_e6_*``, ``test_e11_*``
- AC1.6 bróker de cada perfil .......... ``test_ac1_6_*``
- AC1.7 invariantes I3 e I4 ............ ``test_ac1_7_*``
- AC1.8 sin cambio de comportamiento ... ``test_ac1_8_*``
- E4 firma de caché .................... ``test_e4_*``
- E9 guardián del export ............... ``test_e9_*``

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_symbols.py -q
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import threading
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from poc_web.api import main, profiles, symbols
from sqx.sqx_trades import (
    DEFAULT_COSTS_PATH,
    DEFAULT_CFD_COSTS_PATH,
    load_costs_policy,
    read_trade_table,
)

ROOT = Path(__file__).resolve().parents[2]
SHEET_FIXTURE = ROOT / "tests" / "fixtures" / "mt5_sheet_ftmo_sample.csv"
MANIFEST = ROOT / "scripts" / "export_shareable" / "manifest.yaml"
REAL_USER_FILES = (ROOT / "config" / "symbols.user.yaml", ROOT / "config" / "propfirms.user.yaml")
_REAL_BEFORE = {p for p in REAL_USER_FILES if p.exists()}      # estado al importar el módulo (I4)
TODAY = "2026-10-20"
FTMO_MAPPING = {"XAUUSD": "XAUUSD_ftmo(2)", "US100.cash": "USATECHIDXUSD_ftmo(2)", "EURUSD": "EURUSD_ftmo"}
# PLAN-DESVIO: Símbolos SQX que NO tienen ficha. Los tests E6 usaban símbolos con ficha OFICIAL (``FTMO_MAPPING``) y
# esperaban ``guardable`` True; desde H2 (revisión de código de la F1, brief de la F2) un borrador con ficha oficial no es
# guardable, así que para seguir comprobando SOLO la moneda (E6) se usan símbolos nuevos. El caso "oficial + moneda mala"
# lo cubre ``test_h2_un_motivo_previo_manda_y_el_e7_pasa_a_los_avisos``.
FRESH_MAPPING = {"XAUUSD": "XAUUSD_nuevo", "US100.cash": "US100_nuevo", "EURUSD": "EURUSD_nuevo"}
_DEL = object()                   # sentinela: quitar la clave del cuerpo

_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer_symbols", ROOT / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _ficheros_reales_de_usuario_intactos():
    """I4: ningún test de este módulo crea ni toca ``config/symbols.user.yaml`` ni ``config/propfirms.user.yaml``
    REALES (existan o no): se compara su estado antes y después de todo el módulo."""
    def state():
        return {str(p): (p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in REAL_USER_FILES}

    before = state()
    yield
    assert state() == before, "los tests tocaron un fichero de usuario REAL (I4)"


@pytest.fixture()
def user_file(tmp_path, monkeypatch):
    path = tmp_path / "symbols.user.yaml"
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", path)
    symbols._reload()
    return path


@pytest.fixture()
def profiles_file(tmp_path, monkeypatch):
    """Mismo fixture que ``user_file`` de ``test_profiles.py``: perfiles de usuario en un temporal."""
    path = tmp_path / "propfirms.user.yaml"
    monkeypatch.setattr(profiles, "USER_PROFILES_PATH", path)
    main.reload_firms()
    yield path
    monkeypatch.undo()
    main.reload_firms()          # restaura el estado del proceso (sin perfiles de usuario)


@pytest.fixture()
def api(user_file):
    return TestClient(main.app)


def _send(c, method, url, obj=None, *, text=None):
    body = text if text is not None else json.dumps(obj)
    return c.request(method, url, content=body, headers={"content-type": "application/json"})


def _errors(r):
    return r.json()["detail"]["errors"]


def _doc(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(obj) -> str:
    """YAML con el orden de claves del dict (el orden de brókers y fichas del fichero importa)."""
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mk_brokers(api):
    """Un bróker CFD propio (con carpeta de OHLC) y uno de futuros propio."""
    r = _send(api, "POST", "/api/symbols/brokers", {
        "id": "mi_cfd", "display_name": "Mi CFD", "company": "Mi Compañía", "asset_class": "cfd",
        "clock": "ny_plus_7", "ohlc_dir": "risk_app/inputs/ohlc_mi"})
    assert r.status_code == 200, r.text
    r = _send(api, "POST", "/api/symbols/brokers", {
        "id": "mi_fut", "display_name": "Mi futuros", "company": "Mi Compañía", "asset_class": "futures",
        "clock": "ny_wall"})
    assert r.status_code == 200, r.text


def _sheet(**kw):
    """Cuerpo de una ficha CFD válida en ``mi_cfd``; ``clave=_DEL`` quita la clave."""
    body = dict(broker="mi_cfd", sqx_symbol="XAUUSD_mi", ticker="XAUUSD", point_value_usd=100, lot_step=0.01,
                min_lot=0.01, max_lot=50, commission={"per_lot_rt": 4.5}, spread_px=0.3, ohlc_m1="XAUUSD_mi.csv")
    body.update(kw)
    return {k: v for k, v in body.items() if v is not _DEL}


def _fut_sheet(**kw):
    body = dict(broker="mi_fut", sqx_symbol="@mi_micro", ticker="MX", point_value_usd=2.0,
                commission={"per_lot_rt": 1.12}, tick_size=0.25)
    body.update(kw)
    return {k: v for k, v in body.items() if v is not _DEL}


def _create(api, **kw):
    r = _send(api, "POST", "/api/symbols", _sheet(**kw))
    assert r.status_code == 200, r.text
    return r.json()


def _sheet_text() -> str:
    if not SHEET_FIXTURE.is_file():
        # El fixture no viaja en el export compartible (su manifest no lo incluye): ahí se omite.
        pytest.skip("fixture de la hoja de MT5 ausente (export compartible)")
    return SHEET_FIXTURE.read_text(encoding="utf-8")


def _edit_sheet(text: str, ticker: str, **cols) -> str:
    """Copia de la hoja con columnas (por NOMBRE de cabecera) cambiadas en la fila de ``ticker``; ``cols`` admite
    ``__after_pip_worth`` para la columna sin nombre de la moneda."""
    rows = list(csv.reader(io.StringIO(text)))
    head = rows[0]
    for row in rows[1:]:
        if row[0] != ticker:
            continue
        for name, value in cols.items():
            idx = head.index("1 Pip Worth") + 1 if name == "__after_pip_worth" else head.index(name)
            row[idx] = value
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def _draft(drafts, ticker):
    return next(d for d in drafts if d["ticker"] == ticker)


def _hide(api, keys, hidden=True):
    return _send(api, "POST", "/api/symbols/visibility", {"keys": keys, "hidden": hidden})


def _official_entries():
    """``[(bróker, ruta del YAML de costes, clave de symbols:)]`` de los dos YAML de costes de FTMO."""
    out = []
    for broker, path in (("ftmo_cfd", DEFAULT_CFD_COSTS_PATH), ("ftmo_futures", DEFAULT_COSTS_PATH)):
        for key in load_costs_policy(path)["symbols"]:
            out.append(pytest.param(broker, path, key, id=f"{broker}-{key}"))
    return out


# ---------------------------------------------------------------------------
# AC1.1 (+ E4, E5): paridad de los adaptadores con las funciones de HOY
# ---------------------------------------------------------------------------
def test_ac1_1_el_registro_oficial_carga_sin_errores_y_con_los_tres_brokers():
    reg = symbols.load_registry()
    assert reg.errors == []
    assert list(reg.brokers) == ["ftmo_cfd", "ftmo_futures", "axi"]
    assert all(b.official and b.currency == "USD" for b in reg.brokers.values())
    assert {b.id: (b.asset_class, b.clock, b.ohlc_dir) for b in reg.brokers.values()} == {
        "ftmo_cfd": ("cfd", "ny_plus_7", "risk_app/inputs/ohlc"),
        "ftmo_futures": ("futures", "ny_wall", None),
        "axi": ("cfd", "ny_plus_7", "risk_app/inputs/ohlc_axi")}
    assert reg.hidden == set() and reg.user_file_error is None


@pytest.mark.parametrize("broker_id, costs_path, key", _official_entries())
def test_ac1_1_paridad_ftmo_con_las_funciones_de_hoy(broker_id, costs_path, key):
    """pv = ``tick_value_usd / tick_size`` (``pv_yaml`` de sqx_trades), ``main._sqx_cost_spec`` y la comisión,
    los pasos y el ``ohlc_m1`` del YAML, comparados con ``==`` y sin números copiados."""
    costs = load_costs_policy(costs_path)
    spec = costs["symbols"][key]
    sqx_symbol = spec["sqx_symbol"]
    sheet = symbols.load_registry().sheets[sqx_symbol]
    assert (sheet.broker, sheet.official, sheet.usable, sheet.motivo) == (broker_id, True, True, None)
    assert sheet.asset_class == ("cfd" if costs.get("asset_class") == "cfd" else "futures")
    # sqx/sqx_trades.py: pv_yaml = yaml_tick_value / yaml_tick
    assert sheet.point_value_usd == float(spec["tick_value_usd"]) / float(spec["tick_size"])
    # main._sqx_cost_spec: la entrada y typical_spread_points; spread_px = spread_pts * tick
    spec_today, pts = main._sqx_cost_spec(sqx_symbol, costs_path)
    assert spec_today == spec
    assert sheet.tick_size == float(spec_today["tick_size"])
    # CFD: ``typical_spread_points`` va en pips del instrumento SQX (spread_px = pts · tick, como main._sqx_leg_minutes).
    # Futuros (H6, revisión de código de la F1): ya va en UNIDADES DE PRECIO (cabecera de ``sqx_policy``), spread_px = pts.
    if broker_id == "ftmo_futures":
        assert sheet.spread_px == (pts if pts is not None else None)
    else:
        assert sheet.spread_px == (pts * float(spec_today["tick_size"]) if pts is not None else None)
    assert sheet.ohlc_m1 == spec_today.get("ohlc_m1")
    if spec.get("commission_pct") is not None:
        assert sheet.commission == {"pct_notional_rt": float(spec["commission_pct"])}
    else:
        assert sheet.commission == {"per_lot_rt": float(spec["total_round_trip"])}
    if sheet.asset_class == "cfd":
        assert (sheet.lot_step, sheet.min_lot) == (float(spec["lot_step"]), float(spec["min_lot"]))
    else:
        assert (sheet.lot_step, sheet.min_lot) == (1, 1)
    assert sheet.max_lot is None and sheet.needs_review == () and sheet.verified_on is None
    assert set(sheet.sources) == {"ticker", "point_value_usd", "lot_step", "min_lot", "commission", "tick_size"} \
        | ({"spread_px"} if pts is not None else set()) | ({"ohlc_m1"} if sheet.ohlc_m1 else set())
    assert all(s == {"from": "official", "date": None} for s in sheet.sources.values())


def test_ac1_1_spread_px_ftmo_cfd_es_el_que_calcula_sqx_leg_minutes(tmp_path, monkeypatch):
    """Se ejecuta ``main._sqx_leg_minutes`` de verdad (con el OHLC y la serie simulados) y se captura el
    ``spread_px`` que le pasa a ``_sqx_minute_series``: es el de la ficha, con ``==``."""
    seen = {}
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")        # nunca se cargan los OHLC reales
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    real_ohlc = main._sqx_ohlc

    def fake_ohlc(symbol, costs_path):
        """La ruta sale del YAML como hoy; el fichero (que no existe en el worktree) se sustituye por un objeto."""
        _, path = real_ohlc(symbol, costs_path)
        return (object() if path is not None else None), path

    monkeypatch.setattr(main, "_sqx_ohlc", fake_ohlc)
    monkeypatch.setattr(main, "_sqx_ohlc_consistency", lambda df, ohlc, spread_px, tick: 1.0)

    def fake_series(df_srv, meta, fleg, ohlc, spread_px, to_axis):
        seen["spread_px"] = spread_px
        return {}, {}

    monkeypatch.setattr(main, "_sqx_minute_series", fake_series)
    reg = symbols.load_registry()
    n_with_series = 0
    for sheet in [s for s in reg.sheets.values() if s.broker == "ftmo_cfd"]:
        seen.clear()
        valle, diag, motivo = main._sqx_leg_minutes(None, {"symbol": sheet.sqx_symbol}, None, "cfd",
                                                    DEFAULT_CFD_COSTS_PATH, None)
        if sheet.ohlc_m1 is None:            # hoy: "sin ohlc_m1 en el YAML de costes" <=> la ficha no trae ohlc_m1
            assert valle is None and "sin ohlc_m1" in motivo, sheet.sqx_symbol
            continue
        assert motivo is None, (sheet.sqx_symbol, motivo)
        assert seen["spread_px"] == sheet.spread_px, sheet.sqx_symbol
        n_with_series += 1
    assert n_with_series == 3                 # EURUSD, XAUUSD y US100 llevan ohlc_m1 en el YAML


@pytest.mark.parametrize("broker_id, costs_path, key", _official_entries())
def test_ac1_1_la_ficha_la_acepta_read_trade_table_de_hoy(tmp_path, broker_id, costs_path, key):
    """Un ``.sqx`` sintético con el pv y el tick de la ficha pasa por ``read_trade_table`` de hoy sin error ni
    avisos de pointValue/tickSize, y su ``meta`` (pasos, comisión) coincide con la ficha."""
    spec = load_costs_policy(costs_path)["symbols"][key]
    sheet = symbols.load_registry().sheets[spec["sqx_symbol"]]
    f32 = lambda x: float(np.float32(x))                       # noqa: E731 - el .sqx guarda float32
    tick, pv = sheet.tick_size, sheet.point_value_usd
    lots = sheet.min_lot
    open_px, dist = f32(2048.0), 10 * tick
    sl_px, close_px = f32(open_px - dist), f32(open_px + dist)
    gross = (close_px - open_px) * pv * lots
    if "pct_notional_rt" in sheet.commission:
        comm = sheet.commission["pct_notional_rt"] / 100.0 * open_px * pv * lots
    else:
        comm = sheet.commission["per_lot_rt"] * lots
    trade = dict(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 4, 11, 0), lots=lots,
                 open_px=open_px, close_px=close_px, sl_px=sl_px, pnl=gross - comm, mae=0.5 * dist * pv * lots,
                 flag=1)
    path = write_sqx(tmp_path / "p.sqx", [trade], symbol=sheet.sqx_symbol, instrument="X", pv=pv, tick=tick)
    df, meta = read_trade_table(path, costs_path=costs_path)
    assert meta["symbol"] == sheet.sqx_symbol and meta["asset_class"] == sheet.asset_class
    assert meta["pv"] == pv and meta["tick_size"] == sheet.tick_size
    assert not [w for w in meta["warnings"] if "pointValue" in w or "tickSize" in w]
    if sheet.asset_class == "cfd":
        assert (meta["lot_step"], meta["min_lot"]) == (sheet.lot_step, sheet.min_lot)
        if "pct_notional_rt" in sheet.commission:
            assert meta["commission_pct"] == sheet.commission["pct_notional_rt"]
        else:
            assert meta["commission_per_lot"] == sheet.commission["per_lot_rt"]


def test_ac1_1_el_pv_de_la_ficha_fuera_de_tolerancia_lo_rechaza_el_lector_de_hoy(tmp_path):
    """Contraprueba: con otro pv el lector de hoy falla, así que el test anterior sí mide la paridad."""
    sheet = symbols.load_registry().sheets["EURUSD_ftmo"]
    trade = dict(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 4, 11, 0), lots=0.01, open_px=1.25,
                 close_px=1.26, sl_px=1.24, pnl=9.0, mae=1.0, flag=1)
    path = write_sqx(tmp_path / "p.sqx", [trade], symbol="EURUSD_ftmo", instrument="X",
                     pv=sheet.point_value_usd * 1.01, tick=sheet.tick_size)
    with pytest.raises(Exception, match="pointValue"):
        read_trade_table(path, costs_path=DEFAULT_CFD_COSTS_PATH)


def test_ac1_1_h6_spread_px_de_futuros_en_unidades_de_precio_valores_a_mano():
    """H6: en ``ftmo_futures_costs.yaml`` el spread típico ya está en unidades de precio (1 tick): MNQ 0,25;
    M2K 0,10; MGC 0,10; M6E 0,0001. Antes se multiplicaba otra vez por el tick (MNQ daba 0,0625)."""
    reg = symbols.load_registry()
    esperado = {"@micro_nasdaq": 0.25, "@micro_rusell": 0.10, "@micro_oro": 0.10, "@micro_eurusd": 0.0001}
    for sym, spread in esperado.items():
        assert reg.sheets[sym].spread_px == pytest.approx(spread, rel=0, abs=1e-12), sym
        assert reg.sheets[sym].spread_px == reg.sheets[sym].tick_size            # 1 tick en los cuatro micros
    # y el de los CFD sigue siendo pips · tick (no cambia)
    assert reg.sheets["EURUSD_ftmo"].spread_px == 0.3 * 0.0001
    assert reg.sheets["XAUUSD_ftmo(2)"].spread_px == 45 * 0.01


# E5: AXI con valores a mano (tolerancia 1e-12)
AXI_A_MANO = {          # sqx_symbol: (pv, comisión por lote ida y vuelta, spread_px, paso, mínimo)
    "XAUUSD_axi": (100.0, 4.5, 0.08, 0.01, 0.01),
    "EURUSD_axi": (100000.0, 4.5, 0.00002, None, None),
    "USATECHIDXUSD_axi": (20.0, 0.0, 2.5, None, None),
}


@pytest.mark.parametrize("sqx_symbol", sorted(AXI_A_MANO))
def test_ac1_1_axi_valores_a_mano_e5(sqx_symbol):
    pv, comm, spread, step, mn = AXI_A_MANO[sqx_symbol]
    sheet = symbols.load_registry().sheets[sqx_symbol]
    assert sheet.broker == "axi" and sheet.official and sheet.usable and sheet.asset_class == "cfd"
    assert sheet.point_value_usd == pytest.approx(pv, rel=0, abs=1e-12)
    assert sheet.commission == {"per_lot_rt": pytest.approx(comm, rel=0, abs=1e-12)}
    assert sheet.spread_px == pytest.approx(spread, rel=0, abs=1e-12)
    if step is not None:
        assert (sheet.lot_step, sheet.min_lot) == (pytest.approx(step, abs=1e-12), pytest.approx(mn, abs=1e-12))
    assert sheet.ohlc_m1 == f"{sqx_symbol}.csv"


def test_ac1_1_axi_paridad_con_broker_symbol_y_deuidxeur_no_usable():
    cfg = main.load_broker_config()
    reg = symbols.load_registry()
    axi = [s for s in reg.sheets.values() if s.broker == "axi"]
    assert len(axi) == len(cfg.symbols) == 7
    for s in cfg.symbols:
        sheet = reg.sheets[s.ticker_cascade + "_axi"]               # ticker_cascade + sqx_suffix
        assert sheet.ticker == s.ticker_axi
        assert sheet.point_value_usd == s.point_value_usd
        assert sheet.commission == {"per_lot_rt": 2 * s.commission_usd_per_lot_per_side}
        assert (sheet.lot_step, sheet.min_lot, sheet.max_lot) == (s.lot_step, s.min_lot, s.max_lot)
        assert sheet.spread_px == s.typical_spread_points * s.tick_size
        assert sheet.sources["spread_px"] == {"from": "official", "date": cfg.captured_at.isoformat()}
        assert sheet.tick_size is None
    deu = reg.sheets["DEUIDXEUR_axi"]
    assert deu.usable is False and deu.motivo == "divisa no USD (v1)"
    assert all(s.usable and s.motivo is None for s in axi if s.sqx_symbol != "DEUIDXEUR_axi")


def test_ac1_1_as_dict_es_json_estricto_y_no_comparte_estado():
    sheet = symbols.load_registry().sheets["XAUUSD_ftmo(2)"]
    d = sheet.as_dict()
    json.dumps(d, allow_nan=False)
    d["commission"]["per_lot_rt"] = 99
    d["sources"]["min_lot"]["from"] = "manual"
    d["needs_review"].append("x")
    assert sheet.commission == {"pct_notional_rt": 0.0014} and sheet.sources["min_lot"]["from"] == "official"
    assert sheet.needs_review == ()


# ---------------------------------------------------------------------------
# AC1.2 (+ E7): unicidad de sqx_symbol
# ---------------------------------------------------------------------------
def test_ac1_2_los_sqx_symbol_oficiales_son_unicos():
    """Los tres ficheros de especificaciones no repiten ``sqx_symbol`` y el registro conserva TODAS sus fichas."""
    names: list[str] = []
    for path in (DEFAULT_CFD_COSTS_PATH, DEFAULT_COSTS_PATH):
        names += [s["sqx_symbol"] for s in load_costs_policy(path)["symbols"].values()]
    names += [s.ticker_cascade + "_axi" for s in main.load_broker_config().symbols]
    reg = symbols.load_registry()
    assert len(names) == len(set(names)) == len(reg.sheets) == 16
    assert set(names) == set(reg.sheets) and reg.errors == []


def _write_costs(path: Path, sqx_symbol: str, **extra) -> Path:
    spec = {"sqx_symbol": sqx_symbol, "tick_size": 0.01, "tick_value_usd": 1.0, "total_round_trip": 1.0,
            "lot_step": 0.01, "min_lot": 0.01, **extra}
    path.write_text(_dump({"asset_class": "cfd", "symbols": {"A": spec},
                           "sqx_policy": {sqx_symbol: {"typical_spread_points": 3}}}), encoding="utf-8")
    return path


def _write_brokers(path: Path, brokers: dict, **top) -> Path:
    path.write_text(_dump({"schema_version": 1, "brokers": brokers, **top}), encoding="utf-8")
    return path


def _broker_def(file: Path, **kw) -> dict:
    d = {"display_name": "B", "company": "C", "currency": "USD", "asset_class": "cfd", "clock": "ny_plus_7",
         "source": {"adapter": "costs_yaml", "file": str(file)}}
    d.update(kw)
    return d


def test_ac1_2_un_simbolo_repetido_entre_brokers_oficiales_se_ignora_y_se_avisa(tmp_path, monkeypatch):
    uno = _write_costs(tmp_path / "uno.yaml", "DUP_x")
    dos = _write_costs(tmp_path / "dos.yaml", "DUP_x", tick_size=0.1)
    monkeypatch.setattr(symbols, "BROKERS_PATH", _write_brokers(
        tmp_path / "brokers.yaml", {"b_uno": _broker_def(uno), "b_dos": _broker_def(dos)}))
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "u.yaml")
    reg = symbols.load_registry()
    assert list(reg.sheets) == ["DUP_x"] and reg.sheets["DUP_x"].broker == "b_uno"
    assert len(reg.errors) == 1 and "repetido entre brókers oficiales" in reg.errors[0]


def test_ac1_2_usuario_repite_un_simbolo_oficial_error_e7(api):
    _mk_brokers(api)
    for target in ("mi_cfd", "ftmo_cfd"):                 # en un bróker propio o en el oficial de ese símbolo
        r = _send(api, "POST", "/api/symbols", _sheet(broker=target, sqx_symbol="EURUSD_ftmo"))
        assert r.status_code == 400
        msg = _errors(r)[0]
        assert msg.startswith("el símbolo SQX 'EURUSD_ftmo' ya tiene ficha en el bróker 'ftmo_cfd'; si es otro "
                              "bróker, dale a sus datos en SQX un nombre propio (p. ej. EURUSD_ftmo_mibroker)")
    assert "@micro_nasdaq" in symbols.load_registry().sheets                  # nada se perdió
    r = _send(api, "POST", "/api/symbols", _sheet(broker="axi", sqx_symbol="@micro_nasdaq"))
    assert r.status_code == 400 and "en el bróker 'ftmo_futures'" in _errors(r)[0]


def test_ac1_2_usuario_repite_un_simbolo_de_usuario_error_e7(api, user_file):
    _mk_brokers(api)
    _create(api)
    before = user_file.read_text(encoding="utf-8")
    r = _send(api, "POST", "/api/symbols", _sheet(broker="mi_cfd"))
    assert r.status_code == 400 and "ya tiene ficha en el bróker 'mi_cfd'" in _errors(r)[0]
    assert "XAUUSD_mi_mibroker" in _errors(r)[0]
    # y en OTRO bróker (único entre todos): el mensaje nombra al bróker que ya la tiene
    r = _send(api, "POST", "/api/symbols", _sheet(broker="ftmo_cfd"))
    assert r.status_code == 400 and "ya tiene ficha en el bróker 'mi_cfd'" in _errors(r)[0]
    assert user_file.read_text(encoding="utf-8") == before


def test_ac1_2_un_fichero_de_usuario_con_un_simbolo_repetido_conserva_solo_el_primero(user_file):
    ficha = {"point_value_usd": 1.0, "lot_step": 0.01, "min_lot": 0.01, "commission": {"per_lot_rt": 0.0}}
    user_file.write_text(_dump({
        "brokers": {"mi_cfd": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"}},
        "symbols": {"mi_cfd": {"EURUSD_ftmo": ficha, "NUEVO_a": ficha}, "ftmo_cfd": {"NUEVO_a": ficha}}}),
        encoding="utf-8")
    symbols._reload()
    reg = symbols.load_registry()
    assert reg.sheets["EURUSD_ftmo"].official and reg.sheets["NUEVO_a"].broker == "mi_cfd"
    assert len(reg.errors) == 2 and all("ya tiene ficha" in e for e in reg.errors)
    assert [e for e in reg.errors if "mi_cfd/EURUSD_ftmo" in e] and [e for e in reg.errors if "ftmo_cfd/NUEVO_a" in e]


# ---------------------------------------------------------------------------
# AC1.3: CRUD completo con TestClient
# ---------------------------------------------------------------------------
def test_ac1_3_listado_inicial_con_procedencia_y_sin_fichero_de_usuario(api, user_file):
    r = api.get("/api/symbols")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"brokers", "symbols", "errors", "profiles", "lenses"} and body["errors"] == []
    assert [b["id"] for b in body["brokers"]] == ["ftmo_cfd", "ftmo_futures", "axi"]
    assert len(body["symbols"]) == 16 and all(s["official"] and s["hidden"] is False for s in body["symbols"])
    eur = next(s for s in body["symbols"] if s["sqx_symbol"] == "EURUSD_ftmo")
    assert eur["sources"]["point_value_usd"] == {"from": "official", "date": None}
    assert body["profiles"]["apex_150k_intraday"] is None and body["lenses"] == {"axi": "axi"}
    assert not user_file.exists()                                  # listar no crea el fichero de usuario


def test_ac1_3_crud_completo_alta_edicion_ocultar_borrado_y_ficheros_reales_intactos(api, user_file):
    _mk_brokers(api)
    created = _create(api)
    assert created["official"] is False and created["usable"] is True and created["hidden"] is False
    assert created["sources"]["min_lot"] == {"from": "manual", "date": created["sources"]["min_lot"]["date"]}
    assert created["sources"]["ticker"]["from"] == "manual" and created["needs_review"] == []
    # alta en un bróker OFICIAL (el usuario puede añadir símbolos a un bróker oficial)
    r = _send(api, "POST", "/api/symbols", _sheet(broker="ftmo_cfd", sqx_symbol="GBPJPY_ftmo", ticker="GBPJPY",
                                                  ohlc_m1=_DEL))
    assert r.status_code == 200 and r.json()["broker"] == "ftmo_cfd" and r.json()["official"] is False
    listing = {s["sqx_symbol"]: s for s in api.get("/api/symbols").json()["symbols"]}
    assert len(listing) == 18 and listing["XAUUSD_mi"]["broker"] == "mi_cfd"
    # edición parcial (PUT): se conserva lo no enviado y null borra un campo opcional
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"min_lot": 0.02, "spread_px": None, "notes": "n"})
    assert r.status_code == 200, r.text
    ed = r.json()
    assert (ed["min_lot"], ed["spread_px"], ed["notes"], ed["max_lot"]) == (0.02, None, "n", 50.0)
    assert ed["commission"] == {"per_lot_rt": 4.5} and ed["ohlc_m1"] == "XAUUSD_mi.csv"
    assert "spread_px" not in ed["sources"] and ed["sources"]["min_lot"]["from"] == "manual"
    # ocultar / mostrar (oficial y de usuario): ocultar no borra y `resolve` la deja de ver
    assert symbols.resolve("XAUUSD_mi") is not None and symbols.resolve("GBPUSD_ftmo") is not None
    r = _hide(api, ["ftmo_cfd/GBPUSD_ftmo", "mi_cfd/XAUUSD_mi"])
    assert r.status_code == 200 and r.json() == {"hidden": ["ftmo_cfd/GBPUSD_ftmo", "mi_cfd/XAUUSD_mi"]}
    assert symbols.resolve("XAUUSD_mi") is None and symbols.resolve("GBPUSD_ftmo") is None
    listing = {s["sqx_symbol"]: s for s in api.get("/api/symbols").json()["symbols"]}
    assert listing["XAUUSD_mi"]["hidden"] is True and listing["GBPUSD_ftmo"]["hidden"] is True
    assert listing["EURUSD_ftmo"]["hidden"] is False and "XAUUSD_mi" in symbols.load_registry().sheets
    assert _hide(api, ["mi_cfd/XAUUSD_mi"], False).json() == {"hidden": ["ftmo_cfd/GBPUSD_ftmo"]}
    assert symbols.resolve("XAUUSD_mi").min_lot == 0.02
    # borrado (solo de usuario): la quita también de `hidden`
    assert _hide(api, ["mi_cfd/XAUUSD_mi"]).status_code == 200
    assert api.delete("/api/symbols/mi_cfd/XAUUSD_mi").json() == {"deleted": "mi_cfd/XAUUSD_mi"}
    assert "XAUUSD_mi" not in symbols.load_registry().sheets
    assert _doc(user_file)["hidden"] == ["ftmo_cfd/GBPUSD_ftmo"]
    assert api.delete("/api/symbols/mi_cfd/XAUUSD_mi").status_code == 404
    # el fichero: bróker + ficha de GBPJPY en el oficial, sin restos .tmp
    doc = _doc(user_file)
    assert set(doc) == {"brokers", "symbols", "hidden"} and set(doc["brokers"]) == {"mi_cfd", "mi_fut"}
    assert "GBPJPY_ftmo" in doc["symbols"]["ftmo_cfd"] and "mi_cfd" not in doc["symbols"]
    assert not list(user_file.parent.glob("*.tmp"))
    for p in REAL_USER_FILES:                                      # I4: los tests nunca tocan los ficheros reales
        assert p.exists() == (p in _REAL_BEFORE)


def test_ac1_3_editar_una_ficha_ajena_o_inexistente(api):
    _mk_brokers(api)
    _create(api)
    assert _send(api, "PUT", "/api/symbols/mi_cfd/NO_existe", {"min_lot": 0.02}).status_code == 404
    assert _send(api, "PUT", "/api/symbols/no_broker/XAUUSD_mi", {"min_lot": 0.02}).status_code == 404
    assert _send(api, "PUT", "/api/symbols/ftmo_cfd/XAUUSD_mi", {"min_lot": 0.02}).status_code == 404   # otro bróker
    for k, v in (("sqx_symbol", "OTRO_a"), ("broker", "ftmo_cfd")):         # no se renombra ni se cambia de bróker
        r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {k: v})
        assert r.status_code == 400 and "no se puede cambiar" in _errors(r)[0]
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"min_lot": 0.015})        # la fusión se valida entera
    assert r.status_code == 400 and "múltiplo" in _errors(r)[0]
    assert _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"min_lot": None}).status_code == 400   # obligatorio


def test_ac1_3_la_edicion_conserva_o_actualiza_la_procedencia_y_los_campos_sin_verificar(api):
    _mk_brokers(api)
    _create(api, sources={"point_value_usd": {"from": "sqx", "date": "2026-10-19"},
                          "min_lot": {"from": "mt5_sheet", "date": "2026-10-18"}},
            needs_review=["point_value_usd", "spread_px"])
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"notes": "solo notas"})
    ed = r.json()
    assert ed["sources"]["point_value_usd"] == {"from": "sqx", "date": "2026-10-19"}      # sin tocar: se conserva
    assert ed["sources"]["min_lot"] == {"from": "mt5_sheet", "date": "2026-10-18"}
    assert ed["needs_review"] == ["point_value_usd", "spread_px"]
    # editar a mano un campo sin verificar lo da por revisado; los demás siguen en la lista
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"point_value_usd": 100.5})
    ed = r.json()
    assert ed["sources"]["point_value_usd"]["from"] == "manual" and ed["needs_review"] == ["spread_px"]
    # aceptar un valor de otra fuente campo a campo (E11): la procedencia viaja en el mismo PUT
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {
        "min_lot": 0.02, "sources": {"min_lot": {"from": "mt5_sheet", "date": TODAY}}, "needs_review": []})
    ed = r.json()
    assert ed["sources"]["min_lot"] == {"from": "mt5_sheet", "date": TODAY} and ed["needs_review"] == []


def test_ac1_3_escritura_atomica_el_destino_nunca_queda_a_medias(api, user_file, monkeypatch):
    _mk_brokers(api)
    _create(api)
    old = user_file.read_text(encoding="utf-8")
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        # en el instante del reemplazo, el temporal ya está COMPLETO y es YAML válido; el destino sigue siendo el viejo
        seen["tmp_symbols"] = set(yaml.safe_load(Path(src).read_text(encoding="utf-8"))["symbols"]["mi_cfd"])
        seen["dst_is_old"] = Path(dst).read_text(encoding="utf-8") == old
        real_replace(src, dst)

    monkeypatch.setattr(symbols, "os", types.SimpleNamespace(replace=spy))
    _create(api, sqx_symbol="XAGUSD_mi", ticker="XAGUSD")
    assert seen == {"tmp_symbols": {"XAUUSD_mi", "XAGUSD_mi"}, "dst_is_old": True}
    assert set(_doc(user_file)["symbols"]["mi_cfd"]) == {"XAUUSD_mi", "XAGUSD_mi"}
    # y si el reemplazo falla: 500, el fichero queda como estaba y no sobra el temporal
    good = user_file.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("disco lleno")

    monkeypatch.setattr(symbols, "os", types.SimpleNamespace(replace=boom))
    r = _send(api, "POST", "/api/symbols", _sheet(sqx_symbol="OTRA_mi"))
    assert r.status_code == 500 and "disco lleno" in r.json()["detail"]
    assert user_file.read_text(encoding="utf-8") == good and not list(user_file.parent.glob("*.tmp"))
    monkeypatch.undo()
    symbols._reload()
    assert "OTRA_mi" not in symbols.load_registry().sheets


def test_ac1_3_fichero_corrupto_es_409_en_escrituras_y_error_en_el_listado(api, user_file):
    user_file.write_text("brokers: [sin cerrar\n  - x: {", encoding="utf-8")
    symbols._reload()
    body = api.get("/api/symbols").json()
    assert len(body["symbols"]) == 16 and len(body["errors"]) == 1 and "ilegible" in body["errors"][0]
    assert symbols.load_registry().user_file_error == body["errors"][0]
    peticiones = [("POST", "/api/symbols", _sheet()),
                  ("POST", "/api/symbols/brokers", {"id": "otro_b", "display_name": "x", "company": "y",
                                                    "asset_class": "cfd", "clock": "ny_wall"}),
                  ("PUT", "/api/symbols/ftmo_cfd/EURUSD_ftmo", {"min_lot": 0.02}),
                  ("DELETE", "/api/symbols/ftmo_cfd/EURUSD_ftmo", None),
                  ("POST", "/api/symbols/visibility", {"keys": ["ftmo_cfd/EURUSD_ftmo"], "hidden": True})]
    for method, url, obj in peticiones:
        r = _send(api, method, url, obj) if obj is not None else api.request(method, url)
        if method == "PUT":                       # la ficha oficial se rechaza ANTES de leer el fichero: 400
            assert r.status_code in (400, 409), (method, url, r.text)
            continue
        assert r.status_code == 409 and "ilegible" in r.json()["detail"], (method, url, r.text)
    assert user_file.read_text(encoding="utf-8") == "brokers: [sin cerrar\n  - x: {"   # no se pisa
    for text in ("- a\n- b\n", "brokers: 5\n", "symbols: [1, 2]\n", "symbols: {axi: 5}\n", "hidden: 3\n",
                 "hidden: [1, 2]\n"):
        user_file.write_text(text, encoding="utf-8")
        symbols._reload()
        assert api.get("/api/symbols").json()["errors"], text
        assert _send(api, "POST", "/api/symbols", _sheet()).status_code in (404, 409), text


def test_ac1_3_un_fichero_a_medias_carga_entrada_a_entrada(api, user_file):
    ok = {"point_value_usd": 100, "lot_step": 0.01, "min_lot": 0.01, "commission": {"per_lot_rt": 4.5}}
    user_file.write_text(_dump({
        "brokers": {"mi_cfd": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"},
                    "Mal Id": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"},
                    "axi": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"},
                    "mal_clock": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "utc"}},
        "symbols": {"mi_cfd": {"BUENA_a": {**ok, "verified_on": datetime(2026, 10, 1).date()},
                               "MALA_a": {**ok, "min_lot": 0.015}, "sin_pv": {"lot_step": 0.01},
                               "otra moneda": {**ok, "currency": "EUR"}, "NO_dict": 5},
                    "no_existe": {"X_a": ok}}}), encoding="utf-8")
    symbols._reload()
    reg = symbols.load_registry()
    assert reg.sheets["BUENA_a"].broker == "mi_cfd" and "MALA_a" not in reg.sheets
    buena = reg.sheets["BUENA_a"]
    assert buena.verified_on == "2026-10-01"                                      # fecha YAML sin comillas -> ISO
    assert buena.ticker == "BUENA_a" and buena.official is False and buena.usable is True
    assert buena.sources["point_value_usd"] == {"from": "manual", "date": None}      # sin `sources` en el fichero
    assert set(reg.brokers) == {"ftmo_cfd", "ftmo_futures", "axi", "mi_cfd"} and reg.brokers["axi"].official
    joined = "\n".join(reg.errors)
    for needle in ("Mal Id", "axi", "mal_clock", "mi_cfd/MALA_a", "mi_cfd/sin_pv", "mi_cfd/otra moneda",
                   "mi_cfd/NO_dict", "no_existe"):
        assert needle in joined, needle
    assert reg.user_file_error is None
    assert api.get("/api/symbols").json()["errors"] == reg.errors
    # escribir sobre un fichero a medias NO borra las entradas con error
    _create(api, sqx_symbol="NUEVA_a", ohlc_m1=_DEL)           # `mi_cfd` del fichero no tiene ohlc_dir
    assert {"MALA_a", "sin_pv", "NO_dict"} <= set(_doc(user_file)["symbols"]["mi_cfd"])


def test_ac1_3_redefinir_un_simbolo_oficial_es_400(api, user_file):
    _mk_brokers(api)
    before = user_file.read_text(encoding="utf-8")
    r = _send(api, "POST", "/api/symbols", _sheet(broker="ftmo_cfd", sqx_symbol="XAUUSD_ftmo(2)"))
    assert r.status_code == 400 and "ya tiene ficha en el bróker 'ftmo_cfd'" in _errors(r)[0]
    r = _send(api, "PUT", "/api/symbols/ftmo_cfd/EURUSD_ftmo", {"min_lot": 0.02})
    assert r.status_code == 400 and "ficha oficial" in _errors(r)[0]
    r = _send(api, "PUT", "/api/symbols/axi/XAUUSD_axi", _sheet(broker="axi", sqx_symbol="XAUUSD_axi"))
    assert r.status_code == 400
    r = api.delete("/api/symbols/ftmo_cfd/EURUSD_ftmo")
    assert r.status_code == 400 and "ficha oficial" in _errors(r)[0]
    assert user_file.read_text(encoding="utf-8") == before
    assert symbols.resolve("EURUSD_ftmo").point_value_usd == 100000.0           # la oficial sigue intacta
    # lo único que se hace con una oficial es ocultarla (y mostrarla)
    assert _hide(api, ["ftmo_cfd/EURUSD_ftmo"]).status_code == 200 and symbols.resolve("EURUSD_ftmo") is None
    assert _hide(api, ["ftmo_cfd/EURUSD_ftmo"], False).status_code == 200 and symbols.resolve("EURUSD_ftmo")


def test_ac1_3_visibilidad_atomica_y_validada(api, user_file):
    _mk_brokers(api)
    r = _hide(api, ["ftmo_cfd/EURUSD_ftmo", "no_broker/X_a"])
    assert r.status_code == 404 and r.json()["detail"]["unknown"] == ["no_broker/X_a"]
    assert not user_file.exists() or _doc(user_file)["hidden"] == []                    # todo o nada
    for bad in ({"keys": [], "hidden": True}, {"keys": "x", "hidden": True}, {"keys": [1], "hidden": True},
                {"keys": ["ftmo_cfd/EURUSD_ftmo"], "hidden": "si"}, {"hidden": True}):
        assert _send(api, "POST", "/api/symbols/visibility", bad).status_code == 422, bad


def test_ac1_3_exportar_e_importar_el_sobre_risk_app_symbols_v1(api, user_file):
    _mk_brokers(api)
    created = _create(api, needs_review=["spread_px"], notes="nota", verified_on="2026-10-19")
    exp = api.get("/api/symbols/mi_cfd/XAUUSD_mi/export").json()
    assert exp["format"] == "risk-app-symbols" and exp["version"] == 1
    assert (exp["broker"], exp["sqx_symbol"]) == ("mi_cfd", "XAUUSD_mi")
    assert exp["broker_def"] == {"display_name": "Mi CFD", "company": "Mi Compañía", "currency": "USD",
                                 "asset_class": "cfd", "clock": "ny_plus_7", "ohlc_dir": "risk_app/inputs/ohlc_mi"}
    assert exp["sheet"]["needs_review"] == ["spread_px"] and exp["sheet"]["notes"] == "nota"
    assert api.delete("/api/symbols/mi_cfd/XAUUSD_mi").status_code == 200
    r = _send(api, "POST", "/api/symbols/import", exp)                  # vuelve igual
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["broker_created"] is False and out["message"] == "importado"
    assert out["sheet"] == created
    # importar sobre un bróker de usuario que no existe: el sobre trae su definición y se crea en la misma escritura
    doc = _doc(user_file)
    doc["brokers"].pop("mi_cfd")
    doc["symbols"].pop("mi_cfd")
    user_file.write_text(_dump(doc), encoding="utf-8")
    symbols._reload()
    assert "mi_cfd" not in symbols.load_registry().brokers
    r = _send(api, "POST", "/api/symbols/import", exp)
    assert r.status_code == 200 and r.json()["broker_created"] is True
    assert symbols.load_registry().brokers["mi_cfd"].ohlc_dir == "risk_app/inputs/ohlc_mi"
    # exportar una oficial y volver a importarla: ya existe -> E7; en otro id de bróker propio, la procedencia pasa a manual
    off = api.get("/api/symbols/ftmo_futures/%40micro_nasdaq/export").json()
    assert off["broker_def"] is None and off["sheet"]["sources"]["point_value_usd"]["from"] == "official"
    assert _send(api, "POST", "/api/symbols/import", off).status_code == 400
    # formato y versión
    assert _send(api, "POST", "/api/symbols/import", {**exp, "format": "otra-cosa"}).status_code == 422
    assert _send(api, "POST", "/api/symbols/import", {**exp, "version": 2}).status_code == 422
    assert _send(api, "POST", "/api/symbols/import", {**exp, "version": True}).status_code == 422
    assert _send(api, "POST", "/api/symbols/import", {**exp, "sheet": 5}).status_code == 422
    r = _send(api, "POST", "/api/symbols/import", {**exp, "broker": "no_existe", "broker_def": None})
    assert r.status_code == 404
    bad = copy.deepcopy(exp)
    bad["sheet"]["min_lot"] = 0.015
    bad["sqx_symbol"] = "OTRA_mi"
    r = _send(api, "POST", "/api/symbols/import", bad)
    assert r.status_code == 400 and "múltiplo" in _errors(r)[0]
    assert "OTRA_mi" not in symbols.load_registry().sheets


def test_ac1_3_importar_convierte_la_procedencia_oficial_en_manual(api):
    _mk_brokers(api)
    off = api.get("/api/symbols/ftmo_cfd/EURUSD_ftmo/export").json()
    off.update(broker="mi_cfd", sqx_symbol="EURUSD_mi")
    off["sheet"]["ohlc_m1"] = "EURUSD_mi.csv"
    r = _send(api, "POST", "/api/symbols/import", off)
    assert r.status_code == 200, r.text
    sheet = r.json()["sheet"]
    assert sheet["official"] is False and sheet["point_value_usd"] == 100000.0
    assert {s["from"] for s in sheet["sources"].values()} == {"manual"}
    assert all(s["date"] is None for s in sheet["sources"].values())        # la fecha original (la de la oficial)


def test_ac1_3_alta_de_broker_de_usuario(api, user_file):
    ok = {"id": "mi_broker", "display_name": "  Mi   bróker ", "company": "  Mi  Compañía ", "asset_class": "cfd",
          "clock": "ny_wall", "ohlc_dir": "risk_app/inputs/ohlc_mio"}
    r = _send(api, "POST", "/api/symbols/brokers", ok)
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "mi_broker", "display_name": "Mi bróker", "company": "Mi Compañía",
                        "currency": "USD", "asset_class": "cfd", "clock": "ny_wall",
                        "ohlc_dir": "risk_app/inputs/ohlc_mio", "official": False}
    assert _doc(user_file)["brokers"]["mi_broker"]["currency"] == "USD"
    assert symbols.load_registry().brokers["mi_broker"].official is False
    assert [b["id"] for b in api.get("/api/symbols").json()["brokers"]] == [
        "ftmo_cfd", "ftmo_futures", "axi", "mi_broker"]
    # no reutiliza el id de un oficial ni el de uno propio
    for dup, kind in (("axi", "bróker oficial"), ("mi_broker", "tus brókers")):
        r = _send(api, "POST", "/api/symbols/brokers", {**ok, "id": dup})
        assert r.status_code == 400 and kind in _errors(r)[0], (dup, r.text)
    # un bróker propio acepta fichas propias y los símbolos siguen siendo únicos entre todos los brókers
    r = _send(api, "POST", "/api/symbols", _sheet(broker="mi_broker", ohlc_m1=_DEL))
    assert r.status_code == 200 and r.json()["broker"] == "mi_broker"


def test_ac1_3_concurrencia_no_corrompe_el_fichero(api, user_file):
    _mk_brokers(api)
    errors: list[BaseException] = []

    def creator(prefix):
        try:
            for i in range(5):
                r = _send(api, "POST", "/api/symbols",
                          _sheet(sqx_symbol=f"{prefix}_{i}", ticker=f"{prefix}{i}", ohlc_m1=_DEL))
                assert r.status_code == 200, r.text
        except BaseException as exc:                       # pragma: no cover
            errors.append(exc)

    def toggler():
        try:
            for i in range(10):
                assert _hide(api, ["ftmo_cfd/EURUSD_ftmo", "axi/XAUUSD_axi"], i % 2 == 0).status_code == 200
        except BaseException as exc:                       # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=creator, args=("AA",)), threading.Thread(target=creator, args=("BB",)),
          threading.Thread(target=toggler)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    d = _doc(user_file)
    assert len(d["symbols"]["mi_cfd"]) == 10 and d["hidden"] == []            # i=9 es impar -> mostradas
    assert api.get("/api/symbols").json()["errors"] == [] and not list(user_file.parent.glob("*.tmp"))


def test_ac1_3_el_cuerpo_debe_ser_json_estricto(api):
    for url in ("/api/symbols", "/api/symbols/brokers", "/api/symbols/validate", "/api/symbols/import",
                "/api/symbols/mt5-sheet", "/api/symbols/visibility"):
        for text in ('{"point_value_usd": NaN}', '{"point_value_usd": Infinity}', '{"x": 1e999}', "no es json",
                     "[1, 2]", '"texto"'):
            assert _send(api, "POST", url, text=text).status_code == 422, (url, text)


def test_ac1_3_el_gancho_de_recarga_vacia_la_cache_de_patas_de_main(api, monkeypatch):
    """Una escritura de fichas invalida ``main._leg_cache`` (lo registra main con ``set_reload_hook``)."""
    assert symbols._reload_hook is not None
    _mk_brokers(api)
    monkeypatch.setattr(main, "_leg_cache", {("sentinela",): {"x": 1}})
    _create(api)
    assert main._leg_cache == {}
    monkeypatch.setattr(main, "_leg_cache", {("sentinela",): {"x": 1}})
    assert _hide(api, ["ftmo_cfd/EURUSD_ftmo"]).status_code == 200 and main._leg_cache == {}
    monkeypatch.setattr(main, "_leg_cache", {("sentinela",): {"x": 1}})
    assert api.get("/api/symbols").status_code == 200 and main._leg_cache == {("sentinela",): {"x": 1}}   # leer no
    assert _send(api, "POST", "/api/symbols", _sheet(min_lot=0.015, sqx_symbol="MAL_a")).status_code == 400
    assert main._leg_cache == {("sentinela",): {"x": 1}}                      # un rechazo tampoco


def test_ac1_3_las_rutas_estan_montadas_en_la_app(api):
    paths = api.get("/openapi.json").json()["paths"]
    for method, path in (("get", "/api/symbols"), ("get", "/api/symbols/schema"), ("post", "/api/symbols/validate"),
                         ("post", "/api/symbols"), ("put", "/api/symbols/{broker_id}/{sqx_symbol}"),
                         ("delete", "/api/symbols/{broker_id}/{sqx_symbol}"), ("post", "/api/symbols/visibility"),
                         ("post", "/api/symbols/brokers"), ("get", "/api/symbols/{broker_id}/{sqx_symbol}/export"),
                         ("post", "/api/symbols/import"), ("post", "/api/symbols/mt5-sheet"),
                         ("get", "/api/symbols/draft-from-sqx")):
        # PLAN-DESVIO: en la F1 esta lista no incluía ``draft-from-sqx`` y el test fijaba que NO estaba montada
        # ("es de la fase F2"). La F2 lo añade, así que la aserción pasa a exigirlo.
        assert method in paths.get(path, {}), (method, path)


def test_ac1_3_schema_describe_el_formulario(api):
    s = api.get("/api/symbols/schema").json()
    names = [f["name"] for f in s["sheet_fields"]]
    assert names == ["broker", "sqx_symbol", "ticker", "point_value_usd", "lot_step", "min_lot", "max_lot",
                     "tick_size", "commission", "spread_px", "ohlc_m1", "verified_on", "notes"]
    pv = next(f for f in s["sheet_fields"] if f["name"] == "point_value_usd")
    assert pv["min"] == 0.0 and pv["min_exclusive"] is True and pv["required"] is True
    assert [e["id"] for e in s["enums"]["clock"]] == ["ny_plus_7", "ny_wall"]
    assert {f["name"] for f in s["broker_fields"]} == {"id", "display_name", "company", "currency", "asset_class",
                                                       "clock", "ohlc_dir"}
    json.dumps(s, allow_nan=False)


# ---------------------------------------------------------------------------
# AC1.4: validación
# ---------------------------------------------------------------------------
CFD_INVALIDAS = [
    ("moneda_no_usd", dict(currency="EUR"), "currency"),
    ("pv_cero", dict(point_value_usd=0), "point_value_usd"),
    ("pv_negativo", dict(point_value_usd=-5), "mayor que 0"),
    ("pv_texto", dict(point_value_usd="100"), "debe ser un número"),
    ("pv_booleano", dict(point_value_usd=True), "debe ser un número"),
    ("pv_falta", dict(point_value_usd=_DEL), "falta el valor"),
    ("sin_lot_step", dict(lot_step=_DEL), "lot_step"),
    ("sin_min_lot", dict(min_lot=_DEL), "min_lot"),
    ("lot_step_cero", dict(lot_step=0), "lot_step"),
    ("min_lot_no_multiplo", dict(min_lot=0.015), "no es múltiplo de lot_step"),
    ("min_lot_menor_que_el_paso", dict(lot_step=0.1, min_lot=0.01), "no es múltiplo de lot_step"),
    ("dos_comisiones", dict(commission={"per_lot_rt": 4.5, "pct_notional_rt": 0.001}), "un solo tipo de comisión"),
    ("sin_comision", dict(commission=_DEL), "commission"),
    ("comision_vacia", dict(commission={}), "falta"),
    ("comision_clave_rara", dict(commission={"por_lado": 2}), "clave desconocida"),
    ("comision_negativa", dict(commission={"per_lot_rt": -1}), "commission.per_lot_rt"),
    ("comision_pct_mayor_que_100", dict(commission={"pct_notional_rt": 120}), "commission.pct_notional_rt"),
    ("comision_texto", dict(commission="4.5"), "commission"),
    ("max_lot_menor_que_min_lot", dict(max_lot=0.005), "max_lot"),
    ("spread_negativo", dict(spread_px=-1), "spread_px"),
    ("tick_negativo", dict(tick_size=-1), "tick_size"),
    ("ohlc_con_carpeta_padre", dict(ohlc_m1="../x.csv"), "ohlc_m1"),
    ("ohlc_con_carpeta", dict(ohlc_m1="a/b.csv"), "ohlc_m1"),
    ("campo_desconocido", dict(foo=1), "campos no reconocidos: foo"),
    ("sqx_symbol_con_barra", dict(sqx_symbol="A/B"), "sqx_symbol"),
    ("sqx_symbol_con_espacio", dict(sqx_symbol="A B"), "sqx_symbol"),
    ("sqx_symbol_vacio", dict(sqx_symbol=""), "sqx_symbol"),
    ("sqx_symbol_dos_puntos", dict(sqx_symbol=".."), "sqx_symbol"),
    ("asset_class_distinto_del_broker", dict(asset_class="futures"), "asset_class"),
    ("fuente_oficial_reservada", dict(sources={"min_lot": {"from": "official", "date": None}}), "sources.min_lot"),
    ("fuente_de_campo_desconocido", dict(sources={"foo": {"from": "manual", "date": None}}), "sources.foo"),
    ("fecha_de_verificacion_mala", dict(verified_on="2026-13-45"), "verified_on"),
    ("notas_demasiado_largas", dict(notes="x" * 2001), "notes"),
    ("revision_de_campo_desconocido", dict(needs_review=["foo"]), "needs_review"),
    ("revision_de_currency_no_es_de_ficha", dict(needs_review=["currency"]), "needs_review"),
    ("ticker_en_blanco", dict(ticker=" "), "ticker"),
]


@pytest.mark.parametrize("override, needle", [pytest.param(o, n, id=i) for i, o, n in CFD_INVALIDAS])
def test_ac1_4_ficha_cfd_invalida_es_400_y_no_se_guarda(api, user_file, override, needle):
    _mk_brokers(api)
    before = user_file.read_text(encoding="utf-8")
    r = _send(api, "POST", "/api/symbols", _sheet(**override))
    assert r.status_code == 400, r.text
    assert any(needle in e for e in _errors(r)), _errors(r)
    assert user_file.read_text(encoding="utf-8") == before
    # POST /validate da el mismo veredicto sin guardar
    v = _send(api, "POST", "/api/symbols/validate", _sheet(**override)).json()
    assert v["ok"] is False and any(needle in e for e in v["errors"]) and v["sheet"] is None


FUTUROS_INVALIDAS = [
    ("paso_distinto_de_1", dict(lot_step=0.5), "vale 1"),
    ("minimo_distinto_de_1", dict(min_lot=2), "vale 1"),
    ("comision_porcentual", dict(commission={"pct_notional_rt": 0.001}), "solo vale en CFD"),
    ("ohlc_sin_carpeta", dict(ohlc_m1="x.csv"), "no tiene carpeta de OHLC"),
]


@pytest.mark.parametrize("override, needle", [pytest.param(o, n, id=i) for i, o, n in FUTUROS_INVALIDAS])
def test_ac1_4_ficha_de_futuros_invalida(api, override, needle):
    _mk_brokers(api)
    r = _send(api, "POST", "/api/symbols", _fut_sheet(**override))
    assert r.status_code == 400 and any(needle in e for e in _errors(r)), r.text


def test_ac1_4_futuros_tienen_paso_y_minimo_1_fijos(api):
    _mk_brokers(api)
    r = _send(api, "POST", "/api/symbols", _fut_sheet())            # sin paso ni mínimo: se fijan a 1
    assert r.status_code == 200, r.text
    assert (r.json()["lot_step"], r.json()["min_lot"], r.json()["asset_class"]) == (1.0, 1.0, "futures")
    r = _send(api, "POST", "/api/symbols", _fut_sheet(sqx_symbol="@otro_micro", lot_step=1, min_lot=1.0))
    assert r.status_code == 200 and (r.json()["lot_step"], r.json()["min_lot"]) == (1.0, 1.0)
    # también en un bróker oficial de futuros
    r = _send(api, "POST", "/api/symbols", _fut_sheet(broker="ftmo_futures", sqx_symbol="@micro_dax"))
    assert r.status_code == 200 and r.json()["broker"] == "ftmo_futures"


def test_ac1_4_ficha_valida_se_normaliza(api):
    _mk_brokers(api)
    r = _send(api, "POST", "/api/symbols", _sheet(point_value_usd=100, lot_step=1, min_lot=1, max_lot=10,
                                                  commission={"pct_notional_rt": 0}, spread_px=0, tick_size=1,
                                                  ticker="  Mi   ticker "))
    assert r.status_code == 200, r.text
    s = r.json()
    assert all(isinstance(s[k], float) for k in ("point_value_usd", "lot_step", "min_lot", "max_lot", "tick_size"))
    assert s["commission"] == {"pct_notional_rt": 0.0} and s["spread_px"] == 0.0 and s["ticker"] == "Mi ticker"
    # 0,07 es múltiplo de 0,01 aunque 0,07 / 0,01 no sea exactamente 7 en coma flotante (tolerancia 1e-9)
    r = _send(api, "POST", "/api/symbols", _sheet(sqx_symbol="B_mi", min_lot=0.07, lot_step=0.01))
    assert r.status_code == 200, r.text


def test_ac1_4_validacion_de_brokers_clock_moneda_y_demas(api):
    ok = {"id": "otro_broker", "display_name": "Otro", "company": "Co", "asset_class": "cfd", "clock": "ny_wall"}
    malos = [
        ("clock fuera de la lista", {**ok, "clock": "utc"}, "clock"),
        ("clock IANA (fuera de la v1)", {**ok, "clock": "Europe/Athens"}, "clock"),
        ("sin clock", {k: v for k, v in ok.items() if k != "clock"}, "clock"),
        ("moneda EUR", {**ok, "currency": "EUR"}, "solo USD"),
        ("clase de activo desconocida", {**ok, "asset_class": "fx"}, "asset_class"),
        ("id con mayúsculas", {**ok, "id": "Otro_Broker"}, "id inválido"),
        ("id demasiado corto", {**ok, "id": "ab"}, "id inválido"),
        ("sin id", {k: v for k, v in ok.items() if k != "id"}, "id inválido"),
        ("sin nombre", {k: v for k, v in ok.items() if k != "display_name"}, "display_name"),
        ("compañía en blanco", {**ok, "company": "  "}, "company"),
        ("ohlc_dir absoluto", {**ok, "ohlc_dir": "/datos/ohlc"}, "ohlc_dir"),
        ("ohlc_dir con ..", {**ok, "ohlc_dir": "../ohlc"}, "ohlc_dir"),
        ("clave desconocida", {**ok, "source": {"adapter": "costs_yaml"}}, "no reconocidas"),
    ]
    for etiqueta, body, needle in malos:
        r = _send(api, "POST", "/api/symbols/brokers", body)
        assert r.status_code == 400 and any(needle in e for e in _errors(r)), (etiqueta, r.text)
    assert list(symbols.load_registry().brokers) == ["ftmo_cfd", "ftmo_futures", "axi"]       # ninguno se creó
    for clock in ("ny_plus_7", "ny_wall"):
        r = _send(api, "POST", "/api/symbols/brokers", {**ok, "id": f"b_{clock}", "clock": clock})
        assert r.status_code == 200 and r.json()["clock"] == clock


def test_ac1_4_validate_normaliza_y_distingue_alta_de_edicion(api):
    _mk_brokers(api)
    v = _send(api, "POST", "/api/symbols/validate", _sheet()).json()
    assert v["ok"] is True and v["errors"] == [] and v["sheet"]["point_value_usd"] == 100.0
    assert v["sheet"]["sources"]["min_lot"]["from"] == "manual" and v["sheet"]["official"] is False
    _create(api)
    dup = _send(api, "POST", "/api/symbols/validate", _sheet()).json()           # alta de una que ya existe
    assert dup["ok"] is False and "ya tiene ficha" in dup["errors"][0]
    edit = _send(api, "POST", "/api/symbols/validate", {"broker": "mi_cfd", "sqx_symbol": "XAUUSD_mi",
                                                        "editing": True, "min_lot": 0.02}).json()
    assert edit["ok"] is True and edit["sheet"]["min_lot"] == 0.02 and edit["sheet"]["max_lot"] == 50.0
    bad = _send(api, "POST", "/api/symbols/validate", {"broker": "mi_cfd", "sqx_symbol": "XAUUSD_mi",
                                                       "editing": True, "min_lot": 0.015}).json()
    assert bad["ok"] is False and "múltiplo" in bad["errors"][0]
    nope = _send(api, "POST", "/api/symbols/validate", {"broker": "mi_cfd", "sqx_symbol": "NO_esta_a",
                                                        "editing": True, "min_lot": 0.02}).json()
    assert nope["ok"] is False and "no existe" in nope["errors"][0]
    off = _send(api, "POST", "/api/symbols/validate", {"broker": "ftmo_cfd", "sqx_symbol": "EURUSD_ftmo",
                                                       "editing": True, "min_lot": 0.02}).json()
    assert off["ok"] is False and "ya tiene ficha" in off["errors"][0]
    assert _send(api, "POST", "/api/symbols/validate", _sheet(broker="no_existe")).status_code == 404


# ---------------------------------------------------------------------------
# AC1.5 / AC1.5bis (+ E6, E11): hoja del script de SQX para MT5
# ---------------------------------------------------------------------------
def test_ac1_5_hoja_de_mt5_con_el_fixture(user_file):
    drafts = symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", FTMO_MAPPING, today=TODAY)
    assert [d["ticker"] for d in drafts] == ["US100.cash", "XAUUSD", "EURUSD"]
    xau, us100, eur = _draft(drafts, "XAUUSD"), _draft(drafts, "US100.cash"), _draft(drafts, "EURUSD")
    assert (xau["point_value_usd"], xau["min_lot"], xau["max_lot"], xau["lot_step"]) == (100.0, 0.01, 100.0, 0.01)
    assert xau["spread_px"] == pytest.approx(41.0 * 0.01, rel=1e-12)               # 41 pips x 0,01
    assert (us100["point_value_usd"], us100["min_lot"], us100["max_lot"], us100["lot_step"]) == \
        (1.0, 0.01, 1000.0, 0.01)
    assert us100["spread_px"] == pytest.approx(155.0 * 0.01, rel=1e-12)
    assert (eur["point_value_usd"], eur["min_lot"], eur["max_lot"]) == (100000.0, 0.01, 50.0)
    assert eur["spread_px"] == pytest.approx(0.1 * 0.0001, rel=1e-12)
    assert [d["tick_size"] for d in (us100, xau, eur)] == [0.01, 0.01, 0.0001]        # `Pip (Tick Size)`
    # pv, mínimo y paso coinciden con la ficha OFICIAL ftmo_cfd de su símbolo
    for d in drafts:
        assert d["sqx_symbol"] == FTMO_MAPPING[d["ticker"]] and d["sqx_symbol_suggested"] is False
        off = symbols.resolve(d["sqx_symbol"])
        assert off.broker == "ftmo_cfd" and off.official
        assert (d["point_value_usd"], d["min_lot"], d["lot_step"]) == (off.point_value_usd, off.min_lot, off.lot_step)
        assert d["broker"] == "ftmo_cfd" and d["asset_class"] == "cfd" and d["currency"] == "USD"
        # la comisión sale 0 si no se teclea en el script: queda VACÍA y en needs_review
        assert d["commission"] == {} and d["needs_review"] == ["commission"]
        assert d["faltan"] == ["commission"]            # guardable/motivo: ver H2 más abajo (ya tienen ficha oficial)
        assert d["ohlc_m1"] is None and d["verified_on"] is None and d["notes"] is None
        assert set(d["sources"]) == {"ticker", "point_value_usd", "lot_step", "min_lot", "max_lot", "spread_px",
                                     "tick_size"}
        assert all(s == {"from": "mt5_sheet", "date": TODAY} for s in d["sources"].values())
        # ya tienen ficha OFICIAL: no es editable y no hay conflictos en pv/paso/mínimo
        assert d["existing"] == {"broker": "ftmo_cfd", "sqx_symbol": d["sqx_symbol"], "official": True}
        assert d["editable"] is False
        assert not {c["field"] for c in d["conflicts"]} & {"point_value_usd", "lot_step", "min_lot"}
        # H2 (revisión de código de la F1): con una ficha OFICIAL el borrador no es guardable y su motivo es el E7
        assert d["guardable"] is False and d["motivo"].startswith(
            f"el símbolo SQX '{d['sqx_symbol']}' ya tiene ficha en el bróker 'ftmo_cfd'")
        assert "Las fichas oficiales no se editan" in d["motivo"]
    json.dumps(drafts, allow_nan=False)
    assert not user_file.exists()                       # parse_mt5_sheet es puro: no guarda nada


def test_ac1_5_el_borrador_no_guarda_nada_por_la_api(api, user_file):
    text = _sheet_text()
    r = _send(api, "POST", "/api/symbols/mt5-sheet", {"broker": "ftmo_cfd", "text": text, "mapping": FTMO_MAPPING})
    assert r.status_code == 200
    body = r.json()
    assert body["broker"] == "ftmo_cfd" and [d["ticker"] for d in body["drafts"]] == ["US100.cash", "XAUUSD", "EURUSD"]
    assert not user_file.exists()
    assert len(symbols.load_registry().sheets) == 16
    assert api.get("/api/symbols").json()["errors"] == []


def test_ac1_5_el_borrador_publica_conflictos_con_los_datos_oficiales_reales():
    """Con el export real del 12-oct: el spread de la ficha de oro (45 puntos del YAML, 0,45) difiere del de la hoja
    (41 pips, 0,41) y el tick de US100 (1,0 del instrumento SQX) del pip de MT5 (0,01)."""
    drafts = symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", FTMO_MAPPING, today=TODAY)
    by = {d["ticker"]: {c["field"]: c for c in d["conflicts"]} for d in drafts}
    assert set(by["XAUUSD"]) == {"spread_px"} and set(by["EURUSD"]) == {"spread_px"}
    assert set(by["US100.cash"]) == {"tick_size"}
    c = by["XAUUSD"]["spread_px"]
    assert c["actual"] == symbols.resolve("XAUUSD_ftmo(2)").spread_px == 0.45
    assert c["propuesto"] == pytest.approx(0.41) and (c["fuente_actual"], c["fuente_propuesta"]) == \
        ("official", "mt5_sheet")


def test_ac1_5_sin_mapping_sugiere_por_raiz_entre_las_fichas_del_broker_destino():
    drafts = symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", None, today=TODAY)
    xau, eur, us100 = _draft(drafts, "XAUUSD"), _draft(drafts, "EURUSD"), _draft(drafts, "US100.cash")
    assert (xau["sqx_symbol"], xau["sqx_symbol_suggested"], xau["candidates"]) == \
        ("XAUUSD_ftmo(2)", True, ["XAUUSD_ftmo(2)"])
    assert (eur["sqx_symbol"], eur["sqx_symbol_suggested"]) == ("EURUSD_ftmo", True)
    # `US100.cash` -> raíz US100: ningún símbolo SQX de FTMO CFD la comparte (USATECHIDXUSD_...): sin sugerencia
    assert us100["sqx_symbol"] is None and us100["candidates"] == [] and us100["guardable"] is False
    assert us100["motivo"] == "falta el símbolo SQX" and us100["faltan"][0] == "sqx_symbol"
    assert us100["existing"] is None and us100["editable"] is True and us100["conflicts"] == []
    # el bróker destino manda: en AXI la raíz XAUUSD se sugiere como XAUUSD_axi
    axi = {d["ticker"]: d for d in symbols.parse_mt5_sheet(_sheet_text(), "axi", None, today=TODAY)}
    assert axi["XAUUSD"]["sqx_symbol"] == "XAUUSD_axi" and axi["EURUSD"]["sqx_symbol"] == "EURUSD_axi"
    assert axi["US100.cash"]["sqx_symbol"] is None
    # la raíz: prefijo alfanumérico antes de `_`, `.` o `(`
    assert [symbols._root(s) for s in ("US100.cash", "XAUUSD_ftmo(2)", "EURUSD", "@micro_oro", "")] == \
        ["US100", "XAUUSD", "EURUSD", "", ""]


def test_ac1_5_un_mapping_a_un_simbolo_de_otro_broker_avisa_con_el_mensaje_e7():
    d = _draft(symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", {"XAUUSD": "XAUUSD_axi"}, today=TODAY), "XAUUSD")
    assert d["existing"]["broker"] == "axi" and any("ya tiene ficha en el bróker 'axi'" in w for w in d["warnings"])
    conflicts = {c["field"] for c in d["conflicts"]}
    assert "max_lot" in conflicts and "spread_px" in conflicts                    # 100 vs 20 y 0,41 vs 0,08


def test_ac1_5bis_e11_lote_minimo_divergente_en_la_hoja_es_un_conflicto_con_la_oficial(api, user_file):
    text = _edit_sheet(_sheet_text(), "XAUUSD", **{"Min Lot Size": "0.1"})
    r = _send(api, "POST", "/api/symbols/mt5-sheet", {"broker": "ftmo_cfd", "text": text, "mapping": FTMO_MAPPING})
    assert r.status_code == 200
    xau = _draft(r.json()["drafts"], "XAUUSD")
    mn = next(c for c in xau["conflicts"] if c["field"] == "min_lot")
    assert (mn["actual"], mn["propuesto"], mn["fuente_actual"], mn["fuente_propuesta"]) == \
        (0.01, 0.1, "official", "mt5_sheet")
    assert "la ficha oficial dice 0.01, tu hoja dice 0.1; si tu cuenta es distinta, crea un bróker propio" \
        in mn["texto"]
    assert xau["editable"] is False and xau["min_lot"] == 0.1
    for other in ("US100.cash", "EURUSD"):                      # las demás filas siguen sin conflicto de mínimo
        assert "min_lot" not in {c["field"] for c in _draft(r.json()["drafts"], other)["conflicts"]}
    assert not user_file.exists()                                # nada guardado
    assert symbols.resolve("XAUUSD_ftmo(2)").min_lot == 0.01 and len(symbols.load_registry().sheets) == 16


def test_e11_conflicto_con_ficha_de_usuario_es_editable_y_se_acepta_campo_a_campo(api, user_file):
    _mk_brokers(api)
    _create(api, min_lot=0.1, lot_step=0.1, spread_px=0.5, tick_size=0.01, max_lot=100)
    drafts = symbols.parse_mt5_sheet(_sheet_text(), "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY)
    xau = _draft(drafts, "XAUUSD")
    assert xau["editable"] is True and xau["existing"]["official"] is False
    by = {c["field"]: c for c in xau["conflicts"]}
    assert set(by) == {"lot_step", "min_lot", "spread_px"}
    assert by["min_lot"]["fuente_actual"] == "manual" and by["min_lot"]["actual"] == 0.1
    assert "tu ficha dice" in by["min_lot"]["texto"] and "campo a campo" in by["min_lot"]["texto"]
    # aceptar solo min_lot y lot_step de la hoja (PUT con los campos elegidos); el spread de la ficha se queda
    chosen = {"min_lot": xau["min_lot"], "lot_step": xau["lot_step"],
              "sources": {"min_lot": xau["sources"]["min_lot"], "lot_step": xau["sources"]["lot_step"]}}
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", chosen)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert (saved["min_lot"], saved["lot_step"], saved["spread_px"]) == (0.01, 0.01, 0.5)
    assert saved["sources"]["min_lot"] == {"from": "mt5_sheet", "date": TODAY}
    assert saved["sources"]["spread_px"]["from"] == "manual"
    again = _draft(symbols.parse_mt5_sheet(_sheet_text(), "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert {c["field"] for c in again["conflicts"]} == {"spread_px"}


def test_e11_conflicto_de_comision_entre_una_ficha_de_usuario_y_la_hoja(api):
    _mk_brokers(api)
    _create(api, commission={"per_lot_rt": 4.5})
    text = _edit_sheet(_sheet_text(), "XAUUSD", **{"Commission Value": "5.0"})
    xau = _draft(symbols.parse_mt5_sheet(text, "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert xau["commission"] == {"per_lot_rt": 5.0} and xau["needs_review"] == []
    c = next(c for c in xau["conflicts"] if c["field"] == "commission")
    assert (c["actual"], c["propuesto"]) == ({"per_lot_rt": 4.5}, {"per_lot_rt": 5.0})
    same = _edit_sheet(_sheet_text(), "XAUUSD", **{"Commission Value": "4.5"})
    xau = _draft(symbols.parse_mt5_sheet(same, "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert "commission" not in {c["field"] for c in xau["conflicts"]}
    pct = _edit_sheet(_sheet_text(), "XAUUSD", **{"Commission Type": "PercentageBased", "Commission Value": "0.0014"})
    xau = _draft(symbols.parse_mt5_sheet(pct, "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert xau["commission"] == {"pct_notional_rt": 0.0014}
    assert next(c for c in xau["conflicts"] if c["field"] == "commission")["propuesto"] == {"pct_notional_rt": 0.0014}


def test_e11_la_tolerancia_de_los_conflictos_es_relativa_1e_12(api):
    _mk_brokers(api)
    _create(api, point_value_usd=100 * (1 + 5e-13))
    xau = _draft(symbols.parse_mt5_sheet(_sheet_text(), "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert "point_value_usd" not in {c["field"] for c in xau["conflicts"]}                 # dentro de 1e-12
    r = _send(api, "PUT", "/api/symbols/mi_cfd/XAUUSD_mi", {"point_value_usd": 100 * (1 + 1e-9)})
    assert r.status_code == 200
    xau = _draft(symbols.parse_mt5_sheet(_sheet_text(), "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert "point_value_usd" in {c["field"] for c in xau["conflicts"]}


def test_e6_moneda_vacia_tras_1_pip_worth_no_es_guardable_ni_se_asume_usd():
    text = _edit_sheet(_sheet_text(), "XAUUSD", __after_pip_worth="")
    drafts = symbols.parse_mt5_sheet(text, "ftmo_cfd", FRESH_MAPPING, today=TODAY)
    xau = _draft(drafts, "XAUUSD")
    assert "currency" in xau["needs_review"] and xau["guardable"] is False and xau["currency"] == ""
    assert xau["point_value_usd"] is None and "point_value_usd" not in xau["sources"]
    assert "no se asume USD" in xau["motivo"]
    # las otras filas no se ven afectadas
    assert all(d["guardable"] and "currency" not in d["needs_review"] for d in drafts if d is not xau)


def test_e6_moneda_distinta_de_usd_es_divisa_no_usd_y_no_guardable():
    text = _edit_sheet(_sheet_text(), "EURUSD", __after_pip_worth="EUR")
    eur = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", FRESH_MAPPING, today=TODAY), "EURUSD")
    assert eur["motivo"] == "divisa no USD (v1)" and eur["guardable"] is False
    assert eur["needs_review"] == ["commission", "currency"] and eur["point_value_usd"] is None
    assert eur["currency"] == "EUR"
    low = _edit_sheet(_sheet_text(), "EURUSD", __after_pip_worth="usd")                # mayúsculas: sigue siendo USD
    assert _draft(symbols.parse_mt5_sheet(low, "ftmo_cfd", FRESH_MAPPING, today=TODAY), "EURUSD")["guardable"]


def test_e6_sin_columna_de_moneda_sin_nombre_tras_1_pip_worth_tampoco_se_asume_usd():
    text = _sheet_text().replace("1 Pip Worth,,1 Tick Worth", "1 Pip Worth,Moneda,1 Tick Worth", 1)
    drafts = symbols.parse_mt5_sheet(text, "ftmo_cfd", FTMO_MAPPING, today=TODAY)
    assert all("currency" in d["needs_review"] and d["guardable"] is False for d in drafts)


def test_ac1_5_comisiones_de_la_hoja():
    base = _sheet_text()
    d = lambda **cols: _draft(symbols.parse_mt5_sheet(                                    # noqa: E731
        _edit_sheet(base, "XAUUSD", **cols), "ftmo_cfd", FTMO_MAPPING, today=TODAY), "XAUUSD")
    size = d(**{"Commission Value": "4.5"})
    assert size["commission"] == {"per_lot_rt": 4.5} and size["needs_review"] == [] and size["faltan"] == []
    assert size["sources"]["commission"] == {"from": "mt5_sheet", "date": TODAY}
    pct = d(**{"Commission Type": "PercentageBased", "Commission Value": "0.0014"})
    assert pct["commission"] == {"pct_notional_rt": 0.0014} and pct["needs_review"] == []
    raro = d(**{"Commission Type": "PerTrade", "Commission Value": "2.0"})
    assert raro["commission"] == {} and raro["needs_review"] == ["commission"]
    assert any("PerTrade" in w for w in raro["warnings"])
    sin_valor = d(**{"Commission Value": ""})
    assert sin_valor["commission"] == {} and sin_valor["needs_review"] == ["commission"]


def test_ac1_5_formatos_de_entrada_de_la_hoja():
    base = symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", FTMO_MAPPING, today=TODAY)
    text = _sheet_text()
    variantes = {"bom": "\ufeff" + text, "crlf": text.replace("\n", "\r\n"), "punto y coma": text.replace(",", ";")}
    for nombre, t in variantes.items():
        assert symbols.parse_mt5_sheet(t, "ftmo_cfd", FTMO_MAPPING, today=TODAY) == base, nombre
    # filas en blanco y filas cortas no rompen; una cifra ilegible deja el campo vacío
    rows = list(csv.reader(io.StringIO(text)))
    rows.insert(2, [])
    rows.insert(3, ["", "sin instrumento"])
    xau = next(r for r in rows if r and r[0] == "XAUUSD")
    xau[rows[0].index("Spread (In Pips)")] = "n/a"
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    drafts = symbols.parse_mt5_sheet(out.getvalue(), "ftmo_cfd", FTMO_MAPPING, today=TODAY)
    assert [d["ticker"] for d in drafts] == ["US100.cash", "XAUUSD", "EURUSD"]
    assert _draft(drafts, "XAUUSD")["spread_px"] is None and "spread_px" not in _draft(drafts, "XAUUSD")["sources"]
    corta = ",".join(text.splitlines()[1].split(",")[:14])                       # termina antes de `Min Lot Size`
    d = symbols.parse_mt5_sheet(text.splitlines()[0] + "\n" + corta + "\n", "ftmo_cfd", None, today=TODAY)[0]
    assert d["min_lot"] is None and d["point_value_usd"] is None and d["guardable"] is False
    assert "faltan datos obligatorios" in (d["motivo"] or "falta el símbolo SQX") or d["sqx_symbol"] is None


def test_ac1_5_hoja_con_errores_de_formato_y_destinos_invalidos(api):
    text = _sheet_text()
    with pytest.raises(ValueError, match="faltan columnas.*Point Value"):
        symbols.parse_mt5_sheet(text.replace("Point Value", "Valor"), "ftmo_cfd")
    with pytest.raises(ValueError, match="vacía"):
        symbols.parse_mt5_sheet("  \n\n", "ftmo_cfd")
    with pytest.raises(ValueError, match="bróker desconocido"):
        symbols.parse_mt5_sheet(text, "no_existe")
    with pytest.raises(ValueError, match="solo vale para brókers CFD"):
        symbols.parse_mt5_sheet(text, "ftmo_futures")
    with pytest.raises(ValueError, match="mapping"):
        symbols.parse_mt5_sheet(text, "ftmo_cfd", {"XAUUSD": "con espacio"})
    url = "/api/symbols/mt5-sheet"
    assert _send(api, "POST", url, {"broker": "no_existe", "text": text}).status_code == 404
    r = _send(api, "POST", url, {"broker": "ftmo_futures", "text": text})
    assert r.status_code == 400 and "solo vale para brókers CFD" in _errors(r)[0]
    r = _send(api, "POST", url, {"broker": "ftmo_cfd", "text": text.replace("Point Value", "Valor")})
    assert r.status_code == 400 and "faltan columnas" in _errors(r)[0]
    assert _send(api, "POST", url, {"broker": "ftmo_cfd"}).status_code == 422
    assert _send(api, "POST", url, {"broker": "ftmo_cfd", "text": text, "mapping": [1]}).status_code == 422
    assert _send(api, "POST", url, {"broker": "ftmo_cfd", "text": text, "mapping": {"XAUUSD": "a b"}}).status_code == 400


def test_ac1_5_del_borrador_a_la_ficha_guardada(api, user_file):
    """Flujo de la vista de importación: borrador -> el usuario teclea la comisión -> POST por ficha."""
    _mk_brokers(api)
    d = _draft(symbols.parse_mt5_sheet(_sheet_text(), "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert d["guardable"] and d["existing"] is None and d["editable"] is True and d["conflicts"] == []
    sin_comision = _send(api, "POST", "/api/symbols", d)             # el borrador tal cual NO se guarda: falta la comisión
    assert sin_comision.status_code == 400 and "commission" in _errors(sin_comision)[0]
    d["commission"], d["needs_review"] = {"per_lot_rt": 4.5}, []
    d["sources"]["commission"] = {"from": "manual", "date": TODAY}
    r = _send(api, "POST", "/api/symbols", d)                         # los campos de solo lectura del borrador se ignoran
    assert r.status_code == 200, r.text
    s = r.json()
    assert (s["point_value_usd"], s["min_lot"], s["max_lot"], s["lot_step"], s["tick_size"]) == \
        (100.0, 0.01, 100.0, 0.01, 0.01)
    assert s["spread_px"] == pytest.approx(0.41) and s["commission"] == {"per_lot_rt": 4.5}
    assert s["sources"]["point_value_usd"] == {"from": "mt5_sheet", "date": TODAY}
    assert s["sources"]["commission"]["from"] == "manual" and s["needs_review"] == []


# ---------------------------------------------------------------------------
# AC1.6: bróker de cada perfil
# ---------------------------------------------------------------------------
OFICIALES = {"ftmo_50k": "ftmo_cfd", "ftmo_50k_swing": "ftmo_cfd", "ftmo_100k": "ftmo_cfd",
             "ftmo_futures_growth_50k": "ftmo_futures", "ftmo_futures_growth_100k": "ftmo_futures",
             "ftmo_futures_growth_150k": "ftmo_futures", "ftmo_futures_pro_50k": "ftmo_futures",
             "ftmo_futures_pro_100k": "ftmo_futures", "ftmo_futures_pro_150k": "ftmo_futures",
             "apex_150k_intraday": None}


def test_ac1_6_los_diez_perfiles_oficiales_tienen_broker_o_null_explicito():
    official_keys = list(profiles.official()[0])
    assert len(official_keys) == 10
    reg = symbols.load_registry()
    assert set(reg.profiles) == set(official_keys) == set(OFICIALES)             # los 10 oficiales, ni más ni menos
    assert reg.profiles == OFICIALES and "apex_150k_intraday" in reg.profiles    # Apex: null EXPLÍCITO
    for key in official_keys:
        assert symbols.broker_for_profile(key) == OFICIALES[key]
    assert symbols.lens_broker("axi") == "axi" and symbols.lens_broker("propfirm") is None
    assert reg.lenses == {"axi": "axi"}


def test_ac1_6_un_perfil_de_usuario_hereda_el_broker_de_su_raiz(api, profiles_file):
    for base, key, esperado in (("ftmo_50k", "mi_perfil", "ftmo_cfd"),
                                ("ftmo_futures_pro_50k", "mi_futuros", "ftmo_futures"),
                                ("apex_150k_intraday", "mi_apex", None)):
        r = _send(api, "POST", "/api/profiles", {"base": base, "key": key, "display_name": key})
        assert r.status_code == 200, r.text
        assert profiles.based_on(key) == base
        assert symbols.broker_for_profile(key) == esperado, key
    assert symbols.broker_for_profile("perfil_inexistente") is None             # raíz sin entrada


def test_ac1_6_una_raiz_sin_entrada_en_brokers_yaml_da_none(tmp_path, monkeypatch):
    costs = _write_costs(tmp_path / "c.yaml", "S_x")
    monkeypatch.setattr(symbols, "BROKERS_PATH", _write_brokers(
        tmp_path / "brokers.yaml", {"b_uno": _broker_def(costs)}, profiles={"ftmo_50k": "b_uno"}))
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "u.yaml")
    assert symbols.broker_for_profile("ftmo_50k") == "b_uno"
    assert symbols.broker_for_profile("ftmo_100k") is None


# ---------------------------------------------------------------------------
# AC1.7: I3 (ficheros que no cambian) e I4 (ficheros reales de usuario)
# ---------------------------------------------------------------------------
I3_SHA256 = {
    "propfirms.yaml": "7fadcfe067b32f5e412aa742f81325e39d41b401f1b585a77a2ff0b4cba30461",
    "broker_axi.yaml": "65a19842e3d15f9b86881546ba8ef7080d8137ca0e41f355d6b04246dd72c0bd",
    "ftmo_cfd_costs.yaml": "cd7e36d2ad66c9fefecd93962abd45955d78e7b465649eeb35a3e660131f7299",
    "ftmo_futures_costs.yaml": "5e54e346ef4a018ffecde79125dbe14d30fbed11fae25e74711fb6d8891cbe61",
    "axi_select.yaml": "759290df4fceec595b49d8f19bfa542d031244956ca4171bc64c90924af68aad",
}


@pytest.mark.parametrize("name", sorted(I3_SHA256))
def test_ac1_7_i3_los_ficheros_de_especificaciones_no_cambian(name):
    assert _sha(ROOT / "config" / name) == I3_SHA256[name]


def test_ac1_7_i4_los_tests_no_crean_ni_tocan_los_ficheros_reales_de_usuario(api, user_file, profiles_file):
    """Un recorrido por todas las escrituras con rutas en tmp; los ficheros REALES siguen como estaban (el
    fixture autouse del módulo lo vuelve a comprobar al terminar)."""
    antes = {p: (p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in REAL_USER_FILES}
    _mk_brokers(api)
    _create(api)
    assert _hide(api, ["axi/XAUUSD_axi"]).status_code == 200
    assert _send(api, "POST", "/api/profiles", {"base": "ftmo_50k", "key": "mi_perfil"}).status_code == 200
    assert user_file.exists() and profiles_file.exists()
    despues = {p: (p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in REAL_USER_FILES}
    assert despues == antes
    assert symbols.USER_SYMBOLS_PATH == user_file and symbols.USER_SYMBOLS_PATH != REAL_USER_FILES[0]
    assert profiles.USER_PROFILES_PATH == profiles_file and profiles.USER_PROFILES_PATH != REAL_USER_FILES[1]


# ---------------------------------------------------------------------------
# AC1.8: sin cambio de comportamiento (la carga y el catálogo no usan fichas todavía)
# ---------------------------------------------------------------------------
def test_ac1_8_catalog_legs_csv_igual_y_sqx_con_estado_de_ficha(api, tmp_path, monkeypatch):
    """``/api/catalog``: las patas CSV conservan sus claves y valores de siempre; las ``.sqx`` de símbolos conocidos
    conservan las suyas y la fase F2 les añade el estado de su ficha (``broker``, ``ficha``, ``motivo``; invariante
    I7). Una escritura de fichas ajenas a esas patas no cambia el catálogo.

    PLAN-DESVIO: en la F1 este test se llamaba ``test_ac1_8_catalog_legs_sin_cambios_sobre_sqx_sinteticos`` y fijaba
    el catálogo SIN esas tres claves (la carga aún no usaba fichas); el brief de la F2 cambia el catálogo a propósito."""
    root = tmp_path / "strategies"
    for name, sym, inst, pv, tick in (("Strategy 1.1", "EURUSD_ftmo", "EURUSD_cfd_ftmo", 100000.0, 0.0001),
                                      ("Strategy 1.2", "@micro_nasdaq", "MNQ_ftmo", 2.0, 0.25)):
        write_sqx(root / "bank" / f"{name}.sqx", [_W.long_win()], symbol=sym, instrument=inst, pv=pv, tick=tick)
    trades = tmp_path / "trades"
    (trades / "dir").mkdir(parents=True)
    (trades / "dir" / "a.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", root)
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [])
    monkeypatch.setattr(main, "TRADES_DIR", trades)
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})

    def legs():
        return api.get("/api/catalog").json()["legs"]

    antes = legs()
    csv_leg, cfd_leg, fut_leg = antes
    assert csv_leg == {"path": str(trades / "dir" / "a.csv"), "folder": "dir", "name": "a", "kind": "csv"}
    assert cfd_leg == {"path": str(root / "bank" / "Strategy 1.1.sqx"), "folder": "bank", "name": "Strategy 1.1",
                       "kind": "sqx_futures", "asset": "cfd", "broker": "ftmo_cfd", "ficha": "ok", "motivo": None}
    assert fut_leg["kind"] == "sqx_futures" and fut_leg["asset"] == "futures" and fut_leg["name"] == "Strategy 1.2"
    assert (fut_leg["broker"], fut_leg["ficha"], fut_leg["motivo"]) == ("ftmo_futures", "ok", None)
    _mk_brokers(api)
    _create(api)
    assert legs() == antes


def test_ac1_8_las_fichas_no_cambian_la_carga_ni_la_simulacion(api):
    """La carga sigue leyendo los YAML de costes: dar de alta o cambiar una ficha no toca las constantes ni las
    funciones de ``main`` que la usan (la fase F2 las conectará)."""
    antes = (main.SQX_OHLC_DIR, main.SQX_MINUTE_COHERENCE_MIN, main.KIND_SQX, main._sqx_asset("EURUSD_ftmo"),
             main._sqx_cost_spec("EURUSD_ftmo", DEFAULT_CFD_COSTS_PATH))
    _mk_brokers(api)
    _create(api)
    _create(api, broker="ftmo_cfd", sqx_symbol="GBPJPY_ftmo", ohlc_m1=_DEL)
    assert main._sqx_asset("GBPJPY_ftmo") == (None, None)                    # main no ve las fichas de usuario
    assert (main.SQX_OHLC_DIR, main.SQX_MINUTE_COHERENCE_MIN, main.KIND_SQX, main._sqx_asset("EURUSD_ftmo"),
            main._sqx_cost_spec("EURUSD_ftmo", DEFAULT_CFD_COSTS_PATH)) == antes


# ---------------------------------------------------------------------------
# E4: firma de caché
# ---------------------------------------------------------------------------
def test_e4_sheet_signature_cambia_con_cada_componente():
    reg = symbols.load_registry()
    sheet, broker = reg.sheets["EURUSD_ftmo"], reg.brokers["ftmo_cfd"]
    args = (sheet, broker, "/datos/ohlc/EURUSD_ftmo.csv", 1700000000.0, "point")
    base = symbols.sheet_signature(*args)
    assert len(base) == 64 and set(base) <= set("0123456789abcdef")
    assert symbols.sheet_signature(*args) == base                                         # determinista
    assert symbols.sheet_signature(sheet, broker, Path("/datos/ohlc/EURUSD_ftmo.csv"), 1700000000.0, "point") == base
    rep = dataclasses.replace
    variantes = {
        "point_value_usd": (rep(sheet, point_value_usd=sheet.point_value_usd * 2), broker, *args[2:]),
        "lot_step": (rep(sheet, lot_step=0.1), broker, *args[2:]),
        "min_lot": (rep(sheet, min_lot=0.1), broker, *args[2:]),
        "max_lot": (rep(sheet, max_lot=100.0), broker, *args[2:]),
        "commission_valor": (rep(sheet, commission={"per_lot_rt": 9.0}), broker, *args[2:]),
        "commission_tipo": (rep(sheet, commission={"pct_notional_rt": 5.0}), broker, *args[2:]),
        "spread_px": (rep(sheet, spread_px=sheet.spread_px * 2), broker, *args[2:]),
        "spread_px_none": (rep(sheet, spread_px=None), broker, *args[2:]),
        "ohlc_m1": (rep(sheet, ohlc_m1="OTRO.csv"), broker, *args[2:]),
        "tick_size": (rep(sheet, tick_size=0.00001), broker, *args[2:]),
        "clock": (sheet, rep(broker, clock="ny_wall"), *args[2:]),
        "ohlc_dir": (sheet, rep(broker, ohlc_dir="otra/carpeta"), *args[2:]),
        "broker_id": (sheet, rep(broker, id="otro_id"), *args[2:]),                     # H7
        "broker_asset_class": (sheet, rep(broker, asset_class="futures"), *args[2:]),   # H7
        "ruta_del_ohlc": (sheet, broker, "/datos/ohlc/OTRO.csv", *args[3:]),
        "sin_ruta_del_ohlc": (sheet, broker, None, *args[3:]),
        "mtime_del_ohlc": (sheet, broker, args[2], 1700000001.0, args[4]),
        "sin_mtime_del_ohlc": (sheet, broker, args[2], None, args[4]),
        "synth_mode": (sheet, broker, *args[2:4], "daily"),
    }
    firmas = {k: symbols.sheet_signature(*v) for k, v in variantes.items()}
    assert all(s != base for s in firmas.values()), [k for k, s in firmas.items() if s == base]
    assert len(set(firmas.values())) == len(firmas)                      # y entre sí
    # lo que NO es componente de la firma no la cambia
    for kw in (dict(notes="x"), dict(verified_on="2026-01-01"), dict(sources={}), dict(needs_review=("min_lot",)),
               dict(ticker="OTRO"), dict(official=False), dict(sqx_symbol="otro_a"), dict(usable=False, motivo="m")):
        assert symbols.sheet_signature(rep(sheet, **kw), broker, *args[2:]) == base, kw


# ---------------------------------------------------------------------------
# E9: los ficheros fuente de brokers.yaml viajan en el export
# ---------------------------------------------------------------------------
def test_e9_cada_source_file_de_brokers_yaml_esta_en_config_files_del_manifest():
    if not MANIFEST.is_file():
        pytest.skip("manifest del export ausente (export compartible)")
    config_files = set(yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["config_files"])
    doc = yaml.safe_load((ROOT / "config" / "brokers.yaml").read_text(encoding="utf-8"))
    files = {b["source"]["file"] for b in doc["brokers"].values()}
    assert files == {"config/ftmo_cfd_costs.yaml", "config/ftmo_futures_costs.yaml", "config/broker_axi.yaml"}
    assert files <= config_files, sorted(files - config_files)
    assert "config/brokers.yaml" in config_files


# ---------------------------------------------------------------------------
# Registro: lectura robusta de brokers.yaml, adaptadores y caché por mtime
# ---------------------------------------------------------------------------
def _patch_registry(monkeypatch, tmp_path, brokers_path):
    monkeypatch.setattr(symbols, "BROKERS_PATH", brokers_path)
    monkeypatch.setattr(symbols, "USER_SYMBOLS_PATH", tmp_path / "u.yaml")


def test_registro_brokers_yaml_ausente_o_ilegible_no_rompe(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, tmp_path, tmp_path / "no_existe.yaml")
    reg = symbols.load_registry()
    assert reg.brokers == {} and reg.sheets == {} and len(reg.errors) == 1 and "ilegible" in reg.errors[0]
    assert symbols.resolve("EURUSD_ftmo") is None and symbols.broker_for_profile("ftmo_50k") is None
    bad = tmp_path / "brokers.yaml"
    bad.write_text("brokers: [sin cerrar", encoding="utf-8")
    _patch_registry(monkeypatch, tmp_path, bad)
    assert "ilegible" in symbols.load_registry().errors[0]
    bad.write_text("- una\n- lista\n", encoding="utf-8")
    symbols._reload()
    assert "diccionario" in symbols.load_registry().errors[0]
    bad.write_text("schema_version: 2\nbrokers: {}\n", encoding="utf-8")
    symbols._reload()
    assert "schema_version" in symbols.load_registry().errors[0]
    bad.write_text("schema_version: 1\nbrokers: {}\n", encoding="utf-8")
    symbols._reload()
    assert "al menos un bróker" in symbols.load_registry().errors[0]


def test_registro_broker_o_adaptador_invalido_se_lista_y_el_resto_carga(tmp_path, monkeypatch):
    costs = _write_costs(tmp_path / "c.yaml", "OK_x")
    brokers = {
        "b_bueno": _broker_def(costs),
        "b_clase": _broker_def(costs, asset_class="futures", clock="ny_wall"),          # el YAML es de CFD
        "b_adaptador": _broker_def(costs, source={"adapter": "otro", "file": str(costs)}),
        "b_sin_source": {k: v for k, v in _broker_def(costs).items() if k != "source"},
        "b_sin_fichero": _broker_def(costs, source={"adapter": "costs_yaml", "file": str(tmp_path / "no_hay.yaml")}),
        "B_mal_id": _broker_def(costs),
        "b_clock": _broker_def(costs, clock="utc"),
    }
    _patch_registry(monkeypatch, tmp_path, _write_brokers(tmp_path / "brokers.yaml", brokers))
    reg = symbols.load_registry()
    assert list(reg.sheets) == ["OK_x"]
    assert set(reg.brokers) == {"b_bueno", "b_clase", "b_sin_fichero"}
    joined = "\n".join(reg.errors)
    for needle in ("b_clase: ", "b_adaptador", "b_sin_source", "b_sin_fichero: no se pudo leer", "B_mal_id", "b_clock"):
        assert needle in joined, needle


def test_registro_entrada_de_costes_incompleta_se_omite_con_error(tmp_path, monkeypatch):
    costs = tmp_path / "c.yaml"
    costs.write_text(_dump({"asset_class": "cfd", "symbols": {
        "BIEN": {"sqx_symbol": "BIEN_x", "tick_size": 0.01, "tick_value_usd": 1.0, "total_round_trip": 1.0,
                 "lot_step": 0.01, "min_lot": 0.01},
        "SIN_RT": {"sqx_symbol": "SIN_RT_x", "tick_size": 0.01, "tick_value_usd": 1.0, "lot_step": 0.01,
                   "min_lot": 0.01},
        "SIN_SQX": {"tick_size": 0.01, "tick_value_usd": 1.0, "total_round_trip": 1.0, "lot_step": 0.01,
                    "min_lot": 0.01},
        "TICK_0": {"sqx_symbol": "TICK_0_x", "tick_size": 0, "tick_value_usd": 1.0, "total_round_trip": 1.0,
                   "lot_step": 0.01, "min_lot": 0.01},
        "SIN_PASO": {"sqx_symbol": "SIN_PASO_x", "tick_size": 0.01, "tick_value_usd": 1.0, "total_round_trip": 1.0}},
        "sqx_policy": {}}), encoding="utf-8")
    _patch_registry(monkeypatch, tmp_path, _write_brokers(tmp_path / "brokers.yaml", {"b_uno": _broker_def(costs)}))
    reg = symbols.load_registry()
    assert list(reg.sheets) == ["BIEN_x"] and reg.sheets["BIEN_x"].spread_px is None   # sin sqx_policy: spread None
    assert len(reg.errors) == 4
    futuros = tmp_path / "f.yaml"
    futuros.write_text(_dump({"symbols": {"M": {"sqx_symbol": "@m", "tick_size": 0.25, "tick_value_usd": 0.5,
                                                "commission_pct": 0.001}}}), encoding="utf-8")
    _patch_registry(monkeypatch, tmp_path, _write_brokers(
        tmp_path / "brokers2.yaml", {"b_fut": _broker_def(futuros, asset_class="futures", clock="ny_wall")}))
    reg = symbols.load_registry()
    assert reg.sheets == {} and "commission_pct solo vale en CFD" in reg.errors[0]


def test_registro_perfiles_y_lentes_validan_el_broker_de_destino(tmp_path, monkeypatch):
    costs = _write_costs(tmp_path / "c.yaml", "OK_x")
    _patch_registry(monkeypatch, tmp_path, _write_brokers(
        tmp_path / "brokers.yaml", {"b_uno": _broker_def(costs)},
        profiles={"p_ok": "b_uno", "p_null": None, "p_mal": "no_existe"}, lenses={"l_ok": "b_uno", "l_null": None}))
    reg = symbols.load_registry()
    assert reg.profiles == {"p_ok": "b_uno", "p_null": None} and reg.lenses == {"l_ok": "b_uno"}
    assert len(reg.errors) == 2 and "p_mal" in "".join(reg.errors) and "l_null" in "".join(reg.errors)
    _patch_registry(monkeypatch, tmp_path, _write_brokers(
        tmp_path / "brokers_b.yaml", {"b_uno": _broker_def(costs)}, profiles=[1], lenses="x"))
    assert [e for e in symbols.load_registry().errors if "debe ser un diccionario" in e].__len__() == 2


def test_registro_sufijo_sqx_y_ticker_cascade_vacio_del_adaptador_axi(tmp_path, monkeypatch):
    axi = tmp_path / "axi.yaml"
    axi.write_text((ROOT / "config" / "broker_axi.yaml").read_text(encoding="utf-8").replace(
        'ticker_cascade: "NZDUSD"', "ticker_cascade: null"), encoding="utf-8")
    d = {"display_name": "A", "company": "C", "asset_class": "cfd", "clock": "ny_plus_7",
         "source": {"adapter": "broker_axi", "file": str(axi), "sqx_suffix": "_x"}}
    _patch_registry(monkeypatch, tmp_path, _write_brokers(tmp_path / "brokers.yaml", {"a_axi": d}))
    reg = symbols.load_registry()
    assert "XAUUSD_x" in reg.sheets and "XAUUSD_axi" not in reg.sheets and "NZDUSD_x" not in reg.sheets
    assert len(reg.sheets) == 6 and reg.sheets["XAUUSD_x"].ohlc_m1 == "XAUUSD_x.csv"
    assert len(reg.errors) == 1 and "NZDUSD.pa" in reg.errors[0] and "ticker_cascade" in reg.errors[0]
    axi.write_text("symbols: 5\n", encoding="utf-8")
    symbols._reload()
    assert symbols.load_registry().sheets == {} and "no se pudo leer" in symbols.load_registry().errors[0]


def test_registro_cache_por_mtime_de_brokers_yaml_adaptadores_y_fichero_de_usuario(tmp_path, monkeypatch):
    costs = _write_costs(tmp_path / "c.yaml", "OK_x")
    brokers_path = _write_brokers(tmp_path / "brokers.yaml", {"b_uno": _broker_def(costs)})
    _patch_registry(monkeypatch, tmp_path, brokers_path)
    user = tmp_path / "u.yaml"
    reg = symbols.load_registry()
    assert symbols.load_registry() is reg                                     # sin cambios: la misma instancia

    def bump(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        ns = path.stat().st_mtime_ns + 5_000_000_000
        os.utime(path, ns=(ns, ns))                                           # mtime distinto sin esperar

    bump(costs, costs.read_text(encoding="utf-8").replace("total_round_trip: 1.0", "total_round_trip: 7.0"))
    reg2 = symbols.load_registry()                                            # cambió el fichero fuente del adaptador
    assert reg2 is not reg and reg2.sheets["OK_x"].commission == {"per_lot_rt": 7.0}
    assert symbols.load_registry() is reg2
    bump(user, _dump({"brokers": {"mi_b": {"display_name": "M", "company": "C", "asset_class": "cfd",
                                           "clock": "ny_wall"}}}))
    reg3 = symbols.load_registry()                                            # apareció el fichero de usuario
    assert reg3 is not reg2 and "mi_b" in reg3.brokers
    bump(brokers_path, brokers_path.read_text(encoding="utf-8").replace("display_name: B", "display_name: Otro"))
    reg4 = symbols.load_registry()                                            # cambió brokers.yaml
    assert reg4 is not reg3 and reg4.brokers["b_uno"].display_name == "Otro"
    user.unlink()
    assert "mi_b" not in symbols.load_registry().brokers                       # el fichero de usuario desapareció


def test_registro_resolve_oculta_y_reaparece_y_no_conoce_lo_inexistente(api):
    assert symbols.resolve("@micro_nasdaq").broker == "ftmo_futures"
    assert symbols.resolve("no_existe_a") is None and symbols.resolve("") is None
    assert _hide(api, ["ftmo_futures/@micro_nasdaq"]).status_code == 200
    assert symbols.resolve("@micro_nasdaq") is None
    assert symbols.load_registry().sheets["@micro_nasdaq"].point_value_usd == 2.0        # sigue en el registro
    assert "ftmo_futures/@micro_nasdaq" in symbols.load_registry().hidden
    assert _hide(api, ["ftmo_futures/@micro_nasdaq"], False).status_code == 200
    assert symbols.resolve("@micro_nasdaq") is not None


def test_registro_las_rutas_con_simbolos_especiales_funcionan_codificadas(api):
    for broker, sym in (("ftmo_futures", "@micro_nasdaq"), ("ftmo_cfd", "XAUUSD_ftmo(2)"),
                        ("ftmo_cfd", "USATECHIDXUSD_ftmo(2)")):
        for url in (f"/api/symbols/{broker}/{sym}/export", f"/api/symbols/{broker}/{sym.replace('@', '%40')}/export"):
            r = api.get(url)
            assert r.status_code == 200 and r.json()["sqx_symbol"] == sym, url
    assert api.get("/api/symbols/ftmo_cfd/NO_esta/export").status_code == 404
    assert api.get("/api/symbols/ftmo_futures/XAUUSD_ftmo(2)/export").status_code == 404      # símbolo de otro bróker


# ---------------------------------------------------------------------------
# Revisión de código de la F1 (hallazgos H1-H7), corregidos en la F2. Un test por cada uno de H1-H4.
# ---------------------------------------------------------------------------
def test_h1_el_adaptador_axi_aisla_cada_simbolo_y_el_resto_carga(user_file, monkeypatch):
    """H1: un símbolo de ``broker_axi.yaml`` que falla al construir su ficha va a ``errors`` y no rompe el registro;
    ``ticker_cascade`` ``None`` es el error de esa entrada. Los demás símbolos cargan."""
    real = main.load_broker_config()
    malo = types.SimpleNamespace(ticker_axi="MALO.pa", ticker_cascade="MALO", quote_currency="USD")  # le faltan datos
    sin_nombre = types.SimpleNamespace(ticker_axi="SINNOMBRE.pa", ticker_cascade=None)
    fake = types.SimpleNamespace(captured_at=real.captured_at, symbols=[malo, sin_nombre, *real.symbols])
    monkeypatch.setattr(symbols, "load_broker_config", lambda path: fake)
    reg = symbols._build_registry()                   # directo: no pisa la caché del registro con el cfg falso
    axi = {s.sqx_symbol for s in reg.sheets.values() if s.broker == "axi"}
    assert axi == {s.ticker_cascade + "_axi" for s in real.symbols} and len(axi) == 7
    assert len(reg.errors) == 2
    err_malo = next(e for e in reg.errors if "axi/MALO.pa" in e)
    assert "AttributeError" in err_malo and "broker_axi.yaml" in err_malo
    assert any("axi/SINNOMBRE.pa" in e and "ticker_cascade" in e for e in reg.errors)
    assert "MALO_axi" not in reg.sheets
    # los brókers oficiales de costes no se ven afectados
    assert {"EURUSD_ftmo", "@micro_nasdaq"} <= set(reg.sheets)


def test_h2_borrador_con_ficha_oficial_o_de_otro_broker_no_es_guardable_ni_editable(api):
    _mk_brokers(api)
    _create(api)                                                       # XAUUSD_mi en mi_cfd (de usuario)
    text = _sheet_text()
    e7 = lambda sym, b: f"el símbolo SQX '{sym}' ya tiene ficha en el bróker '{b}'; si es otro bróker, dale a sus " \
                        f"datos en SQX un nombre propio (p. ej. {sym}_mibroker)"                      # noqa: E731
    # 1) ficha OFICIAL del mismo bróker destino
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", {"XAUUSD": "XAUUSD_ftmo(2)"}, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"]) == (False, False) and d["motivo"].startswith(e7("XAUUSD_ftmo(2)", "ftmo_cfd"))
    # 2) ficha OFICIAL de OTRO bróker
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", {"XAUUSD": "XAUUSD_axi"}, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"]) == (False, False) and d["motivo"].startswith(e7("XAUUSD_axi", "axi"))
    assert any(e7("XAUUSD_axi", "axi") in w for w in d["warnings"])             # el aviso de la F1 se conserva
    # 3) ficha de USUARIO en OTRO bróker destino
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"]) == (False, False) and d["motivo"].startswith(e7("XAUUSD_mi", "mi_cfd"))
    # 4) ficha de USUARIO en el MISMO bróker: sigue siendo editable y guardable (aceptar el conflicto campo a campo)
    d = _draft(symbols.parse_mt5_sheet(text, "mi_cfd", {"XAUUSD": "XAUUSD_mi"}, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"], d["motivo"]) == (True, True, None)
    # 5) sin ficha: guardable
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", FRESH_MAPPING, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"], d["motivo"], d["existing"]) == (True, True, None, None)


def test_h2_un_motivo_previo_manda_y_el_e7_pasa_a_los_avisos():
    text = _edit_sheet(_sheet_text(), "EURUSD", __after_pip_worth="EUR")          # divisa no USD
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", {"EURUSD": "EURUSD_ftmo"}, today=TODAY), "EURUSD")
    assert d["guardable"] is False and d["editable"] is False and d["motivo"] == "divisa no USD (v1)"
    assert any("ya tiene ficha en el bróker 'ftmo_cfd'" in w for w in d["warnings"])


def test_h2_las_fichas_ocultas_no_se_sugieren_en_candidates(api):
    assert _hide(api, ["ftmo_cfd/XAUUSD_ftmo(2)"]).status_code == 200
    xau = _draft(symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", None, today=TODAY), "XAUUSD")
    assert xau["candidates"] == [] and xau["sqx_symbol"] is None and xau["sqx_symbol_suggested"] is False
    assert "XAUUSD_ftmo(2)" in symbols.load_registry().sheets          # sigue en el registro (la unicidad la cuenta)
    # y si el usuario lo teclea a mano, sigue siendo un duplicado (E7): no guardable
    d = _draft(symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", {"XAUUSD": "XAUUSD_ftmo(2)"}, today=TODAY), "XAUUSD")
    assert d["guardable"] is False and d["existing"]["sqx_symbol"] == "XAUUSD_ftmo(2)"
    assert _hide(api, ["ftmo_cfd/XAUUSD_ftmo(2)"], False).status_code == 200
    xau = _draft(symbols.parse_mt5_sheet(_sheet_text(), "ftmo_cfd", None, today=TODAY), "XAUUSD")
    assert xau["candidates"] == ["XAUUSD_ftmo(2)"]


@pytest.mark.parametrize("campo, valor", [("lot_step", 5e-324), ("min_lot", 5e-324), ("lot_step", 1e-9),
                                          ("min_lot", 1e-9), ("lot_step", 0.0), ("lot_step", 1e-300)])
def test_h3_lot_step_y_min_lot_no_bajan_de_1e_8_y_dan_400(api, user_file, campo, valor):
    _mk_brokers(api)
    antes = user_file.read_text(encoding="utf-8")
    r = _send(api, "POST", "/api/symbols", _sheet(**{campo: valor}))
    assert r.status_code == 400, r.text
    assert any(campo in e and "1e-08" in e or campo in e and "mayor" in e for e in _errors(r)), _errors(r)
    assert user_file.read_text(encoding="utf-8") == antes
    v = _send(api, "POST", "/api/symbols/validate", _sheet(**{campo: valor})).json()
    assert v["ok"] is False
    # justo en el mínimo (1e-8) sí vale, y el mínimo múltiplo del paso también
    ok = _send(api, "POST", "/api/symbols", _sheet(lot_step=1e-8, min_lot=1e-8, sqx_symbol="MIN_mi"))
    assert ok.status_code == 200, ok.text


def test_h3_un_fichero_con_un_lot_step_subnormal_no_tumba_el_registro(user_file):
    ficha = {"point_value_usd": 1.0, "lot_step": 0.01, "min_lot": 0.01, "commission": {"per_lot_rt": 0.0}}
    user_file.write_text(_dump({
        "brokers": {"mi_cfd": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"}},
        "symbols": {"mi_cfd": {"BUENA_a": ficha, "SUBNORMAL_a": {**ficha, "lot_step": 5e-324, "min_lot": 1.0},
                               "OTRA_a": ficha}}}), encoding="utf-8")
    symbols._reload()
    reg = symbols.load_registry()                                     # antes: OverflowError (round(inf))
    assert {"BUENA_a", "OTRA_a"} <= set(reg.sheets) and "SUBNORMAL_a" not in reg.sheets
    assert len([e for e in reg.errors if "mi_cfd/SUBNORMAL_a" in e]) == 1


def test_h3_cualquier_excepcion_en_una_entrada_va_a_errors_y_las_demas_cargan(user_file, monkeypatch):
    ficha = {"point_value_usd": 1.0, "lot_step": 0.01, "min_lot": 0.01, "commission": {"per_lot_rt": 0.0}}
    user_file.write_text(_dump({
        "brokers": {"mi_cfd": {"display_name": "M", "company": "C", "asset_class": "cfd", "clock": "ny_wall"}},
        "symbols": {"mi_cfd": {"UNO_a": ficha, "EXPLOTA_a": ficha, "TRES_a": ficha}}}), encoding="utf-8")
    real = symbols._sheet_from_entry

    def flaky(broker, sqx_symbol, entry):
        if sqx_symbol == "EXPLOTA_a":
            raise RuntimeError("fallo inesperado")
        return real(broker, sqx_symbol, entry)

    monkeypatch.setattr(symbols, "_sheet_from_entry", flaky)
    symbols._reload()
    reg = symbols.load_registry()
    assert {"UNO_a", "TRES_a"} <= set(reg.sheets) and "EXPLOTA_a" not in reg.sheets
    assert any("mi_cfd/EXPLOTA_a" in e and "RuntimeError" in e and "fallo inesperado" in e for e in reg.errors)
    # lo mismo con un bróker de usuario que explota al parsearse
    real_pb = symbols._parse_broker

    def flaky_broker(bid, d, *, official):
        if bid == "mi_cfd":
            raise KeyError("x")
        return real_pb(bid, d, official=official)

    monkeypatch.setattr(symbols, "_parse_broker", flaky_broker)
    symbols._reload()
    assert any("bróker 'mi_cfd'" in e and "KeyError" in e for e in symbols.load_registry().errors)
    monkeypatch.undo()
    symbols._reload()


def test_h4_min_lot_al_menos_un_paso_de_lote(api):
    _mk_brokers(api)
    url = "/api/symbols/validate"
    # min_lot minúsculo frente a un paso enorme: antes pasaba por "múltiplo 0" (|ratio − round(ratio)| <= tol)
    r = _send(api, "POST", url, _sheet(lot_step=1000, min_lot=1e-8)).json()
    assert r["ok"] is False and any("menor que un paso de lote" in e for e in r["errors"]), r
    # justo un paso: vale; un paso con ruido de 1e-10 (dentro de la tolerancia de 1e-9) también
    assert _send(api, "POST", url, _sheet(lot_step=0.1, min_lot=0.1)).json()["ok"] is True
    assert _send(api, "POST", url, _sheet(lot_step=1000, min_lot=1000 * (1 - 1e-10), max_lot=_DEL)).json()["ok"] is True
    # medio paso: no es múltiplo (mensaje de siempre)
    r = _send(api, "POST", url, _sheet(lot_step=0.1, min_lot=0.05)).json()
    assert r["ok"] is False and any("no es múltiplo" in e for e in r["errors"])
    # dos pasos: vale
    assert _send(api, "POST", url, _sheet(lot_step=0.1, min_lot=0.2)).json()["ok"] is True
    # y por la API de alta real
    bad = _send(api, "POST", "/api/symbols", _sheet(lot_step=1000, min_lot=1e-8))
    assert bad.status_code == 400 and "menor que un paso" in _errors(bad)[0]


def test_h5_el_modulo_documenta_que_los_locks_son_intra_proceso():
    doc = symbols.__doc__
    assert "INTRA-PROCESO" in doc and "run_app.py" in doc and "workers" in doc


# ---------------------------------------------------------------------------
# F3 (guardrail entrada-4-bloques): correcciones de la revisión de código de la F2. H3 y H5 (los demás hallazgos de la
# F2 viven en sus ficheros: H1/H2 en tests/test_sqx_trades.py, H6-H9 en test_sqx_fichas.py).
# ---------------------------------------------------------------------------
def _fake_sqx_instrument(monkeypatch, **over):
    """Un ``.sqx`` del catálogo que propone ``XAUUSD_mi`` (sin tocar disco): proveedor de catálogo y lector falsos."""
    monkeypatch.setattr(symbols, "_catalog_provider", lambda: ["/no/importa/X.sqx"])
    monkeypatch.setattr(symbols, "read_instrument_info", lambda p: {
        "symbol": "XAUUSD_mi", "instrument": "XAUUSD_mi", "point_value": 100.0, "tick_size": 0.01,
        "order_size_step": 0.01, "default_spread": None, "commission": None, **over})
    return "/no/importa/X.sqx"


def test_f3_h3_la_frase_del_conflicto_con_una_ficha_de_usuario_de_otro_broker_no_ofrece_campo_a_campo(
        api, user_file, monkeypatch):
    """H3: ``_conflicts`` recibe el bróker de destino. Con una ficha de USUARIO de OTRO bróker el borrador es
    ``editable=False`` y la frase de cada conflicto lo dice (crea un bróker propio / dale a tus datos en SQX un nombre
    propio); no promete «campo a campo», que es solo para la ficha de usuario del propio bróker."""
    _mk_brokers(api)
    _create(api, min_lot=0.1, lot_step=0.1, spread_px=0.5, tick_size=0.01, max_lot=100)     # XAUUSD_mi en mi_cfd
    text = _sheet_text()
    mapping = {"XAUUSD": "XAUUSD_mi"}
    # destino = OTRO bróker (ftmo_cfd): la ficha de usuario vive en mi_cfd
    d = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", mapping, today=TODAY), "XAUUSD")
    assert (d["guardable"], d["editable"]) == (False, False) and d["existing"]["broker"] == "mi_cfd"
    assert {c["field"] for c in d["conflicts"]} == {"lot_step", "min_lot", "spread_px"}
    for c in d["conflicts"]:
        assert "campo a campo" not in c["texto"], c["texto"]
        assert "crea un bróker propio" in c["texto"] and "un nombre propio" in c["texto"] and "'mi_cfd'" in c["texto"]
    # destino = el MISMO bróker de la ficha: se acepta campo a campo (sin cambio)
    same = _draft(symbols.parse_mt5_sheet(text, "mi_cfd", mapping, today=TODAY), "XAUUSD")
    assert same["editable"] is True and {c["field"] for c in same["conflicts"]} == {"lot_step", "min_lot", "spread_px"}
    assert all("campo a campo" in c["texto"] and "bróker propio" not in c["texto"] for c in same["conflicts"])
    # la ficha OFICIAL conserva su frase de siempre aunque el destino sea otro bróker
    off = _draft(symbols.parse_mt5_sheet(_edit_sheet(text, "XAUUSD", **{"Min Lot Size": "0.1"}), "mi_cfd",
                                         {"XAUUSD": "XAUUSD_ftmo(2)"}, today=TODAY), "XAUUSD")
    mn = next(c for c in off["conflicts"] if c["field"] == "min_lot")
    assert mn["texto"].endswith("tu hoja dice 0.1; si tu cuenta es distinta, crea un bróker propio")
    # sin bróker de destino (alta desde un .sqx) una ficha de usuario es editable y se acepta campo a campo
    ds = symbols.draft_from_sqx(_fake_sqx_instrument(monkeypatch, order_size_step=0.05), today=TODAY)
    assert ds["editable"] is True and ds["conflicts"]
    assert all("puedes quedarte con el valor del .sqx, campo a campo" in c["texto"] for c in ds["conflicts"])


def test_f3_h5_el_alta_rechaza_una_ficha_de_usuario_oculta_y_el_borrador_lo_dice(api, user_file, monkeypatch):
    """H5: el POST de alta rechaza un ``sqx_symbol`` cuya ficha de usuario está OCULTA (E7: la unicidad cuenta las
    ocultas), así que el borrador sobre ella sale ``guardable=False``, ``editable=False`` y con el motivo «ficha oculta:
    muéstrala para editarla»; al mostrarla vuelve a ser como una ficha de usuario visible del propio bróker. Lo mismo
    para el borrador del alta desde un ``.sqx`` (sin bróker de destino)."""
    _mk_brokers(api)
    _create(api, min_lot=0.1, lot_step=0.1, spread_px=0.5, tick_size=0.01, max_lot=100)     # XAUUSD_mi en mi_cfd
    text, mapping = _sheet_text(), {"XAUUSD": "XAUUSD_mi"}
    draft = lambda: _draft(symbols.parse_mt5_sheet(text, "mi_cfd", mapping, today=TODAY), "XAUUSD")   # noqa: E731
    visible = draft()
    assert (visible["guardable"], visible["editable"], visible["motivo"]) == (True, True, None)
    assert _hide(api, ["mi_cfd/XAUUSD_mi"]).status_code == 200
    # lo que hace HOY el POST de alta con ese símbolo (oculto): lo rechaza con el texto E7
    r = _send(api, "POST", "/api/symbols", _sheet())
    assert r.status_code == 400 and "ya tiene ficha en el bróker 'mi_cfd'" in _errors(r)[0]
    # y el borrador es coherente con eso
    d = draft()
    assert (d["guardable"], d["editable"]) == (False, False)
    assert d["motivo"] == "ficha oculta: muéstrala para editarla" == symbols.HIDDEN_MOTIVO
    assert d["existing"] == {"broker": "mi_cfd", "sqx_symbol": "XAUUSD_mi", "official": False}
    assert d["conflicts"] and all("oculta" in c["texto"] and "muéstrala" in c["texto"] for c in d["conflicts"])
    # un motivo previo manda (p. ej. la moneda) y el de oculta pasa a los avisos
    eur = _edit_sheet(text, "XAUUSD", __after_pip_worth="EUR")
    dm = _draft(symbols.parse_mt5_sheet(eur, "mi_cfd", mapping, today=TODAY), "XAUUSD")
    assert dm["motivo"] == "divisa no USD (v1)" and symbols.HIDDEN_MOTIVO in dm["warnings"]
    # una ficha de usuario oculta de OTRO bróker sigue con el motivo E7 (mostrarla no la haría guardable)
    d_otro = _draft(symbols.parse_mt5_sheet(text, "ftmo_cfd", mapping, today=TODAY), "XAUUSD")
    assert d_otro["motivo"].startswith("el símbolo SQX 'XAUUSD_mi' ya tiene ficha en el bróker 'mi_cfd'")
    # el alta guiada desde un .sqx (sin bróker de destino) también
    path = _fake_sqx_instrument(monkeypatch)
    ds = symbols.draft_from_sqx(path, today=TODAY)
    assert (ds["guardable"], ds["editable"], ds["motivo"]) == (False, False, symbols.HIDDEN_MOTIVO)
    # al mostrarla vuelve a ser editable y guardable (por PUT, campo a campo)
    assert _hide(api, ["mi_cfd/XAUUSD_mi"], False).status_code == 200
    again = draft()
    assert (again["guardable"], again["editable"], again["motivo"]) == (True, True, None)
    assert all("campo a campo" in c["texto"] and "oculta" not in c["texto"] for c in again["conflicts"])
    ds2 = symbols.draft_from_sqx(path, today=TODAY)
    assert (ds2["guardable"], ds2["editable"], ds2["motivo"]) == (True, True, None)
