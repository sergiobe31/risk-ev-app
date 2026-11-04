"""Guardrail ``entrada-4-bloques``, fase F4: receta para el EA (``meta.recipe``), filas de carga con la ficha y guarda
``fichas`` (plan §2.8 y enmiendas E2, E3, E12, E14, E18).

Todo sintético y hermético, como ``test_axi_sqx.py`` y ``test_sqx_fichas.py``: los ``.sqx`` se escriben con el writer
de ``tests/test_sqx_trades.py``, los OHLC y el fichero de fichas del usuario van a carpetas temporales (I4: nunca los
reales). Los tests que necesitan la demo (``scripts/export_shareable/make_demo.py``) se omiten si no viaja.

Mapa criterio -> test (el id del criterio está en el nombre):

- AC4.1 ``meta.recipe`` con la forma del brief en prop firm (demo CFD y demo de futuros) y en AXI; ausente en CSV y con
  la respuesta CSV igual que antes de la fase ........................................................ ``test_ac4_1_*``
- AC4.2 golden a mano (riesgos por lote conocidos) y el caso ``max_lot_excede_ficha_by_f`` con su aviso . ``test_ac4_2_*``
- AC4.3 (E2) la receta equivale a la simulación: propiedad con RNG fijo contra un oráculo que sigue el orden de
  operaciones del EA ................................................................................... ``test_ac4_3_*``
- AC4.4 las notas nuevas existen y la meta las cita (los pins de ``config/model_notes.yaml`` viven en
  ``tests/test_model_notes.py``) ...................................................................... ``test_ac4_4_*``
- C filas de ``portfolio_load`` con ``broker``, ``ficha_sin_verificar``, ``mm_backtest`` y ``paridad_frac_ok``; guarda
  ``fichas`` ......................................................................................... ``test_c_*``

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest poc_web/api/test_recipe.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from poc_web.api import main, symbols
from poc_web.api.model_notes import NOTES
from src.risk.futures_legs import fixed_risk_micros
from src.risk.propfirm import eval_contract_cap_micros

ROOT = Path(__file__).resolve().parents[2]
REAL_USER_FILES = (ROOT / "config" / "symbols.user.yaml", ROOT / "config" / "propfirms.user.yaml")


def _load_sibling(name: str, fname: str):
    """Carga un módulo de test hermano por ruta (el patrón del repo: ``poc_web/api`` no es un paquete, y un ``import``
    normal sacaría este test del cierre del export)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIBLINGS: dict[str, object] = {}


def _sib(fname: str):
    if fname not in _SIBLINGS:
        _SIBLINGS[fname] = _load_sibling("_tst_recipe_" + Path(fname).stem, fname)
    return _SIBLINGS[fname]


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
    """Captura las ``StrategyMC`` (con sus lotes) y los bloques que ``simulate_axi`` pasa a ``build_block_records``: lo que
    el motor USA de verdad."""
    seen: dict = {}
    real = main.build_block_records

    def wrapped(strategies, blocks, oos_start=None):
        seen["strats"] = strategies
        seen["blocks"] = real(strategies, blocks, oos_start=oos_start)
        return seen["blocks"]

    monkeypatch.setattr(main, "build_block_records", wrapped)
    return seen


# ---------------------------------------------------------------------------
# AC4.1 (parte CSV): la vía CSV no cambia. Pins tomados ANTES de la fase F4 (con el código de la F3) sobre la cartera
# sintética de ``test_csv_minute_float.py``: respuesta COMPLETA (meta incluida, sin ``elapsed_s``) de los dos endpoints.
# ---------------------------------------------------------------------------
def _canon(out) -> str:
    """Huella (sha256, 16 hex) de una respuesta completa, sin ``elapsed_s`` y con los floats a 6 decimales (estable entre
    plataformas); lo no numérico (textos, claves, listas) entra tal cual."""
    def r(o):
        if isinstance(o, dict):
            return {k: r(v) for k, v in sorted(o.items()) if k != "elapsed_s"}
        if isinstance(o, (list, tuple)):
            return [r(v) for v in o]
        if isinstance(o, float):
            return "nan" if o != o else round(o, 6)
        return o
    blob = json.dumps(r(out), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _canon_sin_bloque(out) -> str:
    """:func:`_canon` de la respuesta SIN lo que el guardrail ``bloque-politis-white`` cambia por diseño: la descripción del
    bloque (``meta.model_context.block_size``) y su nota ``tamano_bloque``, ``meta.bs1`` (la sensibilidad con bloque 1) y los
    avisos de esa sensibilidad. Todo lo demás (reports, recomendación, trayectoria, f*, resto de la meta y del
    ``model_context``, otros avisos) entra entero."""
    o = json.loads(json.dumps(out, default=str))
    mc = o["meta"]["model_context"]
    mc.pop("block_size")
    mc["notes"] = [n for n in mc["notes"] if n["id"] != "tamano_bloque"]
    o["meta"].pop("bs1", None)   # None: el árbol viejo (ce7c5f9) no tiene la clave (revisión de código, H2)
    o["warnings"] = [w for w in o["warnings"] if not w.startswith("Con bloque 1")]
    return _canon(o)


# Tomados con ``main.py`` tal como queda tras la F3 (el parche acumulado F1+F2+F3, sin ningún cambio de la F4), en dos
# corridas con carpetas temporales distintas (mismo valor): ``_canon(simulate_propfirm(ftmo_100k))`` y
# ``_canon(simulate_axi(seed))`` de la cartera CSV sintética.
PIN_CSV_PROPFIRM = "30261b1202163019"
PIN_CSV_AXI = "ea4d4a134dc47272"
# bloque-politis-white (2026-10-21, E7 opción 1): el bloque automático cambia, así que la respuesta se pide con el bloque que daba
# la regla vieja en este escenario (10,0, las dos lentes) y se compara SIN lo que el guardrail cambia por diseño (la descripción
# del bloque en ``meta.model_context.block_size`` y su nota ``tamano_bloque``, ``meta.bs1`` y su aviso: ``_canon_sin_bloque``).
# Las huellas de abajo NO se calcularon con el código nuevo: salen del árbol de ANTES del guardrail (HEAD ce7c5f9) con ese mismo
# canon, pidiendo ``block_size=10.0`` (y la huella de ``_canon`` completa de ese árbol con el bloque automático de entonces sigue
# siendo PIN_CSV_PROPFIRM/PIN_CSV_AXI).
CSV_OLD_BLOCK = 10.0
PIN_CSV_PROPFIRM_SIN_BLOQUE = "263611e776650629"
PIN_CSV_AXI_SIN_BLOQUE = "4ec9fcbdc3e01104"
RNG_CANARY = [0.1257302210933933, -0.1321048632913019]        # np.random.default_rng(0).normal(size=2) (numpy 2.4.4)
# Filas de ``portfolio_load`` de una pata CSV y guardas de una cartera CSV, también de antes de la F4 (la guarda ``fichas`` y
# las claves nuevas de fila son solo de las carteras .sqx).
CSV_ROW_KEYS = ["pata", "símbolo", "modo", "trades", "status", "warnings", "sqx"]
CSV_GUARD_KEYS = ["crossing", "bloques", "cobertura", "valles", "capital", "sqx", "m1"]
# H6: el CONJUNTO de claves de la respuesta CSV de cada lente (también de antes de la F4). Se comprueba SIN el pin de arriba,
# que se omite si cambia el RNG de numpy: una clave de meta que aparezca o desaparezca en una cartera CSV se ve igual.
CSV_TOP_KEYS_PROPFIRM = {"f_star", "intentos_garantizados", "meta", "recommendation", "reports", "trajectory", "warnings"}
# bloque-politis-white (2026-10-21): ``bs1`` (la sensibilidad con bloque 1) es la clave nueva de la meta de las dos lentes.
CSV_META_KEYS_PROPFIRM = {
    "account_size", "block_size", "bs1", "cap_violation_tol", "coverage", "day_axis", "day_axis_note", "elapsed_s", "fee_eur", "firm",
    "float_mode", "float_note", "legs", "model_context", "n_blocks", "n_copied_accounts", "n_mae_bounded", "n_mae_zero",
    "n_no_valley", "n_paths", "oos_start", "pnl_w_mae_bounded", "pnl_w_mae_zero", "pnl_w_no_valley", "ruleset_version",
    "sizing", "sizing_base", "sizing_invalid_any", "source_tz", "trades_in_window", "trailing"}
CSV_TOP_KEYS_AXI = {"f_star", "meta", "recommendation", "reports", "warnings"}
CSV_META_KEYS_AXI = {
    "baseline", "block_size", "bs1", "coverage", "crossing", "days_cal", "elapsed_s", "etapa", "etapa_display", "f_star_shift_note",
    "float_convention", "floors_applied", "horizon_days", "legs", "min_days", "min_unique_trades", "model_context", "motor",
    "n_blocks", "n_blocks_full", "n_mae_bounded", "n_mae_zero", "n_paths", "oos_start", "pass_line", "pnl_w_mae_bounded",
    "pnl_w_mae_zero", "policy_assumption", "ruin_line", "semantics", "sizing", "sizing_base", "trades_in_window"}


def _csv_world(tmp_path, monkeypatch):
    """La cartera CSV sintética de ``test_csv_minute_float.py`` (un trade por día, 10 % sin M1) cargada con
    ``portfolio_load``: ``(módulo del fixture, carga)``."""
    t = _sib("test_csv_minute_float.py")
    s = t._synth_files(tmp_path)
    monkeypatch.setattr(main, "_load_ohlc", lambda sym: s.ohlc)
    res = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(s.csv)], account_size=t._ACCOUNT, f_ref=t._F_REF))
    assert res["ok"], res
    return t, res


def test_ac4_1_la_respuesta_csv_de_las_dos_lentes_es_la_de_antes_de_la_fase_y_no_lleva_recipe(tmp_path, monkeypatch):
    t, load = _csv_world(tmp_path, monkeypatch)
    pf = main.simulate_propfirm(t._req(load["portfolio_id"], block_size=CSV_OLD_BLOCK))     # E7, opción 1 (ver los PIN_*)
    axi = main.simulate_axi(main.AxiSimRequest(portfolio_id=load["portfolio_id"], stage="seed", f_lo=0.01, f_hi=0.02,
                                               f_step=0.01, n_paths=100, block_size=CSV_OLD_BLOCK))
    assert pf["meta"]["block_size"] == axi["meta"]["block_size"] == CSV_OLD_BLOCK
    assert pf["meta"]["bs1"] is not None and axi["meta"]["bs1"] is not None          # bloque 10 > 1: hay sensibilidad
    assert len(pf["meta"]["bs1"]) == len(axi["meta"]["bs1"]) == 1
    assert "recipe" not in pf["meta"] and "recipe" not in axi["meta"]
    assert set(pf) == CSV_TOP_KEYS_PROPFIRM and set(pf["meta"]) == CSV_META_KEYS_PROPFIRM          # H6: sin depender del RNG
    assert set(axi) == CSV_TOP_KEYS_AXI and set(axi["meta"]) == CSV_META_KEYS_AXI
    # las cifras dependen del RNG sembrado de numpy: si su flujo cambia en otra versión, el pin no es comparable (se omite
    # en vez de dar un falso fallo); lo estructural de arriba y de los otros tests de este módulo sí corre
    if np.random.default_rng(0).normal(size=2).tolist() != RNG_CANARY:
        pytest.skip("el flujo del RNG de numpy es otro: los pins de la respuesta CSV no son comparables")
    assert _canon_sin_bloque(pf) == PIN_CSV_PROPFIRM_SIN_BLOQUE
    assert _canon_sin_bloque(axi) == PIN_CSV_AXI_SIN_BLOQUE


def test_c_las_filas_y_las_guardas_de_una_cartera_csv_no_cambian(tmp_path, monkeypatch):
    """Las filas CSV no cambian: ni ``broker``, ni ``ficha_sin_verificar``, ni ``mm_backtest``, ni ``paridad_frac_ok``; y la
    guarda ``fichas`` es solo de las carteras ``.sqx``."""
    _, load = _csv_world(tmp_path, monkeypatch)
    assert [list(r) for r in load["legs"]] == [CSV_ROW_KEYS]
    assert list(load["guards"]) == CSV_GUARD_KEYS and "fichas" not in load["guards"]


# ---------------------------------------------------------------------------
# Utilidades: patas falsas, el oráculo del EA y la comparación contra la receta
# ---------------------------------------------------------------------------
def _fake_leg(name, risk, *, step=0.01, min_lot=0.01, asset="cfd", ficha_max_lot=None, mm_backtest=None, paridad=None):
    """Pata ``.sqx`` mínima con lo que leen ``_lots_for_leg`` y ``_sqx_recipe``: el riesgo por lote ``|open − SL| · pv`` de
    cada orden (``risk``), los pasos de lote, el bróker y la ficha."""
    risk = np.asarray(risk, dtype=float)
    return {"name": name, "kind": main.KIND_SQX, "asset": asset, "broker": "axi" if asset == "cfd" else "ftmo_futures",
            "fleg": types.SimpleNamespace(risk_per_micro=risk, comm_per_lot=np.zeros(len(risk))),
            "meta": {"lot_step": step, "min_lot": min_lot, "symbol": "SIM_" + name},
            "ficha": {"max_lot": ficha_max_lot}, "mm_backtest": mm_backtest, "paridad": paridad}


def _recipe_of(leg, f_grid, account_size, cap=None, lens="axi"):
    """``(lotes, receta, avisos)`` con las funciones de la app: los lotes que entran al motor y la receta que sale de ellos."""
    lots, diag = main._lots_for_leg(leg, np.asarray(f_grid, dtype=float), {}, 0.005, account_size, cap)
    recipe, warns = main._sqx_recipe([leg], {leg["name"]: lots}, {leg["name"]: diag}, f_grid, account_size, lens)
    return lots, recipe, warns


def _round_down(value: float, step: float, decimals: int) -> float:
    """``roundDown(value, step, decimals)`` del MQL5 exportado por SQX: ``MathFloor(value/step)·step`` y luego, truncando a
    ``decimals`` (``MathFloor(v·10^d + 1e-8)/10^d`` normalizado; con 0 decimales, ``(int) v``)."""
    if step > 0:
        value = math.floor(value / step) * step
    if decimals == 0:
        return float(int(value))
    p = 10.0 ** decimals
    return round(math.floor(value * p + 1e-8) / p, decimals)


def _ea_lot(risked_money, risk_one_lot, *, step, decimals, min_lot, lots_if_no_mm, max_lots, multiplier=1.0) -> float:
    """El lote que abre el EA de SQX con ``FixedAmount``, siguiendo SU orden de operaciones, sin usar nada de la app:
    1) ``oneLot = round(pv·|open − SL|, 7)``; 2) ``L = R / oneLot · mmMultiplier``; 3) ``roundDown`` al paso y a ``Decimals``;
    4) si ``L ≤ 0`` → ``LotsIfNoMM``; si ``L > MaxLots`` → ``MaxLots``; si ``L < mínimo`` → mínimo.

    Es el modelo del EA de la base factual §2 (``sqMMFixedAmount`` del MQL5 exportado), NO el MQL5 en sí. El tope del bróker EN
    VIVO (el máximo del símbolo en MT5, que el EA aplica al final) no está modelado: lo cubre ``max_lot_excede_ficha_by_f`` con
    su aviso (cuando es ``True``, la receta no reproduce la simulación y la app lo dice). ``max_lots=math.inf`` = el EA SIN el
    ``MaxLots`` de la receta."""
    one_lot = round(risk_one_lot, 7)
    lot = risked_money / one_lot if one_lot > 0 else 0.0
    lot = _round_down(lot * multiplier, step, decimals)
    if lot <= 0:
        lot = _round_down(lots_if_no_mm, step, decimals)
    if lot > max_lots:
        lot = _round_down(max_lots, step, decimals)
    if lot < min_lot:
        lot = _round_down(min_lot, step, decimals)
    return lot


BAND = 1e-6          # AC4.3: «frontera» = R/(oneLot·mmStep) a menos de esto de un entero
# H1: la app suma 1e-9 al cociente antes del floor y el EA calcula ``(R/oneLot)/mmStep``: las dos rutas difieren del cociente
# real en <= ~2,2e-16·q cada una (dos redondeos), así que el +1e-9 de la app domina el ruido (y la app queda >= EA en toda
# frontera) mientras q < ~2e6 pasos. Es un orden de magnitud irreal (2.000 lotes con paso 0,001): el comparador lo exige.
Q_MAX = 2.0e6


def _compare_with_oracle(recipe_leg: dict, recipe: dict, risks, lots, stats: dict, cap=None) -> None:
    """AC4.3: aplica el oráculo del EA a cada orden y a cada f con SOLO los parámetros de la receta y lo compara con los
    lotes de la app. Las precondiciones del AC están aquí: ``LotsIfNoMM`` = ``min_lot``, ``MaxLots`` = ``max_lot_by_f``,
    ``mmMultiplier`` = 1 y ``mmStep`` = ``lot_step``. Por orden y f:

    - **Fuera de fronteras** (``R/(oneLot·mmStep)`` dista más de ``BAND`` de un entero): igualdad EXACTA con el lote de la app.
    - **En las fronteras**: diferencia ≤ un paso y, H1, la app queda SIEMPRE ≥ el EA (nunca por debajo: la simulación no
      subestima el lote que abriría el EA).
    - **H2**: el EA SIN el ``MaxLots`` de la receta (``ea_sin``) nunca supera ``max_lot_by_f``: la receta no recorta al EA, ni
      fuera ni en las fronteras (``stats["muerde_exact"]`` y ``stats["muerde_frontier"]`` cuentan ``ea_sin > max_lot_by_f``;
      todo lo que cuentan tiene que ser el caso siguiente). La excepción es el tope de contratos del perfil en futuros
      (``cap``): la app también recorta a él, la receta lo recoge en ``MaxLots`` (= el tope) y ahí SÍ corta al EA, por diseño
      (``stats["maxlots_tope_perfil"]``). PLAN-DESVIO (menor): el encargo de la revisión dice «asertar 0 fuera de fronteras»;
      con el tope de contratos del perfil ese 0 literal no puede ser (la receta de esas patas NECESITA que ``MaxLots`` sea el
      tope para reproducir la simulación), así que lo que se aserta es 0 mordiscos SIN EXPLICAR por ese tope, también en las
      fronteras (donde, además, se cuentan y se imprimen los explicados).
    - **H4**: ``max_lot_by_f`` ≥ todo lote del motor, SIN tolerancia (el array del motor ya viene redondeado a 10 decimales)."""
    step = recipe_leg["mmStep"]
    assert recipe_leg["LotsIfNoMM"] == recipe_leg["min_lot"]
    assert recipe_leg["mmMultiplier"] == 1 and recipe_leg["mmStep"] == recipe_leg["lot_step"]
    ea_kw = dict(step=step, decimals=recipe_leg["decimals"], min_lot=recipe_leg["min_lot"],
                 lots_if_no_mm=recipe_leg["LotsIfNoMM"], multiplier=recipe_leg["mmMultiplier"])
    for i, R in enumerate(recipe["risked_money_by_f"]):
        max_lots = recipe_leg["max_lot_by_f"][i]
        assert max_lots >= float(np.max(lots[i])), f"f#{i}: MaxLots {max_lots!r} < el mayor lote simulado {np.max(lots[i])!r}"
        for j, risk in enumerate(risks):
            ea = _ea_lot(R, float(risk), max_lots=max_lots, **ea_kw)
            ea_sin = _ea_lot(R, float(risk), max_lots=math.inf, **ea_kw)
            app = float(lots[i][j])
            q = R / (round(float(risk), 7) * step)
            assert q < Q_MAX, f"f#{i} orden {j}: q={q!r} fuera del rango en que el +1e-9 de la app domina el ruido"
            muerde = ea_sin > max_lots + 1e-12
            por_tope = muerde and cap is not None and ea_sin > cap + 1e-12 and max_lots >= cap - 1e-12
            stats["orders"] += 1
            stats["min_lot"] += q < 1.0
            stats["maxlots_tope_perfil"] += por_tope
            if abs(q - round(q)) > BAND:
                stats["muerde_exact"] += muerde
                assert ea == app, (f"f#{i} orden {j}: R={R!r} riesgo/lote={risk!r} q={q!r} -> EA {ea!r}, app {app!r}")
                assert not muerde or por_tope, (f"f#{i} orden {j}: el EA sin MaxLots abriría {ea_sin!r} > MaxLots {max_lots!r} "
                                                f"(app {app!r}): la receta le corta el lote")
                stats["exact"] += 1
            else:
                stats["muerde_frontier"] += muerde
                assert abs(ea - app) <= step + 1e-9, (f"frontera f#{i} orden {j}: q={q!r} -> EA {ea!r}, app {app!r}")
                assert app >= ea - 1e-12, (f"frontera f#{i} orden {j}: q={q!r} -> la app queda POR DEBAJO del EA "
                                           f"(app {app!r}, EA {ea!r})")
                assert not muerde or por_tope, (f"frontera f#{i} orden {j}: el EA sin MaxLots abriría {ea_sin!r} > MaxLots "
                                                f"{max_lots!r} (app {app!r}): la receta le corta el lote")
                stats["frontier"] += 1
                stats["frontier_diff"] += ea != app
                stats["frontier_app_above"] += app > ea


def _new_stats() -> dict:
    return {"orders": 0, "exact": 0, "frontier": 0, "frontier_diff": 0, "frontier_app_above": 0, "min_lot": 0,
            "muerde_exact": 0, "muerde_frontier": 0, "maxlots_tope_perfil": 0}


# ---------------------------------------------------------------------------
# AC4.1: forma de meta.recipe (prop firm: demo CFD y demo de futuros; AXI: sintético de test_axi_sqx.py)
# ---------------------------------------------------------------------------
RECIPE_KEYS = {"mm", "currency", "account_size", "f_grid", "risked_money_by_f", "legs", "notes"}
LEG_KEYS = {"pata", "symbol", "broker", "min_lot", "lot_step", "decimals", "mmStep", "mmMultiplier", "LotsIfNoMM",
            "max_lot_by_f", "ficha_max_lot", "max_lot_excede_ficha_by_f", "frac_min_lot_by_f",
            "p95_riesgo_real_pct_by_f", "mm_backtest", "paridad"}


def _yaml_note_text(nid: str) -> str:
    return next(n.texto for n in NOTES if n.id == nid)


def _expected_leg(leg: dict, f_grid, account_size: float, cap) -> dict:
    """Lo que debe decir la receta de una pata, recalculado AQUÍ desde su riesgo por lote (no desde ``_sqx_recipe``)."""
    risk = np.asarray(leg["fleg"].risk_per_micro, dtype=float)
    # los pasos de lote de la ficha (los futuros no los llevan en el meta del lector: valen 1 y 1, fijos)
    step, mn = float(leg["meta"].get("lot_step") or 1.0), float(leg["meta"].get("min_lot") or 1.0)
    lots, _ = fixed_risk_micros(risk, f_grid, account_size, cap=cap, comm_per_micro=leg["fleg"].comm_per_lot,
                                lot_step=step, min_lot=mn)
    fs = [float(f) for f in f_grid]
    return {
        "lots": lots, "step": step, "min_lot": mn,
        "max": [float(lots[i].max()) for i in range(len(fs))],
        "frac": [float(np.mean(f * account_size < risk * mn)) for f in fs],
        "p95_pct": [float(np.percentile(lots[i] * risk / (f * account_size), 95)) * f * 100.0 for i, f in enumerate(fs)],
    }


def _check_recipe_shape(recipe: dict, legs_rows: list[dict], legs_by_name: dict, f_grid, account_size: float, cap,
                        note_ids: list[str]) -> None:
    assert set(recipe) == RECIPE_KEYS
    assert (recipe["mm"], recipe["currency"], recipe["account_size"]) == ("FixedAmount", "USD", account_size)
    assert recipe["f_grid"] == [float(f) for f in f_grid]
    assert recipe["risked_money_by_f"] == [float(f) * account_size for f in f_grid]
    assert [l["pata"] for l in recipe["legs"]] == [r["pata"] for r in legs_rows]
    for rl, row in zip(recipe["legs"], legs_rows):
        assert set(rl) == LEG_KEYS, row["pata"]
        leg = legs_by_name[row["pata"]]
        exp = _expected_leg(leg, f_grid, account_size, cap)
        sheet = symbols.resolve(row["símbolo"])
        assert rl["symbol"] == row["símbolo"] and rl["broker"] == sheet.broker == leg["broker"]
        assert (rl["min_lot"], rl["lot_step"]) == (sheet.min_lot, sheet.lot_step) == (exp["min_lot"], exp["step"])
        assert rl["decimals"] == (0 if exp["step"] == 1.0 else 2)
        assert (rl["mmStep"], rl["mmMultiplier"], rl["LotsIfNoMM"]) == (rl["lot_step"], 1, rl["min_lot"])
        assert rl["max_lot_by_f"] == pytest.approx(exp["max"], abs=1e-9)
        for i in range(len(exp["max"])):                       # un tope que nunca muerde: ≥ todo lote simulado (H4: SIN
            assert exp["lots"][i].max() <= rl["max_lot_by_f"][i]     # tolerancia: el array del motor viene a 10 decimales), y ajustado
            assert rl["max_lot_by_f"][i] - exp["step"] < exp["lots"][i].max() + 1e-12
        assert rl["ficha_max_lot"] == sheet.max_lot
        assert rl["max_lot_excede_ficha_by_f"] == [sheet.max_lot is not None and m > sheet.max_lot
                                                   for m in rl["max_lot_by_f"]]
        assert all(type(b) is bool for b in rl["max_lot_excede_ficha_by_f"])        # booleanos reales, no 0/1
        assert rl["frac_min_lot_by_f"] == pytest.approx(exp["frac"], abs=1e-12)
        assert rl["p95_riesgo_real_pct_by_f"] == pytest.approx(exp["p95_pct"], rel=1e-12)
        assert len(rl["max_lot_by_f"]) == len(rl["frac_min_lot_by_f"]) == len(f_grid)
        assert rl["mm_backtest"] is None and rl["paridad"] is None                  # estos .sqx no traen lastSettings.xml
    assert [n["id"] for n in recipe["notes"]] == note_ids
    assert all(set(n) == {"id", "texto"} and n["texto"] == _yaml_note_text(n["id"]) for n in recipe["notes"])
    json.dumps(recipe, allow_nan=False)                                             # JSON puro


DEMO_CASES = {
    "cfd": ("demo_cfd", "ftmo_50k", "ftmo", ("ftmo_cfd", 0.01, 2)),
    "futuros": ("demo_futuros", "ftmo_futures_growth_50k", "servidor", ("ftmo_futures", 1.0, 0)),
}


@pytest.mark.parametrize("caso", sorted(DEMO_CASES))
def test_ac4_1_recipe_en_prop_firm_con_la_demo_cfd_y_la_demo_de_futuros(tmp_path, caso):
    """La forma de B en prop firm (demo CFD + ``ftmo_50k``; demo de futuros + ``ftmo_futures_growth_50k``, donde
    ``lot_step = min_lot = 1`` y ``decimals = 0``), con cada cifra recalculada aquí desde el riesgo por lote de la pata."""
    folder, firm, axis, (broker, step, decimals) = DEMO_CASES[caso]
    _sib("test_sqx_fichas.py")._make_demo(tmp_path / "demo")
    legs = sorted(str(p) for p in (tmp_path / "demo" / folder).glob("*.sqx"))
    account = 50000.0
    load = main.portfolio_load(main.PortfolioLoadRequest(legs=legs, account_size=account, firm_key=firm))
    assert load["ok"], load
    out = main.simulate_propfirm(main.PropfirmSimRequest(
        portfolio_id=load["portfolio_id"], firm_key=firm, f_lo=0.0025, f_hi=0.01, f_step=0.0025, n_paths=100,
        n_challenges=3, n_sequences=100, day_axis=axis))
    f_grid = np.arange(0.0025, 0.01 + 0.0025 / 2, 0.0025)            # la misma expresión que el endpoint
    cap = eval_contract_cap_micros(main._firms[firm]) if caso == "futuros" else None
    by_name = {Path(p).name: main._load_leg(p, account, 0.005) for p in legs}
    meta = out["meta"]
    _check_recipe_shape(meta["recipe"], meta["legs"], by_name, f_grid, account, cap, ["receta_ea", "fichas_broker"])
    rl = meta["recipe"]["legs"]
    assert len(rl) == 3 and {l["broker"] for l in rl} == {broker}
    assert {(l["lot_step"], l["min_lot"], l["decimals"]) for l in rl} == {(step, step, decimals)}
    assert all(l["ficha_max_lot"] is None and not any(l["max_lot_excede_ficha_by_f"]) for l in rl)
    assert not any("Receta para el EA" in w for w in out["warnings"])                # sin exceso: sin aviso de la receta
    # la receta no es lo único que pasa por la meta: el resto sigue siendo el de la lente
    assert meta["leg_kind"] == main.KIND_SQX and meta["sizing"] == main.SQX_SIZING_MODE


def test_ac4_1_en_prop_firm_a_es_el_tamano_de_cuenta_de_la_cartera_no_el_del_perfil(tmp_path):
    """El brief: ``A`` = el ``account_size`` con el que se calcularon los lotes, el de la CARTERA. La cuenta del perfil
    (``ftmo_50k``: 50.000) puede ser otra, y el aviso de siempre lo dice."""
    _sib("test_sqx_fichas.py")._make_demo(tmp_path / "demo")
    legs = sorted(str(p) for p in (tmp_path / "demo" / "demo_cfd").glob("*.sqx"))
    load = main.portfolio_load(main.PortfolioLoadRequest(legs=legs, account_size=100000.0, firm_key="ftmo_50k"))
    out = main.simulate_propfirm(main.PropfirmSimRequest(
        portfolio_id=load["portfolio_id"], firm_key="ftmo_50k", f_lo=0.0025, f_hi=0.005, f_step=0.0025, n_paths=100,
        n_challenges=3, n_sequences=100, day_axis="ftmo"))
    r = out["meta"]["recipe"]
    assert out["meta"]["account_size"] == 50000.0 and r["account_size"] == 100000.0
    assert r["risked_money_by_f"] == [0.0025 * 100000.0, 0.005 * 100000.0]
    assert any("distinto del tamaño de cuenta del perfil" in w for w in out["warnings"])
    by_name = {Path(p).name: main._load_leg(p, 100000.0, 0.005) for p in legs}
    _check_recipe_shape(r, out["meta"]["legs"], by_name, np.array([0.0025, 0.005]), 100000.0, None,
                        ["receta_ea", "fichas_broker"])


def test_ac4_1_recipe_en_la_lente_axi_con_el_sintetico_de_test_axi_sqx_y_booleanos_reales_por_http(tmp_path, api):
    T = _sib("test_axi_sqx.py")
    w = T.make_axi_world(tmp_path, multiday=True)
    load = api.post("/api/portfolio/load", json={"legs": w.paths, "account_size": T.ACCOUNT}).json()
    assert load["ok"], load
    r = api.post("/api/simulate/axi", json=dict(portfolio_id=load["portfolio_id"], stage="seed", n_paths=100, **T.F_KW))
    assert r.status_code == 200, r.text
    meta = r.json()["meta"]
    f_grid = T._f_grid()
    by_name = {Path(p).name: main._load_leg(p, T.ACCOUNT, T.F_REF, with_minutes=True, synth_mode="daily")
               for p in w.paths}
    _check_recipe_shape(meta["recipe"], meta["legs"], by_name, f_grid, T.ACCOUNT, None,
                        ["receta_ea", "fichas_broker", "axi_sqx_flotante"])
    rl = {l["symbol"]: l for l in meta["recipe"]["legs"]}
    assert set(rl) == {"XAUUSD_axi", "EURUSD_axi"} and {l["broker"] for l in rl.values()} == {"axi"}
    assert (rl["XAUUSD_axi"]["ficha_max_lot"], rl["EURUSD_axi"]["ficha_max_lot"]) == (20.0, 100.0)    # broker_axi.yaml
    for l in rl.values():                                                       # en el JSON de HTTP: booleanos, no 0/1
        assert l["max_lot_excede_ficha_by_f"] == [False] * 4
        assert all(x is False for x in l["max_lot_excede_ficha_by_f"])
    assert meta["recipe"]["account_size"] == T.ACCOUNT == 5000.0
    assert meta["recipe"]["risked_money_by_f"] == pytest.approx([f * 5000.0 for f in f_grid], rel=1e-12)


def test_ac4_1_la_receta_describe_las_patas_recargadas_y_el_array_de_lotes_del_motor(tmp_path, spy):
    """La receta se calcula con las patas que SE SIMULARON (como ``meta.legs``, H4) y con el MISMO array de lotes que entró
    al motor (``spy``)."""
    T = _sib("test_axi_sqx.py")
    w = T.make_axi_world(tmp_path, n_days=40)
    pid = T._load(w.paths)["portfolio_id"]
    w2 = T.make_axi_world(tmp_path, n_days=60)                   # el .sqx cambia entre la carga y la simulación
    for p in w2.paths:
        ns = Path(p).stat().st_mtime_ns + 5_000_000_000
        os.utime(p, ns=(ns, ns))
    out = T._sim(pid, n_paths=100)
    r = out["meta"]["recipe"]
    assert [l["pata"] for l in r["legs"]] == ["AXI XAU.sqx", "AXI EUR.sqx"]
    for rl in r["legs"]:
        lots = spy["strats"][rl["pata"]].lots
        assert lots.shape == (4, 60)                              # las 60 operaciones recargadas, no las 40 de la carga
        assert rl["max_lot_by_f"] == [float(lots[i].max()) for i in range(4)]       # exacto: es el array del motor
    assert [l["trades"] for l in out["meta"]["legs"]] == [60, 60]


def test_ac4_1_la_receta_pasa_el_mm_del_backtest_y_la_paridad_de_la_pata_como_datos_informativos(tmp_path):
    """``mm_backtest`` (``{method, params}``) y ``paridad`` de cada pata viajan tal cual en la receta (F2, informativos): con un
    ``.sqx`` de ``FixedAmount`` y lotes que siguen la regla, la paridad es 1,0; la receta es JSON puro con ellos dentro."""
    F = _sib("test_sqx_fichas.py")
    fa = F.w_eur_ftmo(F._root(tmp_path) / "FA.sqx", F._fixed_amount_trades(), last_settings=F.LAST_FIXED_AMOUNT)
    load = main.portfolio_load(main.PortfolioLoadRequest(legs=[fa], account_size=F.ACCOUNT, firm_key="ftmo_50k"))
    assert load["ok"], load
    out = main.simulate_propfirm(main.PropfirmSimRequest(
        portfolio_id=load["portfolio_id"], firm_key="ftmo_50k", f_lo=0.0025, f_hi=0.005, f_step=0.0025, n_paths=100,
        n_challenges=3, n_sequences=100, day_axis="ftmo"))
    rl = out["meta"]["recipe"]["legs"][0]
    leg = main._portfolios[load["portfolio_id"]]["legs"][0]
    assert rl["mm_backtest"] == {"method": "FixedAmount", "params": {
        "RiskedMoney": F.RISKED, "Decimals": 2.0, "LotsIfNoMM": 0.01, "MaxLots": 10.0}}
    assert rl["paridad"] == leg["paridad"] and rl["paridad"]["frac_ok"] == 1.0 and rl["paridad"]["risked_money"] == F.RISKED
    assert rl["paridad"]["n"] == 60 and rl["paridad"]["n_lifm"] == 15 and rl["paridad"]["frac_ok_sin_lifm"] == 1.0
    assert rl["mm_backtest"] is not leg["mm_backtest"]                       # una copia: la caché de la pata no se comparte
    json.dumps(out["meta"]["recipe"], allow_nan=False)


# ---------------------------------------------------------------------------
# AC4.2: golden a mano
# ---------------------------------------------------------------------------
GOLD_RISKS = [50.0, 125.0, 400.0, 6000.0]            # |open − SL| · pv, USD por lote (oro de 100 $/punto: 0,5 · 1,25 · 4 · 60 $)
# A = 10.000, f = [0,005; 0,01; 0,02] -> R = [50, 100, 200]. Paso y mínimo 0,01. Lotes = floor(R/(riesgo·0,01))·0,01, al mínimo
# si da 0 (a mano):
#   R=50 : 50/0,5 = 100 -> 1,00 · 50/1,25 = 40 -> 0,40 · 50/4 = 12,5 -> 0,12 · 50/60 = 0,83 -> 0 -> mínimo 0,01
#   R=100: 100/0,5 = 200 -> 2,00 · 100/1,25 = 80 -> 0,80 · 100/4 = 25 -> 0,25 · 100/60 = 1,67 -> 0,01
#   R=200: 200/0,5 = 400 -> 4,00 · 200/1,25 = 160 -> 1,60 · 200/4 = 50 -> 0,50 · 200/60 = 3,33 -> 0,03
GOLD_LOTS = [[1.00, 0.40, 0.12, 0.01], [2.00, 0.80, 0.25, 0.01], [4.00, 1.60, 0.50, 0.03]]
GOLD_F = [0.005, 0.01, 0.02]


def test_ac4_2_golden_a_mano_de_la_receta_riesgos_50_125_400_y_6000():
    leg = _fake_leg("ORO.sqx", GOLD_RISKS)
    lots, r, warns = _recipe_of(leg, GOLD_F, 10000.0)
    np.testing.assert_array_equal(lots, GOLD_LOTS)                         # exactos: los lotes que entran al motor
    assert r["risked_money_by_f"] == [50.0, 100.0, 200.0]
    assert r["f_grid"] == GOLD_F and r["account_size"] == 10000.0 and r["currency"] == "USD" and r["mm"] == "FixedAmount"
    rl = r["legs"][0]
    assert rl["max_lot_by_f"] == [1.00, 2.00, 4.00]                          # el mayor de cada fila
    assert rl["decimals"] == 2 and rl["lot_step"] == rl["mmStep"] == rl["min_lot"] == rl["LotsIfNoMM"] == 0.01
    assert rl["mmMultiplier"] == 1
    # frac_min_lot: X < riesgo·mín <=> 50 < [0,5 · 1,25 · 4 · 60]: solo la de 6000 a R=50 (1 de 4); con R=100 llega al mínimo
    # sin que haya que subirlo (100 > 60), aunque el lote sea el mínimo
    assert rl["frac_min_lot_by_f"] == [0.25, 0.0, 0.0]
    # riesgo real p95 = p95(n·riesgo/A)·100 (en % de la cuenta). R=50: n·riesgo = [50, 50, 48, 60] -> 0,5 · 0,5 · 0,48 · 0,6 %,
    # p95 por interpolación lineal (posición 2,85 de [0,48 0,5 0,5 0,6]) = 0,5 + 0,85·0,1 = 0,585; R=100: [100 100 100 60] -> 1,0 %;
    # R=200: [200 200 200 180] -> 2,0 %
    assert rl["p95_riesgo_real_pct_by_f"] == pytest.approx([0.585, 1.0, 2.0], abs=1e-12)
    assert rl["ficha_max_lot"] is None and rl["max_lot_excede_ficha_by_f"] == [False, False, False] and warns == []
    assert (rl["pata"], rl["symbol"], rl["broker"], rl["mm_backtest"], rl["paridad"]) == (
        "ORO.sqx", "SIM_ORO.sqx", "axi", None, None)
    assert [n["id"] for n in r["notes"]] == ["receta_ea", "fichas_broker", "axi_sqx_flotante"]


def test_ac4_2_max_lot_excede_la_ficha_es_estricto_y_avisa_con_la_pata_y_las_f():
    """Con ``ficha_max_lot`` = 1,5: ``max_lot_by_f`` = [1, 2, 4] -> [False, True, True] y un aviso (el EA toparía en el máximo
    del bróker y la app no). Con 2,0 el segundo ya no excede (es ``>``, no ``>=``); sin máximo en la ficha, nunca."""
    _, r, warns = _recipe_of(_fake_leg("ORO.sqx", GOLD_RISKS, ficha_max_lot=1.5), GOLD_F, 10000.0)
    rl = r["legs"][0]
    assert rl["ficha_max_lot"] == 1.5 and rl["max_lot_excede_ficha_by_f"] == [False, True, True]
    assert all(type(b) is bool for b in rl["max_lot_excede_ficha_by_f"])
    assert len(warns) == 1
    assert "ORO.sqx" in warns[0] and "1.00%, 2.00%" in warns[0] and "(4)" in warns[0] and "(1.5)" in warns[0]
    assert "el EA toparía en ese máximo y la app no" in warns[0]
    _, r2, w2 = _recipe_of(_fake_leg("ORO.sqx", GOLD_RISKS, ficha_max_lot=2.0), GOLD_F, 10000.0)
    assert r2["legs"][0]["max_lot_excede_ficha_by_f"] == [False, False, True] and "2.00%" in w2[0] and "1.00%" not in w2[0]
    _, r3, w3 = _recipe_of(_fake_leg("ORO.sqx", GOLD_RISKS, ficha_max_lot=None), GOLD_F, 10000.0)
    assert r3["legs"][0]["max_lot_excede_ficha_by_f"] == [False] * 3 and w3 == []
    # una pata con exceso y otra sin él: un aviso solo de la primera
    a, b = _fake_leg("A.sqx", GOLD_RISKS, ficha_max_lot=1.5), _fake_leg("B.sqx", GOLD_RISKS, ficha_max_lot=100.0)
    la, da = main._lots_for_leg(a, np.array(GOLD_F), {}, 0.005, 10000.0, None)
    lb, db = main._lots_for_leg(b, np.array(GOLD_F), {}, 0.005, 10000.0, None)
    _, w4 = main._sqx_recipe([a, b], {"A.sqx": la, "B.sqx": lb}, {"A.sqx": da, "B.sqx": db}, GOLD_F, 10000.0, "axi")
    assert len(w4) == 1 and "A.sqx" in w4[0] and "B.sqx" not in w4[0]


def test_ac4_2_futuros_paso_1_decimales_0_y_el_tope_de_contratos_que_ya_trae_el_array():
    """Futuros (``lot_step = min_lot = 1``): ``decimals = 0``. Con el tope de contratos del perfil (``cap``) los lotes ya
    llegan topados al motor y ``MaxLots`` = el mayor de esos (a mano): riesgos de 40, 7 y 10 $ por contrato, A = 10.000,
    R = [50, 100, 200] -> floor(R/riesgo): R=50 [1, 7, 5] · R=100 [2, 14, 10] · R=200 [5, 28, 20], que con tope 20 queda
    [5, 20, 20]; el máximo por f es [7, 14, 20] con tope y [7, 14, 28] sin él."""
    leg = _fake_leg("MNQ.sqx", [40.0, 7.0, 10.0], step=1.0, min_lot=1.0, asset="futures")
    lots, r, _ = _recipe_of(leg, GOLD_F, 10000.0, cap=20.0)
    np.testing.assert_array_equal(lots, [[1.0, 7.0, 5.0], [2.0, 14.0, 10.0], [5.0, 20.0, 20.0]])
    rl = r["legs"][0]
    assert (rl["lot_step"], rl["min_lot"], rl["decimals"], rl["mmStep"], rl["LotsIfNoMM"]) == (1.0, 1.0, 0, 1.0, 1.0)
    assert rl["max_lot_by_f"] == [7.0, 14.0, 20.0] and rl["broker"] == "ftmo_futures"
    assert rl["frac_min_lot_by_f"] == [0.0, 0.0, 0.0]            # R=50 < 40·1? no; R/riesgo >= 1 en todas
    _, r_sin, _ = _recipe_of(leg, GOLD_F, 10000.0, cap=None)
    assert r_sin["legs"][0]["max_lot_by_f"] == [7.0, 14.0, 28.0]


def test_ac4_2_golden_a_mano_de_punta_a_punta_en_la_lente_axi(tmp_path, spy):
    """El mismo golden por el endpoint (``.sqx`` de oro de AXI con riesgos de 50, 125, 400 y 6.000 $ por lote, 10 operaciones de
    cada, A = 10.000 y f = 0,5 · 1 · 1,5 · 2 %): lotes del motor, ``risked_money_by_f``, ``max_lot_by_f``, ``frac_min_lot_by_f``
    y ``p95_riesgo_real_pct_by_f`` calculados a mano."""
    p = _hand_world(tmp_path, 10000.0)
    load = main.portfolio_load(main.PortfolioLoadRequest(legs=[p], account_size=10000.0))
    assert load["ok"], load
    out = main.simulate_axi(main.AxiSimRequest(portfolio_id=load["portfolio_id"], stage="seed", f_lo=0.005, f_hi=0.02,
                                               f_step=0.005, n_paths=100))
    r = out["meta"]["recipe"]
    assert r["account_size"] == 10000.0 and r["f_grid"] == pytest.approx([0.005, 0.01, 0.015, 0.02], rel=1e-12)
    assert r["risked_money_by_f"] == pytest.approx([50.0, 100.0, 150.0, 200.0], rel=1e-12)
    # f = 1,5 %: R = 150 -> 150/0,5 = 300 -> 3,00 · 150/1,25 = 120 -> 1,20 · 150/4 = 37,5 -> 0,37 · 150/60 = 2,5 -> 0,02
    lots = spy["strats"]["HAND.sqx"].lots
    hand = np.array(GOLD_LOTS[:2] + [[3.00, 1.20, 0.37, 0.02]] + GOLD_LOTS[2:])           # (4 f, 4 riesgos distintos)
    np.testing.assert_array_equal(lots, np.tile(hand, (1, 10)))                           # 40 operaciones: el patrón x10
    rl = r["legs"][0]
    assert (rl["pata"], rl["symbol"], rl["broker"], rl["ficha_max_lot"]) == ("HAND.sqx", "XAUUSD_axi", "axi", 20.0)
    assert rl["max_lot_by_f"] == [1.0, 2.0, 3.0, 4.0] and rl["decimals"] == 2
    assert rl["frac_min_lot_by_f"] == [0.25, 0.0, 0.0, 0.0]                              # 10 de 40 al subir al mínimo (R=50)
    # p95 de 40 valores (posición 37,05): R=50 -> n·riesgo/A = [0,5 %·10, 0,5 %·10, 0,48 %·10, 0,6 %·10] -> 0,6; R=100 -> 1,0;
    # R=150 -> 1,5; R=200 -> 2,0 (siempre el riesgo real del lote mínimo en las de 6.000, o el 100 % de X)
    assert rl["p95_riesgo_real_pct_by_f"] == pytest.approx([0.6, 1.0, 1.5, 2.0], rel=1e-9)
    assert rl["max_lot_excede_ficha_by_f"] == [False] * 4


HAND_DISTS = (0.5, 1.25, 4.0, 60.0)      # |open − SL| en $ (el oro de AXI vale 100 $/punto): riesgos de 50, 125, 400 y 6.000 $/lote


def _hand_world(tmp_path, capital: float, n_days: int = 40, name: str = "HAND.sqx") -> str:
    """Un ``.sqx`` de ``XAUUSD_axi`` (ficha OFICIAL) con una orden corta al día (10:00-11:30, 0,1 lotes, OHLC plano a 2000) cuyo
    SL va a ``HAND_DISTS[k % 4]`` del precio (todas las distancias exactas en float32): el riesgo por lote es EXACTO."""
    T = _sib("test_axi_sqx.py")
    comm = symbols.resolve("XAUUSD_axi").commission["per_lot_rt"]
    rows, trades = [], []
    for k, d in enumerate(pd.bdate_range("2023-01-02", periods=n_days)):
        ot = pd.Timestamp(d) + pd.Timedelta(hours=10)
        ct = ot + pd.Timedelta(minutes=90)
        rows += [(ts, 2000.0, 2000.0, 2000.0, 2000.0) for ts in pd.date_range(ot, ct, freq="min")]
        trades.append(dict(open=ot.to_pydatetime(), close=ct.to_pydatetime(), lots=T.LOTS, open_px=2000.0,
                           close_px=2000.0, sl_px=2000.0 + HAND_DISTS[k % 4], pnl=-comm * T.LOTS, mae=5.0, flag=1))
    odir = tmp_path / "ohlc_axi"
    odir.mkdir(parents=True, exist_ok=True)
    with open(odir / "XAUUSD_axi.csv", "w", encoding="utf-8") as fh:
        for ts, o, h, l, c in rows:
            fh.write(f"{ts:%Y.%m.%d},{ts:%H:%M},{o:.2f},{h:.2f},{l:.2f},{c:.2f},100\n")
    return str(T.write_sqx(tmp_path / "extra" / "bank" / name, trades, symbol="XAUUSD_axi", instrument="XAUUSD_mt5_axi",
                           pv=100.0, tick=0.01, capital=capital))


def test_ac4_2_exceso_de_punta_a_punta_con_el_maximo_oficial_de_la_ficha_y_el_aviso_en_warnings(tmp_path):
    """Oro de AXI (ficha oficial, ``max_lot`` = 20) con A = 100.000: las órdenes de 50 $/lote dan 10 · 20 · 30 · 40 lotes a
    f = 0,5 · 1 · 1,5 · 2 % -> [False, False, True, True] (a f = 1 % el máximo es EXACTAMENTE 20: no excede) y un aviso."""
    p = _hand_world(tmp_path, 100000.0)
    load = main.portfolio_load(main.PortfolioLoadRequest(legs=[p], account_size=100000.0))
    out = main.simulate_axi(main.AxiSimRequest(portfolio_id=load["portfolio_id"], stage="seed", f_lo=0.005, f_hi=0.02,
                                               f_step=0.005, n_paths=100))
    rl = out["meta"]["recipe"]["legs"][0]
    assert rl["max_lot_by_f"] == [10.0, 20.0, 30.0, 40.0] and rl["ficha_max_lot"] == 20.0
    assert rl["max_lot_excede_ficha_by_f"] == [False, False, True, True]
    avisos = [w for w in out["warnings"] if w.startswith("Receta para el EA")]
    assert len(avisos) == 1 and "HAND.sqx" in avisos[0] and "1.50%, 2.00%" in avisos[0]
    assert "(40)" in avisos[0] and "(20)" in avisos[0] and "el EA toparía en ese máximo y la app no" in avisos[0]


# ---------------------------------------------------------------------------
# Auxiliares de la receta
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step, esperado", [(0.01, 2), (0.1, 1), (1.0, 0), (1, 0), (0.05, 2), (0.001, 3), (0.5, 1),
                                            (0.25, 2), (10.0, 0), (1e-05, 5), (0.0001, 4)])
def test_decimals_son_los_decimales_de_lot_step(step, esperado):
    assert main._lot_step_decimals(step) == esperado


def test_max_lot_se_redondea_al_alza_al_paso_y_quita_el_ruido_de_coma_flotante():
    f = main._recipe_max_lot
    assert f(np.array([0.1 + 0.2, 0.1]), 0.01) == 0.3                       # 0,30000000000000004 -> 0,3
    assert f(np.array([0.07]), 0.01) == 0.07 and f(np.array([4.0, 1.0]), 0.01) == 4.0
    assert f(np.array([2.0000000000000004]), 1.0) == 2.0 and f(np.array([3.0]), 1.0) == 3.0
    assert f(np.array([2.5]), 1.0) == 3.0                                    # al alza (un tope que no es múltiplo del paso)
    assert f(np.array([0.01, 0.02]), 0.01) == 0.02
    assert f(np.array([123.45, 7.0]), 0.01) == 123.45


def test_json_keep_bool_conserva_los_booleanos_y_limpia_numpy_y_nan():
    got = main._json_keep_bool({"a": [True, False, np.bool_(True)], "b": np.float64(0.5), "c": float("nan"),
                                "d": (np.int64(3), None, "x"), "e": np.array([1.0, 2.0])})
    assert got == {"a": [True, False, True], "b": 0.5, "c": None, "d": [3, None, "x"], "e": [1.0, 2.0]}
    assert type(got["a"][0]) is bool and type(got["a"][2]) is bool and type(got["b"]) is float
    assert main._jsonable([True]) == [1] and type(main._jsonable([True])[0]) is int       # lo que NO se quiere en la receta


# ---------------------------------------------------------------------------
# AC4.3 (E2): la receta equivale a la simulación
# ---------------------------------------------------------------------------
N_LEGS = 500
N_PER_FAMILY = 10
_TICKS = [(100000.0, 1e-05), (100.0, 0.01), (20.0, 0.1), (2.0, 0.25), (1.0, 1.0), (12500.0, 1e-04)]   # (pv, tick)
_RHOS = [0.0, 1e-13, -1e-13, 1e-11, -1e-11, 1e-10, -1e-10, 1e-9, -1e-9, 5e-9, -5e-9, 1e-8, -1e-8, 1e-7, -1e-7]
_UMIN = [1.0, 1.0000001, 0.9999999, 1.5, 2.0, 10.0, 100.0, 1000.0]
_LOT_PARAMS = [(0.01, 0.01), (0.01, 0.01), (0.01, 0.1), (0.1, 0.1), (0.05, 0.05), (0.001, 0.01)]     # (paso, mínimo) CFD
_ACCOUNTS = [5000.0, 10000.0, 50000.0, 100000.0, 150000.0]
_FS = [0.0025, 0.005, 0.0075, 0.01, 0.0115, 0.015, 0.02, 0.03]


Q_GEN = 1.0e6        # el generador no produce órdenes con más de este nº de pasos (el comparador exige q < Q_MAX = 2e6)


def _risks(rng, step: float, r_mid: float, r_max: float) -> np.ndarray:
    """Riesgos por lote (USD) de una pata: 4 familias de 10 órdenes. (1) rejilla de ticks, ``pv·(n·tick)`` con el ruido de
    coma flotante de los datos reales; (2) log-uniformes con hasta 7 decimales; (3) FRONTERAS de paso, ``R/(k·paso)`` con una
    perturbación relativa diminuta (el cociente cae a ~1e-13..1e-7 de un entero); (4) AL MÍNIMO: ``R/paso · u`` con
    ``u ≈ 1`` y mayores (el lote calculado redondea a 0 o a un solo paso). Ninguna da más de ``Q_GEN`` pasos a la f mayor
    (``r_max``): el riesgo mínimo se sube a ``r_max/(Q_GEN·paso)``, en la rejilla de 7 decimales."""
    out: list[float] = []
    for _ in range(N_PER_FAMILY):
        pv, tick = _TICKS[int(rng.integers(len(_TICKS)))]
        out.append(pv * (int(rng.integers(1, 4000)) * tick))
    for _ in range(N_PER_FAMILY):
        out.append(max(1e-3, round(10 ** float(rng.uniform(0.0, 3.8)), int(rng.integers(0, 8)))))
    for _ in range(N_PER_FAMILY):
        k = int(rng.integers(1, 300))
        out.append(round(r_mid / (k * step) * (1.0 + float(rng.choice(_RHOS))), 7))
    for _ in range(N_PER_FAMILY):
        out.append(round(r_mid / step * float(rng.choice(_UMIN)), 7))
    return np.maximum(np.array(out), round(r_max / (Q_GEN * step), 7) + 1e-7)


def _run_property(app_lots, seed: int = 20261020, n_legs: int = N_LEGS) -> dict:
    """Recorre ``n_legs`` patas aleatorias (CFD y futuros, con y sin tope) y compara la receta con el oráculo en cada
    orden y f. ``app_lots(leg, f_grid, account, cap)`` da ``(lotes, diag)``."""
    rng = np.random.default_rng(seed)
    stats = _new_stats()
    for k in range(n_legs):
        futures = k % 5 == 4
        step, mn = (1.0, 1.0) if futures else _LOT_PARAMS[int(rng.integers(len(_LOT_PARAMS)))]
        account = float(rng.choice(_ACCOUNTS))
        f_grid = sorted(float(x) for x in rng.choice(_FS, size=3, replace=False))
        cap = float(rng.choice([3.0, 10.0, 50.0])) if (futures and k % 2 == 0) else None
        risks = _risks(rng, step, f_grid[1] * account, f_grid[-1] * account)
        leg = _fake_leg(f"L{k}.sqx", risks, step=step, min_lot=mn, asset="futures" if futures else "cfd")
        lots, diag = app_lots(leg, f_grid, account, cap)
        recipe, _ = main._sqx_recipe([leg], {leg["name"]: lots}, {leg["name"]: diag}, f_grid, account, "axi")
        _compare_with_oracle(recipe["legs"][0], recipe, risks, lots, stats, cap=cap)
    return stats


def _app_lots(leg, f_grid, account, cap):
    return main._lots_for_leg(leg, np.asarray(f_grid, dtype=float), {}, 0.005, account, cap)


def test_ac4_3_la_receta_equivale_a_la_simulacion_propiedad_con_el_oraculo_del_ea(capsys):
    """AC4.3 (E2). 500 patas aleatorias (CFD con distintos pasos y mínimos; futuros con y sin tope de contratos) × 3 f × 40
    órdenes = 60.000 casos con RNG fijo (``hypothesis`` no está en el entorno). El oráculo sigue el orden de operaciones del EA
    (``oneLot`` a 7 decimales, ``R/oneLot``, ``roundDown`` al paso y a ``Decimals``, ``LotsIfNoMM``, ``MaxLots``, mínimo) y usa
    SOLO los parámetros de la receta. Igualdad EXACTA con los lotes de la app cuando ``R/(oneLot·mmStep)`` dista más de 1e-6 de
    un entero; en las fronteras, diferencia ≤ un paso (se cuentan y se imprimen: ``-s``).

    H1: en TODAS las fronteras la app queda ≥ el EA (la diferencia, si la hay, es siempre la app un paso por ENCIMA: el
    ``+1e-9`` de su floor; la simulación nunca subestima el lote del EA). H2: el EA sin el ``MaxLots`` de la receta nunca
    supera ``max_lot_by_f`` (fuera de fronteras ni en ellas), salvo donde la app también recorta por el tope de contratos del
    perfil, que la receta recoge en ``MaxLots`` por diseño."""
    stats = _run_property(_app_lots)
    with capsys.disabled():
        print(f"\n[AC4.3] órdenes={stats['orders']} exactas={stats['exact']} fronteras={stats['frontier']} "
              f"(con un paso de diferencia: {stats['frontier_diff']}, de ellas con la app por encima: "
              f"{stats['frontier_app_above']}; con la app por debajo del EA: "
              f"{stats['frontier_diff'] - stats['frontier_app_above']}) al mínimo (q<1)={stats['min_lot']}")
        print(f"[AC4.3] MaxLots corta al EA (ea_sin_tope > max_lot_by_f): fuera de fronteras={stats['muerde_exact']}, en "
              f"fronteras={stats['muerde_frontier']}; de ellas por el tope de contratos del perfil (por diseño)="
              f"{stats['maxlots_tope_perfil']}; sin explicar={stats['muerde_exact'] + stats['muerde_frontier'] - stats['maxlots_tope_perfil']}")
    assert stats["orders"] >= 2000 and stats["orders"] == N_LEGS * 3 * 4 * N_PER_FAMILY
    assert stats["exact"] + stats["frontier"] == stats["orders"]
    # el generador cubre de verdad lo que dice: casos al mínimo y casos en la frontera (y la mayoría, lejos de ella)
    assert stats["min_lot"] >= 2000 and stats["frontier"] >= 2000 and stats["exact"] >= 0.5 * stats["orders"]
    # H1: toda diferencia de las fronteras es la app POR ENCIMA del EA (el comparador ya falla con cualquiera por debajo)
    assert 0 < stats["frontier_diff"] == stats["frontier_app_above"] <= stats["frontier"]
    # H2: todo mordisco de MaxLots al EA (fuera y dentro de las fronteras) es el del tope de contratos del perfil, y el
    # generador lo ejercita (patas de futuros con tope); en las patas sin tope, ninguno
    assert stats["muerde_exact"] + stats["muerde_frontier"] == stats["maxlots_tope_perfil"] > 0


def test_ac4_3_el_arnes_detecta_una_app_que_redondea_mal(monkeypatch):
    """Sensibilidad: la misma propiedad FALLA si la app redondea al más cercano en vez de hacia abajo (o sube de más): no es
    tautológica. H7: la función falsa aplica el MISMO tope de contratos que la real, para que el fallo sea del redondeo."""
    def al_mas_cercano(risk_per_micro, f_grid, account_size, cap=None, comm_per_micro=None, *, lot_step=1.0, min_lot=1.0):
        lots, diag = fixed_risk_micros(risk_per_micro, f_grid, account_size, cap=cap, comm_per_micro=comm_per_micro,
                                       lot_step=lot_step, min_lot=min_lot)
        risk = np.asarray(risk_per_micro, dtype=float)
        bad = np.empty_like(lots)
        for i, f in enumerate(np.atleast_1d(np.asarray(f_grid, dtype=float))):
            bad[i] = np.maximum(min_lot, np.round(np.round(f * account_size / (risk * lot_step)) * lot_step, 10))
            if cap is not None:
                bad[i] = np.minimum(float(cap), bad[i])
        return bad, diag

    monkeypatch.setattr(main, "fixed_risk_micros", al_mas_cercano)
    with pytest.raises(AssertionError, match="orden"):
        _run_property(_app_lots, n_legs=40)
    # y el fallo es del redondeo, no del tope: las patas CFD (sin tope) fallan solas
    with pytest.raises(AssertionError, match="orden"):
        _run_property(_app_lots, n_legs=4)             # las 4 primeras patas son CFD (la de futuros es la 5.ª, k % 5 == 4)


def test_ac4_3_el_arnes_detecta_un_maxlots_que_corta(monkeypatch):
    """Y falla si ``max_lot_by_f`` fuera menor que el lote máximo simulado (un ``MaxLots`` que sí muerde)."""
    real = main._recipe_max_lot
    monkeypatch.setattr(main, "_recipe_max_lot", lambda row, step: real(row, step) - step)
    with pytest.raises(AssertionError):
        _run_property(_app_lots, n_legs=40)


def test_ac4_3_h1_el_arnes_detecta_una_app_sin_su_epsilon_que_deja_la_app_por_debajo_del_ea_en_una_frontera(monkeypatch):
    """H1: sin el ``+1e-9`` del floor de la app (``futures_legs._FLOOR_EPS = 0``), en las fronteras el cociente cae a veces
    por debajo de un entero por ruido de coma flotante y la app abre un paso MENOS que el EA: el comparador lo detecta (fuera
    de fronteras no cambia nada). Es el aserto que impide que la simulación subestime el lote del EA."""
    from src.risk import futures_legs
    assert futures_legs._FLOOR_EPS == 1e-9
    monkeypatch.setattr(futures_legs, "_FLOOR_EPS", 0.0)
    with pytest.raises(AssertionError, match="POR DEBAJO del EA"):
        _run_property(_app_lots, n_legs=200)


def test_ac4_3_h2_el_arnes_ve_el_mordisco_de_maxlots_que_el_tope_ocultaba():
    """H2: ``MaxLots`` recorta al EA al mismo lote que tiene la app y, con el tope puesto, ``ea == app`` no lo ve: la app abre
    0,50 (un paso menos de lo que el EA quiere, 0,51) y ``MaxLots`` = 0,50 hace que el EA acabe también en 0,50. Solo el
    lote del EA SIN tope lo delata."""
    risk = 50.0 / 0.5123                                  # R = 50 -> q = R/(riesgo·0,01) = 51,23: lejos de la frontera, el EA abre 0,51
    leg = {"mmStep": 0.01, "decimals": 2, "min_lot": 0.01, "LotsIfNoMM": 0.01, "mmMultiplier": 1, "lot_step": 0.01,
           "max_lot_by_f": [0.50]}
    recipe = {"risked_money_by_f": [50.0]}
    assert _ea_lot(50.0, round(risk, 7), step=0.01, decimals=2, min_lot=0.01, lots_if_no_mm=0.01, max_lots=0.50) == 0.50
    assert _ea_lot(50.0, round(risk, 7), step=0.01, decimals=2, min_lot=0.01, lots_if_no_mm=0.01, max_lots=math.inf) == 0.51
    with pytest.raises(AssertionError, match="la receta le corta el lote"):
        _compare_with_oracle(leg, recipe, [round(risk, 7)], np.array([[0.50]]), _new_stats())
    # con la app en 0,51 y MaxLots = 0,51 (lo que hace la receta de verdad) no hay mordisco ni diferencia
    ok = dict(leg, max_lot_by_f=[0.51])
    stats = _new_stats()
    _compare_with_oracle(ok, recipe, [round(risk, 7)], np.array([[0.51]]), stats)
    assert (stats["exact"], stats["muerde_exact"], stats["maxlots_tope_perfil"]) == (1, 0, 0)


def test_ac4_3_de_punta_a_punta_los_lotes_del_motor_siguen_el_oraculo_en_la_lente_axi(tmp_path, spy):
    """La misma comparación sobre lo que de verdad entra al motor: el riesgo por lote de la pata cargada del ``.sqx``, los lotes
    de ``build_block_records`` (``spy``) y la receta de la meta, en el sintético de ``test_axi_sqx.py`` y en el golden a mano."""
    T = _sib("test_axi_sqx.py")
    stats = _new_stats()

    def run(paths, account, f_lo, f_hi, f_step):
        load = main.portfolio_load(main.PortfolioLoadRequest(legs=paths, account_size=account))
        assert load["ok"], load
        out = main.simulate_axi(main.AxiSimRequest(portfolio_id=load["portfolio_id"], stage="seed", f_lo=f_lo, f_hi=f_hi,
                                                   f_step=f_step, n_paths=100))
        recipe = out["meta"]["recipe"]
        assert recipe["account_size"] == account
        for rl in recipe["legs"]:
            leg = main._load_leg(next(p for p in paths if Path(p).name == rl["pata"]), account, 0.005, with_minutes=True,
                                 synth_mode="daily")
            _compare_with_oracle(rl, recipe, leg["fleg"].risk_per_micro, spy["strats"][rl["pata"]].lots, stats)

    run(T.make_axi_world(tmp_path, multiday=True).paths, T.ACCOUNT, 0.01, 0.04, 0.01)    # 82 + 80 operaciones, SL a 300..1.200 $/lote
    # 40 operaciones de 50..6.000 $/lote con f = 0,5..2 % (después del primero: su OHLC de oro se sobrescribe). A f = 0,5 % las
    # de 6.000 $ dan R/(riesgo·paso) = 0,83: el lote calculado redondea a 0 y el EA abre LotsIfNoMM (= el mínimo)
    run([_hand_world(tmp_path, 10000.0)], 10000.0, 0.005, 0.02, 0.005)
    assert stats["orders"] == 4 * (82 + 80) + 4 * 40 == 808
    assert stats["exact"] + stats["frontier"] == stats["orders"] and stats["min_lot"] == 10


# ---------------------------------------------------------------------------
# AC4.4: la meta cita las notas nuevas por id
# ---------------------------------------------------------------------------
def test_ac4_4_la_meta_cita_las_notas_nuevas_por_id_en_las_dos_lentes_y_no_las_cita_en_csv(tmp_path, monkeypatch):
    ids = {n.id for n in NOTES}
    assert {"receta_ea", "fichas_broker", "axi_sqx_flotante"} <= ids
    assert main.RECIPE_NOTE_IDS == {"propfirm": ("receta_ea", "fichas_broker"),
                                    "axi": ("receta_ea", "fichas_broker", "axi_sqx_flotante")}
    for lens, nids in main.RECIPE_NOTE_IDS.items():
        notes = main._recipe_notes(lens)
        assert [n["id"] for n in notes] == list(nids) and all(set(n) == {"id", "texto"} for n in notes)
        assert [n["texto"] for n in notes] == [_yaml_note_text(i) for i in nids]
    T = _sib("test_axi_sqx.py")
    w = T.make_axi_world(tmp_path, n_days=40)
    meta = T._sim(T._load(w.paths)["portfolio_id"], n_paths=100)["meta"]
    assert [n["id"] for n in meta["recipe"]["notes"]] == ["receta_ea", "fichas_broker", "axi_sqx_flotante"]
    # E18: LotsIfNoMM = lote mínimo se presenta como CONDICIÓN, nunca como «recomendado»; E14: la cuenta del EA en USD
    t = meta["recipe"]["notes"][0]["texto"]
    assert "condición" in t and "recomend" not in t.lower() and "USD" in t
    # las notas viven en el catálogo (lo que pinta «¿Cómo se ha calculado?») con la misma forma que las demás
    cat = main.catalog()
    en_axi = {n["id"]: n for n in cat["model_notes_axi"]}
    assert {"receta_ea", "fichas_broker", "axi_sqx_flotante"} <= set(en_axi)
    assert set(en_axi["receta_ea"]) == {"id", "titulo", "texto", "sesgo"} and en_axi["receta_ea"]["texto"] == t
    assert en_axi["axi_sqx_flotante"]["sesgo"] == "conservador"
    for firm in cat["firms"]:
        assert {"receta_ea", "fichas_broker"} <= {c["id"] for c in firm["caveats"]}, firm["key"]
        assert "axi_sqx_flotante" not in {c["id"] for c in firm["caveats"]}, firm["key"]
    # una cartera CSV no cita ninguna (no hay receta)
    (tmp_path / "csv").mkdir()
    _, load = _csv_world(tmp_path / "csv", monkeypatch)
    out = main.simulate_axi(main.AxiSimRequest(portfolio_id=load["portfolio_id"], stage="seed", f_lo=0.01, f_hi=0.02,
                                               f_step=0.01, n_paths=100))
    assert "recipe" not in out["meta"] and not any(i in json.dumps(out["meta"]) for i in main.RECIPE_NOTE_IDS["axi"])


# ---------------------------------------------------------------------------
# C: filas de carga con la ficha y guarda ``fichas``
# ---------------------------------------------------------------------------
ROW_KEYS_SQX = {"pata", "símbolo", "modo", "trades", "status", "warnings", "sqx", "backtest_precision", "broker",
                "ficha_sin_verificar", "mm_backtest", "paridad_frac_ok"}


def _user_axi_sheet(api, sym="ZZZ_axi", **kw):
    """Una ficha de USUARIO de oro de AXI (añadida al bróker oficial ``axi``) con campos sin verificar."""
    body = dict(broker="axi", sqx_symbol=sym, ticker=sym, point_value_usd=100.0, lot_step=0.01, min_lot=0.01,
                commission={"per_lot_rt": 4.5}, needs_review=["commission", "min_lot"])
    body.update(kw)
    r = api.post("/api/symbols", content=json.dumps(body), headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    return r.json()


def test_c_las_filas_de_las_patas_sqx_llevan_broker_ficha_mm_y_paridad_y_las_claves_de_antes(tmp_path):
    T = _sib("test_axi_sqx.py")
    w = T.make_axi_world(tmp_path, n_days=40)
    res = T._load(w.paths)
    assert res["ok"]
    for row in res["legs"]:
        assert set(row) == ROW_KEYS_SQX and row["status"] == "ok"
        assert row["broker"] == "axi" and row["ficha_sin_verificar"] == []
        assert row["mm_backtest"] is None and row["paridad_frac_ok"] is None         # sin lastSettings.xml
    json.dumps(res, allow_nan=False)
    # las claves de antes siguen con los mismos valores
    assert [(r["pata"], r["símbolo"], r["modo"], r["trades"]) for r in res["legs"]] == [
        ("AXI XAU.sqx", "XAUUSD_axi", "riesgo fijo (Opción A)", 40), ("AXI EUR.sqx", "EURUSD_axi", "riesgo fijo (Opción A)", 40)]
    # lo mismo en las carteras .sqx de FTMO (CFD y futuros): el bróker de cada ficha
    F = _sib("test_sqx_fichas.py")
    for legs, broker in ((F._eur_ftmo_legs(tmp_path), "ftmo_cfd"), (F._mnq_legs(tmp_path), "ftmo_futures")):
        rows = F._load(legs)["legs"]
        assert rows and all(set(r) == ROW_KEYS_SQX and r["broker"] == broker for r in rows), broker
        assert all(r["ficha_sin_verificar"] == [] and r["mm_backtest"] is None and r["paridad_frac_ok"] is None
                   for r in rows)


def test_c_mm_backtest_es_el_metodo_y_paridad_frac_ok_la_fraccion_de_ordenes_que_siguen_la_regla(tmp_path):
    F = _sib("test_sqx_fichas.py")
    root = F._root(tmp_path)
    fa = F.w_eur_ftmo(root / "FA.sqx", F._fixed_amount_trades(), last_settings=F.LAST_FIXED_AMOUNT)
    atr = F.w_eur_ftmo(root / "ATR.sqx", F._fixed_amount_trades(20),
                       last_settings=F.LAST_FIXED_AMOUNT.replace('type="FixedAmount"', 'type="ATRRiskBasedSizing"'))
    sin = F.w_eur_ftmo(root / "SIN.sqx", F._fixed_amount_trades(20))
    alterado = [dict(t) for t in F._fixed_amount_trades()]
    t = alterado[1]
    t["lots"] = round(t["lots"] * 2.0, 10)                                    # un lote alterado: la paridad baja de 1
    t["pnl"] = (t["pnl"] + 5.0 * F._fixed_amount_trades()[1]["lots"]) * 2.0 - 5.0 * t["lots"]
    t["mae"] = t["mae"] * 2.0
    bad = F.w_eur_ftmo(root / "BAD.sqx", alterado, last_settings=F.LAST_FIXED_AMOUNT)
    res = main.portfolio_load(main.PortfolioLoadRequest(legs=[fa, atr, sin, bad], account_size=F.ACCOUNT))
    rows = {r["pata"]: r for r in res["legs"]}
    assert all(r["status"] == "ok" and r["broker"] == "ftmo_cfd" for r in rows.values())
    assert (rows["FA.sqx"]["mm_backtest"], rows["FA.sqx"]["paridad_frac_ok"]) == ("FixedAmount", 1.0)
    assert (rows["ATR.sqx"]["mm_backtest"], rows["ATR.sqx"]["paridad_frac_ok"]) == ("ATRRiskBasedSizing", None)
    assert (rows["SIN.sqx"]["mm_backtest"], rows["SIN.sqx"]["paridad_frac_ok"]) == (None, None)
    assert rows["BAD.sqx"]["mm_backtest"] == "FixedAmount" and 0.9 < rows["BAD.sqx"]["paridad_frac_ok"] < 1.0
    assert rows["BAD.sqx"]["paridad_frac_ok"] == pytest.approx(59 / 60)
    # y los mismos valores salen de la pata cargada (la fuente de la fila)
    legs = {l["name"]: l for l in main._portfolios[res["portfolio_id"]]["legs"]}
    assert legs["FA.sqx"]["paridad"]["frac_ok"] == 1.0 and legs["ATR.sqx"]["mm_backtest"]["method"] == "ATRRiskBasedSizing"


def test_c_guarda_fichas_ok_con_las_fichas_oficiales(tmp_path):
    T = _sib("test_axi_sqx.py")
    res = T._load(T.make_axi_world(tmp_path, n_days=40).paths)
    g = res["guards"]["fichas"]
    assert g["status"] == "ok" and "2 símbolo(s)" in g["detail"] and "sin verificar" in g["detail"]
    assert list(res["guards"]).index("fichas") > list(res["guards"]).index("ventanas")
    # y en una cartera de FTMO (CFD y futuros) también: sus fichas oficiales no tienen nada por verificar
    F = _sib("test_sqx_fichas.py")
    assert F._load(F._eur_ftmo_legs(tmp_path))["guards"]["fichas"]["status"] == "ok"
    assert F._load(F._mnq_legs(tmp_path))["guards"]["fichas"]["status"] == "ok"


def test_c_guarda_fichas_warn_con_la_pata_y_los_campos_de_una_ficha_de_usuario_con_needs_review(tmp_path, api):
    F = _sib("test_sqx_fichas.py")
    T = _sib("test_axi_sqx.py")
    _user_axi_sheet(api)                                      # fichero de usuario en tmp (la fixture ``_aislado``): I4
    assert symbols.resolve("ZZZ_axi").needs_review == ("commission", "min_lot")
    root = F._root(tmp_path)
    z = F.write_sqx(root / "Z.sqx", F.xau_axi(10, 7), symbol="ZZZ_axi", instrument="ZZZ_axi", pv=100.0, tick=0.01,
                    capital=T.ACCOUNT)
    oficial = F.write_sqx(root / "X.sqx", F.xau_axi(12, 8), symbol="XAUUSD_axi", instrument="XAUUSD_mt5_axi", pv=100.0,
                          tick=0.01, capital=T.ACCOUNT)
    res = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(z), str(oficial)], account_size=T.ACCOUNT))
    assert res["ok"], res                                      # un warn nunca impide cargar ni es error
    g = res["guards"]["fichas"]
    assert g["status"] == "warn"
    assert g["detail"].startswith("ficha de ZZZ_axi con campos sin verificar: commission, min_lot (pata: Z.sqx)")
    assert "XAUUSD_axi" not in g["detail"] and "X.sqx" not in g["detail"]            # la oficial no entra
    assert all(x["status"] != "error" for x in res["guards"].values())
    rows = {r["pata"]: r for r in res["legs"]}
    assert rows["Z.sqx"]["ficha_sin_verificar"] == ["commission", "min_lot"] and rows["X.sqx"]["ficha_sin_verificar"] == []
    # con varias patas de la misma ficha salen juntas, y con TODOS los campos por verificar sigue siendo warn (nunca error)
    z2 = F.write_sqx(root / "Z2.sqx", F.xau_axi(14, 9), symbol="ZZZ_axi", instrument="ZZZ_axi", pv=100.0, tick=0.01,
                     capital=T.ACCOUNT)
    res2 = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(z), str(z2)], account_size=T.ACCOUNT))
    g2 = res2["guards"]["fichas"]
    assert g2["status"] == "warn" and g2["detail"].count("ficha de ZZZ_axi") == 1 and "(patas: Z.sqx, Z2.sqx)" in g2["detail"]
    r = api.put("/api/symbols/axi/ZZZ_axi", content=json.dumps({"needs_review": list(symbols.DATA_FIELDS)}),
                headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text                        # editar la ficha vacía la caché de patas (gancho de recarga)
    res3 = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(z)], account_size=T.ACCOUNT))
    assert res3["guards"]["fichas"]["status"] == "warn" and res3["ok"]
    assert res3["legs"][0]["ficha_sin_verificar"] == list(symbols.DATA_FIELDS)
    assert ", ".join(symbols.DATA_FIELDS) in res3["guards"]["fichas"]["detail"]


def test_c_la_guarda_fichas_nunca_es_error_ni_aparece_si_no_hay_patas_cargadas(tmp_path):
    guard = main._fichas_guard
    assert guard([])["status"] == "ok"
    leg = lambda n, nr: {"name": n, "meta": {"symbol": "S"}, "ficha": {"sqx_symbol": "S", "needs_review": nr}}      # noqa: E731
    muchas = guard([leg(f"P{i}", ["commission"]) for i in range(5)])
    assert muchas["status"] == "warn" and "(patas: P0, P1, P2 y 2 más)" in muchas["detail"]
    todo = guard([leg("P", list(symbols.DATA_FIELDS))])
    assert todo["status"] == "warn" and todo["status"] != "error"
    # sin ninguna pata cargada (sin ficha) no hay guarda de fichas, y la cartera ya es no-ok por otra guarda
    F = _sib("test_sqx_fichas.py")
    raro = F.write_sqx(F._root(tmp_path) / "RARO.sqx", F.eur_ftmo(), symbol="SIMBOLO_RARO", instrument="X", pv=100000.0,
                       tick=0.0001)
    res = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(raro)], account_size=F.ACCOUNT))
    assert not res["ok"] and "fichas" not in res["guards"] and res["guards"]["bloques"]["status"] == "error"


def test_c_una_ficha_editada_a_mano_no_sirve_un_aviso_viejo_needs_review_entra_en_la_clave_de_la_pata(tmp_path, api):
    """``needs_review`` no está en la firma de la ficha, pero la pata cacheada lo lleva (``ficha``) y de él salen la fila y la
    guarda de la carga. Editar el fichero de usuario A MANO (sin la API, cuyo gancho de recarga vacía la caché de patas) cambia
    la clave de la pata: no se sirve el aviso viejo."""
    F = _sib("test_sqx_fichas.py")
    T = _sib("test_axi_sqx.py")
    _user_axi_sheet(api)
    z = F.write_sqx(F._root(tmp_path) / "Z.sqx", F.xau_axi(10, 7), symbol="ZZZ_axi", instrument="ZZZ_axi", pv=100.0,
                    tick=0.01, capital=T.ACCOUNT)
    leg1 = main._load_leg(str(z), T.ACCOUNT, 0.005, with_minutes=True, synth_mode="daily")
    assert leg1["ficha"]["needs_review"] == ["commission", "min_lot"]
    assert main._load_leg(str(z), T.ACCOUNT, 0.005, with_minutes=True, synth_mode="daily") is leg1     # sin cambios: la misma
    user_file = tmp_path / "symbols.user.yaml"
    doc = yaml.safe_load(user_file.read_text(encoding="utf-8"))
    doc["symbols"]["axi"]["ZZZ_axi"]["needs_review"] = []
    user_file.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    ns = user_file.stat().st_mtime_ns + 5_000_000_000
    os.utime(user_file, ns=(ns, ns))                                  # mtime distinto sin esperar
    n_keys = len(main._leg_cache)
    res = main.portfolio_load(main.PortfolioLoadRequest(legs=[str(z)], account_size=T.ACCOUNT))
    assert len(main._leg_cache) > n_keys                              # el gancho NO se disparó: hay una pata nueva en la caché
    assert res["legs"][0]["ficha_sin_verificar"] == [] and res["guards"]["fichas"]["status"] == "ok"
    assert any(v is leg1 for v in main._leg_cache.values())           # y la vieja sigue ahí, bajo su clave antigua
