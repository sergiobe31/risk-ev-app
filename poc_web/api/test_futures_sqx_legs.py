"""Fase C del plan float-valleys-ventanas: patas de futuros nativas desde ``.sqx``.

Tests a nivel de FUNCIÓN (sin levantar FastAPI; patrón de
``test_propfirm_futures.py``) con ``.sqx`` SINTÉTICOS escritos en ``tmp_path``
con el writer inverso del formato de ``tests/test_sqx_trades.py`` (importado
por ruta). Cubre:

- C1 catálogo (``kind="sqx_futures"``; un .sqx fuera de la allowlist o ilegible
  no aparece).
- C2 carga (3 .sqx ok; cartera mixta CSV + .sqx → error explícito).
- C3 simulación ``ftmo_futures_growth_50k`` (3 f, 200 paths) sin 409/422,
  ``meta.float_mode == "windows"``, ``rel_low`` no-None en todos los bloques,
  ``sizing_diag``; perfil no-futures → 409; AXI / par / diag. de pata → 409.
- V6 guarda ``corte_1610`` (trade que cruza el corte; cruce de fin de semana).
- DS-H4: ``use_portfolio`` separado de ``trailing_precompute`` (con cartera
  .sqx no salta el ValueError de ``simulate_challenge``).
- Sizing Opción A por ``_lots_for_leg`` (goldens numéricos) y errores del
  extractor propagados como error de pata.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_futures_sqx_legs.py -q
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import poc_web.api.main as main
from poc_web.api.main import (
    AxiSimRequest,
    LegDiagRequest,
    PortfolioLoadRequest,
    PropfirmSimRequest,
    portfolio_leg_diagnostics,
    portfolio_load,
    simulate_axi,
    simulate_propfirm,
    simulate_propfirm_pair,
)
from src.risk.propfirm import build_phase_tape, simulate_challenge

# --- writer inverso del formato .sqx (tests/test_sqx_trades.py) ---------------
_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer",
    Path(__file__).resolve().parents[2] / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx

FUTURES = "ftmo_futures_growth_50k"
CLASSIC = "ftmo_100k"
ACCOUNT = 50000.0
N_DAYS = 100
COMM_RT = 1.12            # USD por micro MNQ (round trip)
PV = 2.0                  # MNQ: USD por punto y micro


def _trades(slot_hour: int, seed: int, n_days: int = N_DAYS) -> list[dict]:
    """Un trade por día hábil, ventana [slot_hour:00, slot_hour+1:30] ET (misma
    sesión 16:10), largo con SL a 10-25 puntos; 60% ganadoras a 1,5R, resto
    cierran en el SL (MAE = pérdida bruta exacta). Comisión 1,12/lote."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    out = []
    for d in days:
        dist = float(rng.choice([10.0, 15.0, 20.0, 25.0]))
        open_px = 18000.0 + 0.25 * float(rng.integers(-40, 40))
        win = rng.random() < 0.6
        sl_px = open_px - dist
        close_px = open_px + 1.5 * dist if win else sl_px
        gross = (close_px - open_px) * PV
        mae = (float(rng.uniform(0.0, 0.8)) * dist * PV) if win else -gross
        out.append(dict(
            open=datetime(d.year, d.month, d.day, slot_hour, 0),
            close=datetime(d.year, d.month, d.day, slot_hour + 1, 30),
            lots=1.0, open_px=open_px, close_px=close_px, sl_px=sl_px,
            pnl=gross - COMM_RT, mae=max(mae, 0.0), flag=1))
    return out


def _write_legs(root: Path, n: int = 3, replace=None) -> list[str]:
    """``n`` patas .sqx sintéticas (slots 10h/12h/14h) → rutas ABSOLUTAS."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n):
        tr = _trades(10 + 2 * k, seed=100 + k)
        if replace and k in replace:
            tr = replace[k](tr)
        paths.append(str(write_sqx(root / f"Strategy 7.{k}.1.sqx", tr,
                                   capital=ACCOUNT)))
    return paths


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path):
    """Cachés de proceso vacías por test + catálogo aislado del disco real."""
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")


@pytest.fixture
def legs(tmp_path):
    return _write_legs(tmp_path / "extra" / "bank")


def _load(legs, firm_key=FUTURES, **kw):
    res = portfolio_load(PortfolioLoadRequest(
        legs=legs, account_size=ACCOUNT, firm_key=firm_key, **kw))
    return res


def _req(res, **kw):
    base = dict(portfolio_id=res["portfolio_id"], firm_key=FUTURES,
                f_lo=0.0025, f_hi=0.0075, f_step=0.0025, n_paths=200,
                n_challenges=5, n_sequences=100, capital0_eur=3000.0,
                fx_eur_usd=1.08)
    base.update(kw)
    return PropfirmSimRequest(**base)


# --- C1: catálogo ------------------------------------------------------------
def test_c1_catalogo_sqx_futuros_con_kind_y_filtro_allowlist(tmp_path):
    # Cambio autorizado en la fase F2 del guardrail entrada-4-bloques (invariante I7): el catálogo lista TODO .sqx
    # legible, con el estado de su ficha (``broker``, ``ficha``, ``motivo``). Un símbolo SIN ficha (aquí EURUSD, que
    # no es el de ninguna ficha) ya no se omite: sale con ``ficha == "sin_ficha"`` para poder darlo de alta. Solo el
    # .sqx ilegible sigue fuera. Antes el filtro era la allowlist de futuros (de ahí el nombre del test, que se
    # conserva).
    extra = tmp_path / "extra"
    fut = write_sqx(extra / "bank" / "Strategy 1.1.1.sqx", _trades(10, 1, 5))
    # CFD: símbolo sin ficha → aparece como sin_ficha (antes: fuera por la allowlist de futuros).
    cfd = write_sqx(extra / "bank" / "Strategy 2.2.2.sqx", _trades(10, 2, 5),
                    symbol="EURUSD", instrument="EURUSD_x", pv=1.0, tick=0.00001)
    junk = extra / "bank" / "roto.sqx"
    junk.write_bytes(b"esto no es un zip")

    legs = main._scan_legs()
    by_name = {l["name"]: l for l in legs}
    assert "Strategy 1.1.1" in by_name
    assert by_name["Strategy 1.1.1"]["kind"] == "sqx_futures"
    assert by_name["Strategy 1.1.1"]["path"] == str(fut)        # raíz externa → abs
    assert by_name["Strategy 1.1.1"]["folder"] == "bank"
    assert set(by_name["Strategy 1.1.1"]) == {"path", "folder", "name", "kind", "asset", "broker", "ficha", "motivo"}
    assert by_name["Strategy 1.1.1"]["asset"] == "futures"
    assert "Strategy 2.2.2" in by_name                            # sin ficha: ahora SÍ aparece
    sin = by_name["Strategy 2.2.2"]
    assert sin["ficha"] == "sin_ficha" and sin["broker"] is None and sin["asset"] is None
    assert "roto" not in by_name                                  # ilegible fuera
    assert any(l["path"] == str(cfd) for l in legs)
    # los CSV del catálogo (si los hay) conservan kind="csv"
    assert all(l["kind"] in ("csv", "sqx_futures") for l in legs)


def test_c1_catalogo_incluye_strategies_dir_del_repo(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo_strategies"
    p = write_sqx(repo_dir / "Strategy 3.3.3.sqx", _trades(10, 3, 5))
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", repo_dir)
    legs = [l for l in main._scan_legs() if l["kind"] == "sqx_futures"]
    assert [l["path"] for l in legs] == [str(p)]


def test_c1_simbolo_cacheado_por_ruta_y_mtime(tmp_path):
    p = write_sqx(tmp_path / "extra" / "a.sqx", _trades(10, 4, 3))
    assert main._sqx_symbol(p) == "@micro_nasdaq"
    assert (str(p), main._mtime(p)) in main._sqx_symbol_cache


# --- C2: carga ---------------------------------------------------------------
def test_c2_carga_tres_sqx_ok(legs):
    res = _load(legs)
    assert res["ok"], res
    assert [l["status"] for l in res["legs"]] == ["ok"] * 3
    assert all(l["modo"] == "riesgo fijo (Opción A)" for l in res["legs"])
    assert all(l["símbolo"] == "@micro_nasdaq" for l in res["legs"])
    assert all(l["trades"] == N_DAYS for l in res["legs"])
    g = res["guards"]
    assert g["tipo_patas"]["status"] == "ok"
    # no aplican las guardas de valle/M1 (se marcan ok con detalle explícito)
    assert g["valles"]["status"] == "ok" and "no aplica" in g["valles"]["detail"]
    assert g["m1"]["status"] == "ok" and "no aplica" in g["m1"]["detail"]
    assert g["ventanas"]["status"] == "ok"
    assert "backtest_precision=1" in g["ventanas"]["detail"]
    assert g["corte_1610"]["status"] == "ok"
    assert g["corte_1610"]["detail"].startswith("0/")
    assert g["capital"]["status"] == "ok"
    # sin aviso de universo FX / Fase 7 ni guarda overnight de Apex
    assert not any("Fase 7" in w or "universo FX" in w for w in res["warnings"])
    assert "overnight" not in g
    pf = main._portfolios[res["portfolio_id"]]
    assert pf["kind"] == "sqx_futures"
    assert all(l["kind"] == "sqx_futures" and l["tl"] is None for l in pf["legs"])


def test_c2_cartera_mixta_csv_y_sqx_es_error_explicito(legs):
    # El tipo se decide por la extensión ANTES de cargar nada: la ruta CSV no
    # necesita existir (construir un CSV orderstocsv válido no aporta a esta guarda).
    res = _load([legs[0], "risk_app/inputs/trades/x/pata_cfd.csv"])
    assert not res["ok"]
    assert res["guards"]["tipo_patas"]["status"] == "error"
    assert "mixta" in res["guards"]["tipo_patas"]["detail"]
    assert [l["status"] for l in res["legs"]] == ["error", "error"]
    assert all("mixta" in l["error"] for l in res["legs"])
    # la cartera registrada no sirve para simular (ninguna pata cargada)
    with pytest.raises(HTTPException) as ei:
        main._propfirm_setup(_req(res))
    assert ei.value.status_code == 409


def test_c2_extractor_error_es_error_de_pata(tmp_path):
    bad = write_sqx(tmp_path / "extra" / "Strategy 8.8.8.sqx", _trades(10, 5, 5),
                    symbol="EURUSD", instrument="EURUSD_x", pv=1.0, tick=0.00001)
    res = _load([str(bad)])
    assert not res["ok"]
    leg = res["legs"][0]
    assert leg["status"] == "error"
    assert "allowlist" in leg["error"]          # SQXFutureLegError → ValueError con el mensaje
    with pytest.raises(ValueError, match="allowlist"):
        main._load_leg(str(bad), ACCOUNT, 0.005)


def test_c2_aviso_capital_distinto_es_informativo(tmp_path):
    p = write_sqx(tmp_path / "extra" / "Strategy 9.9.1.sqx", _trades(10, 6, 5),
                  capital=100000.0)
    leg = main._load_leg(str(p), ACCOUNT, 0.005)
    assert leg["warn"] and "capital" in leg["warn"]
    assert "account_size" in leg["warn"]
    assert leg["sl_exact"] is None and leg["max_lots"] == float("inf")


def test_trades_de_duracion_cero_entran_en_bloques(tmp_path):
    """PLAN-DESVIO de ``_load_sqx_leg``: un trade con open_ts == close_ts se
    perdería en silencio de los bloques flat-flat; el +1 ns lo conserva."""
    def with_zero(tr):
        for i in (5, 6):
            tr[i] = dict(tr[i], close=tr[i]["open"])
        return tr
    legs = _write_legs(tmp_path / "extra" / "x", n=1, replace={0: with_zero})
    res = _load(legs)
    assert any("duración 0" in w for w in res["warnings"])
    assert res["guards"]["cobertura"]["status"] == "ok"      # 100%: ninguno perdido
    assert res["coverage"] == pytest.approx(1.0)
    leg = main._portfolios[res["portfolio_id"]]["legs"][0]
    assert np.all(leg["s"].close_t > leg["s"].open_t)


# --- eje de sesión (W3) y clave de caché --------------------------------------
def test_labeler_ny_con_cut_y_cache_por_tz_y_cut(tmp_path):
    tr = [dict(_trades(10, 7, 1)[0],
               open=datetime(2023, 1, 3, 15, 0), close=datetime(2023, 1, 3, 16, 30)),
          dict(_trades(10, 8, 1)[0],
               open=datetime(2023, 1, 4, 10, 0), close=datetime(2023, 1, 4, 16, 9))]
    p = write_sqx(tmp_path / "extra" / "Strategy 6.6.6.sqx", tr)
    cal = main._load_leg(str(p), ACCOUNT, 0.005)                       # sin eje
    ses = main._load_leg(str(p), ACCOUNT, 0.005,
                         day_labeler_tz="Europe/Athens",               # se ignora
                         day_labeler_cut="16:10")
    ses18 = main._load_leg(str(p), ACCOUNT, 0.005, day_labeler_tz="x",
                           day_labeler_cut="18:00")
    assert cal is not ses and ses is not ses18
    # regla aritmética: fecha + (hora >= 16:10)
    assert [d.isoformat() for d in cal["s"].close_d] == ["2023-01-03", "2023-01-04"]
    assert [d.isoformat() for d in ses["s"].close_d] == ["2023-01-04", "2023-01-04"]
    assert [d.isoformat() for d in ses18["s"].close_d] == ["2023-01-03", "2023-01-04"]
    # misma clave => misma pata cacheada
    assert main._load_leg(str(p), ACCOUNT, 0.005, day_labeler_tz="y",
                          day_labeler_cut="16:10") is ses


# --- V6: guarda corte_1610 -----------------------------------------------------
def test_v6_trade_que_cruza_el_corte_se_cuenta(tmp_path):
    def with_crossers(tr):
        for i in (3, 9, 20):   # 3 de 100 = 3% > 1%
            d = tr[i]["open"].date()
            tr[i] = dict(tr[i], open=datetime(d.year, d.month, d.day, 15, 0),
                         close=datetime(d.year, d.month, d.day, 17, 30))
        return tr
    legs = _write_legs(tmp_path / "extra" / "x", n=1, replace={0: with_crossers})
    res = _load(legs)
    g = res["guards"]["corte_1610"]
    assert g["status"] == "warn"
    assert g["detail"].startswith("3/100 ")
    assert "16:10" in g["detail"]


def test_v6_cruce_de_fin_de_semana_cuenta_y_1pct_es_ok(tmp_path):
    def with_weekend(tr):
        i = next(k for k, t in enumerate(tr) if t["open"].weekday() == 4)  # viernes
        d = tr[i]["open"]
        tr[i] = dict(tr[i], open=datetime(d.year, d.month, d.day, 15, 0),
                     close=datetime(d.year, d.month, d.day, 15, 0) + timedelta(days=3, hours=-5))
        return tr
    legs = _write_legs(tmp_path / "extra" / "x", n=1, replace={0: with_weekend})
    res = _load(legs)
    g = res["guards"]["corte_1610"]
    assert g["detail"].startswith("1/100 ")
    assert g["status"] == "ok"            # 1% no es > 1%


def test_v6_usa_el_cut_del_perfil_si_viene_firm_key(tmp_path):
    def cross_1700(tr):
        d = tr[0]["open"].date()
        tr[0] = dict(tr[0], open=datetime(d.year, d.month, d.day, 15, 0),
                     close=datetime(d.year, d.month, d.day, 16, 30))
        return tr
    legs = _write_legs(tmp_path / "extra" / "x", n=1, replace={0: cross_1700})
    con_perfil = _load(legs)["guards"]["corte_1610"]               # cut 16:10
    sin_perfil = _load(legs, firm_key="")["guards"]["corte_1610"]  # default 16:10
    assert con_perfil["detail"].startswith("1/100 ")
    assert sin_perfil["detail"].startswith("1/100 ")
    assert "corte 16:10 ET" in sin_perfil["detail"]


# --- sizing Opción A -----------------------------------------------------------
def test_sizing_opcion_a_goldens_via_helper(tmp_path):
    # riesgo por micro = |open-sl|*pv: 10pts->20 USD ; 308,5pts->617 ; 2,75pts->5,5
    base = _trades(10, 9, 3)
    spec = [(10.0, 20.0), (308.5, 617.0), (2.75, 5.5)]
    tr = []
    for t, (dist, _risk) in zip(base, spec):
        t = dict(t, sl_px=t["open_px"] - dist, close_px=t["open_px"] - dist,
                 pnl=-dist * PV - COMM_RT, mae=dist * PV)
        tr.append(t)
    p = write_sqx(tmp_path / "extra" / "Strategy 5.5.5.sqx", tr)
    leg = main._load_leg(str(p), ACCOUNT, 0.005)
    np.testing.assert_allclose(leg["fleg"].risk_per_micro, [20.0, 617.0, 5.5])
    cfg = main._firms[FUTURES]
    lots, diag = main._lots_for_leg(leg, np.array([0.01]), {}, 0.005, ACCOUNT,
                                    main.eval_contract_cap_micros(cfg))
    # X = 0,01 * 50.000 = 500 ; floor(500/20)=25 ; 617 -> 0 -> clamp 1 ; 5,5 -> 90 -> cap 50
    np.testing.assert_array_equal(lots, [[25.0, 1.0, 50.0]])
    d = diag[0]
    assert d["X"] == 500.0
    assert d["frac_clamp_up"] == pytest.approx(1 / 3)
    assert d["frac_capped"] == pytest.approx(1 / 3)
    assert "p95_total_adverse_over_X" in d and "p95_risk_eff_over_X" in d


# --- C3: simulación -------------------------------------------------------------
def test_c3_simulacion_futuros_windows_sin_409_ni_422(legs):
    res = _load(legs)
    assert res["ok"], res
    out = simulate_propfirm(_req(res))
    meta = out["meta"]
    assert meta["float_mode"] == "windows"
    assert meta["sizing"] == "riesgo fijo (Opción A)"
    assert meta["leg_kind"] == "sqx_futures"
    assert meta["funded_model"] == "futures"
    assert meta["n_paths"] == 200 and len(out["reports"]) == 3
    # diag de sizing: por pata y por f (3 f)
    sd = meta["sizing_diag"]
    assert set(sd) == {"Strategy 7.0.1.sqx", "Strategy 7.1.1.sqx", "Strategy 7.2.1.sqx"}
    for diag in sd.values():
        assert len(diag) == 3
        assert {"f", "X", "frac_clamp_up", "p95_risk_eff_over_X", "frac_capped",
                "p95_total_adverse_over_X"} <= set(diag[0])
        assert diag[0]["X"] == pytest.approx(0.0025 * ACCOUNT)
    # etiqueta de la cota
    assert meta["backtest_precision"] == {
        l["pata"]: 1 for l in meta["legs"]}
    assert any("precisión del backtest" in w for w in out["warnings"])
    # sin avisos de universo FX / Fase 7, source_tz del labeler = NY
    assert not any("Fase 7" in w or "universo FX" in w for w in out["warnings"])
    assert meta["source_tz"] == "America/New_York"
    assert meta["day_axis"] == "et_session"
    rep0 = out["reports"][0]
    assert 0.0 <= rep0["p_pass1"] <= 1.0
    assert rep0["frac_cap_violations_eval"] is not None


def test_c3_setup_rel_low_no_none_y_flags_separados(legs, monkeypatch):
    res = _load(legs)
    calls = []
    orig = main.precompute_day_arrays_portfolio

    def spy(*a, **kw):
        calls.append(kw)
        return orig(*a, **kw)

    monkeypatch.setattr(main, "precompute_day_arrays_portfolio", spy)
    ctx = main._propfirm_setup(_req(res))
    # flags separados: el perfil ES trailing, pero la cartera .sqx no usa la
    # infra trailing por minuto y SÍ el camino portfolio.
    assert ctx["trailing"] is True
    assert ctx["use_portfolio"] is True
    assert ctx["float_mode"] == "windows"
    assert ctx["kind"] == "sqx_futures"
    assert len(calls) == 1
    kw = calls[0]
    assert kw["float_mode"] == "windows" and kw["trailing"] is False
    assert set(kw["adverse_by_sym"]) == {0, 1, 2}
    # rel_low no-None en TODOS los bloques de TODA la grilla, rel_low <= rel_closed
    assert ctx["n_no_valley"] == 0
    for per_f in ctx["day_arrays_grid"]:
        assert len(per_f) == ctx["n_blocks"]
        for bda in per_f:
            assert bda.rel_low is not None
            assert np.all(bda.rel_low <= bda.rel_closed + 1e-9)
    assert set(ctx["sizing_diag"]) == {l["name"] for l in ctx["legs_used"]}
    # sin guardas minute_valle/minute_fav ni avisos n_no_valley
    assert not any("sin valle" in w for w in ctx["warnings"])


def test_c3_source_tz_se_ignora_con_warning(legs):
    res = _load(legs)
    ctx = main._propfirm_setup(_req(res, source_tz="Europe/Athens"))
    assert ctx["source_tz"] == "America/New_York"
    assert any("source_tz" in w and "ignorado" in w for w in ctx["warnings"])


def test_ds_h4_use_portfolio_evita_el_valueerror_de_simulate_challenge(legs):
    res = _load(legs)
    ctx = main._propfirm_setup(_req(res))
    tape = build_phase_tape(ctx["idx_matrix"], ctx["idx_ext"], None, None)
    args = (tape, ctx["day_arrays_grid"][0], ctx["durations"], ctx["cfg"])
    # Con portfolio=False (el bug: portfolio=trailing con trailing=False) el
    # dispatch eod_trailing/equity hace raise...
    with pytest.raises(ValueError, match="portfolio"):
        simulate_challenge(*args, portfolio=False)
    # ...con el flag que expone el ctx no.
    oc = simulate_challenge(*args, portfolio=ctx["use_portfolio"])
    assert len(oc.pass1) == ctx["idx_matrix"].shape[0]


def test_c3_perfil_no_futures_con_cartera_sqx_es_409(legs):
    res = _load(legs, firm_key=CLASSIC)
    assert any("no es FTMO Futures" in w for w in res["warnings"])
    with pytest.raises(HTTPException) as ei:
        simulate_propfirm(_req(res, firm_key=CLASSIC))
    assert ei.value.status_code == 409
    assert "FTMO Futures" in ei.value.detail


def test_c3_axi_par_y_diag_de_pata_son_409(legs):
    res = _load(legs)
    pid = res["portfolio_id"]
    with pytest.raises(HTTPException) as e1:
        simulate_axi(AxiSimRequest(portfolio_id=pid))
    assert e1.value.status_code == 409 and ".sqx" in e1.value.detail
    # par: la guarda .sqx va ANTES del 400 de perfiles futures
    with pytest.raises(HTTPException) as e2:
        simulate_propfirm_pair(_req(res))
    assert e2.value.status_code == 409
    with pytest.raises(HTTPException) as e3:
        simulate_propfirm_pair(_req(res, firm_key=CLASSIC))
    assert e3.value.status_code == 409
    with pytest.raises(HTTPException) as e4:
        portfolio_leg_diagnostics(pid, LegDiagRequest(f=0.01))
    assert e4.value.status_code == 409


def test_portfolio_desconocido_sigue_siendo_404():
    with pytest.raises(HTTPException) as e1:
        simulate_axi(AxiSimRequest(portfolio_id="nope"))
    assert e1.value.status_code == 404
    with pytest.raises(HTTPException) as e2:
        portfolio_leg_diagnostics("nope", LegDiagRequest(f=0.01))
    assert e2.value.status_code == 404


def test_sizing_label_modo_nuevo_y_mixto():
    a = {"sizing_mode": "riesgo fijo (Opción A)"}
    assert main._sizing_label([a, a]) == "riesgo fijo (Opción A)"
    assert main._sizing_label([a, {"sizing_mode": "exacto"}]) == "mixto"
    assert main._sizing_label([{"sizing_mode": "exacto"}]) == "exacto"
    assert main._sizing_label([{"sizing_mode": "rápido"}]) == "rápido"
