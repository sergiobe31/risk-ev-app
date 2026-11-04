"""Guardrail ``onboarding-b1``, B1-1: catálogo con ``extended``/``kind``/reglas ``cycle``/
avisos como objetos, ``_bs_explain``, ``data_span_days`` y
``meta.model_context`` (+ ``funded_cut_by_data_frac``) en los tres endpoints.

Guardrail ``bloque-politis-white`` (2026-10-21, E2): ``_bs_explain`` decide SOLO por el modo (``auto`` / ``manual`` /
``fallback_iid``): las cinco ramas de la ACF² (que dependían de una ``BlockSizeCalibration``) pasan a los tres modos nuevos;
el modo automático nunca se etiqueta ``fallback_iid``. ``meta.model_context.block_size`` lleva además ``method``,
``series``, ``n_excluded_terms``, ``opt_m`` y ``m``, y ``bs`` va con un decimal en el texto por corrida.

Las secciones sintéticas no necesitan datos; las de endpoint (smoke) se saltan si no
está la pata fixture ``risk_app/inputs`` (gitignorada).

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_model_context.py -q
"""
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import poc_web.api.main as main
from poc_web.api.main import (
    AxiSimRequest,
    PortfolioLoadRequest,
    PropfirmSimRequest,
    _bs_explain,
    _data_span_days,
    _model_context,
    catalog,
    portfolio_load,
    simulate_axi,
    simulate_propfirm,
    simulate_propfirm_pair,
)
from poc_web.api.model_notes import NOTES, notes_for, render_run

ROOT = Path(__file__).resolve().parents[2]
LEG = "risk_app/inputs/trades/nasdaq_2h/Strategy 1.30.184.csv"
needs_data = pytest.mark.skipif(not (ROOT / LEG).exists(),
                                reason="pata fixture gitignorada no disponible")

# --- I4: `extended` == (_profile_rules de HEAD 6e66324 no es None), 10 perfiles -----
EXTENDED_HEAD = {
    "ftmo_50k": False, "ftmo_50k_swing": False, "ftmo_100k": False,
    "apex_150k_intraday": True,
    "ftmo_futures_growth_50k": True, "ftmo_futures_growth_100k": True,
    "ftmo_futures_growth_150k": True, "ftmo_futures_pro_50k": True,
    "ftmo_futures_pro_100k": True, "ftmo_futures_pro_150k": True,
}
# --- I2: sha256 de json.dumps(_jsonable(_profile_rules(cfg)), sort_keys=True) en HEAD ---
RULES_SHA_HEAD = {
    "apex_150k_intraday": "ad0ce0c572573a0d6f18a2d8711db175f659ff208dc142e230d41b9d72ae6cde",
    "ftmo_futures_growth_50k": "d83798b6cc199b1f5d9b8a18b11030001142944eb6404b43e31089a5000b6310",
    "ftmo_futures_growth_100k": "bb25d4259647cf52729b89d28abe74f04a1e0dd247ebc54d62ff6a43116738e6",
    "ftmo_futures_growth_150k": "a5b831df36fe7ba3d53b752452b306bbe1979fc4a83f56adbd6af65d29b48470",
    "ftmo_futures_pro_50k": "8a9cda96ace21f845961b23cae2f64b4483cd8c6a8f7182d855d66d54687aabf",
    "ftmo_futures_pro_100k": "a8d59bc27ac2df5c49ccd02509b5eea9859633ff7af52f3aa833fe5e89ad9ac2",
    "ftmo_futures_pro_150k": "b940b4650a13a7dc31158776ae604c2ac373196d526d56823c2d8d4c29bffe92",
}


def _firms_cat():
    return {f["key"]: f for f in catalog()["firms"]}


# =============================================================================
# Catálogo
# =============================================================================
def test_I4_extended_igual_a_rules_no_none_de_HEAD_en_los_10_perfiles():
    # Solo los oficiales: el catálogo incluye los perfiles del fichero de usuario real si existe.
    firms = {k: f for k, f in _firms_cat().items() if f.get("official")}
    assert set(firms) == set(EXTENDED_HEAD), "cambió la lista de perfiles: revisa I4"
    for key, ext in EXTENDED_HEAD.items():
        assert bool(firms[key]["extended"]) is ext, key   # _jsonable: bool -> 0/1


def test_kind_cfd_si_cycle_si_no_futures():
    for key, f in _firms_cat().items():
        assert f["kind"] == ("cfd" if f["funded_model"] == "cycle" else "futures"), key


def test_I2_rules_de_los_extendidos_identicas_a_HEAD():
    firms = _firms_cat()
    for key, sha in RULES_SHA_HEAD.items():
        s = json.dumps(firms[key]["rules"], sort_keys=True, ensure_ascii=False)
        assert hashlib.sha256(s.encode()).hexdigest() == sha, key


def test_I2_caveats_futures_y_apex_son_las_cadenas_de_HEAD_en_orden():
    firms = _firms_cat()
    apex = [c for c in firms["apex_150k_intraday"]["caveats"]
            if c["id"].startswith("apex_")]
    assert [c["id"] for c in apex] == [f"apex_{i}" for i in range(1, 8)]
    assert "wander" in apex[0]["texto"]
    assert "Deadline de activación" in apex[6]["texto"]
    for key in [k for k in firms if k.startswith("ftmo_futures")]:
        fut = [c for c in firms[key]["caveats"] if c["id"].startswith("futures_")
               and c["id"][-1].isdigit()]
        assert [c["id"] for c in fut] == [f"futures_{i}" for i in range(1, 7)]
        assert fut[0]["texto"].startswith("Cap de campaña (183d)")
    # (el literal completo está fijado en tests/test_model_notes.py)


def test_caveats_son_objetos_con_las_cuatro_claves():
    for key, f in _firms_cat().items():
        assert f["caveats"], key
        for c in f["caveats"]:
            assert set(c) == {"id", "titulo", "texto", "sesgo"}, (key, c)
            assert c["sesgo"] in ("neutral", "optimista", "conservador")


def test_I3_ftmo_cfd_cycle_con_valores_del_yaml_y_un_aviso_de_cada_sesgo():
    firms = _firms_cat()
    cfg = main._firms["ftmo_50k"]
    f = firms["ftmo_50k"]
    assert not f["extended"] and f["kind"] == "cfd"
    r = f["rules"]
    assert r["model"] == "cycle"
    assert [p["name"] for p in r["phases"]] == [p.name for p in cfg.phases]
    p0 = r["phases"][0]
    assert (p0["profit_target_pct"], p0["max_daily_loss_pct"], p0["max_loss_pct"],
            p0["min_trading_days"]) == (10.0, 5.0, 10.0, 4)
    fu = r["funded"]
    assert fu["model"] == "cycle"
    assert fu["profit_split_pct"] == cfg.profit_split_pct
    assert fu["payout_cycle_days"] == 14 and fu["max_payout_cycles"] == 50
    assert fu["horizon_cap_days"] == 14 * 50
    assert bool(r["refund_fee_on_first_payout"]) is True
    assert {c["sesgo"] for c in f["caveats"]} == {"neutral", "optimista", "conservador"}


def test_ftmo_cfd_caveats_sin_notas_de_futuros_y_futures_sin_ev_condicional():
    ids = {c["id"] for c in _firms_cat()["ftmo_50k"]["caveats"]}
    assert not any(i.startswith(("futures_", "apex_")) for i in ids)
    assert "ev_condicional_resolver" in ids and "ftmo_horizonte_fondeada" in ids
    fut_ids = {c["id"] for c in _firms_cat()["ftmo_futures_pro_50k"]["caveats"]}
    assert "ev_condicional_resolver" not in fut_ids   # enmienda v2 (revisión de fidelidad H1)
    assert "futures_ev_incondicional" in fut_ids


def test_notas_model_notes_axi():
    cat = catalog()
    assert [n["id"] for n in cat["model_notes_axi"]] == [
        n.id for n in notes_for("axi")]
    assert all(set(n) == {"id", "titulo", "texto", "sesgo"}
               for n in cat["model_notes_axi"])
    assert "axi_regla_pase" in {n["id"] for n in cat["model_notes_axi"]}


# =============================================================================
# _bs_explain: un test por modo (guardrail bloque-politis-white, E2)
# =============================================================================
def test_bs_explain_auto_politis_white():
    mode, reason, cl = _bs_explain("auto")
    assert mode == "auto" and cl is None            # ya no se calcula la agrupación de volatilidad
    assert reason == ("tamaño medio elegido con el método de Politis-White sobre el resultado de cada bloque medido en "
                      "riesgos (R): sigue la dependencia real entre bloques y no depende de la f ni de la rejilla")


def test_bs_explain_manual():
    assert _bs_explain("manual") == ("manual", "manual: lo fijaste tú", None)


def test_bs_explain_fallback_iid_menos_de_25_bloques_o_mas_del_10_por_ciento_sin_riesgo():
    assert _bs_explain("fallback_iid") == (
        "fallback_iid",
        "no se pudo medir la dependencia entre bloques (menos de 25 bloques, o más del 10 % de las operaciones sin riesgo "
        "por lote válido): bloques sueltos",
        None)


def test_bs_explain_auto_nunca_se_etiqueta_fallback_iid():
    """E2: antes (``cal is None``) una corrida automática sin ``BlockSizeCalibration`` salía como ``fallback_iid``; el modo
    manda y nada más."""
    assert _bs_explain("auto")[0] == "auto"
    assert {_bs_explain(m)[0] for m in ("auto", "manual", "fallback_iid")} == {"auto", "manual", "fallback_iid"}


def test_bs_explain_modo_desconocido_es_error():
    with pytest.raises(ValueError):
        _bs_explain("calibrado")


def test_bs_explain_contra_el_bloque_automatico_real():
    """Las ramas reales de ``_auto_block_size`` caen en ``auto`` (serie con o sin dependencia) o en ``fallback_iid``, y el
    texto es el del modo."""
    from types import SimpleNamespace
    rng = np.random.default_rng(7)

    def blocks_for(r):
        return [SimpleNamespace(sym_id=np.array([0]), trade_local_id=np.array([i]), pnl_per_lot=np.array([float(v)]))
                for i, v in enumerate(r)]

    leg = {"name": "s", "kind": main.KIND_SQX, "fleg": SimpleNamespace(risk_per_micro=np.ones(400))}
    iid = rng.normal(size=300)                              # sin dependencia: b* ≤ 1 -> 1
    bs, mode, info, warns = main._auto_block_size(blocks_for(iid), [leg], {"f_ref": 0.01})
    assert mode == "auto" and bs >= 1.0 and not warns
    assert _bs_explain(mode)[1] == main._BS_REASONS["auto"]
    reg = np.repeat(rng.choice([-1.0, 1.0], size=40), 10) + 0.5 * rng.normal(size=400)   # rachas: dependencia fuerte
    bs, mode, info, warns = main._auto_block_size(blocks_for(reg), [leg], {"f_ref": 0.01})
    assert mode == "auto" and bs > 3.0 and info["series"] == "R" and info["n_excluded_terms"] == 0
    bs, mode, info, warns = main._auto_block_size(blocks_for(iid[:20]), [leg], {"f_ref": 0.01})
    assert (bs, mode) == (1.0, "fallback_iid") and warns


# =============================================================================
# data_span_days
# =============================================================================
def _blk(start, end):
    return SimpleNamespace(start=pd.Timestamp(start), end=pd.Timestamp(end))


def test_data_span_days_primer_inicio_a_ultimo_fin():
    blocks = [_blk("2020-01-10", "2020-01-12"), _blk("2020-01-01", "2020-01-03"),
              _blk("2020-04-09", "2020-04-10")]
    assert _data_span_days(blocks) == 100   # 2020-01-01 -> 2020-04-10


def test_data_span_days_solo_cuenta_los_bloques_de_la_ventana():
    todos = [_blk("2020-01-01", "2020-01-03"), _blk("2020-06-01", "2020-06-03")]
    ventana = todos[1:]            # corte IS/OOS: solo los bloques usados
    assert _data_span_days(todos) > _data_span_days(ventana) == 2


# =============================================================================
# _model_context sintético (AC8 parcial)
# =============================================================================
def _blocks_syn(n=40):
    t0 = pd.Timestamp("2021-01-01")
    return [_blk(t0 + pd.Timedelta(days=10 * i), t0 + pd.Timedelta(days=10 * i + 2))
            for i in range(n)]


def _no_braces(s: str):
    assert "{" not in s and "}" not in s, s


def test_model_context_propfirm_cycle_rellena_todas_las_notas_en_esta_corrida():
    cfg = main._firms["ftmo_50k"]
    mc = _model_context(
        lente="propfirm", blocks=_blocks_syn(40), n_blocks=40, bs=3.0,
        bs_mode="auto", bs_info={"series": "R", "n_excluded_terms": 0, "opt_m": 2, "m": 4},
        cfg=cfg, extra={"pct_cortadas_datos": "37 %"})
    # perfiles-v2: las notas filtran por compañía y nº de fases (ftmo_50k = FTMO, 2 fases)
    esperadas = [n.id for n in notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2)
                 if n.en_esta_corrida]
    assert [n["id"] for n in mc["notes"]] == esperadas
    assert {"remuestreo_bloques", "tamano_bloque",
            "ftmo_horizonte_fondeada"} <= set(esperadas)
    for n in mc["notes"]:
        _no_braces(n["texto"])
        assert set(n) == {"id", "titulo", "texto", "sesgo"}
    txt = {n["id"]: n["texto"] for n in mc["notes"]}
    assert "40 bloques" in txt["remuestreo_bloques"]
    assert f"{mc['data_span_days']} días" in txt["remuestreo_bloques"]
    assert txt["tamano_bloque"].startswith("Grupos de 3.0 bloques de media (auto): tamaño medio elegido con el método de Politis-White")
    assert "700 días" in txt["ftmo_horizonte_fondeada"]
    assert "37 %" in txt["ftmo_horizonte_fondeada"]
    assert mc["block_size"] == {
        "value": 3.0, "mode": "auto", "reason": mc["block_size"]["reason"],
        "has_real_clustering": None, "n_blocks": 40, "calibrated_at_f": None,
        "method": "politis_white", "series": "R", "n_excluded_terms": 0, "opt_m": 2, "m": 4}
    assert mc["funded_horizon"] == {"cap_days": 700, "window_blocks": 40}
    assert mc["data_span_days"] == 392   # 39*10 + 2


def test_model_context_sin_pct_cortadas_omite_solo_esa_nota():
    cfg = main._firms["ftmo_50k"]
    mc = _model_context(lente="propfirm", blocks=_blocks_syn(), n_blocks=40, bs=1.0,
                        bs_mode="manual", cfg=cfg)
    ids = {n["id"] for n in mc["notes"]}
    assert "ftmo_horizonte_fondeada" not in ids
    assert {"remuestreo_bloques", "tamano_bloque"} <= ids
    assert mc["block_size"]["calibrated_at_f"] is None
    assert mc["block_size"]["mode"] == mc["block_size"]["method"] == "manual"
    assert (mc["block_size"]["series"], mc["block_size"]["n_excluded_terms"]) == (None, None)
    for n in mc["notes"]:
        _no_braces(n["texto"])


def test_model_context_horizonte_por_modelo():
    kw = dict(lente="propfirm", blocks=_blocks_syn(), n_blocks=40, bs=1.0,
              bs_mode="manual")
    fut = main._firms["ftmo_futures_growth_50k"]
    apex = main._firms["apex_150k_intraday"]
    assert _model_context(cfg=fut, **kw)["funded_horizon"]["cap_days"] == \
        fut.subscription.funded_cap_days == 365
    assert _model_context(cfg=apex, **kw)["funded_horizon"]["cap_days"] is None
    # Futures/Apex no tienen la nota de horizonte ``cycle``.
    for cfg in (fut, apex):
        ids = {n["id"] for n in _model_context(cfg=cfg, **kw)["notes"]}
        assert "ftmo_horizonte_fondeada" not in ids


def test_model_context_axi_rellena_regla_de_pase_y_no_trae_funded_horizon():
    mc = _model_context(
        lente="axi", blocks=_blocks_syn(), n_blocks=40, bs=2.0, bs_mode="auto",
        bs_info={"series": "R", "n_excluded_terms": 0, "opt_m": 1, "m": 2},
        extra={"min_dias": 5, "min_trades": 20})
    assert "funded_horizon" not in mc
    esperadas = [n.id for n in notes_for("axi") if n.en_esta_corrida]
    assert [n["id"] for n in mc["notes"]] == esperadas
    txt = {n["id"]: n["texto"] for n in mc["notes"]}
    assert txt["axi_regla_pase"] == "En esta etapa: 20 trades mínimos y 5 días mínimos."
    assert mc["block_size"]["calibrated_at_f"] is None          # la serie en R no tiene f: ya no hay f de calibración
    assert txt["tamano_bloque"].startswith("Grupos de 2.0 bloques de media (auto)")
    for n in mc["notes"]:
        _no_braces(n["texto"])


def test_model_context_axi_etapa_sin_suelos_omite_la_nota_de_regla():
    mc = _model_context(lente="axi", blocks=_blocks_syn(), n_blocks=40, bs=1.0,
                        bs_mode="fallback_iid",
                        extra={"min_dias": None, "min_trades": None})
    assert "axi_regla_pase" not in {n["id"] for n in mc["notes"]}
    assert mc["block_size"]["mode"] == "fallback_iid"


def test_ac8_toda_nota_con_plantilla_se_rellena_con_un_ctx_completo():
    """Con un ctx completo ninguna nota con ``en_esta_corrida`` queda sin rellenar."""
    ctx = {"n_blocks": 40, "bs": "3", "bs_modo": "auto", "bs_razon": "r",
           "span_dias": 392, "cap_dias": 700, "pct_cortadas_datos": "10 %",
           "min_dias": 5, "min_trades": 20, "f_calibracion": "1.00%"}
    for n in NOTES:
        if n.en_esta_corrida:
            out = render_run(n, ctx)
            assert out is not None, n.id
            _no_braces(out)


# =============================================================================
# Endpoints (smoke con la pata fixture; I1: solo se añaden claves)
# =============================================================================
@pytest.fixture(scope="module")
def pf():
    if not (ROOT / LEG).exists():
        pytest.skip("pata fixture gitignorada no disponible")
    res = portfolio_load(PortfolioLoadRequest(legs=[LEG], account_size=50000.0))
    assert res["ok"], res
    return res


def _preq(pf, **kw):
    base = dict(portfolio_id=pf["portfolio_id"], firm_key="ftmo_50k",
                f_lo=0.0025, f_hi=0.005, f_step=0.0025, n_paths=100,
                n_challenges=5, n_sequences=100, capital0_eur=3000.0,
                fx_eur_usd=1.25)
    base.update(kw)
    return PropfirmSimRequest(**base)


@needs_data
def test_endpoint_propfirm_cycle_trae_model_context_coherente_con_meta(pf):
    out = simulate_propfirm(_preq(pf))
    m = out["meta"]
    mc = m["model_context"]
    assert mc["block_size"]["value"] == m["block_size"]
    assert mc["block_size"]["n_blocks"] == m["n_blocks"]
    assert mc["block_size"]["mode"] in ("auto", "fallback_iid")
    assert mc["data_span_days"] > 0
    assert mc["funded_horizon"] == {"cap_days": 700, "window_blocks": m["n_blocks"]}
    assert mc["notes"]
    for n in mc["notes"]:
        _no_braces(n["texto"])
    txt = {n["id"]: n["texto"] for n in mc["notes"]}
    assert f"{m['n_blocks']} bloques" in txt["remuestreo_bloques"]
    assert f"{mc['data_span_days']} días" in txt["remuestreo_bloques"]
    assert f"{mc['block_size']['value']:.1f}" in txt["tamano_bloque"]        # un decimal (E12)
    assert mc["block_size"]["method"] in ("politis_white", "fallback_iid") and mc["block_size"]["series"] == "R"
    # Campo nuevo aditivo en cada reporte (cycle): fracción en [0, 1] o None.
    for r in out["reports"]:
        assert "funded_cut_by_data_frac" in r
        v = r["funded_cut_by_data_frac"]
        assert v is None or 0.0 <= v <= 1.0
        # alive_at_cap_frac sigue ahí con su semántica (disjunto del nuevo).
        assert "alive_at_cap_frac" in r
        if v is not None and r["alive_at_cap_frac"] is not None:
            assert v + r["alive_at_cap_frac"] <= 1.0 + 1e-9
    # Si el reporte de f* tiene fracción, la nota la rellena (y a la inversa).
    star = next(r for r in out["reports"] if abs(r["f"] - out["f_star"]) < 1e-9)
    assert (star["funded_cut_by_data_frac"] is not None) == \
        ("ftmo_horizonte_fondeada" in txt)


@needs_data
def test_endpoint_propfirm_manual_bs_dice_manual(pf):
    out = simulate_propfirm(_preq(pf, block_size=2.0))
    bs = out["meta"]["model_context"]["block_size"]
    assert bs["mode"] == "manual" and bs["value"] == 2.0
    assert bs["reason"] == "manual: lo fijaste tú"
    assert bs["has_real_clustering"] is None and bs["calibrated_at_f"] is None


@needs_data
def test_endpoint_propfirm_futures_ev_semantics_parametrizado(pf, monkeypatch):
    key = "ftmo_futures_growth_50k"
    req = _preq(pf, firm_key=key)
    out = simulate_propfirm(req)
    assert "cap 183d = truncado del modelo" in out["meta"]["ev_semantics"]
    assert out["meta"]["model_context"]["funded_horizon"]["cap_days"] == 365
    for r in out["reports"]:
        assert r.get("funded_cut_by_data_frac") is None   # futures: sin el campo
    cfg = main._firms[key]
    cfg2 = dataclasses.replace(
        cfg, subscription=dataclasses.replace(cfg.subscription,
                                              campaign_cap_days=200))
    monkeypatch.setitem(main._firms, key, cfg2)
    out2 = simulate_propfirm(req)
    assert "cap 200d = truncado del modelo" in out2["meta"]["ev_semantics"]
    # Texto idéntico salvo el número.
    assert out["meta"]["ev_semantics"].replace("183d", "200d") == \
        out2["meta"]["ev_semantics"]


@needs_data
def test_endpoint_pair_trae_model_context(pf):
    out = simulate_propfirm_pair(_preq(pf))
    mc = out["meta"]["model_context"]
    assert mc["block_size"]["n_blocks"] == out["meta"]["n_blocks"]
    assert "ftmo_horizonte_fondeada" not in {n["id"] for n in mc["notes"]}
    for n in mc["notes"]:
        _no_braces(n["texto"])


@needs_data
def test_endpoint_axi_trae_model_context(pf):
    out = simulate_axi(AxiSimRequest(portfolio_id=pf["portfolio_id"], stage="seed",
                                     f_lo=0.01, f_hi=0.02, f_step=0.01, n_paths=100))
    m = out["meta"]
    mc = m["model_context"]
    assert "funded_horizon" not in mc
    assert mc["block_size"]["value"] == m["block_size"]
    assert mc["block_size"]["n_blocks"] == m["n_blocks"]
    assert mc["data_span_days"] > 0
    ids = {n["id"] for n in mc["notes"]}
    assert {"remuestreo_bloques", "tamano_bloque"} <= ids
    if m["min_unique_trades"] is not None and m["min_days"] is not None:
        txt = {n["id"]: n["texto"] for n in mc["notes"]}
        assert txt["axi_regla_pase"] == (
            f"En esta etapa: {m['min_unique_trades']} trades mínimos y "
            f"{m['min_days']} días mínimos.")
    for n in mc["notes"]:
        _no_braces(n["texto"])
