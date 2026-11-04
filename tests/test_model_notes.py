"""Catálogo de notas del modelo (guardrail ``onboarding-b1``, B1-1).

Cubre el loader estricto (I5), el render estricto (I7), el filtrado por ámbito y la
fidelidad literal de las cadenas de Futures y Apex que antes vivían en
``main.py::_profile_caveats`` (I2). No necesita datos.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest tests/test_model_notes.py -q
"""
from pathlib import Path

import pytest
import yaml

from poc_web.api.model_notes import (
    LENTES, MODELOS, NOTES, NOTES_PATH, PLACEHOLDERS, SESGOS, TIPOS,
    Note, load_notes, notes_for, render_run,
)

# Cadenas de ``_profile_caveats`` en HEAD 6e66324 (rama futures, 6; rama Apex, 7),
# copiadas literalmente. Si cambian, el YAML ya no es fiel a lo que mostraba la app.
HEAD_FUTURES = [
    "Cap de campaña (183d) / funded (365d) / 50 ciclos de payout son horizontes de TRUNCADO del modelo (decisión 2026-09-29), NO reglas FTMO — se reportan p_truncated / alive_at_cap.",
    "El breach de drawdown/daily loss usa rel_low GROSS de comisiones → el chequeo de equity es levemente optimista (ligado a ruleset_version).",
    "Fase Beta: consta solo en el blog de lanzamiento (no en la página de reglas); FTMO promete no cambiar reglas retroactivamente en cuentas activas. El 'Withdraw 100%' de Pro es la FRACCIÓN RETIRABLE del profit — el ratio 90/10 aplica sobre lo solicitado.",
    "Growth: el hit de daily loss cierra todo y suspende el día (pause_not_fatal) modelado como liquidación en el suelo sin slippage; un día con hit y cierre ≥ umbral cualifica.",
    "El escalado de contratos de la funded (scaling tiers) NO está verificado (pendiente de verificar): sin diagnóstico de cap PA — solo el cap de contratos de la evaluación.",
    "Las métricas de la funded (días a payout, alive_at_cap, DLL hits, deferrals) se calculan sobre el POOL de paths passed del backtest (las campañas re-dibujan paths con reemplazo): una campaña passed puede contar de nuevo paths ya observados.",
]
HEAD_APEX = [
    "Trailing intradía: el wander entre la última barra M1 y el cierre de cada trade NO está acotado (optimismo heredado; el punto de cierre sí es exacto vía PnL realizado).",
    "Deferral de consistencia (50%): se asume que los días cualificados SE CONSERVAN tras una denegación y se reevalúa cada sesión — supuesto a verificar contra el help center (ligado a ruleset_version).",
    "Backtest tratado como neto de comisiones: si es gross, los días cualificados ($300) y la inactividad ($50) quedan levemente optimistas.",
    "Hit de DLL con clamp sin slippage: PnL de sesión := min(observado, −DLL); la liquidación forzosa real puede quedar peor (conservador en PnL, sin modelar el deslizamiento).",
    "Multi-cuenta (n_copied_accounts > 1): correlación PERFECTA entre cuentas — pasan/queman a la vez; el riesgo de cola no se diversifica.",
    "Cap de contratos de la PA evaluado contra el tier MÍNIMO (cota conservadora): max_observed permite comparar a mano contra tiers altos.",
    "Deadline de activación de 7 días no modelado: la PA siempre se activa al pasar.",
]


def _write(tmp_path: Path, notas: list[dict]) -> Path:
    p = tmp_path / "notes.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "notas": notas}, allow_unicode=True),
                 encoding="utf-8")
    return p


def _nota(**over) -> dict:
    base = {"id": "n1", "ambito": {"lente": ["propfirm"], "modelo": ["*"],
                                   "tipo": ["*"]},
            "sesgo": "neutral", "titulo": "t", "texto": "texto estático",
            "fuente": "x"}
    base.update(over)
    return base


# --- I5: el catálogo real es válido ----------------------------------------------
def test_notas_real_ids_unicos_y_enumerados():
    ids = [n.id for n in NOTES]
    assert len(ids) == len(set(ids))
    for n in NOTES:
        assert set(n.lente) <= LENTES
        assert set(n.modelo) <= MODELOS
        assert set(n.tipo) <= TIPOS
        assert n.sesgo in SESGOS


def test_notas_real_texto_sin_llaves_y_placeholders_conocidos():
    import string
    for n in NOTES:
        assert "{" not in n.texto and "}" not in n.texto, n.id
        if n.en_esta_corrida:
            names = {f for _l, f, _s, _c in string.Formatter().parse(n.en_esta_corrida)
                     if f}
            assert names <= PLACEHOLDERS, (n.id, names - PLACEHOLDERS)


def test_notas_real_toda_combinacion_tiene_al_menos_una_nota():
    # (modelo, tipo) existentes: cycle/cfd, ladder/futures, futures/futures.
    for modelo, tipo in (("cycle", "cfd"), ("ladder", "futures"),
                         ("futures", "futures")):
        assert notes_for("propfirm", modelo, tipo), (modelo, tipo)
    assert notes_for("axi")
    assert notes_for("axi", tipo="cfd")


def test_ftmo_cfd_tiene_al_menos_un_aviso_de_cada_sesgo():
    sesgos = {n.sesgo for n in notes_for("propfirm", "cycle", "cfd")}
    assert sesgos == {"neutral", "optimista", "conservador"}


def test_anexo_a_ids_presentes():
    ids = {n.id for n in NOTES}
    esperados = {
        "remuestreo_bloques", "tamano_bloque", "pocos_bloques", "edge_del_backtest",
        "costes_backtest", "ev_condicional_resolver", "fases_continuas",
        "espera_entre_fases", "futures_ev_incondicional", "ftmo_post_objetivo",
        "ftmo_cobros_ciclos", "ftmo_horizonte_fondeada", "ftmo_noticias_finde",
        "ftmo_split_fijo", "ftmo_eje_dias", "ftmo_sin_consistencia",
        # perfiles-v2: variantes por compañía/fases y notas de perfil de usuario
        "cobros_ciclos_otras", "horizonte_fondeada_otras", "conducta_otras", "split_fijo_otras",
        "eje_dias_otras", "ftmo_1f_consistencia", "consistencia_otras", "reglas_de_usuario",
        "plantilla_base",
        # csv-d1-d2 fase 1 (2026-10-19): base del lote = tamaño de cuenta (Enmienda E2)
        "lote_sobre_cuenta",
        # csv-d1-d2 fase 2 (2026-10-19): flotante por minuto con el realizado (plan §2.4)
        "flotante_por_minuto",
        "axi_regla_pase", "axi_post_objetivo", "axi_lote_minimo",
        "axi_dias_calendario", "axi_puertas",
        # entrada-4-bloques fase F4 (2026-10-20): receta para el EA, de dónde salen las especificaciones y el flotante de
        # los .sqx en la lente AXI
        "receta_ea", "fichas_broker", "axi_sqx_flotante",
    } | {f"futures_{i}" for i in range(1, 7)} | {f"apex_{i}" for i in range(1, 8)}
    assert ids == esperados


def test_enmiendas_v2_ambitos_y_sesgos():
    by = {n.id: n for n in NOTES}
    # ev_condicional_resolver: falso en futuros -> solo cycle y ladder.
    assert set(by["ev_condicional_resolver"].modelo) == {"cycle", "ladder"}
    assert set(by["espera_entre_fases"].modelo) == {"cycle", "ladder"}
    assert by["fases_continuas"].modelo == ("*",)
    assert set(by["futures_ev_incondicional"].modelo) == {"futures"}
    assert by["axi_post_objetivo"].sesgo == "conservador"
    assert by["futures_4"].sesgo == "neutral" and by["apex_4"].sesgo == "neutral"
    assert [by[f"futures_{i}"].sesgo for i in range(1, 7)] == [
        "conservador", "optimista", "neutral", "neutral", "neutral", "neutral"]
    assert [by[f"apex_{i}"].sesgo for i in range(1, 8)] == [
        "optimista", "neutral", "optimista", "neutral", "neutral", "conservador",
        "optimista"]


# --- I2: textos de Futures y Apex literales y en orden --------------------------
def test_textos_futures_literales_y_en_orden():
    got = [n.texto for n in NOTES if n.id.startswith("futures_") and n.id[-1].isdigit()]
    assert got == HEAD_FUTURES


def test_textos_apex_literales_y_en_orden():
    got = [n.texto for n in NOTES if n.id.startswith("apex_")]
    assert got == HEAD_APEX


def test_notas_futures_y_apex_solo_en_su_modelo():
    fut = {n.id for n in notes_for("propfirm", "futures", "futures")}
    apx = {n.id for n in notes_for("propfirm", "ladder", "futures")}
    cyc = {n.id for n in notes_for("propfirm", "cycle", "cfd")}
    assert {f"futures_{i}" for i in range(1, 7)} <= fut
    assert not any(i.startswith("apex_") for i in fut)
    assert {f"apex_{i}" for i in range(1, 8)} <= apx
    assert not any(i.startswith("futures_") for i in apx)
    assert not any(i.startswith(("futures_", "apex_")) for i in cyc)
    # Los de ámbito "*" están en las tres.
    assert {"remuestreo_bloques", "tamano_bloque", "fases_continuas"} <= fut & apx & cyc


# --- notes_for: orden y filtros ----------------------------------------------------
def test_notes_for_respeta_el_orden_del_yaml():
    # Orden y pertenencia literales (revisión de código B1-1, C3: el test anterior era tautológico).
    # perfiles-v2: el orden de HEAD es el de un perfil FTMO de 2 fases (empresa y fases filtran).
    assert [n.id for n in notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2)] == [
        "remuestreo_bloques", "tamano_bloque", "pocos_bloques", "edge_del_backtest", "costes_backtest",
        "lote_sobre_cuenta", "flotante_por_minuto",
        "ev_condicional_resolver", "fases_continuas", "espera_entre_fases", "ftmo_post_objetivo",
        "ftmo_cobros_ciclos", "ftmo_horizonte_fondeada", "ftmo_noticias_finde", "ftmo_split_fijo",
        "ftmo_eje_dias", "ftmo_sin_consistencia",
        "receta_ea", "fichas_broker"]            # F4: las notas nuevas van al final del YAML, luego al final de cada lista
    assert [n.id for n in notes_for("axi")] == [
        "remuestreo_bloques", "tamano_bloque", "pocos_bloques", "edge_del_backtest", "costes_backtest",
        "lote_sobre_cuenta", "flotante_por_minuto",
        "axi_regla_pase", "axi_post_objetivo", "axi_lote_minimo", "axi_dias_calendario", "axi_puertas",
        "receta_ea", "fichas_broker", "axi_sqx_flotante"]


def test_notes_for_lente_axi_ignora_el_modelo():
    assert notes_for("axi", "cycle") == notes_for("axi")
    assert all("axi" in n.lente for n in notes_for("axi"))
    assert not any("propfirm" in n.lente and "axi" not in n.lente
                   for n in notes_for("axi"))


# --- Loader estricto (I5): errores con el id ----------------------------------------
def test_loader_acepta_nota_minima(tmp_path):
    notes = load_notes(_write(tmp_path, [_nota()]))
    assert len(notes) == 1 and isinstance(notes[0], Note)


@pytest.mark.parametrize("notas,fragmento", [
    ([_nota(), _nota()], "duplicado"),
    ([_nota(sesgo="malo")], "sesgo"),
    ([_nota(ambito={"lente": ["otra"], "modelo": ["*"], "tipo": ["*"]})], "lente"),
    ([_nota(ambito={"lente": ["axi"], "modelo": ["raro"], "tipo": ["*"]})], "modelo"),
    ([_nota(ambito={"lente": ["axi"], "modelo": ["*"], "tipo": ["otro"]})], "tipo"),
    ([_nota(texto="hola {x}")], "texto"),
    ([_nota(texto="hola }")], "texto"),
    ([_nota(en_esta_corrida="hay {no_existe}")], "no_existe"),
    ([_nota(en_esta_corrida="hay {bs:.2f}")], "formato"),
    ([_nota(en_esta_corrida="hay {}")], "inválido"),
    ([_nota(en_esta_corrida="hay {bs")], "mal formada"),
])
def test_loader_estricto_lanza_valueerror_con_el_id(tmp_path, notas, fragmento):
    with pytest.raises(ValueError) as exc:
        load_notes(_write(tmp_path, notas))
    assert fragmento in str(exc.value)
    assert "n1" in str(exc.value)


def test_loader_falta_campo_obligatorio(tmp_path):
    n = _nota()
    del n["fuente"]
    with pytest.raises(ValueError, match="n1"):
        load_notes(_write(tmp_path, [n]))


# --- I7: render estricto ----------------------------------------------------------
def _tpl(tmp_path, tpl) -> Note:
    return load_notes(_write(tmp_path, [_nota(en_esta_corrida=tpl)]))[0]


def test_render_run_rellena_con_ctx_completo(tmp_path):
    n = _tpl(tmp_path, "Usa {n_blocks} bloques y {bs}.")
    assert render_run(n, {"n_blocks": 40, "bs": "3", "otro": 1}) == "Usa 40 bloques y 3."


def test_render_run_clave_ausente_devuelve_none_sin_rellenar_a_medias(tmp_path):
    n = _tpl(tmp_path, "Usa {n_blocks} bloques y {bs}.")
    assert render_run(n, {"n_blocks": 40}) is None


def test_render_run_valor_none_cuenta_como_ausente(tmp_path):
    n = _tpl(tmp_path, "Usa {n_blocks} bloques.")
    assert render_run(n, {"n_blocks": None}) is None


def test_render_run_sin_plantilla_devuelve_none(tmp_path):
    n = load_notes(_write(tmp_path, [_nota()]))[0]
    assert render_run(n, {"n_blocks": 1}) is None


def test_render_run_nunca_deja_llaves(tmp_path):
    n = _tpl(tmp_path, "{n_blocks} y {span_dias}")
    out = render_run(n, {"n_blocks": 5, "span_dias": 90})
    assert out == "5 y 90" and "{" not in out and "}" not in out
    assert render_run(n, {}) is None


def test_render_run_cero_es_un_valor_valido(tmp_path):
    n = _tpl(tmp_path, "{min_dias} días")
    assert render_run(n, {"min_dias": 0}) == "0 días"


def test_yaml_real_es_el_que_carga_el_modulo():
    assert NOTES_PATH.name == "model_notes.yaml"
    assert [n.id for n in load_notes()] == [n.id for n in NOTES]


# ===========================================================================
# guardrail `perfiles-v2`, fase 1: dimensiones `empresa`, `fases` y `perfil` del ámbito (Enmienda v2 2 y 3)
# ===========================================================================
import hashlib   # noqa: E402

from poc_web.api.model_notes import EMPRESAS_CON_NOTAS, load_empresas, normalize_company   # noqa: E402


def note_hash(n) -> str:
    """Huella (titulo, texto, en_esta_corrida, sesgo) de una nota; las constantes de abajo son las de 3e79519."""
    return hashlib.sha256(f"{n.titulo}\n{n.texto}\n{n.en_esta_corrida}\n{n.sesgo}".encode("utf-8")).hexdigest()[:16]


# Notas de cada (modelo, tipo) a 3e79519, con su huella: I5 (las notas de los oficiales no cambian).
# Únicas notas añadidas a estas listas desde entonces: `lote_sobre_cuenta` (csv-d1-d2 fase 1, 2026-10-19, Enmienda E2),
# común a todos los modelos y a la lente AXI, justo tras `costes_backtest` (huella de su texto en `_LOTE`), y
# `flotante_por_minuto` (csv-d1-d2 fase 2, 2026-10-19, plan §2.4), solo de la lente AXI y de los perfiles cycle (no de
# ladder ni futures), justo tras `lote_sobre_cuenta` (huella en `_FLOT`). Y, desde entrada-4-bloques fase F4 (2026-10-20),
# `receta_ea` y `fichas_broker` (las dos lentes y todos los modelos) y `axi_sqx_flotante` (solo la lente AXI): se añaden AL
# FINAL del YAML, así que van al final de cada lista (huellas en `_RECETA`, `_FICHAS` y `_AXISQX`).
_LOTE = ("lote_sobre_cuenta", "b4a4134a76a8efd8")
_FLOT = ("flotante_por_minuto", "f67403c67bcb6f31")
_RECETA = ("receta_ea", "68648591c5139980")      # H5 de la revisión de código de la F4: «no hace falta cambiar el MM del backtest»
_FICHAS = ("fichas_broker", "b9e02f26f6d26594")
_AXISQX = ("axi_sqx_flotante", "f21a64445af71ae6")
# `tamano_bloque` cambió el 2026-10-21 (guardrail bloque-politis-white, plan §2.3 y E8): la huella es la del texto NUEVO
# (la vieja era 2f3a9ac48b7f1aca); lo que dice la nota se comprueba literalmente en `test_bloque_politis_white_*` de abajo.
_TAMANO_BLOQUE = ("tamano_bloque", "21cb42e6e1b200e6")
_COMMON = [("remuestreo_bloques", "f569c72c9a516cbd"), _TAMANO_BLOQUE,
           ("pocos_bloques", "7bada2f8d17733bb"), ("edge_del_backtest", "6d869d7219a6ed89"),
           ("costes_backtest", "7c675ca430cdfaf4")]
HEAD_IDS_HASH = {
    ("cycle", "cfd"): _COMMON + [_LOTE, _FLOT,
        ("ev_condicional_resolver", "1876e6a5e6ae85d0"), ("fases_continuas", "ecd588e5020cbef4"),
        ("espera_entre_fases", "e54598219ca2e5e4"), ("ftmo_post_objetivo", "8ffd45584c9d87f4"),
        ("ftmo_cobros_ciclos", "9606a4aa8e7b40ba"), ("ftmo_horizonte_fondeada", "67ee32a85bfa4394"),
        ("ftmo_noticias_finde", "c42273ceb5b417e8"), ("ftmo_split_fijo", "400653f89908a5f2"),
        ("ftmo_eje_dias", "62de34954b1745b9"), ("ftmo_sin_consistencia", "934a6ffbae2f5a6c"), _RECETA, _FICHAS],
    ("futures", "futures"): _COMMON[:4] + [("costes_backtest", "7c675ca430cdfaf4"), _LOTE,
                                           ("fases_continuas", "ecd588e5020cbef4"),
                                           ("futures_ev_incondicional", "25fc0ebe970e38ed"),
                                           ("futures_1", "cc89c2570c7b282d"), ("futures_2", "d075f3b1085ef4b5"),
                                           ("futures_3", "116cedec1f0fc114"), ("futures_4", "0e96e7d6a4226a5d"),
                                           ("futures_5", "6f95adfecfc76bf7"), ("futures_6", "e6c783d75af4ab20"),
                                           _RECETA, _FICHAS],
    ("ladder", "futures"): _COMMON + [_LOTE,
        ("ev_condicional_resolver", "1876e6a5e6ae85d0"), ("fases_continuas", "ecd588e5020cbef4"),
        ("espera_entre_fases", "e54598219ca2e5e4"), ("apex_1", "aa0f9dfecf62007b"),
        ("apex_2", "330f54137a4fc59d"), ("apex_3", "e14c46df8a2bee8f"), ("apex_4", "5413552696b8d168"),
        ("apex_5", "e3e5e61314d7ad00"), ("apex_6", "1a152d5e75659da8"), ("apex_7", "73f92bb29a521ddb"),
        _RECETA, _FICHAS],
}
HEAD_AXI = _COMMON + [_LOTE, _FLOT, ("axi_regla_pase", "56aee381e4063796"), ("axi_post_objetivo", "5aacd120d4899cd4"),
                      ("axi_lote_minimo", "7e838f4d04abd7dc"), ("axi_dias_calendario", "8216d96cf2447f7b"),
                      ("axi_puertas", "16b37cb8e30a2ac7"), _RECETA, _FICHAS, _AXISQX]


def _ids_hash(notes):
    return [(n.id, note_hash(n)) for n in notes]


def _txt(notes) -> str:
    return " ".join(f"{n.titulo} {n.texto} {n.en_esta_corrida or ''}" for n in notes).lower()


def test_empresas_con_notas_fija_y_declarada_al_principio_del_yaml():
    assert EMPRESAS_CON_NOTAS == {"cycle": ("FTMO",), "ladder": ("Apex",), "futures": ("FTMO",)} == load_empresas()
    head = NOTES_PATH.read_text(encoding="utf-8").split("\nnotas:")[0]
    assert "empresas_con_notas: {cycle: [FTMO], ladder: [Apex], futures: [FTMO]}" in head


def test_ac5_i5_las_notas_de_los_oficiales_son_las_de_head_con_el_filtro_de_empresa_y_fases():
    assert _ids_hash(notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2)) == HEAD_IDS_HASH[("cycle", "cfd")]
    assert _ids_hash(notes_for("propfirm", "futures", "futures", empresa="FTMO", fases=1)) == \
        HEAD_IDS_HASH[("futures", "futures")]
    assert _ids_hash(notes_for("propfirm", "ladder", "futures", empresa="Apex", fases=1)) == \
        HEAD_IDS_HASH[("ladder", "futures")]
    # sin filtro de empresa (lente axi y llamadas antiguas) los futuros/apex/axi no cambian
    assert _ids_hash(notes_for("propfirm", "futures", "futures")) == HEAD_IDS_HASH[("futures", "futures")]
    assert _ids_hash(notes_for("propfirm", "ladder", "futures")) == HEAD_IDS_HASH[("ladder", "futures")]
    assert _ids_hash(notes_for("axi")) == HEAD_AXI


def test_ambito_empresa_fases_perfil_de_las_notas_del_plan():
    by = {n.id: n for n in NOTES}
    for nid in ("ftmo_cobros_ciclos", "ftmo_horizonte_fondeada", "ftmo_noticias_finde", "ftmo_split_fijo",
                "ftmo_eje_dias"):
        assert by[nid].empresa == ("FTMO",) and by[nid].fases == ("*",), nid
    assert by["ftmo_post_objetivo"].empresa == ("*",)
    assert by["ftmo_sin_consistencia"].empresa == ("FTMO",) and by["ftmo_sin_consistencia"].fases == (2,)
    assert by["ftmo_1f_consistencia"].empresa == ("FTMO",) and by["ftmo_1f_consistencia"].fases == (1,)
    for nid in ("cobros_ciclos_otras", "horizonte_fondeada_otras", "conducta_otras", "split_fijo_otras",
                "eje_dias_otras", "consistencia_otras"):
        assert by[nid].empresa == ("otras",) and by[nid].modelo == ("cycle",), nid
    assert by["reglas_de_usuario"].perfil == ("usuario",) and by["reglas_de_usuario"].modelo == ("*",)
    assert by["plantilla_base"].perfil == ("copia",) and by["plantilla_base"].modelo == ("ladder", "futures")
    # notas anteriores: ámbito por defecto
    for nid in ("remuestreo_bloques", "futures_1", "apex_3", "axi_puertas"):
        assert by[nid].empresa == ("*",) and by[nid].fases == ("*",) and by[nid].perfil == ("*",)


def test_ac5_otras_es_cualquier_compania_fuera_de_empresas_con_notas():
    for comp in ("FundedNext", "Topstep", "x", "ftmo", "FTMO Futures"):          # exacto: 'ftmo' != 'FTMO'
        ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa=comp, fases=2)}
        assert "cobros_ciclos_otras" in ids and "ftmo_cobros_ciclos" not in ids, comp
    ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2)}
    assert "cobros_ciclos_otras" not in ids and "ftmo_cobros_ciclos" in ids
    # la lista es POR MODELO: "Apex" es `otras` en un perfil cycle (recibe las variantes y ninguna de FTMO)
    ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa="Apex", fases=2)}
    assert not any(i.startswith("ftmo_") and i != "ftmo_post_objetivo" for i in ids)
    assert {"cobros_ciclos_otras", "conducta_otras", "split_fijo_otras", "eje_dias_otras",
            "consistencia_otras", "horizonte_fondeada_otras"} <= ids
    # y "FTMO" no es `otras` en cycle, pero la lista del modelo manda: con otra lista, cambia
    ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2,
                                   empresas_con_notas={"cycle": ("Otra",)})}
    assert "cobros_ciclos_otras" in ids and "ftmo_cobros_ciclos" in ids


def test_ac5_normalizacion_de_la_compania():
    assert normalize_company("  Funded   Next ") == "Funded Next"
    a = notes_for("propfirm", "cycle", "cfd", empresa="  FTMO  ", fases=2)
    assert _ids_hash(a) == HEAD_IDS_HASH[("cycle", "cfd")]


def test_ac5_ninguna_nota_cycle_de_otra_compania_contiene_ftmo_sin_distinguir_mayusculas():
    for comp in ("FundedNext", "Apex", "Otra"):          # Apex incluida: en cycle es `otras`
        for fases in (1, 2):
            notes = notes_for("propfirm", "cycle", "cfd", empresa=comp, fases=fases, perfil={"usuario"})
            assert notes and "ftmo" not in _txt(notes), (comp, fases)
    # pero FTMO sí las lleva
    assert "ftmo" in _txt(notes_for("propfirm", "cycle", "cfd", empresa="FTMO", fases=2))


def test_ac5_consistencia_depende_de_las_fases_y_de_la_compania():
    def ids(comp, fases):
        return {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa=comp, fases=fases)}

    assert {"ftmo_sin_consistencia"} <= ids("FTMO", 2) and not ({"ftmo_1f_consistencia", "consistencia_otras"} & ids("FTMO", 2))
    assert {"ftmo_1f_consistencia"} <= ids("FTMO", 1) and not ({"ftmo_sin_consistencia", "consistencia_otras"} & ids("FTMO", 1))
    for fases in (1, 2):
        assert "consistencia_otras" in ids("Otra", fases)
        assert not ({"ftmo_sin_consistencia", "ftmo_1f_consistencia"} & ids("Otra", fases))


def test_perfil_usuario_y_copia_solo_con_su_etiqueta():
    assert not {"reglas_de_usuario", "plantilla_base"} & {n.id for n in notes_for("propfirm", "ladder", "futures")}
    ids = {n.id for n in notes_for("propfirm", "ladder", "futures", perfil={"usuario"})}
    assert "reglas_de_usuario" in ids and "plantilla_base" not in ids
    ids = {n.id for n in notes_for("propfirm", "ladder", "futures", perfil={"usuario", "copia"})}
    assert {"reglas_de_usuario", "plantilla_base"} <= ids
    # plantilla_base es solo de futuros/ladder; reglas_de_usuario es de todos los modelos, no de la lente axi
    ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", perfil={"usuario", "copia"})}
    assert "reglas_de_usuario" in ids and "plantilla_base" not in ids
    assert not {"reglas_de_usuario", "plantilla_base"} & {n.id for n in notes_for("axi", perfil={"usuario", "copia"})}


def test_plantilla_base_render_estricto():
    n = next(x for x in NOTES if x.id == "plantilla_base")
    assert n.texto == "" and n.texto_plantilla and "{base_nombre}" in n.texto_plantilla
    out = n.as_catalog("Apex 150K Intraday Trail (4.0)")
    assert out["texto"] == ("Este perfil copia la mecánica de Apex 150K Intraday Trail (4.0): los avisos de abajo "
                            "hablan de esa firma y de cómo la modela la app.")
    assert "{" not in out["texto"] and "}" not in out["texto"] and set(out) == {"id", "titulo", "texto", "sesgo"}
    with pytest.raises(ValueError, match="plantilla_base"):
        n.as_catalog()
    with pytest.raises(ValueError):
        n.as_catalog("")
    # una nota normal ignora el argumento
    m = next(x for x in NOTES if x.id == "apex_1")
    assert m.as_catalog("loquesea") == m.as_catalog()


# --- Loader estricto de las dimensiones nuevas --------------------------------------------------
def _write2(tmp_path: Path, notas: list[dict], **top) -> Path:
    p = tmp_path / "notes.yaml"
    p.write_text(yaml.safe_dump({"version": 1, **top, "notas": notas}, allow_unicode=True), encoding="utf-8")
    return p


def _amb(**extra) -> dict:
    return {"lente": ["propfirm"], "modelo": ["cycle"], "tipo": ["cfd"], **extra}


def test_loader_acepta_empresa_fases_y_perfil(tmp_path):
    notes = load_notes(_write2(tmp_path, [
        _nota(ambito=_amb(empresa=["FTMO", "otras"], fases=[1, 2], perfil=["usuario", "copia"]))],
        empresas_con_notas={"cycle": ["FTMO"]}))
    n = notes[0]
    assert n.empresa == ("FTMO", "otras") and n.fases == (1, 2) and n.perfil == ("usuario", "copia")
    d = load_notes(_write2(tmp_path, [_nota()]))[0]
    assert d.empresa == ("*",) and d.fases == ("*",) and d.perfil == ("*",)


@pytest.mark.parametrize("amb, top, fragmento", [
    (_amb(empresa=["Desconocida"]), {"empresas_con_notas": {"cycle": ["FTMO"]}}, "empresa"),
    (_amb(empresa=["FTMO"]), {}, "empresa"),                                  # sin la lista fija declarada
    (_amb(empresa=[]), {"empresas_con_notas": {"cycle": ["FTMO"]}}, "lista no vacía"),
    (_amb(empresa=5), {"empresas_con_notas": {"cycle": ["FTMO"]}}, "lista no vacía"),
    (_amb(fases=[0]), {}, "fases"), (_amb(fases=[True]), {}, "fases"), (_amb(fases=["2"]), {}, "fases"),
    (_amb(fases=[]), {}, "lista no vacía"),
    (_amb(perfil=["admin"]), {}, "perfil"), (_amb(perfil="usuario"), {}, "lista no vacía"),
])
def test_loader_estricto_de_las_dimensiones_nuevas(tmp_path, amb, top, fragmento):
    with pytest.raises(ValueError) as exc:
        load_notes(_write2(tmp_path, [_nota(ambito=amb)], **top))
    assert fragmento in str(exc.value) and "n1" in str(exc.value)


@pytest.mark.parametrize("emp", [
    ["FTMO"], "FTMO", {"raro": ["FTMO"]}, {"*": ["FTMO"]},             # lista plana o modelo desconocido
    {"cycle": "FTMO"}, {"cycle": [1]}, {"cycle": ["FTMO", "FTMO"]}, {"cycle": ["otras"]}, {"cycle": ["*"]},
    {"cycle": [""]}])
def test_loader_empresas_con_notas_invalida(tmp_path, emp):
    with pytest.raises(ValueError, match="empresas_con_notas"):
        load_notes(_write2(tmp_path, [_nota()], empresas_con_notas=emp))


def test_loader_texto_plantilla(tmp_path):
    ok = _nota(texto_plantilla="Copia de {base_nombre}.")
    del ok["texto"]
    n = load_notes(_write2(tmp_path, [ok]))[0]
    assert n.texto == "" and n.as_catalog("X")["texto"] == "Copia de X."
    for tpl, frag in (("Copia de {otro}.", "otro"), ("Copia de {base_nombre:>5}.", "formato"),
                      ("Copia de {}.", "inválido"), ("Copia de {base_nombre", "mal formada")):
        bad = _nota(texto_plantilla=tpl)
        del bad["texto"]
        with pytest.raises(ValueError) as exc:
            load_notes(_write2(tmp_path, [bad]))
        assert frag in str(exc.value) and "n1" in str(exc.value)
    both = _nota(texto_plantilla="a {base_nombre}")                 # `texto` y `texto_plantilla` a la vez
    with pytest.raises(ValueError, match="excluyentes"):
        load_notes(_write2(tmp_path, [both]))
    neither = _nota()
    del neither["texto"]
    with pytest.raises(ValueError, match="falta 'texto'"):
        load_notes(_write2(tmp_path, [neither]))


def test_empresas_con_notas_rechaza_nombres_sin_normalizar(tmp_path):
    """Revisión de código fase 1 (C2): un nombre con espacios sobrantes no se acepta en silencio."""
    import yaml as _yaml
    from poc_web.api import model_notes as mn
    raw = _yaml.safe_load(mn.NOTES_PATH.read_text(encoding="utf-8"))
    raw["empresas_con_notas"]["cycle"] = [" FTMO "]
    p = tmp_path / "notes.yaml"
    p.write_text(_yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    import pytest as _pytest
    with _pytest.raises(ValueError, match="sin normalizar"):
        mn.load_notes(p)


# ===========================================================================
# guardrail `csv-d1-d2`, fase 1: nota `lote_sobre_cuenta` (AC1.7, Enmienda E2)
# ===========================================================================
_E2_TITULO = "Riesgo sobre el tamaño de cuenta"
# Texto literal de la Enmienda v2 E2 del plan (con los backticks de markdown de FixedAmount y «en FTMO»).
_E2_PLAN_LITERAL = (
    "El lote de cada operación se calcula sobre el tamaño de cuenta, sin reinvertir: no crece ni mengua con el saldo "
    "simulado. Si tu EA arriesga un importe fijo (`FixedAmount`), es exactamente lo que hace. Si arriesga un % de la "
    "equity del momento, es una aproximación: en un reto la equity se mueve poco (±10 % en FTMO), pero en cuentas "
    "grandes o con horizontes largos la diferencia crece.")
# PLAN-DESVIO: lo entregado difiere del literal de E2 en dos frases y solo en ellas (ver el comentario de la nota en
# config/model_notes.yaml): sin los backticks, y «(±10 % en FTMO)» → «(del orden de ±10 %)», porque la nota es de ámbito
# `*` y la invariante AC5 de perfiles-v2 prohíbe que una nota mostrada a otra compañía nombre a FTMO.
_E2_TEXTO = _E2_PLAN_LITERAL.replace("`FixedAmount`", "FixedAmount").replace("(±10 % en FTMO)", "(del orden de ±10 %)")


def _lote():
    return next(n for n in NOTES if n.id == "lote_sobre_cuenta")


def test_ac1_7_lote_sobre_cuenta_se_carga_con_titulo_y_texto_de_la_enmienda_e2():
    n = _lote()
    assert n.titulo == _E2_TITULO
    assert n.texto == _E2_TEXTO
    assert "{" not in n.texto and "}" not in n.texto
    assert n.fuente and "csv-d1-d2" in n.fuente
    recargadas = load_notes()                                  # el loader estricto la acepta y la deja igual
    assert recargadas[[x.id for x in recargadas].index("lote_sobre_cuenta")] == n


def test_ac1_7_el_texto_entregado_difiere_del_literal_de_e2_solo_en_las_dos_frases_marcadas():
    n = _lote()
    assert n.texto != _E2_PLAN_LITERAL
    assert n.texto.replace("(FixedAmount)", "(`FixedAmount`)").replace("(del orden de ±10 %)", "(±10 % en FTMO)") \
        == _E2_PLAN_LITERAL
    assert "ftmo" not in (n.titulo + " " + n.texto).lower()   # invariante AC5 de perfiles-v2


def test_ac1_7_lote_sobre_cuenta_ambito_y_sesgo():
    """Las dos lentes, todos los modelos y tipos, todas las compañías/fases/perfiles. Sesgo neutral: el error del
    lote sobre la cuenta frente a un EA de % de equity no tiene dirección garantizada (Enmienda E1)."""
    n = _lote()
    assert set(n.lente) == {"propfirm", "axi"}
    assert n.modelo == ("*",) and n.tipo == ("*",)
    assert n.empresa == ("*",) and n.fases == ("*",) and n.perfil == ("*",)
    assert n.sesgo == "neutral"
    assert n.en_esta_corrida is None    # sin cifra por corrida: no entra en `meta.model_context.notes` (ver catálogo)
    assert n.texto_plantilla is None


def test_ac1_7_lote_sobre_cuenta_sale_en_las_dos_lentes_y_en_todos_los_modelos():
    assert "lote_sobre_cuenta" in {n.id for n in notes_for("axi")}
    assert "lote_sobre_cuenta" in {n.id for n in notes_for("axi", tipo="cfd")}
    for modelo, tipo, empresa, fases in (("cycle", "cfd", "FTMO", 2), ("cycle", "cfd", "FTMO", 1),
                                         ("cycle", "cfd", "FundedNext", 2),
                                         ("ladder", "futures", "Apex", 1), ("futures", "futures", "FTMO", 1)):
        ids = {n.id for n in notes_for("propfirm", modelo, tipo, empresa=empresa, fases=fases)}
        assert "lote_sobre_cuenta" in ids, (modelo, tipo, empresa, fases)
    # también sin filtros de compañía/fases (llamadas antiguas) y con etiquetas de perfil de usuario
    assert "lote_sobre_cuenta" in {n.id for n in notes_for("propfirm", "cycle", "cfd")}
    assert "lote_sobre_cuenta" in {n.id for n in notes_for("propfirm", "cycle", "cfd", perfil={"usuario", "copia"})}


def test_ac1_7_lote_sobre_cuenta_llega_a_model_notes_axi_y_a_los_caveats_de_cada_perfil():
    """Lo que consume el panel «¿Cómo se ha calculado?»: `catalog()["model_notes_axi"]` (lente AXI) y los `caveats`
    de CADA perfil de `catalog()["firms"]` (lente prop firm), con la forma `{id, titulo, texto, sesgo}` y el texto de E2."""
    from poc_web.api.main import catalog
    cat = catalog()
    esperado = {"id": "lote_sobre_cuenta", "titulo": _E2_TITULO, "texto": _E2_TEXTO, "sesgo": "neutral"}
    assert esperado in cat["model_notes_axi"]
    assert cat["firms"], "el catálogo no trae perfiles"
    for firm in cat["firms"]:
        assert esperado in firm["caveats"], firm["key"]


# ===========================================================================
# guardrail `csv-d1-d2`, fase 2: nota `flotante_por_minuto` (AC2.8, plan §2.4) y fuente de `axi_lote_minimo` (E17)
# ===========================================================================
import re   # noqa: E402

# Texto del brief de la fase 2, ajustado por el orquestador tras la revisión de código (C7, 2026-10-19): en la
# evaluación por etapas los trades sin M1 no entran en los bloques (`build_block_records`), así que allí no se acotan.
_TEXTO_BRIEF = (
    "El peor momento de cada día suma lo ya cerrado ese día y el flotante de las posiciones abiertas, minuto a minuto. "
    "En los retos de prop firm, las órdenes sin precios de 1 minuto se acotan con su peor flotante (MAE) desde la "
    "apertura; en la evaluación por etapas quedan fuera de la simulación (ver la cobertura al cargar la cartera)")


def _flot():
    return next(n for n in NOTES if n.id == "flotante_por_minuto")


def test_ac2_8_flotante_por_minuto_se_carga_con_el_texto_del_plan():
    n = _flot()
    assert n.titulo == "El peor momento del día, minuto a minuto"
    assert n.texto == _TEXTO_BRIEF + "."
    assert "{" not in n.texto and "}" not in n.texto
    assert n.fuente and "csv-d1-d2" in n.fuente
    recargadas = load_notes()                                  # el loader estricto la acepta y la deja igual
    assert recargadas[[x.id for x in recargadas].index("flotante_por_minuto")] == n


def test_ac2_8_flotante_por_minuto_ambito_y_sesgo():
    """Las dos lentes; en prop firm solo el modelo cycle (los perfiles estáticos); todas las compañías, fases y perfiles."""
    n = _flot()
    assert set(n.lente) == {"propfirm", "axi"}
    assert n.modelo == ("cycle",) and n.tipo == ("cfd",)
    assert n.empresa == ("*",) and n.fases == ("*",) and n.perfil == ("*",)
    # neutral (revisión de código C1, 2026-10-19): la envolvente muere en el minuto de cierre (E10) y un CSV sin MAE
    # cuenta 0, así que no es conservadora en todos los casos.
    assert n.sesgo == "neutral"
    assert n.en_esta_corrida is None and n.texto_plantilla is None


def test_ac2_8_flotante_por_minuto_sale_en_axi_y_en_cycle_y_no_en_ladder_ni_futures():
    assert "flotante_por_minuto" in {n.id for n in notes_for("axi")}
    assert "flotante_por_minuto" in {n.id for n in notes_for("axi", tipo="cfd")}
    for empresa, fases in (("FTMO", 2), ("FTMO", 1), ("FundedNext", 2), ("Apex", 1), ("Otra casa", 2)):
        ids = {n.id for n in notes_for("propfirm", "cycle", "cfd", empresa=empresa, fases=fases)}
        assert "flotante_por_minuto" in ids, (empresa, fases)
    assert "flotante_por_minuto" in {n.id for n in notes_for("propfirm", "cycle", "cfd")}
    assert "flotante_por_minuto" in {n.id for n in notes_for("propfirm", "cycle", "cfd", perfil={"usuario", "copia"})}
    for modelo, tipo, empresa in (("ladder", "futures", "Apex"), ("futures", "futures", "FTMO"),
                                  ("ladder", "futures", None), ("futures", "futures", None)):
        ids = {n.id for n in notes_for("propfirm", modelo, tipo, empresa=empresa, fases=1 if empresa else None,
                                       perfil={"usuario", "copia"})}
        assert "flotante_por_minuto" not in ids, (modelo, empresa)


def test_ac2_8_flotante_por_minuto_no_nombra_a_ninguna_compania_ni_lleva_markdown():
    """Invariante AC5 de perfiles-v2: una nota mostrada a otra compañía no nombra a FTMO (ni a ninguna); y la app pinta el
    texto tal cual, así que sin markdown."""
    n = _flot()
    visible = f"{n.titulo} {n.texto}".lower()
    for nombre in ("ftmo", "apex", "axi", "fundednext", "topstep"):
        assert nombre not in visible, nombre
    assert not any(ch in n.titulo + n.texto for ch in "`*#_[]<>")
    for comp in ("FundedNext", "Apex", "Otra"):
        notas = notes_for("propfirm", "cycle", "cfd", empresa=comp, fases=2, perfil={"usuario"})
        assert "flotante_por_minuto" in {x.id for x in notas} and "ftmo" not in _txt(notas), comp


def test_ac2_8_flotante_por_minuto_llega_a_model_notes_axi_y_a_los_caveats_de_los_perfiles_cycle():
    """Lo que consume el panel «¿Cómo se ha calculado?»: `catalog()["model_notes_axi"]` y los `caveats` de cada perfil."""
    from poc_web.api.main import catalog
    cat = catalog()
    esperado = {"id": "flotante_por_minuto", "titulo": _flot().titulo, "texto": _flot().texto, "sesgo": "neutral"}
    assert esperado in cat["model_notes_axi"]
    cycle = [f for f in cat["firms"] if f["funded_model"] == "cycle"]
    otros = [f for f in cat["firms"] if f["funded_model"] != "cycle"]
    assert cycle and otros
    for firm in cycle:
        assert esperado in firm["caveats"], firm["key"]
    for firm in otros:
        assert "flotante_por_minuto" not in {c["id"] for c in firm["caveats"]}, firm["key"]


def test_e17_la_fuente_de_axi_lote_minimo_cita_el_bloque_de_simulate_f_por_nombre():
    n = next(x for x in NOTES if x.id == "axi_lote_minimo")
    assert "Bounds fieles al MQL5 de SQX" in n.fuente and "simulate_f" in n.fuente
    assert not re.search(r"mm\.py:\d", n.fuente)               # sin números de línea: envejecen
    mm = (Path(__file__).resolve().parents[1] / "src" / "risk" / "mm.py").read_text(encoding="utf-8")
    assert "Bounds fieles al MQL5 de SQX" in mm                # y el bloque existe con ese nombre


# ===========================================================================
# guardrail `entrada-4-bloques`, fase F4: notas de la receta para el EA (AC4.4)
# ===========================================================================
import json   # noqa: E402
import subprocess   # noqa: E402

_NUEVAS_F4 = ("receta_ea", "fichas_broker", "axi_sqx_flotante")
BASE_COMMIT = "b41903a"
# Resumen (sha256, 16 primeros hex, del JSON canónico: ``sort_keys``, ``ensure_ascii=False``) de la CABECERA y de cada
# entrada de ``config/model_notes.yaml`` tal como la parsea ``yaml.safe_load`` en el commit BASE_COMMIT, antes de añadir las
# notas de la F4. Si una nota vieja cambia (texto, ámbito, sesgo, plantilla o fuente) este pin lo dice con su id.
BASE_HEADER_DIGEST = "7f95c2626a08acd2"
BASE_ENTRY_DIGEST = {
    "remuestreo_bloques": "1116db312f64db52", "tamano_bloque": "2cf17fa02a30502a", "pocos_bloques": "6e693f043666706b",
    "edge_del_backtest": "6ec27cd1b82510b9", "costes_backtest": "e30db5df060d3fa3",
    "lote_sobre_cuenta": "91eb2084c8526e85", "flotante_por_minuto": "a2041ea4b179b2f6",
    "ev_condicional_resolver": "b1bb77ad2bc1fc5d", "fases_continuas": "8d5b3df3e58fead6",
    "espera_entre_fases": "90846cb603ad0c79", "reglas_de_usuario": "dd6f4bd6af5d9675",
    "plantilla_base": "07d4a5132088658b", "futures_ev_incondicional": "a6fd77ba4b6134f7",
    "ftmo_post_objetivo": "27b3d85a37c9d1cf", "ftmo_cobros_ciclos": "524d6b63fa86a304",
    "cobros_ciclos_otras": "6e5bfde5a66396f9", "ftmo_horizonte_fondeada": "6309a2aa9be3907d",
    "horizonte_fondeada_otras": "092bb3d81109e59e", "ftmo_noticias_finde": "d41e4c888226836a",
    "conducta_otras": "5fd3c23e0c607722", "ftmo_split_fijo": "e32d3188bd6e3352", "split_fijo_otras": "0635e716f79bafb6",
    "ftmo_eje_dias": "fad8dee723ad7f74", "eje_dias_otras": "82026b0abb81a9af",
    "ftmo_sin_consistencia": "f9c3fc5e26c7672d", "ftmo_1f_consistencia": "d80203a16b182ce8",
    "consistencia_otras": "5e9c30815aee276f", "futures_1": "909879945b831336", "futures_2": "e82fc8d2dc343fec",
    "futures_3": "d52faecf007b7391", "futures_4": "313c83570d63b469", "futures_5": "3f7951ef2675e5af",
    "futures_6": "d406eb7ec22c649b", "apex_1": "9c27bba7243393e4", "apex_2": "c994a2cbc092b095",
    "apex_3": "4b02bf42c24387c3", "apex_4": "93b736323673c814", "apex_5": "002143cbc260fe52",
    "apex_6": "5ccf5fb752fcd927", "apex_7": "cdcb7a37f1869917", "axi_regla_pase": "03042800c61114a3",
    "axi_post_objetivo": "073ffa0d19e57d04", "axi_lote_minimo": "319a3ff9fa248766",
    "axi_dias_calendario": "0eac0a44d199c9c3", "axi_puertas": "9a399b5adcee850e",
}


# Entradas del YAML que cambiaron DESPUÉS de BASE_COMMIT por decisión de un plan (id -> digest de la entrada nueva). Todas las
# demás siguen siendo, byte a byte, las de la base. `tamano_bloque`: guardrail bloque-politis-white, 2026-10-21 (plan §2.3).
CHANGED_SINCE_BASE = {"tamano_bloque": "11e672d69a7dd038"}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _raw_yaml() -> dict:
    return yaml.safe_load(NOTES_PATH.read_text(encoding="utf-8"))


def test_ac4_4_las_notas_nuevas_existen_van_al_final_y_con_el_ambito_y_el_sesgo_del_plan():
    by = {n.id: n for n in NOTES}
    assert set(_NUEVAS_F4) <= set(by)
    assert [n.id for n in NOTES][-3:] == list(_NUEVAS_F4)               # al final del YAML: las viejas no se mueven
    for nid in ("receta_ea", "fichas_broker"):                           # prop firm y AXI, todos los modelos y tipos
        n = by[nid]
        assert set(n.lente) == {"propfirm", "axi"} and n.modelo == ("*",) and n.tipo == ("*",), nid
        assert n.empresa == ("*",) and n.fases == ("*",) and n.perfil == ("*",) and n.sesgo == "neutral", nid
    n = by["axi_sqx_flotante"]                                           # solo la lente AXI (CFD)
    assert n.lente == ("axi",) and n.modelo == ("*",) and n.tipo == ("cfd",)
    assert n.empresa == ("*",) and n.fases == ("*",) and n.perfil == ("*",)
    assert n.sesgo == "conservador"                                      # «pesimista» del brief = conservador del enumerado
    for nid in _NUEVAS_F4:                                               # catálogo: sin plantilla por corrida ni de nombre
        assert by[nid].en_esta_corrida is None and by[nid].texto_plantilla is None, nid
        assert by[nid].titulo and by[nid].texto and by[nid].fuente, nid
    # salen en el catálogo de las dos lentes (y solo donde toca)
    assert [x.id for x in notes_for("axi")][-3:] == list(_NUEVAS_F4)
    for modelo, tipo in (("cycle", "cfd"), ("futures", "futures"), ("ladder", "futures")):
        ids = [x.id for x in notes_for("propfirm", modelo, tipo)]
        assert ids[-2:] == ["receta_ea", "fichas_broker"] and "axi_sqx_flotante" not in ids, (modelo, tipo)


def test_ac4_4_el_texto_de_las_notas_nuevas_dice_lo_que_pide_el_brief_y_no_lleva_nombres_ni_markdown():
    by = {n.id: n for n in NOTES}
    r = by["receta_ea"].texto.lower()
    # E18: LotsIfNoMM = lote mínimo es la CONDICIÓN (no un «recomendado»); lote entero con el 1,0 por defecto; órdenes sin SL
    assert "lotsifnomm" in r and "lote mínimo" in r and "condición" in r and "redondea a cero" in r
    assert "1,0" in r and "lote entero" in r and "sin stop" in r and "recomend" not in r
    # si cambias la f, regenera la receta; las órdenes abiertas a la vez suman riesgo; el MM va en el EA, sin re-minar
    assert "si cambias la f" in r and "genera la receta de nuevo" in r and "maxlots" in r
    assert "suman su riesgo" in r and "volver a minar" in r and "fixedamount" in r
    # H5: el MM se configura en el EA que se despliega, y el del backtest (el del .sqx, que sí lo lleva) no hay que cambiarlo
    assert ("el mm se configura en el ea que despliegas (fixedamount con estos valores); no hace falta cambiar el mm del "
            "backtest ni volver a minar") in r and "no en el .sqx" not in r
    # E14: confirma la moneda de la cuenta antes de teclear RiskedMoney (v1 solo USD)
    assert "mt5" in r and "usd" in r and "riskedmoney" in r and "no vale" in r
    f = by["fichas_broker"].texto.lower()
    assert "ficha" in f and "bróker" in f and "un solo bróker" in f and "rechaza" in f
    assert "vuelve a testearlas en sqx" in f and "sin verificar" in f
    a = by["axi_sqx_flotante"].texto.lower()
    assert "1 minuto" in a and "spread del propio backtest" in a                       # E20
    assert "cota diaria" in a and "mae" in a and "más pesimista" in a                  # operaciones sin barras
    assert "neto de comisión y de swap" in a and "csv" in a and "bruto" in a           # valle neto frente al bruto del CSV
    # lo que ve cualquier perfil no nombra compañías (AC5 de perfiles-v2); nada de modelos de IA ni de rutas personales
    for nid in ("receta_ea", "fichas_broker"):
        visible = f"{by[nid].titulo} {by[nid].texto}".lower()
        for nombre in ("ftmo", "apex", "axi", "fundednext", "topstep", "mffu"):
            assert nombre not in visible, (nid, nombre)
    for nid in _NUEVAS_F4:
        assert not any(ch in by[nid].titulo + by[nid].texto for ch in "`*#_[]<>{}"), nid   # la app pinta el texto tal cual


def test_ac4_4_las_notas_nuevas_pasan_los_escaneos_del_export_sin_nombrar_modelos_ni_rutas_personales():
    """Nombres de modelos de IA, de personas, rutas personales y documentos privados: los patrones los define
    ``scripts/export_shareable/manifest.yaml`` (los mismos del escaneo duro y blando del export), así que este test no repite
    ninguno de esos nombres. Se aplican al título, al texto y a la fuente de las notas nuevas."""
    manifest = Path(__file__).resolve().parents[1] / "scripts" / "export_shareable" / "manifest.yaml"
    if not manifest.is_file():
        pytest.skip("manifest del export ausente")
    m = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    pats = [re.compile(p) for p in m["hard_patterns"]] + [re.compile(p) for p in m["soft_patterns"].values()]
    assert len(pats) >= 10
    for n in NOTES:
        if n.id in _NUEVAS_F4:
            for campo in (n.titulo, n.texto, n.fuente):
                assert not [p.pattern for p in pats if p.search(campo)], (n.id, campo[:60])


def test_ac4_4_las_notas_existentes_son_byte_a_byte_las_de_la_base():
    """El YAML parseado de CADA nota vieja (y la cabecera) es el de ``BASE_COMMIT``: ni un carácter de texto, ámbito, sesgo,
    plantilla ni fuente; mismo orden y mismos ids, con las 3 nuevas detrás."""
    raw = _raw_yaml()
    assert _digest({k: v for k, v in raw.items() if k != "notas"}) == BASE_HEADER_DIGEST
    entries = raw["notas"]
    assert len(entries) == len(BASE_ENTRY_DIGEST) + len(_NUEVAS_F4) == 48
    assert [e["id"] for e in entries] == [*BASE_ENTRY_DIGEST, *_NUEVAS_F4]
    assert set(CHANGED_SINCE_BASE) <= set(BASE_ENTRY_DIGEST)
    for e in entries[:len(BASE_ENTRY_DIGEST)]:
        want = CHANGED_SINCE_BASE.get(e["id"], BASE_ENTRY_DIGEST[e["id"]])
        assert _digest(e) == want, f"la nota vieja '{e['id']}' cambió respecto a {BASE_COMMIT}"
    # y las que cambiaron lo hicieron de verdad (el pin de la base ya no es el vigente)
    for nid in CHANGED_SINCE_BASE:
        assert CHANGED_SINCE_BASE[nid] != BASE_ENTRY_DIGEST[nid]


def _entry_span(raw: bytes, nid: str) -> tuple[int, int]:
    """(inicio, fin) en bytes de la entrada ``- id: <nid>`` de la lista de notas: hasta la siguiente ``  - id:``."""
    start = raw.index(f"  - id: {nid}\n".encode())
    nxt = raw.find(b"\n  - id: ", start + 1)
    return start, (len(raw) if nxt < 0 else nxt + 1)


def _with_current_entries(base: bytes, now: bytes, ids: tuple[str, ...]) -> bytes:
    """Los bytes de la base con las entradas ``ids`` sustituidas por las actuales: lo único que se autoriza a cambiar desde
    BASE_COMMIT (``CHANGED_SINCE_BASE``). Todo lo demás se sigue comparando byte a byte."""
    out = base
    for nid in ids:
        b0, b1 = _entry_span(out, nid)
        n0, n1 = _entry_span(now, nid)
        out = out[:b0] + now[n0:n1] + out[b1:]
    return out


def test_ac4_4_el_yaml_empieza_por_los_mismos_bytes_que_el_de_la_base_en_git():
    """Comparación literal con la versión de ``BASE_COMMIT`` en git (solo donde el repo tiene ese commit: en una copia
    exportada sin historia se omite; el pin de ``BASE_ENTRY_DIGEST`` cubre ese caso). Las notas nuevas se añaden AL FINAL,
    así que el fichero actual EMPIEZA por los bytes del de la base y el YAML parseado de los ids viejos coincide. Salvo las
    entradas de ``CHANGED_SINCE_BASE`` (cambiadas por un plan): esas se sustituyen por la actual en los bytes de la base."""
    root = Path(__file__).resolve().parents[1]
    try:
        cp = subprocess.run(["git", "-C", str(root), "show", f"{BASE_COMMIT}:config/model_notes.yaml"],
                            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git no disponible")
    if cp.returncode != 0:
        pytest.skip(f"el commit base {BASE_COMMIT} no está en este repo")
    now = NOTES_PATH.read_bytes()
    base_bytes = _with_current_entries(cp.stdout, now, tuple(CHANGED_SINCE_BASE))
    assert now.startswith(base_bytes) and len(now) > len(base_bytes)
    base = yaml.safe_load(cp.stdout.decode("utf-8"))
    cur = _raw_yaml()
    assert {k: v for k, v in cur.items() if k != "notas"} == {k: v for k, v in base.items() if k != "notas"}
    changed = set(CHANGED_SINCE_BASE)
    assert [e for e in cur["notas"][:len(base["notas"])] if e["id"] not in changed] == \
        [e for e in base["notas"] if e["id"] not in changed]


# ===========================================================================
# guardrail `bloque-politis-white` (2026-10-21), plan §2.3 y E8: la única nota vieja que cambia
# ===========================================================================
def test_bloque_politis_white_solo_tamano_bloque_cambio_y_dice_lo_que_pide_el_plan():
    assert set(CHANGED_SINCE_BASE) == {"tamano_bloque"}
    n = next(x for x in NOTES if x.id == "tamano_bloque")
    t = n.texto
    assert t.startswith("Para no romper la dependencia entre bloques seguidos, los bloques se toman en grupos de longitud variable.")
    assert "método de Politis y White (2004, con la corrección de 2009)" in t
    assert "resultado de cada bloque medido en riesgos (R)" in t and "No depende de la f ni de la rejilla" in t
    assert "menos de 25 bloques" in t and "más del 10 % de las operaciones" in t          # E8: los dos umbrales declarados
    for viejo in ("ACF", "agrupación de volatilidad", "entre 1, 2, 3, 5 y 10", "calibrar"):
        assert viejo not in t
    assert n.en_esta_corrida == "Grupos de {bs} bloques de media ({bs_modo}): {bs_razon}."
    assert n.fuente == "src/risk/block_length.py; poc_web/api/main.py (_episode_r_series y bloque automático)"
    assert note_hash(n) == _TAMANO_BLOQUE[1]
