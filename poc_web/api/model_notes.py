"""Catálogo de notas del modelo (guardrail ``onboarding-b1``, fase B1-1).

Fuente única de los supuestos del modelo que la app muestra "junto a cada número":
``config/model_notes.yaml``. La API (catálogo y ``meta.model_context``) y, más
adelante, la guía HTML consumen este módulo; no hay textos duplicados en ``main.py``.

Formato del YAML (ver el propio fichero):

* ``id``: único, snake_case.
* ``ambito``: ``lente`` (propfirm | axi), ``modelo`` (cycle | futures | ladder | "*",
  solo lente propfirm) y ``tipo`` (cfd | futures | "*"). Opcionales (guardrail ``perfiles-v2``):
  ``empresa`` (lista de compañías, ``otras`` = cualquiera fuera de ``empresas_con_notas`` del modelo del perfil;
  ``"*"`` por defecto), ``fases`` (lista de enteros = nº de fases de evaluación; ``"*"`` por
  defecto) y ``perfil`` (``usuario`` | ``copia``; ``"*"`` por defecto).
* ``sesgo``: dirección del error del número (neutral | optimista | conservador).
* ``texto``: estático, SIN ``{`` ni ``}``. Excepción: ``texto_plantilla`` (en lugar de
  ``texto``) con el único placeholder ``{base_nombre}``; lo rellena :meth:`Note.as_catalog`.
* ``en_esta_corrida``: plantilla opcional con ``{placeholders}`` del conjunto cerrado
  :data:`PLACEHOLDERS`, que rellena la API por corrida (:func:`render_run`).
* ``fuente``: trazabilidad (no se muestra en la app).

Validación estricta (I5): cualquier incumplimiento lanza ``ValueError`` con el id de la
nota. Render estricto (I7): :func:`render_run` nunca rellena a medias.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = ROOT / "config" / "model_notes.yaml"

LENTES = frozenset({"propfirm", "axi"})
MODELOS = frozenset({"cycle", "futures", "ladder", "*"})
TIPOS = frozenset({"cfd", "futures", "*"})
SESGOS = frozenset({"neutral", "optimista", "conservador"})
# PLAN-DESVIO: el plan (Enm. 2 y 3) no define una dimension del ambito para que `reglas_de_usuario` (solo
# perfiles de usuario) y `plantilla_base` (solo copias de otra compania) salgan donde toca; se anade la
# clave opcional `perfil` ("*" por defecto) con las etiquetas `usuario` y `copia`.
PERFILES = frozenset({"usuario", "copia", "*"})
OTRAS = "otras"
# PLAN-DESVIO: `plantilla_base` lleva el nombre del oficial en el texto; la nota usa `texto_plantilla` (en lugar
# de `texto`, que no admite llaves) con el unico placeholder `{base_nombre}`, estricto como `render_run`.
# Placeholder único de ``texto_plantilla`` (nombre del oficial cuya mecánica copia el perfil).
PLANTILLA_PLACEHOLDERS = frozenset({"base_nombre"})
# Conjunto cerrado de placeholders que la API sabe rellenar (plan onboarding-b1 §2.3).
PLACEHOLDERS = frozenset({
    "n_blocks", "bs", "bs_modo", "bs_razon", "span_dias", "cap_dias",
    "pct_cortadas_datos", "min_dias", "min_trades", "f_calibracion",
})


@dataclass(frozen=True)
class Note:
    id: str
    lente: tuple[str, ...]
    modelo: tuple[str, ...]
    tipo: tuple[str, ...]
    sesgo: str
    titulo: str
    texto: str
    en_esta_corrida: str | None
    fuente: str
    empresa: tuple[str, ...] = ("*",)
    fases: tuple = ("*",)              # enteros o "*"
    perfil: tuple[str, ...] = ("*",)
    texto_plantilla: str | None = None

    def as_catalog(self, base_nombre: str | None = None) -> dict:
        """Forma que sirve el catálogo: ``{id, titulo, texto, sesgo}``.

        Una nota con ``texto_plantilla`` exige ``base_nombre`` (render estricto: nunca deja
        ``{}`` sueltos)."""
        texto = self.texto
        if self.texto_plantilla is not None:
            if not base_nombre:
                raise ValueError(f"nota '{self.id}': texto_plantilla exige base_nombre")
            texto = self.texto_plantilla.format_map(_Strict({"base_nombre": base_nombre}))
        return {"id": self.id, "titulo": self.titulo, "texto": texto, "sesgo": self.sesgo}


def _placeholders(template: str, nid: str) -> set[str]:
    names: set[str] = set()
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:  # llaves desbalanceadas
        raise ValueError(f"nota '{nid}': en_esta_corrida mal formada ({exc})") from exc
    for _lit, field, spec, conv in parsed:
        if field is None:
            continue
        if field == "" or not field.isidentifier():
            raise ValueError(
                f"nota '{nid}': placeholder inválido '{{{field}}}' en en_esta_corrida")
        if spec or conv:
            raise ValueError(
                f"nota '{nid}': placeholder '{{{field}}}' con formato; usa solo {{nombre}}")
        names.add(field)
    return names


def _enum_list(value, allowed: frozenset, what: str, nid: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"nota '{nid}': ambito.{what} debe ser una lista no vacía")
    bad = [v for v in value if v not in allowed]
    if bad:
        raise ValueError(
            f"nota '{nid}': ambito.{what} con valores fuera del enumerado: {bad}")
    return tuple(value)


def _load_empresas(raw: dict, path) -> dict[str, tuple[str, ...]]:
    """``empresas_con_notas`` POR MODELO: ``{cycle: [..], ladder: [..], futures: [..]}``."""
    emp = raw.get("empresas_con_notas", {})
    if not isinstance(emp, dict):
        raise ValueError(f"{path}: empresas_con_notas debe ser un mapa modelo -> lista de compañías")
    out: dict[str, tuple[str, ...]] = {}
    for modelo, names in emp.items():
        if modelo not in MODELOS or modelo == "*":
            raise ValueError(f"{path}: empresas_con_notas con modelo desconocido: {modelo!r}")
        if not isinstance(names, list) or not all(isinstance(e, str) and e.strip() for e in names) \
                or OTRAS in names or "*" in names or len(set(names)) != len(names):
            raise ValueError(f"{path}: empresas_con_notas.{modelo} debe ser una lista de nombres "
                             f"únicos (sin '{OTRAS}' ni '*')")
        # Nombres ya normalizados (como compara _empresa_ok): " FTMO " no puede reclasificar en silencio.
        bad = [e for e in names if e != " ".join(e.split())]
        if bad:
            raise ValueError(f"{path}: empresas_con_notas.{modelo} con nombres sin normalizar: {bad!r}")
        out[modelo] = tuple(names)
    return out


def _opt_list(amb: dict, what: str, nid: str, check) -> tuple:
    """Lista opcional del ámbito (``"*"`` por defecto); ``check(v)`` valida cada elemento."""
    if what not in amb:
        return ("*",)
    value = amb[what]
    if value == "*":
        return ("*",)
    if not isinstance(value, list) or not value:
        raise ValueError(f"nota '{nid}': ambito.{what} debe ser una lista no vacía")
    bad = [v for v in value if v != "*" and not check(v)]
    if bad:
        raise ValueError(f"nota '{nid}': ambito.{what} con valores inválidos: {bad}")
    return tuple(value)


def load_empresas(path: Path | str = NOTES_PATH) -> dict[str, tuple[str, ...]]:
    """``empresas_con_notas`` del YAML (por modelo; ``otras`` = el resto de ese modelo)."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _load_empresas(raw if isinstance(raw, dict) else {}, path)


def load_notes(path: Path | str = NOTES_PATH) -> list[Note]:
    """Carga y valida el catálogo. Orden = orden del YAML."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "notas" not in raw:
        raise ValueError(f"{path}: falta la clave 'notas'")
    empresas_por_modelo = _load_empresas(raw, path)
    empresas = {e for names in empresas_por_modelo.values() for e in names}
    notes: list[Note] = []
    seen: set[str] = set()
    for item in raw["notas"] or []:
        nid = str(item.get("id", "")) if isinstance(item, dict) else ""
        if not nid:
            raise ValueError(f"{path}: nota sin id")
        if nid in seen:
            raise ValueError(f"nota '{nid}': id duplicado")
        seen.add(nid)
        has_tpl = "texto_plantilla" in item
        for key in ("ambito", "sesgo", "titulo", "fuente") + (() if has_tpl else ("texto",)):
            if key not in item:
                raise ValueError(f"nota '{nid}': falta '{key}'")
        if has_tpl and "texto" in item:
            raise ValueError(f"nota '{nid}': 'texto' y 'texto_plantilla' son excluyentes")
        amb = item["ambito"]
        if not isinstance(amb, dict):
            raise ValueError(f"nota '{nid}': ambito debe ser un mapa")
        lente = _enum_list(amb.get("lente"), LENTES, "lente", nid)
        modelo = _enum_list(amb.get("modelo"), MODELOS, "modelo", nid)
        tipo = _enum_list(amb.get("tipo"), TIPOS, "tipo", nid)
        if item["sesgo"] not in SESGOS:
            raise ValueError(f"nota '{nid}': sesgo '{item['sesgo']}' fuera del enumerado")
        empresa = _opt_list(amb, "empresa", nid, lambda v: isinstance(v, str) and (
            v == OTRAS or v in empresas))
        fases = _opt_list(amb, "fases", nid, lambda v: isinstance(v, int)
                          and not isinstance(v, bool) and v >= 1)
        perfil = _opt_list(amb, "perfil", nid, lambda v: v in PERFILES)
        tpl_texto = None
        if has_tpl:
            tpl_texto = str(item["texto_plantilla"])
            unknown = _placeholders(tpl_texto, nid) - PLANTILLA_PLACEHOLDERS
            if unknown:
                raise ValueError(f"nota '{nid}': placeholders desconocidos en texto_plantilla: "
                                 f"{sorted(unknown)}")
            texto = ""
        else:
            texto = str(item["texto"])
            if "{" in texto or "}" in texto:
                raise ValueError(f"nota '{nid}': 'texto' no admite '{{' ni '}}'")
        tpl = item.get("en_esta_corrida")
        if tpl is not None:
            tpl = str(tpl)
            unknown = _placeholders(tpl, nid) - PLACEHOLDERS
            if unknown:
                raise ValueError(
                    f"nota '{nid}': placeholders desconocidos en en_esta_corrida: "
                    f"{sorted(unknown)}")
        notes.append(Note(
            id=nid, lente=lente, modelo=modelo, tipo=tipo, sesgo=item["sesgo"],
            titulo=str(item["titulo"]), texto=texto, en_esta_corrida=tpl,
            fuente=str(item["fuente"]), empresa=empresa, fases=fases, perfil=perfil,
            texto_plantilla=tpl_texto))
    return notes


# Se carga una vez al importar, como ``_firms`` en main.py.
NOTES: list[Note] = load_notes()
EMPRESAS_CON_NOTAS: dict[str, tuple[str, ...]] = load_empresas()


def normalize_company(value: str) -> str:
    """Compañía normalizada: ``strip`` y espacios internos colapsados."""
    return " ".join(str(value).split())


def _empresa_ok(n: Note, empresa: str, conocidas: tuple[str, ...]) -> bool:
    if "*" in n.empresa or empresa in n.empresa:
        return True
    return OTRAS in n.empresa and empresa not in conocidas


def _conocidas(por_modelo: dict[str, tuple[str, ...]], modelo: str | None) -> tuple[str, ...]:
    """Compañías con notas propias del modelo del perfil (sin modelo: la unión de todos)."""
    if modelo is not None:
        return por_modelo.get(modelo, ())
    return tuple(dict.fromkeys(e for names in por_modelo.values() for e in names))


def notes_for(lente: str, modelo: str | None = None, tipo: str | None = None,
              notes: list[Note] | None = None, *, empresa: str | None = None,
              fases: int | None = None, perfil: tuple[str, ...] | set[str] | None = None,
              empresas_con_notas: dict[str, tuple[str, ...]] | None = None) -> list[Note]:
    """Notas aplicables, en el orden del YAML.

    ``modelo`` solo se filtra en la lente propfirm. Un argumento ``None`` no filtra
    esa dimensión; ``"*"`` en el ámbito de una nota casa con cualquier valor.

    ``empresa`` (compañía del perfil) y ``fases`` (nº de fases de evaluación) filtran las
    dimensiones del mismo nombre; con ``empresa`` ``None`` no se filtra. ``perfil`` es el
    conjunto de etiquetas del perfil (``usuario``, ``copia``): una nota con ``perfil`` distinto
    de ``"*"`` solo sale si el perfil lleva alguna de sus etiquetas (``None`` = ninguna).
    """
    por_modelo = EMPRESAS_CON_NOTAS if empresas_con_notas is None else empresas_con_notas
    conocidas = _conocidas(por_modelo, modelo if lente == "propfirm" else None)
    tags = set(perfil or ())
    out = []
    for n in (NOTES if notes is None else notes):
        if lente not in n.lente:
            continue
        if lente == "propfirm" and modelo is not None \
                and "*" not in n.modelo and modelo not in n.modelo:
            continue
        if tipo is not None and "*" not in n.tipo and tipo not in n.tipo:
            continue
        if empresa is not None and not _empresa_ok(n, normalize_company(empresa), conocidas):
            continue
        if fases is not None and "*" not in n.fases and fases not in n.fases:
            continue
        if "*" not in n.perfil and not tags.intersection(n.perfil):
            continue
        out.append(n)
    return out


class _Strict(dict):
    """Dict que lanza KeyError ante una clave ausente (nunca deja ``{x}``)."""

    def __missing__(self, key):
        raise KeyError(key)


def render_run(note: Note, ctx: dict) -> str | None:
    """Rellena ``en_esta_corrida`` con ``ctx``.

    ``None`` si la nota no tiene plantilla o si falta alguna clave en ``ctx`` (o su
    valor es ``None``): nunca rellena a medias.
    """
    if note.en_esta_corrida is None:
        return None
    needed = _placeholders(note.en_esta_corrida, note.id)
    if any(ctx.get(k) is None for k in needed):
        return None
    try:
        return note.en_esta_corrida.format_map(_Strict(ctx))
    except (KeyError, IndexError, ValueError):
        return None
