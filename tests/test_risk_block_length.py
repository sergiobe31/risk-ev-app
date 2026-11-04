"""Longitud de bloque de Politis-White (guardrail ``bloque-politis-white``, fase A; plan §2.1 y enmiendas E3, E11, E12, E17).

``src/risk/block_length.py`` es un port a numpy de ``arch.bootstrap.base._single_optimal_block`` (arch 8.0.0, columna
"stationary"). Dos capas: ``politis_white_raw`` (literal, puede dar NaN o < 1) y ``politis_white_block`` / ``politis_white_info``
(la envoltura de la app: n >= 25, NaN/inf -> 1, acotada a [1, b_max]).

Mapa criterio -> test:

- AC-A1 golden frente a arch (importorskip: la copia exportada no lo tiene) con un caso cuyo valor crudo es < 1, y los MISMOS
  seis valores fijados para comprobarse sin arch ............................................. ``test_ac_a1_*``
- AC-A2 propiedades de la envoltura (E12): determinista, en [1, b_max], AR(1) mayor que iid, n < 25, serie constante,
  NaN/inf -> 1 ............................................................................... ``test_ac_a2_*``
- E11   cabecera con el aviso de copyright, las tres condiciones y la exclusión de garantía de arch ... ``test_e11_*``
- E17   sin arch en el runtime: ni ``import arch`` en ``src/`` y ``poc_web/api/*.py`` (salvo tests), ni en el requirements
  de la app, ni cargado al importar la API ..................................................... ``test_e17_*``
- AC-A3 guardianes: ``tests/test_risk_no_sqx_import.py`` y ``tests/test_export_shareable.py`` (su propio fichero).

Ejecutar desde la raíz del repo::

    PYTHONPATH=. python -m pytest tests/test_risk_block_length.py -q
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.risk import block_length
from src.risk.block_length import MIN_OBS, politis_white_block, politis_white_info, politis_white_raw

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "risk" / "block_length.py"


# ---------------------------------------------------------------------------
# Las seis series fijas (semilla 0) del AC-A1
# ---------------------------------------------------------------------------
def _ar1(rng, n: int, phi: float) -> np.ndarray:
    e = rng.standard_normal(n + 200)         # 200 de calentamiento
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def _regimes(rng, n: int) -> np.ndarray:
    """Regímenes de varianza persistentes: bloques de 50 con desviación 0,5 o 2,0."""
    sig = np.repeat(rng.choice([0.5, 2.0], size=n // 50 + 1), 50)[:n]
    return rng.standard_normal(n) * sig


RECIPES = {
    "iid_600": lambda: np.random.default_rng(0).standard_normal(600),
    "iid_2000": lambda: np.random.default_rng(0).standard_normal(2000),
    "ar1_p05_1000": lambda: _ar1(np.random.default_rng(0), 1000, 0.5),
    "ar1_pm03_1000": lambda: _ar1(np.random.default_rng(0), 1000, -0.3),
    "t3_800": lambda: np.random.default_rng(0).standard_t(3, size=800),
    "regimes_1500": lambda: _regimes(np.random.default_rng(0), 1500),
}

# Valores fijados, obtenidos con ``arch.bootstrap.optimal_block_length(x)["stationary"].iloc[0]`` (arch 8.0.0, numpy 2.4)
# sobre las series de arriba, más la huella de cada serie (Σx y Σ|x|) para distinguir un cambio del generador de numpy de un
# cambio del port. (valor crudo de arch, Σx, Σ|x|).
GOLDEN = {
    "iid_600": (1.0556930790812296, -13.627792910872552, 477.78953491609786),
    "iid_2000": (0.8271295446654939, -56.051171711712556, 1601.975345264271),
    "ar1_p05_1000": (12.480110709428274, -81.92635793250241, 898.0906118850783),
    "ar1_pm03_1000": (5.780268437379131, -32.35828119400345, 810.751868921776),
    "t3_800": (0.5621526682707599, 4.9868867917681, 864.6146602040476),
    "regimes_1500": (1.504986083554225, -28.346045771451486, 1734.0923204311566),
}
# De las seis, las que tienen valor CRUDO < 1 en arch (E3: la envoltura las sube a 1; el golden compara el crudo).
RAW_BELOW_ONE = {"iid_2000", "t3_800"}


@pytest.fixture(scope="module")
def series() -> dict[str, np.ndarray]:
    return {k: f() for k, f in RECIPES.items()}


# ---------------------------------------------------------------------------
# AC-A1: golden frente a arch (con arch) y goldens fijados (sin arch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(RECIPES))
def test_ac_a1_golden_frente_a_arch_a_1e_9(name, series):
    arch_bootstrap = pytest.importorskip("arch.bootstrap", exc_type=ImportError)   # la copia exportada no tiene arch
    x = series[name]
    ref = float(arch_bootstrap.optimal_block_length(x)["stationary"].iloc[0])
    assert politis_white_raw(x) == pytest.approx(ref, abs=1e-9, rel=0)
    assert ref == pytest.approx(GOLDEN[name][0], abs=1e-9, rel=0)   # y el golden fijado es el de arch


def test_ac_a1_al_menos_un_caso_con_valor_crudo_de_arch_menor_que_1(series):
    arch_bootstrap = pytest.importorskip("arch.bootstrap", exc_type=ImportError)
    below = {k for k, x in series.items()
             if float(arch_bootstrap.optimal_block_length(x)["stationary"].iloc[0]) < 1.0}
    assert below == RAW_BELOW_ONE and below                         # la lista fijada no se ha desfasado
    for k in below:                                                # y la capa literal NO lo sube a 1
        assert politis_white_raw(series[k]) < 1.0


def test_ac_a1_goldens_fijados_sin_arch(series):
    for name, (raw, s, s_abs) in GOLDEN.items():
        x = series[name]
        # la huella de la serie: si falla, cambió el generador de numpy, no el port
        assert float(x.sum()) == pytest.approx(s, rel=1e-9), f"{name}: la serie ya no es la fijada (generador de numpy)"
        assert float(np.abs(x).sum()) == pytest.approx(s_abs, rel=1e-9), name
        assert politis_white_raw(x) == pytest.approx(raw, abs=1e-9, rel=0), name
    assert {k for k, (raw, _s, _a) in GOLDEN.items() if raw < 1.0} == RAW_BELOW_ONE   # E3: hay casos crudos < 1


def test_ac_a1_la_capa_literal_y_la_envoltura_son_funciones_distintas(series):
    """Los goldens (capa literal) y las propiedades (envoltura) viven en tests distintos (E12): con un crudo < 1 las dos
    capas DIFIEREN, así que comparar la envoltura con arch sería un error de la prueba."""
    x = series["iid_2000"]
    assert politis_white_raw(x) < 1.0 and politis_white_block(x) == 1.0
    assert politis_white_info(x).b_raw == politis_white_raw(x)


# ---------------------------------------------------------------------------
# AC-A2: propiedades de la envoltura
# ---------------------------------------------------------------------------
def test_ac_a2_determinista(series):
    for x in series.values():
        a, b = politis_white_block(x), politis_white_block(x.copy())
        assert a == b
        np.random.seed(123)                                        # y no usa ningún generador aleatorio
        assert politis_white_block(x) == a


def test_ac_a2_resultado_en_1_y_b_max_y_coherente_con_la_capa_literal(series):
    extra = [np.sin(np.arange(60) / 3.0), np.cumsum(np.random.default_rng(3).standard_normal(80)),
             np.random.default_rng(4).standard_normal(25), np.ones(40) * 7.0]
    for x in list(series.values()) + extra:
        info = politis_white_info(x)
        n = len(x)
        assert info.n == n and info.b_max == math.ceil(min(3 * math.sqrt(n), n / 3))
        assert 1.0 <= info.b <= info.b_max
        assert info.b == politis_white_block(x)
        if math.isfinite(info.b_raw):
            assert info.b == min(max(info.b_raw, 1.0), info.b_max)   # solo acota; no cambia lo que arch da en [1, b_max]
        assert 1 <= info.m <= info.m_max
        assert info.m == (min(2 * max(info.opt_m, 1), info.m_max) if info.opt_m is not None else info.m_max)


def test_ac_a2_ar1_phi_05_da_bloque_mayor_que_iid_del_mismo_n():
    """Media de 20 réplicas (el estimador es ruidoso): con dependencia positiva el bloque medio crece mucho (E12: margen
    b_ar > 1,5 · b_iid)."""
    n = 1000
    b_ar = np.mean([politis_white_block(_ar1(np.random.default_rng(100 + k), n, 0.5)) for k in range(20)])
    b_iid = np.mean([politis_white_block(np.random.default_rng(200 + k).standard_normal(n)) for k in range(20)])
    assert b_ar > 1.5 * b_iid, (b_ar, b_iid)
    assert b_ar > 5.0 and b_iid < 3.0          # órdenes de magnitud: ~11 frente a ~1,5 con estas semillas


def test_ac_a2_menos_de_25_observaciones_es_value_error():
    for n in (0, 1, 10, MIN_OBS - 1):
        with pytest.raises(ValueError, match="al menos 25"):
            politis_white_block(np.random.default_rng(0).standard_normal(n))
        with pytest.raises(ValueError):
            politis_white_info(np.zeros(n))
    assert MIN_OBS == 25
    b = politis_white_block(np.random.default_rng(0).standard_normal(MIN_OBS))   # 25 sí
    assert 1.0 <= b <= math.ceil(min(3 * math.sqrt(25), 25 / 3))


def test_ac_a2_entrada_invalida_es_value_error():
    ok = np.random.default_rng(0).standard_normal(100)
    for bad in (np.nan, np.inf, -np.inf):
        x = ok.copy()
        x[10] = bad
        with pytest.raises(ValueError, match="no finitos"):
            politis_white_block(x)
    with pytest.raises(ValueError, match="1-D"):
        politis_white_block(ok.reshape(50, 2))


def test_ac_a2_serie_constante_no_rompe_nada_y_da_1():
    """Autocovarianzas 0 -> 0/0: el crudo de arch es NaN (sin avisos de numpy) y la envoltura lo convierte en 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")                            # ningún RuntimeWarning
        for c in (np.full(50, 3.0), np.zeros(30), np.full(400, -0.25)):
            assert math.isnan(politis_white_raw(c))
            info = politis_white_info(c)
            assert math.isnan(info.b_raw) and info.b == 1.0
            assert politis_white_block(c) == 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_ac_a2_un_resultado_nan_o_infinito_se_convierte_en_1(monkeypatch, bad):
    """arch no da infinitos (``min(b_sb, b_max)``), pero la envoltura los trata igual que un NaN."""
    monkeypatch.setattr(block_length, "_stationary_core", lambda x: (bad, 74.0, 1, 2, 30))
    assert politis_white_block(np.arange(100.0)) == 1.0
    info = politis_white_info(np.arange(100.0))
    assert info.b == 1.0 and info.m == 2 and info.opt_m == 1


def test_ac_a2_el_tope_superior_es_b_max(monkeypatch):
    monkeypatch.setattr(block_length, "_stationary_core", lambda x: (500.0, 74.0, 1, 2, 30))
    assert politis_white_block(np.arange(600.0)) == 74.0          # nunca por encima de b_max


# ---------------------------------------------------------------------------
# E11: la cabecera lleva el aviso de copyright y la licencia de arch
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(s.split())


def _docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False) or ""


COPYRIGHT = "Copyright (c) 2017 Kevin Sheppard"
GRANT = ('Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated '
         'documentation files (the "Software"), to deal with the Software without restriction, including without limitation '
         'the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and '
         'to permit persons to whom the Software is furnished to do so, subject to the following conditions:')
CONDITIONS = (
    "Redistributions of source code must retain the above copyright notice, this list of conditions and the following "
    "disclaimers.",
    "Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following "
    "disclaimers in the documentation and/or other materials provided with the distribution.",
    "Neither the names of Kevin Sheppard, nor the names of its contributors may be used to endorse or promote products "
    "derived from this Software without specific prior written permission.",
)
WARRANTY = ('THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED '
            'TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL '
            'THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION '
            'OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER '
            'DEALINGS WITH THE SOFTWARE.')


def test_e11_la_cabecera_lleva_copyright_las_tres_condiciones_y_la_exclusion_de_garantia():
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    header = _norm(ast.get_docstring(tree, clean=False) or "")
    assert header, "src/risk/block_length.py debe empezar por una docstring de módulo con la licencia"
    for needle in (COPYRIGHT, GRANT, *CONDITIONS, WARRANTY):
        assert needle in header, f"falta en la cabecera: {needle[:70]}..."
    # y es CABECERA: la licencia va antes de cualquier código (el primer import)
    assert src.index(COPYRIGHT) < src.index("from __future__ import annotations")


def test_e11_la_cabecera_coincide_con_el_license_md_de_arch_instalado():
    """Con arch instalado, la licencia de su distribución es el oráculo independiente: cada párrafo de su LICENSE.md (sin
    el formato markdown ni la línea ``Developed by``, que lleva correos y se conserva sin ellos) está en la cabecera."""
    pytest.importorskip("arch", exc_type=ImportError)
    from importlib import metadata

    lic = next((f for f in (metadata.files("arch") or []) if f.name == "LICENSE.md"), None)
    if lic is None:
        pytest.skip("la distribución de arch no incluye su LICENSE.md")
    paras = [p.replace("**", "").strip() for p in re.split(r"\n\s*\n", lic.read_text(encoding="utf-8"))]
    paras = [_norm(p) for p in paras if p and not p.startswith("#") and not p.startswith("Developed by")]
    assert len(paras) >= 6
    header = _norm(_docstring(MODULE))
    for p in paras:
        assert p in header, f"párrafo del LICENSE.md de arch ausente en la cabecera: {p[:70]}..."
    # y la línea de autoría conserva el nombre y la web, sin los correos
    assert "Developed by: Kevin Sheppard" in header and "https://www.kevinsheppard.com" in header
    assert "@" not in header and not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", header)


def test_e11_el_aviso_de_copyright_del_port_no_lleva_correos():
    """El escaneo duro del export aborta con cualquier dirección de correo (patrón de ``manifest.yaml``): ni en este módulo."""
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", MODULE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# E17: arch no es una dependencia del runtime (se COPIA el método)
# ---------------------------------------------------------------------------
def _imports_arch(tree: ast.AST) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if a.name == "arch" or a.name.startswith("arch.")]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "arch" or node.module.startswith("arch."):
                bad.append(f"from {node.module} import ...")
    return bad


def test_e17_ni_src_ni_la_api_importan_arch():
    files = sorted((ROOT / "src").rglob("*.py")) + sorted(
        p for p in (ROOT / "poc_web" / "api").glob("*.py") if not p.name.startswith("test_"))
    assert any(p.name == "block_length.py" for p in files) and any(p.name == "main.py" for p in files)
    offenders = []
    for p in files:
        for bad in _imports_arch(ast.parse(p.read_text(encoding="utf-8"), filename=str(p))):
            offenders.append(f"{p.relative_to(ROOT)}: {bad}")
    assert not offenders, "arch no puede ser una dependencia del runtime:\n  " + "\n  ".join(offenders)


def test_e17_arch_no_esta_en_los_requirements_de_la_app():
    """En el repo fuente: la plantilla del export; en la copia exportada: el ``requirements.txt`` de la raíz (que sale de ella)."""
    candidates = [ROOT / "scripts" / "export_shareable" / "templates" / "requirements.txt"]
    found = [p for p in candidates if p.is_file()] or [p for p in (ROOT / "requirements.txt",) if p.is_file()]
    assert found, "no se encontró el requirements de la app"
    for p in found:
        assert not re.search(r"(?im)\barch\b", p.read_text(encoding="utf-8")), f"arch aparece en {p.relative_to(ROOT)}"


def test_e17_importar_la_api_no_carga_arch():
    """Comportamiento, en un proceso limpio (el de pytest puede tener arch ya importado por los tests con importorskip)."""
    code = ("import sys\nsys.path.insert(0, %r)\nimport poc_web.api.main, src.risk.block_length\n"
            "bad = sorted(m for m in sys.modules if m == 'arch' or m.startswith('arch.'))\n"
            "print('ARCH_CARGADO' if bad else 'OK', bad)\n" % str(ROOT))
    cp = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=240, cwd=str(ROOT))
    assert cp.returncode == 0, cp.stderr[-1500:]
    assert cp.stdout.strip().startswith("OK"), cp.stdout
