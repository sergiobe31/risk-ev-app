"""Raíces extra de exportaciones SQX (``EXTRA_INPUT_ROOTS``), plan export-app-fase-a §2.1 (AC3).

Precedencia: env ``RISK_APP_EXTRA_ROOTS`` (``os.pathsep``) > ``config/local.yaml``
(clave ``extra_input_roots``) > ``[]``. ``expanduser`` en cada ruta; las rutas
inexistentes se conservan. ``/api/health`` expone ``extra_roots``.

Todo con ``tmp_path``/``monkeypatch``: no depende de la máquina ni de la
``config/local.yaml`` real del repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from poc_web.api import main

ENV = main.EXTRA_ROOTS_ENV
REPO = Path(__file__).resolve().parent.parent


def _write_yaml(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# --- precedencia -----------------------------------------------------------

def test_precedencia_env_gana_a_local_yaml(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": [str(tmp_path / "desde_yaml")]})
    roots, origin = main._resolve_extra_roots({ENV: str(tmp_path / "desde_env")}, ly)
    assert roots == [tmp_path / "desde_env"]
    assert origin == "env"


def test_precedencia_local_yaml_si_no_hay_env(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": [str(tmp_path / "desde_yaml")]})
    roots, origin = main._resolve_extra_roots({}, ly)
    assert roots == [tmp_path / "desde_yaml"]
    assert origin == "local.yaml"


def test_env_vacia_o_en_blanco_cae_a_local_yaml(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": [str(tmp_path / "y")]})
    for valor in ("", "   ", os.pathsep):
        roots, origin = main._resolve_extra_roots({ENV: valor}, ly)
        assert (roots, origin) == ([tmp_path / "y"], "local.yaml"), repr(valor)


def test_sin_env_ni_yaml_es_lista_vacia(tmp_path):
    roots, origin = main._resolve_extra_roots({}, tmp_path / "no_existe.yaml")
    assert roots == []
    assert origin == "ninguno"


def test_local_yaml_sin_la_clave_o_vacio(tmp_path):
    ly = _write_yaml(tmp_path / "a.yaml", {"otra_clave": 1})
    assert main._resolve_extra_roots({}, ly) == ([], "ninguno")
    vacio = tmp_path / "vacio.yaml"
    vacio.write_text("", encoding="utf-8")
    assert main._resolve_extra_roots({}, vacio) == ([], "ninguno")
    lista_vacia = _write_yaml(tmp_path / "b.yaml", {"extra_input_roots": []})
    assert main._resolve_extra_roots({}, lista_vacia) == ([], "ninguno")
    nula = tmp_path / "c.yaml"
    nula.write_text("extra_input_roots:\n", encoding="utf-8")
    assert main._resolve_extra_roots({}, nula) == ([], "ninguno")


# --- validación fail-fast de local.yaml ------------------------------------

def test_yaml_string_solitario_es_una_ruta(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": str(tmp_path / "sola")})
    assert main._resolve_extra_roots({}, ly) == ([tmp_path / "sola"], "local.yaml")


def test_yaml_escalar_int_es_valueerror_con_fichero(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": 7})
    with pytest.raises(ValueError) as exc:
        main._resolve_extra_roots({}, ly)
    msg = str(exc.value)
    assert str(ly) in msg and "extra_input_roots" in msg and "int" in msg


def test_yaml_bool_es_valueerror(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": True})
    with pytest.raises(ValueError, match="bool"):
        main._resolve_extra_roots({}, ly)


def test_yaml_dict_es_valueerror(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": {"a": 1}})
    with pytest.raises(ValueError) as exc:
        main._resolve_extra_roots({}, ly)
    assert str(ly) in str(exc.value) and "dict" in str(exc.value)


@pytest.mark.parametrize("malo", [None, {"a": 1}, [1, 2], True, ""])
def test_yaml_lista_con_elemento_invalido_es_valueerror(tmp_path, malo):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": ["/ok", malo]})
    with pytest.raises(ValueError) as exc:
        main._resolve_extra_roots({}, ly)
    assert str(ly) in str(exc.value)


def test_yaml_lista_con_null_literal_es_valueerror(tmp_path):
    ly = tmp_path / "local.yaml"
    ly.write_text("extra_input_roots:\n  - /ok\n  -\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NoneType"):
        main._resolve_extra_roots({}, ly)


def test_yaml_malformado_es_valueerror_con_fichero(tmp_path):
    ly = tmp_path / "local.yaml"
    ly.write_text("extra_input_roots: [a, b\n  : : {", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        main._resolve_extra_roots({}, ly)
    assert str(ly) in str(exc.value)
    assert isinstance(exc.value.__cause__, yaml.YAMLError)


def test_yaml_raiz_no_mapa_es_valueerror(tmp_path):
    ly = tmp_path / "local.yaml"
    ly.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        main._resolve_extra_roots({}, ly)


def test_env_valida_no_lee_el_yaml_roto(tmp_path):
    """La env gana antes de abrir el YAML (precedencia): un YAML roto no la afecta."""
    ly = tmp_path / "local.yaml"
    ly.write_text("{{{", encoding="utf-8")
    roots, origin = main._resolve_extra_roots({ENV: str(tmp_path / "e")}, ly)
    assert (roots, origin) == ([tmp_path / "e"], "env")


# --- separador, expanduser, inexistentes -----------------------------------

def test_env_con_dos_rutas_separadas_por_pathsep(tmp_path):
    a, b = tmp_path / "uno", tmp_path / "dos"
    roots, origin = main._resolve_extra_roots({ENV: f"{a}{os.pathsep}{b}"}, tmp_path / "x.yaml")
    assert roots == [a, b]
    assert origin == "env"
    # huecos y espacios alrededor se descartan
    roots, _ = main._resolve_extra_roots(
        {ENV: f" {a} {os.pathsep}{os.pathsep} {b}{os.pathsep}"}, tmp_path / "x.yaml")
    assert roots == [a, b]


def test_expanduser_en_env_y_en_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # Windows
    esperado = Path("~/x").expanduser()
    assert esperado != Path("~/x")
    roots, _ = main._resolve_extra_roots({ENV: "~/x"}, tmp_path / "x.yaml")
    assert roots == [esperado]
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": ["~/x"]})
    roots, origin = main._resolve_extra_roots({}, ly)
    assert roots == [esperado]
    assert origin == "local.yaml"


def test_ruta_inexistente_se_conserva(tmp_path):
    fantasma = tmp_path / "no" / "existe"
    assert not fantasma.exists()
    roots, _ = main._resolve_extra_roots({ENV: str(fantasma)}, tmp_path / "x.yaml")
    assert roots == [fantasma]
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": [str(fantasma)]})
    roots, _ = main._resolve_extra_roots({}, ly)
    assert roots == [fantasma]


def test_yaml_ruta_relativa_se_resuelve_contra_la_base(tmp_path):
    ly = _write_yaml(tmp_path / "local.yaml", {"extra_input_roots": ["datos/sqx"]})
    roots, _ = main._resolve_extra_roots({}, ly, base=tmp_path / "repo")
    assert roots == [tmp_path / "repo" / "datos" / "sqx"]
    # por defecto, la base es la raíz del repo
    roots, _ = main._resolve_extra_roots({}, ly)
    assert roots == [main.ROOT / "datos" / "sqx"]


def test_yaml_con_utf8(tmp_path):
    ly = tmp_path / "local.yaml"
    carpeta = str(tmp_path / "estrategías ñ")
    ly.write_bytes(yaml.safe_dump({"extra_input_roots": [carpeta]}, allow_unicode=True).encode("utf-8"))
    roots, _ = main._resolve_extra_roots({}, ly)
    assert roots == [Path(carpeta)]


# --- estado a nivel de módulo ----------------------------------------------

def test_modulo_expone_lista_de_path():
    assert isinstance(main.EXTRA_INPUT_ROOTS, list)
    assert all(isinstance(r, Path) for r in main.EXTRA_INPUT_ROOTS)
    assert main._EXTRA_ROOTS_ORIGIN in {"env", "local.yaml", "ninguno"}


def test_resolucion_al_importar_con_env(tmp_path):
    """Proceso nuevo: la env fija ``EXTRA_INPUT_ROOTS`` al importar el módulo."""
    a, b = tmp_path / "uno", tmp_path / "dos"
    code = ("import sys; from poc_web.api import main; "
            "print('|'.join(str(r) for r in main.EXTRA_INPUT_ROOTS)); "
            "print(main._EXTRA_ROOTS_ORIGIN)")
    env = {**os.environ, ENV: f"{a}{os.pathsep}{b}", "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=180, check=True)
    lineas = out.stdout.strip().splitlines()
    assert lineas[-2] == f"{a}|{b}"
    assert lineas[-1] == "env"


# --- /api/health -----------------------------------------------------------

def test_health_incluye_extra_roots(tmp_path, monkeypatch):
    existe = tmp_path / "existe"
    existe.mkdir()
    fantasma = tmp_path / "fantasma"
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [existe, fantasma])
    r = TestClient(main.app).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["extra_roots"] == [
        {"path": str(existe), "is_dir": True},
        {"path": str(fantasma), "is_dir": False},
    ]
    # claves existentes conservadas
    assert body["status"] == "ok"
    assert isinstance(body["n_symbols_broker"], int)
    assert isinstance(body["n_portfolios"], int)


def test_health_extra_roots_vacio(monkeypatch):
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [])
    body = TestClient(main.app).get("/api/health").json()
    assert body["extra_roots"] == []


@pytest.mark.parametrize("clave", ["status", "n_symbols_broker", "n_portfolios", "extra_roots"])
def test_health_claves(clave):
    assert clave in TestClient(main.app).get("/api/health").json()
