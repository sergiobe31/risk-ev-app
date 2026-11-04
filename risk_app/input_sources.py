"""Fuentes de inputs para risk_app: escaneo local + emparejamiento CSV↔SQX por nombre.

Mantiene la lógica de descubrimiento de archivos fuera de Streamlit para que sea
unit-testeable y reutilizable entre el modo estrategia única y portfolio.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path


INPUTS_DIR = Path(__file__).resolve().parent / "inputs"
TRADES_DIR = INPUTS_DIR / "trades"
STRATEGIES_DIR = INPUTS_DIR / "strategies"


@dataclasses.dataclass
class LocalFile:
    """Wrapper mínimo para que un archivo local se comporte como un UploadedFile de Streamlit."""

    name: str
    path: Path

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def scan_inputs() -> tuple[list[Path], list[Path]]:
    """Devuelve (csvs, sqxs/xmls) encontrados en inputs/trades e inputs/strategies.

    Búsqueda recursiva: admite subcarpetas por activo (p. ej. ``nasdaq_2h/``, ``oro_4h/``).
    """
    csvs = sorted(TRADES_DIR.rglob("*.csv")) if TRADES_DIR.is_dir() else []
    sqxs: list[Path] = []
    if STRATEGIES_DIR.is_dir():
        sqxs = sorted(STRATEGIES_DIR.rglob("*.sqx")) + sorted(STRATEGIES_DIR.rglob("*.xml"))
    return csvs, sqxs


def match_by_name(csvs: list[Path], sqxs: list[Path]) -> tuple[dict[str, LocalFile], dict[str, LocalFile]]:
    """Empareja CSVs y SQXs por el nombre base (sin extensión).

    Devuelve (csv_by_stem, sqx_by_stem). Si hay colisiones de stem en sqxs, gana el
    primero en orden lexicográfico.
    """
    sqx_by_stem: dict[str, LocalFile] = {}
    for p in sqxs:
        sqx_by_stem.setdefault(p.stem, LocalFile(p.name, p))
    csv_by_stem = {p.stem: LocalFile(p.name, p) for p in csvs}
    return csv_by_stem, sqx_by_stem
