"""Tests para risk_app/input_sources.py (escaneo y emparejamiento CSV↔SQX)."""
from pathlib import Path

import pytest

from risk_app.input_sources import LocalFile, match_by_name, scan_inputs


# Asegura que el test puede importar risk_app sin depender del CWD.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_inputs(tmp_path, monkeypatch):
    """Crea una estructura temporal de inputs/trades + inputs/strategies."""
    inputs = tmp_path / "inputs"
    trades = inputs / "trades"
    strategies = inputs / "strategies"
    trades.mkdir(parents=True)
    strategies.mkdir(parents=True)

    (trades / "strat_a.csv").write_text("csv a")
    (trades / "strat_b.csv").write_text("csv b")
    (trades / "orphan.csv").write_text("csv orphan")
    (strategies / "strat_a.sqx").write_text("sqx a")
    (strategies / "strat_b.xml").write_text("xml b")

    monkeypatch.setattr("risk_app.input_sources.TRADES_DIR", trades)
    monkeypatch.setattr("risk_app.input_sources.STRATEGIES_DIR", strategies)
    return inputs


def test_scan_inputs(sample_inputs):
    csvs, sqxs = scan_inputs()
    assert [p.name for p in csvs] == ["orphan.csv", "strat_a.csv", "strat_b.csv"]
    assert [p.name for p in sqxs] == ["strat_a.sqx", "strat_b.xml"]


def test_match_by_name(sample_inputs):
    csvs, sqxs = scan_inputs()
    csv_by_stem, sqx_by_stem = match_by_name(csvs, sqxs)

    assert set(csv_by_stem.keys()) == {"orphan", "strat_a", "strat_b"}
    assert set(sqx_by_stem.keys()) == {"strat_a", "strat_b"}

    assert csv_by_stem["strat_a"].name == "strat_a.csv"
    assert sqx_by_stem["strat_a"].name == "strat_a.sqx"
    assert sqx_by_stem["strat_b"].name == "strat_b.xml"
    assert "orphan" not in sqx_by_stem


def test_scan_inputs_nested_subfolders(sample_inputs):
    """Las subcarpetas por activo (p. ej. nasdaq_2h/, oro_4h/) también se escanean."""
    (sample_inputs / "trades" / "nasdaq_2h").mkdir()
    (sample_inputs / "trades" / "nasdaq_2h" / "strat_c.csv").write_text("csv c")
    (sample_inputs / "strategies" / "oro_4h").mkdir()
    (sample_inputs / "strategies" / "oro_4h" / "strat_c.sqx").write_text("sqx c")

    csvs, sqxs = scan_inputs()
    assert "strat_c.csv" in [p.name for p in csvs]
    assert "strat_c.sqx" in [p.name for p in sqxs]

    csv_by_stem, sqx_by_stem = match_by_name(csvs, sqxs)
    assert csv_by_stem["strat_c"].name == "strat_c.csv"
    assert sqx_by_stem["strat_c"].name == "strat_c.sqx"


def test_local_file_getvalue(sample_inputs):
    csvs, _ = scan_inputs()
    lf = LocalFile(csvs[0].name, csvs[0])
    assert lf.getvalue() == b"csv orphan"


def test_sxq_lookup_by_uploaded_csv_name(sample_inputs):
    """Simula el flujo real de portfolio_mode: un CSV subido busca su .sqx por stem."""
    _, sqx_by_stem = match_by_name([], scan_inputs()[1])

    uploaded_csv_names = ["strat_a.csv", "strat_b.csv", "no_sqx.csv"]
    pairs = []
    for name in uploaded_csv_names:
        stem = Path(name).stem
        sqx_lf = sqx_by_stem.get(stem)
        pairs.append((name, sqx_lf.name if sqx_lf else None))

    assert pairs == [
        ("strat_a.csv", "strat_a.sqx"),
        ("strat_b.csv", "strat_b.xml"),
        ("no_sqx.csv", None),
    ]
