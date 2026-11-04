"""P4: historial de corridas (JSONL append-only) + endpoints /api/runs.

Cubre: roundtrip POST→GET, filtro por lente, límite, orden desc y
append-only (nunca trunca). El fichero real queda intacto: los tests
monkeypachean ``run_history.HISTORY`` a un tmp_path.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_run_history.py -q
"""
import pytest

from poc_web.api import run_history
from poc_web.api.main import RunIn, get_runs, post_run


@pytest.fixture()
def tmp_history(tmp_path, monkeypatch):
    monkeypatch.setattr(run_history, "HISTORY", tmp_path / "history.jsonl")
    # main.py importa las funciones directamente: redirigimos también su vista.
    import poc_web.api.main as m
    monkeypatch.setattr(m, "append_run", run_history.append_run)
    monkeypatch.setattr(m, "list_runs", run_history.list_runs)
    return run_history.HISTORY


def _rec(lens="propfirm", f_star=0.0075):
    return RunIn(
        lens=lens,
        portfolio_id="pf_test",
        request={"f_lo": 0.0025, "oos_start": None},
        summary={"f_star": f_star, "meta": {"n_blocks": 128}, "reports": [1, 2]},
    )


def test_roundtrip_y_forma(tmp_history):
    out = post_run(_rec())
    assert set(out) >= {"id", "ts", "lens", "portfolio_id", "git_sha",
                        "request", "summary"}
    assert out["lens"] == "propfirm"
    assert out["summary"]["f_star"] == 0.0075
    assert out["diag_group_id"] is None
    got = get_runs()["runs"]
    assert len(got) == 1 and got[0]["id"] == out["id"]


def test_filtro_lente_orden_desc(tmp_history):
    a = post_run(_rec(lens="propfirm"))
    b = post_run(_rec(lens="axi"))
    c = post_run(_rec(lens="propfirm"))
    todos = get_runs()["runs"]
    assert [r["id"] for r in todos] == [c["id"], b["id"], a["id"]]  # desc
    pf = get_runs(lens="propfirm")["runs"]
    assert [r["id"] for r in pf] == [c["id"], a["id"]]
    axi = get_runs(lens="axi")["runs"]
    assert [r["id"] for r in axi] == [b["id"]]


def test_limite(tmp_history):
    for _ in range(5):
        post_run(_rec())
    assert len(get_runs(limit=3)["runs"]) == 3


def test_append_only_no_trunca(tmp_history):
    post_run(_rec())
    first = tmp_history.read_text(encoding="utf-8")
    post_run(_rec())
    second = tmp_history.read_text(encoding="utf-8")
    assert second.startswith(first)  # append puro: nada de lo anterior se pierde
    assert len(get_runs()["runs"]) == 2


def test_diag_group_enlaza_gemela(tmp_history):
    g = "abc123"
    main = post_run(_rec())
    twin = RunIn(
        lens="propfirm", portfolio_id="pf_test", diag_group_id=g,
        request={"oos_start": None}, summary={"f_star": 0.01})
    post_run(twin)
    runs = {r["diag_group_id"]: r for r in get_runs()["runs"]}
    assert runs[None]["id"] == main["id"]
    assert runs[g]["request"]["oos_start"] is None
