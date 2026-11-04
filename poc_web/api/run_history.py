"""Historial de corridas (P4): JSONL append-only en ``poc_web/api/runs/``.

Cada registro: {id, ts, lens, portfolio_id, diag_group_id?, git_sha, request,
summary}. El ``summary`` lo elige el cliente (meta + f_star + recommendation +
reports) y NUNCA incluye trajectory ni objetos seq — historial ligero a
propósito. El path es sustituible en tests vía el atributo ``HISTORY``.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
HISTORY = RUNS_DIR / "history.jsonl"

_git_sha: str | None = None


def _sha() -> str | None:
    """git sha corto del árbol (cacheado; None fuera de un repo o si falla)."""
    global _git_sha
    if _git_sha is None:
        try:
            _git_sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, timeout=5).stdout.strip() or None
        except Exception:
            _git_sha = None
    return _git_sha


def append_run(rec: dict[str, Any]) -> dict[str, Any]:
    """Añade un registro (id/ts/git_sha asignados aquí) y lo devuelve."""
    out = {
        "id": uuid.uuid4().hex[:12],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lens": rec["lens"],
        "portfolio_id": rec["portfolio_id"],
        "diag_group_id": rec.get("diag_group_id"),
        "git_sha": _sha(),
        "request": rec.get("request") or {},
        "summary": rec.get("summary") or {},
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    return out


def list_runs(lens: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Últimos ``limit`` registros (desc); filtro opcional por lente.

    Append-only: se lee el fichero entero y se corta por el final (los
    volúmenes del PoC son pequeños; no hace falta índice).
    """
    if not HISTORY.exists():
        return []
    runs: list[dict[str, Any]] = []
    with HISTORY.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # línea truncada (p. ej. kill a mitad de escritura)
            if lens is None or r.get("lens") == lens:
                runs.append(r)
    return runs[-limit:][::-1]
