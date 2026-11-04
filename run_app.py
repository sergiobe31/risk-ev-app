#!/usr/bin/env python
"""Arranque de la app de riesgo en UN solo proceso (API + front compilado).

    python run_app.py                 # abre http://127.0.0.1:8600 en el navegador
    python run_app.py --no-browser    # sin abrir el navegador
    python run_app.py --port 8700     # otro puerto
    python run_app.py --smoke         # comprobacion automatica (sin abrir sockets); avisa si hay carpetas extra
    python run_app.py --smoke --strict  # exige copia limpia (sin carpetas extra, solo las 8 patas demo): CI

Sirve la API (/api/...) y, en la raiz, el front ya compilado (poc_web/web/dist). No hace falta Node.
Las rutas salen de la ubicacion de este fichero: se puede lanzar desde cualquier carpeta.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DIST = ROOT / "poc_web" / "web" / "dist"
EXPECTED_PYTHON = (3, 11)
ACCOUNT = 50000.0

# Demo sintetica: carpeta -> (perfil de la firma, clase de activo, nombres de las patas). Perfil "" = lente AXI
# (no usa perfil de firma: se simula la etapa Seed con su cuenta de 5.000).
DEMO = {
    "demo_cfd": ("ftmo_50k", "cfd", [
        "DEMO_sintetica_1_GBPUSD_ftmo", "DEMO_sintetica_2_GBPUSD_ftmo", "DEMO_sintetica_3_AUDUSD_ftmo"]),
    "demo_futuros": ("ftmo_futures_growth_50k", "futures", [
        "DEMO_sintetica_4_micro_nasdaq", "DEMO_sintetica_5_micro_oro", "DEMO_sintetica_6_micro_eurusd"]),
    "demo_axi": ("", "cfd", ["DEMO_sintetica_7_XAUUSD_axi", "DEMO_sintetica_8_EURUSD_axi"]),
}
N_DEMO = sum(len(names) for _f, _a, names in DEMO.values())
AXI_ACCOUNT = 5000.0
MIN_BLOCKS = 30


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def build_app():
    """La app FastAPI con el front compilado montado en `/` DESPUES de las rutas /api."""
    from fastapi.staticfiles import StaticFiles

    from poc_web.api import main as api_main

    if Path(api_main.ROOT).resolve() != ROOT:
        raise RuntimeError(f"poc_web.api.main se importa desde {api_main.ROOT}, no de {ROOT}")
    app = api_main.app
    if (DIST / "index.html").is_file() and not getattr(app.state, "web_mounted", False):
        app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
        app.state.web_mounted = True
    return app


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
class SmokeFailure(Exception):
    pass


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise SmokeFailure(msg)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def smoke(strict: bool = False) -> int:
    """Comprobacion completa con TestClient (sin sockets).

    Modo usuario (por defecto): /api/health 200, las 8 patas demo ESTAN en el catálogo (subconjunto) y las
    3 cargas + simulaciones demo (2 de prop firm y 1 de la lente AXI); si hay carpetas extra (config/local.yaml o
    RISK_APP_EXTRA_ROOTS) solo avisa.
    ``strict`` (CI y export): ademas `extra_roots == []` y EXACTAMENTE las 8 patas demo en el catálogo."""
    from fastapi.testclient import TestClient

    app = build_app()
    client = TestClient(app)
    try:
        if DIST.is_dir():
            r = client.get("/")
            _expect(r.status_code == 200 and "html" in r.headers.get("content-type", ""),
                    f"GET / -> {r.status_code} {r.headers.get('content-type')}")
            _say("OK    GET / -> 200 html")
        else:
            _say("SKIP  GET / (no hay poc_web/web/dist/: front sin compilar)")

        r = client.get("/api/health")
        _expect(r.status_code == 200 and r.json().get("status") == "ok", f"/api/health -> {r.status_code}")
        extra = r.json().get("extra_roots")
        if strict:
            _expect(extra == [], f"extra_roots = {extra} (se esperaba []): --strict exige una copia limpia, "
                    "sin config/local.yaml ni RISK_APP_EXTRA_ROOTS")
            _say("OK    GET /api/health -> 200, extra_roots == []")
        elif extra:
            _say(f"AVISO GET /api/health -> 200; hay carpetas extra configuradas ({len(extra)}): el catálogo "
                 "puede incluir tus propias patas (usa --strict para exigir una copia limpia)")
        else:
            _say("OK    GET /api/health -> 200, extra_roots == []")

        r = client.get("/api/catalog")
        _expect(r.status_code == 200, f"/api/catalog -> {r.status_code}")
        legs = r.json()["legs"]
        got = sorted((l["folder"], l["name"], l.get("asset")) for l in legs)
        want = sorted((folder, name, asset) for folder, (_f, asset, names) in DEMO.items() for name in names)
        if strict:
            _expect(got == want, f"catalog: patas {got} != demo esperada {want}")
            _say(f"OK    GET /api/catalog -> {len(legs)} patas (exactamente las {N_DEMO} de la demo)")
        else:
            missing = sorted(set(want) - set(got))
            _expect(not missing, f"catalog: faltan patas demo {missing}")
            _say(f"OK    GET /api/catalog -> {len(legs)} patas (las {N_DEMO} de la demo estan; "
                 f"{len(legs) - N_DEMO} mas)")

        for folder, (firm, asset, names) in DEMO.items():
            paths = sorted(l["path"] for l in legs if l["folder"] == folder and l["name"] in names)
            r = client.post("/api/portfolio/load",
                            json={"legs": paths, "account_size": ACCOUNT if firm else AXI_ACCOUNT,
                                  "firm_key": firm})
            _expect(r.status_code == 200, f"{folder}: portfolio/load -> {r.status_code}: {r.text[:300]}")
            body = r.json()
            errs = {k: v["detail"] for k, v in body["guards"].items() if v["status"] == "error"}
            _expect(body["ok"] and not errs, f"{folder}: guardas en error: {errs}")
            _expect(body["n_blocks"] >= MIN_BLOCKS, f"{folder}: {body['n_blocks']} bloques < {MIN_BLOCKS}")
            _say(f"OK    POST /api/portfolio/load {folder} + {firm or 'lente AXI'}: ok, sin guardas en error, "
                 f"{body['n_blocks']} bloques")

            if not firm:      # lente AXI (etapa Seed)
                r = client.post("/api/simulate/axi", json={
                    "portfolio_id": body["portfolio_id"], "stage": "seed",
                    "f_lo": 0.005, "f_hi": 0.0075, "f_step": 0.0025, "n_paths": 100})
                _expect(r.status_code == 200, f"{folder}: simulate/axi -> {r.status_code}: {r.text[:300]}")
                ps = [rep.get("p_pass") for rep in r.json()["reports"]]
                _expect(ps and all(_finite(p) and 0.0 <= p <= 1.0 for p in ps), f"{folder}: P(pasar) no valida: {ps}")
                _say(f"OK    POST /api/simulate/axi {folder} (etapa seed): 200, P(pasar) valida "
                     f"({', '.join(f'{p:.0%}' for p in ps)} en {len(ps)} f)")
                continue

            req = {"portfolio_id": body["portfolio_id"], "firm_key": firm,
                   "f_lo": 0.0025, "f_hi": 0.005, "f_step": 0.0025,
                   "n_paths": 100, "n_challenges": 3, "n_sequences": 100,
                   "day_axis": "ftmo" if asset == "cfd" else "servidor"}
            r = client.post("/api/simulate/propfirm", json=req)
            _expect(r.status_code == 200, f"{folder}: simulate/propfirm -> {r.status_code}: {r.text[:300]}")
            out = r.json()
            evs = [rep.get("ev_per_attempt") for rep in out["reports"]]
            _expect(evs and all(_finite(e) for e in evs), f"{folder}: EV no finito: {evs}")
            _say(f"OK    POST /api/simulate/propfirm {folder} + {firm}: 200, EV/intento finito "
                 f"({', '.join(f'{e:.0f}' for e in evs)} EUR en {len(evs)} f)")
    except SmokeFailure as exc:
        _say(f"FALLO {exc}")
        return 1
    _say("SMOKE OK")
    return 0


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Arranca la app de riesgo (API + front) en un proceso.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    ap.add_argument("--smoke", action="store_true", help="comprobacion automatica y salir")
    ap.add_argument("--strict", action="store_true",
                    help="con --smoke: exige copia limpia (extra_roots == [] y exactamente las 8 patas demo)")
    args = ap.parse_args(argv)

    if sys.version_info[:2] != EXPECTED_PYTHON:
        _say(f"AVISO: estas usando Python {sys.version_info.major}.{sys.version_info.minor}; las "
             f"versiones de requirements.txt estan verificadas con {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.")
    if args.strict and not args.smoke:
        ap.error("--strict solo tiene sentido junto a --smoke")
    if args.smoke:
        return smoke(strict=args.strict)
    if not (DIST / "index.html").is_file():
        _say("ERROR: falta el front compilado (poc_web/web/dist/index.html).\n"
             "  Esta copia deberia traerlo ya compilado: vuelve a descargar/clonar el repositorio.\n"
             "  Si estas desarrollando: cd poc_web/web && npm ci && npm run build (Node 20.19+ o 22.12+).")
        return 1

    import uvicorn

    app = build_app()
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}/"
    _say(f"Risk App en {url}  (Ctrl+C para parar)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
