"""App web — shim FastAPI sobre el motor de riesgo (SOLO LECTURA).

Shell B (DESIGN_UI_V2.md §4.1): la CARTERA es la entidad central — se carga y
valida UNA vez (``POST /api/portfolio/load``, guardas con semáforo) y las
simulaciones consumen el ``portfolio_id`` resultante
(``POST /api/simulate/propfirm``, ``POST /api/simulate/axi``).

Sizing (F5a, PROPUESTA_F5_PRECISION_MOTOR.md §8): EXACTO por pata cuando hay
``.sqx`` emparejado (ATR real vía ``risk_app.exact_mm``, reuso read-only con
import perezoso — migrará a hogar neutro al retirar Streamlit); fallback
rápido etiquetado (warning por pata Y a nivel cartera) cuando falta el .sqx o
el parse falla. Los CUATRO sitios de sizing pasan por ``_leg_lots``.

Base del lote de las patas CSV (2026-10-19, guardrail ``csv-d1-d2``, defecto D1; plan
csv-d1-d2 §2.1 y Enmiendas v2 E1-E2): el lote de cada
orden se calcula sobre el TAMAÑO DE CUENTA (``account_size`` de la carga = ``tl.initial_equity``), SIN
componer sobre la equity del replay del histórico (``CSV_SIZING_COMPOUND = False`` →
``simulate_f(..., compound=False)``). Con un EA de importe fijo (``FixedAmount``) es exactamente lo que
hace el EA; con un EA de % de la equity del momento es una aproximación (nota ``lote_sobre_cuenta``).
Las patas ``.sqx`` ya usaban X = f·A (Opción A). El eco va en ``meta.sizing_base``. El cross-check
exacto-vs-columna ``Size`` sigue COMPONIENDO (``compound=True`` explícito): reproduce el backtest.

Flotante de las patas CSV (2026-10-19, guardrail ``csv-d1-d2``, defecto D2; plan csv-d1-d2 §2.2-2.3 y Enmiendas
v2 E3-E11): el flotante diario ``joint`` NO sumaba el realizado del día (una pérdida cerrada por la mañana y
otra flotante por la tarde no se sumaban en el peor punto del día). Ahora las patas CSV usan
``float_mode="minute"`` (``CSV_FLOAT_MODE``/``AXI_FLOAT_MODE`` = ``"minute"``; ``"joint"`` reproduce el camino
anterior bit a bit): en la lente prop firm SOLO con un perfil estático (``not _profile_extended(cfg)``), en la
lente AXI siempre. Los trades CSV sin OHLC M1 reciben una serie sintética de un punto con su MAE del CSV
(:func:`_csv_minute_series`): cota "desde la apertura", sin suponer cuándo ocurrió. Los perfiles con trailing
(Apex, futuros) siguen con ``FLOAT_MODE`` + trailing (el trailing ya incluye el realizado).

El lente AXI (F4+F5b) usa el BUCLE TRANSPARENTE de §4.1: en vez de la llamada
opaca a ``run_portfolio_mc`` replica su bucle con las piezas públicas del
motor (``detect_flatflat_blocks`` → ``build_block_records`` →
``stationary_block_indices(seed=42)``) y lo evalúa con el WALKER F4 de
``poc_web/api/axi_path.py`` (``evaluate_path_axi``): regla doble de parada
(pass solo si equity ≥ target ∧ T(d) ≥ min_trades, con T por fecha de
apertura), continuación post-target a f constante, y decisiones sobre la
serie merged precomputada (``mtm="joint"``: con D2 el flotante POR MINUTO con el realizado
intradía, ``precompute_minute``; con ``AXI_FLOAT_MODE = "joint"`` el flotante conjunto M1 de antes) con
``first_cross``/raw SIEMPRE en bound (sub-pase motor; tolerancia cero contra
``run_portfolio_mc`` en ``check_axi.py``, bloque 1). El post-filtro F2b
(``apply_trade_floor``) quedó fuera del flujo del endpoint (adjudicado en
§10.4-2 del análisis).

El pipeline de simulación replica la receta EXACTA de
``risk_app/propfirm_portfolio_mode.py`` (bloques flat-flat, gaps empíricos,
tamaño de bloque automático (ver más abajo: Politis-White desde 2026-10-21), CRN seed 0, ``seed=fi`` por f) con
``float_mode`` FIJADO en esta capa (el default del motor es ``"bound"``: no confiar en él, §2.3): ``"joint"``
(``FLOAT_MODE``) en los perfiles con trailing o extendidos, ``"minute"`` (``CSV_FLOAT_MODE``) en los estáticos.

Patas de futuros nativas (plan float-valleys-ventanas, Fase C): una ruta que
termina en ``.sqx`` es una pata ``kind="sqx_futures"`` (tabla de trades del
``orders.bin`` vía ``sqx.sqx_trades``; sin CSV ni OHLC). Sizing Opción A
(``fixed_risk_micros``: X = f·account_size, n = min(cap, max(1, floor(X/riesgo
del SL)))) y flotante por VENTANAS de trade (``float_mode="windows"``, cota a la
precisión del backtest). Solo lente prop firm con un perfil FTMO Futures; AXI, par
2D y diagnóstico por pata responden 409; cartera mixta CSV + .sqx = error de carga.

Patas .sqx de CFD FTMO (2026-10-12) y flotante por MINUTO (2026-10-14, plan
cfd-sqx-minuto-par, Fase 2): con ``asset="cfd"`` el símbolo SQX puede tener OHLC M1 (clave
``ohlc_m1`` de ``config/ftmo_cfd_costs.yaml``, ficheros en ``risk_app/inputs/ohlc/``, bid, hora
del servidor). Si el OHLC existe y CUADRA con el .sqx (≥ 95% de las entradas dentro de la barra de
ejecución ± spread ± 2 ticks) la pata lleva ``minute_valle`` (serie M1 adversa NETA de spread en
cortos, comisión y swap-cargo, en el eje de días pedido) y el lente prop firm usa
``float_mode="minute"`` (cota por minuto con el realizado intradía; ``PropfirmSimRequest.sqx_float
= "auto"``); si no, cae a ``windows`` con warning que nombra la pata y el motivo
(``sqx_float="windows"`` fuerza el camino anterior, bit a bit). Las carteras .sqx CFD admiten
además el barrido 2D (f_challenge, f_funded); AXI y el diagnóstico por pata siguen en 409 para
cualquier .sqx.

Fichas de símbolo (guardrail entrada-4-bloques, fase F2, 2026-10-20): una pata ``.sqx`` ya no necesita que su
símbolo esté en un YAML de costes; se lee contra la FICHA de ese símbolo (``poc_web/api/symbols.py``, alias
``fichas``), que pertenece a un bróker (``config/brokers.yaml``). El bróker fija la clase de activo, el reloj del
servidor (eje de días), la carpeta del OHLC M1 y las especificaciones. El catálogo lista todo ``.sqx`` legible
(``broker``/``ficha``/``motivo``; un símbolo sin ficha sale como ``sin_ficha`` para poder darlo de alta), una
cartera ``.sqx`` debe ser de un solo bróker y, en prop firm, del bróker del perfil (409 si no). Las funciones
``_sqx_asset``/``_sqx_cost_spec``/``_sqx_ohlc``/``_sqx_leg_minutes`` (camino del YAML) se conservan tal cual.

Lente AXI con patas ``.sqx`` (fase F3, 2026-10-20; plan entrada-4-bloques §2.7 y enmiendas E1/E19): una cartera ``.sqx``
del bróker de la lente (``lenses.axi``) se simula con el bucle transparente de siempre. Las patas se recargan con
``synth_mode="daily"``: una operación sin barras M1 tiene un punto por cada fecha del servidor que abarca con el suelo
de la operación (``−adverse_per_lot/pv − spread si es corto``) y una pata sin OHLC coherente es ENTERAMENTE esa cota
diaria por el MAE. El valle diario de cada operación (``{fecha: mínimo de su serie}``) va en una COPIA de la
``StrategyMC`` (la cacheada, que comparte la lente prop firm, sigue con ``valle == {}``); los lotes son el riesgo fijo
``X = f·A`` hasta el SL inicial. Con otro bróker, 409. Enmienda E20: el spread de la serie por minuto (coherencia con el
OHLC, ajuste de los cortos y suelo de la cota diaria) es el del BACKTEST, ``defaultSpread · tickSize`` del
``InstrumentInfo`` del propio ``.sqx``; el ``spread_px`` de la ficha solo si el ``.sqx`` no lo trae o no es > 0.

Receta para el EA (fase F4, 2026-10-20; plan §2.8 y enmiendas E3, E12, E14, E18): las respuestas de ``simulate_propfirm`` y
``simulate_axi`` con una cartera ``.sqx`` (y solo con ella: las CSV no llevan la clave) añaden ``meta.recipe``, cómo configurar
el MM ``FixedAmount`` del EA para que haga el riesgo fijo que simula la app: ``RiskedMoney`` por f, ``Decimals``, ``mmStep``,
``LotsIfNoMM`` = lote mínimo (la CONDICIÓN para que el EA haga lo simulado), ``MaxLots`` = el lote máximo simulado a cada f
(un tope que nunca muerde) y los avisos/notas del catálogo. Lleva TODA la rejilla de f (el front compone la de la f elegida) y se
adjunta DESPUÉS de ``_jsonable`` para conservar sus booleanos. ``portfolio_load`` añade a las filas de patas ``.sqx`` ``broker``,
``ficha_sin_verificar``, ``mm_backtest`` y ``paridad_frac_ok``, y la guarda ``fichas`` (``warn`` si alguna ficha tiene campos
sin verificar; nunca ``error``). Revisión de código de la F3: H2 (``_sqx_backtest_spread`` nunca veta una pata), H3
(``_sqx_leg_minutes_sheet`` valida ``synth_mode``) y H4 (``meta.legs`` de la lente AXI describe las patas recargadas).

Bloque automático (guardrail ``bloque-politis-white``, 2026-10-21; plan §2.2 y enmiendas E1-E19): con ``block_size <= 0`` el
tamaño medio de bloque del bootstrap estacionario es el b* de Politis-White (2004; corrección de Patton, Politis y White,
2009; ``src/risk/block_length.py``) sobre la serie del resultado de cada episodio medido en R, es decir, Σ pnl por lote /
riesgo por lote al SL de sus operaciones (:func:`_episode_r_series`): no depende de la f ni de la rejilla (antes, la ACF de los
cuadrados del PnL en $ a UNA f). Con menos de 25 episodios, o con más del 10 % de las operaciones sin un riesgo por lote válido,
bloque 1, modo ``fallback_iid`` y un aviso. Con un bloque principal > 1 (automático o manual) la meta añade ``bs1``: la misma
simulación con bloque 1 solo en f* (mismos caminos y semilla que una petición manual de bloque 1); si ahí se supera la
tolerancia de la recomendación (``tol_ruin`` o ``tol_neg``), un aviso en ``warnings``. f*, la recomendación y la receta se
calculan con el bloque principal y no cambian por la sensibilidad. ``meta.model_context.block_size`` lleva ``method``,
``series``, ``n_excluded_terms``, ``opt_m`` y ``m`` además de lo de siempre.

Lanzar desde la raíz del repo:

    python -m uvicorn poc_web.api.main:app --port 8600
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Imports del motor (lectura pura; nada de este archivo escribe) --------
from risk_app.input_sources import match_by_name, scan_inputs
from src.broker_config import load_axi_config, load_broker_config, load_oos_registry
from src.risk import load_trade_list
from src.risk.input_checks import (
    initial_capital_mismatch,
    leg_diagnostics,
    multi_symbol_counts,
)
from src.risk.block_length import MIN_OBS as BS_MIN_EPISODES, politis_white_info
from src.risk.mc import extend_stationary_rows, stationary_block_indices
from src.risk.mc_portfolio import (
    PathOutcome,
    minute_valley_series,
    aggregate_outcomes,
    attach_lots,
    attach_minute_valleys,
    build_block_records,
    count_crossing,
    detect_flatflat_blocks,
    precompute_strategy,
    run_portfolio_mc,
)
from src.risk.mm import simulate_f
from src.risk.ruin import recommend_f
from poc_web.api.axi_path import (
    evaluate_path_axi,
    joint_columns_cache,
    mean_inter_block_gap,
    minute_columns_cache,
    post_target_split,
    precompute_joint,
    precompute_minute,
    precompute_open_days,
    raw_outcome,
)
from poc_web.api.day_axis import DAY_AXIS_HOURS, apply_day_axis_shift
from poc_web.api import profiles
# Fichas de símbolo por bróker. Alias ``fichas``: hay 8 variables locales ``symbols`` (specs AXI) que taparían el módulo.
from poc_web.api import symbols as fichas
from poc_web.api.model_notes import NOTES, notes_for, render_run
from poc_web.api.run_history import append_run, list_runs
from src.risk.propfirm import (
    _draw_gap_tape,
    build_day_blocks_portfolio,
    build_phase_tape,
    combine_challenge_funded,
    contract_cap_diagnostic,
    empirical_gaps,
    eval_contract_cap_micros,
    ev_vector_all,
    firm_report,
    firm_report_pair,
    load_propfirm_config,
    pa_min_tier_cap_micros,
    precompute_day_arrays_portfolio,
    recommend_f_ev,
    recommend_f_pair,
    resolve_fee_eur,
    sequence_ev,
    simulate_challenge,
    simulate_firm_paths,
    sweep_funded,
)
from src.risk.session_axis import make_session_day_labeler, overnight_violations
from src.risk.futures_legs import _FLOOR_EPS, fixed_risk_micros, strategy_from_trade_table
from src.risk.propfirm_futures import sequence_ev_futures
from poc_web.api.futures_report import (
    firm_report_futures,
    withdrawal_policy_cfg,
)
# Patas de futuros nativas desde .sqx (plan float-valleys-ventanas, Fase C).
# poc_web puede importar sqx (el guardián solo cubre src.risk); src.risk nunca.
from sqx.d1.extract_sqx import SQXFormatError
from sqx.sqx_trades import (
    SQXFutureLegError,
    _parse_settings_xml,
    costs_path_for_symbol,
    load_costs_policy,
    read_backtest_mm,
    read_instrument_info,
    read_trade_table,
    read_trade_table_spec,
)

# Seed FIJA de la secuencia futures (enmienda E1): el EV/campaña depende de la
# seed vía _draw_attempt_index, así que seed=fi rompería el CRN entre f y
# haría f* ruidoso. Mismo valor para TODOS los f (y para la paridad web↔CLI).
_FUTURES_SEQ_SEED = 0

ROOT = Path(__file__).resolve().parents[2]
TRADES_DIR = ROOT / "risk_app" / "inputs" / "trades"
EXTRA_ROOTS_ENV = "RISK_APP_EXTRA_ROOTS"
LOCAL_CONFIG_PATH = ROOT / "config" / "local.yaml"


def _resolve_extra_roots(
    env: Mapping[str, str], local_yaml: Path, base: Path | None = None
) -> tuple[list[Path], str]:
    """Raíces extra de exportaciones SQX y su origen (función pura, sin efectos).

    Precedencia: env ``RISK_APP_EXTRA_ROOTS`` (rutas separadas por ``os.pathsep``)
    > clave ``extra_input_roots`` (lista) de ``config/local.yaml`` (ruta relativa
    respecto a ``base``, por defecto la raíz del repo) > ``[]``. A cada ruta se le
    aplica ``expanduser``. Las rutas inexistentes se CONSERVAN (el catálogo ya
    comprueba ``is_dir``). Origen devuelto: "env" | "local.yaml" | "ninguno".

    Fail-fast: en ``local.yaml``, la clave solo admite un ``str`` o una lista/tupla de
    escalares no nulos (str/int/float, ``str()`` en cada uno; sin cadenas vacías); un YAML
    inválido, un tipo distinto o un elemento nulo/dict/bool lanzan ``ValueError`` que
    nombra el fichero y la clave. Clave ausente, fichero vacío, ``null`` o ``[]`` = sin entradas.
    """
    base = ROOT if base is None else base
    raw_env = (env.get(EXTRA_ROOTS_ENV) or "").strip()
    if raw_env:
        parts = [p.strip() for p in raw_env.split(os.pathsep) if p.strip()]
        if parts:
            return [Path(p).expanduser() for p in parts], "env"
    if local_yaml.is_file():
        key = "extra_input_roots"
        prefix = f"{local_yaml}: '{key}' debe ser una lista de rutas (o una ruta)"
        try:
            data = yaml.safe_load(local_yaml.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError(f"{local_yaml}: YAML inválido ({e})") from e
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"{local_yaml}: se esperaba un mapa con la clave '{key}'; "
                             f"recibido {type(data).__name__}")
        entries = data.get(key)
        if entries is None:
            return [], "ninguno"
        if isinstance(entries, str):
            entries = [entries]
        elif not isinstance(entries, (list, tuple)):
            raise ValueError(f"{prefix}; recibido {type(entries).__name__}")
        roots = []
        for entry in entries:
            if (isinstance(entry, bool) or not isinstance(entry, (str, int, float))
                    or not str(entry).strip()):
                raise ValueError(f"{prefix}; elemento no válido: {entry!r} "
                                 f"({type(entry).__name__})")
            path = Path(str(entry).strip()).expanduser()
            roots.append(path if path.is_absolute() else base / path)
        if roots:
            return roots, "local.yaml"
    return [], "ninguno"


# Raíces adicionales de exportaciones SQX (CSV orderstocsv + .sqx/.xml), fuera
# del repo: se ofrecen en el catálogo y entran en el emparejamiento por nombre.
# Las patas de estas raíces se referencian por ruta absoluta (ROOT / abs = abs).
# Se resuelven UNA vez al importar (env > config/local.yaml > []).
EXTRA_INPUT_ROOTS, _EXTRA_ROOTS_ORIGIN = _resolve_extra_roots(os.environ, LOCAL_CONFIG_PATH)
# Logger de uvicorn: es el único con handler al importar la app (``uvicorn.error``, INFO).
logging.getLogger("uvicorn.error").info(
    "EXTRA_INPUT_ROOTS (origen: %s): %s", _EXTRA_ROOTS_ORIGIN,
    [str(r) for r in EXTRA_INPUT_ROOTS] or "ninguna")
# .sqx del repo (misma carpeta que usa el emparejamiento por nombre); el
# catálogo busca aquí además de en EXTRA_INPUT_ROOTS patas de futuros nativas.
SQX_STRATEGIES_DIR = ROOT / "risk_app" / "inputs" / "strategies"
# OHLC M1 de los símbolos SQX de CFD (2026-10-14; gitignorado): nombre de fichero por
# símbolo en la clave ``ohlc_m1`` de ``config/ftmo_cfd_costs.yaml`` (monkeypatcheable).
SQX_OHLC_DIR = ROOT / "risk_app" / "inputs" / "ohlc"
# Carpeta de OHLC por bróker para los tests (``{id del bróker: Path}``, vacío por defecto): manda sobre la carpeta
# que declara ``config/brokers.yaml`` (véase :func:`_ohlc_dir`).
SQX_OHLC_DIR_OVERRIDES: dict[str, Path] = {}
# Coherencia mínima .sqx↔OHLC (fracción de entradas dentro de la barra de ejecución) para
# usar el flotante por minuto; por debajo, hora o símbolo equivocados → ventanas.
SQX_MINUTE_COHERENCE_MIN = 0.95

# Tipos de pata (``leg["kind"]``): "csv" (orderstocsv + OHLC M1, camino
# histórico, SIN clave "kind" en el dict por bit-identidad) y "sqx_futures"
# (.sqx de micro futuros: tabla de trades del orders.bin, valle por ventanas).
KIND_CSV = "csv"
KIND_SQX = "sqx_futures"
SQX_SIZING_MODE = "riesgo fijo (Opción A)"
# Clase de activo de una pata .sqx (``leg["asset"]``, 2026-10-12): la fija el YAML
# de costes cuya allowlist contiene su símbolo (futuros FTMO o CFD FTMO).
SQX_ASSET_FUTURES = "futures"
SQX_ASSET_CFD = "cfd"
# Los .sqx de futuros llevan reloj de pared ET (enmienda A10 del plan): el
# labeler de sesiones es SIEMPRE el de America/New_York (identidad de zona).
SQX_WALL_CLOCK_TZ = "America/New_York"
DEFAULT_SQX_CUT = "16:10"

FLOAT_MODE = "joint"  # canónico (§2.3): fijado en la capa API, no en la request

# D1 (guardrail csv-d1-d2, 2026-10-19; plan csv-d1-d2 §2.1): el lote de las patas CSV se calcula sobre el
# TAMAÑO DE CUENTA (``tl.initial_equity`` = account_size), sin componer sobre la equity del replay del
# histórico. ``True`` = el comportamiento anterior (solo para A/B de investigación).
CSV_SIZING_COMPOUND = False
# Eco de la base del lote en ``meta.sizing_base`` (lo calcula :func:`_sizing_base`).
SIZING_BASE_ACCOUNT = "tamaño de cuenta (sin componer)"
SIZING_BASE_COMPOUND = "equity del replay del backtest (compuesto)"

# D2 (guardrail csv-d1-d2, fase 2; plan csv-d1-d2 §2.2-2.3 y Enmiendas v2 E3, E10): flotante POR MINUTO con el realizado
# intradía en las patas CSV. ``"minute"`` = el modo nuevo; ``"joint"`` = el camino anterior, bit a bit (A/B de la fase 3).
# ``CSV_FLOAT_MODE`` rige solo en la lente prop firm con un perfil ESTÁTICO (``not _profile_extended(cfg)``: modelo
# cycle, sin trailing ni eje de sesiones ET); con trailing (Apex, futuros) sigue ``FLOAT_MODE`` + trailing, que ya
# incluye el realizado. ``AXI_FLOAT_MODE`` rige en la lente AXI. Se leen al calcular, no al importar (los A/B los
# cambian en caliente); un valor distinto de "minute"/"joint" falla en voz alta (:func:`_check_float_flag`).
CSV_FLOAT_MODE = "minute"
AXI_FLOAT_MODE = "minute"

# Umbral C1 del semáforo de régimen ATR (producto gap-régimen-ATR, piezas
# 1+2): umbral C1 calibrado con la curva GAP-ATR-CURVA (decisión de diseño
# 2026-10-12).
_THRESHOLD_PCTILE = 90.0

# Fase 6 (plan Apex): zona horaria por defecto de los timestamps cuando el
# perfil es de sesiones ET (day_axis="et_session") y la request no trae
# source_tz. Los CSVs disponibles HOY son FX en hora de servidor AXI
# (NY-close, EET/EEST ≈ Europe/Athens); los datos de futuros documentarán su
# tz en la Fase 7 del plan. Siempre se hace eco en meta["source_tz"] y se
# advierte cuando se usa el default.
DEFAULT_ET_SOURCE_TZ = "Europe/Athens"

app = FastAPI(title="PoC web — Cartera + lentes", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(profiles.router)
app.include_router(fichas.router)

_broker = load_broker_config()
# Perfiles de prop firm = oficiales (config/propfirms.yaml) + del usuario
# (config/propfirms.user.yaml, ver profiles.py), oficiales PRIMERO. ``_firms`` se muta EN SITIO
# (clear + update) bajo ``_FIRMS_LOCK``: las referencias existentes ven la recarga. Los lectores
# resuelven la firma tomando el lock (``_firm_get`` / ``_firms_snapshot``).
_FIRMS_LOCK = threading.RLock()
_firms: dict = {}


def reload_firms() -> dict:
    """Reconstruye ``_firms`` (oficiales + usuario) fuera del lock y lo intercambia dentro."""
    merged, user_state = profiles.build_firms()
    with _FIRMS_LOCK:
        _firms.clear()
        _firms.update(merged)
        profiles.set_state(user_state)
    return _firms


def _firms_snapshot() -> list:
    """``list(_firms.items())`` tomado bajo el lock (lectores que iteran)."""
    with _FIRMS_LOCK:
        return list(_firms.items())


def _firms_catalog_snapshot() -> list:
    """``[(cfg, campos_de_catálogo)]`` de TODAS las firmas, leídos juntos bajo el lock:
    ``reload_firms`` intercambia ``_firms`` y el estado de ``profiles`` dentro del mismo lock, así
    que una recarga concurrente nunca deja un cfg con los ``official/based_on/verified_on`` de otro
    momento."""
    with _FIRMS_LOCK:
        return [(c, profiles.catalog_fields(k)) for k, c in _firms.items()]


def _firm_get(key):
    """``_firms.get(key)`` bajo el lock."""
    with _FIRMS_LOCK:
        return _firms.get(key)


profiles.set_reload_hook(reload_firms)
reload_firms()
_axi = load_axi_config()

# Cachés de proceso (solo lectura de disco; claves incluyen mtime del fichero)
_ohlc_cache: dict[tuple, pd.DataFrame | None] = {}
_leg_cache: dict[tuple, dict] = {}
# Fichas de símbolo (symbols.py): tras cada escritura se invalidan las patas .sqx cacheadas, y el alta guiada desde un
# .sqx (GET /api/symbols/draft-from-sqx) solo lee rutas que el catálogo de patas lista.
fichas.set_reload_hook(lambda: _leg_cache.clear())
fichas.set_catalog_provider(lambda: [leg["path"] for leg in _scan_legs() if leg["kind"] == KIND_SQX])
# símbolo SQX de un .sqx (lectura barata de settings.xml): (ruta, mtime) → str|None
_sqx_symbol_cache: dict[tuple, str | None] = {}
# OHLC M1 de símbolos SQX (patas CFD, 2026-10-14): (ruta, mtime) → DataFrame | None
_sqx_ohlc_cache: dict[tuple, pd.DataFrame | None] = {}

# Carteras cargadas en memoria (suficiente para el PoC): portfolio_id → datos
_portfolios: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _jsonable(obj):
    """Convierte numpy/dataclasses/tuples a JSON puro."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    return obj


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_sqx_leg_path(path_rel: str) -> bool:
    """Una ruta que termina en ``.sqx`` es una pata de futuros nativa."""
    return str(path_rel).lower().endswith(".sqx")


def _sqx_symbol(p: Path) -> str | None:
    """Símbolo SQX (``SymbolInfo/@symbolName`` del resultado Main) de un
    ``.sqx`` por lectura BARATA de ``settings.xml``; cacheado por (ruta,
    mtime). None si el fichero es ilegible (ZIP roto, sin settings.xml, XML
    sin resultado Main, …): el catálogo lo omite."""
    import zipfile
    key = (str(p), _mtime(p))
    if key not in _sqx_symbol_cache:
        try:
            with zipfile.ZipFile(p) as zf:
                info = _parse_settings_xml(zf.read("settings.xml"))
            _sqx_symbol_cache[key] = str(info["symbol"])
        except Exception:  # noqa: BLE001 - ilegible ⇒ se omite del catálogo
            _sqx_symbol_cache[key] = None
    return _sqx_symbol_cache[key]


def _sqx_asset(symbol: str | None) -> tuple[str | None, Path | None]:
    """(clase de activo, YAML de costes) de un símbolo SQX: ``("futures", …)`` o
    ``("cfd", …)`` según la allowlist que lo contenga; ``(None, None)`` si ninguna."""
    path = costs_path_for_symbol(symbol)
    if path is None:
        return None, None
    return str(load_costs_policy(path).get("asset_class", SQX_ASSET_FUTURES)), path


def _server_to_prague_wallclock(ts: pd.Series) -> pd.Series:
    """Hora del servidor FTMO (EETUS = siempre 7 h por delante de Nueva York) →
    reloj de pared de Europe/Prague (naive). Los cambios de hora de NY y de Praga
    caen en domingo con el mercado cerrado: la hora ambigua se toma como verano y
    la inexistente se desplaza hacia delante (solo pueden afectar a posiciones que
    pasan el fin de semana, en ±1 h de duración)."""
    ny = (pd.to_datetime(ts) - pd.Timedelta(hours=7)).dt.tz_localize(
        "America/New_York", ambiguous=np.ones(len(ts), dtype=bool),
        nonexistent="shift_forward")
    return ny.dt.tz_convert("Europe/Prague").dt.tz_localize(None).astype("datetime64[ns]")


def _sqx_axis_transform(axis_hours: float):
    """Transformación del eje de días de una pata .sqx CFD, hora del servidor → eje.

    ``axis_hours == DAY_AXIS_HOURS["ftmo"]`` → reloj de pared de Praga con DST real
    (:func:`_server_to_prague_wallclock`); otro valor no nulo → desplazamiento
    constante; 0 → identidad. Es la ÚNICA transformación que se aplica a ``open_ts``/
    ``close_ts`` de la tabla de trades Y a las series por minuto (2026-10-14: una sola
    función para que trades y minutos no puedan desfasarse)."""
    if not axis_hours:
        return lambda s: s
    if float(axis_hours) == DAY_AXIS_HOURS["ftmo"]:
        return _server_to_prague_wallclock
    shift = pd.Timedelta(hours=float(axis_hours))
    return lambda s: s + shift


def _sqx_cost_spec(symbol: str, costs_path: Path | None) -> tuple[dict | None, float | None]:
    """``(entrada symbols.* del YAML con sqx_symbol == symbol, typical_spread_points)``.
    Ambos ``None`` si el YAML no los tiene (el llamador decide el motivo)."""
    if costs_path is None:
        return None, None
    costs = load_costs_policy(costs_path)
    spec = next((s for s in (costs.get("symbols") or {}).values()
                 if s.get("sqx_symbol") == symbol), None)
    pts = ((costs.get("sqx_policy") or {}).get(symbol) or {}).get("typical_spread_points")
    return spec, (float(pts) if pts is not None else None)


def _sqx_ohlc(symbol: str, costs_path: Path | None) -> tuple[pd.DataFrame | None, Path | None]:
    """OHLC M1 (bid, hora del servidor) del símbolo SQX de una pata CFD.

    ``(DataFrame | None, ruta | None)``: la ruta sale de ``ohlc_m1`` en la entrada del
    YAML de costes (``SQX_OHLC_DIR / ohlc_m1``); ``None`` si el símbolo no tiene la clave.
    El DataFrame es ``None`` si el fichero no existe o es ilegible. Carga con
    :func:`src.risk.atr.load_ohlc_m1` y cachea por (ruta, mtime)."""
    spec, _ = _sqx_cost_spec(symbol, costs_path)
    name = (spec or {}).get("ohlc_m1")
    if not name:
        return None, None
    path = SQX_OHLC_DIR / str(name)
    return _sqx_load_ohlc(path), path


def _sqx_load_ohlc(path: Path) -> pd.DataFrame | None:
    """OHLC M1 de un fichero (``None`` si no existe o es ilegible), cacheado por (ruta, mtime) en
    ``_sqx_ohlc_cache``. Lo comparten :func:`_sqx_ohlc` (camino del YAML) y la carga por fichas."""
    key = (str(path), _mtime(path))
    if key not in _sqx_ohlc_cache:
        from src.risk.atr import load_ohlc_m1
        try:
            ohlc = load_ohlc_m1(path)
            if not ohlc.index.is_monotonic_increasing:
                ohlc = ohlc.sort_index()
            if not ohlc.index.is_unique:
                ohlc = ohlc[~ohlc.index.duplicated(keep="last")]
        except Exception:  # noqa: BLE001 - ilegible/ausente ⇒ la pata cae a ventanas con motivo
            ohlc = None
        _sqx_ohlc_cache[key] = ohlc
    return _sqx_ohlc_cache[key]


def _ohlc_dir(broker) -> Path:
    """Carpeta de los OHLC M1 de un bróker (``fichas.Broker``).

    1. ``SQX_OHLC_DIR_OVERRIDES[broker.id]`` si existe (tests);
    2. ``SQX_OHLC_DIR`` si ``ROOT / broker.ohlc_dir`` es ``risk_app/inputs/ohlc`` (FTMO CFD): así los tests que
       sustituyen esa constante siguen funcionando;
    3. ``ROOT / broker.ohlc_dir`` en otro caso.
    ``ValueError`` si el bróker no tiene carpeta de OHLC (y no hay override)."""
    override = SQX_OHLC_DIR_OVERRIDES.get(broker.id)
    if override is not None:
        return Path(override)
    if not broker.ohlc_dir:
        raise ValueError(f"el bróker '{broker.id}' no tiene carpeta de OHLC (ohlc_dir)")
    d = ROOT / broker.ohlc_dir
    return SQX_OHLC_DIR if d == ROOT / "risk_app" / "inputs" / "ohlc" else d


def _sheet_ohlc_path(sheet, broker) -> Path | None:
    """Ruta del OHLC M1 de una ficha (``_ohlc_dir(bróker) / ohlc_m1``); ``None`` si la ficha no lo trae o el bróker
    no tiene carpeta de OHLC."""
    if not sheet.ohlc_m1:
        return None
    if broker.id not in SQX_OHLC_DIR_OVERRIDES and not broker.ohlc_dir:
        return None
    return _ohlc_dir(broker) / str(sheet.ohlc_m1)


def _sqx_entry_bars(df: pd.DataFrame, ohlc: pd.DataFrame):
    """``(pos, ok, low, high)`` por trade: posición en ``ohlc`` de la barra
    ``floor(open_ts, "min")`` (hora del servidor), si existe (``ok``) y su low/high.
    Barra ausente ⇒ ``ok = False``."""
    bars = pd.DatetimeIndex(pd.to_datetime(df["open_ts"])).floor("min")
    pos = ohlc.index.get_indexer(bars)
    ok = pos >= 0
    safe = np.where(ok, pos, 0)
    return ok, ohlc["l"].to_numpy()[safe], ohlc["h"].to_numpy()[safe]


def _sqx_ohlc_consistency(df: pd.DataFrame, ohlc: pd.DataFrame,
                          spread_px: float, tick: float) -> float:
    """Fracción de trades con ``open_px`` dentro de ``[low − spread − 2·tick, high +
    spread + 2·tick]`` de la barra ``floor(open_ts, "min")`` (``df`` en hora del SERVIDOR,
    como el OHLC). Barra ausente = fuera. Mide si el OHLC corresponde al .sqx (hora y
    símbolo); el umbral de uso es ``SQX_MINUTE_COHERENCE_MIN``."""
    if len(df) == 0:
        return 0.0
    ok, lo, hi = _sqx_entry_bars(df, ohlc)
    px = df["open_px"].to_numpy(dtype=float)
    tol = spread_px + 2.0 * tick
    return float(np.mean(ok & (px >= lo - tol) & (px <= hi + tol)))


SYNTH_MODES = ("point", "daily")


def _sqx_minute_series(df_srv: pd.DataFrame, meta: dict, fleg, ohlc: pd.DataFrame | None,
                       spread_px: float, to_axis, synth_mode: str = "point") -> tuple[dict, dict]:
    """Series por minuto de una pata .sqx CFD: ``({i: (ts, vals, ot, ct)}, diag)``.

    ``df_srv`` = tabla de ``read_trade_table`` en hora del SERVIDOR y ANTES de pasar al
    eje (la fila i es el trade local i). Por fila: ``ot = open_ts``, ``ct = max(close_ts,
    open_ts + 1 ns)`` y ``s = minute_valley_series(ot, ct, open_px, is_buy, ohlc)`` (barras
    M1 bid adversas en unidades de PRECIO); si no hay barras, serie sintética ``ts=[ot]``,
    ``vals=[−mae_usd/lots/pv]`` (se cuenta en ``diag["sinteticas"]``). Ajuste de costes
    en precio: ``vals − spread (solo cortos: el OHLC es bid, el largo abre al ask que ya
    está en open_px y cierra al bid) − (comm_usd + max(−swap_usd, 0))/lots/pv`` (``comm_usd``
    y ``swap_usd`` son TOTALES del trade; sin ``swap_usd`` → 0). ``ts``/``ot``/``ct`` pasan
    al eje con ``to_axis`` (la MISMA transformación que ``open_ts``/``close_ts``), UNA
    llamada vectorizada sobre todas las barras de la pata; ``ct`` en el eje sigue la regla
    de ``_load_sqx_leg`` (cierre transformado, y si ≤ apertura, apertura + 1 ns).

    Recorte por el MAE de SQX (enmienda v3, 2026-10-14): tras el ajuste de costes y spread,
    cada serie se recorta por debajo en el peor estado del propio trade según SQX,
    ``vals_i := max(vals_i, −adverse_per_lot_i/pv − (spread_px si corto, 0 si largo))`` con
    ``adverse_per_lot = (mae_usd + comm_usd + max(−swap_usd, 0))/lots`` (el MISMO de las
    ventanas). Los ``vals`` son EXCURSIONES con signo en unidades de precio (≤ 0 en lo
    adverso), no precios absolutos. Motivo: ``minute_valley_series`` toma el low/high
    COMPLETO de la barra de ejecución, incluidos los precios ANTERIORES a la ejecución (en
    una orden stop el precio atraviesa el stop; ese low ocurrió antes de entrar) y eso
    hacía la serie peor que el MAE de SQX en 40-80% de los trades de oro; el MAE de SQX se
    mide en su simulación M1 DESDE la ejecución. Con el recorte, la magnitud por trade
    queda acotada por el MAE de SQX (+ costes, + spread en cortos), que a su vez es una
    cota del modelo del backtest, y el minuto aporta la simultaneidad (qué trades están mal
    a la vez); minuto ≥ ventanas por trade y por día en LARGOS; en cortos el suelo suma el
    spread fijo del YAML al MAE, así que puede quedar hasta ``spread·pv`` por corto vivo por
    debajo de ventanas (si el MAE de SQX ya incluye el spread, cuenta dos veces: conservador;
    revisión GLM H1/H2 de la enmienda, 2026-10-14). Se aplica también a las series
    sintéticas (en largos no cambia nada: ya valen ``−adverse/pv``).

    ``diag``: ``n_trades``, ``sinteticas``, fracción de entradas LARGAS dentro de
    ``[low, high]`` (``frac_largos_bid``) y de ``[low + spread, high + spread]``
    (``frac_largos_ask``) de la barra de ejecución (tolerancia 0,05·tick por float32; ``None``
    sin largos); ANTES del recorte: ``frac_serie_peor_que_mae_antes_del_recorte`` /
    ``max_exceso_usd_lote_antes_del_recorte`` de ``−min(serie ajustada)·pv − adverse_per_lot``
    (> 0,01 USD/lote = la serie era peor que el MAE de SQX; el máximo es el valor crudo:
    negativo = ningún trade excede); y ``frac_recortadas`` = fracción de trades con al
    menos un minuto recortado (en más de 0,01 USD/lote, la misma tolerancia).

    ``synth_mode`` (guardrail entrada-4-bloques, fase F3; plan §2.7.2) fija la serie de un trade SIN barras:

    - ``"point"`` (por defecto; lente prop firm): un punto ``−mae/lots/pv`` en ``ot``. Es el comportamiento de siempre,
      bit a bit;
    - ``"daily"`` (lente AXI): un punto por cada FECHA del servidor de ``[fecha(ot), fecha(ct)]``, en ``max(ot,
      inicio de ese día)``, con el valor del suelo del trade, ``−adverse_per_lot/pv − spread_px si es corto``
      (E19: ``adverse_per_lot`` ya incluye comisión y swap-cargo). El valle diario de la lente AXI toma el mínimo de
      cada fecha, así que un trade largo sin barras cuenta su peor estado en TODOS los días que abarca (cota
      conservadora, no suponer cuándo ocurrió).

    ``ohlc=None`` = sin OHLC: TODOS los trades son sintéticos (una pata sin OHLC coherente en la lente AXI, plan
    §2.7.3). En ambos modos cuentan en ``diag["sinteticas"]``."""
    if synth_mode not in SYNTH_MODES:
        raise ValueError(f"synth_mode debe ser uno de {SYNTH_MODES} (recibido {synth_mode!r})")
    n = len(df_srv)
    pv = float(meta["pv"])
    ns1 = np.timedelta64(1, "ns")
    ot_s = pd.to_datetime(df_srv["open_ts"]).to_numpy(dtype="datetime64[ns]")
    close_s = pd.to_datetime(df_srv["close_ts"]).to_numpy(dtype="datetime64[ns]")
    ct_s = np.maximum(close_s, ot_s + ns1)
    lots = df_srv["lots"].to_numpy(dtype=float)
    open_px = df_srv["open_px"].to_numpy(dtype=float)
    is_buy = df_srv["is_buy"].to_numpy(dtype=bool)
    mae = df_srv["mae_usd"].to_numpy(dtype=float)
    comm = df_srv["comm_usd"].to_numpy(dtype=float)
    swap_cost = (np.maximum(-df_srv["swap_usd"].to_numpy(dtype=float), 0.0)
                 if "swap_usd" in df_srv.columns else np.zeros(n))
    cost_px = (comm + swap_cost) / lots / pv

    raw_ts: list[np.ndarray] = []
    raw_vals: list[np.ndarray] = []
    n_synth = 0
    for i in range(n):
        s = (None if ohlc is None
             else minute_valley_series(ot_s[i], ct_s[i], open_px[i], bool(is_buy[i]), ohlc))
        if s is None:
            n_synth += 1
            if synth_mode == "daily":
                # un punto por fecha del servidor de [fecha(ot), fecha(ct)], en max(ot, inicio de ese día)
                days = np.arange(ot_s[i].astype("datetime64[D]"),
                                 ct_s[i].astype("datetime64[D]") + np.timedelta64(1, "D"))
                ts_i = np.maximum(days.astype("datetime64[ns]"), ot_s[i])
                vals_i = np.full(len(days), -mae[i] / lots[i] / pv)
            else:
                ts_i = ot_s[i:i + 1]
                vals_i = np.array([-mae[i] / lots[i] / pv])
        else:
            ts_i = s.index.values
            vals_i = s.to_numpy(dtype=float)
        raw_ts.append(ts_i)
        raw_vals.append(vals_i - (0.0 if is_buy[i] else spread_px) - cost_px[i])

    adverse = np.asarray(fleg.adverse_per_lot, dtype=float)
    # Suelo por trade: peor estado según SQX (excursión con signo, en precio). Corto: + spread.
    floor_px = -adverse / pv - np.where(is_buy, 0.0, spread_px)
    # Diagnóstico ANTES del recorte (exceso de la serie sobre el MAE+costes de SQX).
    exceso = (np.array([-v.min() for v in raw_vals]) * pv - adverse if n else np.empty(0))
    recortada = np.zeros(n, dtype=bool)
    for i in range(n):
        # "recortada" = el recorte mueve algún minuto más de 0,01 USD/lote (misma tolerancia
        # que el diagnóstico: el redondeo a float32 del .sqx no cuenta como exceso)
        recortada[i] = bool(np.any(raw_vals[i] < floor_px[i] - 0.01 / pv))
        raw_vals[i] = np.maximum(raw_vals[i], floor_px[i])

    lens = np.array([len(x) for x in raw_ts], dtype=np.int64)
    allts = np.concatenate(raw_ts + [ot_s, close_s]) if n else np.empty(0, "datetime64[ns]")
    ax = to_axis(pd.Series(allts)).to_numpy(dtype="datetime64[ns]")
    total = int(lens.sum())
    ts_ax, ot_ax, close_ax = ax[:total], ax[total:total + n], ax[total + n:]
    ct_ax = np.where(close_ax <= ot_ax, ot_ax + ns1, close_ax)
    bounds = np.r_[0, np.cumsum(lens)]
    series = {i: (ts_ax[bounds[i]:bounds[i + 1]], raw_vals[i],
                  pd.Timestamp(ot_ax[i]), pd.Timestamp(ct_ax[i])) for i in range(n)}

    long_ = is_buy
    frac_bid = frac_ask = None
    if long_.any() and ohlc is not None:
        ok, lo, hi = _sqx_entry_bars(df_srv, ohlc)
        eps = 0.05 * float(meta.get("tick_size") or 0.0)
        px = open_px
        in_bid = ok & (px >= lo - eps) & (px <= hi + eps)
        in_ask = ok & (px >= lo + spread_px - eps) & (px <= hi + spread_px + eps)
        frac_bid = float(np.mean(in_bid[long_]))
        frac_ask = float(np.mean(in_ask[long_]))
    diag = {
        "n_trades": int(n),
        "sinteticas": int(n_synth),
        "frac_largos_bid": frac_bid,
        "frac_largos_ask": frac_ask,
        "frac_serie_peor_que_mae_antes_del_recorte": float(np.mean(exceso > 0.01)) if n else 0.0,
        "max_exceso_usd_lote_antes_del_recorte": float(exceso.max()) if n else 0.0,
        "frac_recortadas": float(np.mean(recortada)) if n else 0.0,
    }
    return series, diag


def _sqx_leg_minutes(df_srv: pd.DataFrame, meta: dict, fleg, asset: str,
                     costs_path: Path | None, to_axis) -> tuple[dict | None, dict | None, str | None]:
    """``(minute_valle, minute_diag, motivo)`` de una pata .sqx (``with_minutes=True``).

    ``minute_valle`` es ``None`` (con ``motivo``) si la pata no es CFD, el símbolo no tiene
    ``ohlc_m1``/``typical_spread_points`` en el YAML, el OHLC no existe o es ilegible, o su
    coherencia con el .sqx es < ``SQX_MINUTE_COHERENCE_MIN`` (el OHLC no cuadra: hora o
    símbolo equivocados). Si no, la serie por minuto y su diagnóstico (que incluye la
    ``coherencia``)."""
    symbol = str(meta["symbol"])
    if asset != SQX_ASSET_CFD:
        return None, None, "no aplica: pata de futuros (sin OHLC M1)"
    spec, spread_pts = _sqx_cost_spec(symbol, costs_path)
    ohlc, path = _sqx_ohlc(symbol, costs_path)
    if path is None:
        return None, None, f"sin ohlc_m1 en el YAML de costes para {symbol}"
    if ohlc is None:
        return None, None, (f"OHLC {path.name} no disponible o ilegible en "
                            f"{SQX_OHLC_DIR.name}/")
    if spread_pts is None or spec is None or not spec.get("tick_size"):
        return None, None, f"YAML de costes sin typical_spread_points/tick_size para {symbol}"
    tick = float(spec["tick_size"])
    spread_px = spread_pts * tick
    coherence = _sqx_ohlc_consistency(df_srv, ohlc, spread_px, tick)
    if coherence < SQX_MINUTE_COHERENCE_MIN:
        return None, None, (f"el OHLC no cuadra con el .sqx (coherencia {coherence:.0%} < "
                            f"{SQX_MINUTE_COHERENCE_MIN:.0%}): hora o símbolo equivocados "
                            f"({path.name})")
    series, diag = _sqx_minute_series(df_srv, meta, fleg, ohlc, spread_px, to_axis)
    return series, {"coherencia": coherence, "ohlc": path.name, **diag}, None


_LEGACY_SPREAD = object()          # centinela: el llamador no pasa el spread del .sqx (E20 no aplica; spread de la ficha)


def _minute_spread(sheet, sqx_spread_px) -> tuple[float | None, str | None]:
    """E20: ``(spread_px, fuente)`` de la serie por minuto de una pata ``.sqx``: el spread DEL BACKTEST (``defaultSpread ·
    tickSize`` del ``InstrumentInfo`` del propio ``.sqx``, ``sqx_spread_px``) si es finito y > 0 (``"sqx"``); si no, el
    ``spread_px`` de la ficha (``"ficha"``); ``(None, None)`` si no hay ninguno de los dos."""
    if sqx_spread_px is not None and np.isfinite(sqx_spread_px) and sqx_spread_px > 0:
        return float(sqx_spread_px), "sqx"
    if sheet.spread_px is not None:
        return float(sheet.spread_px), "ficha"
    return None, None


def _sqx_backtest_spread(path: Path) -> float | None:
    """E20: spread del backtest de un ``.sqx`` en unidades de precio, ``defaultSpread · tickSize`` de su ``InstrumentInfo``
    (:func:`sqx.sqx_trades.read_instrument_info`); ``None`` si no lo trae, no es finito o no es > 0 (entonces manda el
    ``spread_px`` de la ficha). Sin redondeo: es la misma expresión (``puntos · tick``) que la ficha oficial de FTMO, así
    que cuando ``defaultSpread`` y ``tickSize`` coinciden con los del YAML el resultado es BIT-IDÉNTICO.

    H2 (revisión de código de la F3): la fuente del spread es informativa y NUNCA veta una pata. Un ``default_spread`` o un
    ``tick_size`` no numéricos (atributo vacío, un formato futuro del lector) darían ``ValueError``/``TypeError`` en el
    ``float(...)``, que estaba FUERA del ``try``: la conversión va dentro y esos errores caen al respaldo de la ficha."""
    try:
        info = read_instrument_info(path)
        ds, tick = info.get("default_spread"), info.get("tick_size")
        if ds is None or tick is None:
            return None
        spread = float(ds) * float(tick)
    except (SQXFormatError, OSError, ValueError, TypeError):
        return None
    return spread if (np.isfinite(spread) and spread > 0) else None


def _sqx_leg_minutes_sheet(df_srv: pd.DataFrame, meta: dict, fleg, sheet, broker, to_axis, synth_mode: str = "point",
                           sqx_spread_px=_LEGACY_SPREAD) -> tuple[dict | None, dict | None, str | None]:
    """``(minute_valle, minute_diag, motivo)`` de una pata ``.sqx`` con su FICHA (``with_minutes=True``): la
    versión por fichas de :func:`_sqx_leg_minutes`, con las mismas comprobaciones y en el mismo orden.

    El OHLC sale de ``_ohlc_dir(bróker) / ficha.ohlc_m1``. Solo hay serie por minuto en brókers CFD. ``minute_valle``
    es ``None`` (con ``motivo``) si la ficha no trae ``ohlc_m1``, no hay spread, el OHLC no existe o es ilegible, o su
    coherencia con el .sqx es < ``SQX_MINUTE_COHERENCE_MIN``. Con una ficha OFICIAL los motivos de "sin OHLC" y "sin
    spread" conservan el texto de siempre (citan el YAML de costes, de donde salen esas fichas); con una del usuario
    citan la ficha. H7 (revisión de código de la F2): "la ficha no trae ``ohlc_m1``" y "el bróker no tiene carpeta de
    OHLC" son dos motivos distintos.

    ``sqx_spread_px`` (enmienda E20): el spread del backtest del ``.sqx`` (:func:`_sqx_backtest_spread`; ``None`` = el
    ``.sqx`` no lo trae). Es el que usan la coherencia OHLC ↔ ``.sqx`` (tolerancia ``spread + 2·tick``), el ajuste de los
    cortos de la serie y el suelo de la cota diaria, porque las entradas del backtest se llenan con el spread de SQX y
    no con el típico real del bróker (oro AXI: 0,18 $ del backtest frente a 0,08 $ de la ficha). Solo si el ``.sqx`` no
    lo trae o no es > 0 se usa el ``spread_px`` de la ficha. Con él el ``minute_diag`` lleva ``spread_px_usado`` y
    ``spread_fuente`` (``"sqx"`` | ``"ficha"``; ``None`` si no hay ningún spread). Sin pasar el argumento (los
    llamadores de antes) el spread es el de la ficha y el diagnóstico el de siempre: es la paridad con
    :func:`_sqx_leg_minutes`, que NO usa el spread del ``.sqx``.

    ``synth_mode`` (fase F3; ver :func:`_sqx_minute_series`) se pasa a la serie. Con ``"daily"`` (lente AXI) una
    pata SIN serie por minuto (cualquiera de los motivos de arriba) no queda sin serie: se construye ENTERAMENTE
    sintética, desde la tabla de trades y sin OHLC, con la cota diaria por el MAE (un punto por fecha del servidor y
    trade con el suelo ``−adverse_per_lot/pv − spread``), y el ``motivo`` (no ``None``) lo dice: «… : se usa la cota
    diaria por el MAE». Su diagnóstico lleva ``cota_diaria: True``. Con ``"point"`` (lente prop firm) es lo de
    siempre: ``(None, None, motivo)``.

    H3 (revisión de código de la F3): un ``synth_mode`` inválido falla en voz alta AL PRINCIPIO (``ValueError``), igual que
    :func:`_load_sqx_leg` y :func:`_sqx_minute_series`; antes solo fallaba si la pata llegaba a construir la serie, y en las
    ramas sin serie (futuros, sin OHLC, sin spread) devolvía un motivo silencioso."""
    if synth_mode not in SYNTH_MODES:
        raise ValueError(f"synth_mode debe ser uno de {SYNTH_MODES} (recibido {synth_mode!r})")
    symbol = str(meta["symbol"])
    if broker.asset_class != SQX_ASSET_CFD:
        return None, None, "no aplica: pata de futuros (sin OHLC M1)"
    # PLAN-DESVIO (menor): E20 dice que el ``minute_diag`` añade ``spread_px_usado`` y ``spread_fuente``. Aquí las claves salen
    # solo cuando el llamador pasa ``sqx_spread_px`` (``_load_sqx_leg`` SIEMPRE lo pasa, así que el ``minute_diag`` de toda
    # pata las lleva): el test existente ``test_ac2_1_serie_por_minuto_por_ficha_identica_a_la_del_yaml_en_ftmo`` compara con
    # ``==`` el diagnóstico de esta función (sin el argumento) con el de ``_sqx_leg_minutes`` (YAML, que no se toca) y así
    # sigue valiendo sin editar sus aserciones.
    e20 = sqx_spread_px is not _LEGACY_SPREAD
    spread_px, spread_src = _minute_spread(sheet, sqx_spread_px if e20 else None)
    path = _sheet_ohlc_path(sheet, broker)
    ohlc = coherence = None
    if not sheet.ohlc_m1:
        reason = (f"sin ohlc_m1 en el YAML de costes para {symbol}" if sheet.official
                  else f"sin ohlc_m1 en la ficha de {symbol}")
    elif path is None:                  # la ficha trae ohlc_m1, pero el bróker no tiene dónde buscarlo
        reason = (f"el bróker '{broker.id}' no tiene carpeta de OHLC (ohlc_dir): no se puede leer "
                  f"{sheet.ohlc_m1} de la ficha de {symbol}")
    else:
        ohlc = _sqx_load_ohlc(path)
        if ohlc is None:
            reason = (f"OHLC {path.name} no disponible o ilegible en "
                      f"{path.parent.name}/")
        elif spread_px is None:         # ni el .sqx (E20) ni la ficha traen un spread
            reason = (f"YAML de costes sin typical_spread_points/tick_size para {symbol}" if sheet.official
                      else f"la ficha de {symbol} no trae spread_px")
        else:
            # tolerancia de la coherencia: el tick de la ficha; sin él (AXI), el tickSize del propio .sqx
            tick = float(sheet.tick_size) if sheet.tick_size else float(meta["tick_size"])
            coherence = _sqx_ohlc_consistency(df_srv, ohlc, spread_px, tick)
            if coherence < SQX_MINUTE_COHERENCE_MIN:
                reason = (f"el OHLC no cuadra con el .sqx (coherencia {coherence:.0%} < "
                          f"{SQX_MINUTE_COHERENCE_MIN:.0%}): hora o símbolo equivocados "
                          f"({path.name})")
            else:
                series, diag = _sqx_minute_series(df_srv, meta, fleg, ohlc, spread_px, to_axis, synth_mode)
                used = {"spread_px_usado": spread_px, "spread_fuente": spread_src} if e20 else {}
                return series, {"coherencia": coherence, "ohlc": path.name, **used, **diag}, None
    if synth_mode != "daily":
        return None, None, reason
    # PLAN-DESVIO: el plan no dice qué spread usar en la cota diaria de una pata sin ningún spread (ni en el .sqx ni en la
    # ficha). Se usa 0 (no se puede sumar lo que no se conoce) y el motivo lo avisa; con spread conocido, el mismo de la serie
    # por minuto (E20: el del backtest si el .sqx lo trae, si no el de la ficha) en el suelo de los cortos.
    floor_spread = spread_px if spread_px is not None else 0.0
    series, diag = _sqx_minute_series(df_srv, meta, fleg, None, floor_spread, to_axis, synth_mode="daily")
    why = reason + ": se usa la cota diaria por el MAE" + (" (sin spread en los cortos)" if spread_px is None else "")
    used = {"spread_px_usado": floor_spread, "spread_fuente": spread_src} if e20 else {}
    return series, {"coherencia": coherence, "ohlc": None if path is None else path.name, "cota_diaria": True,
                    "motivo_sin_serie": reason, **used, **diag}, why


def _scan_legs() -> list[dict]:
    """Patas disponibles: CSVs del repo (ruta relativa a ROOT) + raíces extra
    (ruta absoluta), con ``kind="csv"``, MÁS TODO ``.sqx`` legible (en las
    raíces extra y en ``risk_app/inputs/strategies``), con ``kind="sqx_futures"``.

    Cada ``.sqx`` lleva su estado de ficha (guardrail entrada-4-bloques, F2): ``asset`` = la clase de activo de
    su bróker (``None`` sin ficha), ``broker`` (id o ``None``), ``ficha`` ∈ {``"ok"``, ``"sin_ficha"``,
    ``"no_usable"``} y ``motivo`` (``None`` o el texto). Un símbolo sin ficha SALE en el catálogo (con
    ``sin_ficha``) para poder darlo de alta; solo se omite un ``.sqx`` ilegible (sin símbolo parseable).
    Agrupado en la UI por carpeta, igual que en Streamlit."""
    candidates = sorted(TRADES_DIR.rglob("*.csv"))
    for root in EXTRA_INPUT_ROOTS:
        if root.is_dir():
            candidates += sorted(root.rglob("*.csv"))
    legs, seen = [], set()
    for p in candidates:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        try:
            path = str(p.relative_to(ROOT))
        except ValueError:  # raíz externa → ruta absoluta (ROOT / abs == abs)
            path = str(p)
        legs.append({"path": path, "folder": p.parent.name, "name": p.stem,
                     "kind": KIND_CSV})

    sqx_candidates: list[Path] = []
    for root in [SQX_STRATEGIES_DIR, *EXTRA_INPUT_ROOTS]:
        if root.is_dir():
            sqx_candidates += sorted(root.rglob("*.sqx"))
    freg = fichas.load_registry()
    for p in sqx_candidates:
        rp = str(p.resolve())
        if rp in seen:
            continue
        sym = _sqx_symbol(p)
        if sym is None:            # ilegible (ZIP roto, sin settings.xml, sin resultado Main…): fuera
            continue
        seen.add(rp)
        try:
            path = str(p.relative_to(ROOT))
        except ValueError:
            path = str(p)
        st = fichas.catalog_entry(sym, freg)
        legs.append({"path": path, "folder": p.parent.name, "name": p.stem,
                     "kind": KIND_SQX, "asset": st["asset"], "broker": st["broker"],
                     "ficha": st["ficha"], "motivo": st["motivo"]})
    return legs


def _load_ohlc(sym_csv: str) -> pd.DataFrame | None:
    from src.risk.atr import load_ohlc_m1, resolve_ohlc_path
    try:
        resolved = resolve_ohlc_path(sym_csv)
        mtime = _mtime(Path(resolved))
    except Exception:
        resolved, mtime = None, None
    key = (sym_csv, mtime)
    if key not in _ohlc_cache:
        try:
            _ohlc_cache[key] = load_ohlc_m1(resolved) if resolved else None
        except Exception:
            _ohlc_cache[key] = None
    return _ohlc_cache[key]


def _load_leg(path_rel: str, account_size: float, f_ref: float,
              sqx_map: dict[str, Path] | None = None,
              axis_hours: float = 0.0,
              with_peaks: bool = False,
              day_labeler_tz: str | None = None,
              day_labeler_cut: str = "18:00",
              with_minutes: bool = False,
              synth_mode: str = "point") -> dict:
    """Carga + precomputo de una pata (cacheado). Espejo del bucle de la UI.

    Clave de caché (§8.6-4): mtime del CSV + del ``.sqx``/``.xml`` emparejado +
    del OHLC resuelto (+ horas de eje de días, F2c-2) + flags Apex Fase 6
    (``with_peaks``/``day_labeler_tz``). Si hay ``.sqx`` se
    intenta el sizing EXACTO (ATR real, ``risk_app.exact_mm``); si falla,
    fallback rápido etiquetado SIN cachear el fallo (un .sqx corregido se
    recoge en la siguiente carga).

    ``axis_hours`` (F2c-2): shift de eje de días aplicado ANTES del precomputo,
    de modo que bloques, valles M1, sizing exacto y entries viven todos en el
    mismo eje. La clave de caché lo incluye → cero contaminación entre ejes.

    Fase 6 (perfiles Apex): ``with_peaks=True`` rellena además
    ``strategy.minute_fav`` (envolvente favorable M1, requerida por el
    trailing); ``day_labeler_tz`` etiqueta días como SESIONES ET
    (``make_session_day_labeler``) en el groupby de envolventes y ``close_d``
    del precomputo. Ambos entran en la clave de caché.

    ``day_labeler_cut`` (perfiles FTMO Futures): hora ET de corte de la sesión
    del labeler (default "18:00" = Apex, bit-idéntico; FTMO Futures usa
    "16:10" vía ``cfg.day_axis_cut_hhmm``). Entra en la clave de caché: sin
    él una pata cacheada con eje Apex se reusaría para futures.

    Una ruta que termina en ``.sqx`` es una pata de futuros NATIVA
    (``kind="sqx_futures"``, :func:`_load_sqx_leg`): sin CSV, sin OHLC, sin
    ``axis_hours``/``with_peaks`` (no aplican); ``day_labeler_tz`` solo indica
    "pedir eje de sesión" (el labeler es siempre America/New_York).
    ``with_minutes`` (2026-10-14) solo se propaga a las patas ``.sqx`` (en las CSV no
    existe: ya llevan su ``minute_valle``): pide la serie por minuto de las patas CFD.
    ``synth_mode`` (guardrail entrada-4-bloques, fase F3) también solo vale en las patas ``.sqx`` (las CSV lo ignoran):
    ``"point"`` = lo de siempre (lente prop firm); ``"daily"`` = la cota diaria por el MAE de la lente AXI (ver
    :func:`_load_sqx_leg`).

    Campos de régimen ATR (producto gap-régimen-ATR, piezas 1+2; solo patas
    CSV con sizing exacto resuelto): ``sl_live`` (sl vivo en $/lote, ATR
    shift-1 de una entrada justo tras el final del feed), ``pctile_sl_live``
    (percentil del sl vivo en la distribución histórica de sl-en-entradas de
    la pata, empates al rango superior) y ``sl_live_asof`` (última barra M1,
    post eje de días). En modo rápido, sin .sqx, o si el exacto falla, los
    tres son ``None`` y ``regimen_warns == []``. El bloque tiene su PROPIO
    try/except: NUNCA activa ``exact_failed`` ni degrada el sizing (sus
    avisos van solo a ``regimen_warns``, nunca a ``exact_warns``).
    """
    if _is_sqx_leg_path(path_rel):
        return _load_sqx_leg(path_rel, account_size, day_labeler_tz,
                             day_labeler_cut, axis_hours=axis_hours,
                             with_minutes=with_minutes, synth_mode=synth_mode)
    p = ROOT / path_rel
    sqx_path = _sqx_sibling_path(path_rel)
    if sqx_path is None:
        if sqx_map is None:
            sqx_map = _sqx_match_paths()
        sqx_path = sqx_map.get(p.stem)

    tl = load_trade_list(p.read_bytes(), f_ref=f_ref)
    counts = multi_symbol_counts(tl)
    if counts is not None:
        raise ValueError(f"{p.name}: CSV multi-símbolo ({len(counts)} símbolos); "
                         "usa un CSV por estrategia.")
    sym_csv = str(tl.df["symbol"].iloc[0])
    root = sym_csv.split("_")[0].split(".")[0]
    sym = next((s for s in _broker.symbols
                if root in (s.ticker_axi.split(".")[0], s.ticker_cascade or "")), None)
    if sym is None:
        raise ValueError(f"{p.name}: símbolo '{sym_csv}' no está en broker_axi.yaml")

    from src.risk.atr import resolve_ohlc_path
    try:
        ohlc_resolved = str(resolve_ohlc_path(sym_csv))
        ohlc_mtime = _mtime(Path(ohlc_resolved))
    except Exception:
        ohlc_resolved, ohlc_mtime = None, None
    key = (path_rel, account_size, f_ref, _mtime(p),
           str(sqx_path) if sqx_path else None,
           _mtime(sqx_path) if sqx_path else None,
           ohlc_resolved, ohlc_mtime, axis_hours, with_peaks, day_labeler_tz,
           day_labeler_cut)
    if key in _leg_cache:
        return _leg_cache[key]

    ohlc = _load_ohlc(sym_csv)
    if ohlc is None:
        raise ValueError(f"{p.name}: sin OHLC M1 para '{sym_csv}' (en cartera no "
                         "hay fallback silencioso)")

    # F2c-2: eje de días ANTES de precomputo/sizing → bloques flat-flat,
    # valles M1 y ATR-vs-entries (sizing exacto) quedan todos en el mismo eje.
    tl, ohlc = apply_day_axis_shift(tl, ohlc, axis_hours)

    # --- Sizing exacto (F5a, §8.4-Q1/Q4, §8.6-4/5) -----------------------------
    sl_exact = None
    max_lots = float("inf")
    sizing_mode = "rápido"
    exact_warns: list[str] = []
    exact_failed = False
    mm = None
    atr_series = None
    if sqx_path is not None:
        try:
            # Reuso read-only de risk_app (Q1 adjudicada): el módulo migrará a
            # un hogar neutro cuando Streamlit se retire; import perezoso como
            # en _load_ohlc.
            from risk_app.exact_mm import exact_sizing_from_sqx
            sl_exact, mm = exact_sizing_from_sqx(
                str(sqx_path), tl, ohlc, float(sym.point_value_usd))
            sizing_mode = "exacto"
            max_lots = float(mm.max_lots) if mm.max_lots else float("inf")
            # La serie ATR se computa UNA vez y se reusa en la cobertura y en
            # los campos de régimen (refactor coste-cero, §Pieza 1+2).
            from src.risk.atr import compute_atr_series
            atr_series = compute_atr_series(ohlc, mm.timeframe,
                                            int(mm.atr_period))
            n_out = _atr_out_of_coverage(ohlc, tl, mm, atr_series=atr_series)
            if n_out:
                exact_warns.append(
                    f"{n_out} trade(s) fuera de cobertura ATR: el sizing "
                    f"exacto clampa al valor extremo (revisa el rango del OHLC).")
            warn_cc = _exact_size_crosscheck(tl, sl_exact, mm)
            if warn_cc:
                exact_warns.append(warn_cc)
        except Exception as exc:
            exact_failed = True
            exact_warns.append(f"modo exacto falló ({exc}); sizing rápido.")

    # --- Campos de régimen ATR (semáforo informativo C1; producto -----------
    # gap-régimen-ATR, piezas 1+2). Try/except PROPIO: nunca activa
    # exact_failed ni degrada el sizing; sus avisos van solo a regimen_warns.
    sl_live = pctile_sl_live = sl_live_asof = None
    regimen_warns: list[str] = []
    try:
        # exact_failed puede ser True con mm asignado si la cobertura o el
        # crosscheck lanzan después del parse: en ese caso NO se calcula nada.
        if (not exact_failed and mm is not None and atr_series is not None
                and sl_exact is not None):
            sl_live = _sl_live_from_atr(atr_series, ohlc.index[-1],
                                        float(mm.atr_mult),
                                        float(sym.point_value_usd))
            if not np.isfinite(sl_live) or sl_live <= 0:
                regimen_warns.append(
                    f"sl vivo no positivo/finito ({sl_live}): campos de "
                    "régimen a null.")
                sl_live = pctile_sl_live = sl_live_asof = None
            else:
                pctile_sl_live = _pctile_rango_superior(sl_live, sl_exact)
                if pctile_sl_live is None:
                    regimen_warns.append(
                        "sin sl_exact válidos (finitos > 0): percentil del "
                        "sl vivo a null.")
                elif not (0.0 <= pctile_sl_live <= 100.0):
                    regimen_warns.append(
                        f"percentil del sl vivo fuera de [0, 100] "
                        f"({pctile_sl_live}): percentil a null.")
                    pctile_sl_live = None
                sl_live_asof = str(ohlc.index[-1])
    except Exception as exc:  # noqa: BLE001 - el semáforo nunca rompe la carga
        sl_live = pctile_sl_live = sl_live_asof = None
        regimen_warns.append(f"campos de régimen fallaron ({exc}); a null.")

    # E16 (csv-d1-d2, fase 2): el aviso del motor (``initial_capital_mismatch``) dice «los lotes del CSV están
    # escalados a otra equity; el P(pass) no es fiable», y desde D1 es falso: el lote se rebasa al tamaño de cuenta y
    # el PnL por lote no depende de la escala del backtest. La función del motor NO se toca (la usan otras herramientas
    # que siguen componiendo): aquí solo se aprovecha su tolerancia (5 %) como predicado y el mensaje de la API es
    # informativo. Como en las patas .sqx (más abajo), el texto menciona el «capital» (lo recoge la guarda `capital`).
    warn = None
    if initial_capital_mismatch(tl, account_size) is not None:
        warn = (f"capital del backtest ${tl.initial_equity:,.0f} ≠ tamaño de cuenta ${account_size:,.0f}: "
                "informativo, la app dimensiona cada orden sobre el tamaño de cuenta.")
    if abs(tl.initial_equity - account_size) > 1e-9:
        # Convención LegCache del batch FTMO: replay de sizing sobre la cuenta. Desde D1 (csv-d1-d2) esta
        # ``initial_equity`` ES la base del lote de la pata (``_leg_lots``: compound=False). El cross-check
        # de arriba (``_exact_size_crosscheck``) se llama ANTES de esta sustitución, con la equity original
        # del CSV: reproduce la columna ``Size`` del backtest, que compone.
        tl = dataclasses.replace(tl, initial_equity=account_size)
    # Fase 6: etiquetado por sesión ET (perfiles day_axis="et_session") — el
    # MISMO labeler alimenta el groupby de envolventes y close_d aquí, y
    # open_ord en build_day_blocks_portfolio / los minutos trailing en el
    # setup del endpoint (un solo punto de remap por corrida).
    labeler = (make_session_day_labeler(day_labeler_tz,
                                        cut_hhmm=day_labeler_cut)
               if day_labeler_tz else None)
    s = precompute_strategy(tl, ohlc, sym.point_value_usd, key=p.name,
                            day_labeler=labeler)
    attach_minute_valleys(s, ohlc, np.asarray(tl.open_price, dtype=float),
                          with_peaks=with_peaks)
    out = {"tl": tl, "s": s, "sym": sym, "sym_csv": sym_csv, "warn": warn,
           "name": p.name, "path": path_rel,
           "sl_exact": sl_exact, "max_lots": max_lots,
           "sizing_mode": sizing_mode,
           "sqx_name": sqx_path.name if sqx_path else None,
           "exact_warns": exact_warns,
           "sl_live": sl_live, "pctile_sl_live": pctile_sl_live,
           "sl_live_asof": sl_live_asof, "regimen_warns": regimen_warns}
    if not exact_failed:
        _leg_cache[key] = out
    return out


def _load_sqx_leg(path_rel: str, account_size: float,
                  day_labeler_tz: str | None,
                  day_labeler_cut: str = "18:00", *,
                  axis_hours: float = 0.0,
                  with_minutes: bool = False,
                  synth_mode: str = "point") -> dict:
    """Pata de futuros NATIVA desde un ``.sqx`` (``kind="sqx_futures"``).

    La FICHA de símbolo se resuelve por el símbolo del ``.sqx`` (``fichas.resolve``; guardrail
    entrada-4-bloques, fase F2): sin ficha, o con ficha no usable, es un error de pata. Luego
    ``read_trade_table_spec`` (tabla de trades del ``orders.bin`` contra el valor del punto, la comisión y los
    pasos de lote de la ficha) → ``strategy_from_trade_table`` (``StrategyMC`` con ``valle={}`` + datos por
    trade para el sizing Opción A y el valle por ventanas). Si se pide eje de
    sesión (``day_labeler_tz`` truthy) y el reloj del bróker es ``ny_wall`` el labeler es
    ``make_session_day_labeler("America/New_York", cut_hhmm=day_labeler_cut)``
    (reloj de pared ET; el ``source_tz`` de la request NO interviene); si no,
    ``None`` (día calendario). Clave de caché: ruta, mtime, tz y cut (+
    ``account_size``, que entra en el aviso de capital) y la FIRMA de la ficha
    (:func:`poc_web.api.symbols.sheet_signature`: ficha, bróker, OHLC y su mtime), así editar una ficha no sirve
    una pata vieja.

    ``SQXFutureLegError``/``SQXFormatError`` → ``ValueError`` con el mensaje
    (el endpoint lo muestra como error de pata). El ``dict`` devuelto es
    compatible con el de las patas CSV (``tl=None``, ``sl_exact=None``,
    ``max_lots=inf``) y añade ``kind``, ``asset`` (la clase de activo del bróker), ``fleg`` y ``meta``, más
    ``broker`` (id), ``ficha`` (resumen con ``needs_review``), ``mm_backtest`` (MM del ``lastSettings.xml``,
    informativo) y ``paridad`` (E16: lotes del ``.sqx`` frente a la regla de riesgo fijo, solo con ``FixedAmount``).

    Patas CFD (``asset="cfd"``, reloj ``ny_plus_7``; FTMO 2026-10-12, AXI desde la fase F2): hora del servidor, sin
    labeler de sesiones; el eje de días es el desplazamiento ``axis_hours`` de
    las patas CSV (``"ftmo"`` = −1 h → medianoche CE(S)T), aplicado a apertura y
    cierre antes de construir la pata.

    ``with_minutes`` (2026-10-14, plan cfd-sqx-minuto-par): con ``True`` y pata CFD con
    OHLC M1 que CUADRA con el .sqx (coherencia ≥ ``SQX_MINUTE_COHERENCE_MIN``), el dict
    lleva además ``minute_valle`` (``{i: (ts, vals, ot, ct)}`` en el eje pedido, series
    en precio NETAS de spread en cortos, comisión y swap-cargo; ver
    :func:`_sqx_minute_series`) y ``minute_diag``; si no, ``minute_valle=None`` y
    ``minute_reason`` con el motivo. Con ``False`` (default) el resultado es el de siempre
    (sin esas claves). La clave de caché incluye ``with_minutes`` y el mtime del OHLC.

    ``synth_mode`` (fase F3 del guardrail entrada-4-bloques; plan §2.7): qué serie por minuto tienen las operaciones
    sin barras y las patas sin OHLC coherente (solo cuenta con ``with_minutes`` y bróker CFD; en otro caso se ignora y
    no entra en la clave):

    - ``"point"`` (por defecto; lente prop firm): lo de siempre, bit a bit. Una operación sin barras = un punto en
      ``ot``; una pata sin OHLC coherente = ``minute_valle=None`` y ``minute_reason``;
    - ``"daily"`` (lente AXI): una operación sin barras = un punto por fecha del servidor que abarca, con el suelo de
      la operación; una pata sin OHLC coherente, o con ficha sin ``ohlc_m1``/``spread_px``, tiene ``minute_valle``
      ENTERAMENTE sintético (cota diaria por el MAE) y ``minute_reason`` (no ``None``) con el motivo y «cota diaria
      por el MAE».

    El modo entra en la clave de caché (vía la firma de la ficha).

    Spread de la serie (enmienda E20): el del BACKTEST, ``defaultSpread · tickSize`` del ``InstrumentInfo`` del propio
    ``.sqx`` (:func:`_sqx_backtest_spread`), tanto en la coherencia con el OHLC como en el ajuste de los cortos y el suelo
    de la cota diaria; solo si el ``.sqx`` no lo trae o no es > 0, el ``spread_px`` de la ficha. El ``minute_diag`` dice
    cuál se usó (``spread_px_usado`` y ``spread_fuente``: ``"sqx"`` | ``"ficha"``). El fichero ya entra en la clave por su mtime.
    """
    if synth_mode not in SYNTH_MODES:
        raise ValueError(f"synth_mode debe ser uno de {SYNTH_MODES} (recibido {synth_mode!r})")
    p = ROOT / path_rel
    sym = _sqx_symbol(p)
    if sym is None:
        # .sqx ilegible (ZIP roto, sin settings.xml, sin resultado Main…): el lector da el error concreto
        try:
            read_trade_table(p)
        except (SQXFutureLegError, SQXFormatError) as exc:
            raise ValueError(f"{p.name}: {exc}") from exc
        raise ValueError(f"{p.name}: no se pudo leer el símbolo SQX del .sqx")
    sheet = fichas.resolve(sym)
    if sheet is None:
        # PLAN-DESVIO: el texto del plan es "sin ficha para el símbolo 'X': da de alta el símbolo (pestaña Símbolos)".
        # Se le añade "fuera de la allowlist de fichas" porque ``test_c2_extractor_error_es_error_de_pata``
        # (test_futures_sqx_legs.py, sin editar sus aserciones, AC2.7) exige "allowlist" en el error de un .sqx cuyo
        # símbolo no está dado de alta; hoy esa lista de símbolos admitidos ES el conjunto de fichas.
        raise ValueError(f"{p.name}: sin ficha para el símbolo '{sym}': da de alta el símbolo (pestaña Símbolos); "
                         "fuera de la allowlist de fichas" + _hidden_sheet_note(sym))
    if not sheet.usable:
        raise ValueError(f"{p.name}: ficha no usable para el símbolo '{sym}': {sheet.motivo}")
    broker = fichas.load_registry().brokers[sheet.broker]
    asset = broker.asset_class
    is_cfd = asset == SQX_ASSET_CFD
    # El reloj del bróker decide el eje de días (§2.6): "ny_wall" = reloj de pared de Nueva York → labeler de
    # sesiones NY (futuros); "ny_plus_7" = hora del servidor (NY + 7 h) → desplazamiento/transformación CFD.
    wall = broker.clock == "ny_wall"
    tz = SQX_WALL_CLOCK_TZ if (day_labeler_tz and wall) else None
    ohlc_path = _sheet_ohlc_path(sheet, broker) if (with_minutes and is_cfd) else None
    ohlc_mtime = _mtime(ohlc_path) if ohlc_path else None
    key = (KIND_SQX, path_rel, _mtime(p), tz, day_labeler_cut if tz else None,
           account_size) + ((asset, float(axis_hours)) if not wall else ())
    if with_minutes and is_cfd:
        # H9: centinela "" (no "None") cuando no hay ruta de OHLC (ficha sin ohlc_m1 o bróker sin carpeta)
        key += ("minutes", "" if ohlc_path is None else str(ohlc_path), ohlc_mtime)
    # E4: la firma de la ficha (con el bróker, el OHLC y su mtime) invalida la caché si cambia cualquier dato que
    # la pata usó. "point" es el modo de la lente prop firm (las claves de siempre) y "daily" el de la AXI (H11 de la
    # revisión de la F2: ya no va fijo). PLAN-DESVIO (menor): ``synth_mode`` solo entra en la clave cuando se calculan
    # series por minuto de un bróker CFD; sin ellas (``with_minutes=False`` o futuros) el dict devuelto no depende del
    # modo y una entrada por modo duplicaría la caché sin motivo (así las claves de siempre no cambian).
    eff_mode = synth_mode if (with_minutes and is_cfd) else "point"
    # PLAN-DESVIO (menor): ``needs_review`` no entra en la firma de la ficha (``symbols.py`` no está en la lista de la F4),
    # pero la pata cacheada lo lleva en ``ficha`` y de él salen ``ficha_sin_verificar`` y la guarda ``fichas`` de la carga.
    # Va aparte en la clave para que una ficha editada A MANO en el fichero de usuario (sin pasar por la API, cuyo gancho
    # de recarga vacía la caché) no sirva un aviso viejo.
    key += (fichas.sheet_signature(sheet, broker, ohlc_path, ohlc_mtime, synth_mode=eff_mode), tuple(sheet.needs_review))
    if key in _leg_cache:
        return _leg_cache[key]
    try:
        df, meta = read_trade_table_spec(p, fichas.reader_spec(sheet))
    except (SQXFutureLegError, SQXFormatError) as exc:
        raise ValueError(f"{p.name}: {exc}") from exc
    df_srv = df   # hora del servidor, ANTES del eje: el OHLC M1 está en esta hora
    to_axis = _sqx_axis_transform(axis_hours if not wall else 0.0)
    if (not wall) and axis_hours:
        # Eje FTMO = medianoche de Praga con horarios de verano REALES: el servidor
        # sigue el de EE. UU. y Praga el de la UE, así que en las semanas de desfase
        # (marzo y final de octubre) la diferencia es de 2 h, no de 1 (revisión
        # DeepSeek H2). Ver :func:`_sqx_axis_transform`.
        df = df.copy()
        df["open_ts"] = to_axis(df["open_ts"])
        df["close_ts"] = to_axis(df["close_ts"])
    # PLAN-DESVIO: (hallazgo de la Fase C, no previsto en el plan) los trades de
    # duración 0 (entrada y salida en la misma barra, open_ts == close_ts; 46 de
    # 2.989 en Strategy 3.42.158) NO entran en ningún bloque flat-flat:
    # ``detect_flatflat_blocks`` procesa close(-1) antes que open(+1) en el mismo
    # instante (active -1 → 0) y no abre bloque, así que su PnL y su MAE se
    # pierden en silencio (y con otras patas abiertas parten el bloque). Se les
    # suma 1 ns al cierre para que cuenten como un bloque de 1 ns. Con esto la
    # cota de ventanas pasa de 98,2% a 100% de días válidos frente a dailyEquity
    # (tests/test_windows_vs_dailyequity.py). El arreglo de fondo (motor) queda
    # como backlog: no está en la lista de ficheros de esta fase.
    n_zero = int((df["close_ts"] <= df["open_ts"]).sum())
    if n_zero:
        df = df.copy()
        z = df["close_ts"] <= df["open_ts"]
        df.loc[z, "close_ts"] = df.loc[z, "open_ts"] + pd.Timedelta(1, "ns")
        meta = {**meta, "warnings": [*meta["warnings"], (
            f"{n_zero} trade(s) de duración 0 (entrada y salida en la misma "
            "barra): close_ts +1 ns para que entren en los bloques flat-flat "
            "(si no, su PnL y su MAE se perderían en silencio).")]}
    labeler = (make_session_day_labeler(tz, cut_hhmm=day_labeler_cut)
               if tz else None)
    try:
        s, fleg = strategy_from_trade_table(df, meta, key=p.name,
                                            day_labeler=labeler)
    except Exception as exc:  # noqa: BLE001 - incluye horas DST ambiguas (fail-loud)
        raise ValueError(f"{p.name}: {type(exc).__name__}: {exc}") from exc
    cap = meta.get("initial_capital")
    warn = None
    if cap is not None and abs(float(cap) - account_size) > 1e-9:
        warn = (f"capital del .sqx ({float(cap):,.0f}) ≠ account_size "
                f"({account_size:,.0f}): informativo, el sizing Opción A usa "
                "account_size.")
    mm_backtest = read_backtest_mm(p)
    out = {"kind": KIND_SQX, "asset": asset, "tl": None, "s": s, "fleg": fleg,
           "meta": meta, "sym": None, "sym_csv": meta["symbol"], "warn": warn,
           "name": p.name, "path": path_rel, "sqx_name": p.name,
           "sizing_mode": SQX_SIZING_MODE,
           "exact_warns": list(meta["warnings"]),
           "sl_exact": None, "max_lots": float("inf"),
           # guardrail entrada-4-bloques, F2: bróker y ficha con que se leyó la pata, MM del backtest (dato
           # informativo) y paridad del lote del .sqx con la regla de riesgo fijo de la app (E16).
           "broker": broker.id,
           # F4: ``max_lot`` (el máximo del bróker en la ficha, ``None`` si no lo trae) es lo que compara la receta para el
           # EA con el lote máximo simulado (``max_lot_excede_ficha_by_f``). Entra en la firma de la ficha (clave de caché).
           "ficha": {"sqx_symbol": sheet.sqx_symbol, "broker": broker.id, "official": sheet.official,
                     "needs_review": list(sheet.needs_review), "sources": copy.deepcopy(sheet.sources),
                     "max_lot": sheet.max_lot},
           "mm_backtest": mm_backtest,
           "paridad": _fixed_amount_parity(mm_backtest, df["lots"].to_numpy(dtype=float), fleg, meta,
                                           account_size)}
    if with_minutes:
        # E20: el spread de la serie por minuto es el del backtest (``defaultSpread · tickSize`` del propio .sqx) y solo si
        # el .sqx no lo trae o no es > 0, el de la ficha. El fichero entra en la clave de caché por su mtime.
        mv, mdiag, mreason = _sqx_leg_minutes_sheet(df_srv, meta, fleg, sheet, broker, to_axis, synth_mode,
                                                    sqx_spread_px=_sqx_backtest_spread(p) if is_cfd else None)
        out.update(minute_valle=mv, minute_diag=mdiag, minute_reason=mreason)
    _leg_cache[key] = out
    return out


def _hidden_sheet_note(sym: str) -> str:
    """Nota para el error de "sin ficha": si el registro SÍ tiene una ficha de ese símbolo, es que está oculta."""
    sh = fichas.load_registry().sheets.get(sym)
    if sh is None:
        return ""
    return f" (la ficha del bróker '{sh.broker}' está oculta: muéstrala en la pestaña Símbolos)"


# PLAN-DESVIO: el brief escribe "Las patas son de {broker_cartera}; el perfil '{key}' opera en {broker_perfil}…" sin
# decir si el bróker va por id o por nombre. Se usa "<compañía> <CFD|Futures> (<id>)" (FTMO CFD (ftmo_cfd), FTMO Futures
# (ftmo_futures), Axi CFD (axi)) porque ``display_name`` ("FTMO (CFD, MT5)") no contiene "FTMO CFD" y los tests
# existentes ``test_simulacion_cfd_con_perfil_futures_es_409`` y ``test_c3_perfil_no_futures_con_cartera_sqx_es_409``
# (sin editar sus aserciones, AC2.7) exigen "FTMO CFD" / "FTMO Futures" en el 409 aunque ahora mande el del bróker.
def _broker_label(broker) -> str:
    """Nombre de un bróker para los mensajes de compatibilidad: ``"<compañía> <CFD|Futures> (<id>)"``."""
    return f"{broker.company} {'CFD' if broker.asset_class == SQX_ASSET_CFD else 'Futures'} ({broker.id})"


def _fixed_amount_parity(mm: dict | None, lots_sqx: np.ndarray, fleg, meta: dict,
                         account_size: float) -> dict | None:
    """Paridad (E16) del lote del backtest con la regla de riesgo fijo de la app, solo si el MM del backtest es
    ``FixedAmount`` con ``RiskedMoney`` R > 0. Informativa: no veta nada.

    X = R del ``lastSettings.xml`` (no la f de la rejilla): se llama a ``fixed_risk_micros`` con ``f = R/A`` y
    ``A = account_size``, de modo que A se cancela. Compara con los lotes del ``.sqx`` (tolerancia medio paso) y
    devuelve ``{"risked_money": R, "frac_ok", "n", "n_lifm", "frac_ok_sin_lifm"}``: ``n_lifm`` = órdenes cuyo
    ``floor`` da 0 (el backtest las dimensionó con ``LotsIfNoMM``, no con la regla) y ``frac_ok_sin_lifm`` = la
    fracción de acuerdo sin esas órdenes (``None`` si todas son de ese tipo). ``None`` en otro caso.

    H6 (revisión de código de la F2): la paridad es INFORMATIVA y nunca tumba la carga. Si las longitudes de los lotes
    del ``.sqx`` y del riesgo por lote no coinciden, o CUALQUIER excepción salta al calcularla, devuelve ``None``."""
    try:
        if mm is None or mm.get("method") != "FixedAmount":
            return None
        risked = mm["params"].get("RiskedMoney")
        if risked is None or not risked > 0:
            return None
        step = float(meta.get("lot_step") or 1.0)
        min_lot = float(meta.get("min_lot") or 1.0)
        risk = np.asarray(fleg.risk_per_micro, dtype=float)
        lots_sqx = np.asarray(lots_sqx, dtype=float)
        if lots_sqx.shape != risk.shape:
            return None
        lots_app, _ = fixed_risk_micros(risk, [risked / account_size], account_size, cap=None,
                                        comm_per_micro=fleg.comm_per_lot, lot_step=step, min_lot=min_lot)
        ok = np.abs(lots_app[0] - lots_sqx) <= step / 2.0
        lifm = np.floor(risked / (risk * step) + _FLOOR_EPS) <= 0
        return {"risked_money": float(risked), "frac_ok": float(np.mean(ok)), "n": int(len(ok)),
                "n_lifm": int(lifm.sum()),
                "frac_ok_sin_lifm": float(np.mean(ok[~lifm])) if (~lifm).any() else None}
    except Exception:  # noqa: BLE001 - dato corrupto o forma inesperada: la paridad es informativa, no veta la carga
        return None


def _leg_kind(leg: dict) -> str:
    """Tipo de una pata cargada (las CSV no llevan clave ``kind``)."""
    return leg.get("kind", KIND_CSV)


def _leg_n_trades(leg: dict) -> int:
    """Nº de trades de la pata (CSV: ``tl``; ``.sqx``: sin ``tl``)."""
    return leg["tl"].n_trades if leg.get("tl") is not None else leg["s"].n_trades


def _is_sqx_portfolio(pf: dict) -> bool:
    return pf.get("kind", KIND_CSV) == KIND_SQX


def _reject_sqx_portfolio(portfolio_id: str, what: str) -> None:
    """409 explícito si la cartera es de patas ``.sqx`` (v1: solo lente prop
    firm FTMO Futures). Se llama AL INICIO de los endpoints no soportados,
    antes de tocar ``tl``/``s``. Un ``portfolio_id`` desconocido no es de
    esta guarda (lo resuelve el 404 del endpoint)."""
    pf = _portfolios.get(portfolio_id)
    if pf is not None and _is_sqx_portfolio(pf):
        raise HTTPException(
            409, f"{what} no soporta carteras de patas .sqx de futuros (v1): "
            "solo el lente prop firm con un perfil FTMO Futures.")


def _atr_out_of_coverage(ohlc: pd.DataFrame, tl, mm,
                         atr_series: pd.Series | None = None) -> int:
    """Nº de trades cuyo entry cae fuera de la serie ATR (clip silencioso de
    ``sl_in_money_from_atr`` hecho visible, §8.6-5). atr_shift = 1 (SQX).

    ``atr_series`` (opcional, refactor coste-cero del producto
    gap-régimen-ATR): serie ATR ya computada por el caller; con ``None`` se
    calcula aquí (comportamiento histórico, bit-idéntico)."""
    if atr_series is None:
        from src.risk.atr import compute_atr_series
        atr_series = compute_atr_series(ohlc, mm.timeframe, int(mm.atr_period))
    atr = atr_series
    if not len(atr):
        return tl.n_trades
    ent = pd.to_datetime(tl.df["open_time"]).to_numpy()
    pos = np.searchsorted(atr.index.values, ent, side="right") - 1
    sel = pos - 1
    return int(np.sum((sel < 0) | (sel > len(atr) - 1)))


def _sl_live_from_atr(atr_series: pd.Series, last_ts: pd.Timestamp,
                      atr_mult: float, point_value: float) -> float:
    """sl en $/lote de una entrada inmediatamente posterior al final del feed
    (ATR shift-1, receta SQX del sizing exacto).

    El resultado es siempre ``atr_series.iloc[-2] · atr_mult · point_value``:
    salta la última barra TF, posiblemente PARCIAL. Si la última barra TF
    está completa, esto introduce 1 barra de retraso (conservador, aceptado
    para un semáforo mensual — revisión 2026-10-12, R8)."""
    from src.risk.atr import sl_in_money_from_atr
    entries_now = pd.Series([last_ts + pd.Timedelta(minutes=1)])
    return float(sl_in_money_from_atr(
        atr_series, entries_now, atr_mult=atr_mult,
        point_value=point_value)[0])


def _pctile_rango_superior(v: float, sl_exact) -> float | None:
    """Percentil de ``v`` en la distribución histórica de sl-en-entradas de la
    pata (solo sl válidos: finitos y > 0, el filtro del anexo
    capital-e-estrella). Convención de empates = RANGO SUPERIOR
    (searchsorted-right, como ``pct_in_history`` del prototipo σ-path).
    ``None`` si no hay sl válidos."""
    sl = np.asarray(sl_exact, dtype=float)
    sl_valid_sorted = np.sort(sl[np.isfinite(sl) & (sl > 0)])
    n_valid = len(sl_valid_sorted)
    if n_valid == 0:
        return None
    return float(100.0 * np.searchsorted(sl_valid_sorted, v, side="right")
                 / n_valid)


def _exact_size_crosscheck(tl, sl_exact: np.ndarray, mm) -> str | None:
    """Cross-check de cordura exacto-vs-columna ``Size`` (§8.6-5): mediana de
    lots_exacto(@risk del .sqx, equity original del CSV) / Size; banda
    histórica [0.90, 1.01] (warning, no veto — caza .sqx re-exportados con
    otra MM).

    D1 (csv-d1-d2): este replay COMPONE (``compound=True`` explícito, no depende de
    ``CSV_SIZING_COMPOUND``): la columna ``Size`` sale de un backtest que reinvierte, y solo el replay
    compuesto sobre la equity ORIGINAL del CSV (``tl`` antes de que ``_load_leg`` le ponga
    ``initial_equity = account_size``) la reproduce. Con ``compound=False`` el ratio caería fuera de la
    banda en cualquier backtest que haya compuesto."""
    if mm.risk is None:
        return None
    symbols = {s.ticker_axi: s for s in _broker.symbols}
    lots = simulate_f(tl, f_grid=np.array([float(mm.risk) / 100.0]),
                      symbols=symbols, sl_in_money=sl_exact,
                      max_lots=float(mm.max_lots) if mm.max_lots
                      else float("inf"), compound=True).lots[0]
    size = np.asarray(tl.size, dtype=float)
    mask = size > 0
    if not mask.any():
        return None
    med = float(np.median(lots[mask] / size[mask]))
    if not (0.90 <= med <= 1.01):
        return (f"cross-check exacto vs Size: mediana ratio {med:.3f} fuera de "
                f"[0.90, 1.01] — ¿.sqx re-exportado con otra MM que el CSV?")
    return None


def _sqx_match_paths() -> dict[str, Path]:
    """stem → RUTA del .sqx/.xml (matching por nombre): primero
    inputs/strategies del repo, después las raíces extra sin sobrescribir
    colisiones (misma convención setdefault de ``match_by_name``)."""
    _, sqx_by_stem = match_by_name([], scan_inputs()[1])
    out = {stem: lf.path for stem, lf in sqx_by_stem.items()}
    for root in EXTRA_INPUT_ROOTS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.sqx")) + sorted(root.rglob("*.xml")):
            out.setdefault(p.stem, p)
    return out


def _sqx_sibling_path(leg_rel: str) -> Path | None:
    """Emparejamiento preferente: .sqx/.xml en la MISMA carpeta que el CSV
    (los exports de SQX viajan juntos). None si no hay hermano."""
    d = (ROOT / leg_rel).parent
    for ext in (".sqx", ".xml"):
        cand = d / f"{Path(leg_rel).stem}{ext}"
        if cand.is_file():
            return cand
    return None


def _stage_barriers(stage: str):
    """Barreras de la etapa AXI con la receta de ``portfolio_mode.py:97-100``.

    ``baseline = allocation_at(5000)`` (equity de referencia 5k, hardcodeado en
    Streamlit), ``pass_line = baseline · target%``, ``ruin_line = baseline ·
    max_loss%`` (negativo). Sin overrides.
    """
    try:
        phase = _axi.get_phase(stage)
    except KeyError:
        raise HTTPException(404, f"Etapa AXI '{stage}' desconocida.")
    if phase.profit_target_pct is None:
        raise HTTPException(422, f"La etapa {phase.display_name} no tiene profit "
                                 "target definido para evaluar pass.")
    baseline = phase.allocation_at(5000.0)
    return (phase, baseline,
            baseline * phase.profit_target_pct / 100.0,
            baseline * phase.max_loss_pct / 100.0)


def _leg_lots(leg: dict, f_grid: np.ndarray, symbols: dict, f_ref: float):
    """Lots de la pata CSV sobre la grilla: sizing EXACTO si la carga lo resolvió
    (``sl_exact`` presente, con ``max_lots`` del .sqx); si no, rápido con
    ``f_ref`` (fallback etiquetado, §8.4-Q4). Único punto de sizing de las patas
    CSV de la API (los cuatro sitios: _strats_with_lots, lots_ref de carga,
    propfirm vía _lots_for_leg y leg_diagnostics).

    D1 (csv-d1-d2): el lote se calcula sobre el TAMAÑO DE CUENTA, sin componer —
    ``lots = floor(f·A/sl)`` con ``A = tl.initial_equity`` (= ``account_size`` de la carga) y los
    clamps de ``simulate_f`` (``LotsIfNoMM``, ``MaxLots``, lote mínimo) —, es decir
    ``simulate_f(..., compound=CSV_SIZING_COMPOUND)`` con ``CSV_SIZING_COMPOUND = False``. El modo
    exacto o rápido solo cambia cómo se obtiene el riesgo por lote (``sl``), no la base.

    Solo patas CSV (E9): una pata ``.sqx`` no tiene ``tl`` y usa ``fixed_risk_micros``
    (:func:`_lots_for_leg`)."""
    assert _leg_kind(leg) == KIND_CSV, (
        "_leg_lots solo dimensiona patas CSV; una pata .sqx usa fixed_risk_micros (_lots_for_leg)")
    sl = leg.get("sl_exact")
    return simulate_f(
        leg["tl"], f_grid=f_grid, symbols=symbols, sl_in_money=sl,
        f_ref=None if sl is not None else f_ref,
        max_lots=leg.get("max_lots", float("inf")),
        compound=CSV_SIZING_COMPOUND).lots


# PLAN-DESVIO: el plan fija el literal ``meta.sizing_base = "tamaño de cuenta (sin componer)"``. Con el valor por
# defecto de ``CSV_SIZING_COMPOUND`` (False) este helper devuelve exactamente ese literal; solo con el flag activado
# (A/B de la fase 3) el eco cambia, para no afirmar una base que no se ha aplicado.
def _sizing_base(kind: str = KIND_CSV) -> str:
    """Eco ``meta.sizing_base``: base sobre la que se dimensiona el lote (D1, csv-d1-d2).

    Las patas ``.sqx`` ya usaban X = f·A (Opción A, ``fixed_risk_micros``); las CSV, el tamaño de
    cuenta sin componer salvo que ``CSV_SIZING_COMPOUND`` se haya activado (A/B de investigación),
    caso en que el eco lo dice en vez de afirmar lo que no se ha aplicado."""
    if kind == KIND_CSV and CSV_SIZING_COMPOUND:
        return SIZING_BASE_COMPOUND
    return SIZING_BASE_ACCOUNT


# ---------------------------------------------------------------------------
# D2 (csv-d1-d2, fase 2): flotante por MINUTO con el realizado en las patas CSV
# ---------------------------------------------------------------------------
# PLAN-DESVIO: el brief fija las dos constantes pero no su validación. Sin ella, un valor mal escrito en un script de A/B
# (p. ej. "minuto") caería en silencio al camino ``joint``; con ella falla en voz alta. Con "minute"/"joint" no cambia nada.
def _check_float_flag(name: str, value: str) -> str:
    """Valida ``CSV_FLOAT_MODE``/``AXI_FLOAT_MODE``: un valor mal escrito en un A/B no debe caer en silencio al
    camino ``joint``."""
    if value not in ("minute", "joint"):
        raise ValueError(f"{name} debe ser 'minute' o 'joint' (recibido {value!r})")
    return value


# PLAN-DESVIO: el brief escribe ``enumerate(legs_used)``. Comprobado en el código: ``strats`` se construye iterando
# ``legs_used`` en orden con clave ``leg["name"]`` y ``lots_by``/``pv_by`` enumeran ``strats.values()``, así que con
# nombres únicos el orden es idéntico. Con nombres repetidos (la carga CSV no los rechaza, a diferencia de las .sqx) el
# dict colapsa y ``enumerate(legs_used)`` desalinearía el índice de pata con los lotes y con ``sym_id``; este helper
# sigue el orden de ``strats`` (como ya hace la rama trailing con ``enumerate(strats.values())``).
def _legs_in_strats_order(legs: list[dict], strats: dict) -> list[dict]:
    """Patas en el orden de ``strats`` (= el de ``lots_by``/``pv_by`` y el de ``sym_id`` en los bloques).

    ``strats`` se construye iterando las patas con clave ``leg["name"]``: el orden coincide con el de ``legs`` salvo
    que haya nombres repetidos (el dict colapsa); usar el de ``strats`` mantiene el índice de pata alineado con los
    lotes y con los bloques en ese caso también (igual que ``enumerate(strats.values())`` en la rama trailing)."""
    by_name = {leg["name"]: leg for leg in legs}
    return [by_name[k] for k in strats]


# PLAN-DESVIO: el brief da como causas del valor 0 ``tl.size[i] <= 0`` y ``not tl.has_mae``; la Enmienda E8 habla además de
# «MAE no finito». Se cubren las tres: tamaño no positivo o no finito, MAE no finito (también un MAE vacío en una fila
# del CSV, que ``load_trade_list`` normaliza a 0 y sería un «acotado con MAE 0» optimista y mudo) y valor del punto no
# positivo. Para los CSV cargados con ``load_trade_list`` normales el resultado es el del brief.
def _csv_mae_masks(leg: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(sintetica, cero)`` por trade de una pata CSV: qué trades no tienen serie M1 propia (``minute_valle``) y
    de ellos cuáles reciben valor 0 en vez de su MAE.

    ``sintetica[i]``: el trade ``i`` no tiene entrada en ``s.minute_valle`` (sin cobertura OHLC M1) y tiene
    apertura y cierre válidos (los de NaT no entran en ningún bloque). ``cero[i]``: de esos, los que no se pueden
    acotar con su MAE y cuentan como 0 (optimista), que tienen su propio contador y aviso (E8): la pata no trae
    columna MAE (``not tl.has_mae``), el tamaño original no es > 0 o el MAE/tamaño/valor del punto no es finito
    (incluye un MAE vacío en la fila del CSV, que ``load_trade_list`` normaliza a 0)."""
    assert _leg_kind(leg) == KIND_CSV, (
        "el flotante por minuto con MAE solo sirve a patas CSV; una pata .sqx lleva su propia serie por minuto o "
        "sus ventanas (_load_sqx_leg)")
    s, tl = leg["s"], leg["tl"]
    n = int(s.n_trades)
    size = np.asarray(tl.size, dtype=float)
    mae = np.asarray(tl.mae, dtype=float)
    if len(size) != n or len(mae) != n:
        raise ValueError(f"{leg.get('name')}: la pata CSV tiene {n} trades en la estrategia y {len(size)}/{len(mae)} "
                         "en la lista de trades (tamaño/MAE)")
    have = s.minute_valle or {}
    synth = np.array([(i not in have) and not (pd.isna(s.open_t[i]) or pd.isna(s.close_t[i]))
                      for i in range(n)], dtype=bool)
    pv = float(s.pv)
    if not tl.has_mae or not (np.isfinite(pv) and pv > 0.0):
        zero = np.ones(n, dtype=bool)
    else:
        with np.errstate(invalid="ignore"):
            zero = ~(np.isfinite(size) & (size > 0.0) & np.isfinite(mae))
            df = getattr(tl, "df", None)
            if df is not None and "mae" in getattr(df, "columns", ()):
                raw = pd.to_numeric(df["mae"], errors="coerce").to_numpy(dtype=float)
                if len(raw) == n:
                    zero |= ~np.isfinite(raw)
    return synth, synth & zero


def _csv_minute_series(leg: dict) -> tuple[dict, int, int]:
    """``(series, n_mae_bounded, n_mae_zero)`` de una pata CSV para el modo ``minute`` (D2; plan csv-d1-d2 §2.2,
    E8-E10).

    ``series`` = ``{**s.minute_valle, **sintéticas}`` con el formato exacto de ``minute_valle``
    (``{local_id: (ts, vals, ot, ct)}``, en unidades de PRECIO adverso). El modo ``minute`` del motor exige serie de
    TODOS los trades del bloque, y los trades CSV sin cobertura OHLC M1 no la tienen: se les añade una serie de UN
    punto en la apertura, ``v = tl.mae[i] / (tl.size[i] · pv)`` (MAE del CSV en $ a su tamaño original, convertido a
    precio adverso por unidad), que el constructor del trailing mantiene constante desde ``ot`` hasta ``ct`` (la
    envolvente está activa en ``[ot, ct)``): el peor flotante de la operación contado desde que abre, sin suponer
    cuándo ocurrió — cota conservadora. El motor lo reescala por ``lots·pv``, así que el flotante en $ es ``MAE ·
    lots/size``. ``ot``/``ct`` son los de ``s`` (ya en el eje de días de la carga). Con ``cero`` (ver
    :func:`_csv_mae_masks`) el valor es 0 y se cuenta aparte en ``n_mae_zero``; el resto, en ``n_mae_bounded``.

    No modifica ``s.minute_valle``: el camino trailing de Apex sigue viendo las series originales. Los contadores
    son de TODA la pata; los de una simulación (ventana y bloques) los da :func:`_mae_usage`."""
    synth, zero = _csv_mae_masks(leg)      # incluye el assert de pata CSV (E9)
    s, tl = leg["s"], leg["tl"]
    series = dict(s.minute_valle or {})
    pv = float(s.pv)
    for i in np.flatnonzero(synth):
        i = int(i)
        v = 0.0 if zero[i] else float(tl.mae[i] / (tl.size[i] * pv))
        ot, ct = s.open_t[i], s.close_t[i]
        series[i] = (np.array([ot], dtype="datetime64[ns]"), np.array([v], dtype=float),
                     pd.Timestamp(ot), pd.Timestamp(ct))
    n_zero = int(zero.sum())
    return series, int(synth.sum()) - n_zero, n_zero


# PLAN-DESVIO: el brief devuelve los contadores desde ``_csv_minute_series`` (de TODA la pata) y pide ``meta.n_mae_bounded`` /
# ``meta.n_mae_zero`` más el peso PnL en las dos lentes. Para que el aviso hable de lo que la simulación realmente usa
# (la ventana OOS, los trades que están en los bloques) y coincida con ``n_no_valley`` (= acotados + a cero), los de la
# ``meta`` y los avisos se cuentan sobre los trades de los bloques con esta función; los de la pata entera siguen en
# ``_csv_minute_series``. En AXI salen a 0 hoy: ``build_block_records`` solo incluye trades con valle.
def _mae_usage(blocks, legs_ord: list[dict], lots_by: dict, f_idx: int) -> dict:
    """Trades de ``blocks`` servidos por una serie sintética de MAE (D2, E8): contadores y peso PnL.

    ``blocks``: ``PortfolioDayBlock`` (prop firm) o ``BlockRecord`` (AXI); ``legs_ord``: patas en el orden de
    ``sym_id`` (:func:`_legs_in_strats_order`); ``lots_by`` = ``{sym_id: lotes (n_f, n_trades)}``. El peso PnL es la
    parte de Σ|lotes·PnL por lote| (a ``f_idx``) de los trades de los bloques. Devuelve ``n_mae_bounded``,
    ``n_mae_zero`` (acotados con su MAE / con valor 0, optimista) y ``pnl_w_mae_bounded``, ``pnl_w_mae_zero``
    (NaN si no hay peso)."""
    masks = [_csv_mae_masks(leg) for leg in legs_ord]
    n_b = n_z = 0
    w_b = w_z = w_tot = 0.0
    for b in blocks:
        for j in range(len(b.sym_id)):
            si, loc = int(b.sym_id[j]), int(b.trade_local_id[j])
            w = abs(lots_by[si][f_idx, loc] * b.pnl_per_lot[j])
            w_tot += w
            synth, zero = masks[si]
            if synth[loc]:
                if zero[loc]:
                    n_z += 1
                    w_z += w
                else:
                    n_b += 1
                    w_b += w
    return {"n_mae_bounded": n_b, "n_mae_zero": n_z,
            "pnl_w_mae_bounded": (w_b / w_tot) if w_tot else float("nan"),
            "pnl_w_mae_zero": (w_z / w_tot) if w_tot else float("nan")}


# PLAN-DESVIO: en el primer aviso ("N trades sin OHLC M1: su flotante se acota con el MAE…") N son los trades ACOTADOS con su
# MAE, no todos los que no tienen M1: los de valor 0 no se acotan con nada y llevan su propio aviso (E8); sumados dan
# ``n_no_valley``. Los dos textos son los del brief.
def _mae_warnings(usage: dict) -> list[str]:
    """Avisos de los trades sin OHLC M1 del modo ``minute`` (los mismos en prop firm y AXI): los acotados con su MAE y,
    aparte, los que cuentan como 0 por no tener MAE utilizable (optimista)."""
    out = []
    if usage["n_mae_bounded"]:
        out.append(f"{usage['n_mae_bounded']} trades sin OHLC M1: su flotante se acota con el MAE del CSV desde la "
                   f"apertura (peso PnL f central {usage['pnl_w_mae_bounded']:.1%}).")
    if usage["n_mae_zero"]:
        out.append(f"{usage['n_mae_zero']} trades sin MAE en el CSV: su flotante intradía cuenta como 0 (optimista).")
    return out


CSV_MINUTE_FLOAT_NOTE = (
    "flotante por minuto (OHLC M1 de cada pata): COTA por minuto — el peor precio de cada minuto para todas las "
    "posiciones a la vez —, con el realizado intradía dentro del peor momento del día: lo ya cerrado ese día se suma "
    "al flotante de lo que sigue abierto. La envolvente de cada operación termina en su minuto de cierre, donde entra "
    "el realizado: lo que ocurriera dentro de la barra de cierre no se cuenta (limitación: si todo lo adverso de una "
    "operación cae en esa barra, la cota no lo ve). Las órdenes sin precios M1 se acotan con el MAE del CSV desde la "
    "apertura (su peor flotante, sin suponer cuándo ocurrió); sin MAE en el CSV su flotante cuenta como 0 "
    "(optimista). No es la equity exacta.")


def _csv_minute_meta(ctx: dict) -> dict:
    """Claves de ``meta`` de una cartera CSV simulada con ``float_mode="minute"`` (lente prop firm, 1D y par 2D):
    la nota del flotante y los contadores de trades sin OHLC M1 (E8, E10). ``{}`` en cualquier otro caso (los
    perfiles con trailing y las carteras ``.sqx`` llevan las suyas)."""
    usage = ctx.get("csv_minute")
    if usage is None:
        return {}
    return {"float_note": CSV_MINUTE_FLOAT_NOTE, **usage}


def _leg_regimen_fields(leg: dict, f: float, account_size: float) -> dict:
    """Campos de régimen/canal de dinero por pata (producto gap-régimen-ATR,
    piezas 1+2) para el endpoint ``leg_diagnostics``.

    Aditivos y null-tolerant: NUNCA lanza y NUNCA muta ``leg`` ni
    ``sl_exact``. Sin ``sym`` (patas .sqx de futuros), sin ``sl_exact``
    (modo rápido), con ``f <= 0``/``account_size <= 0`` o sin sl válidos
    (finitos > 0, el filtro del anexo capital-e-estrella) los campos
    derivados salen a ``None``.

    - ``clamp_rate``: ``mean(f·E < min_lot·sl)`` — fórmula fixed-E del anexo
      capital-e-estrella: clampea cuando el presupuesto f·E NO cubre el suelo
      min_lot·sl (a mayor sl, MÁS clamp). No es ``n_sub_min_lot`` de
      ``simulate_f`` (replay con equity compuesta); equivalencia a equity
      fija candada por test.
    - ``floor_risk_med_pct_e`` / ``floor_risk_p99_pct_e``: riesgo suelo
      ``min_lot·mediana/P99(sl_exact)`` como % de ``account_size``. DISTINTO
      del legacy ``floor_risk_pct`` de :func:`leg_diagnostics`
      (``src/risk/input_checks.py``): aquel usa sl BACK-DERIVADO del modo
      rápido sobre el ``baseline`` de la etapa; este usa el sl EXACTO (ATR
      real del .sqx) sobre la equity de la cartera.
    - ``e_star_med`` / ``e_star_p99``: capital que des-clampea la pata a la
      mediana / P99 del sl (``min_lot·sl/f``).
    - ``sl_live_per_lot`` / ``pctile_sl_live`` / ``sl_live_asof``:
      passthrough de los campos de régimen calculados en la carga (la clave
      de pata es ``sl_live``; mapeo explícito).
    - ``would_be_clamp_now``: ``f·E < min_lot·sl_live`` (cláusula 2 del
      trigger compuesto C1); ``None`` si no hay sl vivo.
    """
    fields: dict = {
        "clamp_rate": None,
        "floor_risk_med_pct_e": None,
        "floor_risk_p99_pct_e": None,
        "e_star_med": None,
        "e_star_p99": None,
        # Passthrough con mapeo explícito de nombres (clave pata → clave API).
        "sl_live_per_lot": leg.get("sl_live"),
        "pctile_sl_live": leg.get("pctile_sl_live"),
        "sl_live_asof": leg.get("sl_live_asof"),
        "would_be_clamp_now": None,
    }
    _DERIVED = ("clamp_rate", "floor_risk_med_pct_e", "floor_risk_p99_pct_e",
                "e_star_med", "e_star_p99", "would_be_clamp_now")
    sym = leg.get("sym")
    sl_exact = leg.get("sl_exact")
    try:
        f = float(f)
        E = float(account_size)
        if sym is None or sl_exact is None or f <= 0 or E <= 0:
            return fields
        sl = np.asarray(sl_exact, dtype=float)
        sl_valid = sl[np.isfinite(sl) & (sl > 0)]
        if not len(sl_valid):
            return fields
        ml = float(sym.min_lot)
        fields["clamp_rate"] = float(np.mean(f * E < ml * sl_valid))
        fields["floor_risk_med_pct_e"] = (
            100.0 * ml * float(np.median(sl_valid)) / E)
        fields["floor_risk_p99_pct_e"] = (
            100.0 * ml * float(np.percentile(sl_valid, 99)) / E)
        fields["e_star_med"] = ml * float(np.median(sl_valid)) / f
        fields["e_star_p99"] = ml * float(np.percentile(sl_valid, 99)) / f
        sl_live = leg.get("sl_live")
        if sl_live is not None:
            fields["would_be_clamp_now"] = bool(f * E < ml * float(sl_live))
    except Exception:  # noqa: BLE001 - el diagnóstico nunca tumba el endpoint
        fields.update({k: None for k in _DERIVED})
    return fields


def _sizing_label(legs: list[dict]) -> str:
    """Etiqueta de sizing para meta: exacto / rápido / mixto (§8.4-Q4)."""
    modes = {l["sizing_mode"] for l in legs}
    if modes == {"exacto"}:
        return "exacto"
    if modes == {"rápido"}:
        return "rápido"
    if modes == {SQX_SIZING_MODE}:
        return SQX_SIZING_MODE
    return "mixto"


def _lots_for_leg(leg: dict, f_grid: np.ndarray, symbols: dict, f_ref: float,
                  account_size: float, cap: float | None):
    """Único helper de sizing por ``kind`` (enmienda V7). Devuelve
    ``(lots (n_f, n_trades), diag | None)``.

    CSV → :func:`_leg_lots` (``diag=None``): lote sobre el tamaño de cuenta, sin
    componer (D1, csv-d1-d2: ``lots = floor(f·A/sl)`` con los clamps de
    ``simulate_f``). ``.sqx`` → Opción A
    (:func:`src.risk.futures_legs.fixed_risk_micros`): ``X = f · account_size``
    estático, ``n = min(cap, max(1, floor(X/riesgo)))``; ``diag`` = lista por f
    con ``frac_clamp_up``/``p95_risk_eff_over_X``/``frac_capped``/
    ``p95_total_adverse_over_X``. Las dos ramas dimensionan sobre el tamaño de cuenta."""
    if _leg_kind(leg) == KIND_SQX:
        fleg = leg["fleg"]
        meta = leg.get("meta") or {}
        if leg.get("asset") == SQX_ASSET_CFD:
            cap = None   # el tope de contratos (micros) es de futuros; CFD no lo tiene
        return fixed_risk_micros(fleg.risk_per_micro, f_grid, account_size,
                                 cap=cap, comm_per_micro=fleg.comm_per_lot,
                                 lot_step=float(meta.get("lot_step") or 1.0),
                                 min_lot=float(meta.get("min_lot") or 1.0))
    return _leg_lots(leg, f_grid, symbols, f_ref), None


def _strats_with_lots(pf: dict, f_grid: np.ndarray, symbols: dict) -> dict:
    """Estrategias de la cartera con lots de la grilla (sizing por pata vía
    :func:`_leg_lots`: exacto si la carga lo resolvió, rápido si no; en ambos casos sobre el
    tamaño de cuenta, sin componer — D1, csv-d1-d2)."""
    strats = {}
    for leg in pf["legs"]:
        strats[leg["name"]] = attach_lots(
            leg["s"], _leg_lots(leg, f_grid, symbols, pf["f_ref"]))
    return strats


# ---------------------------------------------------------------------------
# Perfiles extendidos (plan Apex Fase 6): trailing / sesiones ET / ladder
# ---------------------------------------------------------------------------
def _profile_flags(cfg) -> dict:
    """Banderas del perfil: trailing (alguna fase o la PA con drawdown
    trailing) y et_session (eje de sesiones CME 18:00→16:59 ET)."""
    trailing = any(ph.drawdown is not None for ph in cfg.phases) \
        or cfg.funded.drawdown is not None
    return {"trailing": trailing, "et_session": cfg.day_axis == "et_session"}


def _cfd_profile_ok(cfg) -> bool:
    """Perfil válido para patas .sqx CFD FTMO: FTMO de ciclo (``ftmo_*``, modelo
    ``cycle``) con reglas estáticas (pérdida diaria y máxima sobre el inicial, sin
    trailing ni eje de sesiones). Revisión DeepSeek H4: identidad explícita, no la
    mera negación de futures."""
    flags = _profile_flags(cfg)
    # onboarding-c: un perfil de usuario hereda la identidad del oficial RAIZ (``based_on``),
    # leída bajo el lock (coherente con el ``_firms`` del que salió ``cfg``).
    with _FIRMS_LOCK:
        root_id = profiles.based_on(str(cfg.key))
    return (cfg.funded.model == "cycle"
            and root_id.startswith("ftmo_")
            and not flags["trailing"] and not flags["et_session"])


def _profile_extended(cfg) -> bool:
    """Perfil "extendido" (trailing, sesiones ET o modelo distinto de ``cycle``).

    Equivale EXACTAMENTE a ``_profile_rules(cfg) is not None`` de antes del plan
    onboarding-b1 (invariante I4): el front usa este flag, no la presencia de
    ``rules``, para decidir eje de días, multi-cuenta, caps y ``source_tz``."""
    flags = _profile_flags(cfg)
    return bool(flags["trailing"] or flags["et_session"]
                or cfg.funded.model != "cycle")


def _profile_kind(cfg) -> str:
    """``"cfd"`` si el modelo es ``cycle``; si no ``"futures"`` (Apex opera
    futuros micro; FTMO Futures también)."""
    return "cfd" if cfg.funded.model == "cycle" else "futures"


def _cycle_rules(cfg) -> dict:
    """Reglas del perfil FTMO CFD (``cycle``) para el panel "Reglas del perfil"
    (plan onboarding-b1 §2.2). ``horizon_cap_days`` es un truncado del modelo."""
    f = cfg.funded
    return {
        "model": "cycle",
        "ruleset_version": cfg.ruleset_version,
        "phases": [
            {"name": ph.name, "profit_target_pct": ph.profit_target_pct,
             "max_daily_loss_pct": ph.max_daily_loss_pct,
             "max_loss_pct": ph.max_loss_pct,
             "min_trading_days": ph.min_trading_days}
            for ph in cfg.phases
        ],
        "funded": {
            "model": "cycle",
            "profit_split_pct": cfg.profit_split_pct,
            "payout_cycle_days": f.payout_cycle_days,
            "min_payout_profit_pct": f.min_payout_profit_pct,
            "max_payout_cycles": f.max_payout_cycles,
            "horizon_cap_days": f.payout_cycle_days * f.max_payout_cycles,
        },
        "refund_fee_on_first_payout": cfg.refund_fee_on_first_payout,
    }


def _profile_rules(cfg) -> dict:
    """Bloques de reglas del perfil para el panel read-only "Reglas del
    perfil" (plan Apex Fase 6). Desde onboarding-b1 devuelve reglas para TODOS
    los perfiles: el dict extendido de siempre (idéntico, I2) si
    :func:`_profile_extended`, y el dict ``cycle`` (:func:`_cycle_rules`) si no."""
    if not _profile_extended(cfg):
        return _cycle_rules(cfg)
    ph = cfg.phases[0]
    f = cfg.funded
    rules = {
        "ruleset_version": cfg.ruleset_version,
        "day_axis": cfg.day_axis,
        # Fase 3 (FTMO Futures): corte ET del eje de sesiones del perfil.
        "day_axis_cut_hhmm": cfg.day_axis_cut_hhmm,
        "evaluation": {
            "profit_target_usd": cfg.account_size * ph.profit_target_pct / 100.0,
            "lifetime_days": ph.lifetime_days,
            "on_horizon_exhausted": ph.on_horizon_exhausted,
            "trailing_usd": ph.drawdown.amount_usd if ph.drawdown else None,
            "max_contracts": (
                {"max_mini": ph.max_contracts.max_mini,
                 "max_micro": ph.max_contracts.max_micro}
                if ph.max_contracts else None),
        },
        "funded": {
            "model": f.model,
            "profit_split_pct": cfg.profit_split_pct,
            "trailing": (
                {"amount_usd": f.drawdown.amount_usd,
                 "lock_at_balance": f.drawdown.lock_at_balance,
                 "frozen_floor": f.drawdown.frozen_floor}
                if f.drawdown else None),
            "dll_by_tier_usd": (list(f.daily_loss.by_tier_usd)
                                if f.daily_loss else None),
            "dll_post_hit": f.daily_loss.post_hit if f.daily_loss else None,
            "scaling_tiers": [
                {"level": t.level, "profit_lo": t.profit_lo,
                 "profit_hi": t.profit_hi, "max_mini": t.max_mini,
                 "max_micro": t.max_micro}
                for t in f.scaling_tiers
            ],
            "payouts": (
                {"qualifying_day_min_usd": f.payouts.qualifying_day_min_usd,
                 "days_per_payout": f.payouts.days_per_payout,
                 "ladder_usd": list(f.payouts.ladder_usd),
                 "min_payout_usd": f.payouts.min_payout_usd,
                 "min_balance_first": f.payouts.min_balance_first,
                 "residual_floor": f.payouts.residual_floor,
                 "max_payouts": f.payouts.max_payouts,
                 "consistency": {
                     "max_best_day_share": f.payouts.consistency.max_best_day_share,
                     "strict": f.payouts.consistency.strict,
                     "window": f.payouts.consistency.window,
                 }}
                if f.payouts else None),
            "inactivity": (
                {"window_days": f.inactivity.window_days,
                 "min_days": f.inactivity.min_days,
                 "min_profit_usd": f.inactivity.min_profit_usd}
                if f.inactivity else None),
            "activation": (
                {"fee_usd": cfg.activation.fee_usd,
                 "deadline_days": cfg.activation.deadline_days}
                if cfg.activation else None),
        },
    }
    if f.model == "futures":
        # Fase 3: bloques futures para el panel "Reglas del perfil" (aditivos;
        # los Apex de arriba quedan a None/[] en estos perfiles).
        sub = cfg.subscription
        rules["evaluation"].update({
            "drawdown": (
                {"type": ph.drawdown.type,
                 "amount_usd": ph.drawdown.amount_usd,
                 "lock_at_balance": ph.drawdown.lock_at_balance,
                 "breach_on": ph.drawdown.breach_on}
                if ph.drawdown else None),
            "daily_loss_usd": ph.daily_loss_usd,   # Pro: hard fatal; Growth: None
            "consistency": (
                {"max_best_day_share": ph.consistency.max_best_day_share,
                 "strict": ph.consistency.strict}
                if ph.consistency else None),
            "min_trading_days": ph.min_trading_days,
        })
        rules["funded"].update({
            "drawdown": (
                {"type": f.drawdown.type,
                 "amount_usd": f.drawdown.amount_usd,
                 "lock_at_balance": f.drawdown.lock_at_balance,
                 "breach_on": f.drawdown.breach_on}
                if f.drawdown else None),
            "dll": (
                {"type": f.daily_loss.type,
                 "amount_usd": f.daily_loss.by_tier_usd[0],
                 "post_hit": f.daily_loss.post_hit}
                if f.daily_loss else None),
            "subscription": (
                {"monthly_fee_usd": sub.monthly_fee_usd,
                 "reset_fee_usd": sub.reset_fee_usd,
                 "billing_period_days": sub.billing_period_days,
                 "campaign_cap_days": sub.campaign_cap_days,
                 "funded_cap_days": sub.funded_cap_days,
                 "funded_start_delay_days": sub.funded_start_delay_days}
                if sub else None),
            "payouts_futures": (
                {"qualifying_day_min_usd": pf.qualifying_day_min_usd,
                 "days_per_payout": pf.days_per_payout,
                 "withdraw_share": pf.withdraw_share,
                 "request_cap_usd": pf.request_cap_usd,
                 "min_new_profit_frac": pf.min_new_profit_frac,
                 "min_payout_usd": pf.min_payout_usd,
                 "withdrawal_policy": pf.withdrawal_policy,
                 "buffer_usd": pf.buffer_usd}
                if (pf := f.payouts_futures) else None),
        })
    return rules


_FTMO_COMPANY = "FTMO"


def _profile_notes(cfg) -> tuple[list, str | None]:
    """``(notas aplicables, base_nombre)`` del perfil (guardrail perfiles-v2): además de
    modelo y tipo, filtran su compañía, su nº de fases de evaluación y sus etiquetas
    (``usuario`` -> ``reglas_de_usuario``; ``copia`` -> ``plantilla_base``)."""
    with _FIRMS_LOCK:
        ctx = profiles.note_context(str(cfg.key))
    notes = notes_for("propfirm", cfg.funded.model, _profile_kind(cfg),
                      empresa=ctx["company"], fases=len(cfg.phases), perfil=ctx["tags"])
    return notes, ctx["base_nombre"]


def _profile_broker_problem(cfg, pf_broker: str) -> str | None:
    """Por qué las patas ``.sqx`` del bróker ``pf_broker`` no encajan con el perfil ``cfg`` (guardrail
    entrada-4-bloques, §2.5), o ``None`` si encajan. El bróker del perfil es el de su perfil OFICIAL raíz
    (``config/brokers.yaml``; un perfil del usuario hereda el de su ``based_on``):

    - perfil sin bróker (p. ej. Apex): "El perfil '<clave>' aún no admite patas .sqx.";
    - bróker distinto: las horas, los precios y la comisión del backtest serían los de otro bróker, y la regla única
      evita errores silenciosos: "Las patas son de <bróker>; el perfil '<clave>' opera en <bróker>. Re-testea las
      estrategias en SQX con los datos de ese bróker."."""
    with _FIRMS_LOCK:
        profile_broker = fichas.broker_for_profile(str(cfg.key))
    if profile_broker is None:
        return f"El perfil '{cfg.key}' aún no admite patas .sqx."
    if profile_broker == pf_broker:
        return None
    brokers = fichas.load_registry().brokers
    label = lambda bid: _broker_label(brokers[bid]) if bid in brokers else bid        # noqa: E731
    return (f"Las patas son de {label(pf_broker)}; el perfil '{cfg.key}' opera en {label(profile_broker)}. "
            "Re-testea las estrategias en SQX con los datos de ese bróker.")


def _sqx_cfd_company_warning(cfg) -> str | None:
    """Enmienda v2 7: las patas ``.sqx`` CFD se dimensionan con las especificaciones, comisiones y
    hora de servidor de FTMO; con un perfil de otra compañía se avisa (no se bloquea)."""
    with _FIRMS_LOCK:
        company = profiles.company_of(str(cfg.key))
    if company is None or company == _FTMO_COMPANY:
        return None
    return (f"Patas .sqx CFD con un perfil de {company}: se dimensionan con las especificaciones, "
            "comisiones y hora de servidor de FTMO (config/ftmo_cfd_costs.yaml); en tu firma "
            "pueden ser otras.")


def _profile_caveats(cfg) -> list[dict]:
    """Supuestos del modelo del perfil como objetos ``{id, titulo, texto, sesgo}``
    (plan onboarding-b1): fuente única ``config/model_notes.yaml``. Los textos de
    Apex y Futures son, literalmente, las cadenas que antes vivían aquí."""
    notes, base_nombre = _profile_notes(cfg)
    return [n.as_catalog(base_nombre) for n in notes]


# ---------------------------------------------------------------------------
# Tamaño de bloque automático (guardrail bloque-politis-white, 2026-10-21; plan §2.2 y enmiendas E1-E19)
# ---------------------------------------------------------------------------
# Con ``block_size <= 0`` las dos lentes eligen el tamaño medio del bloque con el método de Politis-White sobre la serie
# del resultado de cada episodio medido en R (riesgos): no depende de la f ni de la rejilla. Antes era la ACF de los
# cuadrados del PnL en $ a UNA f (``calibrate_block_size``, que sigue en el motor para los batch de ``sqx/``).
_BS_MIN_EPISODES = BS_MIN_EPISODES   # menos episodios que estos → bloques sueltos (decisión del plan, declarada en la nota)
_BS_MAX_EXCLUDED_PCT = 10            # más de este % de las operaciones sin riesgo por lote válido → bloques sueltos (idem)
_BS1_EPS = 1e-9                      # "bloque principal > 1" es ``bs > 1 + 1e-9`` (E12)
_PROPFIRM_SEED = 0                   # CRN base de la lente prop firm (``idx_matrix``); la prolongación usa [0, 2]
_AXI_SEED = 42                       # semilla de la matriz de índices de la lente AXI (la de ``run_portfolio_mc``)
_BS_REASONS = {
    "auto": "tamaño medio elegido con el método de Politis-White sobre el resultado de cada bloque medido en riesgos (R): "
            "sigue la dependencia real entre bloques y no depende de la f ni de la rejilla",
    "manual": "manual: lo fijaste tú",
    "fallback_iid": "no se pudo medir la dependencia entre bloques (menos de 25 bloques, o más del 10 % de las operaciones "
                    "sin riesgo por lote válido): bloques sueltos",
}
# ``meta.model_context.block_size.method`` por modo
_BS_METHODS = {"auto": "politis_white", "manual": "manual", "fallback_iid": "fallback_iid"}


def _leg_risk_per_lot(leg: dict, pf: dict) -> np.ndarray:
    """Riesgo por lote de CADA operación de una pata (USD por lote hasta el SL), alineado con sus trades.

    * Pata ``.sqx``: ``fleg.risk_per_micro`` (``|apertura − SL| · pv``), el array que ya usa ``fixed_risk_micros``.
    * Pata CSV con sizing exacto: ``sl_exact`` (el ``slInMoney`` por operación del ATR real).
    * Pata CSV en modo rápido (sin ``sl_exact``): ``tl.sl_in_money(f_ref)`` con el ``f_ref`` de la cartera, la misma fuente
      que ``_leg_lots``."""
    if _leg_kind(leg) == KIND_SQX:
        return np.asarray(leg["fleg"].risk_per_micro, dtype=float)
    sl = leg.get("sl_exact")
    if sl is not None:
        return np.asarray(sl, dtype=float)
    return np.asarray(leg["tl"].sl_in_money(pf["f_ref"]), dtype=float)


def _episode_r_series(blocks, legs_ord: list[dict], pf: dict) -> tuple[np.ndarray, int]:
    """Serie del resultado de cada episodio flat-flat medido en R, y nº de operaciones excluidas.

    ``R_episodio = Σ_{operaciones j del episodio} pnl_por_lote_j / riesgo_por_lote_j`` (la fórmula de la serie R del A/B
    del tamaño de bloque). Es adimensional y NO depende de la f ni de la rejilla: ni el PnL por lote ni el riesgo por lote
    (distancia al SL en $) dependen del lote que dimensione la f.

    * ``blocks``: EXACTAMENTE los bloques que remuestrea la lente, ya con el corte ``oos_start`` aplicado (``PortfolioDayBlock``
      en prop firm, ``BlockRecord`` en AXI: los dos llevan ``sym_id``, ``trade_local_id`` y ``pnl_per_lot``). La serie sale
      en el mismo orden y con el mismo largo (E1: ``len(serie) == len(blocks)``).
    * ``legs_ord``: las patas en el orden de ``strats`` (``_legs_in_strats_order``), el de ``sym_id``.
    * Riesgo por lote: :func:`_leg_risk_per_lot`. En una cartera CSV con TODAS las patas en modo rápido el riesgo es
      ``f_ref · equity_before / Size``: lineal en ``f_ref`` con el mismo factor para todas las operaciones, así que la serie
      se escala en 1/``f_ref`` y el b* de Politis-White (invariante de escala) no cambia (E5). Con patas exactas y rápidas
      mezcladas, el peso relativo de las rápidas sí depende de ``f_ref``.
    * Exclusiones (E8): una operación con riesgo ≤ 0 o no finito, o con un cociente no finito, NO suma a su episodio y se
      cuenta. Los episodios nunca se descartan.

    Suma vectorizada por episodio (``np.add.at``, en el orden de las operaciones del bloque: el mismo orden de suma que un
    bucle plano). Devuelve ``(serie (n_bloques,), n_operaciones_excluidas)``."""
    n_blocks = len(blocks)
    if n_blocks == 0:
        return np.zeros(0), 0
    sizes = np.array([len(b.sym_id) for b in blocks], dtype=np.int64)
    ep = np.repeat(np.arange(n_blocks), sizes)
    sym = np.concatenate([np.asarray(b.sym_id, dtype=np.int64) for b in blocks])
    tid = np.concatenate([np.asarray(b.trade_local_id, dtype=np.int64) for b in blocks])
    ppl = np.concatenate([np.asarray(b.pnl_per_lot, dtype=float) for b in blocks])
    if len(sym) and int(sym.max()) >= len(legs_ord):
        raise ValueError(f"sym_id {int(sym.max())} fuera de las {len(legs_ord)} patas: serie R desalineada")
    risk = np.full(len(ep), np.nan)
    for si, leg in enumerate(legs_ord):
        sel = sym == si
        if sel.any():
            arr, t = _leg_risk_per_lot(leg, pf), tid[sel]
            # cota del índice de operación (revisión de código de la fase A, H1): un trade_local_id fuera del array de
            # riesgo de su pata daría un IndexError sin contexto o, con índices negativos, el riesgo de otra operación
            if int(t.min()) < 0 or int(t.max()) >= arr.size:
                raise ValueError(f"trade_local_id fuera del riesgo de la pata {si} ({arr.size} operaciones): serie R "
                                 "desalineada")
            risk[sel] = arr[t]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = ppl / risk
        ok = np.isfinite(risk) & (risk > 0.0) & np.isfinite(term)
    r = np.zeros(n_blocks)
    np.add.at(r, ep[ok], term[ok])
    return r, int(np.count_nonzero(~ok))


def _auto_block_size(blocks, legs_ord: list[dict], pf: dict) -> tuple[float, str, dict, list[str]]:
    """Bloque automático de las dos lentes (``block_size <= 0``): ``(bs, modo, info, avisos)``.

    * Menos de 25 episodios, o más del 10 % de las operaciones de ``blocks`` sin riesgo por lote válido →
      ``bs = 1``, modo ``fallback_iid`` y un aviso que da el motivo (los dos umbrales son decisiones del plan, declaradas
      en la nota ``tamano_bloque``).
    * Si no → ``bs = politis_white_block(serie R)``, modo ``auto`` (``info`` lleva ``opt_m`` y ``m`` del estimador, E10).

    ``info`` = ``{series, n_excluded_terms, opt_m, m}`` (eco de ``meta.model_context.block_size``)."""
    n_blocks = len(blocks)
    r, n_excl = _episode_r_series(blocks, legs_ord, pf)
    assert len(r) == n_blocks, "la serie R debe tener un valor por episodio que remuestrea la lente (E1)"
    n_ops = int(sum(len(b.sym_id) for b in blocks))
    info = {"series": "R", "n_excluded_terms": n_excl, "opt_m": None, "m": None}
    if n_blocks < _BS_MIN_EPISODES:
        return 1.0, "fallback_iid", info, [
            f"Solo {n_blocks} bloques (<{_BS_MIN_EPISODES}): no se puede medir la dependencia entre bloques; "
            "block_size = 1 (bloques sueltos)."]
    if 100 * n_excl > _BS_MAX_EXCLUDED_PCT * n_ops:       # enteros: sin redondeos en el umbral del 10 %
        return 1.0, "fallback_iid", info, [
            f"{n_excl} de {n_ops} operaciones ({n_excl / n_ops:.1%}) sin riesgo por lote válido (más del "
            f"{_BS_MAX_EXCLUDED_PCT} %): no se puede medir el resultado de los bloques en riesgos; "
            "block_size = 1 (bloques sueltos)."]
    pw = politis_white_info(r)
    info["opt_m"], info["m"] = pw.opt_m, pw.m
    return pw.b, "auto", info, []


# ---------------------------------------------------------------------------
# Contexto de modelo por corrida (plan onboarding-b1 §2.3)
# ---------------------------------------------------------------------------
def _bs_explain(mode: str) -> tuple[str, str, bool | None]:
    """Explica en español por qué el tamaño de bloque es el que es.

    Devuelve ``(mode, reason, has_real_clustering)``. Decide SOLO por el modo (enmienda E2): ``auto`` (Politis-White sobre
    la serie en R), ``manual`` o ``fallback_iid``. El modo automático nunca se etiqueta ``fallback_iid``. Con Politis-White
    ya no se calcula la agrupación de volatilidad, así que ``has_real_clustering`` es siempre ``None``."""
    if mode not in _BS_REASONS:
        raise ValueError(f"modo de bloque desconocido: {mode!r}")
    return mode, _BS_REASONS[mode], None


def _data_span_days(blocks) -> int:
    """Días naturales entre la primera apertura y el último cierre de los bloques
    que usa la simulación (con corte IS/OOS, solo los de la ventana usada)."""
    start = min(pd.Timestamp(b.start) for b in blocks)
    end = max(pd.Timestamp(b.end) for b in blocks)
    return int(round((end - start).total_seconds() / 86400.0))


def _pct_label(x: float) -> str:
    return f"{100.0 * x:.0f} %"


# PLAN-DESVIO (menor): el plan no dice qué pasa con los argumentos ``cal`` y ``f_cal`` de ``_model_context``. Se conservan
# (opcionales, ignorados) porque ``poc_web/api/test_profiles_templates.py::_ctx_notes`` (fuera de la lista de esta fase)
# llama con ``cal=None, f_cal=0.01``.
def _model_context(*, lente: str, blocks, n_blocks: int, bs: float,
                   bs_mode: str, bs_info: dict | None = None, cal=None, f_cal: float | None = None,
                   cfg=None, extra: dict | None = None) -> dict:
    """``meta.model_context`` (aditivo): tamaño de bloque y su razón, días de
    backtest, horizonte de la fondeada (solo prop firm) y notas por corrida.

    ``extra`` aporta placeholders específicos de la lente (``pct_cortadas_datos``,
    ``min_dias``, ``min_trades``). Solo entran en ``notes`` las notas con
    ``en_esta_corrida`` completamente rellenable (render estricto).

    Bloque (guardrail bloque-politis-white): ``block_size`` lleva ``value`` (float), ``mode`` (``auto`` | ``manual`` |
    ``fallback_iid``), ``method`` (``politis_white`` | ``manual`` | ``fallback_iid``), ``reason``, ``n_blocks`` y, de
    ``bs_info`` (la salida de :func:`_auto_block_size`; ``None`` en modo manual): ``series`` (``"R"``),
    ``n_excluded_terms`` y los parámetros del estimador ``opt_m`` y ``m`` (E10). ``has_real_clustering`` y
    ``calibrated_at_f`` ya no se calculan (la serie en R no tiene f): salen ``null``. ``cal`` y ``f_cal`` se aceptan por
    compatibilidad de firma y se ignoran. En el texto por corrida ``bs`` va con un decimal (E12)."""
    mode, reason, clustering = _bs_explain(bs_mode)
    info = bs_info or {}
    span = _data_span_days(blocks)
    ctx = {
        "block_size": {
            "value": float(bs), "mode": mode, "reason": reason,
            "has_real_clustering": clustering, "n_blocks": n_blocks,
            "calibrated_at_f": None,
            "method": _BS_METHODS[mode], "series": info.get("series"),
            "n_excluded_terms": info.get("n_excluded_terms"),
            "opt_m": info.get("opt_m"), "m": info.get("m"),
        },
        "data_span_days": span,
    }
    fill = {
        "n_blocks": n_blocks, "bs": f"{bs:.1f}", "bs_modo": mode,
        "bs_razon": reason, "span_dias": span,
    }
    if lente == "propfirm":
        if cfg.funded.model == "cycle":
            cap = cfg.funded.payout_cycle_days * cfg.funded.max_payout_cycles
        elif cfg.funded.model == "futures" and cfg.subscription is not None:
            cap = cfg.subscription.funded_cap_days
        else:
            cap = None
        ctx["funded_horizon"] = {"cap_days": cap, "window_blocks": n_blocks}
        fill["cap_dias"] = cap
        notes, _base = _profile_notes(cfg)
    else:
        notes = notes_for("axi")
    fill.update(extra or {})
    rendered = []
    for n in notes:
        txt = render_run(n, fill)
        if txt is not None:
            rendered.append({"id": n.id, "titulo": n.titulo, "texto": txt,
                             "sesgo": n.sesgo})
    ctx["notes"] = rendered
    return ctx


# ---------------------------------------------------------------------------
# Sensibilidad con bloque 1 (guardrail bloque-politis-white; plan §2.2.4-5 y enmiendas E4, E15, E16, E18, E19)
# ---------------------------------------------------------------------------
# Con un bloque principal > 1 (automático o manual) se repite la simulación SOLO en f* con bloque 1: mismo número de caminos,
# generadores NUEVOS con la semilla de la lente y TODAS las matrices que dependen del bloque reconstruidas, de modo que es
# reproducible: misma semilla ⇒ mismos resultados que una petición manual con bloque 1. NO consume el generador de la
# petición principal y no cambia f*, la recomendación ni la receta (E19). ``meta.bs1`` = lista con UNA fila ``{f, KPIs}``
# (E16: S = {f*}); ``null`` si el bloque principal es ≤ 1 + 1e-9 o el modo es ``fallback_iid`` (la cifra sería la misma).
def _bs1_wanted(bs: float) -> bool:
    """¿Hay que calcular la sensibilidad con bloque 1? Solo si el bloque principal es > 1 (E12: ``bs > 1 + 1e-9``)."""
    return bs > 1.0 + _BS1_EPS


# PLAN-DESVIO (menor): el texto del plan termina en "por encima de tu tolerancia"; aquí se añade el valor de la tolerancia entre
# paréntesis para que el aviso se entienda sin mirar el formulario.
def _bs1_warning(f_star: float, kpi: str, value: float, tol: float) -> str:
    """Aviso (solo en ``warnings``) de que con bloque 1 se supera la tolerancia de la recomendación en f*."""
    return (f"Con bloque 1 (sin la dependencia entre bloques de tu backtest), en f* = {100.0 * f_star:.2f} % la {kpi} "
            f"sería {100.0 * value:.1f} %, por encima de tu tolerancia ({100.0 * tol:g} %).")


def _axi_qual_report(outs, f: float, bs: float, n_paths: int, horizon_days):
    """``RiskReport`` F4 de los outcomes de UNA f (post-continuación: el que da ``p_pass``/``p_ruin`` y alimenta
    ``recommend_f``)."""
    qual = [PathOutcome(o.outcome, o.days, o.trades, o.max_dd, o.terminal) for o in outs]
    return aggregate_outcomes(qual, float(f), "block", bs, n_paths, horizon_days)


def _axi_bs1(outs_at, f_grid, f_star: float, n_blocks: int, req) -> tuple[list[dict], list[str]]:
    """Sensibilidad con bloque 1 de la lente AXI en f*: ``([{f, p_pass, p_ruin}], avisos)``.

    ``outs_at(fi, idx_matrix)`` evalúa los caminos de la f ``fi`` de la rejilla con una matriz de índices (es la MISMA
    función que evalúa la rejilla de la petición). La matriz del bloque 1 sale de un ``default_rng(42)`` NUEVO (la semilla
    de la lente). Aviso si ``p_ruin > tol_ruin`` (el criterio con que ``recommend_f`` llama factible a una f)."""
    fi = int(np.argmin(np.abs(np.asarray(f_grid, dtype=float) - f_star)))
    f = float(f_grid[fi])
    idx1 = stationary_block_indices(n_blocks, req.n_paths, 1.0, np.random.default_rng(_AXI_SEED))
    rep = _axi_qual_report(outs_at(fi, idx1), f, 1.0, req.n_paths, req.horizon_days)
    warns = []
    if rep.p_ruin > req.tol_ruin:
        warns.append(_bs1_warning(f, "P(ruina)", rep.p_ruin, req.tol_ruin))
    return [{"f": f, "p_pass": rep.p_pass, "p_ruin": rep.p_ruin}], warns


def apply_trade_floor(outcomes: list[PathOutcome],
                      min_unique_trades: int | None) -> list[PathOutcome]:
    """Post-filtro del suelo de trades (§3.4, F2b) — función pura.

    CONGELADA como artefacto histórico (§10.4-2 de ANALISIS_CASOS_PASS_F4):
    F4 (``evaluate_path_axi``) la subsume resolviendo la continuación real del
    camino; ya NO está en el flujo del endpoint. Se conserva con sus tests
    (``test_floor_filter.py``) como referencia del comportamiento F2b.

    Un outcome ``pass`` que llegó al target con ``trades < min_unique_trades``
    NO ha pasado la etapa: debe seguir operando, así que se reclasifica como
    "sin resolver" (``timeout``, el outcome sin resolver del portfolio). Con
    ``min_unique_trades is None`` (etapa sin suelo) es un no-op: devuelve la
    entrada intacta (mismo objeto, sin copiar).
    """
    if min_unique_trades is None:
        return outcomes
    return [
        dataclasses.replace(o, outcome="timeout")
        if o.outcome == "pass" and o.trades < min_unique_trades else o
        for o in outcomes
    ]


# ---------------------------------------------------------------------------
# Modelos de la API
# ---------------------------------------------------------------------------
class PortfolioLoadRequest(BaseModel):
    legs: list[str] = Field(min_length=1)
    account_size: float = Field(gt=0)
    f_ref: float = Field(default=0.005, gt=0.0)
    # Fase 6 (Apex): si se indica la firm y su perfil es de sesiones ET, la
    # carga añade la guarda de compliance overnight (flat a 16:59 ET) y el
    # aviso de universo de datos. "" = carga agnóstica de perfil (histórico).
    firm_key: str = ""
    source_tz: str | None = None  # tz de los timestamps; default documentado


class PropfirmSimRequest(BaseModel):
    portfolio_id: str
    firm_key: str = ""
    f_lo: float = 0.0025
    f_hi: float = 0.0175
    f_step: float = 0.0025
    n_paths: int = Field(default=800, ge=100, le=5000)
    n_challenges: int = Field(default=20, ge=1, le=50)
    n_sequences: int = Field(default=1000, ge=100, le=10000)
    capital0_eur: float = 3000.0
    fx_eur_usd: float = 1.08
    tol_neg: float = Field(default=0.5, gt=0.0, lt=1.0)
    block_size: float = 0.0            # 0 = automático (Politis-White sobre la serie en R); > 0 = manual
    stop_on_insolvency: bool = False
    oos_start: str | None = None       # F2c-1: corte IS/OOS; null = ventana completa
    # F2c-2: eje de días. Default "servidor" = camino histórico sin shift (el
    # baseline de paridad queda intacto); la UI manda "ftmo" explícito.
    # Ignorado con perfiles day_axis="et_session" (el eje lo fija el labeler).
    day_axis: Literal["ftmo", "servidor"] = "servidor"
    # Fase 6 (Apex): tz de los timestamps para el etiquetado de sesiones ET
    # (null → DEFAULT_ET_SOURCE_TZ con warning); copy trading y umbral del
    # diagnóstico de caps de contratos (Fase 5 del plan).
    source_tz: str | None = None
    n_copied_accounts: int = Field(default=1, ge=1, le=20)
    cap_violation_tol: float = Field(default=0.0, ge=0.0, le=1.0)
    # Fase 2 (FTMO Futures): políticas de la campaña de suscripción. Solo
    # aplican con funded.model == "futures"; con otro perfil y valor no-default
    # se ignoran + warning (con defaults: silencio total, regresión estricta).
    reset_policy: Literal["billing_retry", "paid_reset"] = "billing_retry"
    withdrawal_policy: Literal["greedy", "buffer_preserving"] = "greedy"
    # Carteras .sqx CFD (2026-10-14): "auto" = flotante por MINUTO (OHLC M1) si todas
    # las patas tienen OHLC coherente, si no ventanas con warning; "windows" = cota por
    # ventanas (camino anterior, bit a bit). Ignorado en carteras CSV y de futuros.
    sqx_float: Literal["auto", "windows"] = "auto"


class AxiSimRequest(BaseModel):
    portfolio_id: str
    stage: str = "seed"
    f_lo: float = 0.005
    f_hi: float = 0.03
    f_step: float = 0.0025
    n_paths: int = Field(default=2000, ge=100, le=20000)
    tol_ruin: float = Field(default=0.10, gt=0.0, lt=1.0)
    block_size: float = 0.0            # 0 = automático (Politis-White sobre la serie en R); > 0 = manual
    horizon_days: float | None = None  # control avanzado (§3.4); null = sin horizonte
    oos_start: str | None = None       # F2c-1: corte IS/OOS; null = ventana completa


class LegDiagRequest(BaseModel):
    f: float = Field(gt=0)
    stage: str = "seed"
    n_paths: int = Field(default=1000, ge=100, le=20000)
    block_size: float = Field(default=1.0, gt=0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "n_symbols_broker": len(_broker.symbols),
            "n_portfolios": len(_portfolios),
            "user_profiles_errors": profiles.get_errors(),
            "extra_roots": [{"path": str(r), "is_dir": Path(r).is_dir()}
                            for r in EXTRA_INPUT_ROOTS]}


@app.get("/api/catalog")
def catalog():
    out = _jsonable({
        "firms": [
            {"key": c.key, "display_name": c.display_name,
             "account_size": c.account_size, "fee": c.fee,
             # Fase 6 (Apex): metadatos del perfil para el frontal (selector
             # con moneda correcta + panel "Reglas del perfil" read-only).
             "fee_ccy": c.fee_ccy,
             "ruleset_version": c.ruleset_version or None,
             "day_axis": c.day_axis,
             # onboarding-c: official / based_on / verified_on (None en los oficiales);
             # perfiles-v2: company / hidden (los ocultos NO se excluyen: el front los filtra).
             **pf,
             # Fase 3: funded.model ("cycle" | "ladder" | "futures") para que
             # el frontal ramifique la renderización.
             "funded_model": c.funded.model,
             # onboarding-b1: ``extended`` sustituye a "rules is not None" como
             # criterio de perfil extendido; ``kind`` = cfd | futures.
             "extended": _profile_extended(c),
             "kind": _profile_kind(c),
             "rules": _profile_rules(c),
             "caveats": _profile_caveats(c)}
            for c, pf in _firms_catalog_snapshot()
        ],
        # onboarding-b1: supuestos de la lente AXI (panel "¿Cómo se ha calculado?").
        "model_notes_axi": [n.as_catalog() for n in notes_for("axi")],
        "axi_stages": [
            {"key": p.name, "display_name": p.display_name,
             "profit_target_pct": p.profit_target_pct,
             "min_days": p.min_days,
             "min_unique_trades": p.min_unique_trades,
             "max_loss_pct": p.max_loss_pct,
             # Baseline de la etapa con la convención de la app Streamlit
             # (equity de referencia 5k → allocation_at(5000)).
             "allocation_usd": p.allocation_at(5000.0)}
            for p in sorted(_axi.phases, key=lambda p: p.order)
        ],
        "legs": _scan_legs(),
        # F2c-1: registro de cortes IS/OOS por campaña (config/oos.yaml, M2-8).
        "oos_campaigns": [
            {"key": k, "cut": c.cut.isoformat(), "source": c.source}
            for k, c in load_oos_registry().items()
        ],
    })
    # PLAN-DESVIO: post-paso en catalog() (el plan solo pedía añadir campos). ``_jsonable``
    # convierte bool -> int (bool es subclase de int): ``official`` debe salir como booleano JSON
    # real (el resto de la salida queda tal cual, I2).
    for f in out["firms"]:
        f["official"] = bool(f["official"])
        f["hidden"] = bool(f["hidden"])
    return out


def _sqx_load_guards(loaded: list[dict], req,
                     asset: str | None = SQX_ASSET_FUTURES, *, lens_axi: bool = False) -> dict[str, dict]:
    """Guardas de carga de una cartera de patas ``.sqx`` (V6/W1).

    - ``ventanas``: todos los trades con MAE y SL válidos (finitos, MAE ≥ 0,
      riesgo inicial > 0) — lo garantiza el extractor; aquí se comprueba sobre
      los arrays que consumirá el sizing y las ventanas. Informa de la
      ``backtest_precision`` de cada pata (la cota vale a esa precisión).
    - ``corte_1610`` (V6, regla 16:10 explícita): nº y % de trades cuya
      etiqueta de sesión de apertura ≠ la de cierre (labeler NY con el cut del
      perfil futures si viene ``firm_key``, "16:10" por defecto). Incluye los
      cruces de fin de semana. Son imposibles en vivo (auto-flatten de FTMO):
      ``warn`` si > 1%.

    ``lens_axi`` (fase F3): la cartera es de CFD del bróker de la lente AXI. El MAE y el SL se comprueban igual, pero
    el detalle de ``ventanas`` habla del MAE que acota el valle diario de esa lente (no de ventanas) y ``corte_1610``
    no menciona a FTMO (la lente AXI no tiene cierre de sesión)."""
    out: dict[str, dict] = {}
    n_tot = n_bad = 0
    precs: dict[str, int] = {}
    for leg in loaded:
        fl = leg["fleg"]
        n_tot += len(fl.adverse_per_lot)
        ok = (np.isfinite(fl.adverse_per_lot) & (fl.adverse_per_lot >= 0)
              & np.isfinite(fl.risk_per_micro) & (fl.risk_per_micro > 0))
        n_bad += int((~ok).sum())
        precs[str(fl.backtest_precision)] = precs.get(
            str(fl.backtest_precision), 0) + 1
    prec_txt = ", ".join(f"backtest_precision={k} ({v} pata(s))"
                         for k, v in sorted(precs.items()))
    out["ventanas"] = (
        {"status": "ok", "detail": f"100% de {n_tot} trades con MAE y SL "
         f"válidos; " + ("el MAE acota la serie por minuto y es la cota diaria de la lente AXI, a precisión del "
                         "backtest " if lens_axi else "valle por ventanas = cota a precisión del backtest ")
         + f"({prec_txt})"}
        if not n_bad else
        {"status": "error", "detail": f"{n_bad}/{n_tot} trades sin MAE/SL "
         "válidos: la cota de ventanas no es calculable."})

    if asset == SQX_ASSET_CFD:
        out["corte_1610"] = {
            "status": "ok",
            "detail": ("no aplica: lente AXI (CFD sin cierre de sesión, se puede mantener de noche y en fin de "
                       "semana)" if lens_axi else
                       "no aplica: CFD FTMO Swing (sin cierre de sesión, se puede "
                       "mantener de noche y en fin de semana)")}
        return out
    cfg = _firm_get(req.firm_key) if req.firm_key else None
    cut = (cfg.day_axis_cut_hhmm
           if cfg is not None and cfg.funded.model == "futures"
           else DEFAULT_SQX_CUT)
    labeler = make_session_day_labeler(SQX_WALL_CLOCK_TZ, cut_hhmm=cut)
    n_cross = 0
    for leg in loaded:
        lo = np.asarray(labeler(pd.to_datetime(leg["s"].open_t)))
        lc = np.asarray(labeler(pd.to_datetime(leg["s"].close_t)))
        n_cross += int(np.sum(lo != lc))
    pct = (n_cross / n_tot) if n_tot else 0.0
    out["corte_1610"] = {
        "status": "warn" if pct > 0.01 else "ok",
        "detail": (f"{n_cross}/{n_tot} trades ({pct:.2%}) abren y cierran en "
                   f"sesiones distintas (corte {cut} ET; incluye fines de "
                   "semana): imposibles en vivo por el auto-flatten de FTMO"
                   + (" — > 1%: revisa el backtest." if pct > 0.01 else "."))}
    return out


def _fichas_guard(loaded: list[dict]) -> dict:
    """Guarda ``fichas`` de la carga (F4; SOLO carteras ``.sqx``): ``ok`` si ninguna de las fichas de las patas cargadas
    tiene campos sin verificar (``needs_review``: los que propuso el ``.sqx`` o la comisión 0 de la hoja de MT5); si
    alguna los tiene, ``warn`` con la ficha, su pata y los campos. NUNCA ``error``: una ficha por verificar no impide
    simular, solo dice que su dimensionado (valor del punto, pasos de lote, comisión, spread) descansa en un dato sin
    confirmar. Las patas del mismo símbolo comparten ficha y salen juntas."""
    pending: dict[str, tuple[list[str], list[str]]] = {}
    n_symbols = set()
    for leg in loaded:
        fl = leg.get("ficha") or {}
        sym = str(fl.get("sqx_symbol") or (leg.get("meta") or {}).get("symbol"))
        n_symbols.add(sym)
        fields = [str(f) for f in (fl.get("needs_review") or [])]
        if fields:
            pending.setdefault(sym, (fields, []))[1].append(str(leg["name"]))
    if not pending:
        return {"status": "ok",
                "detail": f"las fichas de {len(n_symbols)} símbolo(s) no tienen campos sin verificar"}
    parts = []
    for sym, (fields, names) in pending.items():
        shown = ", ".join(names[:3]) + (f" y {len(names) - 3} más" if len(names) > 3 else "")
        parts.append(f"ficha de {sym} con campos sin verificar: {', '.join(fields)} "
                     f"(pata{'s' if len(names) > 1 else ''}: {shown})")
    return {"status": "warn",
            "detail": "; ".join(parts) + ". Verifícalos en la ficha (pestaña Símbolos): el dimensionado de la cartera "
                                         "descansa en ellos."}


@app.post("/api/portfolio/load")
def portfolio_load(req: PortfolioLoadRequest):
    """Carga y valida las patas UNA vez; guardas con semáforo ok/warn/error.

    Carteras .sqx CFD (2026-10-14): las patas se cargan con ``with_minutes=True`` y la guarda
    nueva ``flotante_m1`` (solo en estas carteras) avisa (``warn``, nunca error) de las patas
    sin OHLC M1 coherente: la simulación usará la cota por ventanas.

    Cartera ``.sqx`` del bróker de la lente AXI (fase F3 del guardrail entrada-4-bloques): las patas se cargan con
    ``synth_mode="daily"`` (la misma entrada de caché que recarga ``simulate_axi``), ``flotante_m1`` avisa de las patas que
    usarán la cota diaria por el MAE y las guardas ``tipo_patas``, ``valles``, ``ventanas`` y ``corte_1610`` hablan de la
    lente AXI (no de FTMO ni de ventanas).

    Cartera ``.sqx`` (fase F4): las filas de las patas cargadas añaden ``broker`` (id), ``ficha_sin_verificar`` (los campos
    ``needs_review`` de su ficha, posiblemente ``[]``), ``mm_backtest`` (el método del ``lastSettings.xml`` o ``null``) y
    ``paridad_frac_ok`` (la fracción de órdenes cuyo lote reproduce la regla de riesgo fijo, o ``null``), y la guarda
    ``fichas`` (:func:`_fichas_guard`) avisa (``warn``, nunca ``error``) de las fichas con campos sin verificar. Las filas y
    guardas de las carteras CSV no cambian."""
    symbols = {s.ticker_axi: s for s in _broker.symbols}
    sqx_map = _sqx_match_paths()
    legs_out: list[dict] = []
    loaded: list[dict] = []
    warnings: list[str] = []

    # Tipo de cartera (enmienda V7): .sqx de futuros vs CSV. NO se mezclan en
    # v1 (cartera mixta → error explícito, ninguna pata se carga).
    kinds = {KIND_SQX if _is_sqx_leg_path(rel) else KIND_CSV for rel in req.legs}
    mixed = len(kinds) > 1
    pf_kind = KIND_SQX if kinds == {KIND_SQX} else KIND_CSV
    is_sqx = pf_kind == KIND_SQX
    # Bróker de las patas .sqx (guardrail entrada-4-bloques, F2): TODAS deben ser del mismo bróker (la clase de
    # activo, el reloj y las especificaciones salen de él). Un .sqx ilegible o sin ficha no cuenta aquí: su error
    # sale en la carga de la pata (revisión DeepSeek H9).
    # (la vía CSV no toca el registro de fichas: invariante I1)
    freg = fichas.load_registry() if is_sqx else None
    sqx_sheets = ([fichas.resolve(s) if s else None for s in (_sqx_symbol(ROOT / rel) for rel in req.legs)]
                  if is_sqx else [])
    sqx_broker_ids = sorted({sh.broker for sh in sqx_sheets if sh is not None})
    sqx_broker = freg.brokers[sqx_broker_ids[0]] if len(sqx_broker_ids) == 1 else None
    sqx_asset = sqx_broker.asset_class if sqx_broker is not None else None
    # F3: una cartera .sqx del bróker de la lente AXI solo se simula con la lente AXI (un perfil de prop firm da 409 por el
    # bróker): sus patas se leen con la cota diaria por el MAE (``synth_mode="daily"``) y sus guardas hablan de esa lente.
    # PLAN-DESVIO (menor): el brief pide "guardas coherentes" y detalla ``flotante_m1``; se interpreta que también
    # ``tipo_patas``, ``valles``, ``ventanas`` y ``corte_1610`` dejan de hablar de FTMO y de ventanas en esta cartera.
    axi_sqx = (is_sqx and not mixed and sqx_broker is not None and sqx_broker.id == fichas.lens_broker("axi")
               and sqx_asset == SQX_ASSET_CFD)
    guards: dict[str, dict] = {}
    if is_sqx and len(sqx_broker_ids) > 1:
        mixed = True
        detail = (f"patas de brókers distintos ({', '.join(sqx_broker_ids)}): carga una cartera de un solo bróker")
        if len({freg.brokers[b].asset_class for b in sqx_broker_ids}) > 1:
            # PLAN-DESVIO: se conserva la frase de la guarda de hoy ("futuros y de CFD") porque
            # ``test_cartera_mixta_futuros_y_cfd_es_error`` (test_cfd_sqx_legs.py, sin editar sus aserciones, AC2.7)
            # la exige en el detalle de la mezcla de futuros con CFD.
            detail += (". Cartera mixta de patas .sqx de futuros y de CFD: no soportada "
                       "(reglas y costes distintos).")
        guards["tipo_patas"] = {"status": "error", "detail": detail}
    elif mixed:
        guards["tipo_patas"] = {
            "status": "error",
            "detail": "cartera mixta CSV + .sqx: no soportada en v1 "
                      "(las patas .sqx usan sizing Opción A y valle por "
                      "ventanas; las CSV, sizing por MM y valle OHLC). Carga "
                      "una cartera de un solo tipo."}
    elif axi_sqx:
        guards["tipo_patas"] = {
            "status": "ok",
            "detail": f"patas .sqx nativas de CFD del bróker {sqx_broker.display_name}: lente AXI con el valle diario de "
                      "la serie por minuto (o su cota diaria por el MAE), sizing Opción A con los pasos de lote de su "
                      "ficha"}
    elif is_sqx and sqx_asset == SQX_ASSET_CFD:
        guards["tipo_patas"] = {
            "status": "ok",
            "detail": ("patas .sqx nativas de CFD FTMO (valle por ventanas con "
                       "swap, sizing Opción A con lotes de 0,01)" if sqx_broker.id == "ftmo_cfd" else
                       f"patas .sqx nativas de CFD del bróker {sqx_broker.display_name}: valle por ventanas con "
                       "swap, sizing Opción A con los pasos de lote de su ficha")}
    elif is_sqx:
        guards["tipo_patas"] = {
            "status": "ok",
            "detail": ("patas .sqx nativas de futuros (valle por ventanas, "
                       "sizing Opción A)" if sqx_broker is None or sqx_broker.id == "ftmo_futures" else
                       f"patas .sqx nativas de futuros del bróker {sqx_broker.display_name}: valle por "
                       "ventanas, sizing Opción A")}

    if mixed:
        for rel in req.legs:
            legs_out.append({"pata": Path(rel).name, "status": "error",
                             "error": guards["tipo_patas"]["detail"],
                             "warnings": []})

    # --- Carga por pata (guardas por pata) ----------------------------------
    for rel in ([] if mixed else req.legs):
        name = Path(rel).name
        try:
            # Patas .sqx CFD: se pide también la serie por minuto (guarda flotante_m1;
            # 2026-10-14). En CSV y futuros .sqx el flag se ignora/vale False.
            leg = _load_leg(rel, req.account_size, req.f_ref, sqx_map=sqx_map,
                            with_minutes=(is_sqx and sqx_asset == SQX_ASSET_CFD),
                            synth_mode="daily" if axi_sqx else "point")
        except (ValueError, FileNotFoundError) as exc:
            legs_out.append({"pata": name, "status": "error", "error": str(exc),
                             "warnings": []})
            continue
        leg_warns: list[str] = []
        if leg["warn"]:
            leg_warns.append(leg["warn"])
            warnings.append(f"{name}: {leg['warn']}")
        for w in leg["exact_warns"]:
            leg_warns.append(w)
            warnings.append(f"{name}: {w}")
        if _leg_kind(leg) == KIND_SQX:
            # Sin guarda minute_valle ni "sin .sqx": la pata ES el .sqx.
            legs_out.append({"pata": name, "símbolo": leg["sym_csv"],
                             "modo": leg["sizing_mode"],
                             "trades": _leg_n_trades(leg), "status": "ok",
                             "warnings": leg_warns, "sqx": leg["sqx_name"],
                             "backtest_precision":
                                 leg["meta"].get("backtest_precision"),
                             # F4 (solo filas de patas .sqx; las CSV no cambian): bróker de la ficha, campos de la ficha
                             # sin verificar (lista, posiblemente vacía), método del MM del backtest (``null`` si el
                             # .sqx no trae ``lastSettings.xml`` o no tiene método activo) y fracción de órdenes cuyo
                             # lote reproduce la regla de riesgo fijo (``null`` si el MM no es ``FixedAmount``).
                             "broker": leg["broker"],
                             "ficha_sin_verificar": list(leg["ficha"]["needs_review"]),
                             "mm_backtest": (leg["mm_backtest"] or {}).get("method"),
                             "paridad_frac_ok": (leg["paridad"] or {}).get("frac_ok")})
            loaded.append(leg)
            continue
        if leg["sqx_name"] is None:
            leg_warns.append("sin .sqx/.xml en inputs/strategies: sizing rápido.")
        if not leg["s"].minute_valle:
            # None (sin M1) o {} (M1 sin trades con cobertura, §8.5): el joint
            # produciría flotante 0 silencioso.
            legs_out.append({"pata": name, "símbolo": leg["sym_csv"],
                             "modo": leg["sizing_mode"],
                             "trades": leg["tl"].n_trades, "status": "error",
                             "error": "sin serie M1 usable para el flotante intradía "
                                     "(sin M1 o sin trades con cobertura M1).",
                             "warnings": leg_warns})
            continue
        legs_out.append({"pata": name, "símbolo": leg["sym_csv"],
                         "modo": leg["sizing_mode"],
                         "trades": leg["tl"].n_trades, "status": "ok",
                         "warnings": leg_warns, "sqx": leg["sqx_name"]})
        loaded.append(leg)

    # Warning a nivel cartera (§8.4-Q4): precisión heterogénea si alguna pata
    # cayó a rápido.
    n_rapido = sum(1 for l in loaded if l["sizing_mode"] == "rápido")
    if loaded and n_rapido:
        warnings.append(
            f"sizing heterogéneo: {n_rapido} de {len(loaded)} pata(s) en "
            f"rápido (sin .sqx o modo exacto fallido) — precisión no uniforme "
            f"dentro de la simulación.")

    blocks = []
    n_window = n_in_blocks = n_no_valley = 0
    pnl_w_nv = float("nan")
    coverage = float("nan")

    if loaded:
        strats = {leg["name"]: leg["s"] for leg in loaded}
        oc = {k: (s.open_t, s.close_t) for k, s in strats.items()}
        n_cross = count_crossing(strats, detect_flatflat_blocks(oc))
        guards["crossing"] = (
            {"status": "ok", "detail": "count_crossing = 0"} if not n_cross else
            {"status": "warn", "detail": f"count_crossing = {n_cross} (debe ser 0): "
             "trades que no caben en la estructura flat-flat."})

        blocks = build_day_blocks_portfolio(strats, oos_start=None)
        n_blocks = len(blocks)
        if n_blocks < 5:
            guards["bloques"] = {"status": "error",
                                 "detail": f"Solo {n_blocks} bloques flat-flat: "
                                 "backtest demasiado corto o denso."}
        elif n_blocks < 30:
            guards["bloques"] = {"status": "warn",
                                 "detail": f"n_blocks = {n_blocks} < 30: bootstrap "
                                 "degenerado (M2-11)."}
        else:
            guards["bloques"] = {"status": "ok", "detail": f"{n_blocks} bloques flat-flat"}

        n_window = int(sum(s.n_trades for s in strats.values()))
        n_in_blocks = int(sum(b.n_trades for b in blocks))
        coverage = (n_in_blocks / n_window) if n_window else float("nan")
        if n_in_blocks < n_window:
            guards["cobertura"] = {"status": "warn",
                                   "detail": f"{n_in_blocks}/{n_window} trades de la "
                                   f"ventana entran en bloques ({coverage:.1%})."}
        else:
            guards["cobertura"] = {"status": "ok",
                                   "detail": f"cobertura {coverage:.1%} "
                                   f"({n_in_blocks}/{n_window})"}

        # n_no_valley + peso de PnL a f_ref (diagnóstico de carga; el de
        # ejecución se recalcula a f central en cada simulación). D1 (csv-d1-d2): los
        # lotes de ``_leg_lots`` van sobre el tamaño de cuenta, sin componer, igual que
        # en la ejecución; a f_ref ya NO son el ``Size`` compuesto del CSV (cambia el peso).
        # Patas .sqx: NO aplica (valle={} por construcción; el flotante sale de
        # las ventanas por trade) → sin cálculo ni warning de n_no_valley.
        n_no_valley = 0 if is_sqx else int(sum(b.n_no_valley for b in blocks))
        if is_sqx:
            guards["valles"] = ({"status": "ok",
                                 "detail": "no aplica: el valle diario de las patas .sqx sale de su serie por minuto "
                                           "(o de su cota diaria por el MAE) al simular con la lente AXI"}
                                if axi_sqx else
                                {"status": "ok",
                                 "detail": "no aplica: patas .sqx usan ventanas"})
            guards.update(_sqx_load_guards(loaded, req, sqx_asset, lens_axi=axi_sqx))
            guards["fichas"] = _fichas_guard(loaded)
        elif n_no_valley:
            lots_ref = {
                i: _leg_lots(leg, np.array([req.f_ref]), symbols, req.f_ref)
                for i, leg in enumerate(loaded)
            }
            w_nv = w_tot = 0.0
            for b in blocks:
                for j in range(b.n_trades):
                    w = abs(lots_ref[b.sym_id[j]][0, b.trade_local_id[j]]
                            * b.pnl_per_lot[j])
                    w_tot += w
                    if not b.valle_precio[j]:
                        w_nv += w
            pnl_w_nv = (w_nv / w_tot) if w_tot else float("nan")
            # D2 (csv-d1-d2, fase 2): esta guarda no conoce el perfil. Con perfil estático (modo "minute") el flotante
            # de esos trades se acota con su MAE desde la apertura (0 si el CSV no trae MAE); con trailing (Apex,
            # futuros) cuenta como 0; en la lente AXI quedan fuera de los bloques (``build_block_records`` solo
            # incluye trades con valle). Revisión de código de la fase 2 (C4, C7), 2026-10-19.
            n_sin_mae = sum(1 for l in loaded if _leg_kind(l) == KIND_CSV and not l["tl"].has_mae)
            sin_mae = (f" (salvo en {n_sin_mae} pata(s) sin columna MAE en el CSV: cuenta como 0)"
                       if n_sin_mae else "")
            guards["valles"] = {"status": "warn",
                                "detail": f"{n_no_valley} trades sin valle OHLC "
                                f"(peso PnL a f_ref {pnl_w_nv:.1%}): en retos estáticos su flotante "
                                f"se acota con su MAE desde la apertura{sin_mae}; con trailing "
                                "(Apex o Futures) cuenta como 0; en la lente AXI esos trades quedan "
                                "fuera de la simulación."}
        else:
            guards["valles"] = {"status": "ok",
                                "detail": "todos los trades en bloques tienen valle OHLC"}
    else:
        guards["bloques"] = {"status": "error",
                             "detail": "ninguna pata válida cargada."}

    # Guardas agregadas por pata (resumen para el semáforo de la cabecera)
    per_leg_warns = [w for l in legs_out for w in l.get("warnings", [])]
    cap_warns = [w for w in per_leg_warns if "capital" in w.lower()
                 or "equity" in w.lower()]
    guards["capital"] = (
        {"status": "warn", "detail": f"{len(cap_warns)} pata(s) con capital de "
         "backtest ≠ account_size (replay sobre la cuenta del escenario)."}
        if cap_warns else
        {"status": "ok", "detail": "capital de backtest = account_size en todas"})
    sqx_missing = [l["pata"] for l in legs_out
                   if l.get("status") == "ok" and not l.get("sqx")]
    sqx_failed = [l["pata"] for l in legs_out
                  if l.get("status") == "ok" and l.get("sqx")
                  and l.get("modo") == "rápido"]
    if is_sqx:
        guards["sqx"] = {"status": "ok",
                         "detail": f"patas .sqx nativas de "
                                   f"{'CFD' if sqx_asset == SQX_ASSET_CFD else 'futuros'} "
                                   "(sizing Opción A; sin emparejado CSV/.sqx)"}
    elif sqx_missing or sqx_failed:
        parts = []
        if sqx_missing:
            parts.append(f"{len(sqx_missing)} sin .sqx: "
                         f"{', '.join(sqx_missing[:3])}"
                         f"{'…' if len(sqx_missing) > 3 else ''}")
        if sqx_failed:
            parts.append(f"{len(sqx_failed)} con modo exacto fallido: "
                         f"{', '.join(sqx_failed[:3])}"
                         f"{'…' if len(sqx_failed) > 3 else ''}")
        guards["sqx"] = {"status": "warn",
                         "detail": f"pata(s) en sizing rápido ({'; '.join(parts)})"}
    else:
        guards["sqx"] = {"status": "ok",
                         "detail": ".sqx encontrado y modo exacto en todas las patas"}
    m1_bad = [l["pata"] for l in legs_out if l["status"] == "error"
              and "minute_valle" in (l.get("error") or "")]
    if is_sqx and sqx_asset == SQX_ASSET_CFD:
        guards["m1"] = {"status": "ok",
                        "detail": "no aplica: patas .sqx CFD (el OHLC M1 se valida en "
                                  "la guarda flotante_m1)"}
        # Guarda nueva (2026-10-14, SOLO carteras .sqx CFD): ¿todas las patas tienen
        # OHLC M1 coherente → flotante por minuto? Nunca es error: sin OHLC coherente la
        # simulación usa la cota por ventanas (warn con pata y motivo).
        # F3 (lente AXI): las patas se cargaron con ``synth_mode="daily"``, así que TODAS tienen serie; las que no tienen
        # OHLC coherente se reconocen por su ``minute_reason`` y la lente AXI usará su cota diaria por el MAE.
        no_min = [f"{l['name']}: {l.get('minute_reason') or 'sin serie por minuto'}"
                  for l in loaded if (l.get("minute_reason") if axi_sqx else l.get("minute_valle") is None)]
        guards["flotante_m1"] = (
            {"status": "warn",
             "detail": f"{len(no_min)}/{len(loaded)} pata(s) sin OHLC M1 coherente "
                       f"({'; '.join(no_min[:3])}{'…' if len(no_min) > 3 else ''}): "
                       + ("la lente AXI usará la cota diaria por el MAE (más pesimista)." if axi_sqx else
                          "la simulación usará la cota por ventanas (más pesimista).")}
            if no_min else
            {"status": "ok",
             "detail": f"OHLC M1 coherente en las {len(loaded)} pata(s): flotante por "
                       "minuto disponible (cota por minuto con el realizado intradía)."})
    elif is_sqx:
        guards["m1"] = {"status": "ok",
                        "detail": "no aplica: patas .sqx usan ventanas"}
    else:
        guards["m1"] = (
            {"status": "error", "detail": f"{len(m1_bad)} pata(s) sin M1 para "
             f"el flotante intradía: {', '.join(m1_bad[:3])}"}
            if m1_bad else
            {"status": "ok", "detail": "M1 disponible: flotante intradía por minuto viable"})

    # Fase 6 (Apex): guarda de compliance para perfiles de sesiones ET —
    # ningún trade puede cruzar el cierre de sesión (16:59 ET; flat obligado).
    # Se ADVIERTE (no se rechaza) porque el universo disponible hoy es FX en
    # hora de servidor AXI y la tz es una convención provisional hasta la
    # Fase 7 (datos de futuros con tz documentada).
    # FTMO Futures (funded.model "futures"): la guarda NO aplica — su corte
    # es 16:10 ET (day_axis_cut_hhmm) y la semántica 16:59/17-18h CME de
    # overnight_violations no lo parametriza a propósito (riesgo 8 del plan).
    # El aviso de universo FX se conserva para todo perfil et_session.
    firm_cfg = _firm_get(req.firm_key) if req.firm_key else None
    if req.firm_key and firm_cfg is None:
        warnings.append(f"firm_key '{req.firm_key}' desconocida: carga sin "
                        "guardas de perfil.")
    if is_sqx and sqx_asset == SQX_ASSET_CFD:
        # Patas .sqx CFD: hora del servidor FTMO; solo perfiles FTMO CFD.
        if firm_cfg is not None and not _cfd_profile_ok(firm_cfg):
            warnings.append(
                f"El perfil {firm_cfg.display_name} no es un perfil FTMO CFD "
                "(pérdida diaria y máxima estáticas): las carteras de patas .sqx "
                "CFD solo se simulan con esos perfiles (la simulación devolverá 409).")
        elif firm_cfg is not None and (w := _sqx_cfd_company_warning(firm_cfg)):
            warnings.append(w)
    elif is_sqx:
        # Patas .sqx: reloj de pared ET, sin aviso de universo FX ni guarda
        # overnight de Apex (la regla 16:10 la cubre la guarda corte_1610).
        if req.source_tz:
            warnings.append(
                "source_tz ignorado: las patas .sqx de futuros llevan reloj de "
                "pared ET (America/New_York).")
        if firm_cfg is not None and firm_cfg.funded.model != "futures":
            warnings.append(
                f"El perfil {firm_cfg.display_name} no es FTMO Futures: las "
                "carteras de patas .sqx solo se simulan con perfiles futures "
                "(la simulación devolverá 409).")
    elif firm_cfg is not None and firm_cfg.day_axis == "et_session":
        tz = req.source_tz or DEFAULT_ET_SOURCE_TZ
        if firm_cfg.funded.model != "futures":
            per_leg = []
            for leg in loaded:
                df = leg["tl"].df
                idx = overnight_violations(
                    list(zip(pd.to_datetime(df["open_time"]),
                             pd.to_datetime(df["close_time"]))),
                    source_tz=tz)
                if len(idx):
                    per_leg.append(f"{leg['name']}: {len(idx)} trade(s)")
            n_viol = sum(int(p.rsplit(": ", 1)[1].split(" ")[0])
                         for p in per_leg)
            guards["overnight"] = (
                {"status": "warn",
                 "detail": f"{n_viol} trade(s) cruzan el cierre de sesión "
                 f"16:59 ET ({'; '.join(per_leg[:3])}"
                 f"{'…' if len(per_leg) > 3 else ''}): Apex exige flat — esos "
                 "trades son imposibles en la cuenta real "
                 f"(tz asumida: {tz})."}
                if per_leg else
                {"status": "ok",
                 "detail": f"sin posiciones overnight (flat 16:59 ET, tz {tz})"})
        warnings.append(
            f"Perfil {firm_cfg.display_name} sobre universo FX (servidor AXI, "
            f"tz {tz}{' por defecto' if not req.source_tz else ''}): los datos "
            "NO son los futuros de la firma — resultados exploratorios hasta "
            "la Fase 7 (datos de futuros con tz documentada).")
    if is_sqx and not mixed and sqx_broker is not None and firm_cfg is not None:
        # Bróker de las patas frente al del perfil (guardrail entrada-4-bloques, F2): si no coinciden, la simulación
        # devolverá 409. Mismo estilo que los avisos de arriba.
        problem = _profile_broker_problem(firm_cfg, sqx_broker.id)
        if problem:
            warnings.append(f"{problem} La simulación devolverá 409.")

    ok = bool(loaded) and all(g["status"] != "error" for g in guards.values()) \
        and not any(l["status"] == "error" for l in legs_out)

    portfolio_id = uuid.uuid4().hex[:12]
    _portfolios[portfolio_id] = {
        "legs": loaded, "account_size": req.account_size, "f_ref": req.f_ref,
        "kind": "mixed" if mixed else pf_kind,
        "sqx_asset": sqx_asset if is_sqx and not mixed else None,
        # bróker de las patas .sqx (id; ``None`` en CSV, en carteras mezcladas o sin ninguna pata con ficha)
        "broker": sqx_broker.id if is_sqx and not mixed and sqx_broker is not None else None,
        "guards": guards, "n_blocks": len(blocks), "coverage": coverage,
        "n_no_valley": n_no_valley, "pnl_w_no_valley": pnl_w_nv,
    }
    return _jsonable({"portfolio_id": portfolio_id, "ok": ok, "legs": legs_out,
                      "guards": guards, "warnings": warnings,
                      "n_blocks": len(blocks), "coverage": coverage,
                      "account_size": req.account_size, "f_ref": req.f_ref})


def _with_profile_snapshot(out: dict, ctx: dict) -> dict:
    """Añade ``meta.profile_snapshot`` (solo perfiles de usuario) DESPUÉS de ``_jsonable``:
    ``_jsonable`` serializa los bool como 0/1 y el snapshot debe conservar los booleanos reales
    para que su ``sha256`` se pueda recalcular desde el spec serializado. El snapshot ya es JSON
    puro (dicts/listas/str/números/bool) y ``_jsonable`` no se toca (I1)."""
    snap = ctx.get("profile_snapshot")
    if snap is not None:
        out["meta"]["profile_snapshot"] = snap
    return out


def _propfirm_index_matrices(n_blocks: int, n_paths: int, bs: float):
    """``(rng, idx_matrix, idx_ext)`` de la lente prop firm para un tamaño de bloque ``bs``.

    ``idx_matrix`` sale de ``default_rng(0)`` (el CRN base; el ``rng`` devuelto queda en el estado posterior a dibujarla, y
    el par 2D sigue con él para los gaps fijos) e ``idx_ext`` prolonga cada fila en ``2·n_blocks`` posiciones con un
    generador PROPIO (``default_rng([0, 2])``) para la F2 y la fondeada (plan ftmo-fases-secuencias). Las DOS dependen de
    ``bs``: la sensibilidad con bloque 1 (:func:`_propfirm_bs1`) vuelve a llamar aquí con ``bs = 1`` y generadores nuevos."""
    rng = np.random.default_rng(_PROPFIRM_SEED)
    idx_matrix = stationary_block_indices(n_blocks, n_paths, bs, rng)
    # Continuación de la secuencia (plan ftmo-fases-secuencias): la F2 y la fondeada
    # son ventanas posteriores de la MISMA fila prolongada en 2n bloques. Generador
    # PROPIO (no consume ``rng``: los gaps_fixed del par 2D y la F1 no cambian).
    idx_ext = extend_stationary_rows(idx_matrix, n_blocks, bs, 2 * n_blocks,
                                     np.random.default_rng([_PROPFIRM_SEED, 2])).astype(np.int32)
    return rng, idx_matrix, idx_ext


def _propfirm_setup(req: PropfirmSimRequest) -> dict:
    """Setup común de los endpoints prop firm (barrido 1D y par 2D F3).

    Sentencias idénticas y en el MISMO orden que el flujo histórico del 1D
    (paridad F1): eje de días → patas → cfg/grilla/corte → bloques →
    day-arrays → CRN base (rng seed 0 → idx_matrix; ``idx_ext`` = prolongación de
    las filas en 2n bloques con ``default_rng([0, 2])``, para la F2 y la fondeada).
    Devuelve el contexto compartido; el ``rng`` sale expuesto para que el par 2D dibuje
    sus gaps fijos del MISMO generador (CRN global, como en Streamlit).

    Carteras .sqx CFD (2026-10-14): con ``req.sqx_float == "auto"`` las patas se recargan
    con ``with_minutes=True`` en el eje pedido y, si TODAS tienen ``minute_valle``,
    ``float_mode = "minute"`` (``ctx["minute_diag"]`` = diagnóstico por pata); si alguna no,
    ``windows`` + warning con pata y motivo. ``sqx_float == "windows"`` = camino anterior.

    Carteras CSV (2026-10-19, csv-d1-d2 fase 2, D2): con un perfil estático (``not _profile_extended(cfg)``) y
    ``CSV_FLOAT_MODE == "minute"``, ``float_mode = "minute"`` con ``minute_by_sym`` de :func:`_csv_minute_series`
    por pata (los trades sin OHLC M1 se acotan con su MAE) y ``ctx["csv_minute"]`` = contadores de esos trades;
    en otro caso (``"joint"``, o un perfil con trailing/extendido) el camino de siempre.

    Tamaño de bloque (guardrail bloque-politis-white): con ``req.block_size <= 0``, :func:`_auto_block_size` sobre los
    ``blocks`` de la ventana (``ctx["bs"]``, ``ctx["bs_mode"]``, ``ctx["bs_info"]``). El par 2D hereda este bloque y no lleva
    sensibilidad.
    """
    pf = _portfolios.get(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, f"portfolio_id '{req.portfolio_id}' desconocido: "
                             "carga la cartera primero (POST /api/portfolio/load).")
    warnings: list[str] = []

    # --- Perfil de firm (Fase 6: resolución ANTES del eje de días) ----------
    # onboarding-c (enmiendas 5 y 7): se resuelve UNA vez, bajo el lock. ``firm_key`` vacío =
    # primer oficial (histórico); una clave NO vacía y desconocida (o borrada) = 404, no un
    # fallback silencioso. La instantánea del perfil de usuario sale del MISMO ``cfg`` simulado.
    with _FIRMS_LOCK:
        if req.firm_key:
            cfg = _firms.get(req.firm_key)
            if cfg is None:
                raise HTTPException(404, f"perfil desconocido: '{req.firm_key}'")
        else:
            cfg = next(iter(_firms.values()))
        profile_snapshot = profiles.snapshot_for(cfg.key)
    try:
        # fee/fee de activación USD→EUR UNA vez aquí (fx del request); con
        # fee_ccy EUR es no-op. Todos los consumidores (ev_vector_all, refund,
        # stop_on_insolvency, intentos garantizados) ven el fee en EUR.
        cfg = resolve_fee_eur(cfg, req.fx_eur_usd)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    flags = _profile_flags(cfg)
    trailing, et_session = flags["trailing"], flags["et_session"]
    # Carteras de patas .sqx de futuros (enmienda V7): solo con un perfil FTMO
    # Futures (el valle por ventanas alimenta el walker eod_trailing).
    is_sqx = _is_sqx_portfolio(pf)
    sqx_cfd = is_sqx and pf.get("sqx_asset") == SQX_ASSET_CFD
    # Compatibilidad bróker ↔ perfil (guardrail entrada-4-bloques, §2.5): ANTES de las comprobaciones de capacidad
    # de abajo, que no cambian. Un perfil sin bróker (Apex) o con otro bróker que el de las patas → 409.
    if is_sqx and pf.get("broker") is not None:
        problem = _profile_broker_problem(cfg, pf["broker"])
        if problem:
            raise HTTPException(409, problem)
    if sqx_cfd and not _cfd_profile_ok(cfg):
        raise HTTPException(
            409, f"Las carteras de patas .sqx de CFD solo se simulan con un "
            f"perfil FTMO CFD (reglas estáticas, sin trailing ni sesiones ET); "
            f"'{cfg.key}' no lo es.")
    if sqx_cfd and (w := _sqx_cfd_company_warning(cfg)):
        warnings.append(w)
    if is_sqx and not sqx_cfd and cfg.funded.model != "futures":
        raise HTTPException(
            409, f"Las carteras de patas .sqx de futuros solo se simulan con "
            f"un perfil FTMO Futures (funded.model == 'futures'); '{cfg.key}' "
            f"es '{cfg.funded.model}'.")
    # Flags separados (DS-H4): ``use_portfolio`` → simulate_firm_paths
    # (portfolio=...); ``trailing_precompute`` → with_peaks, guarda minute_fav
    # y precompute(trailing=). Cartera CSV: ambos = ``trailing`` del perfil
    # (bit-idéntico al histórico). Cartera .sqx: portfolio=True SIN infra
    # trailing por minuto (el rel_low sale de las ventanas).
    use_portfolio = True if is_sqx else trailing
    trailing_precompute = False if is_sqx else trailing

    # --- Eje de días (F2c-2 + Fase 6): recarga de patas al eje elegido ------
    # La cartera vive en eje servidor (eje de carga); para "ftmo" se resuelven
    # las mismas patas vía _load_leg(axis_hours=-1), que cachea por eje → la
    # primera corrida ftmo paga el precomputo y las siguientes son hit.
    # Fase 6 (perfiles et_session, Apex): el eje lo fija el LABELER de
    # sesiones ET (18:00→16:59), no el shift ftmo/servidor — axis_h = 0 y las
    # patas se recargan con day_labeler + picos M1 (with_peaks) si hay
    # trailing. Con trailing sin et_session solo hacen falta los picos.
    labeler = None
    source_tz = None
    if sqx_cfd:
        # Patas .sqx CFD: hora del servidor FTMO, eje de días por desplazamiento
        # (``req.day_axis``) como las patas CSV; sin labeler de sesiones.
        if req.source_tz is not None:
            warnings.append(
                f"source_tz='{req.source_tz}' ignorado: las patas .sqx CFD van en "
                "hora del servidor FTMO y el eje de días lo fija day_axis.")
        if req.day_axis != "ftmo":
            warnings.append(
                f"day_axis='{req.day_axis}': FTMO resetea la pérdida diaria a "
                "medianoche CE(S)T; con patas CFD usa day_axis='ftmo' (con 'servidor' "
                "el día corta a medianoche de la hora del servidor).")
    elif is_sqx:
        # W3/V7: reloj de pared ET → labeler NY con el cut del perfil,
        # construido ANTES del bloque et_session y usado en la carga de patas
        # (close_d) y en build_day_blocks_portfolio (open_ord). req.source_tz
        # se IGNORA; sin avisos de "universo FX / Fase 7".
        source_tz = SQX_WALL_CLOCK_TZ
        labeler = make_session_day_labeler(
            source_tz, cut_hhmm=cfg.day_axis_cut_hhmm)
        if req.source_tz is not None:
            warnings.append(
                f"source_tz='{req.source_tz}' ignorado: las patas .sqx de "
                "futuros llevan reloj de pared ET (America/New_York).")
        if req.day_axis != "servidor":
            warnings.append(
                f"day_axis='{req.day_axis}' ignorado con perfil et_session: "
                "el eje lo fija el day_labeler de sesiones ET.")
    elif et_session:
        source_tz = req.source_tz or DEFAULT_ET_SOURCE_TZ
        labeler = make_session_day_labeler(
            source_tz, cut_hhmm=cfg.day_axis_cut_hhmm)
        if req.source_tz is None:
            warnings.append(
                f"source_tz no especificado: se asume {DEFAULT_ET_SOURCE_TZ} "
                "(hora de servidor AXI, FX NY-close). Los CSV disponibles son "
                "FX — el universo de datos NO es el de futuros de la firma "
                "(Fase 7); resultados exploratorios.")
        else:
            warnings.append(
                "Los CSV disponibles son FX (servidor AXI) — el universo de "
                "datos NO es el de futuros de la firma (Fase 7); resultados "
                "exploratorios.")
        if req.day_axis != "servidor":
            warnings.append(
                f"day_axis='{req.day_axis}' ignorado con perfil et_session: "
                "el eje lo fija el day_labeler de sesiones ET.")
    axis_h = 0.0 if et_session else DAY_AXIS_HOURS[req.day_axis]
    # Flotante por minuto de las patas .sqx CFD (2026-10-14): se recargan con
    # ``with_minutes=True`` en el eje pedido (también con eje servidor, donde la
    # recarga es un hit de la caché de la carga).
    sqx_minute = sqx_cfd and req.sqx_float == "auto"
    legs_used = pf["legs"]
    if axis_h or et_session or trailing or sqx_minute:
        legs_used = []
        for leg in pf["legs"]:
            try:
                legs_used.append(_load_leg(
                    leg["path"], pf["account_size"], pf["f_ref"],
                    axis_hours=axis_h,
                    with_peaks=trailing_precompute,
                    day_labeler_tz=source_tz if et_session else None,
                    day_labeler_cut=cfg.day_axis_cut_hhmm,
                    with_minutes=sqx_minute))
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(
                    409, f"Pata '{leg['name']}' no recargable en el eje "
                    f"{req.day_axis}: {exc}")
    if is_sqx:
        # Sin guardas minute_valle/minute_fav: las patas .sqx no usan M1.
        names = [l["name"] for l in legs_used]
        if len(set(names)) != len(names):
            raise HTTPException(409, "Patas .sqx con el mismo nombre de "
                                f"fichero ({names}): renómbralas.")
        bad = []
    else:
        bad = [l["name"] for l in legs_used if not l["s"].minute_valle]
    if bad or not legs_used:
        raise HTTPException(409, "La cartera tiene patas con error de carga; "
                            "recarga sin ellas.")
    if trailing_precompute:
        # Firma trailing sin picos en una pata = raise ruidoso en setup
        # (adjudicado Fase 2): sin la envolvente favorable no hay HWM.
        no_peaks = [l["name"] for l in legs_used if not l["s"].minute_fav]
        if no_peaks:
            raise HTTPException(
                409, f"Perfil con drawdown trailing: pata(s) {no_peaks} sin "
                "picos M1 (minute_fav) — sin envolvente favorable no se "
                "puede construir el high-water mark. Recarga la cartera.")

    symbols = {s.ticker_axi: s for s in _broker.symbols}
    f_grid = np.arange(req.f_lo, req.f_hi + req.f_step / 2, req.f_step)
    if len(f_grid) < 2:
        raise HTTPException(400, "Grilla de f inválida (mínimo 2 puntos).")

    try:
        oos_ts = pd.Timestamp(req.oos_start) if req.oos_start else None
    except ValueError as exc:
        raise HTTPException(400, f"oos_start inválido (se esperaba ISO): {exc}")
    if oos_ts is None:
        # Regla M2-8: evaluar sobre el IS infl P(pass)/EV si hubo selección ahí.
        warnings.append(
            "Ventana completa (sin corte IS/OOS): métricas infladas si las "
            "estrategias se seleccionaron sobre este tramo (regla M2-8); "
            "elige campaña en el selector de corte.")

    # --- Patas (sizing por pata: exacto si la carga lo resolvió, F5a) --------
    strats, leg_rows = {}, []
    sizing_diag: dict[str, list[dict]] = {}
    cap_eval = eval_contract_cap_micros(cfg)
    if len(cfg.phases) > 1 and cap_eval is not None:
        # Plan ftmo-fases-secuencias: el diagnóstico de caps de evaluación recorre
        # solo la ventana de la F1; con 2 fases no cubriría la F2 (hoy ningún perfil
        # de 2 fases declara max_contracts).
        raise HTTPException(
            422, "caps de evaluación con 2 fases: no soportado; el diagnóstico de caps "
            "cubre solo la F1.")
    for leg in legs_used:
        lots, diag = _lots_for_leg(leg, f_grid, symbols, pf["f_ref"],
                                   pf["account_size"], cap_eval)
        strats[leg["name"]] = attach_lots(leg["s"], lots)
        row = {"pata": leg["name"], "símbolo": leg["sym_csv"],
               "modo": leg["sizing_mode"], "trades": _leg_n_trades(leg)}
        if diag is not None:
            sizing_diag[leg["name"]] = diag
            row["backtest_precision"] = leg["fleg"].backtest_precision
        leg_rows.append(row)
    if is_sqx and abs(pf["account_size"] - cfg.account_size) > 1e-9:
        warnings.append(
            f"Opción A usa X = f · account_size de la carga "
            f"({pf['account_size']:,.0f}), distinto del tamaño de cuenta del "
            f"perfil ({cfg.account_size:,.0f}): recarga la cartera con el "
            "tamaño del perfil si no es intencionado.")

    # --- Bloques conjuntos + guardas de ejecución (idéntico a la UI) --------
    oc = {k: (s.open_t, s.close_t) for k, s in strats.items()}
    n_cross = count_crossing(strats, detect_flatflat_blocks(oc))
    if n_cross:
        warnings.append(f"count_crossing = {n_cross} (debe ser 0).")
    # Fase 6: con perfil et_session el open_ord se etiqueta por sesión ET (el
    # MISMO labeler del precomputo de patas — close_d y claves de valle ya
    # vienen etiquetadas de _load_leg).
    blocks = build_day_blocks_portfolio(strats, oos_start=oos_ts,
                                        day_labeler=labeler)
    n_blocks = len(blocks)
    if n_blocks < 5:
        raise HTTPException(422, f"Solo {n_blocks} bloques flat-flat en la ventana.")
    if n_blocks < 30:
        warnings.append(f"n_blocks = {n_blocks} < 30: bootstrap degenerado (M2-11).")
    if oos_ts is not None:
        n_window = int(sum(int(np.sum(pd.to_datetime(s.open_t) >= oos_ts))
                           for s in strats.values()))
    else:
        n_window = int(sum(s.n_trades for s in strats.values()))
    n_in_blocks = int(sum(b.n_trades for b in blocks))
    coverage = (n_in_blocks / n_window) if n_window else float("nan")

    lots_by = {i: s.lots for i, s in enumerate(strats.values())}
    pv_by = {i: s.pv for i, s in enumerate(strats.values())}
    n_f = len(f_grid)
    f_mid = n_f // 2
    csv_minute_usage = None   # D2: contadores de trades sin M1 (solo carteras CSV en modo "minute")

    if is_sqx:
        # Patas .sqx: valle={} por construcción → n_no_valley NO aplica (sin
        # cálculo ni warning); el flotante sale de las ventanas por trade.
        n_no_valley = 0
        pnl_w_nv = 0.0
        float_mode = "windows"
        minute_diag = None
        if sqx_minute:
            # Patas .sqx CFD (2026-10-14): flotante por MINUTO si TODAS las patas
            # tienen serie M1 coherente; si no, ventanas + warning con pata y motivo.
            no_min = [f"{l['name']}: {l.get('minute_reason') or 'sin serie por minuto'}"
                      for l in legs_used if l.get("minute_valle") is None]
            if no_min:
                warnings.append(
                    "Flotante por ventanas (cota más pesimista): "
                    f"{len(no_min)}/{len(legs_used)} pata(s) sin OHLC M1 coherente — "
                    + "; ".join(no_min))
            else:
                float_mode = "minute"
                minute_diag = {l["name"]: l["minute_diag"] for l in legs_used}
        if float_mode == "minute":
            # mismo orden de patas que lots_by/pv_by (= orden de ``strats``)
            minute_by_sym = {i: leg["minute_valle"]
                             for i, leg in enumerate(legs_used)}
            day_arrays_grid = precompute_day_arrays_portfolio(
                blocks, lots_by, pv_by, n_f,
                float_mode="minute", minute_by_sym=minute_by_sym,
                trailing=False, day_labeler=labeler)
        else:
            adverse_by_sym = {i: leg["fleg"].adverse_per_lot
                              for i, leg in enumerate(legs_used)}
            day_arrays_grid = precompute_day_arrays_portfolio(
                blocks, lots_by, pv_by, n_f,
                float_mode="windows", adverse_by_sym=adverse_by_sym,
                trailing=False, day_labeler=labeler)
    else:
        minute_diag = None
        n_no_valley = int(sum(b.n_no_valley for b in blocks))
        w_nv = w_tot = 0.0
        for b in blocks:
            for j in range(b.n_trades):
                w = abs(lots_by[b.sym_id[j]][f_mid, b.trade_local_id[j]] * b.pnl_per_lot[j])
                w_tot += w
                if not b.valle_precio[j]:
                    w_nv += w
        pnl_w_nv = (w_nv / w_tot) if w_tot else float("nan")
        # D2 (csv-d1-d2, fase 2; plan §2.2 y E3): las patas CSV usan el flotante POR MINUTO con el realizado intradía
        # si y solo si el perfil es ESTÁTICO (``not _profile_extended``: cycle, sin trailing ni et_session). En esos
        # perfiles ``labeler`` es None (solo se asigna con .sqx de futuros o et_session), así que joint y minute
        # agrupan por el mismo día de calendario. Con trailing (Apex, futuros) sigue el camino de antes.
        if (_check_float_flag("CSV_FLOAT_MODE", CSV_FLOAT_MODE) == "minute"
                and not _profile_extended(cfg)):
            # Orden de patas = el de ``strats`` (= lots_by/pv_by y ``sym_id`` de los bloques).
            legs_ord = _legs_in_strats_order(legs_used, strats)
            csv_minute_usage = _mae_usage(blocks, legs_ord, lots_by, f_mid)
            warnings.extend(_mae_warnings(csv_minute_usage))
            float_mode = "minute"
            minute_by_sym = {i: _csv_minute_series(leg)[0] for i, leg in enumerate(legs_ord)}
            day_arrays_grid = precompute_day_arrays_portfolio(
                blocks, lots_by, pv_by, n_f,
                float_mode="minute", minute_by_sym=minute_by_sym,
                trailing=False, day_labeler=labeler)
        else:
            if n_no_valley:
                warnings.append(f"{n_no_valley} trades sin valle OHLC (peso PnL f central "
                                f"{pnl_w_nv:.1%}): flotante intradía cuenta como 0.")

            # float_mode="joint" FIJO en esta capa (§2.3): minute_valle validado en carga.
            float_mode = FLOAT_MODE
            minute_by_sym = {i: s.minute_valle for i, s in enumerate(strats.values())}
            # Fase 6 (trailing): envolventes favorables por pata + labeler de sesión
            # para rel_low/rel_high/rel_exc (validados arriba: raise ruidoso si falta).
            minute_fav_by_sym = (
                {i: s.minute_fav for i, s in enumerate(strats.values())}
                if trailing_precompute else None)
            day_arrays_grid = precompute_day_arrays_portfolio(
                blocks, lots_by, pv_by, n_f,
                float_mode=FLOAT_MODE, minute_by_sym=minute_by_sym,
                trailing=trailing_precompute, minute_fav_by_sym=minute_fav_by_sym,
                day_labeler=labeler)
    durations = np.array([b.duration_days for b in blocks])
    gaps_emp = empirical_gaps(blocks)

    # Tamaño de bloque (guardrail bloque-politis-white): manual si ``block_size > 0``; si no, Politis-White sobre la serie
    # en R de EXACTAMENTE estos ``blocks`` (los de la ventana ``oos_start``), con las patas en el orden de ``strats``.
    bs = float(req.block_size)
    bs_mode, bs_info = "manual", None   # eco para meta.model_context
    if bs <= 0:
        bs, bs_mode, bs_info, bs_warns = _auto_block_size(blocks, _legs_in_strats_order(legs_used, strats), pf)
        warnings.extend(bs_warns)

    rng, idx_matrix, idx_ext = _propfirm_index_matrices(n_blocks, req.n_paths, bs)

    return {
        "warnings": warnings, "cfg": cfg, "f_grid": f_grid, "oos_ts": oos_ts,
        "profile_snapshot": profile_snapshot,
        "strats": strats, "leg_rows": leg_rows, "blocks": blocks,
        "legs_used": legs_used, "lots_by": lots_by,
        # F4: el tamaño de cuenta con que se calcularon los lotes (``X = f · A``: el de la CARTERA, no el del perfil)
        "lots_account_size": float(pf["account_size"]),
        "trailing": trailing, "et_session": et_session,
        "use_portfolio": use_portfolio, "float_mode": float_mode,
        "csv_minute": csv_minute_usage,
        "kind": KIND_SQX if is_sqx else KIND_CSV, "sizing_diag": sizing_diag,
        "sqx_asset": pf.get("sqx_asset") if is_sqx else None,
        "minute_diag": minute_diag,
        "source_tz": source_tz,
        "cap_eval_micros": cap_eval,
        "cap_pa_micros": pa_min_tier_cap_micros(cfg),
        "n_blocks": n_blocks, "n_window": n_window, "coverage": coverage,
        "n_no_valley": n_no_valley, "pnl_w_nv": pnl_w_nv, "bs": bs,
        "bs_mode": bs_mode, "bs_info": bs_info,
        "day_arrays_grid": day_arrays_grid, "durations": durations,
        "gaps_emp": gaps_emp, "idx_matrix": idx_matrix, "idx_ext": idx_ext,
        "rng": rng,
    }


def _sqx_meta_block(ctx: dict, leg_rows: list[dict]) -> dict:
    """Claves de meta de una cartera .sqx (1D y 2D): ``leg_kind``, ``sizing_diag``,
    ``backtest_precision`` y ``float_note`` según el modo de flotante; ``minute_diag`` SOLO
    en modo ``minute`` (en ``windows`` la clave no existe, ni null ni vacía)."""
    precs = {l["pata"]: l.get("backtest_precision") for l in leg_rows}
    prec_txt = ", ".join(sorted({str(v) for v in precs.values()}))
    if ctx["float_mode"] == "minute":
        note = (
            "flotante por minuto (OHLC M1 bid del símbolo SQX): COTA por minuto — el "
            "peor precio de cada minuto para todas las posiciones a la vez —, con el "
            "realizado intradía dentro del mínimo del día, NETO de la comisión RT y del "
            "swap que es cargo (ambos cargados desde la apertura) y con el spread fijo "
            "del YAML sumado en los cortos (el OHLC es bid; si el MAE de SQX ya lo incluye, "
            "cuenta dos veces: conservador). La magnitud por trade queda "
            "acotada por el MAE de SQX (+ costes, + spread en cortos), que a su vez es una "
            "cota del modelo del backtest; el minuto aporta la simultaneidad (qué trades "
            "están mal a la vez). Residuales ANTI-conservadores: "
            "el vaivén entre la última barra y el cierre no se acota (≈ 0 con cierres en "
            "minuto exacto) y el spread en vivo puede ser más ancho que el fijo (noticias, "
            "rollover). No es la equity exacta (backtest_precision=" + prec_txt + ").")
    else:
        note = (
            "valle por ventanas de trade (cota inferior sin suponer "
            "cuándo ocurrió el MAE), NETO de la comisión RT"
            + ("; en CFD el swap que es cargo entra entero desde la apertura"
               if ctx.get("sqx_asset") == SQX_ASSET_CFD else "")
            + "; incluye el realized intradía. Cota a precisión del backtest "
            "(backtest_precision=" + prec_txt + ").")
    out = {"leg_kind": KIND_SQX, "sizing_diag": ctx["sizing_diag"],
           "backtest_precision": precs, "float_note": note}
    if ctx["float_mode"] == "minute":
        out["minute_diag"] = ctx["minute_diag"]
    return out


def _futures_sequence(outcomes, cfg, req, seed):
    """Secuencia de campañas FTMO Futures (helper de módulo para
    testabilidad/paridad web↔CLI). ``n_sequences`` = nº de campañas;
    seed fija (_FUTURES_SEQ_SEED) para CRN entre f (enmienda E1)."""
    return sequence_ev_futures(
        outcomes, cfg, fx_eur_usd=req.fx_eur_usd, n_paths=req.n_sequences,
        initial_capital_eur=req.capital0_eur, seed=seed,
        reset_policy=req.reset_policy,
        stop_on_insolvency=req.stop_on_insolvency)


def _propfirm_eval_f(ctx: dict, req: PropfirmSimRequest, cfg, futures: bool, fi: int, f: float,
                     idx_matrix, idx_ext):
    """Evalúa UN f de la rejilla de la lente prop firm con la matriz de índices dada: ``(report, seq, error)``.

    Es la evaluación del barrido 1D (``seed=fi`` por f, diagnósticos de caps de contratos, secuencia de campañas y
    reporte) parametrizada por ``(f, matrices del bloque)``: el barrido la llama con las matrices de la petición y la
    sensibilidad con bloque 1 (:func:`_propfirm_bs1`) con las de bloque 1 (guardrail bloque-politis-white, E4). ``error`` es
    el mensaje del ``ValueError`` de la secuencia (sin paths resueltos); entonces ``report`` y ``seq`` son ``None``. El
    reporte es un ``FirmReport`` o, en perfiles futures, un ``FuturesReport``. ``seq`` es ``None`` solo con error."""
    blocks, lots_by = ctx["blocks"], ctx["lots_by"]
    outcomes = simulate_firm_paths(idx_matrix, ctx["day_arrays_grid"][fi], ctx["durations"],
                                   cfg, gap_days=ctx["gaps_emp"], seed=fi,
                                   portfolio=ctx["use_portfolio"],
                                   idx_ext=idx_ext)
    if ctx["cap_eval_micros"] is not None:
        outcomes.contract_cap_eval = contract_cap_diagnostic(
            idx_matrix, blocks, lots_by, fi,
            cap_micros=ctx["cap_eval_micros"], scope="evaluation")
    if ctx["cap_pa_micros"] is not None:
        # Cap de la fondeada sobre SUS ventanas (continuación de la secuencia).
        outcomes.contract_cap_pa = contract_cap_diagnostic(
            outcomes.funded_windows,
            blocks, lots_by, fi,
            cap_micros=ctx["cap_pa_micros"], scope="funded",
            mask=outcomes.both)
    try:
        if futures:
            seq = _futures_sequence(outcomes, cfg, req, seed=_FUTURES_SEQ_SEED)
        else:
            seq = sequence_ev(outcomes, cfg, fx_eur_usd=req.fx_eur_usd,
                              n_challenges=req.n_challenges,
                              n_sequences=req.n_sequences,
                              initial_capital_eur=req.capital0_eur, seed=fi,
                              stop_on_insolvency=req.stop_on_insolvency,
                              n_copied_accounts=req.n_copied_accounts,
                              cap_violation_tol=req.cap_violation_tol)
    except ValueError as exc:
        return None, None, str(exc)
    if futures:
        return firm_report_futures(
            float(f), outcomes, seq, cfg=cfg,
            reset_policy=req.reset_policy,
            withdrawal_policy=req.withdrawal_policy,
            cap_violation_tol=req.cap_violation_tol), seq, None
    return firm_report(float(f), outcomes, seq), seq, None


def _propfirm_bs1(ctx: dict, req: PropfirmSimRequest, cfg, futures: bool, f_grid,
                  f_star: float) -> tuple[list[dict] | None, list[str]]:
    """Sensibilidad con bloque 1 de la lente prop firm en f*: ``([{f, p_both, p_net_negative, ev_per_attempt}], avisos)``.

    Vuelve a evaluar f* con :func:`_propfirm_eval_f` y las matrices de bloque 1 que salen de generadores NUEVOS
    (:func:`_propfirm_index_matrices`: ``idx_matrix`` con ``default_rng(0)`` e ``idx_ext`` con ``default_rng([0, 2])``), de modo
    que el resultado es el de una petición manual con ``block_size = 1`` en la fila de f*. Aviso si
    ``p_net_negative > tol_neg`` (el criterio con que ``recommend_f_ev`` llama factible a una f). Si la secuencia no resuelve
    ningún camino, ``(None, [aviso])``."""
    fi = int(np.argmin(np.abs(np.asarray(f_grid, dtype=float) - f_star)))
    f = float(f_grid[fi])
    _, idx1, idx_ext1 = _propfirm_index_matrices(ctx["n_blocks"], req.n_paths, 1.0)
    rep, _seq, err = _propfirm_eval_f(ctx, req, cfg, futures, fi, f, idx1, idx_ext1)
    if rep is None:
        return None, [f"Sensibilidad con bloque 1 no disponible en f* = {100.0 * f:.2f} %: {err}"]
    warns = []
    if rep.p_net_negative > req.tol_neg:
        warns.append(_bs1_warning(f, "P(neto<0)", rep.p_net_negative, req.tol_neg))
    return [{"f": f, "p_both": rep.p_both, "p_net_negative": rep.p_net_negative,
             "ev_per_attempt": rep.ev_per_attempt}], warns


# ---------------------------------------------------------------------------
# Receta para el EA (guardrail entrada-4-bloques, fase F4; plan §2.8 y enmiendas E3, E12, E14, E18)
# ---------------------------------------------------------------------------
# Notas del modelo (``config/model_notes.yaml``) que cita ``meta.recipe.notes``, por lente: la receta y de dónde salen las
# especificaciones (las dos lentes) y, en la lente AXI, cómo se calcula el flotante de las patas ``.sqx``.
# PLAN-DESVIO (interpretación): el brief dice que «la meta las cita por id» sin decir dónde; se citan aquí, en ``meta.recipe.notes``
# con la forma ``{id, texto}`` del §2.8 (el catálogo de «Supuestos del perfil» y «¿Cómo se ha calculado?» las lista además por su
# ámbito). No pueden ir en ``meta.model_context.notes``: solo admite notas con ``en_esta_corrida`` y ``test_model_context.py``
# fija esa lista con las notas del catálogo que la tienen.
RECIPE_NOTE_IDS = {"propfirm": ("receta_ea", "fichas_broker"),
                   "axi": ("receta_ea", "fichas_broker", "axi_sqx_flotante")}


def _lot_step_decimals(step: float) -> int:
    """Nº de decimales de ``lot_step`` (el ``Decimals`` del EA): 0,01 → 2; 0,1 → 1; 1 → 0."""
    for n in range(13):
        if abs(round(step, n) - step) <= 1e-12 * max(1.0, abs(step)):
            return n
    return 12


def _recipe_max_lot(lots_row, step: float) -> float:
    """``MaxLots`` de la receta a una f: el lote MÁXIMO de la pata en la simulación (el mismo array que entra al motor,
    una fila de ``_lots_for_leg``), redondeado al alza al paso con ``round(…, 10)``. Es un tope que nunca muerde en lo
    simulado (E3): ningún lote simulado lo supera, y quita el ruido de coma flotante de ``k · 0,01``.

    Tolerancia (H4 de la revisión de código de la F4): el redondeo es al paso, salvo ruido de coma flotante < 1e-12 en un
    array que NO venga redondeado (``ceil(x/paso − 1e-9)`` da ``0,3`` para ``0,30000000000000004``: un 4e-17 por debajo, que el
    EA tampoco ve porque trunca igual). El array de ``fixed_risk_micros`` SÍ viene redondeado a 10 decimales, así que en la
    práctica el tope es ≥ todo lote simulado sin ninguna tolerancia, y los tests lo exigen así sobre los arrays reales."""
    return round(math.ceil(float(np.max(lots_row)) / step - _FLOOR_EPS) * step, 10)


def _json_keep_bool(obj):
    """Como :func:`_jsonable` (numpy/NaN → JSON puro) pero CONSERVA los booleanos: ``_jsonable`` mira ``int`` antes que
    ``bool`` y los vuelve 0/1 (``catalog()`` corrige ``official`` y ``hidden`` a mano). La receta lleva listas de
    booleanos (``max_lot_excede_ficha_by_f``) que el front tipa como ``boolean[]``."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_keep_bool(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_keep_bool(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_keep_bool(v) for v in obj.tolist()]
    return _jsonable(obj)


def _recipe_notes(lens: str) -> list[dict]:
    """``meta.recipe.notes``: ``[{id, texto}]`` de las notas del catálogo que cita la receta en esa lente (E18: el lote
    mínimo en ``LotsIfNoMM`` se presenta como CONDICIÓN para que el EA haga lo simulado; E14: la cuenta del EA en USD)."""
    by_id = {n.id: n for n in NOTES}
    return [{"id": nid, "texto": by_id[nid].texto} for nid in RECIPE_NOTE_IDS[lens]]


def _sqx_recipe(legs: list[dict], lots_by: dict[str, np.ndarray], diag_by: dict[str, list[dict]],
                f_grid, account_size: float, lens: str) -> tuple[dict, list[str]]:
    """``(receta, avisos)`` para configurar en el EA el riesgo fijo que simula la app (``meta.recipe``; solo carteras
    ``.sqx``). El front la compone para la f elegida (F5): aquí va TODA la rejilla.

    ``legs`` = las patas SIMULADAS (en el orden de la simulación), ``lots_by[pata]`` = el array de lotes ``(n_f, n_trades)``
    que entra al motor y ``diag_by[pata]`` = el ``diag`` por f de :func:`_lots_for_leg`. ``account_size`` = el tamaño de
    cuenta con el que se calcularon esos lotes (el de la cartera). Por pata y f:

    - ``max_lot_by_f`` → ``MaxLots`` (:func:`_recipe_max_lot`); ``max_lot_excede_ficha_by_f`` → el máximo simulado supera el
      máximo del bróker de la ficha (``ficha_max_lot``): el EA toparía en el del bróker y la app no (aviso en ``avisos``);
    - ``frac_min_lot_by_f`` = ``frac_clamp_up`` del ``diag`` (fracción de órdenes cuyo riesgo ``X`` no llega al lote
      mínimo) y ``p95_riesgo_real_pct_by_f`` = ``p95_risk_eff_over_X · f · 100`` (E12): el riesgo real p95 en % de la cuenta;
    - ``LotsIfNoMM`` = ``min_lot`` (la CONDICIÓN para que el EA haga lo simulado, E18), ``mmStep`` = ``lot_step``,
      ``mmMultiplier`` = 1 y ``Decimals`` = decimales de ``lot_step`` (E3)."""
    f_list = [float(f) for f in f_grid]
    a = float(account_size)
    legs_out: list[dict] = []
    warns: list[str] = []
    for leg in legs:
        name = leg["name"]
        meta = leg.get("meta") or {}
        # los mismos paso y mínimo que usa ``_lots_for_leg`` para dimensionar
        step = float(meta.get("lot_step") or 1.0)
        min_lot = float(meta.get("min_lot") or 1.0)
        lots = np.asarray(lots_by[name], dtype=float)
        diag = diag_by[name]
        if lots.ndim != 2 or lots.shape[0] != len(f_list) or len(diag) != len(f_list):
            raise ValueError(f"receta: los lotes de la pata '{name}' no cubren la rejilla de f "
                             f"({lots.shape} lotes, {len(diag)} diag, {len(f_list)} f)")
        max_by_f = [_recipe_max_lot(lots[i], step) for i in range(len(f_list))]
        fm = (leg.get("ficha") or {}).get("max_lot")
        fm = None if fm is None else float(fm)
        exceeds = [fm is not None and m > fm for m in max_by_f]
        if any(exceeds):
            fs = ", ".join(f"{f:.2%}" for f, e in zip(f_list, exceeds) if e)
            top = max(m for m, e in zip(max_by_f, exceeds) if e)
            warns.append(f"Receta para el EA: en la pata '{name}' el lote máximo simulado ({top:g}) supera el máximo del "
                         f"bróker de su ficha ({fm:g}) con f = {fs}: el EA toparía en ese máximo y la app no.")
        legs_out.append({
            "pata": name, "symbol": str(meta.get("symbol", leg.get("sym_csv", ""))), "broker": leg.get("broker"),
            "min_lot": min_lot, "lot_step": step, "decimals": _lot_step_decimals(step),
            "mmStep": step, "mmMultiplier": 1, "LotsIfNoMM": min_lot,
            "max_lot_by_f": max_by_f,
            "ficha_max_lot": fm,
            "max_lot_excede_ficha_by_f": exceeds,
            "frac_min_lot_by_f": [float(d["frac_clamp_up"]) for d in diag],
            "p95_riesgo_real_pct_by_f": [float(d["p95_risk_eff_over_X"]) * float(d["f"]) * 100.0 for d in diag],
            "mm_backtest": copy.deepcopy(leg.get("mm_backtest")),
            "paridad": copy.deepcopy(leg.get("paridad")),
        })
    recipe = {"mm": "FixedAmount", "currency": "USD", "account_size": a, "f_grid": f_list,
              "risked_money_by_f": [f * a for f in f_list], "legs": legs_out, "notes": _recipe_notes(lens)}
    return _json_keep_bool(recipe), warns


@app.post("/api/simulate/propfirm")
def simulate_propfirm(req: PropfirmSimRequest):
    """Lente prop firm sobre una cartera cargada. Pipeline EXACTO del PoC F0.

    Cartera ``.sqx`` (fase F4): la meta añade ``recipe`` (:func:`_sqx_recipe`), la receta para configurar el EA con el riesgo
    fijo simulado, con el tamaño de cuenta de la CARTERA y los lotes (topados por el perfil, si los hay) que entraron al motor.
    Las carteras CSV no llevan la clave."""
    t0 = time.perf_counter()
    ctx = _propfirm_setup(req)
    warnings: list[str] = ctx["warnings"]
    cfg = ctx["cfg"]
    f_grid = ctx["f_grid"]
    strats = ctx["strats"]
    leg_rows = ctx["leg_rows"]
    legs_used = ctx["legs_used"]
    n_blocks = ctx["n_blocks"]
    n_window = ctx["n_window"]
    coverage = ctx["coverage"]
    n_no_valley = ctx["n_no_valley"]
    pnl_w_nv = ctx["pnl_w_nv"]
    bs = ctx["bs"]
    idx_matrix = ctx["idx_matrix"]
    blocks = ctx["blocks"]
    trailing = ctx["trailing"]
    et_session = ctx["et_session"]
    is_sqx = ctx["kind"] == KIND_SQX

    # --- Rama futures (plan §3.2-3.3): economía de suscripción por campaña ---
    # La política de retirada se aplica ANTES del bucle f: la lee sweep_funded
    # dentro de simulate_firm_paths (riesgo 7 del plan). n_copied_accounts y
    # n_challenges no aplican; n_sequences = nº de campañas.
    futures = cfg.funded.model == "futures"
    if futures:
        cfg = withdrawal_policy_cfg(cfg, req.withdrawal_policy)
        if req.n_copied_accounts > 1:
            warnings.append(
                f"n_copied_accounts={req.n_copied_accounts} ignorado en "
                "perfiles futures: la campaña corre sobre una cuenta y la "
                "correlación perfecta no aplica al EV de suscripción.")
        warnings.append(
            "n_challenges no aplica a perfiles futures: la campaña corre "
            "hasta pase/cap (183d truncado del modelo, no regla FTMO); "
            "n_sequences = nº de campañas.")

    # --- Barrido 1D (seed=fi como la UI) ------------------------------------
    # Fase 6: el dispatch trailing va por el camino PORTFOLIO (mono = cartera
    # de 1 pata); los diagnósticos de caps (Fase 5) se calculan fuera y se
    # acoplan a los outcomes ANTES de la secuencia (umbral cap_violation_tol
    # → sizing_invalid). El acople corre para AMBAS ramas (enmienda E3).
    # Guardrail bloque-politis-white (E4): la evaluación de cada f es ``_propfirm_eval_f``, parametrizada por la matriz de
    # índices (la de la petición aquí; la de bloque 1 en la sensibilidad). Mismas sentencias y orden que antes.
    reports, trajectories = [], {}
    for fi, f in enumerate(f_grid):
        report, seq, err = _propfirm_eval_f(ctx, req, cfg, futures, fi, f, idx_matrix, ctx["idx_ext"])
        if report is None:
            warnings.append(f"f = {f:.2%}: {err}")
            continue
        reports.append(report)
        if futures:
            continue
        trajectories[float(f)] = {
            "steps": list(range(req.n_challenges + 1)),
            "p05": seq.trajectory_p05, "p50": seq.trajectory_p50,
            "p95": seq.trajectory_p95,
        }

    if not reports:
        raise HTTPException(422, "Ningún f produjo paths resueltos.")
    if not futures:
        max_unres = max(r.p_unresolved for r in reports)
        if max_unres > 0.05:
            warnings.append(f"Hasta {max_unres:.0%} de paths sin resolver (neither); "
                            "EV condicional a resolver.")
    rec = recommend_f_ev(reports, req.tol_neg)

    f_star = rec.f if rec is not None else reports[len(reports) // 2].f
    if futures:
        # sequence_ev_futures NO produce trayectorias (enmienda E2: con
        # trajectories == {} el next(iter(...)) histórico revienta).
        traj = None
    else:
        traj = trajectories.get(float(f_star)) or next(
            iter(trajectories.values()))

    # Fase 6: los warnings estructurados del motor (SIZING_INVALID, MULTI_CUENTA)
    # viven por reporte; se surfcean también a nivel corrida desde el f* (el
    # punto de decisión) para que el bloque de warnings de la UI los muestre.
    star_rep = next((r for r in reports if abs(r.f - f_star) < 1e-9), None)
    if star_rep is not None:
        for w in star_rep.warnings:
            if w not in warnings:
                warnings.append(f"@f*={f_star:.2%}: {w}")

    # Sensibilidad con bloque 1 en f* (guardrail bloque-politis-white, E4/E15/E16): solo con un bloque principal > 1 (modo
    # automático o manual). Se calcula DESPUÉS de f* y no toca nada de lo ya calculado (E19).
    bs1 = None
    if _bs1_wanted(bs):
        bs1, bs1_warns = _propfirm_bs1(ctx, req, cfg, futures, f_grid, f_star)
        if rec is not None:   # revisión de código de la fase A, H3: sin recomendación, f* no es una f recomendada
            warnings.extend(bs1_warns)

    meta = {
        "firm": cfg.display_name, "account_size": cfg.account_size, "fee_eur": cfg.fee,
        "legs": leg_rows, "n_blocks": n_blocks, "coverage": coverage,
        "n_no_valley": n_no_valley, "pnl_w_no_valley": pnl_w_nv,
        "block_size": bs, "bs1": bs1, "float_mode": ctx["float_mode"],
        "sizing": _sizing_label(legs_used),
        # D1 (csv-d1-d2): eco de la base del lote (tamaño de cuenta, sin componer).
        "sizing_base": _sizing_base(ctx["kind"]),
        # F2c: eco de ventana y eje (el diagnóstico/historial etiquetan con esto).
        "oos_start": req.oos_start,
        "trades_in_window": n_window,
        "day_axis": "et_session" if et_session else req.day_axis,
        **({"day_axis_note": (
            "eje servidor AXI (00:00 EET/EEST): el daily loss de FTMO resetea a "
            "00:00 CE(S)T = 01:00 servidor; a este eje P(pass) queda ~1-5pp "
            "optimista — usar solo como comparación.")}
           if req.day_axis == "servidor" and not et_session else {}),
        # Fase 6 (Apex): eco del perfil extendido y del plumbing aplicado.
        "ruleset_version": cfg.ruleset_version or None,
        "source_tz": ctx["source_tz"],
        "trailing": trailing,
        "n_copied_accounts": req.n_copied_accounts,
        "cap_violation_tol": req.cap_violation_tol,
        "sizing_invalid_any": any(r.sizing_invalid for r in reports),
        "n_paths": req.n_paths, "elapsed_s": time.perf_counter() - t0,
    }
    if is_sqx:
        # Patas .sqx (Opción A + ventanas o minuto): eco de la fuente del valle, del
        # sizing por pata y f, y de la precisión del backtest (V8/DS-H7).
        meta.update(_sqx_meta_block(ctx, leg_rows))
        if ctx["float_mode"] != "minute":
            precs = meta["backtest_precision"]
            warnings.append(
                "Cota a precisión del backtest (backtest_precision="
                + ", ".join(f"{k}: {v}" for k, v in precs.items())
                + "): el MAE del .sqx se mide a esa resolución (1 = barra del "
                "timeframe de la estrategia; un retest M1 da un MAE más fiel). La "
                "cota de ventanas es algo conservadora entre patas solapadas y "
                "puede serlo más que FTMO (comisión RT viva).")
    # D2 (csv-d1-d2, fase 2): cartera CSV con flotante por minuto → nota del flotante y contadores de los trades
    # sin OHLC M1 (``{}`` en cualquier otro caso: sin claves nuevas).
    meta.update(_csv_minute_meta(ctx))
    if futures:
        # Fase 2 (FTMO Futures): semántica del EV y políticas aplicadas.
        meta.update({
            "funded_model": "futures",
            "reset_policy": req.reset_policy,
            "withdrawal_policy": req.withdrawal_policy,
            "day_axis_cut_hhmm": cfg.day_axis_cut_hhmm,
            "ev_semantics": (
                "EV/campaña incondicional en EUR (cap "
                f"{cfg.subscription.campaign_cap_days}d = truncado del "
                "modelo, no regla FTMO); ev_per_attempt = ev_per_campaign; "
                "n_sequences = nº de campañas; p_pass1/p_unresolved "
                "incondicionales sobre N paths; seed fija de secuencia "
                "(CRN entre f)"),
        })
    elif req.reset_policy != "billing_retry" \
            or req.withdrawal_policy != "greedy":
        warnings.append(
            f"reset_policy/withdrawal_policy ignoradas: el perfil {cfg.key} "
            "no es futures")
    # onboarding-b1: contexto de modelo (aditivo). ``pct_cortadas_datos`` solo
    # existe en ``cycle`` (fondeadas vivas al acabarse los datos, al f*).
    mc_extra = {}
    # (``FuturesReport`` no trae el campo: getattr con default.)
    cut_frac = getattr(star_rep, "funded_cut_by_data_frac", None)
    if cut_frac is not None and np.isfinite(cut_frac):
        mc_extra["pct_cortadas_datos"] = _pct_label(cut_frac)
    meta["model_context"] = _model_context(
        lente="propfirm", blocks=blocks, n_blocks=n_blocks, bs=bs,
        bs_mode=ctx["bs_mode"], bs_info=ctx["bs_info"],
        cfg=cfg, extra=mc_extra)
    intentos = int(req.capital0_eur // cfg.fee) if cfg.fee > 0 else None
    recipe = None
    if is_sqx:
        # F4: receta para el EA (solo carteras .sqx; las CSV no llevan la clave). Con el MISMO array de lotes que entró al
        # motor (``strats[...].lots``, ya con el tope de contratos del perfil en futuros) y el tamaño de cuenta de la
        # cartera con el que se calcularon. Toda la rejilla de f: el front compone la de la f elegida (F5).
        recipe, recipe_warns = _sqx_recipe(legs_used, {n: s.lots for n, s in strats.items()}, ctx["sizing_diag"],
                                           f_grid, ctx["lots_account_size"], "propfirm")
        warnings.extend(recipe_warns)
    out = _with_profile_snapshot(_jsonable({
        "meta": meta, "warnings": warnings,
        "reports": reports, "recommendation": rec,
        "trajectory": traj, "f_star": float(f_star),
        "intentos_garantizados": intentos,
    }), ctx)
    if recipe is not None:
        # DESPUÉS de ``_jsonable`` (que vuelve los booleanos 0/1): la receta ya es JSON puro y conserva los booleanos
        out["meta"]["recipe"] = recipe
    return out


@app.post("/api/simulate/propfirm_pair")
def simulate_propfirm_pair(req: PropfirmSimRequest):
    """F3: barrido 2D (f_challenge, f_funded) con ΔEV pareado vs la diagonal.

    Composición de capa-app idéntica al tab par de Streamlit
    (``risk_app/propfirm_mode.py``): la cinta de gaps (``_draw_gap_tape``: los
    ``gaps_fixed`` de la F1 + su prolongación) dibujada UNA vez del MISMO
    rng que la idx_matrix (CRN global), ``simulate_challenge``/``sweep_funded``
    sobre la :class:`PhaseTape` (la F2 y la fondeada continúan la secuencia de la F1)
    por f_challenge, y por celda ``combine_challenge_funded`` →
    ``sequence_ev(seed=0)`` → ``firm_report_pair``; ``recommend_f_pair`` trae el
    IC95 bootstrap pareado del ΔEV (n_boot=2000) y el gate ``recommend_pair``.
    Sensibilidad al cap SOLO en 2 celdas (par recomendado + mejor diagonal):
    cap máximo vs cap10. El endpoint 1D no se toca (baseline intacto).

    2026-10-14: admite carteras de patas ``.sqx`` CFD FTMO (perfil FTMO CFD; flotante por
    minuto o ventanas según ``sqx_float``, ver :func:`_propfirm_setup`); las de futuros
    siguen en 409. ``meta`` lleva el ``float_mode`` real y, con ``.sqx``, ``leg_kind``,
    ``sizing_diag``, ``backtest_precision``, ``float_note`` (y ``minute_diag`` solo en
    modo minuto).
    """
    t0 = time.perf_counter()
    # Guarda .sqx ANTES que el 400 de perfiles futures y que cualquier acceso
    # a tl/s (enmienda V7). 2026-10-14: las carteras .sqx CFD SÍ admiten el 2D (el
    # perfil FTMO CFD lo exige ``_propfirm_setup``); las de futuros siguen en 409. La
    # guarda NO se relaja (la comparten el lente AXI y el diagnóstico por pata).
    _pf2d = _portfolios.get(req.portfolio_id)
    if not (_pf2d is not None and _pf2d.get("sqx_asset") == SQX_ASSET_CFD):
        _reject_sqx_portfolio(req.portfolio_id, "El barrido 2D (par)")
    ctx = _propfirm_setup(req)
    if ctx["cfg"].funded.model == "futures":
        raise HTTPException(
            400, "Barrido 2D no soportado para perfiles FTMO Futures: la "
            "economía es por campaña de suscripción (f_funded no aplica) — "
            "usa el barrido 1D.")
    n_f_grid = len(ctx["f_grid"])
    if n_f_grid > 8:
        raise HTTPException(
            400, f"Grilla {n_f_grid}×{n_f_grid} = {n_f_grid**2} celdas excede "
            "el límite 8×8: amplía el paso f o acota el rango.")
    warnings: list[str] = ctx["warnings"]
    cfg = ctx["cfg"]
    f_grid = ctx["f_grid"]
    leg_rows = ctx["leg_rows"]
    day_arrays_grid = ctx["day_arrays_grid"]
    durations = ctx["durations"]
    gaps_emp = ctx["gaps_emp"]
    idx_matrix = ctx["idx_matrix"]
    blocks = ctx["blocks"]
    lots_by = ctx["lots_by"]
    trailing = ctx["trailing"]
    et_session = ctx["et_session"]

    # Gaps fijos (CRN): misma cinta de paths y gaps para TODA la grilla 2D
    # (continúa el stream del rng que dibujó idx_matrix, como en Streamlit).
    # ``_draw_gap_tape``: los primeros n_blocks son los gaps_fixed de siempre (F1);
    # después, la prolongación para la F2 y la fondeada (continuación de la secuencia).
    rng_pair = ctx["rng"]
    if gaps_emp is not None and len(gaps_emp):
        gaps_fixed, gaps_ext = _draw_gap_tape(rng_pair, gaps_emp, *idx_matrix.shape)
    else:
        gaps_fixed, gaps_ext = None, None
    tape = build_phase_tape(idx_matrix, ctx["idx_ext"], gaps_fixed, gaps_ext)
    del gaps_fixed, gaps_ext      # la cinta ya los lleva: no dejar copias vivas

    # Fase 1: challenges y fondeada por f_challenge (passers comparten paths).
    # Fase 6: dispatch trailing por el camino portfolio (mono = cartera de 1
    # pata); diagnóstico de caps (Fase 5) acoplado por celda — evaluación a
    # f_c, PA a f_f sobre los passers de f_c (combine los recoge).
    challenge_by_fc, funded_by_fc = {}, {}
    for fi_c, f_c in enumerate(f_grid):
        ch = simulate_challenge(tape, day_arrays_grid[fi_c], durations,
                                cfg, portfolio=ctx["use_portfolio"])
        if ctx["cap_eval_micros"] is not None:
            ch.contract_cap = contract_cap_diagnostic(
                idx_matrix, blocks, lots_by, fi_c,
                cap_micros=ctx["cap_eval_micros"], scope="evaluation")
        challenge_by_fc[float(f_c)] = ch
        funded_by_fc[float(f_c)] = sweep_funded(
            tape=tape, funded_start=ch.funded_start,
            day_arrays_f_grid=day_arrays_grid, passers_mask=ch.both,
            durations=durations, cfg=cfg, f_grid=f_grid)
        if ctx["cap_pa_micros"] is not None:
            fwin = tape.windows(ch.funded_start)     # ventanas de la fondeada (por f_c)
            for fi_f, f_f in enumerate(f_grid):
                fu = funded_by_fc[float(f_c)][float(f_f)]
                fu.contract_cap = contract_cap_diagnostic(
                    fwin, blocks, lots_by, fi_f,
                    cap_micros=ctx["cap_pa_micros"], scope="funded",
                    mask=ch.both)

    # Fase 2: celda a celda — secuencia de challenges combinados (seed=0 CRN
    # aproximado a nivel de secuencias, mismo criterio que 1D para celdas sin
    # paths resueltos).
    reports_2d, cells = {}, {}
    for f_c in f_grid:
        for f_f in f_grid:
            oc_cell = combine_challenge_funded(challenge_by_fc[float(f_c)],
                                               funded_by_fc[float(f_c)][float(f_f)])
            try:
                seq = sequence_ev(
                    oc_cell, cfg, fx_eur_usd=req.fx_eur_usd,
                    n_challenges=req.n_challenges, n_sequences=req.n_sequences,
                    initial_capital_eur=req.capital0_eur, seed=0,
                    stop_on_insolvency=req.stop_on_insolvency,
                    n_copied_accounts=req.n_copied_accounts,
                    cap_violation_tol=req.cap_violation_tol)
            except ValueError:
                continue  # celda sin paths resueltos (mismo criterio que 1D)
            reports_2d[(float(f_c), float(f_f))] = firm_report_pair(
                float(f_c), float(f_f), oc_cell, seq)
            cells[(float(f_c), float(f_f))] = (
                ev_vector_all(oc_cell, cfg, fx_eur_usd=req.fx_eur_usd),
                oc_cell.resolved)

    if not reports_2d:
        raise HTTPException(422, "Ninguna celda del par produjo paths resueltos.")
    pair_rec = recommend_f_pair(reports_2d, cells, req.tol_neg)

    # Sensibilidad al cap: 4 llamadas sequence_ev en las 2 celdas de referencia.
    cap_sens = []
    if pair_rec is not None:
        cap10 = min(10, cfg.funded.max_payout_cycles)
        keys = {(pair_rec.f_challenge, pair_rec.f_funded): "par recomendado",
                (pair_rec.f_single, pair_rec.f_single): "mejor f único (diagonal)"}
        for key, label in keys.items():
            oc_cell = combine_challenge_funded(challenge_by_fc[key[0]],
                                               funded_by_fc[key[0]][key[1]])
            s_cap = sequence_ev(oc_cell, cfg, fx_eur_usd=req.fx_eur_usd,
                                n_challenges=req.n_challenges,
                                n_sequences=req.n_sequences,
                                initial_capital_eur=req.capital0_eur, seed=0,
                                stop_on_insolvency=req.stop_on_insolvency,
                                n_copied_accounts=req.n_copied_accounts,
                                cap_violation_tol=req.cap_violation_tol)
            s_c10 = sequence_ev(oc_cell, cfg, fx_eur_usd=req.fx_eur_usd,
                                n_challenges=req.n_challenges,
                                n_sequences=req.n_sequences,
                                initial_capital_eur=req.capital0_eur, seed=0,
                                cap_cycles=cap10,
                                stop_on_insolvency=req.stop_on_insolvency,
                                n_copied_accounts=req.n_copied_accounts,
                                cap_violation_tol=req.cap_violation_tol)
            cap_sens.append({
                "celda": label, "f_challenge": key[0], "f_funded": key[1],
                "ev_cap_max": s_cap.ev_per_attempt,
                "ev_cap10": s_c10.ev_per_attempt,
                "vivo_al_cap": s_cap.alive_at_cap_frac})

    meta = {
        "firm": cfg.display_name, "account_size": cfg.account_size,
        "fee_eur": cfg.fee, "legs": leg_rows,
        "n_blocks": ctx["n_blocks"], "coverage": ctx["coverage"],
        "n_no_valley": ctx["n_no_valley"], "pnl_w_no_valley": ctx["pnl_w_nv"],
        "block_size": ctx["bs"], "float_mode": ctx["float_mode"],
        "sizing": _sizing_label(ctx["legs_used"]),
        # D1 (csv-d1-d2): eco de la base del lote (tamaño de cuenta, sin componer).
        "sizing_base": _sizing_base(ctx["kind"]),
        # F2c: eco de ventana y eje (mismo convenio que el 1D).
        "oos_start": req.oos_start,
        "trades_in_window": ctx["n_window"],
        "day_axis": "et_session" if et_session else req.day_axis,
        **({"day_axis_note": (
            "eje servidor AXI (00:00 EET/EEST): el daily loss de FTMO resetea a "
            "00:00 CE(S)T = 01:00 servidor; a este eje P(pass) queda ~1-5pp "
            "optimista — usar solo como comparación.")}
           if req.day_axis == "servidor" and not et_session else {}),
        # Fase 6 (Apex): eco del perfil extendido (mismo convenio que el 1D).
        "ruleset_version": cfg.ruleset_version or None,
        "source_tz": ctx["source_tz"],
        "trailing": trailing,
        "n_copied_accounts": req.n_copied_accounts,
        "cap_violation_tol": req.cap_violation_tol,
        "sizing_invalid_any": any(r.sizing_invalid for r in reports_2d.values()),
        "grid_n": n_f_grid, "cells_resolved": len(reports_2d),
        "n_paths": req.n_paths, "elapsed_s": time.perf_counter() - t0,
    }
    if ctx["kind"] == KIND_SQX:
        meta.update(_sqx_meta_block(ctx, leg_rows))
    # D2 (csv-d1-d2, fase 2): nota del flotante y contadores de trades sin OHLC M1 (cartera CSV en modo minute).
    meta.update(_csv_minute_meta(ctx))
    # onboarding-b1: contexto de modelo (aditivo; sin f* único no hay
    # ``pct_cortadas_datos``, así que esa nota no se rellena en el par).
    meta["model_context"] = _model_context(
        lente="propfirm", blocks=ctx["blocks"], n_blocks=ctx["n_blocks"],
        bs=ctx["bs"], bs_mode=ctx["bs_mode"], bs_info=ctx["bs_info"], cfg=cfg)
    cells_out = [
        {"f_challenge": k[0], "f_funded": k[1], **dataclasses.asdict(r)}
        for k, r in sorted(reports_2d.items())
    ]
    return _with_profile_snapshot(_jsonable({
        "meta": meta, "warnings": warnings, "cells": cells_out,
        "recommendation": pair_rec, "cap_sensitivity": cap_sens,
    }), ctx)


# D2 (csv-d1-d2, fase 2): textos de la lente AXI con el flotante por MINUTO (``AXI_FLOAT_MODE == "minute"``). Con
# ``"joint"`` se emiten los de antes (literales en ``simulate_axi``, sin tocar).
# PLAN-DESVIO: el brief solo cambia ``meta.float_convention``. ``policy_assumption`` describía el flotante «JOINT (F5b)»
# (min diario de la suma de flotantes vivas; trades sin M1 aportan 0) y la interfaz lo muestra junto a la convención
# («Suposición de política»): dejarlo así contradiría a ``float_convention``. Con "minute" se emite este texto; con
# "joint" el literal de siempre.
_AXI_FLOAT_CONVENTION_MINUTE = (
    "flotante por minuto con el realizado intradía (decisiones); raw y first_cross en bound")
_AXI_POLICY_MINUTE = (
    "F4 regla doble sobre flotante por MINUTO con el realizado intradía (D2): ruin al primer toque "
    "de max_loss y pass el primer día con equity ≥ target ∧ T(d) ≥ "
    "min_trades, ambos sobre la serie merged cuyo peor punto de cada día suma "
    "lo ya cerrado ese día y el flotante M1 de las posiciones abiertas, minuto "
    "a minuto (ffill físico entre barras; la envolvente de cada trade termina "
    "en su minuto de cierre, donde entra el realizado; los trades sin cobertura "
    "M1 no entran en los bloques, ver coverage); si falta T, continuación a f "
    "CONSTANTE hasta cumplirlo (sobrestima 2a / subestima 2b respecto al juego "
    "óptimo con reducción de riesgo); D (min_days) se cumple esperando "
    "plano (gratis) y solo desplaza la elegibilidad, nunca gatea el "
    "pass. T(d) cuenta trades COLOCADOS por fecha de apertura. Orden "
    "de chequeos por día: ruin→pass→timeout; agotar horizon_days con "
    "T bajo = timeout (sin resolver). Los campos raw (primer cruce, "
    "ancla de fidelidad) se mantienen en la convención bound "
    "(peor-punto-por-día, referencia motor): la convención por minuto y bound "
    "pueden diferir en cualquier sentido según el día — ninguna comparación "
    "raw↔F4 implica dirección sistemática. Evaluación por día (el peor minuto "
    "del día, con el realizado, entra en la serie; no se evalúa minuto a "
    "minuto). Puertas no modeladas: Edge Score ≥60, $1.000 mínimo "
    "antes de Next Stage, cierre previo de posiciones.")
# PLAN-DESVIO (menor, fase F3): el brief no toca ``policy_assumption``. Con patas ``.sqx`` TODAS las operaciones entran en
# los bloques (la serie de cada una es su OHLC M1 o su cota diaria por el MAE), así que la frase «los trades sin cobertura M1
# no entran en los bloques» sería falsa en la interfaz («Suposición de política»); solo cambia esa cláusula.
_AXI_POLICY_MINUTE_SQX = _AXI_POLICY_MINUTE.replace(
    "los trades sin cobertura M1 no entran en los bloques, ver coverage",
    "las operaciones sin barras M1 usan su cota diaria por el MAE y entran en los bloques, ver float_note")


# ---------------------------------------------------------------------------
# Lente AXI con patas .sqx (guardrail entrada-4-bloques, fase F3; plan §2.7 y enmiendas E1 y E19)
# ---------------------------------------------------------------------------
def _axi_sqx_admission(pf: dict) -> None:
    """409 si una cartera de patas ``.sqx`` no es del bróker de la lente AXI (``config/brokers.yaml → lenses.axi``).

    Sustituye a :func:`_reject_sqx_portfolio` SOLO en esta lente (el par 2D y el diagnóstico por pata siguen con su
    409). El detalle lleva «.sqx», «AXI» y el bróker de las patas (los tests de FTMO fijan «.sqx»). Una cartera sin
    bróker (ninguna pata con ficha cargada) no se rechaza aquí: la rechaza el «sin patas válidas» del endpoint."""
    lens_id = fichas.lens_broker("axi")
    pf_broker = pf.get("broker")
    if pf_broker is None or pf_broker == lens_id:
        return
    brokers = fichas.load_registry().brokers
    label = lambda bid: _broker_label(brokers[bid]) if bid in brokers else str(bid)        # noqa: E731
    if lens_id is None:
        raise HTTPException(
            409, "La lente AXI no tiene bróker asignado en config/brokers.yaml (lenses.axi): no puede admitir carteras "
            f"de patas .sqx (las patas de esta cartera son de {label(pf_broker)}).")
    raise HTTPException(
        409, f"La lente AXI solo admite carteras de patas .sqx del bróker de AXI ({label(lens_id)}); las patas de esta "
        f"cartera son de {label(pf_broker)}. Re-testea las estrategias en SQX con los datos de AXI.")


def _daily_valley(ts, vals) -> dict:
    """Valle diario de UNA operación desde su serie por minuto (E1): ``{fecha del servidor: mínimo de vals de esa
    fecha}``, la misma agrupación que ``precompute_strategy`` (``groupby(s.index.date).min()``).

    Las claves son ``datetime.date`` (``pd.DatetimeIndex(ts).date``, el mismo tipo que ``close_d`` de la
    ``StrategyMC``) y los valores ``float`` en unidades de PRECIO: excursiones netas de comisión, swap y spread en
    cortos (como la serie ``.sqx`` de la prop firm; más conservadoras que el valle bruto del CSV). E21: el valor es el
    MÍNIMO de la excursión del día y puede ser > 0 si la operación estuvo en beneficio todo el día, igual que el valle
    CSV (``precompute_strategy``); "≤ 0" no es un invariante y no se recorta."""
    idx = pd.DatetimeIndex(ts)
    g = pd.Series(np.asarray(vals, dtype=float), index=idx).groupby(idx.date).min()
    return {d: float(v) for d, v in g.items()}


def _axi_sqx_strats(pf: dict, f_grid: np.ndarray, symbols: dict,
                    diag_out: dict | None = None) -> tuple[list[dict], dict]:
    """``(patas recargadas, strats)`` de una cartera ``.sqx`` para la lente AXI (plan §2.7).

    ``diag_out`` (F4, opcional): si se pasa, se rellena ``{nombre de pata: diag}`` con el ``diag`` por f de
    :func:`_lots_for_leg` (``frac_clamp_up``, ``p95_risk_eff_over_X``…), que consume la receta para el EA.

    - Las patas se RECARGAN con ``with_minutes=True`` y ``synth_mode="daily"`` en el eje del servidor (identidad: el día
      de la lente AXI es la fecha del servidor), igual que ``_propfirm_setup`` recarga las suyas. Con la caché de patas,
      si la carga (``portfolio_load``) ya las leyó así, es un acierto.
    - ``valle[i]`` = :func:`_daily_valley` de la serie de la operación ``i`` (toda operación tiene entrada: sin barras
      tiene su cota diaria; una pata sin OHLC coherente es entera de la cota diaria). Se construye sobre una COPIA de
      la ``StrategyMC`` (``dataclasses.replace``): la cacheada se queda con ``valle == {}`` (I6: la comparte la lente
      prop firm). Así ``build_block_records`` no pierde ninguna operación (no hay D3 en las patas ``.sqx``).
    - Lotes por :func:`_lots_for_leg` (riesgo fijo ``X = f·A`` hasta el SL inicial, ``A`` = tamaño de cuenta de la
      carga, sin tope) sobre la copia.

    El orden de ``strats`` es el de las patas (= el de ``lots_by``/``pv_by`` y el de ``sym_id`` de los bloques)."""
    legs_used: list[dict] = []
    for leg in pf["legs"]:
        try:
            legs_used.append(_load_leg(leg["path"], pf["account_size"], pf["f_ref"],
                                       with_minutes=True, synth_mode="daily"))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, f"Pata '{leg['name']}' no recargable para la lente AXI: {exc}")
    names = [l["name"] for l in legs_used]
    if len(set(names)) != len(names):
        raise HTTPException(409, f"Patas .sqx con el mismo nombre de fichero ({names}): renómbralas.")
    if any(l.get("broker") != pf.get("broker") for l in legs_used):
        raise HTTPException(409, "Las fichas de símbolo cambiaron desde la carga y las patas ya no son del bróker de "
                                 "la cartera; recarga la cartera.")
    strats: dict = {}
    for leg in legs_used:
        mv = leg.get("minute_valle")
        if mv is None:
            raise HTTPException(409, f"La pata '{leg['name']}' no tiene serie por minuto ni cota diaria: "
                                     f"{leg.get('minute_reason')}")
        s = leg["s"]
        valle = {i: _daily_valley(mv[i][0], mv[i][1]) for i in range(s.n_trades)}
        lots, diag = _lots_for_leg(leg, f_grid, symbols, pf["f_ref"], account_size=pf["account_size"], cap=None)
        if diag_out is not None:
            diag_out[leg["name"]] = diag
        strats[leg["name"]] = attach_lots(dataclasses.replace(s, valle=valle), lots)
    return legs_used, strats


def _axi_sqx_float_note(legs_used: list[dict], float_mode: str) -> str:
    """``meta.float_note`` de la lente AXI con patas ``.sqx``: cómo se calcula el flotante y CUÁNTAS patas (y por qué)
    usan la cota diaria por el MAE en vez del OHLC M1 (``minute_reason`` no nulo con ``synth_mode="daily"``), más las
    operaciones sin barras dentro de las patas que sí tienen OHLC y, desde E20, de dónde sale el spread de la serie (el
    del backtest o el de la ficha)."""
    n = len(legs_used)
    daily = [l for l in legs_used if l.get("minute_reason")]
    n_ops = sum(int((l.get("minute_diag") or {}).get("sinteticas", 0)) for l in legs_used if not l.get("minute_reason"))
    if float_mode == "minute":
        txt = ("flotante por minuto (OHLC M1 bid de cada pata, hora del servidor): COTA por minuto — el peor precio de "
               "cada minuto para todas las posiciones a la vez —, con el realizado intradía dentro del peor momento "
               "del día, NETO de comisión, swap que es cargo y spread en los cortos, y acotado por el MAE de SQX (la "
               "magnitud por operación no pasa de su MAE + costes). ")
    else:
        txt = ("flotante conjunto M1 sin el realizado intradía (modo joint, A/B de investigación), NETO de comisión, "
               "swap que es cargo y spread en los cortos, y acotado por el MAE de SQX. ")
    if daily:
        why = "; ".join(f"{l['name']} ({(l.get('minute_diag') or {}).get('motivo_sin_serie') or l['minute_reason']})"
                        for l in daily)
        txt += (f"{len(daily)} de {n} pata(s) usan la cota diaria por el MAE porque no tienen OHLC M1 coherente: "
                f"{why}. ")
    else:
        txt += f"Las {n} pata(s) tienen OHLC M1 coherente. "
    # E20: de dónde sale el spread de la serie (coherencia con el OHLC, ajuste de los cortos y suelo de la cota diaria)
    srcs = [(l.get("minute_diag") or {}).get("spread_fuente") for l in legs_used]
    n_sqx, n_ficha = srcs.count("sqx"), srcs.count("ficha")
    if n_sqx or n_ficha:
        parts = ([f"{n_sqx} de {n} pata(s) usan el del backtest (defaultSpread del .sqx)"] if n_sqx else []) + (
            [f"{n_ficha} de {n} el de la ficha"] if n_ficha else [])
        txt += f"Spread de la serie (coherencia con el OHLC, cortos y cota diaria): {'; '.join(parts)}. "
    if n_ops:
        txt += (f"{n_ops} operación(es) sin barras M1 dentro de patas con OHLC usan también la cota diaria por el MAE "
                "(un punto por cada día que abarcan, con el peor estado de la operación). ")
    return txt + ("La cota diaria por el MAE es más pesimista que el minuto: cuenta el peor estado de la operación entera "
                  "en cada día que abarca. No es la equity exacta.")


@app.post("/api/simulate/axi")
def simulate_axi(req: AxiSimRequest):
    """Lente AXI (ruina) sobre una cartera cargada — F4 + F5b (joint).

    Réplica de la receta del modo portfolio de Streamlit
    (``risk_app/portfolio_mode.py``): barreras directas de la etapa
    (``baseline = allocation_at(5000)``, ``pass_line = target%``,
    ``ruin_line = max_loss%``), sizing por pata vía ``_leg_lots`` (exacto si la
    carga lo resolvió, F5a) y, si ``block_size <= 0``, block_size automático de Politis-White sobre la serie en R de los
    bloques (:func:`_auto_block_size`; guardrail bloque-politis-white: ya no depende de f_ref ni de la rejilla). Con un
    bloque principal > 1 la meta añade ``bs1`` (la sensibilidad con bloque 1 en f*, :func:`_axi_bs1`).

    BUCLE TRANSPARENTE (§4.1): en vez de la llamada opaca a
    ``run_portfolio_mc`` se replica su bucle con piezas públicas del motor
    (``detect_flatflat_blocks`` → ``build_block_records`` →
    ``stationary_block_indices`` con ``default_rng(42)``, la seed por defecto
    de ``run_portfolio_mc``/``sweep_f_portfolio``; un ``idx_matrix`` compartido
    por toda la grilla f) y se evalúa con el WALKER F4 en modo JOINT
    (``axi_path.evaluate_path_axi``, ``mtm="joint"``; §8.6-1 de
    PROPUESTA_F5): regla doble E∧T(d)-apertura con continuación a f constante
    sobre la serie merged con flotante conjunto M1. Los campos RAW (primer
    cruce, semántica del motor) salen del sub-pase bound del propio walker
    (``axi_path.raw_outcome``) y se verifican con tolerancia cero contra
    ``run_portfolio_mc`` en ``check_axi.py``.

    D2 (csv-d1-d2, fase 2): con ``AXI_FLOAT_MODE == "minute"`` (por defecto) la serie merged precomputada que
    consume el walker es la del flotante POR MINUTO con el realizado intradía (``minute_columns_cache`` +
    ``precompute_minute`` en lugar de ``joint_columns_cache`` + ``precompute_joint``); el walker, el sub-pase
    bound y el resto no cambian. Con ``"joint"`` el camino de antes, bit a bit.

    SUELOS (§3.4, F4): el suelo de trades ya no reclasifica a posteriori — el
    walker continúa el camino hasta cumplir T o resolver (ruin/fin de datos),
    destapando las ruinas 2a y recuperando los passes 2b (reparto expuesto en
    ``post_target`` por f). El suelo de días NO descuenta probabilidad, solo
    desplaza calendario: días-elegibilidad = máx(``days_cal``, min_days) sobre
    passes cualificados, con ``days_cal`` = estimación calendario etiquetada
    (busy + huecos medios; ver ``meta.days_cal``). ``recommend_f`` se alimenta
    de los reports F4. maxDD/terminales se reportan RAW (primer cruce, bound)
    como ancla de fidelidad; el walker F4 los des-censura internamente para
    los paths del cubo caso-2.

    Carteras de patas ``.sqx`` (guardrail entrada-4-bloques, fase F3; plan §2.7 y enmiendas E1/E19): solo las del bróker
    de la lente (``lenses.axi`` de ``config/brokers.yaml``; otro bróker = 409). Las patas se recargan con ``with_minutes``
    y ``synth_mode="daily"`` en el eje del servidor; el valle diario de cada operación sale de su serie por minuto
    (``valle[i] = {fecha: min(serie de esa fecha)}``) y se pone en una COPIA de la ``StrategyMC`` (la cacheada la
    comparte la lente prop firm y sigue con ``valle == {}``); los lotes son el riesgo fijo ``X = f·A`` hasta el SL
    inicial (:func:`_lots_for_leg`); el flotante por minuto usa la serie de cada pata (las operaciones sin barras y las
    patas sin OHLC coherente, su cota diaria por el MAE). Las barreras de etapa, el walker y la semilla son los de siempre.
    ``_csv_minute_series`` y ``_mae_usage`` no se llaman con ``.sqx``.

    Fase F4: con una cartera ``.sqx`` la meta añade ``recipe`` (:func:`_sqx_recipe`, receta para el EA con ``A`` =
    ``pf["account_size"]``) y los ecos por pata (``meta.legs``, ``meta.sizing``) salen de las patas RECARGADAS (H4 de la
    revisión de código de la F3), no de las de ``portfolio_load``.
    """
    t0 = time.perf_counter()
    pf = _portfolios.get(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, f"portfolio_id '{req.portfolio_id}' desconocido: "
                             "carga la cartera primero (POST /api/portfolio/load).")
    is_sqx = _is_sqx_portfolio(pf)
    if is_sqx:
        _axi_sqx_admission(pf)            # F3: solo el bróker de la lente (antes: 409 para cualquier .sqx)
    if not pf["legs"]:
        raise HTTPException(409, "La cartera no tiene patas válidas; recarga.")
    if not is_sqx:
        bad = [l["name"] for l in pf["legs"] if not l["s"].minute_valle]
        if bad:
            raise HTTPException(409, "La cartera tiene patas sin minute_valle usable "
                                "(sin M1 o sin trades con cobertura); recarga.")
    phase, baseline, pass_line, ruin_line = _stage_barriers(req.stage)
    warnings: list[str] = []

    symbols = {s.ticker_axi: s for s in _broker.symbols}
    f_grid = np.arange(req.f_lo, req.f_hi + req.f_step / 2, req.f_step)
    if len(f_grid) < 2:
        raise HTTPException(400, "Grilla de f inválida (mínimo 2 puntos).")

    diag_by: dict[str, list[dict]] = {}      # F4: diag por f de los lotes de cada pata .sqx (receta para el EA)
    if is_sqx:
        legs_used, strats = _axi_sqx_strats(pf, f_grid, symbols, diag_by)
        daily_legs = [l for l in legs_used if l.get("minute_reason")]
        if daily_legs:
            warnings.append(
                "Flotante por la cota diaria por el MAE (más pesimista): "
                f"{len(daily_legs)}/{len(legs_used)} pata(s) sin OHLC M1 coherente — "
                + "; ".join(f"{l['name']}: {l['minute_reason']}" for l in daily_legs))
    else:
        legs_used = pf["legs"]
        strats = _strats_with_lots(pf, f_grid, symbols)

    # --- Corte IS/OOS (F2c-1): descarte de bloques previos al corte ----------
    try:
        oos_ts = pd.Timestamp(req.oos_start) if req.oos_start else None
    except ValueError as exc:
        raise HTTPException(400, f"oos_start inválido (se esperaba ISO): {exc}")
    if oos_ts is None:
        # Regla M2-8: evaluar sobre el IS infla P(pass) si hubo selección ahí.
        warnings.append(
            "Ventana completa (sin corte IS/OOS): P(pass) inflada si las "
            "estrategias se seleccionaron sobre este tramo (regla M2-8); "
            "elige campaña en el selector de corte.")

    # --- Bloques flat-flat (una vez; sirven para el bloque automático y para el MC) ---
    # Misma construcción que run_portfolio_mc (mc_portfolio.py:567-572):
    # lots_by/pv_by por sym_id en orden de inserción; F2c-1: oos_start de la
    # request descarta bloques enteros previos al corte (misma semántica que
    # build_day_blocks_portfolio en el lente prop firm).
    lots_by = {si: s.lots for si, s in enumerate(strats.values())}
    pv_by = {si: s.pv for si, s in enumerate(strats.values())}
    oc = {k: (s.open_t, s.close_t) for k, s in strats.items()}
    blocks_all = detect_flatflat_blocks(oc)
    n_cross = count_crossing(strats, blocks_all)
    blocks = build_block_records(strats, blocks_all, oos_start=oos_ts)
    if not blocks:
        raise HTTPException(422, "No hay bloques flat-flat en la ventana "
                            "(¿corte IS/OOS posterior al último trade?).")
    if len(blocks) < 30:
        warnings.append(f"n_blocks = {len(blocks)} < 30: bootstrap "
                        "degenerado (M2-11).")
    if n_cross:
        warnings.append(f"count_crossing = {n_cross} (debe ser 0).")

    # --- block_size automático (guardrail bloque-politis-white): Politis-White sobre la serie en R de EXACTAMENTE estos
    # ``blocks`` (los de la ventana ``oos_start``), con las patas en el orden de ``strats``. Ya no depende de f_ref ni de
    # la rejilla. Manual si ``block_size > 0``.
    bs = float(req.block_size)
    bs_mode, bs_info = "manual", None   # eco para meta.model_context
    if bs <= 0:
        bs, bs_mode, bs_info, bs_warns = _auto_block_size(blocks, _legs_in_strats_order(legs_used, strats), pf)
        warnings.extend(bs_warns)

    # --- Bucle transparente F4: misma idx_matrix que sweep_f_portfolio -------
    # (mc_portfolio.py:527-535): un rng=default_rng(42) → idx_matrix común a
    # toda la grilla; la evaluación por path y por f la hace el WALKER F4
    # (axi_path.evaluate_path_axi): regla doble E≥target ∧ T(d)-apertura ≥ min,
    # continuación a f constante si falta T. open_days (T por apertura) y ḡ
    # (hueco medio inter-bloque, runtime) se precomputan UNA vez.
    # F5b (§8.6-1): el walker expuesto es 100% JOINT (mtm="joint"): columnas
    # M1 f-indep una vez por simulación (cols_cache; NO se persisten en
    # _portfolios, §8.6-9) y merged0 por (bloque, f) precomputado;
    # first_cross/fc_*/raw siguen en bound (sub-pase motor del walker).
    # D2 (csv-d1-d2, fase 2; plan §2.3, E6-E8): con AXI_FLOAT_MODE == "minute" la serie merged0 por (bloque, f)
    # es la del flotante POR MINUTO con el realizado intradía (axi_path.precompute_minute, mismo formato que
    # precompute_joint: el walker no cambia). ``minute_by_sym`` = serie propia de cada trade + una serie sintética
    # con el MAE para los que no tengan M1 (en los bloques AXI no hay ninguno hoy: ``build_block_records`` solo
    # incluye trades con valle, así que los contadores salen a 0; se calculan igual para que el aviso no dependa de eso).
    float_mode_axi = _check_float_flag("AXI_FLOAT_MODE", AXI_FLOAT_MODE)
    min_trades = phase.min_unique_trades
    min_days = phase.min_days
    open_days = precompute_open_days(blocks, strats)
    gap_mean = mean_inter_block_gap(blocks)
    mae_usage = None
    if is_sqx:
        # F3: ``minute_by_sym`` = la serie de cada pata en el orden de ``strats`` (contrato E7 de ``minute_columns_cache``);
        # toda operación tiene la suya (barras M1 o cota diaria por el MAE), así que ni ``_csv_minute_series`` ni
        # ``_mae_usage`` (solo CSV) se llaman. Las series viven en la pata, no en ``StrategyMC.minute_valle``.
        minute_by_sym = {i: leg["minute_valle"] for i, leg in enumerate(legs_used)}
        if float_mode_axi == "minute":
            cols_cache = minute_columns_cache(blocks, list(strats.values()), minute_by_sym)
        else:
            cols_cache = joint_columns_cache(blocks, minute_by_sym)
    elif float_mode_axi == "minute":
        # mismo orden que ``strats``/``lots_by``/``pv_by`` (E7): ``strategies = list(strats.values())``
        legs_ord = _legs_in_strats_order(pf["legs"], strats)
        minute_by_sym = {i: _csv_minute_series(leg)[0] for i, leg in enumerate(legs_ord)}
        cols_cache = minute_columns_cache(blocks, list(strats.values()), minute_by_sym)
        mae_usage = _mae_usage(blocks, legs_ord, lots_by, len(f_grid) // 2)
        warnings.extend(_mae_warnings(mae_usage))
    else:
        minute_by_sym = {i: s.minute_valle for i, s in enumerate(strats.values())}
        cols_cache = joint_columns_cache(blocks, minute_by_sym)
    idx_matrix = stationary_block_indices(len(blocks), req.n_paths, bs, np.random.default_rng(_AXI_SEED))

    def axi_outs_at(fi: int, idx: np.ndarray) -> list:
        """Caminos de la f ``fi`` de la rejilla con la matriz de índices ``idx``: la de la petición en el barrido y la de
        bloque 1 en la sensibilidad (:func:`_axi_bs1`). Una sola evaluación para las dos (guardrail bloque-politis-white, E4)."""
        if float_mode_axi == "minute":
            joint_pre = precompute_minute(blocks, lots_by, pv_by, fi, cols_cache)
        else:
            joint_pre = precompute_joint(blocks, lots_by, pv_by, fi,
                                         minute_by_sym, cols_cache)
        return [
            evaluate_path_axi(idx[p], blocks, open_days, lots_by, pv_by,
                              fi, pass_line=pass_line, ruin_line=ruin_line,
                              baseline=baseline, min_unique_trades=min_trades,
                              gap_mean=gap_mean, horizon_days=req.horizon_days,
                              mtm="joint", joint_pre=joint_pre)
            for p in range(req.n_paths)
        ]

    axi_by_f: list[list] = [axi_outs_at(fi, idx_matrix) for fi in range(len(f_grid))]

    # --- Métricas F4 por f: raw derivado de first_cross (V4) + reparto 2a/2b/2c
    reports: list[dict] = []
    qual_reports = []  # RiskReport F4 (alimentan recommend_f)
    for fi, f in enumerate(f_grid):
        outs = axi_by_f[fi]
        n_paths = len(outs)
        # RAW = semántica del motor (primer cruce), derivada del snapshot
        # first_cross sin recomputar; ancla de fidelidad tolerancia cero.
        raw = [raw_outcome(o) for o in outs]
        base = aggregate_outcomes(raw, float(f), "block", bs, req.n_paths,
                                  req.horizon_days)
        # F4: outcomes post-continuación (trades = T(d)-apertura en el stop).
        qrep = _axi_qual_report(outs, f, bs, req.n_paths, req.horizon_days)
        # Pass ya cualificado EN el primer toque (T(d) ≥ suelo): ancla de la
        # partición V3 (n_deg = n·(p_pass_raw − p_pass_suelo)).
        n_suelo = sum(1 for o in outs
                      if o.first_cross == "pass"
                      and (min_trades is None or o.fc_trades_day >= min_trades))
        post_target = post_target_split(outs, min_trades)
        # Días-elegibilidad = máx(days_cal, min_days) y trades, sobre passes
        # cualificados F4 (si no hay, null → la UI muestra "—"). days_cal es
        # estimación calendario etiquetada (ver meta.days_cal).
        days_q = np.array([
            (max(o.days_cal, min_days) if min_days is not None else o.days_cal)
            for o in outs if o.outcome == "pass"], dtype=float)
        trades_q = np.array([o.trades for o in outs if o.outcome == "pass"],
                            dtype=float)
        pct = lambda a, q: float(np.nanpercentile(a, q)) if len(a) \
            else float("nan")
        elig_p50, elig_p90 = pct(days_q, 50), pct(days_q, 90)
        tq_p50, tq_p90 = pct(trades_q, 50), pct(trades_q, 90)
        reports.append({
            "f": qrep.f, "n_paths": qrep.n_paths,
            # P(pass)/P(ruin) F4 (post-continuación: 2b recuperado, 2a
            # destapado) + anclas raw (primer cruce, semántica motor).
            "p_pass": qrep.p_pass, "p_pass_ci": list(qrep.p_pass_ci),
            "p_pass_raw": base.p_pass, "p_pass_raw_ci": list(base.p_pass_ci),
            "p_pass_suelo": n_suelo / n_paths,
            "p_ruin": qrep.p_ruin, "p_ruin_ci": list(qrep.p_ruin_ci),
            "p_ruin_raw": base.p_ruin, "p_ruin_raw_ci": list(base.p_ruin_ci),
            # "sin resolver" F4 = neither + timeout post-continuación
            # (§4.1: pass+ruin+sin_resolver = 1; p_blown no se muestra).
            "p_sin_resolver": 1.0 - qrep.p_pass - qrep.p_ruin,
            # Reparto del cubo degradado (casos 2a/2b/2c de §4 del análisis).
            "post_target": post_target,
            "median_days_eligibility": elig_p50,
            "p90_days_eligibility": elig_p90,
            "median_trades_qual": tq_p50, "p90_trades_qual": tq_p90,
            # Campos raw (primer cruce), compat F2 + fidelidad del bucle.
            "median_days_to_target": base.median_days_to_target,
            "p90_days_to_target": base.p90_days_to_target,
            "median_trades_to_target": base.median_trades_to_target,
            "p90_trades_to_target": base.p90_trades_to_target,
            "median_max_dd": base.median_max_dd, "p10_max_dd": base.p10_max_dd,
            "terminal_p05": base.terminal_p05, "terminal_p50": base.terminal_p50,
            "terminal_p95": base.terminal_p95,
        })
        qual_reports.append(dataclasses.replace(
            qrep, median_days_to_target=elig_p50, p90_days_to_target=elig_p90))

    rec = recommend_f(qual_reports, tolerance=req.tol_ruin)
    f_star = rec.f if rec is not None else reports[len(reports) // 2]["f"]

    # Sensibilidad con bloque 1 en f* (guardrail bloque-politis-white, E4/E15/E16): solo con un bloque principal > 1 (modo
    # automático o manual). Se calcula DESPUÉS de f* y no toca nada de lo ya calculado (E19).
    bs1 = None
    if _bs1_wanted(bs):
        bs1, bs1_warns = _axi_bs1(axi_outs_at, f_grid, f_star, len(blocks), req)
        if rec is not None:   # revisión de código de la fase A, H3: sin recomendación, f* no es una f recomendada
            warnings.extend(bs1_warns)

    if oos_ts is not None:
        n_window = int(sum(int(np.sum(pd.to_datetime(s.open_t) >= oos_ts))
                           for s in strats.values()))
    else:
        n_window = int(sum(s.n_trades for s in strats.values()))
    n_in_blocks = int(sum(b.n_unique_trades for b in blocks))
    policy_minute = _AXI_POLICY_MINUTE_SQX if is_sqx else _AXI_POLICY_MINUTE
    meta = {
        "etapa": phase.name, "etapa_display": phase.display_name,
        "min_days": min_days,
        "min_unique_trades": min_trades,
        "floors_applied": {"min_days": min_days,
                           "min_unique_trades": min_trades},
        "semantics": "F4",
        # F5b (§8.6-6): convención expuesta única; raw/ancla en bound.
        "float_convention": _AXI_FLOAT_CONVENTION_MINUTE if float_mode_axi == "minute" else (
            "joint-m1 en las decisiones (min diario de la suma de flotantes "
            "M1 de las posiciones vivas); first_cross/raw en bound "
            "(peor-punto-por-día, referencia motor)"),
        # Texto A2 (§10.5): política de continuación y convenciones.
        "policy_assumption": policy_minute if float_mode_axi == "minute" else (
            "F4 regla doble sobre flotante JOINT (F5b): ruin al primer toque "
            "de max_loss y pass el primer día con equity ≥ target ∧ T(d) ≥ "
            "min_trades, ambos sobre la serie merged con flotante conjunto "
            "intradía (min diario de la suma de flotantes M1 de las posiciones "
            "vivas, ffill físico dentro de [open, close]; trades sin cobertura "
            "M1 aportan 0, BAJO-3); si falta T, continuación a f CONSTANTE "
            "hasta cumplirlo (sobrestima 2a / subestima 2b respecto al juego "
            "óptimo con reducción de riesgo); D (min_days) se cumple esperando "
            "plano (gratis) y solo desplaza la elegibilidad, nunca gatea el "
            "pass. T(d) cuenta trades COLOCADOS por fecha de apertura. Orden "
            "de chequeos por día: ruin→pass→timeout; agotar horizon_days con "
            "T bajo = timeout (sin resolver). Los campos raw (primer cruce, "
            "ancla de fidelidad) se mantienen en la convención bound "
            "(peor-punto-por-día, referencia motor): bound y joint difieren "
            "en AMBOS sentidos según el día (§8.2) — ninguna comparación "
            "raw↔F4 implica dirección sistemática. Evaluación por día (el min "
            "intradía entra vía el flotante del día; no se evalúa minuto a "
            "minuto). Puertas no modeladas: Edge Score ≥60, $1.000 mínimo "
            "antes de Next Stage, cierre previo de posiciones."),
        # §8.3-D2/§8.6-6: desplazamiento del consejo de riesgo respecto al
        # histórico rápido+bound.
        "f_star_shift_note": (
            "El sizing exacto, el flotante intradía conjunto (por minuto, con lo "
            "cerrado en el día, desde el 2026-10-19) y el lote sobre el tamaño de "
            "cuenta (sin componer, desde el 2026-10-19) cambian las probabilidades "
            "respecto a corridas anteriores ⇒ el f* NO es comparable con ellas "
            "(desplazamiento sistemático, no ruido; recalibrar la tolerancia es "
            "decisión del usuario)."),
        # A3 (§10.5): estimación calendario de la edad de etapa, etiquetada.
        "days_cal": {
            "label": "estimación calendario (no exacta)",
            "formula": "days_busy + (n_bloques_consumidos − 1) · gap_mean",
            "gap_mean_days": gap_mean,
            "gap_mean_source": "(span − Σ duration_days) / (n_bloques − 1), "
                               "derivado del dato en runtime",
            "nota": "la edad simulada arranca en el inicio de la ventana de "
                    "datos, no en el alta real de etapa; sobreconteo "
                    "intrabloque ≤ duración del bloque en curso",
        },
        "baseline": baseline, "pass_line": pass_line, "ruin_line": ruin_line,
        "horizon_days": req.horizon_days,
        # F2c-1: eco de la ventana evaluada (diagnóstico/historial etiquetan).
        "oos_start": req.oos_start,
        "trades_in_window": n_window,
        # F3: ``_leg_n_trades`` sirve a las dos clases de pata (CSV: ``tl``; ``.sqx``: sin ``tl``).
        # H4 (revisión de código de la F3): los ecos por pata salen de las patas que SE SIMULARON (``legs_used``: con
        # ``.sqx`` son las RECARGADAS por ``_axi_sqx_strats``, que pueden diferir de las de ``portfolio_load`` si el
        # ``.sqx`` o la ficha cambiaron entre la carga y la simulación); con CSV ``legs_used`` ES ``pf["legs"]``.
        "legs": [{"pata": leg["name"], "símbolo": leg["sym_csv"],
                  "modo": leg["sizing_mode"], "trades": _leg_n_trades(leg)}
                 for leg in legs_used],
        "motor": "bloques flat-flat", "sizing": _sizing_label(legs_used),
        # D1 (csv-d1-d2): eco de la base del lote (tamaño de cuenta, sin componer). Las patas ``.sqx`` ya usaban X = f·A.
        "sizing_base": _sizing_base(KIND_SQX if is_sqx else KIND_CSV),
        "block_size": bs, "bs1": bs1, "n_paths": req.n_paths,
        "n_blocks": len(blocks), "n_blocks_full": len(blocks_all),
        "crossing": n_cross,
        "coverage": (n_in_blocks / n_window) if n_window else float("nan"),
        "elapsed_s": time.perf_counter() - t0,
    }
    if mae_usage is not None:
        # D2 (csv-d1-d2, fase 2; E8): trades de los bloques acotados con su MAE / con valor 0 (y su peso PnL a f central).
        meta.update(mae_usage)
    if is_sqx:
        # F3 (AC3.6): modo de flotante y su nota (cuántas patas usan la cota diaria por el MAE y por qué) y el bróker
        meta.update(float_mode=float_mode_axi, float_note=_axi_sqx_float_note(legs_used, float_mode_axi),
                    broker=pf["broker"])
    # onboarding-b1: contexto de modelo (aditivo).
    meta["model_context"] = _model_context(
        lente="axi", blocks=blocks, n_blocks=len(blocks), bs=bs,
        bs_mode=bs_mode, bs_info=bs_info,
        extra={"min_dias": min_days, "min_trades": min_trades})
    recipe = None
    if is_sqx:
        # F4: receta para el EA (solo carteras .sqx; las CSV no llevan la clave). Con el MISMO array de lotes que entró al
        # motor y el tamaño de cuenta de la cartera con el que se calcularon (``pf["account_size"]``, el que usa
        # ``_axi_sqx_strats``); las patas son las RECARGADAS. Toda la rejilla de f: el front compone la de la f elegida.
        recipe, recipe_warns = _sqx_recipe(legs_used, {n: s.lots for n, s in strats.items()}, diag_by, f_grid,
                                           pf["account_size"], "axi")
        warnings.extend(recipe_warns)
    out = _jsonable({
        "meta": meta, "warnings": warnings,
        "reports": reports, "recommendation": rec,
        "f_star": float(f_star),
    })
    if recipe is not None:
        # DESPUÉS de ``_jsonable`` (que vuelve los booleanos 0/1): la receta ya es JSON puro y conserva los booleanos
        out["meta"]["recipe"] = recipe
    return out


@app.post("/api/portfolio/{portfolio_id}/leg_diagnostics")
def portfolio_leg_diagnostics(portfolio_id: str, req: LegDiagRequest):
    """Diagnóstico por pata BAJO DEMANDA (L2, panel cartera).

    Estático (``leg_diagnostics``: %min_lot ref, riesgo suelo, coste
    implícito/esperado) + P(pass)/P(ruin) de cada pata SOLA @ f dado, con la
    receta de ``portfolio_mode.py:258-292`` (un ``run_portfolio_mc`` por pata,
    seed por defecto, mismas barreras de etapa). El baseline de la etapa se
    necesita para el riesgo suelo, así que TODO el diagnóstico vive aquí (no
    en la carga, que es agnóstica de lente/etapa).
    """
    pf = _portfolios.get(portfolio_id)
    if pf is None:
        raise HTTPException(404, f"portfolio_id '{portfolio_id}' desconocido.")
    _reject_sqx_portfolio(portfolio_id, "El diagnóstico por pata")
    if not pf["legs"]:
        raise HTTPException(409, "La cartera no tiene patas válidas; recarga.")
    phase, baseline, pass_line, ruin_line = _stage_barriers(req.stage)

    symbols = {s.ticker_axi: s for s in _broker.symbols}
    f_arr = np.array([req.f])
    rows = []
    for leg in pf["legs"]:
        diag = leg_diagnostics(leg["tl"], leg["sym"],
                               f_ref=pf["f_ref"], baseline=baseline)
        lots = _leg_lots(leg, f_arr, symbols, pf["f_ref"])
        single = dataclasses.replace(leg["s"], lots=lots)
        p_pass = p_ruin = None
        err = None
        try:
            r1 = run_portfolio_mc(
                {leg["name"]: single}, f_arr,
                n_paths=req.n_paths, block_size=req.block_size, oos_start=None,
                pass_line=pass_line, ruin_line=ruin_line, baseline=baseline)
            p_pass, p_ruin = r1.reports[0].p_pass, r1.reports[0].p_ruin
        except Exception as exc:  # p.ej. bloques < bs en la pata sola
            err = str(exc)
        rows.append({
            "pata": leg["name"], "símbolo": leg["sym_csv"],
            "modo": leg["sizing_mode"],
            "trades": diag.n_trades,
            "frac_min_lot_ref": diag.frac_min_lot_ref,
            "floor_risk_pct": diag.floor_risk_pct,
            "implied_cost_per_lot": diag.implied_cost_per_lot,
            "expected_cost_per_lot": diag.expected_cost_per_lot,
            "p_pass_solo": p_pass, "p_ruin_solo": p_ruin,
            "warnings": list(diag.warnings), "error": err,
            # Campos de régimen/canal de dinero (producto gap-régimen-ATR,
            # piezas 1+2): ADITIVOS, null-tolerant; el trigger compuesto
            # (percentil AND clamp-hoy) NO va en la API — lo compone el
            # frontend.
            **_leg_regimen_fields(leg, req.f, pf["account_size"]),
            "regimen_warns": list(leg.get("regimen_warns", [])),
        })
    return _jsonable({
        "f": req.f, "stage": phase.name, "stage_display": phase.display_name,
        "baseline": baseline, "n_paths": req.n_paths,
        "block_size": req.block_size, "rows": rows,
        # §8.6-6: el MC por pata corre con el motor (bound); las P solas NO se
        # comparan con el lente AXI (joint) sin ese contexto.
        "float_convention": "bound (referencia motor)",
        "account_size": pf["account_size"],
        "threshold_pctile": _THRESHOLD_PCTILE,
        "pending_calibration": False,
        "threshold_source": "curva GAP-ATR-CURVA 2026-10-12",
        "reference_population": (
            "distribución histórica de sl-en-entradas de la pata (no barras "
            "calendario; expanding completa, sin corte OOS)"),
        "hybrid_label": "escenario híbrido, dirección no acotada",
    })


# ---------------------------------------------------------------------------
# Historial de corridas (P4): JSONL append-only, sin trajectory/seq
# ---------------------------------------------------------------------------
class RunIn(BaseModel):
    lens: Literal["propfirm", "axi"]
    portfolio_id: str
    diag_group_id: str | None = None  # enlaza la gemela del diagnóstico (P2)
    request: dict = {}
    summary: dict = {}  # meta + f_star + recommendation + reports; SIN trajectory


@app.post("/api/runs")
def post_run(rec: RunIn):
    """Registra una corrida (el cliente manda el resumen ya recortado)."""
    return append_run(rec.model_dump())


@app.get("/api/runs")
def get_runs(lens: str | None = None, limit: int = 50):
    return {"runs": list_runs(lens=lens, limit=min(limit, 200))}
