"""Tabla de trades de un ``.sqx`` de micro futuros o de CFD FTMO (Fase A del plan
float-valleys-ventanas; CFD añadido el 2026-10-12).

Lee solo ``orders.bin`` y ``settings.xml`` (sin datos de precios) y devuelve, para
el test ``Main``, una fila por trade con lo que necesita el motor de riesgo para
construir una pata de futuros nativa: PnL neto, MAE en USD, precios de
apertura/cierre/SL inicial y comisión implícita.

Anatomía de ``orders.bin`` (registros de 149 B, big-endian; verificada el
2026-10-10 sobre 131.076 trades MNQ M30; documentación completa en
``sqx/d1/extract_sqx.py``):

    [1]      uint8   idx del test (``Main:``)
    [3]      uint8   flag de dirección (=1 en todos los largos observados)
    [15:23]  int64   hora de COLOCACIÓN de la orden, epoch-ms de RELOJ DE PARED (naive; no convertir tz)
    [24:28]  float32 lotes
    [28:32]  float32 precio de apertura
    [32:40]  int64   hora de EJECUCIÓN (apertura real; = "Open time" del orderstocsv de SQX en el 100% de
                     los trades, 2026-10-14). Con orden a mercado coincide con [15:23]; con stop/límite es
                     posterior (88% de los trades de una cartera CFD de producción, 10,6% del banco MNQ M30)
    [44:52]  int64   close time (epoch-ms, reloj de pared)
    [52:56]  float32 precio de cierre
    [56:60]  float32 precio del SL INICIAL
    [66:70]  float32 PnL NETO en USD (comisión incluida)
    [99:103] float32 MAE en USD (>= 0)

Esquema de salida (``read_trade_table``): ``open_ts``, ``close_ts`` (datetime64[ns]
naive), ``lots``, ``pnl_net``, ``mae_usd``, ``open_px``, ``close_px``, ``sl_px``,
``is_buy``, ``comm_usd``, ``dir_flag``, ``order_ts``; ordenado por ``close_ts`` (estable).
``open_ts`` es la hora de EJECUCIÓN ``[32:40]`` y ``order_ts`` la de colocación ``[15:23]``.
Hasta el 2026-10-14 ``open_ts`` era la de colocación: las ventanas de los trades con orden
pendiente empezaban antes de abrirse (cota más holgada, solapes y swap mal fechados).

Decisiones (plan v2, W1 + enmienda V1):
- ``symbol`` = ``SymbolInfo/@symbolName`` del resultado ``Main`` (p.ej.
  ``@micro_nasdaq``), NO ``instrumentName`` (``NQ_ftmo``). ``pv``/``tick_size``
  salen del ``InstrumentInfo`` de ese ``SymbolInfo``.
- ``is_buy`` = lado del SL inicial (``sl_px < open_px``). El byte ``[3]`` NO es la
  dirección: vale 1 también en los cortos (riesgo R1 del plan cerrado el 2026-10-12
  con 8.467 cortos CFD EURUSD reales: PnL = (open - close)·pv·lots - coste).
- Trades abiertos al final del backtest (cierre en el último instante con |PnL| <= 0,01
  USD y precio movido): SQX no los liquida de verdad; se descartan con aviso.
- ``comm_usd = (close_px - open_px) * (+1 largo / -1 corto) * pv * lots - pnl_net``.
- Fail-loud (``SQXFutureLegError``, subclase de ``SQXFormatError``): lotes <= 0;
  precios open/close/SL <= 0; MAE negativo (estricto, < 0); comisión implícita < -0,02 USD/lote; orders.bin
  truncado o ilegible (nunca escapa un ``struct.error``); claves de coste ausentes
  en el YAML; valores no finitos; ``sl_px == open_px``; MAE menor que la pérdida bruta
  (tolerancia 0,011 USD por lote con suelo de 0,011 USD); comisión/lote no
  constante (> 0,02 USD/lote respecto a la mediana); horas fuera de orden (se exige
  colocación <= ejecución <= cierre); símbolo fuera de la
  allowlist (claves de ``sqx_policy`` de ``config/ftmo_futures_costs.yaml``);
  ``pv`` != ``tick_value/tick_size`` del YAML (tol. relativa 1e-6); sin trades Main.
- Contrato con el consumidor (``src/risk/futures_legs.py``): ``mae_usd >= 0`` ESTRICTO
  (cualquier MAE < 0 => error) y ``comm_usd >= 0``: valores de ``comm_usd`` en
  ``[-0,02*lotes, 0)`` (ruido float32 en backtests sin comisión) se fijan a ``0.0``
  con un warning con el nº de trades ajustados; por debajo de ``-0,02*lotes`` =>
  error. El ajuste ocurre ANTES de la comprobación de constancia y del cruce con el YAML.
- Warning (no error): comisión/lote vs ``total_round_trip`` del YAML > 0,02 USD
  (backtests antiguos pueden llevar otra comisión); ``tickSize`` distinto del YAML;
  trades cortos o flags de dirección no observados.

CFD (``asset_class: cfd`` en el YAML, ``config/ftmo_cfd_costs.yaml``): lotes
fraccionarios (``lot_step``/``min_lot`` del YAML, en ``meta``); el PnL neto incluye el
SWAP de las noches, así que la comisión/lote se estima con la mediana de los trades que
no cruzan el rollover (23:00 hora del servidor; error si se desvía más de 0,02 USD/lote
+ medio céntimo por lote, YAML si hay menos de 5) y el resto del coste implícito va a la columna extra ``swap_usd`` (positivo
= abono). El MAE se cruza contra la pérdida BRUTA de precio. Con ``commission_pct`` en el YAML (oro FTMO,
2026-10-14) la comisión es la ``PercentageBased`` de SQX, ``pct/100 · lotes · precio de apertura · pv``, cobrada
entera al abrir: varía con el precio, y los trades sin rollover se contrastan uno a uno con ese modelo (misma
tolerancia) en vez de con una mediana constante.

Fuente de las especificaciones (guardrail ``entrada-4-bloques``, fase F2, 2026-10-20): ``read_trade_table`` las
toma del YAML de costes (allowlist = ``sqx_policy``); ``read_trade_table_spec`` de la FICHA de símbolo de un
bróker (``poc_web/api/symbols.py``), sin YAML ni allowlist: el símbolo del ``.sqx`` debe ser el de la ficha. Las
dos comparten ``_read_trade_table_impl`` (mismas comprobaciones, mismo ``meta``); solo los mensajes que citan la
fuente cambian (con una ficha nombran los dos valores y su procedencia). Rechazos nuevos, para las dos:
salidas parciales (cualquier elemento ``ATMs`` con ``enable="true"`` en ``strategy_Portfolio.xml``) y dos o más
órdenes ``Main`` con la misma hora de colocación y de ejecución (0 casos en 4.945 ``.sqx`` reales).
``read_backtest_mm`` devuelve el MM del backtest (``lastSettings.xml``) como dato informativo. Con una ficha, el
``tickSize`` del ``.sqx`` debe ser > 0 aunque la ficha no traiga ``tick_size`` (fase F3, H2: la serie por minuto lo usa
como tolerancia); ``read_instrument_info`` devuelve un ``SQXFutureLegError`` claro si falta el ``SymbolInfo`` o su
``InstrumentInfo`` (H1).

CLI: ``python sqx/sqx_trades.py --bank <dir> --out <parquet>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):  # ejecución directa: python sqx/sqx_trades.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqx.d1.extract_sqx import (  # noqa: E402
    _FMT_PREFIX,
    _ORDER_REC_SIZE,
    SQXFormatError,
    _depacketize,
    _read_settings,
)

__all__ = ["SQXFutureLegError", "read_trade_table", "read_trade_table_spec", "read_backtest_mm",
           "read_instrument_info", "MM_CLASSES", "load_costs_policy", "costs_path_for_symbol",
           "DEFAULT_CFD_COSTS_PATH"]

DEFAULT_COSTS_PATH = Path(__file__).resolve().parents[1] / "config" / "ftmo_futures_costs.yaml"
DEFAULT_CFD_COSTS_PATH = Path(__file__).resolve().parents[1] / "config" / "ftmo_cfd_costs.yaml"

COMM_PER_LOT_TOL = 0.02  # USD/lote: desviación máx. respecto a la mediana, y vs YAML
MAE_TOL_PER_LOT = 0.011  # USD/lote: redondeo float32 del MAE vs pérdida bruta
PV_REL_TOL = 1e-6
MAE_VS_RISK_TOL = 0.5  # USD: margen para frac_mae_gt_initial_risk (informativo)
EOB_MIN_GROSS_USD = 0.01  # USD: |PnL| <= esto con |bruto| mayor, en el último cierre = trade abierto al final
PNL_HALF_CENT = 0.005  # USD: el PnL neto de SQX va redondeado a céntimos (error <= medio céntimo)
ROLLOVER_HOUR = 23  # hora del servidor del swap en SQX (rolloutHour del instrumento)
MIN_NO_ROLL_TRADES = 5  # trades sin rollover mínimos para estimar la comisión CFD

# Offsets (ver docstring del módulo).
_REC_DTYPE = np.dtype(
    {
        "names": ["test_idx", "dir_flag", "order_ms", "lots", "open_px", "open_ms",
                  "close_ms", "close_px", "sl_px", "pnl_net", "mae_usd"],
        "formats": ["u1", "u1", ">i8", ">f4", ">f4", ">i8", ">i8", ">f4", ">f4", ">f4", ">f4"],
        "offsets": [1, 3, 15, 24, 28, 32, 44, 52, 56, 66, 99],
        "itemsize": _ORDER_REC_SIZE,
    }
)


class SQXFutureLegError(SQXFormatError):
    """El ``.sqx`` no cumple los requisitos de una pata de futuros nativa."""


# --------------------------------------------------------------------------
# YAML de costes / allowlist
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_costs_cached(path_str: str) -> dict:
    with open(path_str, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_costs_policy(costs_path: str | Path | None = None) -> dict:
    """Carga ``ftmo_futures_costs.yaml``.

    El resultado se CACHEA por ruta (``lru_cache``) durante la vida del proceso:
    editar el YAML exige reiniciar el proceso (o usar otra ruta) para verlo.
    """
    return _load_costs_cached(str(costs_path or DEFAULT_COSTS_PATH))


def costs_path_for_symbol(symbol: str | None) -> Path | None:
    """YAML de costes cuya allowlist (``sqx_policy``) contiene ``symbol``: el de futuros
    o el de CFD FTMO. ``None`` si no está en ninguno."""
    if not symbol:
        return None
    for path in (DEFAULT_COSTS_PATH, DEFAULT_CFD_COSTS_PATH):
        if path.exists() and symbol in (load_costs_policy(path).get("sqx_policy") or {}):
            return path
    return None


def _spec_for_symbol(costs: dict, symbol: str) -> dict:
    """Entrada de ``symbols:`` cuyo ``sqx_symbol`` == symbol."""
    for spec in (costs.get("symbols") or {}).values():
        if spec.get("sqx_symbol") == symbol:
            return spec
    raise SQXFutureLegError(
        f"símbolo {symbol!r} en sqx_policy pero sin entrada 'symbols:' con sqx_symbol coincidente"
    )


def _spec_float(spec: dict, key: str, symbol: str) -> float:
    """``spec[key]`` como float, o ``SQXFutureLegError`` claro si falta o no es numérico."""
    if key not in spec or spec[key] is None:
        raise SQXFutureLegError(f"YAML de costes: falta la clave {key!r} para {symbol}")
    try:
        return float(spec[key])
    except (TypeError, ValueError) as exc:
        raise SQXFutureLegError(
            f"YAML de costes: {key!r} no numérico ({spec[key]!r}) para {symbol}"
        ) from exc


# --------------------------------------------------------------------------
# settings.xml
# --------------------------------------------------------------------------
def _parse_settings_xml(xml: bytes) -> dict:
    """Símbolo, pv, tick_size y precisión del resultado ``Main``."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SQXFutureLegError(f"settings.xml no es XML válido: {exc}") from exc

    results = list(root.iter("Result"))
    mains = [r for r in results if r.get("resultKey", "").startswith("Main:")]
    if not mains and len(results) == 1:
        mains = results
    if len(mains) != 1:
        raise SQXFutureLegError(
            f"se esperaba exactamente 1 resultado 'Main:' en settings.xml, hay {len(mains)}"
        )
    main = mains[0]

    sym_el = next(main.iter("Symbol"), None)
    symbol = (sym_el.text or "").strip() if sym_el is not None else ""
    if not symbol:
        raise SQXFutureLegError("resultado Main sin <Symbol>")

    sym_info = next((si for si in root.iter("SymbolInfo") if si.get("symbolName") == symbol), None)
    if sym_info is None:
        raise SQXFutureLegError(f"SymbolInfo con symbolName={symbol!r} ausente en settings.xml")
    instr = sym_info.find("InstrumentInfo")
    if instr is None:
        raise SQXFutureLegError(f"SymbolInfo {symbol!r} sin InstrumentInfo")
    try:
        pv = float(instr.get("pointValue"))
        tick_size = float(instr.get("tickSize"))
    except (TypeError, ValueError) as exc:
        raise SQXFutureLegError(f"pointValue/tickSize ilegibles para {symbol!r}") from exc

    prec_el = next(main.iter("BacktestPrecision"), None)
    if prec_el is None:
        prec_el = next(root.iter("BacktestPrecision"), None)
    try:
        precision = int(prec_el.text) if prec_el is not None and prec_el.text else None
    except ValueError:
        precision = None

    return {"symbol": symbol, "pv": pv, "tick_size": tick_size, "backtest_precision": precision}


# --------------------------------------------------------------------------
# orders.bin
# --------------------------------------------------------------------------
def _take(payload: bytes, off: int, n: int, what: str) -> bytes:
    """``payload[off:off+n]`` o ``SQXFutureLegError`` si el payload está truncado."""
    if off < 0 or off + n > len(payload):
        raise SQXFutureLegError(
            f"orders.bin truncado: faltan bytes de {what} (offset {off}, {n} B, total {len(payload)} B)"
        )
    return payload[off : off + n]


def _parse_main_records(payload: bytes) -> np.ndarray:
    """Registros del test ``Main:`` como array estructurado (orden del fichero).

    Cualquier fallo de parseo (payload truncado, cabecera inesperada) se eleva como
    ``SQXFutureLegError``; nunca escapa un ``struct.error``/``ValueError`` crudo.
    """
    try:
        return _parse_main_records_impl(payload)
    except SQXFutureLegError:
        raise
    except (struct.error, ValueError, IndexError) as exc:
        raise SQXFutureLegError(f"orders.bin ilegible: {type(exc).__name__}: {exc}") from exc


def _parse_main_records_impl(payload: bytes) -> np.ndarray:
    if len(payload) < 2:
        raise SQXFutureLegError("orders.bin vacío")
    slen = struct.unpack(">H", _take(payload, 0, 2, "longitud de cabecera"))[0]
    head = _take(payload, 2, slen, "cabecera SQOrderFileFormat")
    if not head.startswith(_FMT_PREFIX):
        raise SQXFutureLegError("cabecera SQOrderFileFormat ausente")
    fmt_ver = head[len(_FMT_PREFIX) :]
    if fmt_ver != b"11":
        raise SQXFutureLegError(f"versión de formato no soportada: {fmt_ver!r}")
    off = 2 + slen + 24
    n1, n_strings = struct.unpack(">ii", _take(payload, off, 8, "contadores de cabecera"))
    if n1 != 1 or not (0 < n_strings < 64):
        raise SQXFutureLegError(f"cabecera inesperada ({n1}, {n_strings})")
    off += 8
    strings = []
    for k in range(n_strings):
        sl = struct.unpack(">H", _take(payload, off, 2, f"longitud del string {k}"))[0]
        off += 2
        strings.append(_take(payload, off, sl, f"string {k}").decode("utf-8", "replace"))
        off += sl
    body = len(payload) - off
    if body < 0:
        raise SQXFutureLegError("orders.bin truncado: la tabla de strings excede el payload")
    if body % _ORDER_REC_SIZE:
        raise SQXFutureLegError(f"cuerpo de órdenes {body} B no múltiplo de {_ORDER_REC_SIZE}")
    n_orders = body // _ORDER_REC_SIZE
    if n_orders == 0:
        raise SQXFutureLegError("orders.bin sin órdenes: ningún trade Main")
    recs = np.frombuffer(payload, dtype=_REC_DTYPE, count=n_orders, offset=off)
    main_idx = next((i + 1 for i, s in enumerate(strings) if s.startswith("Main:")), None)
    if main_idx is None:  # sin etiqueta Main: un solo test => todo vale
        ids = np.unique(recs["test_idx"])
        if len(ids) != 1:
            raise SQXFutureLegError("sin test 'Main:' y hay varios test ids")
        main_idx = int(ids[0])
    return recs[recs["test_idx"] == main_idx]


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def read_trade_table(
    path: str | Path, *, costs_path: str | Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Tabla de trades ``Main`` de un ``.sqx`` de micro futuros + metadatos.

    Lanza ``SQXFutureLegError`` (subclase de ``SQXFormatError``) ante cualquier
    violación de las reglas fail-loud del docstring del módulo.

    Las especificaciones salen del YAML de costes (allowlist = claves de ``sqx_policy``). Es la
    misma lectura que :func:`read_trade_table_spec` con otra fuente de especificaciones: los
    mensajes, el orden de las comprobaciones y el ``meta`` son los de siempre (los tests los fijan).
    """
    costs = load_costs_policy(costs_path)
    return _read_trade_table_impl(Path(path), lambda info: _resolve_costs_spec(costs, info))


def read_trade_table_spec(path: str | Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Igual que :func:`read_trade_table`, con las especificaciones de una FICHA de símbolo en vez del YAML.

    ``spec`` (normalizado; lo construye ``poc_web.api.symbols.reader_spec``):

    - ``symbol``: el símbolo SQX de la ficha; el del ``.sqx`` debe ser EXACTAMENTE ese;
    - ``asset_class`` (``"cfd"`` | ``"futures"``), ``point_value_usd``, ``lot_step``, ``min_lot`` (los dos
      últimos solo se usan en CFD);
    - ``tick_size``: ``None`` = no se compara el ``tickSize`` del ``.sqx`` (aviso solo si la ficha lo trae);
    - ``commission``: ``{"per_lot_rt": x}`` (USD por lote, ida y vuelta) o ``{"pct_notional_rt": x}`` (% del
      nocional de apertura, ida y vuelta);
    - ``source``: ``{campo: procedencia}`` (``official`` | ``mt5_sheet`` | ``sqx`` | ``manual``); los
      desacuerdos ``.sqx`` ↔ ficha nombran los dos valores y esta fuente (enmienda E11).

    La lógica de comisión, MAE, swap y rechazos es la misma que la de ``read_trade_table``: solo cambia de
    dónde salen el valor del punto, la comisión y los pasos de lote.
    """
    return _read_trade_table_impl(Path(path), lambda info: _resolve_sheet_spec(spec, info))


def _resolve_costs_spec(costs: dict, info: dict) -> dict:
    """Spec normalizado desde el YAML de costes, con las comprobaciones de siempre y EN EL MISMO ORDEN:
    allowlist → entrada ``symbols:`` → claves de coste → ``tick_size`` > 0 → ``pointValue`` →
    ``asset_class`` → comisión proporcional solo en CFD → ``lot_step``/``min_lot`` → ``tickSize`` (aviso)."""
    symbol, pv, tick_size = info["symbol"], info["pv"], info["tick_size"]
    allow = set((costs.get("sqx_policy") or {}).keys())

    # --- allowlist + coherencia pv con el YAML ---
    if symbol not in allow:
        raise SQXFutureLegError(
            f"símbolo {symbol!r} fuera de la allowlist del YAML de costes {sorted(allow)}"
        )
    spec = _spec_for_symbol(costs, symbol)
    yaml_tick = _spec_float(spec, "tick_size", symbol)
    yaml_tick_value = _spec_float(spec, "tick_value_usd", symbol)
    # Comisión proporcional (PercentageBased de SQX: pct/100 · lotes · precio de apertura · pv,
    # cobrada entera al abrir; oro FTMO, 2026-10-14): sustituye a total_round_trip.
    comm_pct = None
    if spec.get("commission_pct") is not None:
        comm_pct = _spec_float(spec, "commission_pct", symbol)
        yaml_rt = None
    else:
        yaml_rt = _spec_float(spec, "total_round_trip", symbol)
    if yaml_tick <= 0:
        raise SQXFutureLegError(f"YAML de costes: tick_size <= 0 para {symbol}")
    pv_yaml = yaml_tick_value / yaml_tick
    if abs(pv - pv_yaml) > PV_REL_TOL * abs(pv_yaml):
        raise SQXFutureLegError(
            f"pointValue del .sqx ({pv}) != tick_value/tick_size del YAML ({pv_yaml}) para {symbol}"
        )
    asset_class = str(costs.get("asset_class", "futures"))
    if asset_class not in ("futures", "cfd"):
        raise SQXFutureLegError(f"YAML de costes: asset_class desconocido {asset_class!r}")
    if comm_pct is not None and asset_class != "cfd":
        raise SQXFutureLegError(f"YAML de costes: commission_pct solo vale en CFD ({symbol})")
    lot_step = min_lot = None
    if asset_class == "cfd":
        lot_step = _spec_float(spec, "lot_step", symbol)
        min_lot = _spec_float(spec, "min_lot", symbol)
        if lot_step <= 0 or min_lot <= 0:
            raise SQXFutureLegError(f"YAML de costes: lot_step/min_lot <= 0 para {symbol}")
    warnings: list[str] = []
    if abs(tick_size - yaml_tick) > PV_REL_TOL * yaml_tick:
        warnings.append(
            f"tickSize del .sqx ({tick_size}) != tick_size del YAML ({yaml_tick}) para {symbol}"
        )
    return {"origin": "yaml", "symbol": symbol, "asset_class": asset_class, "comm_pct": comm_pct,
            "comm_rt": yaml_rt, "lot_step": lot_step, "min_lot": min_lot, "warnings": warnings,
            "source": {}}


def _sheet_float(spec: dict, key: str, symbol: str) -> float:
    """``spec[key]`` como float finito, o ``SQXFutureLegError`` claro (ficha incompleta)."""
    if spec.get(key) is None:
        raise SQXFutureLegError(f"ficha: falta la clave {key!r} para {symbol}")
    try:
        x = float(spec[key])
    except (TypeError, ValueError) as exc:
        raise SQXFutureLegError(f"ficha: {key!r} no numérico ({spec[key]!r}) para {symbol}") from exc
    if not np.isfinite(x):
        raise SQXFutureLegError(f"ficha: {key!r} no finito ({spec[key]!r}) para {symbol}")
    return x


def _src(source: dict, field: str) -> str:
    """Procedencia de un campo de la ficha para los mensajes (E11)."""
    return str(source.get(field) or "desconocida")


def _resolve_sheet_spec(spec: dict, info: dict) -> dict:
    """Spec normalizado desde una ficha: mismas comprobaciones que :func:`_resolve_costs_spec` (símbolo →
    ``pointValue`` → ``asset_class`` → comisión → pasos de lote → ``tickSize``), con mensajes que nombran los
    dos valores y la fuente del campo de la ficha (E11)."""
    symbol, pv, tick_size = info["symbol"], info["pv"], info["tick_size"]
    if not isinstance(spec, dict) or spec.get("symbol") is None:
        raise SQXFutureLegError("ficha: spec sin 'symbol'")
    if symbol != spec["symbol"]:
        raise SQXFutureLegError(
            f"el símbolo del .sqx ({symbol!r}) no es el de la ficha ({spec['symbol']!r})"
        )
    source = spec.get("source") or {}
    # H2 (revisión de código de la F2): el ``tickSize`` del .sqx debe ser > 0 aunque la ficha no traiga ``tick_size``
    # (AXI): la serie por minuto lo usa como tolerancia de la coherencia con el OHLC cuando la ficha no tiene tick.
    if not tick_size > 0:                      # ``not >`` y no ``<=``: un NaN también es inválido
        raise SQXFutureLegError(f"tickSize del .sqx ({tick_size}) <= 0 (o no finito) para {symbol}")
    sheet_pv = _sheet_float(spec, "point_value_usd", symbol)
    if sheet_pv <= 0:
        raise SQXFutureLegError(f"ficha: point_value_usd <= 0 para {symbol}")
    if abs(pv - sheet_pv) > PV_REL_TOL * abs(sheet_pv):
        raise SQXFutureLegError(
            f"pointValue del .sqx ({pv}) ≠ point_value_usd de la ficha ({sheet_pv}, "
            f"fuente: {_src(source, 'point_value_usd')})"
        )
    asset_class = str(spec.get("asset_class"))
    if asset_class not in ("futures", "cfd"):
        raise SQXFutureLegError(f"ficha: asset_class desconocido {asset_class!r}")
    commission = spec.get("commission")
    if (not isinstance(commission, dict) or len(commission) != 1
            or next(iter(commission)) not in ("per_lot_rt", "pct_notional_rt")):
        raise SQXFutureLegError(
            "ficha: la comisión debe ser exactamente una de per_lot_rt o pct_notional_rt "
            f"(recibido {commission!r}) para {symbol}"
        )
    kind = next(iter(commission))
    comm_pct = comm_rt = None
    if kind == "pct_notional_rt":
        comm_pct = _sheet_float(commission, kind, symbol)
        if asset_class != "cfd":
            raise SQXFutureLegError(f"ficha: pct_notional_rt solo vale en CFD ({symbol})")
    else:
        comm_rt = _sheet_float(commission, kind, symbol)
    lot_step = min_lot = None
    if asset_class == "cfd":
        lot_step = _sheet_float(spec, "lot_step", symbol)
        min_lot = _sheet_float(spec, "min_lot", symbol)
        if lot_step <= 0 or min_lot <= 0:
            raise SQXFutureLegError(f"ficha: lot_step/min_lot <= 0 para {symbol}")
    warnings: list[str] = []
    if spec.get("tick_size") is not None:
        sheet_tick = _sheet_float(spec, "tick_size", symbol)
        if sheet_tick <= 0:                    # H2: como el camino del YAML ("tick_size <= 0"), no se omite en silencio
            raise SQXFutureLegError(f"ficha: tick_size <= 0 para {symbol}")
        if abs(tick_size - sheet_tick) > PV_REL_TOL * sheet_tick:
            warnings.append(
                f"tickSize del .sqx ({tick_size}) ≠ tick_size de la ficha ({sheet_tick}, "
                f"fuente: {_src(source, 'tick_size')})"
            )
    return {"origin": "ficha", "symbol": symbol, "asset_class": asset_class, "comm_pct": comm_pct,
            "comm_rt": comm_rt, "lot_step": lot_step, "min_lot": min_lot, "warnings": warnings,
            "source": dict(source)}


# PLAN-DESVIO (menor): los rechazos nuevos (ATM y órdenes compartidas) elevan ``SQXFutureLegError``, la subclase de
# ``SQXFormatError`` que usan todas las reglas fail-loud de este módulo (``pytest.raises(SQXFormatError)`` los capta).
# En "N órdenes comparten…" N = nº de registros Main que pertenecen a algún grupo con la misma colocación y ejecución.
def _atm_enabled(portfolio_xml: bytes) -> bool:
    """¿Alguna estrategia del ``.sqx`` tiene activadas las salidas parciales (ATM)? Enmienda E8: se parsea el
    ``strategy_Portfolio.xml`` y vale CUALQUIER elemento ``ATMs`` con ``enable`` igual a ``true`` (sin
    distinguir mayúsculas). Un XML que no parsea no se rechaza por esto."""
    try:
        root = ET.fromstring(portfolio_xml)
    except ET.ParseError:
        return False
    return any(el.tag.split("}")[-1] == "ATMs" and str(el.get("enable")).lower() == "true"
               for el in root.iter())


# PLAN-DESVIO: el brief firma ``_read_trade_table_impl(path, spec)`` con un ``spec`` ya construido. Aquí el segundo
# parámetro es un RESOLVEDOR ``info -> spec normalizado`` que el impl llama tras leer el símbolo del .sqx: es la única
# forma de que ``read_trade_table`` conserve EXACTAMENTE el orden de comprobaciones de hoy (la allowlist y las claves del
# YAML dependen del símbolo, que solo se conoce después de abrir el ZIP y antes de la del pointValue). Es una función
# interna: la firma pública de ``read_trade_table`` y la de ``read_trade_table_spec`` son las del brief.
def _read_trade_table_impl(path: Path, resolve) -> tuple[pd.DataFrame, dict]:
    """Cuerpo común de :func:`read_trade_table` y :func:`read_trade_table_spec`.

    ``resolve(info)`` recibe el símbolo, el ``pointValue`` y el ``tickSize`` del ``.sqx``
    (:func:`_parse_settings_xml`) y devuelve el spec normalizado (``origin``, ``asset_class``, ``comm_pct``,
    ``comm_rt``, ``lot_step``, ``min_lot``, ``warnings``, ``source``), o eleva ``SQXFutureLegError``. Se llama
    en el punto exacto donde antes iba la comprobación contra el YAML, así el orden de los errores no cambia.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            strategy_name, _note, capital = _read_settings(zf)
            settings_xml = zf.read("settings.xml")
            try:
                payload = _depacketize(zf.read("orders.bin"))
            except struct.error as exc:
                raise SQXFutureLegError(f"orders.bin ilegible (bloques truncados): {exc}") from exc
            try:
                portfolio_xml = zf.read("strategy_Portfolio.xml")
            except KeyError as exc:
                raise SQXFutureLegError("strategy_Portfolio.xml ausente") from exc
    except zipfile.BadZipFile as exc:
        raise SQXFutureLegError(f"no es un ZIP válido: {exc}") from exc

    info = _parse_settings_xml(settings_xml)
    symbol, pv, tick_size = info["symbol"], info["pv"], info["tick_size"]

    # --- allowlist/ficha + coherencia del pointValue y del tickSize con la fuente de especificaciones ---
    norm = resolve(info)
    yaml_origin = norm["origin"] == "yaml"
    asset_class, comm_pct, yaml_rt = norm["asset_class"], norm["comm_pct"], norm["comm_rt"]
    lot_step, min_lot = norm["lot_step"], norm["min_lot"]
    comm_src = _src(norm["source"], "commission")
    warnings: list[str] = list(norm["warnings"])

    # Enmienda E8: las salidas parciales (ATM) no están soportadas en la v1.
    if _atm_enabled(portfolio_xml):
        raise SQXFutureLegError(
            "salidas parciales (ATM) activadas en la estrategia: no soportadas en la v1"
        )

    # --- trades ---
    recs = _parse_main_records(payload)
    if len(recs) == 0:
        raise SQXFutureLegError("ningún trade Main en orders.bin")

    def _arrays(r: np.ndarray) -> dict:
        return {k: r[k].astype(np.float64) for k in
                ("lots", "open_px", "close_px", "sl_px", "pnl_net", "mae_usd")}

    a = _arrays(recs)
    for name, arr in a.items():
        if not np.isfinite(arr).all():
            raise SQXFutureLegError(f"valores no finitos en {name}")
    if (a["lots"] <= 0).any():
        raise SQXFutureLegError(f"lotes <= 0 en {int((a['lots'] <= 0).sum())} trades")
    for name in ("open_px", "close_px", "sl_px"):
        if (a[name] <= 0).any():
            raise SQXFutureLegError(f"{name} <= 0 en {int((a[name] <= 0).sum())} trades")
    neg_mae = a["mae_usd"] < 0  # estricto: el consumidor (futures_legs) exige mae_usd >= 0
    if neg_mae.any():
        raise SQXFutureLegError(f"MAE negativo en {int(neg_mae.sum())} trades (el MAE es >= 0)")
    if (a["sl_px"] == a["open_px"]).any():
        raise SQXFutureLegError(
            f"sl_px == open_px en {int((a['sl_px'] == a['open_px']).sum())} trades")
    bad_t = (recs["open_ms"] < recs["order_ms"]) | (recs["close_ms"] < recs["open_ms"])
    if bad_t.any():
        raise SQXFutureLegError(
            f"horas incoherentes en {int(bad_t.sum())} trades (se exige colocación <= ejecución <= cierre)")
    # Dos o más órdenes Main con la MISMA hora de colocación y de ejecución = posibles salidas parciales (una
    # entrada que se cierra en trozos): 0 casos en 4.945 .sqx reales. No soportado en la v1.
    _, shared = np.unique(
        np.stack([recs["order_ms"].astype(np.int64), recs["open_ms"].astype(np.int64)], axis=1),
        axis=0, return_counts=True)
    n_shared = int(shared[shared > 1].sum())
    if n_shared:
        raise SQXFutureLegError(
            f"{n_shared} órdenes comparten hora de colocación y de ejecución (¿salidas parciales?): "
            "no soportado en la v1")

    # Trades abiertos al final del backtest: SQX los cierra en el último instante con
    # PnL exactamente 0 aunque el precio se haya movido (visto en CFD EURUSD 2026-10-12).
    # No son trades reales: se descartan con aviso.
    sign0 = np.where(a["sl_px"] < a["open_px"], 1.0, -1.0)
    gross0 = (a["close_px"] - a["open_px"]) * sign0 * pv * a["lots"]
    eob = ((recs["close_ms"] == recs["close_ms"].max())
           & (np.abs(a["pnl_net"]) <= EOB_MIN_GROSS_USD)
           & (np.abs(gross0) > EOB_MIN_GROSS_USD))
    if eob.any():
        recs = recs[~eob]
        if len(recs) == 0:
            raise SQXFutureLegError("ningún trade Main en orders.bin")
        a = _arrays(recs)
        warnings.append(
            f"{int(eob.sum())} trade(s) abiertos al final del backtest (cierre en el último "
            "instante con PnL 0 y precio movido): descartados.")

    lots, open_px, close_px = a["lots"], a["open_px"], a["close_px"]
    sl_px, pnl_net, mae_usd = a["sl_px"], a["pnl_net"], a["mae_usd"]
    dir_flag = recs["dir_flag"].astype(np.uint8)
    # Lado = posición del SL inicial. El byte [3] NO es la dirección: vale 1 también
    # en los cortos (8.467 cortos CFD EURUSD verificados el 2026-10-12, con el PnL
    # cuadrando con (open - close)·pv·lots menos la comisión).
    is_buy = sl_px < open_px
    sign = np.where(is_buy, 1.0, -1.0)
    gross = (close_px - open_px) * sign * pv * lots
    # Tolerancia del MAE: redondeo float32 por lote con suelo absoluto de un céntimo
    # (el MAE viene redondeado a céntimos; con lotes fraccionarios de CFD el término
    # por lote solo se quedaría corto). Con lotes >= 1 (futuros) es la de siempre.
    mae_tol = MAE_TOL_PER_LOT * np.maximum(lots, 1.0)

    swap_usd = None
    if asset_class == "cfd":
        # El PnL neto incluye el SWAP de las noches: la comisión/lote se estima con
        # los trades que no cruzan el rollover (23:00 hora del servidor) y el resto
        # del coste implícito es swap (positivo = abono).
        implied = gross - pnl_net
        o_ts = pd.to_datetime(recs["open_ms"].astype(np.int64), unit="ms")
        c_ts = pd.to_datetime(recs["close_ms"].astype(np.int64), unit="ms")
        same_day = np.asarray(o_ts.normalize() == c_ts.normalize())
        # Abrir justo en el rollover (23:00:00) también paga esa noche: SQX la cuenta (NASDAQ M30 a las 23:00, 14-oct).
        o_at_roll = (o_ts.hour == ROLLOVER_HOUR) & (o_ts.minute == 0) & (o_ts.second == 0)
        crosses = np.asarray(((o_ts.hour < ROLLOVER_HOUR) | o_at_roll) & (c_ts.hour >= ROLLOVER_HOUR))
        no_roll = same_day & ~crosses
        enough = no_roll.sum() >= MIN_NO_ROLL_TRADES
        if enough:
            per_lot = implied[no_roll] / lots[no_roll]
            # float32 de precios (<= 0,02 USD/lote) + redondeo a céntimos del PnL repartido
            # por lote (revisión DeepSeek H6: el término por lote se deriva del redondeo).
            tol = COMM_PER_LOT_TOL + PNL_HALF_CENT / lots[no_roll]
        if comm_pct is not None:
            comm_lot = comm_pct / 100.0 * open_px * pv            # por trade
            if enough:
                dev = np.abs(per_lot - comm_lot[no_roll])
                if (dev > tol).any():
                    if yaml_origin:
                        raise SQXFutureLegError(
                            f"comisión/lote de los trades sin rollover no cuadra con commission_pct "
                            f"({comm_pct}% del nocional de apertura): desviación máx. {dev.max():.4f} USD/lote")
                    worst = int(np.argmax(dev))
                    raise SQXFutureLegError(
                        f"comisión/lote de los trades sin rollover del .sqx ({per_lot[worst]:.4f} USD/lote) no "
                        f"cuadra con pct_notional_rt de la ficha ({comm_pct}% del nocional de apertura = "
                        f"{comm_lot[no_roll][worst]:.4f} USD/lote, fuente: {comm_src}): "
                        f"desviación máx. {dev.max():.4f} USD/lote")
            elif yaml_origin:
                warnings.append(
                    f"solo {int(no_roll.sum())} trades sin rollover: comisión proporcional del "
                    f"YAML ({comm_pct}%) sin verificar contra el backtest")
            else:
                warnings.append(
                    f"solo {int(no_roll.sum())} trades sin rollover: comisión proporcional de la ficha "
                    f"({comm_pct}%, fuente: {comm_src}) sin verificar contra el backtest")
        elif enough:
            comm_lot = float(np.median(per_lot))
            dev = np.abs(per_lot - comm_lot)
            if (dev > tol).any():
                raise SQXFutureLegError(
                    f"comisión/lote no constante en trades sin rollover: desviación máx. "
                    f"{dev.max():.4f} USD/lote respecto a la mediana {comm_lot:.4f}")
        else:
            comm_lot = yaml_rt
            if yaml_origin:
                warnings.append(
                    f"solo {int(no_roll.sum())} trades sin rollover: comisión/lote tomada del "
                    f"YAML ({yaml_rt}) sin verificar contra el backtest")
            else:
                warnings.append(
                    f"solo {int(no_roll.sum())} trades sin rollover: comisión/lote tomada de la ficha "
                    f"({yaml_rt}, fuente: {comm_src}) sin verificar contra el backtest")
        comm_lot = np.broadcast_to(np.asarray(comm_lot, dtype=float), lots.shape)
        if (comm_lot < -COMM_PER_LOT_TOL).any():
            raise SQXFutureLegError(f"comisión/lote implícita negativa ({comm_lot.min():.4f})")
        comm_lot = np.maximum(comm_lot, 0.0)
        comm_usd = comm_lot * lots
        swap_usd = pnl_net - gross + comm_usd
        comm_med = float(np.median(comm_lot))
        mae_viol = mae_usd < -gross - mae_tol
    else:
        comm_usd = gross - pnl_net
        # El consumidor exige comm_usd >= 0. Ruido float32 en [-0,02*lotes, 0) => 0.0
        # (backtests sin comisión); por debajo de -0,02*lotes => error.
        comm_floor = -COMM_PER_LOT_TOL * lots
        if (comm_usd < comm_floor).any():
            raise SQXFutureLegError(
                f"comisión implícita negativa en {int((comm_usd < comm_floor).sum())} trades "
                f"(< -{COMM_PER_LOT_TOL} USD/lote; mín. {float((comm_usd / lots).min()):.4f})"
            )
        comm_clamped = (comm_usd < 0) & (comm_usd >= comm_floor)
        n_comm_clamped = int(comm_clamped.sum())
        if n_comm_clamped:
            comm_usd = np.where(comm_clamped, 0.0, comm_usd)
            warnings.append(
                f"comm_usd fijada a 0.0 en {n_comm_clamped} trades (ruido float32 en "
                f"[-{COMM_PER_LOT_TOL}*lotes, 0))"
            )

        comm_per_lot = comm_usd / lots
        comm_med = float(np.median(comm_per_lot))
        dev = np.abs(comm_per_lot - comm_med)
        if (dev > COMM_PER_LOT_TOL).any():
            raise SQXFutureLegError(
                f"comisión/lote no constante: desviación máx. {dev.max():.4f} USD/lote "
                f"respecto a la mediana {comm_med:.4f} (tol {COMM_PER_LOT_TOL})"
            )
        mae_viol = mae_usd < -(pnl_net + comm_usd) - mae_tol

    if mae_viol.any():
        raise SQXFutureLegError(
            f"MAE menor que la pérdida bruta en {int(mae_viol.sum())} trades (el MAE no puede "
            "ser inferior a la pérdida del trade)"
        )

    if yaml_rt is not None and abs(comm_med - yaml_rt) > COMM_PER_LOT_TOL:
        if yaml_origin:
            warnings.append(
                f"comisión/lote del .sqx ({comm_med:.4f}) difiere de total_round_trip del YAML "
                f"({yaml_rt:.4f}) para {symbol} en más de {COMM_PER_LOT_TOL}"
            )
        else:
            warnings.append(
                f"comisión/lote del .sqx ({comm_med:.4f}) ≠ per_lot_rt de la ficha ({yaml_rt:.4f}, "
                f"fuente: {comm_src}) para {symbol} en más de {COMM_PER_LOT_TOL}"
            )
    other_flags = sorted({int(f) for f in np.unique(dir_flag)} - {1})
    if other_flags:
        warnings.append(f"valores de flag de dirección no observados antes: {other_flags}")
    if capital is None:
        warnings.append("InitialCapital ausente en settings.xml")

    initial_risk = np.abs(open_px - sl_px) * pv * lots
    frac_mae_gt = float(np.mean(mae_usd > initial_risk + MAE_VS_RISK_TOL))

    cols = {
        # pandas >= 3 devuelve datetime64[ms]; se fija [ns] (contrato del esquema).
        "open_ts": pd.to_datetime(recs["open_ms"].astype(np.int64), unit="ms").astype("datetime64[ns]"),
        "close_ts": pd.to_datetime(recs["close_ms"].astype(np.int64), unit="ms").astype("datetime64[ns]"),
        "lots": lots,
        "pnl_net": pnl_net,
        "mae_usd": mae_usd,
        "open_px": open_px,
        "close_px": close_px,
        "sl_px": sl_px,
        "is_buy": is_buy,
        "comm_usd": comm_usd,
        "dir_flag": dir_flag,
        "order_ts": pd.to_datetime(recs["order_ms"].astype(np.int64), unit="ms").astype("datetime64[ns]"),
    }
    if swap_usd is not None:
        cols["swap_usd"] = swap_usd
    df = pd.DataFrame(cols)
    order = np.argsort(df["close_ts"].to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)

    meta = {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "pv": pv,
        "tick_size": tick_size,
        "initial_capital": capital,
        "backtest_precision": info["backtest_precision"],
        "sha1": hashlib.sha1(portfolio_xml).hexdigest(),
        "n_trades": int(len(df)),
        "warnings": warnings,
        "frac_mae_gt_initial_risk": frac_mae_gt,
        "frac_pending_entry": float(np.mean(recs["open_ms"] > recs["order_ms"])),
        "asset_class": asset_class,
    }
    if asset_class == "cfd":
        meta.update(lot_step=lot_step, min_lot=min_lot, commission_per_lot=comm_med)
        if comm_pct is not None:
            meta["commission_pct"] = comm_pct
    return df, meta


# --------------------------------------------------------------------------
# Datos informativos del .sqx: MM del backtest e InstrumentInfo
# --------------------------------------------------------------------------
# Clases de Money Management de SQX (tipo del <Method> de lastSettings.xml): los 12 métodos built-in
# (skill sqx-build-patterns, references/money-management.md).
MM_CLASSES = frozenset({
    "FixedSize", "FixedAmount", "RiskFixedBalancePct", "RiskFixedPctOfAccount", "ATRRiskBasedSizing",
    "StocksSizeByPrice", "CryptoFixedAmount", "CryptoFixedBalancePct", "CryptoSizeByPrice",
    "PickerFixedAmount", "PickerRiskFixedPctOfAccount", "SimpleMartingaleMM",
})


def _finite_float(raw) -> float | None:
    """``raw`` como float finito, o ``None`` si no lo es."""
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def read_backtest_mm(path: str | Path) -> dict | None:
    """Money Management del backtest que produjo el ``orders.bin`` (``lastSettings.xml`` del ``.sqx``).

    Devuelve ``{"method": <tipo>, "params": {<clave sin #>: float}}`` del ``Method`` con ``use="true"`` cuyo
    ``type`` es una clase de MM (:data:`MM_CLASSES`), p. ej. ``{"method": "FixedAmount", "params":
    {"RiskedMoney": 57.5, "Decimals": 2.0, "LotsIfNoMM": 0.01, "MaxLots": 10.0}}``. Es un dato INFORMATIVO (el
    MM del backtest no cambia las operaciones, solo los lotes): ``None`` si no hay ``lastSettings.xml``, no
    parsea o no hay un método de MM activo; nunca eleva."""
    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("lastSettings.xml")
        root = ET.fromstring(raw)
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return None
    for el in root.iter():
        if (el.tag.split("}")[-1] == "Method" and str(el.get("use")).lower() == "true"
                and el.get("type") in MM_CLASSES):
            params: dict[str, float] = {}
            for prm in el.iter("Param"):
                key = (prm.get("key") or "").strip().strip("#")
                val = _finite_float(prm.get("value") if prm.get("value") is not None else (prm.text or "").strip())
                if key and val is not None:
                    params[key] = val
            return {"method": str(el.get("type")), "params": params}
    return None


def _instrument_commission(attr: str | None) -> dict | None:
    """``{"type", "value"}`` del método de comisión ACTIVO del atributo ``commissions`` del ``InstrumentInfo``
    (el XML de los ``<Method>`` va escapado dentro del atributo; el parser ya lo devuelve sin escapar).
    ``None`` si no hay método activo o no parsea."""
    if not attr or not attr.strip():
        return None
    text = attr.strip()
    xml = text if text.startswith("<Commissions") else f"<Commissions>{text}</Commissions>"
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    for m in root.iter():
        if m.tag.split("}")[-1] == "Method" and str(m.get("use")).lower() == "true":
            value = None
            for prm in m.iter("Param"):
                if (prm.get("key") or "").strip().strip("#") == "Commission":
                    value = _finite_float(prm.get("value") if prm.get("value") is not None
                                          else (prm.text or "").strip())
                    break
            return {"type": str(m.get("type")), "value": value}
    return None


def read_instrument_info(path: str | Path) -> dict:
    """``InstrumentInfo`` del ``SymbolInfo`` del resultado ``Main`` de ``settings.xml``: lo que un ``.sqx`` PROPONE
    para dar de alta una ficha de símbolo (nunca la verifica). Devuelve ``symbol`` (``SymbolInfo/@symbolName``),
    ``instrument``, ``point_value``, ``tick_size``, ``order_size_step``, ``default_spread`` (en unidades de
    ``tickSize``) y ``commission`` (``{"type": "SizeBased", "value": 4.5}`` o ``None``), con ``None`` en lo ausente
    o ilegible. El lote mínimo y el máximo no están en el ``.sqx``. ``SQXFutureLegError`` si el ZIP o el
    ``settings.xml`` no sirven."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("settings.xml")
    except zipfile.BadZipFile as exc:
        raise SQXFutureLegError(f"no es un ZIP válido: {exc}") from exc
    except KeyError as exc:
        raise SQXFutureLegError("settings.xml ausente") from exc
    info = _parse_settings_xml(xml)
    root = ET.fromstring(xml)                        # ya parseó bien en _parse_settings_xml
    # H1 (revisión de código de la F2): el ``SymbolInfo`` del símbolo del resultado Main (``info["symbol"]``) y su
    # ``InstrumentInfo`` se buscan SIN suponer que existen: si faltan, un error claro (el endpoint lo devuelve como 400),
    # nunca un ``StopIteration``/``AttributeError`` que acabaría en 500.
    sym_info = next((si for si in root.iter("SymbolInfo") if si.get("symbolName") == info["symbol"]), None)
    if sym_info is None:
        raise SQXFutureLegError(
            f"settings.xml sin SymbolInfo para el símbolo del resultado Main ({info['symbol']!r})")
    instr = sym_info.find("InstrumentInfo")
    if instr is None:
        raise SQXFutureLegError(
            f"el SymbolInfo del símbolo {info['symbol']!r} no trae InstrumentInfo: no hay nada que proponer")
    return {
        "symbol": info["symbol"],
        "instrument": instr.get("instrument"),
        "point_value": info["pv"],
        "tick_size": info["tick_size"],
        "order_size_step": _finite_float(instr.get("orderSizeStep")),
        "default_spread": _finite_float(instr.get("defaultSpread")),
        "commission": _instrument_commission(instr.get("commissions")),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _bank_to_parquet(bank: Path, out: Path) -> int:
    files = sorted(bank.glob("*.sqx"))
    if not files:
        print(f"ERROR: sin ficheros .sqx en {bank}", file=sys.stderr)
        return 2
    frames, metas, failed = [], [], []
    for f in files:
        try:
            df, meta = read_trade_table(f)
        except Exception as exc:  # noqa: BLE001 - se lista, nunca se traga
            failed.append((f.name, f"{type(exc).__name__}: {exc}"))
            continue
        df = df.assign(file=f.name, sha1=meta["sha1"])
        frames.append(df)
        metas.append({**meta, "file": f.name, "warnings": json.dumps(meta["warnings"])})
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        pd.DataFrame(metas).to_parquet(out.with_name(f"{out.stem}_meta.parquet"), index=False)
    print(f"{len(frames)}/{len(files)} ficheros OK -> {out}")
    if failed:
        print(f"FALLARON {len(failed)} ficheros:", file=sys.stderr)
        for name, msg in failed:
            print(f"  {name}: {msg}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", required=True, type=Path, help="directorio con .sqx")
    ap.add_argument("--out", required=True, type=Path, help="parquet de salida")
    args = ap.parse_args(argv)
    return _bank_to_parquet(args.bank, args.out)


if __name__ == "__main__":
    sys.exit(main())
