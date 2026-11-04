"""Extractor puro-Python de resultados de un .sqx (StrategyQuant X, build 143).

Un .sqx es un ZIP. Se leen tres entradas:

- ``settings.xml`` → ``<StrategyName>``, ``<Note>`` (sector) y
  ``MoneyManagement.InitialCapital``.
- ``orders.bin`` → stream con magic Java-serialization ``aced 0005`` troceado en
  bloques TC_BLOCKDATALONG (0x7A + uint32BE len, len ≤ 1024) y un bloque final
  TC_BLOCKDATA (0x77; longitud de 1 byte cuando cabe — variante no estándar de
  SQX). Payload:
    - uint16BE len + UTF ``"SQOrderFileFormat:11"``
    - 24 bytes a cero (3×int64 reservados)
    - int32BE = 1, int32BE = N_STRINGS, N_STRINGS × (uint16BE len + UTF):
      tabla de strings (1ª vacía; incluye "Main: SYMBOL/TF", "Strategy x.y.z",
      "SYMBOL", y en retests multi-test también "CrossCheck_...", etc.)
    - registros de órdenes de 149 bytes exactos hasta el final del payload.
      Layout (big-endian; floats = float32):
        [0]   uint8  idx (1-based) de símbolo en la tabla de strings
        [1]   uint8  idx (1-based) del test ("Main: ..." es el retest principal)
        [2]   uint8  idx (1-based) del nombre de estrategia
        [3]   uint8  flags/dirección (=1 en los 131.076 trades largos observados;
                     el valor de los cortos NO está observado)
        [4:8]    int32   ticket
        [8:12]   int32   0
        [12:15]  3 bytes flags
        [15:23]  int64   open time (epoch-ms; ver nota tz abajo)
        [23]     uint8   (constante 6)
        [24:28]  float32 lotes
        [28:32]  float32 precio apertura
        [32:40]  int64   2º timestamp (≈ open; puede diferir)
        [40:44]  float32 2º precio (== apertura)
        [44:52]  int64   close time (epoch-ms; ver nota tz abajo)
        [52:56]  float32 precio cierre
        [56:60]  float32 precio del SL INICIAL (nunca 0; en largos < apertura)  ★
                         (verificado 2026-10-10 sobre 131.076 trades MNQ M30)
        [60]     uint8 flags
        [61:65]  float32 (constante, ~-0.367)
        [65]     uint8 flags
        [66:70]  float32 PnL NETO en money (comisiones/swap horneados)  ★
        [70:94]  6×float32 PnL en pips/ticks y variantes
        [94:98]  float32 swap
        [98]     uint8 (=1)
        [99:103] float32 MAE en USD (≥ 0; peor flotante bruto de toda la vida
                         del trade)  ★
        [103:107] float32 MAE en ticks (MNQ: 2×USD, tick $0,50)
        [107:111] float32 MFE en USD
        [111:115] float32 MFE en ticks
                         ([99:115] verificado 2026-10-10 sobre 131.076 trades MNQ
                         M30: 0 violaciones de MAE ≥ −(PnL neto + comisión); el
                         74% de las perdedoras cierra con MAE = pérdida bruta)
        [115:119] int32  duración del trade en segundos
        [119:123] float32 equity/balance tras el cierre (money, absoluta)
        [123:127] float32 PnL pips (variante)
        [127:135] int64  magic number (p.ej. 11111)
        [135]    uint8 (=1)
        [136:140] float32 (~drawdown % al cierre)
        [140:144] float32 0
        [144]    uint8 (0xFF = -1)
        [145:149] float32 (métrica pequeña, ~pips)
- ``Results/Main: <SYM>_LOM_<TF>/dailyEquity.bin`` → mismo esquema de bloques.
  Payload = int32BE count + count × 16 bytes: (int64BE fecha epoch-ms a las
  00:00 del día, float64BE equity). La equity es el MÍNIMO INTRADÍA de la equity
  (balance realizado + flotante) de ese día calendario, RELATIVA al capital
  inicial (empieza en 0.0) — NO es la equity de cierre de día: en días sin
  actividad de trades coincide exactamente con el balance; en días con actividad
  es ≤ balance de las 00:00 (verificado numéricamente en 15 ficheros). El
  último registro corresponde al fin de los datos (puede ser intradía). El
  primer registro es el inicio del lead-in de datos (p.ej. 2021-05-11); hay que
  recortar a la ventana de test. Validación: en días sin trades activos,
  equity == suma(PnL de trades cerrados antes del día) con error < 0.02 (oro)
  / < 0.15 (EURUSD, redondeo interno float32 de SQX).

NOTA TZ (verificado 2026-10-08, enmienda A10 del plan float-valleys): los
epoch-ms de open/close y las fechas de dailyEquity son del RELOJ DE PARED de
los datos — ET wall-clock naive (hueco de opens en la hora 17 = break CME
17:00-18:00 ET); NO son UTC. ``extract_sqx`` los etiqueta como datetime aware
UTC por comodidad de tipado, pero esa etiqueta NO implica conversión de
meridiano. Prohibido aplicar zoneinfo a estos valores.

Solo librería estándar. Pensado para procesar ~12k ficheros en minutos.
"""
from __future__ import annotations

import re
import struct
import zipfile
from datetime import date, datetime, timedelta, timezone

__all__ = ["SQXFormatError", "extract_sqx"]

_FMT_PREFIX = b"SQOrderFileFormat:"
_ORDER_REC_SIZE = 149
_EQUITY_REC_SIZE = 16
_OFF_CLOSE_TIME = 44
_OFF_PROFIT = 66
_OFF_OPEN_TIME = 15
_OFF_LOTS = 24

_RE_STRATEGY = re.compile(rb"<StrategyName type=\"String\">([^<]*)</StrategyName>")
_RE_NOTE = re.compile(rb"<Note type=\"String\">([^<]*)</Note>")
_RE_CAPITAL = re.compile(
    rb"<MoneyManagement\.InitialCapital type=\"Double\">([^<]*)</MoneyManagement\.InitialCapital>"
)


class SQXFormatError(ValueError):
    """El fichero no sigue el layout esperado de SQX build 143."""


def _depacketize(buf: bytes) -> bytes:
    """Reensambla el payload de un stream SQX (bloques 0x7A/0x77) tras 'aced0005'."""
    if buf[:4] != b"\xac\xed\x00\x05":
        raise SQXFormatError("magic java-serialization ausente")
    out = bytearray()
    off = 4
    n = len(buf)
    while off < n:
        tag = buf[off]
        if tag == 0x7A:  # TC_BLOCKDATALONG + uint32BE
            ln = struct.unpack(">I", buf[off + 1 : off + 5])[0]
            off += 5
        elif tag == 0x77:  # TC_BLOCKDATA; SQX escribe len de 1 byte cuando < 256
            ln2 = struct.unpack(">H", buf[off + 1 : off + 3])[0]
            if off + 3 + ln2 <= n:
                ln = ln2
                off += 3
            else:
                ln = buf[off + 1]
                off += 2
        else:
            raise SQXFormatError(f"tag de bloque desconocido 0x{tag:02x} @ {off}")
        if off + ln > n:
            raise SQXFormatError("bloque excede el tamaño del buffer")
        out += buf[off : off + ln]
        off += ln
    return bytes(out)


def _read_settings(zf: zipfile.ZipFile) -> tuple[str, str, float | None]:
    xml = zf.read("settings.xml")
    m = _RE_STRATEGY.search(xml)
    strategy_name = m.group(1).decode("utf-8", "replace") if m else ""
    m = _RE_NOTE.search(xml)
    note = m.group(1).decode("utf-8", "replace") if m else ""
    m = _RE_CAPITAL.search(xml)
    capital = float(m.group(1)) if m else None
    return strategy_name, note, capital


def _parse_orders(payload: bytes) -> list[tuple[int, float]]:
    """Devuelve [(close_time_epoch_ms, profit)] del test principal ('Main: ...')."""
    slen = struct.unpack(">H", payload[:2])[0]
    if not payload[2 : 2 + slen].startswith(_FMT_PREFIX):
        raise SQXFormatError("cabecera SQOrderFileFormat ausente")
    fmt_ver = payload[2 + len(_FMT_PREFIX) : 2 + slen]
    if fmt_ver != b"11":
        raise SQXFormatError(f"versión de formato no soportada: {fmt_ver!r}")
    off = 2 + slen + 24  # 24 bytes reservados
    n1, n_strings = struct.unpack(">ii", payload[off : off + 8])
    if n1 != 1 or not (0 < n_strings < 64):
        raise SQXFormatError(f"cabecera inesperada ({n1}, {n_strings})")
    off += 8
    strings = []
    for _ in range(n_strings):
        sl = struct.unpack(">H", payload[off : off + 2])[0]
        off += 2
        strings.append(payload[off : off + sl].decode("utf-8", "replace"))
        off += sl
    body = len(payload) - off
    if body % _ORDER_REC_SIZE:
        raise SQXFormatError(f"cuerpo de órdenes {body} B no múltiplo de {_ORDER_REC_SIZE}")
    n_orders = body // _ORDER_REC_SIZE
    main_idx = next((i + 1 for i, s in enumerate(strings) if s.startswith("Main:")), None)
    if main_idx is None:  # sin etiqueta Main: un solo test → todo vale
        test_ids = {payload[off + k * _ORDER_REC_SIZE + 1] for k in range(n_orders)}
        if len(test_ids) != 1:
            raise SQXFormatError("sin test 'Main:' y hay varios test ids")
        main_idx = test_ids.pop()

    unpack_f = struct.Struct(">f").unpack_from
    unpack_q = struct.Struct(">q").unpack_from
    trades = []
    for k in range(n_orders):
        ro = off + k * _ORDER_REC_SIZE
        if payload[ro + 1] != main_idx:
            continue
        trades.append((unpack_q(payload, ro + _OFF_CLOSE_TIME)[0], unpack_f(payload, ro + _OFF_PROFIT)[0]))
    trades.sort(key=lambda t: t[0])
    return trades


def _parse_orders_full(payload: bytes) -> list[tuple[int, int, float, float]]:
    """Devuelve [(open_time_epoch_ms, close_time_epoch_ms, lots, profit)] del
    test principal ('Main: ...'). Misma cabecera y filtro que `_parse_orders`;
    añade open time (offset 15:23) y lotes (offset 24:28).

    Los timestamps son epoch-ms del RELOJ DE PARED de los datos (ET wall-clock
    naive; verificado 2026-10-08: hueco de opens en la hora 17 = break CME
    17:00-18:00 ET) — NO convertir con zoneinfo (enmienda A10).
    """
    slen = struct.unpack(">H", payload[:2])[0]
    if not payload[2 : 2 + slen].startswith(_FMT_PREFIX):
        raise SQXFormatError("cabecera SQOrderFileFormat ausente")
    fmt_ver = payload[2 + len(_FMT_PREFIX) : 2 + slen]
    if fmt_ver != b"11":
        raise SQXFormatError(f"versión de formato no soportada: {fmt_ver!r}")
    off = 2 + slen + 24  # 24 bytes reservados
    n1, n_strings = struct.unpack(">ii", payload[off : off + 8])
    if n1 != 1 or not (0 < n_strings < 64):
        raise SQXFormatError(f"cabecera inesperada ({n1}, {n_strings})")
    off += 8
    strings = []
    for _ in range(n_strings):
        sl = struct.unpack(">H", payload[off : off + 2])[0]
        off += 2
        strings.append(payload[off : off + sl].decode("utf-8", "replace"))
        off += sl
    body = len(payload) - off
    if body % _ORDER_REC_SIZE:
        raise SQXFormatError(f"cuerpo de órdenes {body} B no múltiplo de {_ORDER_REC_SIZE}")
    n_orders = body // _ORDER_REC_SIZE
    main_idx = next((i + 1 for i, s in enumerate(strings) if s.startswith("Main:")), None)
    if main_idx is None:  # sin etiqueta Main: un solo test → todo vale
        test_ids = {payload[off + k * _ORDER_REC_SIZE + 1] for k in range(n_orders)}
        if len(test_ids) != 1:
            raise SQXFormatError("sin test 'Main:' y hay varios test ids")
        main_idx = test_ids.pop()

    unpack_f = struct.Struct(">f").unpack_from
    unpack_q = struct.Struct(">q").unpack_from
    trades = []
    for k in range(n_orders):
        ro = off + k * _ORDER_REC_SIZE
        if payload[ro + 1] != main_idx:
            continue
        trades.append(
            (
                unpack_q(payload, ro + _OFF_OPEN_TIME)[0],
                unpack_q(payload, ro + _OFF_CLOSE_TIME)[0],
                unpack_f(payload, ro + _OFF_LOTS)[0],
                unpack_f(payload, ro + _OFF_PROFIT)[0],
            )
        )
    trades.sort(key=lambda t: t[1])
    return trades


def _parse_daily_equity(payload: bytes) -> list[tuple[int, float]]:
    """Devuelve [(fecha_epoch_ms_00UTC, equity_relativa)] sin recortar."""
    count = struct.unpack(">i", payload[:4])[0]
    if len(payload) - 4 != count * _EQUITY_REC_SIZE:
        raise SQXFormatError(
            f"dailyEquity: {len(payload) - 4} B de cuerpo != {count}×{_EQUITY_REC_SIZE}"
        )
    unpack_q = struct.Struct(">q").unpack_from
    unpack_d = struct.Struct(">d").unpack_from
    return [
        (unpack_q(payload, 4 + i * 16)[0], unpack_d(payload, 12 + i * 16)[0])
        for i in range(count)
    ]


def extract_sqx(
    path: str,
    *,
    clip: bool = True,
    clip_start: date = date(2022, 1, 1),
    clip_end: date = date(2026, 6, 19),
) -> dict:
    """Extrae resultados del retest principal de un .sqx.

    Devuelve dict con:
      - ``path``, ``strategy_name``, ``note``, ``initial_capital``
      - ``n_trades``: nº de trades del test principal en orders.bin
      - ``trades``: [(close_time_utc: datetime aware, profit: float)]
      - ``daily_equity``: [(date, equity)]; equity = MÍNIMO intradía de la
        equity (balance + flotante) del día, RELATIVA al capital inicial
        (equity absoluta = valor + initial_capital). No es equity de cierre.

    Con ``clip=True`` la equity diaria se recorta a [clip_start, clip_end].
    """
    with zipfile.ZipFile(path) as zf:
        strategy_name, note, capital = _read_settings(zf)
        trades = _parse_orders(_depacketize(zf.read("orders.bin")))
        eq_name = next(
            (
                nm
                for nm in zf.namelist()
                if nm.startswith("Results/Main:") and nm.endswith("dailyEquity.bin")
            ),
            None,
        )
        equity = _parse_daily_equity(_depacketize(zf.read(eq_name))) if eq_name else []

    utc = timezone.utc
    trade_list = [(datetime.fromtimestamp(ms / 1000, utc), round(p, 4)) for ms, p in trades]

    eq_list = []
    for ms, val in equity:
        d = (datetime.fromtimestamp(ms / 1000, utc)).date()
        if clip and not (clip_start <= d <= clip_end):
            continue
        eq_list.append((d, val))

    return {
        "path": path,
        "strategy_name": strategy_name,
        "note": note,
        "initial_capital": capital,
        "n_trades": len(trade_list),
        "trades": trade_list,
        "daily_equity": eq_list,
    }
