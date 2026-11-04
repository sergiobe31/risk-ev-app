"""Parse del bloque Money Management (ATRRiskBasedSizing) de un .sqx o .xml.

Vive en ``sqx/`` (es parsing SQX-especifico; puede importar src si hiciera
falta). ``risk_app.py`` lo importa perezoso. Devuelve un dataclass plano con los
params que el motor exacto (``src.risk.atr``) necesita.

Fuentes DENTRO del .sqx (ZIP), por fiabilidad (corregido 2026-07-18):
  * **MM**: ``lastSettings.xml`` tiene PRIORIDAD — es la config del run del
    Tester/Retester que produjo los resultados (orders/dailyEquity). OJO al
    matiz: sus ``<Chart>`` NO son fiables para el símbolo (son la "vista" de la
    GUI al guardar, pueden ser de OTRA estrategia — bug real: 2 .sqx DAX
    reportaban XAUUSD_axi). ``strategy_Portfolio.xml`` es la definición de la
    estrategia en el databank y puede estar VIEJA (caso real: FixedSize 0.1
    porque el Builder no portó el bloque MM por .tpl faltante; el retester
    corregido escribió la MM buena en lastSettings.xml) — queda como fallback.
  * **symbol/timeframe**: del path ``Results/Main: <SYM>_..._<TF>/`` dentro del
    ZIP (el resultKey del backtest = verdad del run). Fallback: ``<Chart>`` /
    ``<Symbol>``/``<Timeframe>`` del XML.
  * **InitialCapital**: elemento hermano ``<InitialCapital>5000</InitialCapital>``
    en lastSettings.xml (NO es un Param del Method; antes siempre salía None).

Un ``.sqx`` de PORTFOLIO (``Results/Portfolio/`` o entradas ``Results/Strategy
<id>:`` o ``mergedResult="true"``) NO tiene un único MM/símbolo: se RECHAZA con
error explícito listando sus estrategias (la MM por estrategia vive en los .sqx
individuales).

Formatos tolerados:
  * ``.xml`` (estrategia exportada): ``<MoneyManagement type="ATRRiskBasedSizing">
    <Method><Params><Param key="#Risk#" value="2.0">2.0</Param>...``
  * ``.sqx`` (ZIP): ver arriba.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

# key normalizado -> (campo, cast)
_PARAM_KEYS: dict[str, tuple[str, type]] = {
    "risk": ("risk", float),
    "atrperiod": ("atr_period", int),
    "atrmult": ("atr_mult", float),
    "decimals": ("decimals", int),
    "lotsifnomm": ("lots_if_no_mm", float),
    "maxlots": ("max_lots", float),
    "initialcapital": ("initial_capital", float),
}
_NUMERIC_FIELDS = [f for f, _ in _PARAM_KEYS.values()]

# "XAUUSD_axi_LOM_H4" -> (XAUUSD_axi, H4); "Main: EURUSD_axi_D1" -> (EURUSD_axi, D1)
_RESULT_KEY_RE = re.compile(
    r"^(?P<sym>.+?)(?:_LOM)?_(?P<tf>M1|M5|M15|M30|H1|H2|H3|H4|H6|H8|H12|D1|W1|MN1)$"
)


@dataclass(frozen=True)
class StrategyMM:
    """Params del bloque ATRRiskBasedSizing + symbol/TF. ``None`` si ausente."""

    symbol: str | None = None
    timeframe: str | None = None
    mm_type: str | None = None
    risk: float | None = None
    atr_period: int | None = None
    atr_mult: float | None = None
    decimals: int | None = None
    lots_if_no_mm: float | None = None
    max_lots: float | None = None
    initial_capital: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _read_strategy_xml(xml_bytes: bytes) -> StrategyMM:
    root = ET.fromstring(xml_bytes)

    # 1) Encontrar el elemento del MM activo. lastSettings.xml lista VARIOS
    # <Method type=...> (con use="true"/"false"); hay que quedarse con el
    # activo (use="true"), no con el primero que tenga "Risk" en el tipo.
    candidates = []
    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()  # sin namespace
        t = el.get("type")
        if t and tag in ("method", "moneymanagement"):
            candidates.append((el, t))
    mm_el, mm_type = None, None
    # preferir use="true" (metodo activo en lastSettings)
    for el, t in candidates:
        if el.get("use") == "true":
            mm_el, mm_type = el, t
            break
    # si no, preferir ATRRiskBasedSizing explicitamente
    if mm_el is None:
        for el, t in candidates:
            if t and "ATRRisk" in t:
                mm_el, mm_type = el, t
                break
    # ultimo recurso: el primero
    if mm_el is None and candidates:
        mm_el, mm_type = candidates[0]

    params: dict[str, float] = {}
    if mm_el is not None:
        for p in mm_el.iter("Param"):
            key = (p.get("key") or "").strip().strip("#").lower()
            val = p.get("value")
            if val is None:
                val = (p.text or "").strip()
            if key in _PARAM_KEYS and val not in ("", None):
                field, cast = _PARAM_KEYS[key]
                try:
                    params[field] = cast(float(val))
                except (TypeError, ValueError):
                    pass

    # InitialCapital: elemento hermano (lastSettings), NO un Param del Method.
    if "initial_capital" not in params:
        for el in root.iter():
            if el.tag.split("}")[-1] == "InitialCapital" and (el.text or "").strip():
                try:
                    params["initial_capital"] = float(el.text.strip())
                except ValueError:
                    pass
                break

    # 2) symbol / timeframe (fallback XML; el path Results/ del ZIP manda).
    symbol = timeframe = None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Chart":
            symbol = el.get("symbol") or symbol
            timeframe = el.get("timeframe") or timeframe
    if symbol is None:
        for el in root.iter("Symbol"):
            if (el.text or "").strip() or el.get("name"):
                symbol = el.get("name") or el.text.strip()
                break
    if timeframe is None:
        for el in root.iter("Timeframe"):
            if (el.text or "").strip():
                timeframe = el.text.strip()
                break

    return StrategyMM(
        symbol=symbol, timeframe=timeframe, mm_type=mm_type, **{f: params.get(f) for f in _NUMERIC_FIELDS}
    )


def _symbol_tf_from_results(names: set[str]) -> tuple[str | None, str | None]:
    """symbol/TF del resultKey ``Results/Main: <SYM>_..._<TF>/`` (verdad del run)."""
    for n in sorted(names):
        m = re.match(r"Results/Main: ([^/]+)/", n)
        if m:
            km = _RESULT_KEY_RE.match(m.group(1).strip())
            if km:
                return km.group("sym"), km.group("tf")
    return None, None


def _check_not_portfolio(names: set[str], zf: zipfile.ZipFile) -> None:
    """Veta .sqx de PORTFOLIO: no hay un único MM/símbolo que parsear."""
    parts = sorted({n.split("/")[1] for n in names if n.startswith("Results/Strategy")})
    is_portfolio = bool(parts) or any(n.startswith("Results/Portfolio/") for n in names)
    if not is_portfolio and "strategy_Portfolio.xml" in names:
        head = zf.read("strategy_Portfolio.xml")[:400].decode("utf-8", errors="replace")
        is_portfolio = 'mergedResult="true"' in head
    if is_portfolio:
        raise ValueError(
            "Este .sqx es un PORTFOLIO, no una estrategia: no tiene un MM/simbolo "
            f"unico. Estrategias detectadas: {parts or 'ver Results/ en el ZIP'}. "
            "Exporta/parsea los .sqx individuales de cada estrategia."
        )


def _parse_sqx(zf: zipfile.ZipFile) -> StrategyMM:
    names = set(zf.namelist())
    _check_not_portfolio(names, zf)
    # MM: lastSettings.xml (config del run que produjo los resultados) tiene
    # prioridad; strategy_Portfolio.xml es fallback (puede estar vieja).
    mm: StrategyMM | None = None
    for member in ("lastSettings.xml", "strategy_Portfolio.xml", "settings.xml"):
        if member in names:
            mm = _read_strategy_xml(zf.read(member))
            break
    if mm is None:
        raise ValueError(f".sqx sin XML de estrategia conocido: {sorted(names)}")
    # symbol/TF: el resultKey Results/ manda sobre los <Chart> del XML.
    sym, tf = _symbol_tf_from_results(names)
    if sym or tf:
        mm = StrategyMM(
            symbol=sym or mm.symbol, timeframe=tf or mm.timeframe,
            mm_type=mm.mm_type, risk=mm.risk, atr_period=mm.atr_period,
            atr_mult=mm.atr_mult, decimals=mm.decimals,
            lots_if_no_mm=mm.lots_if_no_mm, max_lots=mm.max_lots,
            initial_capital=mm.initial_capital,
        )
    return mm


def parse_strategy_mm(path: Path | str | bytes) -> StrategyMM:
    """Parsea ``.xml`` (directo) o ``.sqx`` (ZIP).

    Acepta ruta/Path o ``bytes`` (lo que da ``st.file_uploader``).
    Rechaza .sqx de portfolio con error explícito.
    """
    # bytes (uploader) -> zip si empieza con 'PK', sino xml
    if isinstance(path, (bytes, bytearray)):
        data = bytes(path)
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return _parse_sqx(z)
        return _read_strategy_xml(data)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".sqx":
        with zipfile.ZipFile(p) as z:
            return _parse_sqx(z)
    return _read_strategy_xml(p.read_bytes())
