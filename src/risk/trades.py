"""Carga y validacion de la lista de trades exportada de SQX (orderstocsv).

El CSV orderstocsv (16 columnas, ``;``-separado) es el contrato de artefacto
entre SQX y este motor. Formato verificado en
``docs/sqx_cli_commands.md`` §6 y ``sqx/sts_pilot/run_pilot.sh``::

    Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;
    Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment

Convencion decimal: depende del flag ``usecomma`` pasado a orderstocsv, PERO
no es uniforme por fichero: SQX B143 con ``usecomma=true`` exporta los PRECIOS
y el Size con punto (formato invariante) y solo el DINERO (PnL/Balance/
MAE/MFE) con coma europea (los 101 CSV de PM_nstrategies son asi). Sin el
flag -> todo con ``.``. Un decimal unico por fichero corrompe silenciosamente
la mitad de las columnas en el caso mixto (``"1.09195"`` leido con coma ->
``109195``, ``"0.22"`` -> ``22``), asi que husmeamos el separador POR COLUMNA
(:func:`_sniff_decimal_column`) con el sniffer global como fallback para
columnas sin evidencia (solo enteros).

Linealidad: con sizing ATR-risk dinamico, el retorno % por trade es
``r_n(f) = f * y_n`` donde ``y_n`` es el R-multiplo neto (f-independiente; la
equity se cancela al componer). Desde el CSV + el ``f_ref`` del backtest de
referencia, todo es derivable:

    equity_before_n = Balance_n - PnL_n        (Balance es post-trade)
    r_n        = PnL_n / equity_before_n       (retorno % realizado a f_ref)
    y_n        = r_n / f_ref                   (R-multiplo neto, f-indep.)
    slInMoney_n= f_ref * equity_before_n / Size_n   (USD de riesgo por lote)
    pnl_per_lot_n = PnL_n / Size_n             (PnL por lote; comision size-based incluida)

CAVEAT DE EXACTITUD (importante): la identidad es EXACTA a ``f = f_ref`` por
construction (es como se back-deriva ``slInMoney``). A ``f != f_ref`` es
aproximada cuando el ``Size`` de referencia fue lot-step-floored o
min_lot-clamped por SQX: el back-derivo ``slInMoney = f_ref*E/Size`` solo
coincide con el verdadero si ``Size`` fue el lote continuo natural. La
direccion peligrosa es el CLAMPING a min_lot (``Size = min_lot >
lots_cont_ref``), que sesga ``slInMoney`` BAJO y hace que el sweep
SOBREDIMENSIONE a otro ``f``. No se puede detectar desde el CSV solo; usa
:func:`src.risk.mm.reference_clamping_fraction` como heuristica y VALIDA el
sweep contra 2-3 re-runs de SQX antes de fiarte (gate del plan).

PRECONDICION: los exits deben ser price/time/ATR (p.ej. ExitAfterBars=N,
SL=k*ATR, señales de indicador — cualquier N, no solo 1). Un exit
equity-contingent (trailing sobre equity, stop por drawdown, BE sobre PnL)
hace que la secuencia de trades dependa de ``f`` y rompe la linealidad para
esa estrategia. No se puede verificar desde el CSV solo -> se expone como
``has_equity_contingent_exit_hint`` (default None; el llamador/assert lo seta).
El MAE($) da el valle MTM del trade entero sea cual sea N.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema orderstocsv — mapeo tolerante por nombre normalizado (no por posicion).
# ---------------------------------------------------------------------------
_CANONICAL = [
    "ticket", "symbol", "type", "open_time", "open_price", "size",
    "close_time", "close_price", "pnl", "balance", "sample_type",
    "close_type", "mae", "mfe", "time_in_trade", "comment",
]

# Alias heuristicos (header normalizado -> canonico). Cubre variantes de
# capitalizacion y nombres exactos de SQX B143.
_ALIASES: dict[str, str] = {
    "ticket": "ticket", "order": "ticket", "ticketid": "ticket",
    "symbol": "symbol", "instrument": "symbol",
    "type": "type", "direction": "type", "trade_type": "type",
    "open time": "open_time", "opentime": "open_time", "open_time": "open_time",
    "open price": "open_price", "openprice": "open_price", "open_price": "open_price",
    "size": "size", "lots": "size", "volume": "size",
    "close time": "close_time", "closetime": "close_time", "close_time": "close_time",
    "close price": "close_price", "closeprice": "close_price", "close_price": "close_price",
    "profit/loss": "pnl", "profit loss": "pnl", "profit-loss": "pnl",
    "pnl": "pnl", "profit": "pnl", "net profit": "pnl", "profit/loss ": "pnl",
    "balance": "balance", "equity": "balance",
    "sample type": "sample_type", "sampletype": "sample_type", "sample_type": "sample_type",
    "close type": "close_type", "closetype": "close_type", "close_type": "close_type",
    "mae ($)": "mae", "mae": "mae", "mae($)": "mae", "mae_usd": "mae", "mae ($) ": "mae",
    "mfe ($)": "mfe", "mfe": "mfe", "mfe($)": "mfe", "mfe_usd": "mfe",
    "time in trade": "time_in_trade", "timeintrade": "time_in_trade",
    "time_in_trade": "time_in_trade", "duration": "time_in_trade",
    "comment": "comment", "comments": "comment",
}

# Columnas obligatorias para el motor de riesgo.
_REQUIRED: tuple[str, ...] = ("symbol", "size", "pnl", "balance")
# MAE solo se usa para el MTM de ruin; si falta, se desactiva el MTM intra-trade.
_OPTIONAL: tuple[str, ...] = ("mae", "mfe", "open_time", "close_time", "open_price")


def _norm(header: str) -> str:
    return header.strip().lower().replace("_", " ")


_DECIMAL_RE = __import__("re").compile(r"^-?\d+([,.])\d+$")


def sniff_decimal(raw: bytes | str) -> str:
    """Husmea el separador decimal del CSV orderstocsv.

    Devuelve ``','`` (europeo, ``usecomma=true``) o ``'.'``. Heuristica robusta:
    cuenta separadores en campos que son NUMEROS DECIMALES puros
    (regex ``^-?\\d+([,.])\\d+$``), ignorando fechas (``2024.01.01`` tiene varios
    puntos y no matchea) y enteros sin decimal. Si las comas dominan -> europeo.
    Default ``'.'`` si no hay campos decimales (enteros/fechas solo).

    Asume separador de campos ``';'`` (constante en orderstocsv usecomma o no).
    """
    if isinstance(raw, str):
        data = raw
    else:
        data = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in data.splitlines() if ln.strip()]
    n_comma = 0
    n_dot = 0
    for ln in lines[1:8]:  # primeros renglones de datos
        if ";" not in ln:
            continue
        for field in ln.split(";"):
            m = _DECIMAL_RE.match(field.strip().strip('"').strip())
            if m:
                if m.group(1) == ",":
                    n_comma += 1
                else:
                    n_dot += 1
    if n_comma == 0 and n_dot == 0:
        return "."
    return "," if n_comma > n_dot else "."


def _sniff_decimal_column(values: pd.Series, fallback: str) -> str:
    """Decimal de UNA columna numerica (misma heuristica regex que :func:`sniff_decimal`).

    SQX B143 con ``usecomma=true`` exporta MIXTO: precios/Size con punto y
    dinero con coma. Cada columna elige por evidencia propia (misma regla de
    mayoria); sin campos decimales en los primeros renglones (solo enteros o
    texto) hereda el decimal global del fichero (``fallback``).
    """
    n_comma = 0
    n_dot = 0
    for v in values.head(8):
        m = _DECIMAL_RE.match(str(v).strip().strip('"').strip())
        if m:
            if m.group(1) == ",":
                n_comma += 1
            else:
                n_dot += 1
    if n_comma == 0 and n_dot == 0:
        return fallback
    return "," if n_comma > n_dot else "."


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    n_trades: int
    decimal_char: str
    net_profit: float
    initial_equity: float
    final_balance: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TradeList:
    """Lista de trades parseada + arrays derivados listos para el motor.

    Todos los arrays estan alineados por indice de trade (0..n_trades-1) en
    orden cronologico (por ``close_time`` si existe, si no por orden de fila).
    """

    df: pd.DataFrame  # normalizado, columnas canonicas, numericas parseadas
    symbol: np.ndarray
    size: np.ndarray  # lotes
    pnl: np.ndarray  # PnL neto (USD), incluye comision+swap
    equity_before: np.ndarray  # equity justo antes de cada trade
    mae: np.ndarray  # flotante negativo en el peor punto del trade (<=0), 0 si desconocido
    close_time: pd.Series  # timestamp de cierre (NaT si ausente)
    open_price: np.ndarray  # precio de apertura (para margin clip; 0 si ausente)
    n_trades: int
    initial_equity: float  # E_0 = equity_before del primer trade
    decimal_char: str
    has_mae: bool
    validation: ValidationResult

    # --- Derivados que dependen de f_ref ---------------------------------
    def r_multiples(self, f_ref: float) -> np.ndarray:
        """R-multiplos netos ``y_n`` (f-independientes bajo linealidad).

        ``y_n = r_n / f_ref`` donde ``r_n = PnL_n / equity_before_n``.
        """
        if f_ref <= 0:
            raise ValueError("f_ref debe ser > 0 (fraccion de riesgo de referencia)")
        r = self.pnl / self.equity_before
        return r / f_ref

    def sl_in_money(self, f_ref: float) -> np.ndarray:
        """USD de riesgo por lote por trade, back-solveados del run de referencia.

        ``slInMoney_n = f_ref * equity_before_n / Size_n``.
        """
        if f_ref <= 0:
            raise ValueError("f_ref debe ser > 0")
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = np.where(self.size > 0, f_ref * self.equity_before / self.size, np.nan)
        return sl

    def pnl_per_lot(self) -> np.ndarray:
        """PnL neto por lote (comision size-based ya incluida en PnL)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.size > 0, self.pnl / self.size, np.nan)


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapped = {}
    for col in df.columns:
        key = _norm(col).replace("  ", " ")
        mapped[col] = _ALIASES.get(key) or _ALIASES.get(key.replace(" ", "_"))
    df = df.rename(columns={k: v for k, v in mapped.items() if v})
    return df


def _to_float(series: pd.Series, decimal: str) -> pd.Series:
    """Fuerza numerico: reemplaza separador de miles (raro en SQX) y parsea."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    if decimal == ",":
        # europeo: ',' decimal, '.' ignorado como separador de miles
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_trade_list(
    source: Path | str | bytes | BinaryIO,
    *,
    f_ref: float | None = None,
    net_profit_reference: float | None = None,
) -> TradeList:
    """Carga + valida un CSV orderstocsv y devuelve :class:`TradeList`.

    Args:
        source: ruta, bytes o file-like del CSV (``;``-separado).
        f_ref: fraccion de riesgo del backtest de referencia (p.ej. 0.01 para
            1%). No se requiere para cargar, pero los derivados ``r_multiples``
            lo piden; se guarda para conveniencia si se da.
        net_profit_reference: si se da, se cruza contra ``sum(PnL)`` y se emite
            warning/error si difiere >1% (trust gate).

    Raises:
        ValueError: schema incompleto, PnL no parsea, o net profit fuera de
            tolerancia vs referencia.
    """
    raw: bytes
    if isinstance(source, (str, Path)) and Path(source).exists():
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read()  # type: ignore[union-attr]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        raise ValueError(f"source no soportado: {type(source)!r}")

    decimal = sniff_decimal(raw)
    # dtype=str: leemos crudo y re-parseamos en _to_float con el decimal husmeado
    # (asi el arg decimal de read_csv es irrelevante y evitamos mismatches).
    df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str)
    df = _rename_columns(df)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV orderstocsv incompleto: faltan columnas {missing}. "
            f"Detectadas: {list(df.columns)}"
        )

    # Parseo numerico robusto (rehacemos con el decimal husmeado por si acaso).
    # El decimal se husmea POR COLUMNA (el export B143 usecomma es mixto:
    # precios/Size con punto, dinero con coma); `decimal` es solo el fallback.
    for col in ("size", "open_price", "close_price", "pnl", "balance", "mae", "mfe", "time_in_trade"):
        if col in df.columns:
            df[col] = _to_float(df[col], _sniff_decimal_column(df[col], decimal))
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    else:
        df["close_time"] = pd.NaT

    # Orden cronologico por cierre (si no hay timestamp, orden de fila).
    if df["close_time"].notna().any():
        df = df.sort_values("close_time", kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Validacion.
    warnings: list[str] = []
    errors: list[str] = []
    n_endtest = 0
    if "close_type" in df.columns:
        n_endtest = int(
            df["close_type"].astype(str).str.contains("EndTest", case=False, na=False).sum()
        )
    n_pnl_nan = int(df["pnl"].isna().sum())
    # Filas con PnL NaN (posicion abierta al final del backtest, Close type
    # EndTest, sin PnL realizado) -> se DESCARTAN, no son trades cerrados.
    if n_pnl_nan:
        warnings.append(
            f"{n_pnl_nan} trade(s) sin PnL realizado descartado(s) "
            f"({n_endtest} con Close type EndTest = posicion abierta al cierre)."
        )
    if (df["size"].fillna(-1) <= 0).any():
        errors.append(f"{int((df['size'].fillna(-1) <= 0).sum())} trades con Size <= 0")
    df = df.dropna(subset=["pnl", "size", "balance"])
    df = df[df["size"] > 0].reset_index(drop=True)

    pnl = df["pnl"].to_numpy(dtype=float)
    balance = df["balance"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)
    equity_before = balance - pnl  # Balance es post-trade
    initial_equity = float(equity_before[0]) if len(equity_before) else 0.0
    net_profit = float(pnl.sum())
    final_balance = float(balance[-1]) if len(balance) else initial_equity

    has_mae = "mae" in df.columns and df["mae"].notna().any()
    if has_mae:
        mae = df["mae"].to_numpy(dtype=float)
        # MAE($) = excursion ADVERSA: siempre perdida o 0 (nunca ganancia; el
        # MFE cubre lo favorable). Forzar no-positivo INCONDICIONALMENTE con
        # -abs(): cubre la convencion signada (negativo) y la de magnitud
        # (positivo), y evita que un +suelto sobreviva y se trate como
        # ganancia en el MTM de ruin (bug: el signo se normalizaba solo por la
        # media de la columna, y entradas positivas en columnas de media<0
        # colaban).
        mae = -np.abs(np.nan_to_num(mae, nan=0.0))
    else:
        mae = np.zeros_like(pnl)
        warnings.append("Sin columna MAE ($): ruin MTM intra-trade desactivado (solo realized).")

    open_price = df["open_price"].to_numpy(dtype=float) if "open_price" in df.columns else np.zeros_like(pnl)
    symbol = df["symbol"].astype(str).to_numpy()

    if net_profit_reference is not None and initial_equity > 0:
        rel = abs(net_profit - net_profit_reference) / max(abs(net_profit_reference), 1.0)
        if rel > 0.01:
            errors.append(
                f"sum(Profit/Loss)={net_profit:.2f} difiere {rel:.1%} del "
                f"referencia {net_profit_reference:.2f} (gate de confianza)"
            )

    ok = not errors and len(pnl) > 0
    validation = ValidationResult(
        ok=ok,
        n_trades=len(pnl),
        decimal_char=decimal,
        net_profit=net_profit,
        initial_equity=initial_equity,
        final_balance=final_balance,
        warnings=warnings,
        errors=errors,
    )
    if not ok:
        raise ValueError("CSV invalido: " + " | ".join(errors))

    return TradeList(
        df=df,
        symbol=symbol,
        size=size,
        pnl=pnl,
        equity_before=equity_before,
        mae=mae,
        close_time=df["close_time"],
        open_price=open_price,
        n_trades=len(pnl),
        initial_equity=initial_equity,
        decimal_char=decimal,
        has_mae=has_mae,
        validation=validation,
    )