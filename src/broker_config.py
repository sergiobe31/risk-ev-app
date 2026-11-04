"""Configuracion operativa del broker / prop firm.

Carga las reglas de Axi Select y los specs de simbolos desde YAML, con
validacion estructural via Pydantic. Separado de src/config.py porque
src/config.py es CANON METODOLOGICO (constantes Sesion4); este modulo
es OPERATIVO (cambia segun broker/firma).

Uso:
    from src.broker_config import load_axi_config, load_broker_config

    axi = load_axi_config()
    incubation = axi.get_phase("incubation")
    assert incubation.max_loss_pct == -7.0

    broker = load_broker_config()
    nas = broker.get_symbol("NAS100")
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Raiz del proyecto y rutas por defecto.
ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = ROOT / "config"
AXI_SELECT_YAML: Path = CONFIG_DIR / "axi_select.yaml"
BROKER_AXI_YAML: Path = CONFIG_DIR / "broker_axi.yaml"


# ---------------------------------------------------------------------------
# Modelos Axi Select
# ---------------------------------------------------------------------------

class HFTProxy(BaseModel):
    """Umbrales proxy para detectar HFT en el validador.

    No son fail automatico — son warnings configurables. La regla AXI prohibe
    HFT/latency arbitrage, no scalping legitimo.
    """
    model_config = ConfigDict(extra="forbid")

    avg_trade_duration_warn_minutes: int = Field(ge=0)
    trades_per_day_per_symbol_warn: int = Field(ge=0)
    min_holding_seconds_hard_flag: int = Field(ge=0)


class SymbolEligibility(BaseModel):
    """Tabla oficial AXI de simbolos elegibles para copiar a Allocation Account."""
    model_config = ConfigDict(extra="forbid")

    forex: str | list[str]
    indices: str | list[str]
    index_futures: str | list[str]
    commodities_spot: list[str]
    metals_spot: list[str]
    crypto: list[str]
    stocks: list[str]
    excluded: dict[str, str | list[str]]


class AXIProgram(BaseModel):
    """Reglas globales del programa (independientes de etapa)."""
    model_config = ConfigDict(extra="forbid")

    quarantine_days: int = Field(ge=0)
    quarantine_applies_to: list[str]
    max_quarantines_before_seed_reset: int = Field(ge=1)
    news_restriction: bool
    weekend_hold_allowed: bool
    hedging_allowed: bool
    hedge_counts_as_unique_trade: bool
    edge_score_reset_period_days: int = Field(ge=0)
    hft_proxy: HFTProxy
    symbol_eligibility: SymbolEligibility


class AXIPhase(BaseModel):
    """Una etapa del programa Axi Select.

    Campos `null`-ables (profit_target_pct, min_days, min_unique_trades) reflejan
    el caso Pro M, que no tiene criterios adicionales — solo Max Loss -10%.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    order: int = Field(ge=1, le=6)
    min_equity_usd: int = Field(gt=0)
    edge_score_min: int = Field(ge=0, le=100)
    max_funding_multiplier: int = Field(gt=0)
    max_funding_usd: int = Field(gt=0)
    profit_share_pct: float = Field(ge=0.0, le=100.0)
    profit_target_pct: float | None
    min_days: int | None
    min_unique_trades: int | None
    leverage: int = Field(gt=0)
    max_loss_pct: float = Field(lt=0.0)
    quarantine_applies: bool
    previous_phase: str | None

    def allocation_at(self, equity_usd: float) -> float:
        """Calcula la Allocation Account para un equity dado en esta etapa.

        Formula: min(equity * max_multiplier, max_funding).
        """
        return min(equity_usd * self.max_funding_multiplier, self.max_funding_usd)


class AXIConfig(BaseModel):
    """Bundle completo de reglas Axi Select."""
    model_config = ConfigDict(extra="forbid")

    prop_firm: str
    account_currency: str
    program: AXIProgram
    phases: list[AXIPhase]

    @model_validator(mode="after")
    def _validate_phase_chain(self) -> "AXIConfig":
        # Order tiene que ser 1..6 sin huecos
        orders = sorted(p.order for p in self.phases)
        if orders != list(range(1, len(self.phases) + 1)):
            raise ValueError(f"phases.order debe ser consecutivo 1..N, got {orders}")

        # previous_phase tiene que apuntar a una phase existente o ser null en seed
        names = {p.name for p in self.phases}
        for p in self.phases:
            if p.previous_phase is not None and p.previous_phase not in names:
                raise ValueError(
                    f"Phase {p.name}: previous_phase '{p.previous_phase}' no existe"
                )
            if p.order == 1 and p.previous_phase is not None:
                raise ValueError(f"Phase {p.name} es order=1, previous_phase debe ser null")
        return self

    def get_phase(self, name: str) -> AXIPhase:
        """Devuelve la etapa por nombre. Lanza KeyError si no existe."""
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(f"Phase '{name}' no encontrada. Disponibles: {[p.name for p in self.phases]}")

    def next_phase(self, current: str) -> AXIPhase | None:
        """Devuelve la siguiente etapa, o None si current es la ultima."""
        cur = self.get_phase(current)
        for p in self.phases:
            if p.previous_phase == cur.name:
                return p
        return None


# ---------------------------------------------------------------------------
# Modelos Broker (specs por simbolo)
# ---------------------------------------------------------------------------

SymbolCategory = Literal["forex", "index", "crypto", "metal", "commodity", "stock"]

# Tipos de swap observados en AXI (MT5):
#   percent_open_price   — % anualizado sobre el nominal calculado con el precio
#                          de apertura del trade (indices: NAS100, GER40, JPN225,
#                          AUS200).
#   percent_current_price— % anualizado sobre el nominal recalculado cada noche
#                          con el precio actual (cripto: BTCUSD, ETHUSD).
#   points               — swap fijo en points por noche, indiferente al precio
#                          (forex .pro: EURUSD.pro, USDCHF.pro).
SwapType = Literal["percent_open_price", "percent_current_price", "points"]

# Tier de cuenta AXI. Determina si los simbolos forex/bullion llevan sufijo
# ".pro" + comision. En indices/cripto no hay diferencia entre Standard y Pro.
AccountTier = Literal["standard", "pro"]


class SpreadSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_utc: str
    spread_points: float = Field(ge=0)


class SessionWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open: str
    close: str


class BrokerSymbol(BaseModel):
    """Specs de un simbolo capturados desde MT5 demo AXI.

    `swap_long_value` y `swap_short_value` se interpretan segun `swap_type`:
      - "percent_open_price" / "percent_current_price": % anualizado (e.g. -20.0)
      - "points": valor fijo de swap por noche en points (e.g. -7.7)
    Pueden ser positivos (carry favorable) o negativos (coste).
    """
    model_config = ConfigDict(extra="forbid")

    ticker_axi: str
    ticker_cascade: str | None = None
    category: SymbolCategory
    eligible_for_allocation: bool

    # Divisa de cotizacion del simbolo (USD, EUR, JPY, AUD, etc.). Si no coincide
    # con la divisa de la cuenta, hay conversion implicita en P/L y margin.
    quote_currency: str = Field(min_length=3, max_length=4)

    # Contrato y precision
    contract_size: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    pip_size: float = Field(gt=0)         # 1 pip puede ser N ticks (forex 5dig)
    tick_value_usd: float = Field(gt=0)
    point_value_usd: float = Field(gt=0)  # USD por 1.0 de precio por 1.0 lote = contract_size × (cotizacion->USD)

    # Sizing
    min_lot: float = Field(gt=0)
    lot_step: float = Field(gt=0)
    max_lot: float = Field(gt=0)

    # Coste por trade
    typical_spread_points: float = Field(ge=0)
    worst_spread_points: float = Field(ge=0)
    spread_samples: list[SpreadSample] = Field(default_factory=list)
    commission_usd_per_lot_per_side: float = Field(ge=0)
    # Política SQX por símbolo (sqx/audit_cfx_symbol_params.py — fail-closed si ausente)
    sqx_hp_spread_points: float | None = Field(default=None, ge=0)
    sqx_mc_spread_range: list[float] | None = None
    sqx_mc_slippage_range: list[float] | None = None

    # Swap
    swap_type: SwapType
    swap_long_value: float
    swap_short_value: float
    triple_swap_day: Literal["monday", "tuesday", "wednesday", "thursday", "friday"]

    # Operativa — sesiones en HORA SERVIDOR AXI (EST+7), NO UTC. Por-dia o por-tramo
    # (mon_thu / fri / sun_fri / sat / mon_fri_am / mon_fri_pm) segun el instrumento.
    sessions_broker_est7: dict[str, SessionWindow] = Field(default_factory=dict)
    stop_level_points: float = Field(ge=0)
    margin_currency: str = "USD"
    leverage_max: int = Field(gt=0)
    notes: str = ""

    @model_validator(mode="after")
    def _check_spread_and_lot(self) -> "BrokerSymbol":
        if self.worst_spread_points < self.typical_spread_points:
            raise ValueError(
                f"{self.ticker_axi}: worst_spread_points ({self.worst_spread_points}) "
                f"< typical_spread_points ({self.typical_spread_points})"
            )
        if self.max_lot < self.min_lot:
            raise ValueError(
                f"{self.ticker_axi}: max_lot ({self.max_lot}) < min_lot ({self.min_lot})"
            )
        return self

    # -----------------------------------------------------------------------
    # Helpers para configurar SQX (calculan los valores listos para meter
    # en la pantalla "Commission & swap settings")
    # -----------------------------------------------------------------------

    DAYS_PER_YEAR: ClassVar[int] = 360  # convencion mercado swap anualizado

    def swap_per_day_per_lot(
        self, current_price: float, fx_quote_to_usd: float = 1.0,
    ) -> tuple[float, float]:
        """Calcula swap LONG y SHORT en USD/dia por LOTE estandar.

        Estos son los valores a meter en SQX en "Swap type: money" cuando
        la cuenta esta en USD. Para simbolos en divisa no-USD (JPN225 en
        JPY), pasar `fx_quote_to_usd` con el cross actual (e.g. 1/155 para JPY).

        Returns:
            (long_usd_per_day, short_usd_per_day)
        """
        if self.swap_type == "points":
            # Valor fijo en points. Convertir a USD via tick_value.
            long_per_lot = self.swap_long_value * self.tick_value_usd
            short_per_lot = self.swap_short_value * self.tick_value_usd
        elif self.swap_type in ("percent_open_price", "percent_current_price"):
            # % anualizado sobre nominal. Daily = nominal * % / DAYS_PER_YEAR.
            nominal_quote = current_price * self.contract_size
            long_per_lot = nominal_quote * self.swap_long_value / 100 / self.DAYS_PER_YEAR
            short_per_lot = nominal_quote * self.swap_short_value / 100 / self.DAYS_PER_YEAR
        else:
            raise ValueError(f"swap_type desconocido: {self.swap_type}")
        return long_per_lot * fx_quote_to_usd, short_per_lot * fx_quote_to_usd

    def spread_usd_per_lot(self) -> tuple[float, float]:
        """Devuelve (typical_usd, worst_usd) para 1 lote estandar."""
        return (
            self.typical_spread_points * self.tick_value_usd,
            self.worst_spread_points * self.tick_value_usd,
        )

    def commission_round_trip_usd(self) -> float:
        """Comision round-trip (apertura + cierre) por lote."""
        return 2 * self.commission_usd_per_lot_per_side

    def sqx_settings(
        self, current_price: float, fx_quote_to_usd: float = 1.0,
        rollout_hour_utc: str = "22:00",
    ) -> dict:
        """Bloque listo para copiar a la pantalla Commission&Swap de SQX.

        Args:
            current_price: precio actual del simbolo (en su quote_currency).
            fx_quote_to_usd: cross para convertir nominal a USD si el simbolo
                no cotiza en USD. Default 1.0 (simbolo USD-quoted).
            rollout_hour_utc: hora UTC del rollover broker. Default 22:00.

        Returns dict con campos exactos del SQX UI.
        """
        long_swap, short_swap = self.swap_per_day_per_lot(current_price, fx_quote_to_usd)
        typical_spread_usd, worst_spread_usd = self.spread_usd_per_lot()
        return {
            "ticker": self.ticker_axi,
            "commission": {
                # SIEMPRE "Size based" (escala con size). "Per trade" cobra fijo
                # y NO escala = bug (convención verificada con la comisión de SQX).
                "mode": "Size based" if self.commission_usd_per_lot_per_side > 0 else "None",
                "value_usd_per_lot_per_side": self.commission_usd_per_lot_per_side,
                "round_trip_usd_per_lot": self.commission_round_trip_usd(),
                # Valor a teclear en el campo SQX "Size based" = round-trip (4.5),
                # no el per-side (SQX cobra ese valor por round-trip por lote).
                "sqx_size_based_value": self.commission_round_trip_usd(),
            },
            "swap": {
                "type": "money",  # SQX dropdown — coincide con todos los casos
                "long_usd_per_day_per_lot": round(long_swap, 4),
                "short_usd_per_day_per_lot": round(short_swap, 4),
                "triple_swap_on": self.triple_swap_day.upper(),
                "rollout_hour_utc": rollout_hour_utc,
            },
            "spread": {
                "typical_points": self.typical_spread_points,
                "worst_points": self.worst_spread_points,
                "typical_usd_per_lot": round(typical_spread_usd, 4),
                "worst_usd_per_lot": round(worst_spread_usd, 4),
            },
            "context": {
                "current_price_assumed": current_price,
                "fx_quote_to_usd_assumed": fx_quote_to_usd,
                "quote_currency": self.quote_currency,
                "swap_type_internal": self.swap_type,
            },
        }

    def format_sqx_block(
        self, current_price: float, fx_quote_to_usd: float = 1.0,
        rollout_hour_utc: str = "22:00",
    ) -> str:
        """Misma info que sqx_settings() pero como texto legible para copy-paste."""
        s = self.sqx_settings(current_price, fx_quote_to_usd, rollout_hour_utc)
        lines = [
            f"=== SQX settings for {s['ticker']} ===",
            f"(precio asumido: {current_price} {self.quote_currency}"
            f"{', fx=' + str(fx_quote_to_usd) if fx_quote_to_usd != 1.0 else ''})",
            "",
            "COMMISSION:",
            f"  Mode: {s['commission']['mode']}",
            f"  Value: ${s['commission']['value_usd_per_lot_per_side']} per lot per side"
            f"  (round-trip ${s['commission']['round_trip_usd_per_lot']})",
            "",
            "SWAP:",
            f"  Use: ON",
            f"  Swap type: {s['swap']['type']}",
            f"  Long:  ${s['swap']['long_usd_per_day_per_lot']:>8.4f} per day per lot",
            f"  Short: ${s['swap']['short_usd_per_day_per_lot']:>8.4f} per day per lot",
            f"  Triple swap on: {s['swap']['triple_swap_on']}",
            f"  Rollout hour: {s['swap']['rollout_hour_utc']}",
            "",
            "SPREAD (referencia, SQX no lo configura aqui):",
            f"  Typical: {s['spread']['typical_points']:.0f} points "
            f"(${s['spread']['typical_usd_per_lot']:.4f}/lote)",
            f"  Worst:   {s['spread']['worst_points']:.0f} points "
            f"(${s['spread']['worst_usd_per_lot']:.4f}/lote)",
        ]
        return "\n".join(lines)


class BrokerConfig(BaseModel):
    """Bundle de specs broker. Puede contener lista vacia de simbolos."""
    model_config = ConfigDict(extra="forbid")

    broker: str
    account_tier: AccountTier
    captured_at: date
    reconciled_at: date | None = None    # fecha de reconciliacion contra el export del script MT5
    symbols: list[BrokerSymbol] = Field(default_factory=list)

    def get_symbol(self, ticker_axi: str) -> BrokerSymbol:
        """Devuelve symbol spec por ticker AXI. KeyError si no existe."""
        for s in self.symbols:
            if s.ticker_axi == ticker_axi:
                return s
        raise KeyError(
            f"Symbol '{ticker_axi}' no encontrado. Disponibles: "
            f"{[s.ticker_axi for s in self.symbols]}"
        )

    def has_symbol(self, ticker_axi: str) -> bool:
        return any(s.ticker_axi == ticker_axi for s in self.symbols)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_axi_config(path: Path = AXI_SELECT_YAML) -> AXIConfig:
    """Carga config/axi_select.yaml y devuelve AXIConfig validado."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AXIConfig.model_validate(raw)


def load_broker_config(path: Path = BROKER_AXI_YAML) -> BrokerConfig:
    """Carga config/broker_axi.yaml y devuelve BrokerConfig validado.

    La lista de simbolos puede estar vacia — el archivo es funcional desde el
    primer dia para arrancar el validador y se va rellenando en MT5 demo.
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # Si symbols viene como None (YAML `symbols:` sin items), normalizar a []
    if raw.get("symbols") is None:
        raw["symbols"] = []
    return BrokerConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Registro de cortes IS/OOS por campaña de minado (config/oos.yaml)
# ---------------------------------------------------------------------------

OOS_YAML: Path = CONFIG_DIR / "oos.yaml"


class OOSCut(BaseModel):
    """Corte IS/OOS registrado para una campaña de minado.

    `cut` = dateFrom de la cola OOS final del build SQX. La evaluación MC usa
    solo bloques con inicio >= cut (adjudicación M2-8: evaluar sobre el IS del
    PM infla P(pass), diferencial en k).
    """
    model_config = ConfigDict(extra="forbid")

    cut: date
    source: str = ""
    registered: date | None = None


def load_oos_registry(path: Path = OOS_YAML) -> dict[str, OOSCut]:
    """Carga config/oos.yaml → {campaña: OOSCut}. Vacío si no hay campañas."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    campaigns = raw.get("campaigns") or {}
    return {k: OOSCut.model_validate(v) for k, v in campaigns.items()}
