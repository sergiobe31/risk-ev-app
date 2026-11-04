"""Tests for src/broker_config.py.

Cubre: carga real de config/axi_select.yaml y config/broker_axi.yaml, business
rules de los Pydantic models, helpers get_phase / next_phase / allocation_at.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.broker_config import (
    AXI_SELECT_YAML,
    BROKER_AXI_YAML,
    AXIConfig,
    AXIPhase,
    BrokerConfig,
    BrokerSymbol,
    load_axi_config,
    load_broker_config,
)


# ---------------------------------------------------------------------------
# Carga real desde config/
# ---------------------------------------------------------------------------

def test_load_axi_config_real_yaml() -> None:
    cfg = load_axi_config()
    assert cfg.prop_firm == "Axi Select"
    assert cfg.account_currency == "USD"
    assert len(cfg.phases) == 6


def test_load_broker_config_real_yaml() -> None:
    cfg = load_broker_config()
    assert cfg.broker.startswith("Axi Select")
    assert cfg.account_tier == "pro"
    # 4 seed + USATECHIDXUSD.sa (2026-07-17) + DAX40.fs (2026-07-18) +
    # XAGUSD.pa (2026-08-30). Los dos primeros anadidos para el modo portfolio
    # de la app de riesgo; plata añadida como metal spot operable.
    assert len(cfg.symbols) == 7


# ---------------------------------------------------------------------------
# Estructura de fases AXI
# ---------------------------------------------------------------------------

def test_phases_in_correct_order() -> None:
    cfg = load_axi_config()
    expected = ["seed", "incubation", "acceleration", "pro", "pro_500", "pro_m"]
    actual = [p.name for p in sorted(cfg.phases, key=lambda x: x.order)]
    assert actual == expected


def test_seed_has_no_previous_phase() -> None:
    cfg = load_axi_config()
    seed = cfg.get_phase("seed")
    assert seed.previous_phase is None


def test_pro_m_has_max_loss_minus_10() -> None:
    cfg = load_axi_config()
    pro_m = cfg.get_phase("pro_m")
    assert pro_m.max_loss_pct == -10.0


def test_other_phases_have_max_loss_minus_7() -> None:
    cfg = load_axi_config()
    for name in ["seed", "incubation", "acceleration", "pro", "pro_500"]:
        assert cfg.get_phase(name).max_loss_pct == -7.0


def test_pro_m_has_no_advancement_criteria() -> None:
    cfg = load_axi_config()
    pro_m = cfg.get_phase("pro_m")
    assert pro_m.profit_target_pct is None
    assert pro_m.min_days is None
    assert pro_m.min_unique_trades is None


def test_quarantine_applies_only_to_incubation_and_acceleration() -> None:
    cfg = load_axi_config()
    for name in ["incubation", "acceleration"]:
        assert cfg.get_phase(name).quarantine_applies is True
    for name in ["seed", "pro", "pro_500", "pro_m"]:
        assert cfg.get_phase(name).quarantine_applies is False


def test_leverage_seed_1000_resto_100() -> None:
    # Axi Select (PDF oficial + axi.com, verificado 2026-10-09): Seed hasta 1000:1, resto 100:1.
    cfg = load_axi_config()
    assert cfg.get_phase("seed").leverage == 1000
    for n in ["incubation", "acceleration", "pro", "pro_500", "pro_m"]:
        assert cfg.get_phase(n).leverage == 100


def test_profit_share_progression() -> None:
    cfg = load_axi_config()
    shares = [cfg.get_phase(n).profit_share_pct
              for n in ["seed", "incubation", "acceleration", "pro", "pro_500", "pro_m"]]
    assert shares == [0.0, 40.0, 50.0, 60.0, 70.0, 80.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_next_phase_chain() -> None:
    cfg = load_axi_config()
    assert cfg.next_phase("seed").name == "incubation"
    assert cfg.next_phase("incubation").name == "acceleration"
    assert cfg.next_phase("acceleration").name == "pro"
    assert cfg.next_phase("pro").name == "pro_500"
    assert cfg.next_phase("pro_500").name == "pro_m"
    assert cfg.next_phase("pro_m") is None


def test_get_phase_raises_on_unknown() -> None:
    cfg = load_axi_config()
    with pytest.raises(KeyError):
        cfg.get_phase("nonexistent")


def test_allocation_at_below_cap() -> None:
    # En Incubation con $1000 → Allocation = min(1000*10, 20000) = 10000
    cfg = load_axi_config()
    inc = cfg.get_phase("incubation")
    assert inc.allocation_at(1000.0) == 10000.0


def test_allocation_at_hits_cap() -> None:
    # En Incubation con $4000 → Allocation = min(4000*10, 20000) = 20000 (cap)
    cfg = load_axi_config()
    inc = cfg.get_phase("incubation")
    assert inc.allocation_at(4000.0) == 20000.0
    assert inc.allocation_at(10000.0) == 20000.0  # sigue capado


def test_allocation_at_seed() -> None:
    cfg = load_axi_config()
    seed = cfg.get_phase("seed")
    # equity 500 × 10 = 5000 (en el cap)
    assert seed.allocation_at(500.0) == 5000.0


# ---------------------------------------------------------------------------
# Program-level rules
# ---------------------------------------------------------------------------

def test_program_rules() -> None:
    cfg = load_axi_config()
    p = cfg.program
    assert p.quarantine_days == 7
    assert p.quarantine_applies_to == ["incubation", "acceleration"]
    assert p.max_quarantines_before_seed_reset == 3
    assert p.news_restriction is False
    assert p.weekend_hold_allowed is True
    assert p.hedging_allowed is True
    assert p.hedge_counts_as_unique_trade is False


def test_symbol_eligibility_crypto_btc_eth() -> None:
    cfg = load_axi_config()
    assert "BTCUSD" in cfg.program.symbol_eligibility.crypto
    assert "ETHUSD" in cfg.program.symbol_eligibility.crypto


def test_symbol_eligibility_stocks_list() -> None:
    cfg = load_axi_config()
    stocks = cfg.program.symbol_eligibility.stocks
    for ticker in ["META", "MSFT", "AMZN", "GOOGL", "TSLA", "NFLX", "AAPL"]:
        assert ticker in stocks


# ---------------------------------------------------------------------------
# Pydantic validators — rechazos esperados
# ---------------------------------------------------------------------------

def test_phase_rejects_positive_max_loss() -> None:
    with pytest.raises(ValidationError):
        AXIPhase(
            name="bad", display_name="Bad", order=1, min_equity_usd=500,
            edge_score_min=50, max_funding_multiplier=10, max_funding_usd=5000,
            profit_share_pct=0.0, profit_target_pct=7.0, min_days=30,
            min_unique_trades=20, leverage=1000,
            max_loss_pct=7.0,  # positivo → debe fallar
            quarantine_applies=False, previous_phase=None,
        )


def test_axiconfig_rejects_nonconsecutive_orders() -> None:
    """Si fases tienen order [1, 2, 4] (con hueco) → fail."""
    cfg = load_axi_config().model_dump()
    cfg["phases"] = cfg["phases"][:3]
    cfg["phases"][2]["order"] = 4  # introducir hueco (1, 2, 4 en vez de 1, 2, 3)
    with pytest.raises(ValidationError):
        AXIConfig.model_validate(cfg)


def test_axiconfig_rejects_unknown_previous_phase() -> None:
    cfg = load_axi_config().model_dump()
    cfg["phases"][1]["previous_phase"] = "ghost_phase"
    with pytest.raises(ValidationError):
        AXIConfig.model_validate(cfg)


# ---------------------------------------------------------------------------
# BrokerSymbol validation
# ---------------------------------------------------------------------------

def _valid_symbol_kwargs() -> dict:
    return dict(
        ticker_axi="NAS100", ticker_cascade="USATECHIDXUSD", category="index",
        eligible_for_allocation=True, quote_currency="USD",
        contract_size=1.0, tick_size=0.01, pip_size=1.0, tick_value_usd=0.01,
        point_value_usd=1.0,
        min_lot=0.10, lot_step=0.10, max_lot=100.0,
        typical_spread_points=100.0, worst_spread_points=350.0,
        commission_usd_per_lot_per_side=0.0,
        swap_type="percent_open_price",
        swap_long_value=-6.75, swap_short_value=0.75,
        triple_swap_day="friday", stop_level_points=50.0, leverage_max=100,
    )


def test_broker_symbol_valid() -> None:
    sym = BrokerSymbol(**_valid_symbol_kwargs())
    assert sym.ticker_axi == "NAS100"
    assert sym.quote_currency == "USD"
    assert sym.swap_type == "percent_open_price"


def test_broker_symbol_rejects_worst_lt_typical() -> None:
    kw = _valid_symbol_kwargs()
    kw["worst_spread_points"] = 50.0  # < typical 100
    with pytest.raises(ValidationError):
        BrokerSymbol(**kw)


def test_broker_symbol_rejects_max_lot_lt_min_lot() -> None:
    kw = _valid_symbol_kwargs()
    kw["max_lot"] = 0.05  # < min_lot 0.10
    with pytest.raises(ValidationError):
        BrokerSymbol(**kw)


def test_broker_symbol_rejects_unknown_category() -> None:
    kw = _valid_symbol_kwargs()
    kw["category"] = "unknown"
    with pytest.raises(ValidationError):
        BrokerSymbol(**kw)


def test_broker_symbol_rejects_unknown_swap_type() -> None:
    kw = _valid_symbol_kwargs()
    kw["swap_type"] = "logarithmic"  # no es uno de los 3 modos
    with pytest.raises(ValidationError):
        BrokerSymbol(**kw)


def test_broker_symbol_accepts_all_swap_types() -> None:
    for swap_type in ("percent_open_price", "percent_current_price", "points"):
        kw = _valid_symbol_kwargs()
        kw["swap_type"] = swap_type
        sym = BrokerSymbol(**kw)
        assert sym.swap_type == swap_type


def test_broker_config_get_and_has_symbol() -> None:
    cfg = BrokerConfig(
        broker="test", account_tier="pro", captured_at=date(2026, 5, 23),
        symbols=[BrokerSymbol(**_valid_symbol_kwargs())],
    )
    assert cfg.has_symbol("NAS100")
    assert not cfg.has_symbol("GHOST")
    sym = cfg.get_symbol("NAS100")
    assert sym.ticker_cascade == "USATECHIDXUSD"
    with pytest.raises(KeyError):
        cfg.get_symbol("GHOST")


def test_broker_config_rejects_unknown_account_tier() -> None:
    with pytest.raises(ValidationError):
        BrokerConfig(
            broker="test", account_tier="vip",  # no es standard ni pro
            captured_at=date(2026, 5, 23), symbols=[],
        )


# ---------------------------------------------------------------------------
# Carga real del broker_axi.yaml — cartera Seed minimalista (4 simbolos)
# ---------------------------------------------------------------------------

def test_broker_yaml_has_exactly_the_seed_plus_risk_app_symbols() -> None:
    """Decision miercoles 2026-06-24 (MINIMALISTA): los 4 tickers Seed, capturados
    del run Live Axi-US51-Live. Los no-seed (USTECH/USDCHF/AUS200) se borraron;
    viven en git + export del script MT5. Fuente: la decisión de tickers de la cuenta Seed.
    Excepcion posterior: USATECHIDXUSD.sa (2026-07-17) y DAX40.fs (2026-07-18) se
    re-anadieron NO para la cartera Seed sino para el modo portfolio de la app de
    riesgo (necesita resolver simbolo -> point value / OHLC).
    Excepcion posterior 2026-08-30: XAGUSD.pa se anade como metal spot operable
    (elegible en axi_select.yaml) con specs del export MT5 Axi-US51-Live."""
    SEED = {"EURUSD.pa", "ETHUSD.sa", "XAUUSD.pa", "NZDUSD.pa"}
    RISK_APP = {"USATECHIDXUSD.sa", "DAX40.fs"}
    EXTRA = {"XAGUSD.pa"}  # metal spot anadido 2026-08-30
    cfg = load_broker_config()
    tickers = {s.ticker_axi for s in cfg.symbols}
    assert tickers == SEED | RISK_APP | EXTRA
    # Los 4 seed + plata cargan sin error de validacion y son elegibles para copiar.
    for t in SEED | EXTRA:
        assert cfg.get_symbol(t).eligible_for_allocation is True


def test_broker_yaml_account_tier_pro() -> None:
    cfg = load_broker_config()
    assert cfg.account_tier == "pro"


def test_commission_invariant_pa_vs_sa() -> None:
    """Invariante AXI (confirmado usuario 2026-06-24, incluido XAUUSD): los .pa
    (FX + metal, cuenta Pro) cobran 4.5 round-trip = 2.25/lado; los .sa
    (indices + cripto) y .fs (indices-futuro) = 0. Se testea la PROPIEDAD por
    sufijo, no la lista literal."""
    cfg = load_broker_config()
    for s in cfg.symbols:
        if s.ticker_axi.endswith(".pa"):
            assert s.commission_usd_per_lot_per_side == 2.25, s.ticker_axi
        elif s.ticker_axi.endswith((".sa", ".fs")):
            assert s.commission_usd_per_lot_per_side == 0.0, s.ticker_axi
        else:
            raise AssertionError(f"ticker sin sufijo .pa/.sa/.fs reconocido: {s.ticker_axi}")


def test_swap_types_match_broker_observed() -> None:
    """Verifica que los swap_type de los 4 seed coinciden con las capturas.
    (Las ramas percent_open_price / fx-conversion las cubren los tests sinteticos
    de abajo, ya que ningun simbolo seed las ejerce.)"""
    cfg = load_broker_config()
    # Cripto: % precio actual
    assert cfg.get_symbol("ETHUSD.sa").swap_type == "percent_current_price"
    # Forex / oro: points
    assert cfg.get_symbol("EURUSD.pa").swap_type == "points"
    assert cfg.get_symbol("NZDUSD.pa").swap_type == "points"
    assert cfg.get_symbol("XAUUSD.pa").swap_type == "points"


def test_triple_swap_day_convention_split_fx_vs_index() -> None:
    """Cripto: viernes. FX/oro: miercoles (convencion T+2)."""
    cfg = load_broker_config()
    assert cfg.get_symbol("ETHUSD.sa").triple_swap_day == "friday"
    for ticker in ("EURUSD.pa", "NZDUSD.pa", "XAUUSD.pa"):
        assert cfg.get_symbol(ticker).triple_swap_day == "wednesday"


def test_quote_currencies_match_capture() -> None:
    """Los 4 seed cotizan todos en USD (esquivan el bug de doble-conversion del
    export, que solo afecta a no-USD)."""
    cfg = load_broker_config()
    for ticker in ("EURUSD.pa", "NZDUSD.pa", "XAUUSD.pa", "ETHUSD.sa"):
        assert cfg.get_symbol(ticker).quote_currency == "USD"


def test_ethusd_spread_matches_2024_2026_analysis() -> None:
    """Mediana 2024-2026 fue 1700 points; P95 capado en 2300."""
    eth = load_broker_config().get_symbol("ETHUSD.sa")
    assert eth.typical_spread_points == 1700
    assert eth.worst_spread_points == 2300


def test_all_symbols_eligible_for_allocation() -> None:
    """Los 4 simbolos seed estan dentro de la tabla de elegibilidad AXI Select."""
    cfg = load_broker_config()
    for s in cfg.symbols:
        assert s.eligible_for_allocation is True


# ---------------------------------------------------------------------------
# Helpers para configurar SQX
# ---------------------------------------------------------------------------

def test_swap_per_day_points_mode_eurusd() -> None:
    """EURUSD.pa swap_type=points: long × tick_value_usd."""
    eur = load_broker_config().get_symbol("EURUSD.pa")
    long_swap, short_swap = eur.swap_per_day_per_lot(current_price=1.10)
    assert long_swap == pytest.approx(eur.swap_long_value * eur.tick_value_usd, abs=0.01)
    assert short_swap == pytest.approx(eur.swap_short_value * eur.tick_value_usd, abs=0.01)


def test_swap_per_day_points_mode_xauusd() -> None:
    """XAUUSD.pa swap_type=points: long × tick_value_usd $1.0."""
    xau = load_broker_config().get_symbol("XAUUSD.pa")
    long_swap, short_swap = xau.swap_per_day_per_lot(current_price=4150.0)
    assert long_swap == pytest.approx(xau.swap_long_value * xau.tick_value_usd, abs=0.01)
    assert short_swap == pytest.approx(xau.swap_short_value * xau.tick_value_usd, abs=0.01)


def test_swap_per_day_percent_open_synthetic() -> None:
    """Rama swap_type=percent_open_price (indice -6.75% anual sobre nominal).
    A precio $22k -> -$4.125/dia/lote. Simbolo SINTETICO: ningun seed la ejerce
    (cuando se reanada un indice como USTECH/AUS200 se cubre desde el YAML)."""
    sym = BrokerSymbol(**_valid_symbol_kwargs())  # index, percent_open_price, -6.75/0.75
    long_swap, short_swap = sym.swap_per_day_per_lot(current_price=22000.0)
    # 22000 * 1 (contract) * 0.0675 / 360 = $4.125
    assert long_swap == pytest.approx(-4.125, abs=0.01)
    assert short_swap == pytest.approx(0.4583, abs=0.01)


def test_swap_per_day_percent_current_ethusd() -> None:
    """ETHUSD swap=-20% anual. A precio $3500 → -$1.944/dia/lote."""
    eth = load_broker_config().get_symbol("ETHUSD.sa")
    long_swap, _ = eth.swap_per_day_per_lot(current_price=3500.0)
    # 3500 * 1 * 0.20 / 360 = $1.9444
    assert long_swap == pytest.approx(-1.9444, abs=0.01)


def test_swap_per_day_with_fx_conversion_synthetic() -> None:
    """Rama fx_quote_to_usd para simbolo no-USD (indice AUD). A precio 8800,
    swap -7.35% anual, AUDUSD=0.70 -> swap en USD. Simbolo SINTETICO: ningun seed
    cotiza fuera de USD (cuando se reanada AUS200 se cubre desde el YAML)."""
    kw = _valid_symbol_kwargs()
    kw["quote_currency"] = "AUD"
    kw["swap_long_value"] = -7.35
    kw["swap_short_value"] = 1.35
    sym = BrokerSymbol(**kw)
    long_swap, _ = sym.swap_per_day_per_lot(current_price=8800.0, fx_quote_to_usd=0.70)
    # 8800 * 1 (contract) * 0.0735 / 360 = 1.7967 AUD/dia ; × 0.70 = $1.2577
    assert long_swap == pytest.approx(-1.2577, abs=0.02)


def test_commission_round_trip() -> None:
    # AXI Select = 4.5 round-trip (facturado en entrada; modelado 2.25/lado × 2).
    # SQX SizeBased se pone a 4.5. El .sa (cripto ETHUSD) = 0.
    cfg = load_broker_config()
    for ticker in ("EURUSD.pa", "NZDUSD.pa", "XAUUSD.pa"):
        assert cfg.get_symbol(ticker).commission_round_trip_usd() == 4.5
    assert cfg.get_symbol("ETHUSD.sa").commission_round_trip_usd() == 0.0


def test_spread_usd_per_lot_ethusd() -> None:
    """ETHUSD typical 1700 points × tick_value $0.001 = $1.70/lote."""
    eth = load_broker_config().get_symbol("ETHUSD.sa")
    typical_usd, worst_usd = eth.spread_usd_per_lot()
    assert typical_usd == pytest.approx(1.70, abs=0.01)
    assert worst_usd == pytest.approx(2.30, abs=0.01)


def test_sqx_settings_returns_dict_with_expected_keys() -> None:
    eth = load_broker_config().get_symbol("ETHUSD.sa")
    s = eth.sqx_settings(current_price=3500.0)
    assert set(s.keys()) == {"ticker", "commission", "swap", "spread", "context"}
    assert s["ticker"] == "ETHUSD.sa"
    assert s["swap"]["triple_swap_on"] == "FRIDAY"
    assert s["commission"]["mode"] == "None"  # ETHUSD sin comision


def test_sqx_settings_commission_size_based() -> None:
    """Un .pa con comision (EURUSD) debe reportar mode='Size based' (NUNCA 'Per
    trade', que es bug: cobra fijo, no escala). El valor a teclear en SQX =
    round-trip 4.5."""
    eur = load_broker_config().get_symbol("EURUSD.pa")
    s = eur.sqx_settings(current_price=1.10)
    assert s["commission"]["mode"] == "Size based"
    assert s["commission"]["value_usd_per_lot_per_side"] == 2.25
    assert s["commission"]["sqx_size_based_value"] == 4.5
    assert s["swap"]["triple_swap_on"] == "WEDNESDAY"


def test_format_sqx_block_human_readable() -> None:
    eth = load_broker_config().get_symbol("ETHUSD.sa")
    text = eth.format_sqx_block(current_price=3500.0)
    assert "ETHUSD" in text
    assert "FRIDAY" in text
    assert "$  -1.9444 per day per lot" in text or "$ -1.9444 per day per lot" in text


# ---------------------------------------------------------------------------
# Comprobacion de inmutabilidad: paths por defecto resuelven a config/
# ---------------------------------------------------------------------------

def test_default_paths_resolve() -> None:
    assert AXI_SELECT_YAML.exists()
    assert BROKER_AXI_YAML.exists()
    assert AXI_SELECT_YAML.parent.name == "config"
