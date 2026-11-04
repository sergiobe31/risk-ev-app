"""Tests del sweep de f (linealidad + re-floor + guardas).

HECHO CANONICO: el replay a f_ref debe reproducir EXACTO el path de equity de
referencia (Balance). El re-floor a otro f debe igualar una simulacion bruta
lot-discretizada.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.broker_config import BrokerSymbol
from src.risk.mm import default_f_grid, simulate_f
from src.risk.trades import load_trade_list

CSV = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
    "1;EURUSD.pa;Buy;2024.01.01 00:00;1.1000;0.20;2024.01.02 00:00;1.1030;300;10300;"
    "IST;Exit After 1 Bars;-50;60;1;-\n"
    "2;EURUSD.pa;Buy;2024.01.03 00:00;1.1000;0.20;2024.01.04 00:00;1.0980;-200;10100;"
    "IST;Exit After 1 Bars;-200;30;1;-\n"
    "3;EURUSD.pa;Buy;2024.01.05 00:00;1.1000;0.20;2024.01.06 00:00;1.1050;500;10600;"
    "IST;Exit After 1 Bars;-40;80;1;-\n"
)


def _eurusd() -> BrokerSymbol:
    return BrokerSymbol(
        ticker_axi="EURUSD.pa",
        category="forex",
        eligible_for_allocation=True,
        quote_currency="USD",
        contract_size=100000.0,
        tick_size=0.00001,
        pip_size=0.0001,
        tick_value_usd=1.0,
        point_value_usd=100000.0,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=100.0,
        typical_spread_points=2.0,
        worst_spread_points=3.0,
        commission_usd_per_lot_per_side=2.25,
        swap_type="points",
        swap_long_value=-7.7,
        swap_short_value=-7.7,
        triple_swap_day="wednesday",
        stop_level_points=0.0,
        leverage_max=100,
    )


@pytest.fixture
def tl():
    return load_trade_list(CSV.encode(), f_ref=0.01)


def test_sweep_a_f_ref_reproduce_path_referencia(tl):
    """A f=f_ref, el replay con re-floor debe dar EXACTO el Balance original."""
    res = simulate_f(tl, np.array([0.01]), f_ref=0.01, symbols={"EURUSD.pa": _eurusd()}, leverage=100)
    # equity[:,0]=E0=10000, luego 10300, 10100, 10600
    np.testing.assert_allclose(res.equity[0], [10000, 10300, 10100, 10600])
    assert res.n_sub_min_lot[0] == 0
    assert res.n_margin_clipped[0] == 0


def test_sweep_f_doble_refloor_coincide_con_bruto(tl):
    """A f=0.02 el replay re-floor debe igualar el calculo manual discreto."""
    res = simulate_f(tl, np.array([0.02]), f_ref=0.01, symbols={"EURUSD.pa": _eurusd()}, leverage=100)
    # Manual: trade1 lots=0.40, pnl=600 -> 10600
    # trade2 lots=floor(0.02*10600/515/0.01)*0.01=floor(0.41165/0.01)*0.01=0.41, pnl=-410 -> 10190
    # trade3 lots=floor(0.02*10190/505/0.01)*0.01=floor(0.40356/0.01)*0.01=0.40, pnl=1000 -> 11190
    np.testing.assert_allclose(res.equity[0], [10000, 10600, 10190, 11190])
    np.testing.assert_allclose(res.lots[0], [0.40, 0.41, 0.40])


def test_naive_vs_refloor_difieren_por_discretizacion(tl):
    """El naive prod(1+f*y) difiere del re-floor (el bug que el re-floor corrige)."""
    res = simulate_f(tl, np.array([0.02]), f_ref=0.01, symbols={"EURUSD.pa": _eurusd()}, leverage=100)
    y = tl.r_multiples(0.01)
    naive_terminal = 10000 * np.prod(1 + 0.02 * y)
    # No deben ser identicos (la discretizacion de 0.41 vs continuo).
    assert abs(res.terminal()[0] - naive_terminal) > 1e-6
    # Y el error relativo es pequeno (<0.25%, como documenta el plan).
    rel = abs(res.terminal()[0] - naive_terminal) / abs(naive_terminal)
    assert rel < 0.0025


def test_lotsifnomm_guard_clampa_a_min_lot(tl):
    """Si f es tan bajo que los lotes continuos < min_lot, se clampea (no 1.0)."""
    # f=0.0005 (0.05%) -> lots_cont ~ 0.01 (en el limite). Aun menor -> clamp.
    res = simulate_f(tl, np.array([0.0001]), f_ref=0.01, symbols={"EURUSD.pa": _eurusd()}, leverage=100)
    # Con f=0.0001, lots_cont=0.0001*10000/500=0.002 < min_lot 0.01 -> clamp 0.01.
    assert (res.lots[0] == 0.01).all()
    assert res.n_sub_min_lot[0] == 3  # los 3 trades clampeados
    # Jamas 1.0 lote (la trampa LotsIfNoMM).
    assert (res.lots[0] != 1.0).all()


def test_default_grid_rango():
    g = default_f_grid()
    assert g.min() == pytest.approx(0.0025)
    assert g.max() == pytest.approx(0.02)


def test_reference_clamping_fraction_detecta_min_lot():
    """Heuristica: trades con Size==min_lot cuentan como sospecha de clamping."""
    from src.risk.mm import reference_clamping_fraction
    # Un trade a 0.20 (limpio) y otro a 0.01 (== min_lot del fixture).
    csv2 = (
        "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
        "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
        "1;EURUSD.pa;Buy;2024.01.01 00:00;1.1000;0.20;2024.01.02 00:00;1.1030;300;10300;"
        "IST;Exit After 1 Bars;-50;60;1;-\n"
        "2;EURUSD.pa;Buy;2024.01.03 00:00;1.1000;0.01;2024.01.04 00:00;1.0980;-15;10285;"
        "IST;Exit After 1 Bars;-15;5;1;-\n"
    )
    tl2 = load_trade_list(csv2.encode(), f_ref=0.01)
    frac = reference_clamping_fraction(tl2, {"EURUSD.pa": _eurusd()})
    assert frac == pytest.approx(0.5)  # 1 de 2 trades en min_lot


def test_maxlots_cap_topea_el_tamano(tl):
    """#MaxLots# del bloque SQX se respeta: lots nunca superan el tope."""
    # f=0.02 -> lots_cont=0.4; con max_lots=0.3 se topea a 0.30.
    res = simulate_f(tl, np.array([0.02]), f_ref=0.01,
                     symbols={"EURUSD.pa": _eurusd()}, leverage=100, max_lots=0.3)
    assert (res.lots[0] <= 0.3 + 1e-9).all()
    assert res.lots[0, 0] == pytest.approx(0.30)
