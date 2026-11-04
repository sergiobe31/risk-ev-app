"""Tests de src/risk/input_checks.py — guards de inputs del modo portfolio."""
from __future__ import annotations

import numpy as np

from src.broker_config import load_broker_config
from src.risk.input_checks import (
    initial_capital_mismatch,
    leg_diagnostics,
    multi_symbol_counts,
)
from src.risk.trades import load_trade_list

_HEADER = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
)

# XAUUSD.pa: pv $100, comm 4.5 round-trip, swap long -58.3 pts x $1 = -$58.3/lote/noche.
# Buy 1 lote 2000.00 -> 2001.00: bruto = $100. 1 noche. Coste esperado = 4.5 + 58.3 = 62.8.
CSV_XAU_CLEAN = _HEADER + (
    "1;XAUUSD.pa;Buy;2024.01.01 00:00;2000.00;1.00;2024.01.02 00:00;2001.00;37.2;5037.2;"
    "IST;Exit After 1 Bars;-10;60;1;-\n"
)
# Mismo trade con el coste x10 (net = 100 - 628 = -528): ratio 10 -> warning.
CSV_XAU_COST_X10 = _HEADER + (
    "1;XAUUSD.pa;Buy;2024.01.01 00:00;2000.00;1.00;2024.01.02 00:00;2001.00;-528.0;4472.0;"
    "IST;Exit After 1 Bars;-10;60;1;-\n"
)
# DAX40.fs: pv $28.49, comm 0, swap 0. Buy 1 lote 9000 -> 9001: bruto = $28.49.
CSV_DAX_CLEAN = _HEADER + (
    "1;DAX40.fs;Buy;2024.01.01 00:00;9000.00;1.00;2024.01.02 00:00;9001.00;28.49;5028.49;"
    "IST;Exit After 1 Bars;-5;30;1;-\n"
)
# DAX con net = -100 -> coste implicito 128.49 con 0 esperado -> warning (pv distinto).
CSV_DAX_PV_MISMATCH = _HEADER + (
    "1;DAX40.fs;Buy;2024.01.01 00:00;9000.00;1.00;2024.01.02 00:00;9001.00;-100.0;4900.0;"
    "IST;Exit After 1 Bars;-5;30;1;-\n"
)
# Portfolio Master: dos simbolos mezclados.
CSV_MULTI = _HEADER + (
    "1;XAUUSD.pa;Buy;2024.01.01 00:00;2000.00;1.00;2024.01.02 00:00;2001.00;37.2;5037.2;"
    "IST;Exit After 1 Bars;-10;60;1;-\n"
    "2;DAX40.fs;Buy;2024.01.03 00:00;9000.00;1.00;2024.01.04 00:00;9001.00;28.49;5065.69;"
    "IST;Exit After 1 Bars;-5;30;1;-\n"
)


def test_multi_symbol_detectado():
    tl = load_trade_list(CSV_MULTI.encode())
    counts = multi_symbol_counts(tl)
    assert counts == {"XAUUSD.pa": 1, "DAX40.fs": 1}


def test_single_symbol_limpio():
    tl = load_trade_list(CSV_XAU_CLEAN.encode())
    assert multi_symbol_counts(tl) is None


def test_capital_mismatch_avisa_y_cuando_cuadra_no():
    tl = load_trade_list(CSV_XAU_CLEAN.encode())  # E_0 = 5000
    assert initial_capital_mismatch(tl, baseline=5000.0) is None
    msg = initial_capital_mismatch(tl, baseline=20000.0)
    assert msg is not None and "5,000" in msg and "20,000" in msg


def test_coste_implicito_limpio_ratio_1():
    xau = load_broker_config().get_symbol("XAUUSD.pa")
    tl = load_trade_list(CSV_XAU_CLEAN.encode())
    d = leg_diagnostics(tl, xau, f_ref=0.02, baseline=5000.0)
    assert d.implied_cost_per_lot == np.float64(62.8) or abs(d.implied_cost_per_lot - 62.8) < 1e-9
    assert abs(d.expected_cost_per_lot - 62.8) < 1e-9
    assert abs(d.cost_ratio - 1.0) < 1e-9
    assert not any("coste" in w for w in d.warnings)


def test_coste_implicito_x10_dispara_warning():
    xau = load_broker_config().get_symbol("XAUUSD.pa")
    tl = load_trade_list(CSV_XAU_COST_X10.encode())
    d = leg_diagnostics(tl, xau, f_ref=0.02, baseline=5000.0)
    assert d.cost_ratio > 2.0
    assert any("lo esperado" in w for w in d.warnings)


def test_coste_implicito_comm0_limpio_no_avisa():
    dax = load_broker_config().get_symbol("DAX40.fs")
    tl = load_trade_list(CSV_DAX_CLEAN.encode())
    d = leg_diagnostics(tl, dax, f_ref=0.02, baseline=5000.0)
    assert abs(d.implied_cost_per_lot) < 1e-9
    assert d.expected_cost_per_lot == 0.0
    assert not any("coste" in w for w in d.warnings)


def test_coste_implicito_comm0_con_coste_dispara_warning():
    dax = load_broker_config().get_symbol("DAX40.fs")
    tl = load_trade_list(CSV_DAX_PV_MISMATCH.encode())
    d = leg_diagnostics(tl, dax, f_ref=0.02, baseline=5000.0)
    assert d.implied_cost_per_lot > 0  # 28.49 - (-100) = 128.49
    assert any("point value" in w for w in d.warnings)


def test_clamp_frac_y_riesgo_suelo():
    # sizes 0.01 y 0.05 (min_lot 0.01) -> frac_min = 0.5 -> warning de suelo.
    csv = _HEADER + (
        "1;XAUUSD.pa;Buy;2024.01.01 00:00;2000.00;0.01;2024.01.02 00:00;2000.50;0.5;5000.5;"
        "IST;Exit After 1 Bars;-1;2;1;-\n"
        "2;XAUUSD.pa;Buy;2024.01.03 00:00;2000.00;0.05;2024.01.04 00:00;2000.50;2.5;5003.0;"
        "IST;Exit After 1 Bars;-1;2;1;-\n"
    )
    xau = load_broker_config().get_symbol("XAUUSD.pa")
    tl = load_trade_list(csv.encode())
    d = leg_diagnostics(tl, xau, f_ref=0.02, baseline=5000.0)
    assert d.frac_min_lot_ref == 0.5
    assert any("min_lot" in w for w in d.warnings)
    # sl = f_ref*eq/size: t1 = 0.02*5000/0.01 = 10000; t2 = 0.02*5000.5/0.05 = 2000.2
    # floor = 0.01 * mediana(10000, 2000.2) = 0.01*6000.1 = 60.001 -> 1.20002% de 5000
    assert abs(d.floor_risk_pct - 1.20002) < 1e-3
