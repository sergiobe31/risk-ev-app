"""Tests del loader de trade list (orderstocsv) + derivados y_n/slInMoney."""
from __future__ import annotations

import numpy as np

from src.risk.trades import load_trade_list, sniff_decimal

# CSV sintetico (3 trades, EURUSD.pa). E_0 = 10000 (Balance_1 - PnL_1).
# Decimal anglosajon ('.').
CSV_DOT = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
    "1;EURUSD.pa;Buy;2024.01.01 00:00;1.1000;0.20;2024.01.02 00:00;1.1030;300;10300;"
    "IST;Exit After 1 Bars;-50;60;1;-\n"
    "2;EURUSD.pa;Buy;2024.01.03 00:00;1.1000;0.20;2024.01.04 00:00;1.0980;-200;10100;"
    "IST;Exit After 1 Bars;-200;30;1;-\n"
    "3;EURUSD.pa;Buy;2024.01.05 00:00;1.1000;0.20;2024.01.06 00:00;1.1050;500;10600;"
    "IST;Exit After 1 Bars;-40;80;1;-\n"
)
# Mismo contenido, decimal europeo (',') como lo produce usecomma=true.
CSV_COMMA = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
    "1;EURUSD.pa;Buy;2024.01.01 00:00;1,1000;0,20;2024.01.02 00:00;1,1030;300;10300;"
    "IST;Exit After 1 Bars;-50;60;1;-\n"
    "2;EURUSD.pa;Buy;2024.01.03 00:00;1,1000;0,20;2024.01.04 00:00;1,0980;-200;10100;"
    "IST;Exit After 1 Bars;-200;30;1;-\n"
    "3;EURUSD.pa;Buy;2024.01.05 00:00;1,1000;0,20;2024.01.06 00:00;1,1050;500;10600;"
    "IST;Exit After 1 Bars;-40;80;1;-\n"
)


def test_sniff_detecta_ambas_convenciones():
    assert sniff_decimal(CSV_DOT.encode()) == "."
    assert sniff_decimal(CSV_COMMA.encode()) == ","


def test_load_parsea_ambas_convenciones_igual():
    t_dot = load_trade_list(CSV_DOT.encode(), f_ref=0.01)
    t_com = load_trade_list(CSV_COMMA.encode(), f_ref=0.01)
    np.testing.assert_allclose(t_dot.pnl, t_com.pnl)
    np.testing.assert_allclose(t_dot.size, t_com.size)
    np.testing.assert_allclose(t_dot.equity_before, [10000, 10300, 10100])


def test_derivados_y_slmoney():
    tl = load_trade_list(CSV_DOT.encode(), f_ref=0.01)
    assert tl.initial_equity == 10000.0
    # y_n = (PnL/equity_before)/f_ref
    y = tl.r_multiples(0.01)
    np.testing.assert_allclose(y, [(300 / 10000) / 0.01, (-200 / 10300) / 0.01, (500 / 10100) / 0.01])
    # slInMoney_n = f_ref * equity_before / Size
    sl = tl.sl_in_money(0.01)
    np.testing.assert_allclose(sl, [0.01 * 10000 / 0.2, 0.01 * 10300 / 0.2, 0.01 * 10100 / 0.2])


def test_mae_se_normaliza_negativo():
    tl = load_trade_list(CSV_DOT.encode(), f_ref=0.01)
    assert tl.has_mae
    assert (tl.mae <= 0).all()  # MAE en flotante negativo


def test_validation_ok_y_net_profit():
    tl = load_trade_list(CSV_DOT.encode(), f_ref=0.01)
    v = tl.validation
    assert v.ok
    assert v.n_trades == 3
    assert v.net_profit == 600.0  # 300 - 200 + 500
    assert v.decimal_char == "."


# MAE con signo mixto (regresion: antes un + suelto sobrevivia por la media<0).
CSV_MIXED_MAE = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
    "1;EURUSD.pa;Buy;2024.01.01 00:00;1.1000;0.20;2024.01.02 00:00;1.1030;300;10300;"
    "IST;Exit After 1 Bars;-50;60;1;-\n"
    "2;EURUSD.pa;Buy;2024.01.03 00:00;1.1000;0.20;2024.01.04 00:00;1.0980;-200;10100;"
    "IST;Exit After 1 Bars;60;30;1;-\n"   # MAE +60 (magnitud) en columna de media<0
    "3;EURUSD.pa;Buy;2024.01.05 00:00;1.1000;0.20;2024.01.06 00:00;1.1050;500;10600;"
    "IST;Exit After 1 Bars;-40;80;1;-\n"
)


def test_mae_signo_mixto_se_fuerza_no_positivo():
    """Toda MAE debe ser <=0 aunque la columna tenga media negativa y un + suelto."""
    tl = load_trade_list(CSV_MIXED_MAE.encode(), f_ref=0.01)
    assert tl.has_mae
    assert (tl.mae <= 0).all()
    assert tl.mae[1] == -60.0  # el +60 (magnitud) se vuelve -60, no sobrevive


# Formato real del export orderstocsv de SQX B143 con usecomma=true (los 101
# CSV de PM_nstrategies, exportados 2026-08): MIXTO — precios y Size con
# punto, dinero (PnL/Balance/MAE/MFE) con coma europea, campos entrecomillados.
# Regresion: el sniffer global por fichero elegia coma (4 campos de dinero vs
# 3 de precio por renglon) y corrompia en silencio precios ("1.1000" ->
# 11000.0) y Size ("0.20" -> 20.0) -> valles/flotantes absurdos en el MC.
CSV_MIXED_FORMAT = (
    '"Ticket";"Symbol";"Type";"Open time";"Open price";"Size";"Close time";"Close price";'
    '"Profit/Loss";"Balance";"Sample type";"Close type";"MAE ($)";"MFE ($)";"Time in trade";"Comment"\n'
    '"1";"EURUSD_axi";"Buy";"2024.01.01 00:00:00";"1.1000";"0.20";"2024.01.02 00:00:00";"1.1030";'
    '"300,00";"10300,00";"IST";"Exit After 1 Bars";"-50,00";"60,00";"1d 0h 0m";""\n'
    '"2";"EURUSD_axi";"Buy";"2024.01.03 00:00:00";"1.1000";"0.20";"2024.01.04 00:00:00";"1.0980";'
    '"-200,00";"10100,00";"IST";"Exit After 1 Bars";"-200,00";"30,00";"1d 0h 0m";""\n'
    '"3";"EURUSD_axi";"Buy";"2024.01.05 00:00:00";"1.1000";"0.20";"2024.01.06 00:00:00";"1.1050";'
    '"500,00";"10600,00";"IST";"Exit After 1 Bars";"-40,00";"80,00";"1d 0h 0m";""\n'
)


def test_formato_mixto_b143_precios_punto_dinero_coma():
    """El CSV mixto de B143 (usecomma) debe parsear identico al de punto puro."""
    t_mix = load_trade_list(CSV_MIXED_FORMAT.encode(), f_ref=0.01)
    t_dot = load_trade_list(CSV_DOT.encode(), f_ref=0.01)
    np.testing.assert_allclose(t_mix.pnl, t_dot.pnl)
    np.testing.assert_allclose(t_mix.size, t_dot.size)
    np.testing.assert_allclose(t_mix.open_price, t_dot.open_price)
    np.testing.assert_allclose(t_mix.mae, t_dot.mae)
    np.testing.assert_allclose(t_mix.equity_before, t_dot.equity_before)
