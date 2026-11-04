"""Patas .sqx de CFD FTMO en el motor (2026-10-12): sizing Opción A con paso de lote y
lote mínimo, y swap en el peor flotante de la pata (src/risk/futures_legs.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.futures_legs import fixed_risk_micros, strategy_from_trade_table


# --- sizing ------------------------------------------------------------------
def test_cfd_lotes_al_paso_de_001_hacia_abajo_y_minimo():
    """X = 0,5% de 50k = 250 USD. Riesgo por lote 150 / 95 / 400 / 60.000 USD →
    1,66 / 2,63 / 0,62 / clamp a 0,01 (como redondea FixedAmount del EA)."""
    risk = np.array([150.0, 95.0, 400.0, 60000.0])
    lots, diag = fixed_risk_micros(risk, [0.005], 50000.0, lot_step=0.01, min_lot=0.01)
    np.testing.assert_array_equal(lots, [[1.66, 2.63, 0.62, 0.01]])
    assert diag[0]["frac_clamp_up"] == pytest.approx(0.25)      # X < risk·min_lot solo en el último


def test_cfd_lotes_sin_ruido_de_coma_flotante():
    """k·0,01 exacto a 10 decimales (sin 0,30000000000000004)."""
    risk = np.array([250.0 / 0.3, 250.0 / 0.07])
    lots, _ = fixed_risk_micros(risk, [0.005], 50000.0, lot_step=0.01, min_lot=0.01)
    assert lots[0, 0] == 0.3 and lots[0, 1] == 0.07


def test_futuros_por_defecto_sin_cambios():
    risk = np.array([150.0, 95.0, 400.0])
    lots, diag = fixed_risk_micros(risk, [0.005], 50000.0)
    np.testing.assert_array_equal(lots, [[1.0, 2.0, 1.0]])
    assert diag[0]["frac_clamp_up"] == pytest.approx(1 / 3)


def test_cap_se_aplica_tambien_con_paso_fraccionario():
    lots, diag = fixed_risk_micros(np.array([10.0]), [0.005], 50000.0, cap=5.0,
                                   lot_step=0.01, min_lot=0.01)
    assert lots[0, 0] == 5.0 and diag[0]["frac_capped"] == 1.0


@pytest.mark.parametrize("kw", [dict(lot_step=0.0), dict(min_lot=-0.01),
                                dict(lot_step=float("nan"))])
def test_paso_o_minimo_invalidos(kw):
    with pytest.raises(ValueError, match="lot_step"):
        fixed_risk_micros(np.array([100.0]), [0.005], 50000.0, **kw)


# --- swap en el peor flotante ------------------------------------------------
def _table(swap=None) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({
        "open_ts": pd.to_datetime(["2024-03-04 10:00", "2024-03-05 10:00"]),
        "close_ts": pd.to_datetime(["2024-03-04 12:00", "2024-03-07 12:00"]),
        "lots": [0.5, 0.25],
        "pnl_net": [-50.0, 20.0],
        "mae_usd": [47.5, 30.0],
        "open_px": [1.125, 1.125],
        "close_px": [1.126, 1.12],
        "sl_px": [1.127, 1.128],
        "is_buy": [False, False],
        "comm_usd": [2.5, 1.25],
    })
    if swap is not None:
        df["swap_usd"] = swap
    return df, {"pv": 100000.0, "symbol": "EURUSD_ftmo"}


def test_swap_cargo_entra_en_el_peor_flotante():
    df, meta = _table(swap=[0.0, -8.0])              # 2.º trade: cargo de 8 USD
    _, leg = strategy_from_trade_table(df, meta, key="c")
    np.testing.assert_allclose(leg.adverse_per_lot, [(47.5 + 2.5) / 0.5,
                                                     (30.0 + 1.25 + 8.0) / 0.25])


def test_swap_abono_no_reduce_el_peor_flotante():
    df, meta = _table(swap=[0.0, 3.0])
    _, leg_abono = strategy_from_trade_table(df, meta, key="c")
    _, leg_sin = strategy_from_trade_table(*_table(), key="c")
    np.testing.assert_allclose(leg_abono.adverse_per_lot, leg_sin.adverse_per_lot)


def test_swap_no_finito_es_error():
    df, meta = _table(swap=[0.0, np.nan])
    with pytest.raises(ValueError, match="swap_usd"):
        strategy_from_trade_table(df, meta, key="c")


def test_cap_menor_que_el_lote_minimo_es_error():
    with pytest.raises(ValueError, match="lote mínimo"):
        fixed_risk_micros(np.array([100.0]), [0.005], 50000.0, cap=0.005,
                          lot_step=0.01, min_lot=0.01)
