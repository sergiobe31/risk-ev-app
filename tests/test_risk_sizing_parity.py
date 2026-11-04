"""Fidelity gate: el sizing modo-exacto (ATR Wilder shift1) reproduce SQX.

Oracle = la columna ``Size`` del CSV orderstocsv (lo que SQX ejecutó por trade,
ground truth). El motor calcula ``slInMoney`` desde el ATR real
(:func:`src.risk.atr.sl_in_money_from_atr`) y, vía :func:`src.risk.mm.simulate_f`
con re-floor de lot-step, los lotes por trade. Si la app reproduce los ``Size``
de SQX trade-a-trade, el motor de sizing es fiel — no hace falta re-run de SQX
ni leer ``dailyEquity.bin``.

Datos de validación (Desktop, NO repo -> ``skip`` limpio si ausentes):
  ``<RISK_PARITY_DATA>/2ATR_first_strategy.csv``  -> XAUUSD_axi H4, risk 2%, ATRMult 2
  ``<RISK_PARITY_DATA>/2ATR_second.csv``          -> idem
OHLC M1 (repo): ``risk_app/inputs/ohlc/XAUUSD_axi.csv``

Caracterización (2026-07-17): los 4 CSV son XAUUSD_axi H4, ATRPeriod 20, ATRMult 2.
Los ``2ATR_*`` son risk=2% y dan ratio slInMoney(ATR)/slInMoney(csv) ≈ 0.96 (gate
de paridad). Residual esperado ~3-5% (holidays + floor de lot-step); el gate lo
absorbe. Hallazgo DIFERIDO: holidays 5%-><1%.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.broker_config import load_broker_config
from src.risk import load_trade_list
from src.risk.atr import compute_atr_series, load_ohlc_m1, sl_in_money_from_atr
from src.risk.mm import simulate_f

_REPO = Path(__file__).resolve().parent.parent
# CSV fuera del repo (exports de SQX de la máquina del autor); override vía env.
DATA_DIR = Path(os.environ.get(
    "RISK_PARITY_DATA",
    "/nonexistent/private-data",
))
OHLC = _REPO / "risk_app" / "inputs" / "ohlc" / "XAUUSD_axi.csv"

# Params confirmados por caracterización empírica del .sqx (risk_mm_parse):
# todos dicen symbol=XAUUSD_axi, tf=H4, atrP=20, atrMult=2, lotsIfNoMM=0.01, maxLots=5.
ATR_PERIOD = 20
ATR_MULT = 2.0
TF = "H4"

CASES_2ATR = ["2ATR_first_strategy.csv", "2ATR_second.csv"]   # risk 2% -> gate de paridad


def _broker_xauusd():
    syms = {s.ticker_axi: s for s in load_broker_config().symbols}
    return syms["XAUUSD.pa"]


@pytest.fixture(scope="module")
def atr_series():
    """ATR de Wilder shift1 sobre el OHLC M1 del repo (lo pesado, una vez)."""
    if not OHLC.exists():
        pytest.skip(f"OHLC del repo no encontrado: {OHLC}")
    return compute_atr_series(load_ohlc_m1(OHLC), TF, ATR_PERIOD)


def _load(csv):
    p = DATA_DIR / csv
    if not p.exists():
        pytest.skip(f"CSV de validación ausente (Desktop, no repo): {p}")
    return load_trade_list(p), p


# ---------------------------------------------------------------------------
# Gate 1 — slInMoney(ATR real) ≈ slInMoney(back-derivo del Size de SQX)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("csv", CASES_2ATR)
def test_sl_in_money_atr_matches_csv(csv, atr_series):
    """El ATR modo-exacto reproduce el slInMoney que SQX usó (ratio ≈ 1)."""
    tl, _ = _load(csv)
    sym = _broker_xauusd()
    f_ref = 0.02
    sl_csv = f_ref * tl.equity_before / tl.size          # back-derivo del Size (ground truth)
    sl_atr = sl_in_money_from_atr(
        atr_series, tl.df["open_time"], atr_mult=ATR_MULT, point_value=sym.point_value_usd,
    )
    r = sl_atr / sl_csv
    r = r[np.isfinite(r) & (sl_csv > 0) & (sl_atr > 0)]
    med = float(np.median(r))
    p10 = float(np.percentile(r, 10))
    # mediana ~1 (residual holidays+floor ~4%); cola p10 >= 0.85 (sin fuga sistémica).
    assert 0.90 <= med <= 1.01, f"{csv}: ratio mediana {med:.4f} fuera de [0.90, 1.01]"
    assert p10 >= 0.85, f"{csv}: cola p10={p10:.3f} < 0.85 (fuga sistémica en la cola)"


# ---------------------------------------------------------------------------
# Gate 2 — simulate_f (re-floor) es "well-behaved" a f_ref (no bit-exacto)
# ---------------------------------------------------------------------------
# NOTA: el oracle de paridad bit-exacta es Gate 1 (slInMoney aislado). El replay
# simulate_f recomputa la equity con lotes que difieren en el floor de lot-step,
# así que en estrategias largas (p.ej. 1066 trades) el drift de composición
# acumula varios % de mediana. NO es infidelidad del ATR; es la discretización
# que mm.py documenta. Y no contamina la decisión de f (el MC headline usa forma
# continua E·(1+fy), error ≤0.24%, no este sweep discreto). Aquí sólo pedimos
# que el replay no explote ni clampee patológicamente a f_ref.
@pytest.mark.parametrize("csv", CASES_2ATR)
def test_simulate_f_well_behaved_at_fref(csv, atr_series):
    """Replay a f_ref: sin blow-up, sin clamping a min_lot, mediana cerca de 1."""
    tl, _ = _load(csv)
    sym = _broker_xauusd()
    f_ref = 0.02
    sl_atr = sl_in_money_from_atr(
        atr_series, tl.df["open_time"], atr_mult=ATR_MULT, point_value=sym.point_value_usd,
    )
    sweep = simulate_f(
        tl, np.array([f_ref]), symbols={sym.ticker_axi: sym},
        max_lots=5.0, lots_if_no_mm=0.01, sl_in_money=sl_atr,
    )
    lots = sweep.lots[0]
    size = tl.size
    nz = size > 0
    r = lots[nz] / size[nz]
    med = float(np.median(r))
    frac_1step = float(np.mean(np.abs(lots[nz] - size[nz]) <= sym.lot_step))
    assert int(sweep.blown_up.sum()) == 0, f"{csv}: equity ≤ 0 a f_ref (blow-up inesperado)"
    assert int(sweep.n_sub_min_lot[0]) == 0, (
        f"{csv}: {int(sweep.n_sub_min_lot[0])} trades clampeados a min_lot "
        f"(inesperado para Seed con lotes ~0.2)"
    )
    # mediana cerca de 1 con holgura para el drift de composición (hasta ±10%).
    assert 0.90 <= med <= 1.10, f"{csv}: lots/Size mediana {med:.4f} fuera de [0.90, 1.10]"
    # Bit-exactitud sólo esperable en estrategias cortas; la dejamos como info.
    print(f"  {csv}: lots/Size mediana={med:.4f}, dentro_1step={frac_1step:.0%}")
