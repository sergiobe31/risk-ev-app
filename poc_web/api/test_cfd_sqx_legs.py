"""Patas ``.sqx`` de CFD FTMO en la app (2026-10-12): catálogo, carga, guardas y
simulación con el perfil ``ftmo_50k_swing`` (sizing Opción A con lotes de 0,01 y valle
por ventanas con swap). Mismo patrón que ``test_futures_sqx_legs.py``: tests a nivel
de función con ``.sqx`` sintéticos escritos con el writer de ``tests/test_sqx_trades.py``.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest poc_web/api/test_cfd_sqx_legs.py -q
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from poc_web.api import main
from poc_web.api.main import (
    PortfolioLoadRequest,
    PropfirmSimRequest,
    _lots_for_leg,
    _load_sqx_leg,
    portfolio_load,
    simulate_propfirm,
)

_SPEC = importlib.util.spec_from_file_location(
    "_tst_sqx_trades_writer_cfd",
    Path(__file__).resolve().parents[2] / "tests" / "test_sqx_trades.py")
_W = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_W)
write_sqx = _W.write_sqx

SWING = "ftmo_50k_swing"
FUTURES = "ftmo_futures_growth_50k"
ACCOUNT = 50000.0
PV = 100000.0
COMM = 5.0                 # USD por lote ida y vuelta (FTMO forex)
LOTS = 0.5


def _f32(x: float) -> float:
    return float(np.float32(x))


def _cfd_trades(slot_hour: int, seed: int, n_days: int = 100) -> list[dict]:
    """Un corto EURUSD por día hábil en [slot_hour, slot_hour+1:30] (hora del
    servidor), SL a 10-25 pips, 60% ganadoras a 1,5R; precios redondeados a float32
    para que PnL y MAE cuadren con lo que relee el extractor. Cada 10.º trade se
    mantiene 2 noches con abono de swap (corto EURUSD: +0,10 USD/lote/noche)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    out = []
    for k, d in enumerate(days):
        dist = float(rng.choice([0.0010, 0.0015, 0.0020, 0.0025]))
        open_px = _f32(1.10 + 0.0001 * float(rng.integers(-40, 40)))
        win = rng.random() < 0.6
        sl_px = _f32(open_px + dist)
        close_px = _f32(open_px - 1.5 * dist) if win else sl_px
        gross = (open_px - close_px) * PV * LOTS
        mae = (float(rng.uniform(0.0, 0.8)) * dist * PV * LOTS) if win else -gross
        overnight = k % 10 == 9 and k + 2 < n_days
        swap = 0.10 * 2 * LOTS if overnight else 0.0
        close_day = days[k + 2] if overnight else d
        out.append(dict(
            open=datetime(d.year, d.month, d.day, slot_hour, 0),
            close=datetime(close_day.year, close_day.month, close_day.day, slot_hour + 1, 30),
            lots=LOTS, open_px=open_px, close_px=close_px, sl_px=sl_px,
            pnl=gross - COMM * LOTS + swap, mae=max(mae, 0.0), flag=1))
    return out


def _write_cfd(path: Path, trades: list[dict]) -> str:
    return str(write_sqx(path, trades, symbol="EURUSD_ftmo", instrument="EURUSD_cfd_ftmo",
                         pv=PV, tick=0.0001, capital=ACCOUNT))


def _write_cfd_legs(root: Path, n: int = 3) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    return [_write_cfd(root / f"Strategy 8.{k}.1.sqx", _cfd_trades(10 + 2 * k, 200 + k))
            for k in range(n)]


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_leg_cache", {})
    monkeypatch.setattr(main, "_portfolios", {})
    monkeypatch.setattr(main, "_sqx_symbol_cache", {})
    # sin OHLC real (hallazgo A9, 2026-10-14): estos tests son del camino de ventanas
    monkeypatch.setattr(main, "_sqx_ohlc_cache", {})
    monkeypatch.setattr(main, "SQX_OHLC_DIR", tmp_path / "sin_ohlc")
    monkeypatch.setattr(main, "EXTRA_INPUT_ROOTS", [tmp_path / "extra"])
    monkeypatch.setattr(main, "SQX_STRATEGIES_DIR", tmp_path / "no_strategies")


@pytest.fixture
def legs(tmp_path):
    return _write_cfd_legs(tmp_path / "extra" / "bank")


def _load(legs, firm_key=SWING):
    return portfolio_load(PortfolioLoadRequest(legs=legs, account_size=ACCOUNT,
                                               firm_key=firm_key))


def _req(res, **kw):
    base = dict(portfolio_id=res["portfolio_id"], firm_key=SWING,
                f_lo=0.0025, f_hi=0.0075, f_step=0.0025, n_paths=200,
                n_challenges=5, n_sequences=100, capital0_eur=3000.0,
                fx_eur_usd=1.08)
    base.update(kw)
    return PropfirmSimRequest(**base)


# --- catálogo ------------------------------------------------------------------
def test_catalogo_incluye_cfd_con_asset(tmp_path):
    p = _write_cfd(tmp_path / "extra" / "bank" / "Strategy 9.1.1.sqx", _cfd_trades(10, 1, 10))
    by_name = {l["name"]: l for l in main._scan_legs()}
    leg = by_name["Strategy 9.1.1"]
    assert leg["kind"] == "sqx_futures" and leg["asset"] == "cfd" and leg["path"] == p


# --- carga ---------------------------------------------------------------------
def test_carga_cfd_con_perfil_swing_ok_y_guardas(legs):
    res = _load(legs)
    assert res["ok"], res
    g = res["guards"]
    assert g["tipo_patas"]["status"] == "ok" and "CFD" in g["tipo_patas"]["detail"]
    assert g["corte_1610"]["status"] == "ok" and "no aplica" in g["corte_1610"]["detail"]
    assert g["ventanas"]["status"] == "ok"
    assert "CFD" in g["sqx"]["detail"]
    assert not any("perfil FTMO CFD" in w for w in res["warnings"])
    assert all(l["status"] == "ok" for l in res["legs"])


def test_carga_cfd_con_perfil_futures_avisa(legs):
    res = _load(legs, firm_key=FUTURES)
    assert any("no es un perfil FTMO CFD" in w for w in res["warnings"])


def test_cartera_mixta_futuros_y_cfd_es_error(tmp_path, legs):
    fut = str(write_sqx(tmp_path / "extra" / "fut" / "Strategy 1.1.1.sqx",
                        [_W.long_win(open=datetime(2023, 1, 2 + k, 10, 0),
                                     close=datetime(2023, 1, 2 + k, 11, 0)) for k in range(5)]))
    res = _load([legs[0], fut])
    assert not res["ok"]
    assert res["guards"]["tipo_patas"]["status"] == "error"
    assert "futuros y de CFD" in res["guards"]["tipo_patas"]["detail"]


# --- sizing y eje de días ----------------------------------------------------
def test_sizing_cfd_lotes_de_001_y_swap_en_el_valle(legs):
    leg = _load_sqx_leg(legs[0], ACCOUNT, None)
    assert leg["asset"] == "cfd" and leg["meta"]["lot_step"] == 0.01
    lots, diag = _lots_for_leg(leg, np.array([0.005]), {}, 0.005, ACCOUNT, None)
    risk = leg["fleg"].risk_per_micro
    exp = np.maximum(0.01, np.round(np.floor(250.0 / (risk * 0.01) + 1e-9) * 0.01, 10))
    np.testing.assert_allclose(lots[0], exp)
    assert np.all(np.abs(lots[0] * 100 - np.round(lots[0] * 100)) < 1e-9)   # múltiplos de 0,01
    # los trades con noches llevan abono: el valle no lo descuenta (solo cargos)
    fl = leg["fleg"]
    assert np.all(fl.adverse_per_lot >= fl.comm_per_lot - 1e-9)


def test_eje_ftmo_desplaza_una_hora_y_cambia_el_dia(tmp_path):
    """Un trade que cierra a las 00:30 del servidor pertenece, en el eje FTMO
    (−1 h, medianoche CE(S)T), al día anterior."""
    tr = _cfd_trades(10, 3, 10)
    tr[0] = dict(tr[0], open=datetime(2023, 1, 2, 23, 40), close=datetime(2023, 1, 3, 0, 30))
    p = _write_cfd(tmp_path / "extra" / "x" / "Strategy 9.9.9.sqx", tr)
    srv = _load_sqx_leg(p, ACCOUNT, None)
    ftmo = _load_sqx_leg(p, ACCOUNT, None, axis_hours=-1.0)
    i = int(np.argmin(srv["s"].close_t))
    assert str(srv["s"].close_d[i]) == "2023-01-03"
    assert str(ftmo["s"].close_d[i]) == "2023-01-02"
    assert ftmo is not srv                                   # caché por eje


# --- simulación ----------------------------------------------------------------
def test_simulacion_cfd_swing_con_ventanas(legs):
    res = _load(legs)
    out = simulate_propfirm(_req(res, day_axis="ftmo"))
    meta = out["meta"]
    assert meta["float_mode"] == "windows"
    assert meta["leg_kind"] == "sqx_futures"
    assert meta["sizing"] == "riesgo fijo (Opción A)"
    assert "swap" in meta["float_note"]
    assert len(out["reports"]) == 3
    for diag in meta["sizing_diag"].values():
        assert diag[0]["X"] == pytest.approx(0.0025 * ACCOUNT)
    rep0 = out["reports"][0]
    assert 0.0 <= rep0["p_pass1"] <= 1.0
    assert not any("America/New_York" in w for w in out["warnings"])


def test_simulacion_cfd_con_perfil_futures_es_409(legs):
    res = _load(legs, firm_key=FUTURES)
    with pytest.raises(HTTPException) as ei:
        simulate_propfirm(_req(res, firm_key=FUTURES))
    assert ei.value.status_code == 409 and "FTMO CFD" in ei.value.detail


# --- revisión DeepSeek (2026-10-12) -------------------------------------------
def test_eje_ftmo_respeta_el_desfase_de_horario_de_verano(tmp_path):
    """H2: entre el cambio de hora de EE. UU. (12-mar-2023) y el de la UE (26-mar) el
    servidor va 2 h por delante de Praga, no 1. Un cierre a las 01:30 del servidor del
    21-mar es 23:30 del 20-mar en Praga (con −1 h fijo saldría el 21)."""
    tr = _cfd_trades(10, 4, 60)
    tr[0] = dict(tr[0], open=datetime(2023, 3, 20, 22, 0), close=datetime(2023, 3, 21, 1, 30))
    p = _write_cfd(tmp_path / "extra" / "dst" / "Strategy 9.8.7.sqx", tr)
    leg = _load_sqx_leg(p, ACCOUNT, None, axis_hours=-1.0)
    s = leg["s"]
    i = int(np.where(pd.to_datetime(s.close_t) == pd.Timestamp("2023-03-20 23:30"))[0][0])
    assert str(s.close_d[i]) == "2023-03-20"
    assert pd.Timestamp(s.open_t[i]) == pd.Timestamp("2023-03-20 20:00")


def test_cfd_con_eje_servidor_avisa(legs):
    res = _load(legs)
    out = simulate_propfirm(_req(res, day_axis="servidor"))
    assert any("medianoche CE(S)T" in w for w in out["warnings"])


def test_lotes_cfd_ignoran_el_tope_de_contratos(legs):
    leg = _load_sqx_leg(legs[0], ACCOUNT, None)
    lots_cap, _ = _lots_for_leg(leg, np.array([0.005]), {}, 0.005, ACCOUNT, 0.5)
    lots_none, _ = _lots_for_leg(leg, np.array([0.005]), {}, 0.005, ACCOUNT, None)
    np.testing.assert_array_equal(lots_cap, lots_none)
