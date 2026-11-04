"""Guardrail ``csv-d1-d2``, fase 1 (D1): ``simulate_f(..., compound=False)``.

Defecto D1 (``docs/backlog.md`` P0, medido el 2026-10-17): las patas CSV de la app de riesgo
dimensionaban cada orden sobre la equity del *replay* del histórico (``lots = f·E_n/sl``), que
compone a lo largo de todo el backtest. El plan csv-d1-d2 (§2.1 y Enmiendas v2 E1, E5, E15) añade
``compound: bool = True`` a :func:`src.risk.mm.simulate_f`:

* ``compound=False``: ``lots_cont = f·e0/sl`` con ``e0 = tl.initial_equity`` FIJO (tamaño de cuenta, sin
  componer). Todo lo demás (margen opcional, suelo a ``lot_step``, ``LotsIfNoMM``, ``MaxLots``, lote
  mínimo, ``n_sub_min_lot``, PnL, equity, ``mtm_min``, ``blown_up``) queda idéntico.
* ``compound=True`` (por defecto): bit a bit igual a como era antes del cambio.

Tests-de-plan de este fichero:

* AC1.1 (golden a mano, Enmienda E5): ``test_ac1_1_*``.
* AC1.2 (invariancia del lote ante el PnL previo, property test de semilla fija): ``test_ac1_2_*``.
* AC1.3 (``True`` por defecto == comportamiento previo, bit a bit): ``test_ac1_3_*``. Los tests
  existentes de ``simulate_f`` (``test_risk_mm_*``, ``test_risk_sizing_parity``) no se editan.
* Resto: bounds (suelo, ``LotsIfNoMM``, ``MaxLots``, lote mínimo), margen opcional y semántica de
  ``equity``/``mtm_min``/``blown_up`` a lote fijo (Enmienda E15).

Datos sintéticos: ni este fichero ni sus fixtures necesitan ``risk_app/inputs/``.

Ejecutar desde la raíz del repo:
    PYTHONPATH=. python -m pytest tests/test_risk_mm_compound.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from src.broker_config import BrokerSymbol
from src.risk.mm import SweepResult, simulate_f
from src.risk.trades import load_trade_list

_HEADER = (
    "Ticket;Symbol;Type;Open time;Open price;Size;Close time;Close price;"
    "Profit/Loss;Balance;Sample type;Close type;MAE ($);MFE ($);Time in trade;Comment\n"
)
_TICKER = "SYN.pa"


def _symbol(*, lot_step: float = 0.01, min_lot: float = 0.01,
            contract_size: float = 100000.0) -> BrokerSymbol:
    """``BrokerSymbol`` sintético (mismos campos que ``_eurusd`` de test_risk_mm_linearity)."""
    return BrokerSymbol(
        ticker_axi=_TICKER,
        category="forex",
        eligible_for_allocation=True,
        quote_currency="USD",
        contract_size=contract_size,
        tick_size=0.00001,
        pip_size=0.0001,
        tick_value_usd=1.0,
        point_value_usd=100000.0,
        min_lot=min_lot,
        lot_step=lot_step,
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


def _trade_list(pnls, *, size: float = 1.0, balance0: float = 10_000.0, maes=None, prices=None):
    """``TradeList`` sintética: una orden por día, ``Size`` constante, ``Balance`` = acumulado.

    ``equity_before[0] = balance0`` → ``tl.initial_equity == balance0``. ``pnl_per_lot = pnl/size``.
    """
    pnls = [round(float(p), 2) for p in pnls]
    rows, bal = [], float(balance0)
    t0 = pd.Timestamp("2024-01-02")
    for i, p in enumerate(pnls):
        bal += p
        mae = 0.0 if maes is None else float(maes[i])
        price = 1.1 if prices is None else float(prices[i])
        d = (t0 + pd.Timedelta(days=i)).strftime("%Y.%m.%d")
        rows.append(f"{i + 1};{_TICKER};Buy;{d} 00:00;{price:.4f};{size:.2f};{d} 12:00;{price:.4f};"
                    f"{p:.2f};{bal:.2f};IST;Exit After 1 Bars;{mae:.2f};0;1;-\n")
    return load_trade_list((_HEADER + "".join(rows)).encode())


def _sim(tl, f, sl, *, sym=None, compound=None, **kw):
    """``simulate_f`` en modo exacto (``sl_in_money`` dado); ``compound=None`` = no pasar el argumento."""
    extra = {} if compound is None else {"compound": compound}
    return simulate_f(tl, np.atleast_1d(np.asarray(f, dtype=float)), symbols={_TICKER: sym or _symbol()},
                      sl_in_money=np.asarray(sl, dtype=float), **extra, **kw)


def _closed_form_lots(f_grid, e0, sl, *, step, min_lot, max_lots=float("inf"), lots_if_no_mm=None):
    """Oráculo independiente de ``compound=False``: el lote NO depende del camino, así que se calcula
    vectorizado sobre las órdenes. ``floor(f·e0/sl, step)`` y, en este orden: ``LotsIfNoMM`` si ≤ 0,
    ``MaxLots``, lote mínimo. Una orden sin riesgo definido (``sl ≤ 0`` o no finito) no abre (lote 0)."""
    f = np.asarray(f_grid, dtype=float)[:, None]
    sl = np.asarray(sl, dtype=float)[None, :]
    linm = min_lot if lots_if_no_mm is None else lots_if_no_mm
    with np.errstate(divide="ignore", invalid="ignore"):
        lots = np.floor(f * e0 / sl / step) * step
    lots = np.where(lots <= 0, linm, lots)
    lots = np.minimum(lots, max_lots)
    lots = np.where(lots < min_lot, min_lot, lots)
    return np.where((sl <= 0) | ~np.isfinite(sl), 0.0, lots)


# ===========================================================================
# AC1.1: golden a mano (Enmienda E5)
# A = 10 000, f = 1 %, sl = 100 $/lote en las 3 órdenes, lot_step = min_lot = 0,1,
# PnL por lote +1.500, +1.500, -500.
# ===========================================================================
_E5_PNLS = [1500, 1500, -500]
_E5_SL = [100.0, 100.0, 100.0]
_E5_SYM = dict(lot_step=0.1, min_lot=0.1)


def _e5_tl(**kw):
    return _trade_list(_E5_PNLS, size=1.0, balance0=10_000.0, **kw)


def test_ac1_1_golden_e5_compound_false_lotes_exactamente_1_1_1():
    tl = _e5_tl()
    assert tl.initial_equity == 10_000.0
    res = _sim(tl, 0.01, _E5_SL, sym=_symbol(**_E5_SYM), compound=False)
    # lots_cont = 0,01 · 10.000 / 100 = 1,0 en las tres órdenes: la equity del replay no entra.
    assert res.lots[0].tolist() == [1.0, 1.0, 1.0]
    # La equity del replay sí avanza con esos lotes fijos (diagnóstico): 10.000 → 11.500 → 13.000 → 12.500.
    np.testing.assert_allclose(res.equity[0], [10_000, 11_500, 13_000, 12_500], rtol=0, atol=1e-9)
    np.testing.assert_allclose(res.pnl[0], [1500, 1500, -500], rtol=0, atol=1e-9)


def test_ac1_1_golden_e5_compound_true_los_lotes_crecen_con_la_equity():
    tl = _e5_tl()
    res = _sim(tl, 0.01, _E5_SL, sym=_symbol(**_E5_SYM), compound=True)
    # 1.ª: 1,0. 2.ª: E = 11.500 → 1,15 → suelo a 0,1 = 1,1. 3.ª: E = 11.500 + 1,1·1.500 = 13.150 → 1,315 → 1,3.
    np.testing.assert_allclose(res.lots[0], [1.0, 1.1, 1.3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.equity[0], [10_000, 11_500, 13_150, 12_500], rtol=0, atol=1e-9)


def test_ac1_1_golden_e5_sin_pasar_compound_es_el_compuesto():
    """El valor por defecto es el compuesto de siempre (``True``)."""
    res = _sim(_e5_tl(), 0.01, _E5_SL, sym=_symbol(**_E5_SYM))
    np.testing.assert_allclose(res.lots[0], [1.0, 1.1, 1.3], rtol=0, atol=1e-12)


def test_ac1_1_golden_e5_el_lote_fijo_vale_cuenta_por_f_entre_sl_con_otra_cuenta():
    """Con ``compound=False`` el tamaño de cuenta es ``tl.initial_equity``: con 50.000 el lote es ×5."""
    tl = dataclasses.replace(_e5_tl(), initial_equity=50_000.0)   # lo que hace _load_leg (account_size)
    res = _sim(tl, 0.01, _E5_SL, sym=_symbol(**_E5_SYM), compound=False)
    assert res.lots[0].tolist() == [5.0, 5.0, 5.0]


# ===========================================================================
# AC1.2: el lote de la orden i no depende del PnL de las órdenes anteriores (compound=False)
# Property test con casos aleatorios de semilla fija.
# ===========================================================================
@pytest.mark.parametrize("seed", range(10))
def test_ac1_2_lote_de_la_orden_i_no_cambia_si_se_alteran_o_permutan_los_pnl_previos(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 15))
    sl = rng.uniform(40.0, 400.0, n)
    f = float(rng.uniform(0.003, 0.02))
    sym = _symbol()
    tl = _trade_list(rng.normal(0.0, 700.0, n))
    base = _sim(tl, f, sl, sym=sym, compound=False).lots[0]
    # el lote fijo ES la fórmula cerrada
    np.testing.assert_array_equal(
        base, _closed_form_lots([f], tl.initial_equity, sl, step=0.01, min_lot=0.01)[0])
    for i in range(1, n):
        altered = tl.pnl.copy()
        altered[:i] = rng.normal(0.0, 3000.0, i)                 # PnL previos cualesquiera
        permuted = tl.pnl.copy()
        permuted[:i] = rng.permutation(tl.pnl[:i])               # mismos PnL previos, otro orden
        for pnl in (altered, permuted):
            other = _sim(dataclasses.replace(tl, pnl=pnl), f, sl, sym=sym, compound=False).lots[0]
            assert other[i] == base[i], (seed, i)


def test_ac1_2_control_con_compound_true_el_lote_si_depende_del_pnl_previo():
    """Control: sin esto, el property test anterior podría ser vacuo (lotes que nunca dependen del PnL)."""
    tl = _trade_list([800.0, -300.0, 900.0, 400.0, -700.0, 600.0])
    sl = [100.0] * 6
    ricos = _sim(tl, 0.01, sl, compound=True).lots[0]
    pobres = _sim(dataclasses.replace(tl, pnl=-tl.pnl), 0.01, sl, compound=True).lots[0]
    assert ricos[0] == pobres[0]                      # la 1.ª orden no tiene historia
    assert (ricos[1:] != pobres[1:]).any()            # el resto sí depende de ella
    fijos = _sim(tl, 0.01, sl, compound=False).lots[0]
    fijos_pobres = _sim(dataclasses.replace(tl, pnl=-tl.pnl), 0.01, sl, compound=False).lots[0]
    np.testing.assert_array_equal(fijos, fijos_pobres)


# ===========================================================================
# AC1.3: compound=True (por defecto) es bit a bit el comportamiento anterior
# ===========================================================================
# sha256 de TODOS los campos de SweepResult calculados con el ``simulate_f`` ANTERIOR al cambio
# (``src/risk/mm.py`` de e0cbf9a, sin el argumento ``compound``). Receta de regeneración: extraer ese
# fichero con ``git show e0cbf9a:src/risk/mm.py``, importarlo como módulo temporal en lugar de ``src.risk.mm``
# y llamar a ``_digest(_golden_run(margin=...))``.
_G_PNLS = [400, -250, 900, -600, -1500, 300, -200, 1200, -800, 50, -3000, 700, 2500, -400]   # a Size 0,5
_G_MAES = [-60, -90, -40, -350, -800, -30, -120, -150, -500, -10, -2000, -80, -100, -240]
_G_PRICES = [1.10 + 0.01 * i for i in range(len(_G_PNLS))]
# Entradas LITERALES (sin RNG: la semilla de numpy no es contrato entre versiones). Riesgo por lote ($):
# incluye un sl = 0 y un NaN (órdenes sin riesgo definido → lote 0, PnL 0), un sl enorme (lote continuo ≈ 0 →
# LotsIfNoMM) y sl pequeños (con MaxLots = 2,5 se topa a f = 2 %); a f = 2 % el replay se arruina (blown_up).
_G_SL = [100.0, 80.0, 120.0, 0.0, 95.0, 60.0, float("nan"), 110.0, 200.0, 50000.0, 90.0, 70.0, 65.0, 300.0]
_G_F = [0.005, 0.01, 0.02]
_G_FIELDS = ("f_grid", "equity", "mtm_min", "lots", "pnl", "n_sub_min_lot", "n_margin_clipped", "blown_up")
_GOLDEN_DIGEST = {
    False: "9b676b5cb193ea95a8215ca5534db091f6c18fc3a7c21b8240a03d477970539e",   # sin margen (la app)
    True: "72ab0edde4f39bf591d39c8599e116bdb03d40a993a1680c81212a6c10f42694",    # con clip de margen
}


def _digest(res) -> str:
    """sha256 de los bytes de todos los campos de ``SweepResult`` (enteros/bool como int64)."""
    h = hashlib.sha256()
    for name in _G_FIELDS:
        a = np.asarray(getattr(res, name))
        if a.dtype.kind in "iub":
            a = a.astype(np.int64)
        h.update(name.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _golden_tl():
    return _trade_list(_G_PNLS, size=0.5, maes=_G_MAES, prices=_G_PRICES)


def _golden_run(*, margin: bool, **extra):
    return simulate_f(_golden_tl(), np.array(_G_F), symbols={_TICKER: _symbol()}, sl_in_money=np.array(_G_SL),
                      max_lots=2.5, leverage=20, use_margin_clip=margin, **extra)


def test_ac1_3_compound_es_keyword_only_y_true_por_defecto():
    p = inspect.signature(simulate_f).parameters["compound"]
    assert p.default is True
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("margin", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"compound": True}], ids=["por_defecto", "compound=True"])
def test_ac1_3_compound_true_es_bit_a_bit_el_comportamiento_previo(margin, kwargs):
    assert _digest(_golden_run(margin=margin, **kwargs)) == _GOLDEN_DIGEST[margin]


def test_ac1_3_el_golden_cubre_los_caminos_del_replay():
    """Cinturón: el escenario del digest ejercita bounds, skip, blow-up y clip (si no, el digest no protege)."""
    res = _golden_run(margin=False)
    assert res.blown_up.tolist() == [False, False, True]
    assert (res.n_sub_min_lot >= 1).all()                      # LotsIfNoMM / lote mínimo
    assert res.lots.max() == 2.5                               # MaxLots se topa a f = 2 %
    assert (res.lots[:, [3, 6]] == 0.0).all()                  # sl = 0 y sl = NaN: no abren
    assert not np.array_equal(res.lots, _golden_run(margin=True).lots)   # el clip de margen muerde


# ===========================================================================
# compound=False: el resto de la mecánica es idéntico
# ===========================================================================
@pytest.mark.parametrize("seed", range(5))
def test_compound_false_coincide_con_la_formula_cerrada_en_el_escenario_golden(seed):
    """Sobre el escenario del digest (incluye sl = 0/NaN, LotsIfNoMM y MaxLots) y varias cuentas."""
    e0 = [10_000.0, 25_000.0, 7_500.0, 100_000.0, 50_000.0][seed]
    tl = dataclasses.replace(_golden_tl(), initial_equity=e0)
    res = simulate_f(tl, np.array(_G_F), symbols={_TICKER: _symbol()}, sl_in_money=np.array(_G_SL),
                     max_lots=2.5, compound=False)
    exp = _closed_form_lots(_G_F, e0, _G_SL, step=0.01, min_lot=0.01, max_lots=2.5)
    np.testing.assert_array_equal(res.lots, exp)


def test_compound_false_bounds_suelo_lotsifnomm_maxlots_y_lote_minimo():
    """Hecho a mano. A = 10.000, f = 1 % → presupuesto 100 $ por orden. lot_step 0,1, min_lot 0,5, MaxLots 4."""
    sym = _symbol(lot_step=0.1, min_lot=0.5)
    sl = [30.0, 400.0, 1e6, 10.0, 0.0, float("nan")]
    tl = _trade_list([100.0] * 6)
    res = _sim(tl, 0.01, sl, sym=sym, compound=False, max_lots=4.0)
    # 100/30 = 3,33 → suelo 3,3 | 100/400 = 0,25 → suelo 0,2 < min_lot → 0,5 | 100/1e6 ≈ 0 → LotsIfNoMM (= min_lot) 0,5
    # 100/10 = 10 → MaxLots 4,0 | sl = 0 y sl = NaN: sin riesgo definido → no abre
    np.testing.assert_allclose(res.lots[0], [3.3, 0.5, 0.5, 4.0, 0.0, 0.0], rtol=0, atol=1e-12)
    assert res.n_sub_min_lot[0] == 2                            # las dos que subieron a lote mínimo
    np.testing.assert_allclose(res.pnl[0], res.lots[0] * (100.0 / 1.0), rtol=0, atol=1e-9)   # PnL = lote · PnL/lote
    # LotsIfNoMM explícito (0,7 ≥ min_lot): sustituye al lote 0 de la 3.ª orden y no se vuelve a subir
    res2 = _sim(tl, 0.01, sl, sym=sym, compound=False, max_lots=4.0, lots_if_no_mm=0.7)
    np.testing.assert_allclose(res2.lots[0], [3.3, 0.5, 0.7, 4.0, 0.0, 0.0], rtol=0, atol=1e-12)


def test_compound_false_modo_rapido_f_ref_el_lote_a_f_ref_es_size_por_cuenta_sobre_equity():
    """Modo rápido (``sl = f_ref·E_n/Size_n`` back-derivado): con ``compound=False`` y f = f_ref el lote es
    ``floor(Size_n · A / E_n)``, no el ``Size`` del CSV (que sí compone). Hecho a mano con E5: A = 10.000,
    equity antes de cada orden [10.000, 11.500, 13.000] → 1,0 / 0,87 / 0,77 → suelo 0,1 → 1,0 / 0,8 / 0,7."""
    tl = _e5_tl()
    res = simulate_f(tl, np.array([0.01]), symbols={_TICKER: _symbol(**_E5_SYM)}, f_ref=0.01, compound=False)
    np.testing.assert_allclose(res.lots[0], [1.0, 0.8, 0.7], rtol=0, atol=1e-12)


def test_compound_false_margen_opcional_sigue_usando_la_equity_del_replay():
    """«Todo lo demás idéntico»: el clip de margen (apagado en la app) sigue usando la equity del replay."""
    tl = _trade_list([5000.0, 0.0], prices=[1.0, 1.0])          # tras la 1.ª orden la equity del replay es 15.000
    kw = dict(sl=[1.0, 1.0], sym=_symbol(lot_step=0.1, min_lot=0.1), use_margin_clip=True, leverage=10)
    # lots_margin = E·10/(100000·1) → 1,0 con E = 10.000 y 1,5 con E = 15.000
    for compound in (True, False):
        res = _sim(tl, 0.01, compound=compound, **kw)
        np.testing.assert_allclose(res.lots[0], [1.0, 1.5], rtol=0, atol=1e-12, err_msg=f"compound={compound}")
    # sin clip, en cambio, el lote fijo no se mueve y el compuesto sí (control)
    kw["use_margin_clip"] = False
    assert _sim(tl, 0.01, compound=False, **kw).lots[0].tolist() == [100.0, 100.0]
    assert _sim(tl, 0.01, compound=True, **kw).lots[0].tolist() != [100.0, 100.0]


# ===========================================================================
# Enmienda E15: con compound=False, equity / mtm_min / blown_up describen la estrategia a lote fijo
# ===========================================================================
def test_e15_equity_y_mtm_min_a_lote_fijo_hechos_a_mano():
    # E5 con MAE por orden (a Size 1): -200, -100, -700; lotes fijos 1,0
    tl = _e5_tl(maes=[-200, -100, -700])
    res = _sim(tl, 0.01, _E5_SL, sym=_symbol(**_E5_SYM), compound=False)
    # mtm_n = min(E_n + mae·lote, E_{n+1}): min(10.000-200, 11.500) | min(11.500-100, 13.000) | min(13.000-700, 12.500)
    np.testing.assert_allclose(res.mtm_min[0], [9_800, 11_400, 12_300], rtol=0, atol=1e-9)
    assert not res.blown_up[0]


def test_e15_blown_up_describe_el_lote_fijo_y_el_lote_ya_no_se_hunde_con_el_replay():
    """PnL por lote -1.500 (arruina el replay en la 1.ª orden), -600, -600; A = 1.000, f = 1 %, sl = 10 → 1,0 lote."""
    tl = _trade_list([-1500, -600, -600], balance0=1_000.0)
    sym = _symbol(**_E5_SYM)
    fijo = _sim(tl, 0.01, [10.0] * 3, sym=sym, compound=False)
    assert fijo.blown_up[0]
    # la equity del replay se congela tras el blow-up; el LOTE, en cambio, sigue siendo el de la cuenta
    np.testing.assert_allclose(fijo.equity[0], [1_000, -500, -500, -500], rtol=0, atol=1e-9)
    assert fijo.lots[0].tolist() == [1.0, 1.0, 1.0]
    comp = _sim(tl, 0.01, [10.0] * 3, sym=sym, compound=True)
    assert comp.blown_up[0]
    np.testing.assert_allclose(comp.lots[0], [1.0, 0.1, 0.1], rtol=0, atol=1e-12)   # E ≤ 0 → LotsIfNoMM


def test_e15_los_docstrings_lo_dicen():
    for doc in (simulate_f.__doc__, SweepResult.__doc__):
        assert "compound=False" in doc and "lote fijo" in doc
        for campo in ("equity", "mtm_min", "blown_up"):
            assert campo in doc
