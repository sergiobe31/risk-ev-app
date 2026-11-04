"""Tests del constructor de datos del trailing intradía (plan Apex Fase 2).

Todo calculado a mano. Cubre: ejemplo doc §1 (flotante +2,000 que vuelve a 0),
contraejemplo ronda-1 de GLM (excursión real 4,100 que la fórmula v1 sin
realized NO detectaba), sobre-flag inverso (pico +6,000 / cierre +5,990 / valle
−500 ⇒ excursión acotada ≈510, no 6,500), doble conteo en t=ct (cota puntual
lower ≤ eq ≤ upper con igualdad post-fill), trade sin cobertura detectado vía
realized, fallback realized-only (bloque 100% sin cobertura), cortos (espejo
exacto), ot==ct, varios closes en el mismo minuto, rel_exc cross-día con cummax
de bloque, etiquetado por sesión ET, raise ruidoso sin picos, bit-paridad de
``minute_valley_series`` tras el refactor, y el property test T1 (gate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import (
    StrategyMC,
    minute_float_series,
    minute_valley_series,
)
from src.risk.propfirm import (
    PortfolioDayBlock,
    block_day_arrays_portfolio,
    build_day_blocks_portfolio,
)
from src.risk.propfirm_trailing import (
    build_trailing_columns,
    trailing_minute_bounds,
)
from src.risk.session_axis import make_session_day_labeler

D = pd.Timestamp("2024-01-02").date()
D_ORD = pd.Timestamp(D).toordinal()


def _ohlc(bars):
    """DataFrame OHLC desde [(ts, o, h, l, c)]."""
    idx = pd.DatetimeIndex([pd.Timestamp(b[0]) for b in bars])
    return pd.DataFrame(
        {"o": [b[1] for b in bars], "h": [b[2] for b in bars],
         "l": [b[3] for b in bars], "c": [b[4] for b in bars]},
        index=idx,
    )


def _entries(ot, ct, op, is_buy, ohlc):
    """(entrada minute_valle, entrada minute_fav) o (None, None) sin cobertura."""
    par = minute_float_series(pd.Timestamp(ot), pd.Timestamp(ct), op, is_buy, ohlc)
    if par is None:
        return None, None
    s_adv, s_fav = par
    ots, cts = pd.Timestamp(ot), pd.Timestamp(ct)
    return ((s_adv.index.values, s_adv.to_numpy(), ots, cts),
            (s_fav.index.values, s_fav.to_numpy(), ots, cts))


def _blk(trades):
    """PortfolioDayBlock directo (narrativas con trades secuenciales).

    trades: dicts con sym, ot, ct, pnl (USD por lote) y opcionalmente
    ``valle`` (dict {date: vp}) y ``tloc``.
    """
    n = len(trades)
    return PortfolioDayBlock(
        idx=0,
        start=pd.Timestamp(min(t["ot"] for t in trades)),
        end=pd.Timestamp(max(t["ct"] for t in trades)),
        duration_days=0.0,
        sym_id=np.array([t["sym"] for t in trades], dtype=int),
        trade_local_id=np.array([t.get("tloc", 0) for t in trades], dtype=int),
        open_ord=np.array([pd.Timestamp(t["ot"]).toordinal() for t in trades], dtype=int),
        close_ord=np.array([pd.Timestamp(t["ct"]).toordinal() for t in trades], dtype=int),
        valle_precio=[t.get("valle", {}) for t in trades],
        pnl_per_lot=np.array([t["pnl"] for t in trades], dtype=float),
        n_trades=n,
        n_no_valley=n,
        open_t=np.array([pd.Timestamp(t["ot"]).to_datetime64() for t in trades]),
        close_t=np.array([pd.Timestamp(t["ct"]).to_datetime64() for t in trades]),
    )


def _run(blk, covered, pv_by, lots_by=None):
    """block_day_arrays_portfolio con trailing=True.

    ``covered``: {sym: {tloc: (ot, ct, op, is_buy, ohlc)}} — las entradas
    minute se derivan de ``minute_float_series`` (camino real).
    """
    minute_adv, minute_fav = {}, {}
    for sym, trs in covered.items():
        minute_adv[sym], minute_fav[sym] = {}, {}
        for tloc, (ot, ct, op, is_buy, ohlc) in trs.items():
            e_adv, e_fav = _entries(ot, ct, op, is_buy, ohlc)
            assert e_adv is not None, "trade que debía estar cubierto no lo está"
            minute_adv[sym][tloc] = e_adv
            minute_fav[sym][tloc] = e_fav
    syms = set(int(s) for s in blk.sym_id)
    for s in syms:
        minute_adv.setdefault(s, {})
        minute_fav.setdefault(s, {})
    if lots_by is None:
        lots_by = {s: np.ones((1, 1)) for s in syms}
    return block_day_arrays_portfolio(
        blk, lots_by, pv_by, 0, float_mode="bound", trailing=True,
        minute_by_sym=minute_adv, minute_fav_by_sym=minute_fav,
    )


# ---------------------------------------------------------------------------
# Golden 1: ejemplo doc §1 — flotante +2,000 que vuelve a 0
# ---------------------------------------------------------------------------

def _escenario_doc1(is_buy):
    """Largo op=100 pv=1000: pico +2 en m1, cierre en 100 (pnl 0)."""
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 100.5, 99.5, 100.2),
        ("2024-01-02 09:01", 100.2, 102.0, 100.0, 101.0),
        ("2024-01-02 09:02", 101.0, 101.0, 99.8, 100.0),
    ])
    if not is_buy:  # espejo exacto alrededor de 100
        ohlc = _ohlc([
            ("2024-01-02 09:00", 100.0, 100.5, 99.5, 99.8),
            ("2024-01-02 09:01", 99.8, 100.0, 98.0, 99.0),
            ("2024-01-02 09:02", 99.0, 100.2, 99.0, 100.0),
        ])
    blk = _blk([dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 09:02", pnl=0.0)])
    return blk, {0: {0: ("2024-01-02 09:00", "2024-01-02 09:02", 100.0, is_buy, ohlc)}}


def test_doc1_flotante_2000_vuelve_a_cero():
    blk, covered = _escenario_doc1(is_buy=True)
    bda = _run(blk, covered, {0: 1000.0})
    # adv = [-0.5, 0, -0.2], fav = [+0.5, +2, +1] (precio) ×1000.
    # Envolvente muere EN ct (09:02): lower/upper allí = realized = 0.
    assert bda.block_low == pytest.approx(-500.0, abs=1e-9)
    assert bda.block_high == pytest.approx(2000.0, abs=1e-9)
    # La excursión sube 2,000 y NO retrocede cuando el flotante vuelve a 0.
    assert bda.block_exc == pytest.approx(2000.0, abs=1e-9)
    np.testing.assert_allclose(bda.rel_low, [-500.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_high, [2000.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_exc, [2000.0], atol=1e-9)


def test_cortos_espejo_exacto():
    blk_l, cov_l = _escenario_doc1(is_buy=True)
    blk_s, cov_s = _escenario_doc1(is_buy=False)
    bda_l = _run(blk_l, cov_l, {0: 1000.0})
    bda_s = _run(blk_s, cov_s, {0: 1000.0})
    assert bda_s.block_low == pytest.approx(bda_l.block_low, abs=1e-9)
    assert bda_s.block_high == pytest.approx(bda_l.block_high, abs=1e-9)
    assert bda_s.block_exc == pytest.approx(bda_l.block_exc, abs=1e-9)
    np.testing.assert_allclose(bda_s.rel_exc, bda_l.rel_exc, atol=1e-9)


# ---------------------------------------------------------------------------
# Golden 2: contraejemplo ronda-1 de GLM — excursión real 4,100 detectada
# ---------------------------------------------------------------------------

def test_contraejemplo_ronda1_glm():
    # A cierra +1,000 (10:00, SIN cobertura — entra vía realized);
    # B abre 10:30, pico flotante +800 (11:00), cierra −500 (13:00);
    # C valle −2,800 (14:00), cierra 0 (15:00). pv=100, 1 lote.
    ohlc_b = _ohlc([
        ("2024-01-02 10:30", 100.0, 101.0, 99.0, 100.0),
        ("2024-01-02 11:00", 100.0, 108.0, 99.5, 107.0),
        ("2024-01-02 12:00", 107.0, 107.5, 106.0, 106.5),
        ("2024-01-02 12:30", 106.5, 106.5, 95.0, 95.5),
    ])
    ohlc_c = _ohlc([
        ("2024-01-02 12:00", 50.0, 50.5, 49.0, 49.5),
        ("2024-01-02 13:00", 49.5, 50.0, 48.5, 49.0),
        ("2024-01-02 14:00", 49.0, 49.0, 22.0, 48.5),
    ])
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=1000.0),
        dict(sym=1, ot="2024-01-02 10:30", ct="2024-01-02 13:00", pnl=-500.0),
        dict(sym=2, ot="2024-01-02 12:00", ct="2024-01-02 15:00", pnl=0.0),
    ])
    covered = {
        1: {0: ("2024-01-02 10:30", "2024-01-02 13:00", 100.0, True, ohlc_b)},
        2: {0: ("2024-01-02 12:00", "2024-01-02 15:00", 50.0, True, ohlc_c)},
    }
    bda = _run(blk, covered, {0: 100.0, 1: 100.0, 2: 100.0})
    # A mano (calendario 10:00,10:30,11:00,12:00,12:30,13:00,14:00,15:00):
    # realized = 1000,1000,1000,1000,1000,500,500,500
    # lower = 1000, 900, 950, 1500, 400, 350, -2300, 500
    # upper = 1000, 1100, 1800, 1800, 1700, 500, 400, 500
    # cummax(upper) = 1000,1100,1800×6 → exc = 0,200,850,300,1400,1450,4100,1300
    assert bda.block_low == pytest.approx(-2300.0, abs=1e-9)
    assert bda.block_high == pytest.approx(1800.0, abs=1e-9)
    # La excursión real de 4,100 QUEDA DETECTADA (la v1 sin stream realized
    # daba 3,600: cummax de envolvente 800 − valle envolvente −2,800).
    assert bda.block_exc == pytest.approx(4100.0, abs=1e-9)
    np.testing.assert_allclose(bda.rel_low, [-2300.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_high, [1800.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_exc, [4100.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Golden 3: sobre-flag inverso — excursión acotada ≈510, no 6,500
# ---------------------------------------------------------------------------

def test_sobre_flag_inverso_acotado():
    # T1: largo op=100 pv=100, 60 barras de subida limpia +1/minuto (sin
    # mechas bajistas: low = open = cierre previo), pico 160 a las 09:59,
    # cierre 159.9 (10:00) → realized +5,990. T2: largo op=200 con valle
    # −500 en su primer minuto (10:00), cierre plano (11:00).
    bars = [("2024-01-02 09:%02d" % k, 100.0 + k, 101.0 + k, 100.0 + k, 101.0 + k)
            for k in range(60)]
    ohlc_t1 = _ohlc(bars)
    ohlc_t2 = _ohlc([("2024-01-02 10:00", 200.0, 200.0, 195.0, 195.0)])
    pnl_t1 = (159.9 - 100.0) * 100.0  # ≈ 5,990
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=pnl_t1),
        dict(sym=1, ot="2024-01-02 10:00", ct="2024-01-02 11:00", pnl=0.0),
    ])
    covered = {
        0: {0: ("2024-01-02 09:00", "2024-01-02 10:00", 100.0, True, ohlc_t1)},
        1: {0: ("2024-01-02 10:00", "2024-01-02 11:00", 200.0, True, ohlc_t2)},
    }
    bda = _run(blk, covered, {0: 100.0, 1: 100.0})
    # Durante la subida: exc = fav_k − adv_k = 100 por minuto (tope 100).
    # A las 10:00: lower = 5,990 − 500 = 5,490; cummax(upper) = 6,000 →
    # exc = 510. La v1 sobre-flageaba ~6,500 (cummax envolvente 6,000 contra
    # valle posterior sin realized).
    assert bda.block_exc == pytest.approx(510.0, abs=1e-6)
    assert bda.block_exc < 1000.0
    assert bda.block_low == pytest.approx(0.0, abs=1e-9)
    assert bda.block_high == pytest.approx(6000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden 4: doble conteo en t=ct — cota puntual con igualdad post-fill
# ---------------------------------------------------------------------------

def _perdedor_fav_neg():
    """Largo que cierra perdiendo; su envolvente favorable es <0 antes de ct."""
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 101.0, 99.0, 99.5),
        ("2024-01-02 09:30", 99.5, 99.5, 98.0, 98.5),
    ])
    return ohlc, -200.0   # cierra en 98 → pnl −200 (pv=100)


def _ganador_adv_pos():
    """Largo que cierra ganando; su envolvente adversa es >0 antes de ct."""
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 102.0, 100.0, 101.5),
        ("2024-01-02 09:30", 101.5, 103.0, 101.0, 102.5),
    ])
    return ohlc, +300.0   # cierra en 103 → pnl +300 (pv=100)


def test_doble_conteo_ct_cota_puntual():
    ohlc_l, pnl_l = _perdedor_fav_neg()
    ohlc_g, pnl_g = _ganador_adv_pos()
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=pnl_l),
        dict(sym=1, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=pnl_g),
    ])
    covered = {
        0: {0: ("2024-01-02 09:00", "2024-01-02 10:00", 100.0, True, ohlc_l)},
        1: {0: ("2024-01-02 09:00", "2024-01-02 10:00", 100.0, True, ohlc_g)},
    }
    minute_adv = {s: {0: _entries(*c[:2], c[2], c[3], c[4])[0]} for s, c in
                  ((0, covered[0][0]), (1, covered[1][0]))}
    minute_fav = {s: {0: _entries(*c[:2], c[2], c[3], c[4])[1]} for s, c in
                  ((0, covered[0][0]), (1, covered[1][0]))}
    # Las series constituyentes ejercen el riesgo de doble conteo:
    _, adv_l, _, _ = minute_adv[0][0]
    _, fav_l, _, _ = minute_fav[0][0]
    _, adv_g, _, _ = minute_adv[1][0]
    assert fav_l[-1] * 100.0 == pytest.approx(-50.0)   # perdedor con fav<0
    assert adv_g[-1] * 100.0 == pytest.approx(+100.0)  # ganador con adv>0
    tcols = build_trailing_columns(blk, minute_adv, minute_fav)
    lots = np.ones(2)
    lower, upper = trailing_minute_bounds(blk, lots, {0: 100.0, 1: 100.0}, tcols)
    # Cota puntual lower ≤ eq ≤ upper; en t=ct (ambos closes en el mismo
    # minuto) la igualdad es exacta: eq post-fill = realized = −200 + 300.
    assert np.all(lower <= upper + 1e-12)
    for j in range(2):
        p = tcols.ct_pos[j]
        assert lower[p] == pytest.approx(100.0, abs=1e-9)
        assert upper[p] == pytest.approx(100.0, abs=1e-9)
        # La envolvente muere EN ct (cero, no ffill) — sin doble conteo.
        for col in tcols.adv_cols + tcols.fav_cols:
            assert col[p] == 0.0
    # Cada trade por separado: igualdad exacta en su ct con su realized.
    for sym, pnl in ((0, pnl_l), (1, pnl_g)):
        b1 = _blk([dict(sym=sym, ot="2024-01-02 09:00", ct="2024-01-02 10:00",
                        pnl=pnl)])
        tc1 = build_trailing_columns(b1, {sym: minute_adv[sym]},
                                     {sym: minute_fav[sym]})
        lo1, up1 = trailing_minute_bounds(b1, np.ones(1), {sym: 100.0}, tc1)
        assert lo1[tc1.ct_pos[0]] == pytest.approx(pnl, abs=1e-9)
        assert up1[tc1.ct_pos[0]] == pytest.approx(pnl, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden 5: trade sin cobertura (−2,600) tras pico +1,500 — vía realized
# ---------------------------------------------------------------------------

def test_sin_cobertura_detectado_via_realized():
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 115.0, 100.0, 115.0),
        ("2024-01-02 09:30", 115.0, 115.0, 114.0, 114.5),
    ])
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=1500.0),
        dict(sym=1, ot="2024-01-02 09:30", ct="2024-01-02 11:00", pnl=-2600.0),
    ])
    covered = {0: {0: ("2024-01-02 09:00", "2024-01-02 10:00", 100.0, True, ohlc)}}
    bda = _run(blk, covered, {0: 100.0, 1: 100.0})
    # realized: 0, 0, +1500 (10:00), −1100 (11:00). upper pega 1,500 (pico de
    # T1 + su cierre); lower cae a −1,100 cuando el trade SIN cobertura cierra
    # −2,600 — su close exacto entra vía realized EN su minuto ct.
    assert bda.block_low == pytest.approx(-1100.0, abs=1e-9)
    assert bda.block_high == pytest.approx(1500.0, abs=1e-9)
    assert bda.block_exc == pytest.approx(2600.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden 6: bloque 100% sin cobertura — fallback realized-only
# ---------------------------------------------------------------------------

def test_bloque_sin_cobertura_fallback_realized():
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 10:00", pnl=1000.0),
        dict(sym=1, ot="2024-01-02 09:30", ct="2024-01-02 11:00", pnl=-3000.0),
    ])
    bda = _run(blk, {}, {0: 100.0, 1: 100.0})
    # Calendario = {10:00, 11:00} (solo cts); lower == upper == cumsum closes.
    assert bda.block_low == pytest.approx(-2000.0, abs=1e-9)
    assert bda.block_high == pytest.approx(1000.0, abs=1e-9)
    # Nunca no-op silencioso: la excursión realized 1,000 → −2,000 se mide.
    assert bda.block_exc == pytest.approx(3000.0, abs=1e-9)
    np.testing.assert_allclose(bda.rel_low, [-2000.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_high, [1000.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_exc, [3000.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Golden 7: trade ot==ct (mismo minuto) — solo aporta realized
# ---------------------------------------------------------------------------

def test_ot_igual_ct_solo_realized():
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 100.0, 100.0, 100.0),
        ("2024-01-02 10:00", 100.0, 100.0, 100.0, 100.0),
        ("2024-01-02 10:30", 100.0, 100.0, 100.0, 100.0),
    ])
    blk = _blk([
        dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-02 11:00", pnl=0.0),
        dict(sym=1, ot="2024-01-02 10:00", ct="2024-01-02 10:00", pnl=700.0),
    ])
    covered = {0: {0: ("2024-01-02 09:00", "2024-01-02 11:00", 100.0, True, ohlc)}}
    bda = _run(blk, covered, {0: 100.0, 1: 100.0})
    # El trade instantáneo no tiene serie (ct <= ot → None) pero su close
    # +700 entra en realized en su minuto ct: block_high = 700, exc = 0.
    assert bda.block_low == pytest.approx(0.0, abs=1e-9)
    assert bda.block_high == pytest.approx(700.0, abs=1e-9)
    assert bda.block_exc == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden 8: rel_exc cross-día — cummax DE BLOQUE, atribución al día del valle
# ---------------------------------------------------------------------------

def test_rel_exc_cross_dia():
    # Largo de 3 días: pico +1,000 el día 1, valle −1,500 el día 3, cierre 0.
    ohlc = _ohlc([
        ("2024-01-02 09:00", 100.0, 110.0, 100.0, 109.0),
        ("2024-01-03 09:00", 109.0, 109.0, 108.0, 108.5),
        ("2024-01-04 09:00", 108.5, 108.5, 85.0, 86.0),
    ])
    d1, d2, d3 = (pd.Timestamp(f"2024-01-0{i}").date() for i in (2, 3, 4))
    blk = _blk([dict(sym=0, ot="2024-01-02 09:00", ct="2024-01-04 15:00",
                     pnl=0.0, valle={d1: 0.0, d2: 8.0, d3: -15.0})])
    covered = {0: {0: ("2024-01-02 09:00", "2024-01-04 15:00", 100.0, True, ohlc)}}
    bda = _run(blk, covered, {0: 100.0})
    # upper: 1000, 900, 850, 0(ct) — cummax de bloque = 1,000 desde el día 1.
    # lower: 0, 800, −1500, 0(ct). exc: 1000, 200, 2500, 1000.
    ords = [pd.Timestamp(d).toordinal() for d in (d1, d2, d3)]
    np.testing.assert_array_equal(bda.day_ord, ords)
    np.testing.assert_allclose(bda.rel_low, [0.0, 800.0, -1500.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_high, [1000.0, 900.0, 850.0], atol=1e-9)
    # La excursión cross-día (pico día 1 → valle día 3) se atribuye al DÍA 3.
    np.testing.assert_allclose(bda.rel_exc, [1000.0, 200.0, 2500.0], atol=1e-9)
    assert bda.block_exc == pytest.approx(2500.0, abs=1e-9)
    assert bda.block_low == pytest.approx(-1500.0, abs=1e-9)
    assert bda.block_high == pytest.approx(1000.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden 9: etiquetado por sesión ET (day_labeler) en los minutos
# ---------------------------------------------------------------------------

def test_day_labeler_sesion_et():
    # X cierra 21:30 UTC (16:30 ET → sesión del 2); Y cierra 00:30 UTC del 3
    # (19:30 ET del 2 → sesión del 3). Mismo bloque, dos sesiones.
    labeler = make_session_day_labeler("UTC")
    x = StrategyMC(
        key="X", open_t=np.array([np.datetime64("2024-01-02T20:00")]),
        close_t=np.array([np.datetime64("2024-01-02T21:30")]),
        close_d=np.array([pd.Timestamp("2024-01-02").date()]),
        valle={}, pnl_per_lot=np.array([500.0]), is_buy=np.array([True]),
        pv=1.0, lots=np.ones((1, 1)), n_trades=1,
    )
    y = StrategyMC(
        key="Y", open_t=np.array([np.datetime64("2024-01-02T21:00")]),
        close_t=np.array([np.datetime64("2024-01-03T00:30")]),
        close_d=np.array([pd.Timestamp("2024-01-03").date()]),
        valle={}, pnl_per_lot=np.array([-200.0]), is_buy=np.array([True]),
        pv=1.0, lots=np.ones((1, 1)), n_trades=1,
    )
    blk = build_day_blocks_portfolio({"X": x, "Y": y}, day_labeler=labeler)[0]
    bda = block_day_arrays_portfolio(
        blk, {0: x.lots, 1: y.lots}, {0: 1.0, 1: 1.0}, 0,
        float_mode="bound", trailing=True,
        minute_by_sym={0: {}, 1: {}}, minute_fav_by_sym={0: {}, 1: {}},
        day_labeler=labeler,
    )
    ords = [pd.Timestamp("2024-01-02").toordinal(),
            pd.Timestamp("2024-01-03").toordinal()]
    np.testing.assert_array_equal(bda.day_ord, ords)
    # Sesión del 2: realized 500. Sesión del 3: realized 300; excursión
    # cross-sesión con cummax de bloque: 500 − 300 = 200.
    np.testing.assert_allclose(bda.rel_low, [500.0, 300.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_high, [500.0, 300.0], atol=1e-9)
    np.testing.assert_allclose(bda.rel_exc, [0.0, 200.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Humo: firma trailing sin picos ⇒ raise ruidoso
# ---------------------------------------------------------------------------

def test_trailing_sin_picos_raise_ruidoso():
    blk, covered = _escenario_doc1(is_buy=True)
    e_adv, _e_fav = _entries(*covered[0][0][:2], covered[0][0][2],
                             covered[0][0][3], covered[0][0][4])
    with pytest.raises(ValueError, match="minute_fav"):
        block_day_arrays_portfolio(
            blk, {0: np.ones((1, 1))}, {0: 1000.0}, 0, float_mode="bound",
            trailing=True, minute_by_sym={0: {0: e_adv}}, minute_fav_by_sym=None,
        )
    # Adversa presente pero favorable ausente para el mismo trade.
    with pytest.raises(ValueError, match="with_peaks"):
        block_day_arrays_portfolio(
            blk, {0: np.ones((1, 1))}, {0: 1000.0}, 0, float_mode="bound",
            trailing=True, minute_by_sym={0: {0: e_adv}}, minute_fav_by_sym={0: {}},
        )


# ---------------------------------------------------------------------------
# Bit-paridad: minute_valley_series == proyección adversa de minute_float_series
# ---------------------------------------------------------------------------

def test_bit_paridad_valley_tras_refactor():
    rng = np.random.default_rng(7)
    for is_buy in (True, False):
        n = 200
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        op = np.r_[100.0, close[:-1]]
        high = np.maximum(op, close) + rng.uniform(0, 1, n)
        low = np.minimum(op, close) - rng.uniform(0, 1, n)
        ohlc = pd.DataFrame({"o": op, "h": high, "l": low, "c": close},
                            index=pd.date_range("2024-01-02 09:00", periods=n,
                                                freq="min"))
        ot, ct = ohlc.index[5], ohlc.index[150]
        s_v = minute_valley_series(ot, ct, 100.0, is_buy, ohlc)
        par = minute_float_series(ot, ct, 100.0, is_buy, ohlc)
        assert par is not None and s_v is not None
        np.testing.assert_array_equal(s_v.to_numpy(), par[0].to_numpy())
        assert s_v.index.equals(par[0].index)
        # La favorable acota a la adversa en todo minuto.
        assert np.all(par[0].to_numpy() <= par[1].to_numpy() + 1e-12)


# ---------------------------------------------------------------------------
# T1 (GATE): property test contra simulador de referencia minuto-a-minuto
# ---------------------------------------------------------------------------

class _T1Trade:
    __slots__ = ("si", "ot", "ct", "op", "cp", "is_buy", "lots", "covered")

    def pnl_usd(self, pv_by_sym):
        signo = 1.0 if self.is_buy else -1.0
        return (self.cp - self.op) * signo * self.lots * pv_by_sym[self.si]


def _gen_bars(rng, start, n_min, base=100.0, step=1.0):
    """Barras M1 continuas en precio (open = close anterior de la secuencia)."""
    idx = pd.date_range(start, periods=n_min, freq="min")
    o = np.empty(n_min)
    c = np.empty(n_min)
    prev = base
    for i in range(n_min):
        o[i] = prev
        c[i] = prev + rng.uniform(-step, step)
        prev = c[i]
    h = np.maximum(o, c) + rng.uniform(0, step, n_min)
    l = np.minimum(o, c) - rng.uniform(0, step, n_min)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)


def _t1_try(rng, unambiguous: bool):
    """Un intento de escenario: 1-3 símbolos, 1-6 trades encadenados.

    ``unambiguous``: rejilla M1 completa compartida, todos los trades
    cubiertos, fills dentro de cobertura (op = open de la barra en ot,
    cp = close de la barra en ct), sin ot==ct y UNA SOLA dirección por
    símbolo (con direcciones mezcladas en el mismo símbolo, la cota de la
    fórmula — que no asume orden intra-minuto — no es alcanzable por ningún
    path real y la igualdad exacta no aplica) — en estos casos la fórmula y
    el referente deben COINCIDIR exactamente. Si no: trades sin cobertura,
    fills arbitrarios fuera de cobertura y ot==ct permitidos — vale la
    propiedad unilateral (fórmula ⊇ referente).

    Devuelve None si los trades no encadenan en un único bloque flat-flat
    (un ot==ct parte el busy-interval en ``detect_flatflat_blocks``: el close
    se procesa antes que el open en el mismo timestamp).
    """
    start = pd.Timestamp("2024-03-11 09:00")
    span = int(rng.integers(60, 240))
    n_sym = int(rng.integers(1, 4))
    bars_by_sym = {}
    dir_by_sym = {}
    for si in range(n_sym):
        if unambiguous:
            off, n = 0, span
        else:
            off = int(rng.integers(0, span // 2 + 1))
            n = int(rng.integers(10, span - off + 1))
        bars_by_sym[si] = _gen_bars(rng, start + pd.Timedelta(minutes=off), n)
        dir_by_sym[si] = bool(rng.integers(0, 2))
    pv_by_sym = {si: float(rng.choice([1.0, 10.0, 100.0])) for si in range(n_sym)}
    n_tr = int(rng.integers(1, 7))
    trades: list[_T1Trade] = []
    busy_end = None
    first_ot = int(rng.integers(0, 8))
    for k in range(n_tr):
        tr = _T1Trade()
        tr.si = int(rng.integers(0, n_sym))
        if k == 0:
            ot_min = first_ot
            dur = int(rng.integers(1, 5 if unambiguous else 30))
        else:
            ot_min = int(rng.integers(first_ot, busy_end))
            dur = int(rng.integers(1, 4) if unambiguous
                      else int(rng.choice([0, 1, 2, 5, 15, 40])))
        if unambiguous:
            ot_min = min(ot_min, span - 6)
            dur = min(dur, span - 1 - ot_min)
        tr.ot = start + pd.Timedelta(minutes=ot_min)
        tr.ct = start + pd.Timedelta(minutes=ot_min + dur)
        tr.is_buy = dir_by_sym[tr.si] if unambiguous else bool(rng.integers(0, 2))
        tr.lots = float(rng.integers(1, 4))
        bars = bars_by_sym[tr.si]
        tr.covered = minute_float_series(tr.ot, tr.ct, 1.0, tr.is_buy, bars) is not None
        if unambiguous:
            assert tr.covered and tr.ct > tr.ot
            tr.op = float(bars.loc[tr.ot, "o"])
            tr.cp = float(bars.loc[tr.ct, "c"])
        else:
            # Fills arbitrarios: dentro o fuera de cobertura, precio libre.
            tr.op = float(rng.uniform(90, 110))
            tr.cp = float(rng.uniform(90, 110))
        trades.append(tr)
        busy_end = ot_min + dur if busy_end is None else max(busy_end, ot_min + dur)
    # Rechazo: un trade ot==ct parte el busy-interval en detect_flatflat_blocks
    # (close procesado antes que open en el mismo timestamp) — exigimos un
    # único bloque flat-flat que contenga TODOS los trades.
    from src.risk.mc_portfolio import detect_flatflat_blocks
    blocks = detect_flatflat_blocks(
        {0: (np.array([tr.ot.to_datetime64() for tr in trades]),
             np.array([tr.ct.to_datetime64() for tr in trades]))}
    )
    if len(blocks) != 1:
        return None
    b0, b1 = blocks[0]
    if not all(tr.ot >= b0 and tr.ct <= b1 for tr in trades):
        return None
    return bars_by_sym, pv_by_sym, trades


def _t1_scenario(rng, unambiguous: bool):
    """Escenario válido (1 bloque); re-muestrea si el intento no encadena."""
    for _ in range(100):
        out = _t1_try(rng, unambiguous)
        if out is not None:
            return out
    raise AssertionError("no se pudo generar un escenario de un solo bloque")

def _reference_maxdd(bars_by_sym, pv_by_sym, trades, exact: bool, rng):
    """Referente minuto-a-minuto de fuerza bruta (estado POST-FILL en ct).

    Recorre el calendario (barras ∪ ots ∪ cts); en cada minuto con barras
    aplica fases OHLC sin supuesto de orden global. En ``exact`` (casos sin
    ambigüedad, una dirección por símbolo) evalúa las ESQUINAS ADVERSARIALES
    del minuto — todos los símbolos en su extremo favorable, luego todos en
    el adverso, luego el cierre — realizable intra-minuto porque los símbolos
    son independientes y no se asume orden; así los extremos de la fórmula
    son ALCANZADOS. Si no, fases aleatorias por símbolo (paths realistas que
    en general no alcanzan la cota). Los trades cubiertos se marcan al precio
    de su símbolo (inicializado en su fill op); los NO cubiertos aportan
    flotante 0 hasta su close exacto (no hay información de precio en su
    vida — el wander real es el residual R2 documentado, fuera de este test).
    En ct la posición cierra al fill exacto y el minuto ct se modela
    post-fill completo. Devuelve el max drawdown desde el running-peak (peak
    anclado en 0 = inicio de bloque flat), relativo al saldo de inicio.
    """
    bar_at = {si: {t: (float(r.o), float(r.h), float(r.l), float(r.c))
                   for t, r in bars.iterrows()}
              for si, bars in bars_by_sym.items()}
    closes_at: dict = {}
    for tr in trades:
        closes_at[tr.ct] = closes_at.get(tr.ct, 0.0) + tr.pnl_usd(pv_by_sym)
    cal = sorted(set().union(*[set(b.index) for b in bars_by_sym.values()],
                             {tr.ot for tr in trades}, {tr.ct for tr in trades}))
    marks: dict[int, float] = {}
    realized = 0.0
    peak = 0.0
    dd = 0.0

    def equity(open_trs):
        eq = realized
        for tr in open_trs:
            if tr.covered:
                signo = 1.0 if tr.is_buy else -1.0
                eq += (marks[id(tr)] - tr.op) * signo * tr.lots * pv_by_sym[tr.si]
        return eq

    for t in cal:
        realized += closes_at.get(t, 0.0)  # post-fill desde el inicio del minuto
        open_trs = [tr for tr in trades if tr.ot <= t < tr.ct]
        for tr in open_trs:
            if tr.covered and id(tr) not in marks:
                marks[id(tr)] = tr.op
        syms = [si for si in bar_at if t in bar_at[si]]

        def upd(si, ph):
            o, h, l, c = bar_at[si][t]
            px = {"o": o, "h": h, "l": l, "c": c}[ph]
            for tr in open_trs:
                if tr.covered and tr.si == si:
                    marks[id(tr)] = px

        def eval_eq():
            nonlocal peak, dd
            eq = equity(open_trs)
            peak = max(peak, eq)
            dd = max(dd, peak - eq)

        if not syms:
            eval_eq()
            continue
        if exact:
            # Esquinas adversariales: extremo favorable de cada símbolo, luego
            # el adverso (prev_close = open por continuidad de las barras).
            for corner in ("fav", "adv"):
                for si in syms:
                    o, h, l, c = bar_at[si][t]
                    buys = [tr for tr in open_trs if tr.covered and tr.si == si]
                    if not buys:
                        continue
                    dirn = buys[0].is_buy  # una dirección por símbolo
                    px = (max(h, o) if dirn else min(l, o)) if corner == "fav" \
                        else (min(l, o) if dirn else max(h, o))
                    for tr in buys:
                        marks[id(tr)] = px
                eval_eq()
            for si in syms:
                upd(si, "c")
            eval_eq()
            continue
        for si in syms:
            upd(si, "o")
        eval_eq()
        events = []
        for si in syms:
            phs = [p for p in ("h", "l") if rng.random() < 0.9]
            rng.shuffle(phs)
            events += [(si, p) for p in phs]
        rng.shuffle(events)
        for si, ph in events:
            upd(si, ph)
            eval_eq()
        for si in syms:
            upd(si, "c")
        eval_eq()
    return dd


def _formula_dd(bars_by_sym, pv_by_sym, trades):
    """max(−block_low, block_exc) de la fórmula (hwm = S, bloque único)."""
    by_sym: dict[int, list] = {}
    for tr in trades:
        by_sym.setdefault(tr.si, []).append(tr)
    sims = sorted(by_sym)  # sym_id del bloque = orden de enumeración del dict
    strategies = {}
    for si in sims:
        trs = by_sym[si]
        n = len(trs)
        mv, mf = {}, {}
        for i, tr in enumerate(trs):
            e_adv, e_fav = _entries(tr.ot, tr.ct, tr.op, tr.is_buy,
                                    bars_by_sym[si])
            if e_adv is not None:
                mv[i], mf[i] = e_adv, e_fav
        pnl_per_lot = np.array([
            (tr.cp - tr.op) * (1.0 if tr.is_buy else -1.0) * pv_by_sym[si]
            for tr in trs
        ])
        strategies[f"S{si}"] = StrategyMC(
            key=f"S{si}",
            open_t=np.array([tr.ot.to_datetime64() for tr in trs]),
            close_t=np.array([tr.ct.to_datetime64() for tr in trs]),
            close_d=np.array([tr.ct.date() for tr in trs]),
            valle={}, pnl_per_lot=pnl_per_lot,
            is_buy=np.array([tr.is_buy for tr in trs]),
            pv=pv_by_sym[si],
            lots=np.array([[tr.lots for tr in trs]]), n_trades=n,
            minute_valle=mv, minute_fav=mf,
        )
    blocks = build_day_blocks_portfolio(strategies)
    assert len(blocks) == 1, "el generador debe encadenar los trades en 1 bloque"
    blk = blocks[0]
    assert blk.n_trades == len(trades)
    # sym_id del bloque = índice de enumeración del dict de estrategias.
    lots_by = {k: strategies[f"S{si}"].lots for k, si in enumerate(sims)}
    pv_by = {k: pv_by_sym[si] for k, si in enumerate(sims)}
    minute_adv = {k: strategies[f"S{si}"].minute_valle for k, si in enumerate(sims)}
    minute_fav = {k: strategies[f"S{si}"].minute_fav for k, si in enumerate(sims)}
    bda = block_day_arrays_portfolio(
        blk, lots_by, pv_by, 0, float_mode="bound", trailing=True,
        minute_by_sym=minute_adv, minute_fav_by_sym=minute_fav,
    )
    return max(-bda.block_low, bda.block_exc), bda


def test_t1_property_formula_superset_referente():
    """GATE Fase 2. Propiedad unilateral: para todo path, si el referente
    tiene breach trailing (dd desde running-peak ≥ amount) la fórmula TAMBIÉN
    flaggea — equivalente a ``formula_dd ≥ ref_dd`` para todo umbral. En los
    casos sin ambigüedad (todo cubierto, fills dentro de cobertura) deben
    coincidir EXACTAMENTE. El referente tiene convención propia post-fill en
    ct (no hereda la del joint-float), o el test sería vacuo.
    """
    rng = np.random.default_rng(20260914)
    n_unambiguous = 0
    for it in range(400):
        unambiguous = (it % 4 == 0)
        bars_by_sym, pv_by_sym, trades = _t1_scenario(rng, unambiguous)
        ref_dd = _reference_maxdd(bars_by_sym, pv_by_sym, trades,
                                  exact=unambiguous, rng=rng)
        formula_dd, bda = _formula_dd(bars_by_sym, pv_by_sym, trades)
        assert bda.block_low <= bda.block_high + 1e-9
        assert formula_dd >= ref_dd - 1e-6, (
            f"iter {it}: referente {ref_dd} > fórmula {formula_dd} "
            f"(la fórmula SUB-flaggea — la cota no se sostiene)"
        )
        if unambiguous:
            n_unambiguous += 1
            assert formula_dd == pytest.approx(ref_dd, rel=1e-9, abs=1e-6), (
                f"iter {it}: sin ambigüedad deben coincidir exactamente "
                f"(referente {ref_dd}, fórmula {formula_dd})"
            )
    assert n_unambiguous >= 80
