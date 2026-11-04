"""Tests del front-half del motor propfirm (src/risk/propfirm.py).

Los tests de evaluate_* (test_risk_propfirm.py) construyen ``BlockDayArrays`` a
mano; aqui se cubre la CONSTRUCCION desde ``StrategyMC``:

  - build_day_blocks: cortes en instantes flat-flat, incluida la convencion
    critica del motor: en el mismo timestamp el close(-1) se ordena ANTES que
    el open(+1) (``events.sort(key=lambda e: (e[0], e[1]))`` en
    ``detect_flatflat_blocks``), para no romper close-then-open.
  - block_day_arrays: semantica de rel_before / rel_closed / rel_float con
    valores hand-computed (razonados en los comentarios).
  - empirical_gaps: dias flat entre bloques consecutivos.

Datos sinteticos pequenos (timestamps controlados), sin CSVs ni OHLC real.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.risk.mc_portfolio import StrategyMC
from src.risk.propfirm import (
    block_day_arrays,
    build_day_blocks,
    empirical_gaps,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _strategy(trades, valle, ppl, pv=100.0):
    """StrategyMC sintetica mono-estrategia.

    trades: list[(open_ts, close_ts, close_date)]; valle: {i: {date: vp}}
    (valle en precio, <=0 adverso); ppl: pnl_per_lot por trade.
    """
    n = len(trades)
    return StrategyMC(
        key="S",
        open_t=np.array([_ts(t[0]) for t in trades]),
        close_t=np.array([_ts(t[1]) for t in trades]),
        close_d=np.array([t[2] for t in trades]),
        valle=valle,
        pnl_per_lot=np.array(ppl, dtype=float),
        is_buy=np.ones(n, dtype=bool),
        pv=pv,
        lots=np.empty((0, n)),  # block_day_arrays recibe la fila lots_f aparte
        n_trades=n,
    )


D1, D2, D3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)


# ---------------------------------------------------------------------------
# build_day_blocks: cortes flat-flat
# ---------------------------------------------------------------------------

def test_build_day_blocks_corta_en_instantes_flat_flat():
    # Dos episodios disjuntos -> 2 bloques, cada uno con su trade.
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 12:00", D1),
            ("2024-01-02 10:00", "2024-01-02 12:00", D2),
        ],
        valle={0: {D1: -1.0}, 1: {D2: -1.0}},
        ppl=[10.0, 20.0],
    )
    blocks = build_day_blocks(s)
    assert len(blocks) == 2
    assert [b.idx for b in blocks] == [0, 1]
    assert blocks[0].trade_local_id.tolist() == [0]
    assert blocks[1].trade_local_id.tolist() == [1]
    # Duracion = 2h = 1/12 de dia (calendario real, no dias de trading).
    assert blocks[0].duration_days == pytest.approx(2.0 / 24.0)
    # Ordinales de apertura/cierre por trade (calendario real por fecha).
    assert blocks[0].open_ord.tolist() == [D1.toordinal()]
    assert blocks[0].close_ord.tolist() == [D1.toordinal()]
    assert blocks[1].open_ord.tolist() == [D2.toordinal()]
    assert blocks[1].close_ord.tolist() == [D2.toordinal()]
    assert blocks[1].pnl_per_lot.tolist() == [20.0]


def test_build_day_blocks_solapados_un_solo_bloque():
    # trade1 abre mientras trade0 sigue abierto -> active nunca baja a 0 -> 1 bloque.
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 14:00", D1),
            ("2024-01-01 11:00", "2024-01-01 13:00", D1),
        ],
        valle={0: {D1: -1.0}, 1: {D1: -1.0}},
        ppl=[10.0, 20.0],
    )
    blocks = build_day_blocks(s)
    assert len(blocks) == 1
    assert blocks[0].start == _ts("2024-01-01 10:00")
    assert blocks[0].end == _ts("2024-01-01 14:00")
    assert blocks[0].trade_local_id.tolist() == [0, 1]
    assert blocks[0].n_trades == 2


def test_close_then_open_mismo_timestamp_no_fusiona_ni_rompe():
    # CONVENCION CRITICA: trade0 cierra EXACTAMENTE cuando trade1 abre (12:00).
    # El sort (ts, delta) procesa close(-1) antes que open(+1): active baja a 0
    # en 12:00 -> corte flat-flat -> DOS bloques [10:00,12:00] y [12:00,14:00].
    # Si el orden fuese el contrario, active haria 1->2->1 sin tocar 0 y saldria
    # UN bloque fusionado [10:00,14:00] con ambos trades (bug silencioso: el
    # bloque mezclaria episodios que en la realidad estan separados por un
    # instante flat, y su duration_days contaria doble el solape).
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 12:00", D1),
            ("2024-01-01 12:00", "2024-01-01 14:00", D1),
        ],
        valle={0: {D1: -1.0}, 1: {D1: -1.0}},
        ppl=[10.0, 20.0],
    )
    blocks = build_day_blocks(s)
    assert len(blocks) == 2
    # El trade del close pertenece al primer bloque (ct <= bend, con bend=12:00);
    # el del open, al segundo (ot >= bstart, con bstart=12:00). Ninguno se pierde
    # ni se duplica.
    assert blocks[0].trade_local_id.tolist() == [0]
    assert blocks[1].trade_local_id.tolist() == [1]
    assert blocks[0].end == _ts("2024-01-01 12:00")
    assert blocks[1].start == _ts("2024-01-01 12:00")


def test_build_day_blocks_oos_start_filtra_bloques_y_renumera():
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 12:00", D1),
            ("2024-01-03 10:00", "2024-01-03 12:00", D3),
        ],
        valle={0: {D1: -1.0}, 1: {D3: -1.0}},
        ppl=[10.0, 20.0],
    )
    blocks = build_day_blocks(s, oos_start=_ts("2024-01-02"))
    assert len(blocks) == 1
    assert blocks[0].idx == 0  # renumerado tras el filtrado
    assert blocks[0].trade_local_id.tolist() == [1]


# ---------------------------------------------------------------------------
# block_day_arrays: semantica rel_before / rel_closed / rel_float
# ---------------------------------------------------------------------------

def test_block_day_arrays_semantica_rel_hand_computed():
    # UN bloque con 2 trades solapados que cubre 3 dias calendario:
    #   trade0: abre D1 10:00, cierra D2 12:00 (close_d=D2), valle {D1:-2, D2:-1}
    #   trade1: abre D2 09:00, cierra D3 15:00 (close_d=D3), valle {D2:-3, D3:-1.5}
    # lots_f = [0.1, 0.2], pv = 100.
    #
    # Calculo manual:
    #   pnl_trade = lots*ppl = [0.1*10, 0.2*20] = [1.0, 4.0]
    #   PnL cerrado por dia de cierre: D2 -> 1.0, D3 -> 4.0
    #     closed_cum = [1.0, 5.0] (uniq_c = [D2, D3])
    #   day_float = suma de valles*lot*pv de trades abiertos por dia:
    #     D1: -2*0.1*100                    = -20.0
    #     D2: -1*0.1*100 + -3*0.2*100       = -10 - 60 = -70.0
    #     D3: -1.5*0.2*100                  = -30.0
    #   day_opens: D1 -> 1 (trade0), D2 -> 1 (trade1), D3 -> 0
    #   all_days = {D1, D2, D3} (union de floats, closes y opens)
    #
    #   D1 (sin close): rel_before = 0 (no hay cierre previo), rel_closed = 0,
    #     rel_float = 0 + (-20) = -20, opens = 1
    #   D2 (cierra trade0): rel_before = 0 (el close de D2 aun no entra: el
    #     flotante worst del dia ocurre ANTES del close), rel_closed = 1.0,
    #     rel_float = 0 + (-70) = -70, opens = 1
    #   D3 (cierra trade1): rel_before = 1.0 (trade0 ya cerro en D2),
    #     rel_closed = 5.0, rel_float = 1.0 + (-30) = -29, opens = 0
    #   total_pnl = 5.0
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-02 12:00", D2),
            ("2024-01-02 09:00", "2024-01-03 15:00", D3),
        ],
        valle={0: {D1: -2.0, D2: -1.0}, 1: {D2: -3.0, D3: -1.5}},
        ppl=[10.0, 20.0],
    )
    (block,) = build_day_blocks(s)
    assert block.n_trades == 2  # solapados -> un solo bloque

    ba = block_day_arrays(block, lots_f=np.array([0.1, 0.2]), pv=100.0)

    assert ba.day_ord.tolist() == [D1.toordinal(), D2.toordinal(), D3.toordinal()]
    np.testing.assert_allclose(ba.rel_before, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(ba.rel_closed, [0.0, 1.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(ba.rel_float, [-20.0, -70.0, -29.0], atol=1e-12)
    assert ba.opens_day.tolist() == [1, 1, 0]
    assert ba.total_pnl == pytest.approx(5.0)


def test_block_day_arrays_dia_sin_close_hereda_rel_closed_del_before():
    # trade0 abre y cierra en D1 (+1.0 cerrado); trade1 abre D2 y cierra D3.
    # En D2 NO hay close: rel_closed(D2) = rel_before(D2) = 1.0 (trade0 ya cerro).
    #   pnl_trade = [0.1*10, 0.1*20] = [1.0, 2.0]; closes: D1->1.0, D3->2.0
    #   day_float: D1: -1*0.1*100 = -10; D2: -2*0.1*100 = -20; D3: -0.5*0.1*100 = -5
    #   D1: rel_before=0, rel_closed=1.0, rel_float=-10
    #   D2: rel_before=1.0 (close previo D1), rel_closed=1.0 (sin close propio),
    #       rel_float = 1.0 - 20 = -19
    #   D3: rel_before=1.0, rel_closed=3.0, rel_float = 1.0 - 5 = -4
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 12:00", D1),
            ("2024-01-01 11:00", "2024-01-03 15:00", D3),
        ],
        valle={0: {D1: -1.0}, 1: {D2: -2.0, D3: -0.5}},
        ppl=[10.0, 20.0],
    )
    (block,) = build_day_blocks(s)
    ba = block_day_arrays(block, lots_f=np.array([0.1, 0.1]), pv=100.0)

    assert ba.day_ord.tolist() == [D1.toordinal(), D2.toordinal(), D3.toordinal()]
    np.testing.assert_allclose(ba.rel_before, [0.0, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(ba.rel_closed, [1.0, 1.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(ba.rel_float, [-10.0, -19.0, -4.0], atol=1e-12)
    assert ba.opens_day.tolist() == [2, 0, 0]
    assert ba.total_pnl == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# empirical_gaps
# ---------------------------------------------------------------------------

def test_empirical_gaps_dias_flat_entre_bloques():
    # Bloque 0: [D1 10:00, D1 12:00]; bloque 1: [D3 12:00, D3 14:00].
    # Gap flat = D3 12:00 - D1 12:00 = exactamente 2.0 dias.
    s = _strategy(
        trades=[
            ("2024-01-01 10:00", "2024-01-01 12:00", D1),
            ("2024-01-03 12:00", "2024-01-03 14:00", D3),
        ],
        valle={0: {D1: -1.0}, 1: {D3: -1.0}},
        ppl=[10.0, 20.0],
    )
    blocks = build_day_blocks(s)
    gaps = empirical_gaps(blocks)
    np.testing.assert_allclose(gaps, [2.0], atol=1e-12)


def test_empirical_gaps_un_solo_bloque_es_cero():
    s = _strategy(
        trades=[("2024-01-01 10:00", "2024-01-01 12:00", D1)],
        valle={0: {D1: -1.0}},
        ppl=[10.0],
    )
    gaps = empirical_gaps(build_day_blocks(s))
    np.testing.assert_allclose(gaps, [0.0], atol=1e-12)
