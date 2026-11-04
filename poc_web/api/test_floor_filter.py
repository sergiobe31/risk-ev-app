"""Tests unitarios del post-filtro del suelo de trades (§3.4, F2b).

``apply_trade_floor`` (capa API, función pura): un outcome "pass" con
``trades < min_unique_trades`` se reclasifica como "sin resolver" (timeout);
``min_unique_trades is None`` → no-op que devuelve la entrada intacta.

Uso:  PYTHONPATH=. python -m pytest poc_web/api/test_floor_filter.py
"""
from __future__ import annotations

from src.risk.mc_portfolio import PathOutcome, aggregate_outcomes

from poc_web.api.main import apply_trade_floor


def _mk(outcome: str, trades: int, days: float = 5.0) -> PathOutcome:
    return PathOutcome(outcome=outcome, days=days, trades=trades,
                       max_dd=-0.01, terminal=100.0)


def test_noop_sin_suelo_devuelve_la_entrada_intacta():
    outs = [_mk("pass", 3), _mk("pass", 25), _mk("ruin", 40),
            _mk("timeout", 10), _mk("neither", 7)]
    assert apply_trade_floor(outs, None) is outs


def test_reclasifica_pass_bajo_el_suelo_a_sin_resolver():
    outs = [_mk("pass", 3), _mk("pass", 20), _mk("pass", 35),
            _mk("ruin", 10), _mk("neither", 5)]
    out = apply_trade_floor(outs, 20)
    assert [o.outcome for o in out] == ["timeout", "pass", "pass", "ruin",
                                        "neither"]
    # Función pura: la entrada no se muta.
    assert [o.outcome for o in outs] == ["pass", "pass", "pass", "ruin",
                                         "neither"]
    # El resto de campos del outcome reclasificado se conservan.
    assert out[0].trades == 3 and out[0].days == 5.0


def test_frontera_trades_igual_al_suelo_pasa():
    outs = [_mk("pass", 19), _mk("pass", 20)]
    out = apply_trade_floor(outs, 20)
    assert [o.outcome for o in out] == ["timeout", "pass"]


def test_sin_resolver_absorbe_los_degradados_en_la_agregacion():
    outs = [_mk("pass", 3), _mk("pass", 25), _mk("ruin", 40), _mk("neither", 7)]
    out = apply_trade_floor(outs, 20)
    rep = aggregate_outcomes(out, 0.01, "block", 1.0, len(out), None)
    assert rep.p_pass == 1 / 4
    assert rep.p_ruin == 1 / 4
    # p_sin_resolver de la API = 1 − p_pass − p_ruin (timeout + neither).
    assert (1.0 - rep.p_pass - rep.p_ruin) == 2 / 4
    # Los días/trades a target se calculan solo sobre los passes cualificados.
    assert rep.median_trades_to_target == 25.0
