"""Motor de riesgo de ruina para AXI Select (consume exports de SQX).

Subpackage AISLADO de la metodologia cascade. NO importa ``src.methodology``
ni ``sqx``. Sus unicas dependencias son ``src.broker_config`` (umbrales de
etapa AXI + specs de coste), ``vectorbtpro`` (primitivas de MC), ``numpy`` y
``pandas``. Ver ``tests/test_risk_no_sqx_import.py`` para el guardián.

Flujo: CSV orderstocsv -> :mod:`trades` (valida + extrae R-multiplos)
-> :mod:`mm` (sweep de %riesgo ``f`` con re-floor) -> :mod:`mc` (MC dual)
-> :mod:`ruin` (barrera doble de primer paso) -> :mod:`stages` (umbrales AXI).
"""
from __future__ import annotations

from src.risk.trades import TradeList, load_trade_list, sniff_decimal
from src.risk.stages import StageBarriers, barriers_from_phase

__all__ = [
    "TradeList",
    "load_trade_list",
    "sniff_decimal",
    "StageBarriers",
    "barriers_from_phase",
]