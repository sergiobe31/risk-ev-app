"""Tests del parser del bloque MM ATRRiskBasedSizing (.xml y .sqx)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from sqx.risk_mm_parse import parse_strategy_mm

# .xml (formato estrategia exportada: key="#Risk#" value=...)
_XML_FMT = """<StrategyFile><Strategy>
  <Backtest><Symbol>XAUUSD_axi</Symbol><Timeframe>H4</Timeframe></Backtest>
  <MoneyManagement type="ATRRiskBasedSizing"><Method><Params>
    <Param key="#Risk#" value="1.5">1.5</Param>
    <Param key="#ATRPeriod#" value="14">14</Param>
    <Param key="#ATRMult#" value="2.5">2.5</Param>
    <Param key="#Decimals#" value="2">2</Param>
    <Param key="#LotsIfNoMM#" value="0.01">0.01</Param>
    <Param key="#MaxLots#" value="3.0">3.0</Param>
  </Params></Method></MoneyManagement>
</Strategy></StrategyFile>"""

# .sqx -> lastSettings.xml (varios Method; el activo es use="true")
_LASTSETTINGS = """<root>
  <Methods>
    <Method type="RiskFixedBalancePct" use="false"><Params>
      <Param key="Risk" className="x">5</Param>
    </Params></Method>
    <Method type="ATRRiskBasedSizing" use="true"><Params>
      <Param key="Risk" className="x">2</Param>
      <Param key="ATRPeriod" className="x">20</Param>
      <Param key="ATRMult" className="x">2</Param>
      <Param key="Decimals" className="x">2</Param>
      <Param key="LotsIfNoMM" className="x">0.01</Param>
      <Param key="MaxLots" className="x">5</Param>
    </Params></Method>
  </Methods>
  <Chart symbol="XAUUSD_axi" timeframe="H4"/>
</root>"""


def test_parse_xml_format(tmp_path):
    p = tmp_path / "s.xml"
    p.write_text(_XML_FMT, encoding="utf-8")
    m = parse_strategy_mm(p)
    assert m.mm_type == "ATRRiskBasedSizing"
    assert m.symbol == "XAUUSD_axi" and m.timeframe == "H4"
    assert m.risk == 1.5 and m.atr_period == 14 and m.atr_mult == 2.5
    assert m.decimals == 2 and m.lots_if_no_mm == 0.01 and m.max_lots == 3.0


def test_parse_sqx_elige_metodo_activo(tmp_path):
    # empaquetar lastSettings.xml en un .sqx (ZIP)
    sqx = tmp_path / "s.sqx"
    with zipfile.ZipFile(sqx, "w") as z:
        z.writestr("lastSettings.xml", _LASTSETTINGS)
    m = parse_strategy_mm(sqx)
    # debe elegir ATRRiskBasedSizing (use=true), NO RiskFixedBalancePct
    assert m.mm_type == "ATRRiskBasedSizing"
    assert m.risk == 2.0  # no 5.0 (del metodo inactivo)
    assert m.atr_period == 20 and m.atr_mult == 2
    assert m.lots_if_no_mm == 0.01 and m.max_lots == 5.0
    assert m.symbol == "XAUUSD_axi" and m.timeframe == "H4"


_DESKTOP = Path("/nonexistent/private-data")
_REAL_SQX = _DESKTOP / "riesgo1ATRvs2ATR/sqx_5_100_2Strategy 2.29.103.sqx"


@pytest.mark.skipif(not _REAL_SQX.exists(), reason="sin .sqx de ejemplo en Desktop")
def test_real_sqx_parses_atr_params():
    m = parse_strategy_mm(_REAL_SQX)
    assert m.mm_type == "ATRRiskBasedSizing"
    assert m.symbol == "XAUUSD_axi" and m.timeframe == "H4"
    assert m.atr_period == 20 and m.atr_mult == 2.0
    assert m.lots_if_no_mm == 0.01 and m.max_lots == 5.0


# ---------------------------------------------------------------------------
# Fix 2026-07-18: strategy_Portfolio.xml manda, symbol de Results/, IC, veto portfolio
# ---------------------------------------------------------------------------

# strategy_Portfolio.xml con la MM REAL (risk=3) + lastSettings con la "vista"
# (Chart de OTRO simbolo + MM distinta, risk=2). El parser debe preferir la
# definicion de la estrategia y el resultKey de Results/.
_STRATEGY_PORTFOLIO = """<StrategyFile Version="3.9.133"><Strategy name="">
  <MoneyManagement type="ATRRiskBasedSizing"><Params>
    <Param key="Risk" className="ATRRiskBasedSizing">3</Param>
    <Param key="ATRPeriod" className="ATRRiskBasedSizing">20</Param>
    <Param key="ATRMult" className="ATRRiskBasedSizing">2</Param>
    <Param key="Decimals" className="ATRRiskBasedSizing">2</Param>
    <Param key="LotsIfNoMM" className="ATRRiskBasedSizing">0.01</Param>
    <Param key="MaxLots" className="ATRRiskBasedSizing">5</Param>
  </Params></MoneyManagement>
</Strategy></StrategyFile>"""

_LASTSETTINGS_VISTA_ENGANOSA = """<root>
  <Methods>
    <Method type="ATRRiskBasedSizing" use="true"><Params>
      <Param key="Risk" className="x">2</Param>
      <Param key="ATRPeriod" className="x">20</Param>
      <Param key="ATRMult" className="x">2</Param>
    </Params></Method>
  </Methods>
  <InitialCapital>5000</InitialCapital>
  <Chart symbol="XAUUSD_axi" timeframe="H4"/>
</root>"""


def _make_sqx(tmp_path, members: dict[str, str]) -> Path:
    sqx = tmp_path / "s.sqx"
    with zipfile.ZipFile(sqx, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return sqx


def test_mm_prioriza_lastsettings_run_config(tmp_path):
    """La MM del run (lastSettings, risk=2) manda sobre la definicion del databank
    (strategy_Portfolio.xml, risk=3 — puede estar vieja: caso FixedSize del oro)."""
    sqx = _make_sqx(tmp_path, {
        "strategy_Portfolio.xml": _STRATEGY_PORTFOLIO,
        "lastSettings.xml": _LASTSETTINGS_VISTA_ENGANOSA,
    })
    m = parse_strategy_mm(sqx)
    assert m.risk == 2.0
    assert m.mm_type == "ATRRiskBasedSizing"


def test_symbol_tf_desde_results_path_no_del_chart(tmp_path):
    """El Chart de lastSettings (la vista) decia XAUUSD pero el run fue DEUIDXEUR."""
    sqx = _make_sqx(tmp_path, {
        "strategy_Portfolio.xml": _STRATEGY_PORTFOLIO,
        "lastSettings.xml": _LASTSETTINGS_VISTA_ENGANOSA,
        "Results/Main: DEUIDXEUR_axi_LOM_H4/dailyEquity.bin": "PK",
    })
    m = parse_strategy_mm(sqx)
    assert m.symbol == "DEUIDXEUR_axi" and m.timeframe == "H4"


def test_initial_capital_elemento_hermano(tmp_path):
    sqx = _make_sqx(tmp_path, {"lastSettings.xml": _LASTSETTINGS_VISTA_ENGANOSA})
    m = parse_strategy_mm(sqx)
    assert m.initial_capital == 5000.0


def test_portfolio_sqx_rechazado_con_lista_estrategias(tmp_path):
    sqx = _make_sqx(tmp_path, {
        "strategy_Portfolio.xml": _STRATEGY_PORTFOLIO,
        "Results/Portfolio/dailyEquity.bin": "PK",
        "Results/Strategy 1.52.100:  XAUUSD_axi_LOM_H4/dailyEquity.bin": "PK",
        "Results/Strategy 3.57.150:  DEUIDXEUR_axi_LOM_H4/dailyEquity.bin": "PK",
    })
    with pytest.raises(ValueError, match="PORTFOLIO") as exc:
        parse_strategy_mm(sqx)
    assert "Strategy 3.57.150" in str(exc.value)


def test_portfolio_merged_result_rechazado(tmp_path):
    merged = _STRATEGY_PORTFOLIO.replace(
        '<StrategyFile Version="3.9.133">',
        '<StrategyFile Version="3.9.133" mergedResult="true">',
    )
    sqx = _make_sqx(tmp_path, {"strategy_Portfolio.xml": merged})
    with pytest.raises(ValueError, match="PORTFOLIO"):
        parse_strategy_mm(sqx)
