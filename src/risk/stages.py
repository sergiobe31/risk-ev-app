"""Umbrales de barrera por etapa AXI Select.

Wrapper fino sobre :class:`src.broker_config.AXIPhase` (NO reimplementa la
tabla de etapas). Convierte los campos de la etapa en las lineas de barrera
que el motor de ruin necesita:

    baseline   = Allocation Account inicial = allocation_at(equity)
    target_line= baseline * (1 + profit_target_pct/100)
    ruin_line  = baseline * (1 + max_loss_pct/100)        # estatico, NO trailing

Regla AXI verificada (docs/axi_select_program_rules.md §Respuestas Perplexity,
citando DevRev oficial): el max loss es drawdown ESTATICO desde la baseline de
Allocation inicial, evaluado en MTM tiempo real; SIN daily-DD ni consistency.
Pro M = -10%, resto = -7%.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.broker_config import AXIPhase, load_axi_config


@dataclass(frozen=True)
class StageBarriers:
    """Lineas de barrera + filtros de una etapa, ya escaladas a USD."""

    phase_name: str
    display_name: str
    baseline: float  # Allocation Account inicial (USD)
    target_line: float  # equity para APROBAR (USD)
    ruin_line: float  # equity de RUIN (USD, estatico)
    profit_target_pct: float  # %
    max_loss_pct: float  # % (negativo)
    min_unique_trades: int | None
    min_days: int | None
    leverage: int

    def target_return(self) -> float:
        """Fraccion de retorno hasta target (p.ej. 0.07)."""
        return self.profit_target_pct / 100.0

    def ruin_return(self) -> float:
        """Fraccion de perdida hasta ruin (negativo, p.ej. -0.07)."""
        return self.max_loss_pct / 100.0


def barriers_from_phase(
    phase: AXIPhase | str,
    equity_usd: float,
    *,
    ruin_pct: float | None = None,
    target_pct: float | None = None,
    min_days: int | None = None,
    min_trades: int | None = None,
) -> StageBarriers:
    """Construye las barreras para una etapa y un equity de trading.

    Args:
        phase: instancia :class:`AXIPhase` o nombre. Si es str, carga via
            :func:`load_axi_config`.
        equity_usd: equity de la cuenta Axi Select (NO la Allocation).
        ruin_pct: **override** del % de max loss (negativo, p.ej. -7). Default =
            el de la etapa.
        target_pct: **override** del % de profit target (positivo, p.ej. 7).
        min_days: **override** del minimo de dias (puerta de pass).
        min_trades: **override** del minimo de trades.

    Pro M tiene ``profit_target_pct``/``min_days``/``min_unique_trades`` nulos
    (solo rige max_loss -10%): se normalizan a ``+inf`` para que nunca filtren.
    """
    if isinstance(phase, str):
        phase = load_axi_config().get_phase(phase)

    baseline = phase.allocation_at(equity_usd)
    profit_target = target_pct if target_pct is not None else (
        phase.profit_target_pct if phase.profit_target_pct is not None else float("inf")
    )
    max_loss = ruin_pct if ruin_pct is not None else phase.max_loss_pct
    md = min_days if min_days is not None else phase.min_days
    mt = min_trades if min_trades is not None else phase.min_unique_trades

    return StageBarriers(
        phase_name=phase.name,
        display_name=phase.display_name,
        baseline=float(baseline),
        target_line=float(baseline * (1 + profit_target / 100.0)),
        ruin_line=float(baseline * (1 + max_loss / 100.0)),
        profit_target_pct=float(profit_target),
        max_loss_pct=float(max_loss),
        min_unique_trades=mt,
        min_days=md,
        leverage=int(phase.leverage),
    )
