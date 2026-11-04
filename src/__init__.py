"""Core compartido del flujo SQX + riesgo.

Este paquete contiene únicamente los módulos operativos que usan tanto `sqx/`
como `risk_app/`:

- `src.broker_config` — specs de broker y reglas de Axi Select.
- `src.risk` — utilidades de gestión de riesgo, Monte-Carlo y sizing.
"""

from src.broker_config import load_axi_config, load_broker_config

__all__ = [
    "load_axi_config",
    "load_broker_config",
]
