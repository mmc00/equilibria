"""Public scenario simulation API."""

from equilibria.simulations.pep import PepSimulator
from equilibria.simulations.presets import (
    export_tax,
    government_spending,
    import_price,
    import_shock,
)
from equilibria.simulations.simulator import Simulator, register_adapter, run_scenarios
from equilibria.simulations.types import Scenario, Shock, ShockDefinition

__all__ = [
    "Scenario",
    "Shock",
    "ShockDefinition",
    "Simulator",
    "PepSimulator",
    "export_tax",
    "import_price",
    "import_shock",
    "government_spending",
    "register_adapter",
    "run_scenarios",
]
