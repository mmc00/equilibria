"""Public scenario simulation API."""

from equilibria.simulations.pep import PepSimulator
from equilibria.simulations.presets import (
    available_presets,
    export_tax,
    government_spending,
    import_price,
    import_shock,
    make_preset,
)
from equilibria.simulations.simulator import Simulator, register_adapter, run_scenarios
from equilibria.simulations.types import Scenario, Shock, ShockDefinition

__all__ = [
    "Scenario",
    "Shock",
    "ShockDefinition",
    "Simulator",
    "PepSimulator",
    "available_presets",
    "make_preset",
    "export_tax",
    "import_price",
    "import_shock",
    "government_spending",
    "register_adapter",
    "run_scenarios",
]
