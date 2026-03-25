"""Model adapters used by the generic simulation API."""

from equilibria.simulations.adapters.base import BaseModelAdapter
from equilibria.simulations.adapters.gtap import GTAPAdapter
from equilibria.simulations.adapters.icio import ICIOAdapter
from equilibria.simulations.adapters.ieem import IEEMAdapter
from equilibria.simulations.adapters.mapping import MappingAdapter
from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.adapters.pep_co2 import PepCO2Adapter

__all__ = [
    "BaseModelAdapter",
    "MappingAdapter",
    "PepAdapter",
    "PepCO2Adapter",
    "IEEMAdapter",
    "GTAPAdapter",
    "ICIOAdapter",
]
