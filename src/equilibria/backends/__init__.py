"""Backends module for equilibria CGE framework.

Provides solver backends for translating and solving CGE models.
"""

from equilibria.backends.base import Backend, Solution

try:
    from equilibria.backends.pyomo_backend import PyomoBackend

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

__all__ = [
    "Backend",
    "Solution",
]

if PYOMO_AVAILABLE:
    __all__.append("PyomoBackend")
