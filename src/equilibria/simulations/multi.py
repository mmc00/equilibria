"""Model-specific convenience wrappers for non-PEP simulators."""

from __future__ import annotations

from typing import Any

from equilibria.simulations.simulator import Simulator


class IEEMSimulator(Simulator):
    """Convenience wrapper with model fixed to ``ieem``."""

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="ieem", **model_options)


class GTAPSimulator(Simulator):
    """Convenience wrapper with model fixed to ``gtap``."""

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="gtap", **model_options)


class ICIOSimulator(Simulator):
    """Convenience wrapper with model fixed to ``icio``."""

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="icio", **model_options)
