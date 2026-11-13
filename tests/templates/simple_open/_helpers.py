"""Build one simple_open block alone at the calibrated base, and measure residuals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pyomo.environ as pyo

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.base import Block
from equilibria.core.sets import Set as ESet
from equilibria.core.variables import Variable
from equilibria.model import Model


def _domains_of(name: str) -> tuple[str, ...]:
    """Domains of every stdcge variable (see the plan's names table)."""
    return {
        "VA": ("J",), "FD": ("F", "J"), "PVA": ("J",), "WF": ("F",),
        "XI": ("I", "J"), "Z": ("J",), "PZ": ("J",),
        "E": ("I",), "D": ("I",), "PD": ("I",),
        "Q": ("I",), "M": ("I",), "PQ": ("I",),
        "PE": ("I",), "PM": ("I",), "EPS": (),
        "XP": ("I",), "SP": (), "UU": (),
        "TD": (), "TZ": ("J",), "TM": ("I",), "XG": ("I",), "SG": (),
        "XV": ("I",), "WALRAS": (),
    }[name]  # fmt: skip


class Given(Block):
    """Test-only block: declares variables fixed at given values."""

    name: str = "Given"
    given: dict[str, Any] = {}

    def setup(self, set_manager, parameters, variables):  # noqa: ANN001, ANN201
        for var, value in self.given.items():
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            variables[var] = Variable(
                name=var,
                value=arr,
                domains=_domains_of(var),
                domain="Reals",
                lower=arr if arr.ndim else float(arr),
                upper=arr if arr.ndim else float(arr),
            )
        return []


def model_sets(calibration: Any) -> list[ESet]:
    return [
        ESet(name="I", elements=calibration.goods),
        ESet(name="J", elements=calibration.goods),
        ESet(name="F", elements=calibration.factors),
    ]


def block_at_base(block: Block, calibration: Any) -> Any:
    """Pyomo model of ``block`` alone, every other variable fixed at base."""
    model = Model(name=f"{block.name}_at_base")
    model.add_sets(model_sets(calibration))
    model.add_block(block)
    others = {
        k: v for k, v in calibration.base.items() if k not in model.variable_manager
    }
    model.add_block(Given(given=others))
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model


def max_residual(pm: Any, equation_names: Iterable[str]) -> float:
    """Largest |body - bound| over the named equations, at current values."""
    worst = 0.0
    for name in equation_names:
        con = pm.find_component(f"{name}_con")
        assert con is not None, f"no equation {name}"
        for c in con.values():
            worst = max(worst, abs(pyo.value(c.body) - pyo.value(c.upper)))
    return worst
