"""Shared pieces of the generic level-form blocks (production, trade, demand...).

A level-form block receives its calibrated parameter values (``values``) and the
starting level of each variable it declares (``start``) by name, declares them,
and writes each equation as a rule over the live Pyomo model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

# stdcge bounds every level and price below by 1e-5 to avoid divisions by zero
LEVEL_FLOOR = 1e-5


class RuleEquation(SymbolicEquation):
    """An equation given by ``rule(pyomo_model, indices) -> lhs == rhs``."""

    rule: Callable[[Any, tuple[str, ...]], Any] = Field(exclude=True)

    def build_expression(self, pyomo_model: Any, indices: tuple[str, ...]) -> Any:
        return self.rule(pyomo_model, indices)


class LevelBlock(Block):
    """A block with calibrated ``values`` and starting levels ``start``."""

    values: Mapping[str, Any] = Field(default_factory=dict, exclude=True)
    start: Mapping[str, Any] = Field(default_factory=dict, exclude=True)

    def value(self, name: str) -> np.ndarray:
        if name not in self.values:
            raise ValueError(f"block {self.name!r} needs the value of {name!r}")
        return np.asarray(self.values[name], dtype=float)

    def param(
        self,
        parameters: dict[str, Parameter],
        name: str,
        domains: tuple[str, ...],
        *,
        mutable: bool = False,
    ) -> None:
        parameters[name] = Parameter(
            name=name, value=self.value(name), domains=domains, mutable=mutable
        )

    def var(
        self,
        variables: dict[str, Variable],
        name: str,
        domains: tuple[str, ...],
        shape: tuple[int, ...],
        *,
        lower: float = LEVEL_FLOOR,
        domain: str = "Reals",
    ) -> None:
        init = np.asarray(self.start.get(name, 1.0), dtype=float)
        value = np.broadcast_to(init, shape).copy() if shape else np.atleast_1d(init)
        variables[name] = Variable(
            name=name,
            value=value,
            domains=domains,
            domain=domain,
            lower=lower,
            upper=float("inf"),
        )


def eq(name: str, domains: tuple[str, ...], rule: Callable[..., Any]) -> RuleEquation:
    return RuleEquation(name=name, domains=domains, rule=rule)
