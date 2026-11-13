"""The block of simple_open that no generic block fits (design decision)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from equilibria.blocks._levels import LevelBlock, eq
from equilibria.core.symbolic_equations import SymbolicEquation


class StdcgeInvestment(LevelBlock):
    """Investment demand: fixed shares of total saving.

    ``XV = lambda_V*(SP + SG + EPS*Sf)/PQ`` (stdcge eqXv).
    """

    name: str = Field(default="StdcgeInvestment")

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["I"]
        self.uses = ["SP", "SG", "EPS", "PQ"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        ni = len(set_manager.get("I"))
        self.param(parameters, "lambda_V", ("I",))
        self.param(parameters, "Sf", ())
        self.var(variables, "XV", ("I",), (ni,))

        def xv(m, idx):
            (i,) = idx
            return m.XV[i] == m.lambda_V[i] * (m.SP + m.SG + m.EPS * m.Sf) / m.PQ[i]

        return [eq("eq_xv", ("I",), xv)]
