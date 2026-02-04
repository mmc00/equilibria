"""Demand blocks for CGE models.

This module provides consumer demand-related equation blocks including:
- LES (Linear Expenditure System)
- Cobb-Douglas demand
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks.base import Block, ParameterSpec, VariableSpec
from equilibria.core.equations import Equation
from equilibria.core.parameters import Parameter
from equilibria.core.sets import SetManager
from equilibria.core.variables import Variable


class LESConsumer(Block):
    """Linear Expenditure System (LES) consumer demand block.

    Implements LES demand system where consumers have:
    - Subsistence consumption (minimum requirements)
    - Supernumerary consumption (discretionary spending)

    The LES demand function is:
    QD[i] = gamma[i] + (beta[i] / PA[i]) * (Y - sum_j PA[j] * gamma[j])

    Where:
    - QD[i] = demand for commodity i
    - gamma[i] = subsistence consumption
    - beta[i] = marginal budget share
    - PA[i] = price of commodity i
    - Y = total income
    - sum_j PA[j] * gamma[j] = subsistence expenditure

    Attributes:
        name: Block name (default: "LES_Consumer")
    """

    name: str = Field(default="LES_Consumer", description="Block name")
    description: str = Field(
        default="Linear Expenditure System consumer demand",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]

        self.parameters = {
            "gamma": ParameterSpec(
                name="gamma",
                domains=("I",),
                description="Subsistence consumption (minimum requirements)",
            ),
            "beta": ParameterSpec(
                name="beta",
                domains=("I",),
                description="Marginal budget share",
            ),
        }

        self.variables = {
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Commodity demand",
            ),
            "PA": VariableSpec(
                name="PA",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
            "Y": VariableSpec(
                name="Y",
                lower=0.0,
                description="Total household income",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the LES consumer block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create parameters
        # Subsistence consumption (initialize to small values)
        gamma_vals = np.full((n_comm,), 0.1)
        parameters["gamma"] = Parameter(
            name="gamma",
            value=gamma_vals,
            domains=("I",),
            description="Subsistence consumption",
        )

        # Marginal budget shares (initialize equally)
        beta_vals = np.ones((n_comm,)) / n_comm
        parameters["beta"] = Parameter(
            name="beta",
            value=beta_vals,
            domains=("I",),
            description="Marginal budget shares",
        )

        # Create variables
        qd_vals = np.ones((n_comm,))
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity demand",
        )

        pa_vals = np.ones((n_comm,))
        variables["PA"] = Variable(
            name="PA",
            value=pa_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        y_val = np.array([100.0])
        variables["Y"] = Variable(
            name="Y",
            value=y_val,
            lower=0.0,
            description="Household income",
        )

        return []


class CobbDouglasConsumer(Block):
    """Cobb-Douglas consumer demand block.

    Implements Cobb-Douglas utility with constant expenditure shares.

    The Cobb-Douglas demand function is:
    QD[i] = (alpha[i] * Y) / PA[i]

    Where:
    - QD[i] = demand for commodity i
    - alpha[i] = expenditure share (constant)
    - Y = total income
    - PA[i] = price of commodity i

    Attributes:
        name: Block name (default: "CD_Consumer")
    """

    name: str = Field(default="CD_Consumer", description="Block name")
    description: str = Field(
        default="Cobb-Douglas consumer demand", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]

        self.parameters = {
            "alpha": ParameterSpec(
                name="alpha",
                domains=("I",),
                description="Expenditure share",
            ),
        }

        self.variables = {
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Commodity demand",
            ),
            "PA": VariableSpec(
                name="PA",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
            "Y": VariableSpec(
                name="Y",
                lower=0.0,
                description="Total household income",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the Cobb-Douglas consumer block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create parameters
        # Expenditure shares (initialize equally, sum to 1)
        alpha_vals = np.ones((n_comm,)) / n_comm
        parameters["alpha"] = Parameter(
            name="alpha",
            value=alpha_vals,
            domains=("I",),
            description="Expenditure shares",
        )

        # Create variables
        qd_vals = np.ones((n_comm,))
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity demand",
        )

        pa_vals = np.ones((n_comm,))
        variables["PA"] = Variable(
            name="PA",
            value=pa_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        y_val = np.array([100.0])
        variables["Y"] = Variable(
            name="Y",
            value=y_val,
            lower=0.0,
            description="Household income",
        )

        return []
