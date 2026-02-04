"""Institution blocks for CGE models.

This module provides institution-related equation blocks including:
- Household income and expenditure
- Government budget
- Rest of world (trade balance)
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


class Household(Block):
    """Household income and expenditure block.

    Models household income from factor payments and
    expenditure on commodities.

    Income sources:
    - Factor payments (labor, capital)
    - Transfers from government
    - Transfers from abroad

    Attributes:
        name: Block name (default: "Household")
    """

    name: str = Field(default="Household", description="Block name")
    description: str = Field(
        default="Household income and expenditure", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I", "F"]  # Commodities and factors

        self.parameters = {
            "shry": ParameterSpec(
                name="shry",
                domains=("F",),
                description="Share of factor income to households",
            ),
        }

        self.variables = {
            "YH": VariableSpec(
                name="YH",
                lower=0.0,
                description="Household income",
            ),
            "WF": VariableSpec(
                name="WF",
                domains=("F",),
                lower=0.0,
                description="Factor price (wage/rental)",
            ),
            "FSUP": VariableSpec(
                name="FSUP",
                domains=("F",),
                lower=0.0,
                description="Factor supply",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the household block."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        # Create parameters
        # Factor income shares (all to households initially)
        shry_vals = np.ones((n_factors,))
        parameters["shry"] = Parameter(
            name="shry",
            value=shry_vals,
            domains=("F",),
            description="Factor income shares",
        )

        # Create variables
        yh_val = np.array([100.0])
        variables["YH"] = Variable(
            name="YH",
            value=yh_val,
            lower=0.0,
            description="Household income",
        )

        wf_vals = np.ones((n_factors,))
        variables["WF"] = Variable(
            name="WF",
            value=wf_vals,
            domains=("F",),
            lower=0.0,
            description="Factor prices",
        )

        fsup_vals = np.ones((n_factors,))
        variables["FSUP"] = Variable(
            name="FSUP",
            value=fsup_vals,
            domains=("F",),
            lower=0.0,
            description="Factor supply",
        )

        return []


class Government(Block):
    """Government budget block.

    Models government revenue (taxes) and expenditure.

    Revenue sources:
    - Production taxes
    - Import tariffs
    - Income taxes

    Expenditures:
    - Government consumption
    - Transfers to households
    - Savings

    Attributes:
        name: Block name (default: "Government")
    """

    name: str = Field(default="Government", description="Block name")
    description: str = Field(
        default="Government budget", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {
            "tau_p": ParameterSpec(
                name="tau_p",
                domains=("I",),
                description="Production tax rate",
            ),
            "tau_m": ParameterSpec(
                name="tau_m",
                domains=("I",),
                description="Import tariff rate",
            ),
        }

        self.variables = {
            "YG": VariableSpec(
                name="YG",
                lower=0.0,
                description="Government revenue",
            ),
            "XG": VariableSpec(
                name="XG",
                domains=("I",),
                lower=0.0,
                description="Government consumption",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the government block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create parameters
        # Tax rates (initialize to 0)
        tau_p_vals = np.zeros((n_comm,))
        parameters["tau_p"] = Parameter(
            name="tau_p",
            value=tau_p_vals,
            domains=("I",),
            description="Production tax rates",
        )

        tau_m_vals = np.zeros((n_comm,))
        parameters["tau_m"] = Parameter(
            name="tau_m",
            value=tau_m_vals,
            domains=("I",),
            description="Import tariff rates",
        )

        # Create variables
        yg_val = np.array([0.0])
        variables["YG"] = Variable(
            name="YG",
            value=yg_val,
            lower=0.0,
            description="Government revenue",
        )

        xg_vals = np.zeros((n_comm,))
        variables["XG"] = Variable(
            name="XG",
            value=xg_vals,
            domains=("I",),
            lower=0.0,
            description="Government consumption",
        )

        return []


class RestOfWorld(Block):
    """Rest of World (foreign sector) block.

    Models trade balance and foreign transfers.

    Attributes:
        name: Block name (default: "ROW")
    """

    name: str = Field(default="ROW", description="Block name")
    description: str = Field(
        default="Rest of world (foreign sector)", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {
            "pwm": ParameterSpec(
                name="pwm",
                domains=("I",),
                description="World price of imports (CIF)",
            ),
            "pwe": ParameterSpec(
                name="pwe",
                domains=("I",),
                description="World price of exports (FOB)",
            ),
        }

        self.variables = {
            "FSAV": VariableSpec(
                name="FSAV",
                description="Foreign savings (trade balance)",
            ),
            "QM": VariableSpec(
                name="QM",
                domains=("I",),
                lower=0.0,
                description="Import quantity",
            ),
            "QE": VariableSpec(
                name="QE",
                domains=("I",),
                lower=0.0,
                description="Export quantity",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the ROW block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create parameters
        # World prices (initialize to 1)
        pwm_vals = np.ones((n_comm,))
        parameters["pwm"] = Parameter(
            name="pwm",
            value=pwm_vals,
            domains=("I",),
            description="World import prices",
        )

        pwe_vals = np.ones((n_comm,))
        parameters["pwe"] = Parameter(
            name="pwe",
            value=pwe_vals,
            domains=("I",),
            description="World export prices",
        )

        # Create variables
        fsav_val = np.array([0.0])
        variables["FSAV"] = Variable(
            name="FSAV",
            value=fsav_val,
            description="Foreign savings",
        )

        qm_vals = np.zeros((n_comm,))
        variables["QM"] = Variable(
            name="QM",
            value=qm_vals,
            domains=("I",),
            lower=0.0,
            description="Imports",
        )

        qe_vals = np.zeros((n_comm,))
        variables["QE"] = Variable(
            name="QE",
            value=qe_vals,
            domains=("I",),
            lower=0.0,
            description="Exports",
        )

        return []
