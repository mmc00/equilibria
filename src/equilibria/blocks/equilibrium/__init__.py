"""Equilibrium blocks for CGE models.

This module provides market equilibrium-related equation blocks:
- Market clearing conditions
- Price normalization
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


class MarketClearing(Block):
    """Market clearing condition block.

    Ensures supply equals demand for all commodities:
    QS[i] = QD[i] for all commodities i

    Where:
    - QS[i] = total supply of commodity i (domestic + imports)
    - QD[i] = total demand for commodity i (intermediate + final)

    Attributes:
        name: Block name (default: "MarketClearing")
    """

    name: str = Field(default="MarketClearing", description="Block name")
    description: str = Field(
        default="Market clearing conditions", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {}

        self.variables = {
            "QS": VariableSpec(
                name="QS",
                domains=("I",),
                lower=0.0,
                description="Total commodity supply",
            ),
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Total commodity demand",
            ),
            "P": VariableSpec(
                name="P",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the market clearing block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create variables
        qs_vals = np.ones((n_comm,))
        variables["QS"] = Variable(
            name="QS",
            value=qs_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity supply",
        )

        qd_vals = np.ones((n_comm,))
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity demand",
        )

        p_vals = np.ones((n_comm,))
        variables["P"] = Variable(
            name="P",
            value=p_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        return []


class PriceNormalization(Block):
    """Price normalization block.

    Sets the numeraire price to fix the price level:
    P[numeraire] = 1

    Attributes:
        name: Block name (default: "PriceNorm")
        numeraire: Name of numeraire commodity (default: first commodity)
    """

    name: str = Field(default="PriceNorm", description="Block name")
    description: str = Field(
        default="Price normalization (numeraire)", description="Block description"
    )
    numeraire: str = Field(default="", description="Numeraire commodity name")

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {}

        self.variables = {
            "P": VariableSpec(
                name="P",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the price normalization block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Set numeraire if not specified
        if not self.numeraire:
            self.numeraire = list(commodities.iter_elements())[0]

        # Create variables
        p_vals = np.ones((n_comm,))
        variables["P"] = Variable(
            name="P",
            value=p_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        # Fix numeraire price to 1
        numeraire_idx = list(commodities.iter_elements()).index(self.numeraire)
        variables["P"].value[numeraire_idx] = 1.0

        return []


class FactorMarketClearing(Block):
    """Factor market clearing block.

    Ensures factor supply equals factor demand:
    FSUP[f] = FD[f] for all factors f

    Where:
    - FSUP[f] = supply of factor f
    - FD[f] = demand for factor f

    Attributes:
        name: Block name (default: "FactorMarket")
    """

    name: str = Field(default="FactorMarket", description="Block name")
    description: str = Field(
        default="Factor market clearing", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["F"]  # Factors

        self.parameters = {}

        self.variables = {
            "FSUP": VariableSpec(
                name="FSUP",
                domains=("F",),
                lower=0.0,
                description="Factor supply",
            ),
            "FD": VariableSpec(
                name="FD",
                domains=("F",),
                lower=0.0,
                description="Factor demand",
            ),
            "WF": VariableSpec(
                name="WF",
                domains=("F",),
                lower=0.0,
                description="Factor price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the factor market clearing block."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        # Create variables
        fsup_vals = np.ones((n_factors,))
        variables["FSUP"] = Variable(
            name="FSUP",
            value=fsup_vals,
            domains=("F",),
            lower=0.0,
            description="Factor supply",
        )

        fd_vals = np.ones((n_factors,))
        variables["FD"] = Variable(
            name="FD",
            value=fd_vals,
            domains=("F",),
            lower=0.0,
            description="Factor demand",
        )

        wf_vals = np.ones((n_factors,))
        variables["WF"] = Variable(
            name="WF",
            value=wf_vals,
            domains=("F",),
            lower=0.0,
            description="Factor prices",
        )

        return []
