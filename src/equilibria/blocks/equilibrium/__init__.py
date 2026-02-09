"""Equilibrium blocks for CGE models.

This module provides market equilibrium-related equation blocks:
- Market clearing conditions
- Price normalization
"""

from __future__ import annotations

import typing
from typing import Any, TYPE_CHECKING

import numpy as np
from pydantic import Field

from equilibria.blocks.base import Block, ParameterSpec, VariableSpec
from equilibria.core.calibration_phase import CalibrationPhase
from equilibria.core.symbolic_equations import (
    SymbolicEquation,
)
from equilibria.core.parameters import Parameter
from equilibria.core.sets import SetManager
from equilibria.core.variables import Variable

if TYPE_CHECKING:
    from equilibria.core.calibration_data import CalibrationData


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
    ) -> list[SymbolicEquation]:
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

        equations = []

        # Market clearing: QS[i] = QD[i]
        class MarketClearingEq(SymbolicEquation):
            name: str = "Market_Clearing"
            domains: tuple = ("I",)
            description: str = "Commodity market clearing"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for market clearing."""
                QS = getattr(pyomo_model, "QS")
                QD = getattr(pyomo_model, "QD")

                i = indices[0]
                return QS[i] == QD[i]

        equations.append(MarketClearingEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for market clearing."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Get data from other blocks
            trade_params = data.get_block_params("Armington")
            demand_params = data.get_block_params("LES_Consumer")

            if "QA0" in trade_params:
                QS0 = trade_params["QA0"]  # Supply
            else:
                QS0 = np.ones(n_comm)

            if "QD0" in demand_params:
                QD0 = demand_params["QD0"]  # Demand
            else:
                QD0 = QS0  # Balanced in base year

            P0 = np.ones(n_comm)  # Normalized prices

        else:  # dummy mode
            QS0 = self._get_dummy_value("QS0", (n_comm,), 1.0)
            QD0 = QS0  # Balanced
            P0 = np.ones(n_comm)

        return {
            "QS0": QS0,
            "QD0": QD0,
            "P0": P0,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QS0" in calibrated:
            if "QS" in var_manager:
                var_manager.get("QS").value = calibrated["QS0"].copy()
        if "QD0" in calibrated:
            if "QD" in var_manager:
                var_manager.get("QD").value = calibrated["QD0"].copy()
        if "P0" in calibrated:
            if "P" in var_manager:
                var_manager.get("P").value = calibrated["P0"].copy()


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
    ) -> list[SymbolicEquation]:
        """Set up the price normalization block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Set numeraire if not specified
        if not self.numeraire:
            self.numeraire = list(commodities)[0]

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
        numeraire_idx = list(commodities).index(self.numeraire)
        variables["P"].value[numeraire_idx] = 1.0

        equations = []

        # Price normalization: P[numeraire] = 1
        # Capture numeraire in closure to avoid scope issues
        numeraire_value = self.numeraire

        class PriceNormEq(SymbolicEquation):
            name: str = "Price_Normalization"
            domains: tuple = ()  # Scalar
            description: str = "Numeraire price normalization"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for price normalization."""
                P = getattr(pyomo_model, "P")

                # Fix the numeraire price to 1
                return P[numeraire_value] == 1.0

        equations.append(PriceNormEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for price normalization."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Get numeraire
        if not self.numeraire:
            self.numeraire = list(commodities)[0]

        # Prices are normalized to 1
        P0 = np.ones(n_comm)

        return {
            "P0": P0,
            "numeraire": self.numeraire,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "P0" in calibrated:
            if "P" in var_manager:
                var_manager.get("P").value = calibrated["P0"].copy()


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
    ) -> list[SymbolicEquation]:
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

        # Note: FD (factor demand) is defined by CESValueAdded block with 2D indexing (F x J)
        # We don't redefine it here to avoid dimension mismatch

        wf_vals = np.ones((n_factors,))
        variables["WF"] = Variable(
            name="WF",
            value=wf_vals,
            domains=("F",),
            lower=0.0,
            description="Factor prices",
        )

        equations = []

        # Factor market clearing: FSUP[f] = sum_j FD[f,j]
        class FactorMarketClearingEq(SymbolicEquation):
            name: str = "Factor_Market_Clearing"
            domains: tuple = ("F",)
            description: str = "Factor market clearing condition"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for factor market clearing."""
                FSUP = getattr(pyomo_model, "FSUP")
                FD = getattr(pyomo_model, "FD")

                f = indices[0]
                # Sum factor demand over all sectors
                J_set = pyomo_model.J
                total_demand = sum(FD[f, j] for j in J_set)

                return FSUP[f] == total_demand

        equations.append(FactorMarketClearingEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for factor market clearing."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        if mode == "sam":
            # Get data from household block
            hh_params = data.get_block_params("Household")

            if "FSUP0" in hh_params:
                FSUP0 = hh_params["FSUP0"]
            else:
                FSUP0 = np.ones(n_factors)

            WF0 = np.ones(n_factors)

        else:  # dummy mode
            FSUP0 = self._get_dummy_value("FSUP0", (n_factors,), 1.0)
            WF0 = np.ones(n_factors)

        # Note: FD0 is NOT created here - it comes from CESValueAdded block
        # FactorMarketClearing only uses FD, it doesn't define the base year values
        return {
            "FSUP0": FSUP0,
            "WF0": WF0,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "FSUP0" in calibrated:
            if "FSUP" in var_manager:
                var_manager.get("FSUP").value = calibrated["FSUP0"].copy()
        if "FD0" in calibrated:
            if "FD" in var_manager:
                var_manager.get("FD").value = calibrated["FD0"].copy()
        if "WF0" in calibrated:
            if "WF" in var_manager:
                var_manager.get("WF").value = calibrated["WF0"].copy()
