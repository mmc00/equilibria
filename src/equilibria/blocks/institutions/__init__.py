"""Institution blocks for CGE models.

This module provides institution-related equation blocks including:
- Household income and expenditure
- Government budget
- Rest of world (trade balance)
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
    ) -> list[SymbolicEquation]:
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

        equations = []

        # Household income: YH = sum_f shry[f] * WF[f] * FSUP[f]
        class HouseholdIncomeEq(SymbolicEquation):
            name: str = "Household_Income"
            domains: tuple = ()  # Scalar
            description: str = "Household income from factor payments"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for household income."""
                YH = getattr(pyomo_model, "YH")
                WF = getattr(pyomo_model, "WF")
                FSUP = getattr(pyomo_model, "FSUP")
                shry = getattr(pyomo_model, "shry")

                F_set = pyomo_model.F
                income = sum(shry[f] * WF[f] * FSUP[f] for f in F_set)

                return YH == income

        equations.append(HouseholdIncomeEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.INSTITUTIONS]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for household."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        if mode == "sam":
            # Get factor payments from production block
            prod_params = data.get_block_params("CES_VA")
            if "FD0" in prod_params:
                FD0 = prod_params["FD0"]
                WF0 = np.ones(n_factors)  # Assume price = 1 in base year
                FSUP0 = FD0.sum(axis=1)  # Total factor supply
                YH0 = (WF0 * FSUP0).sum()
                shry = np.ones(n_factors)
            else:
                WF0 = np.ones(n_factors)
                FSUP0 = np.ones(n_factors)
                YH0 = n_factors
                shry = np.ones(n_factors)
        else:  # dummy mode
            WF0 = self._get_dummy_value("WF0", (n_factors,), 1.0)
            FSUP0 = self._get_dummy_value("FSUP0", (n_factors,), 1.0)
            YH0 = (WF0 * FSUP0).sum()
            shry = np.ones(n_factors)

        return {
            "WF0": WF0,
            "FSUP0": FSUP0,
            "YH0": YH0,
            "shry": shry,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "WF0" in calibrated:
            if "WF" in var_manager:
                var_manager.get("WF").value = calibrated["WF0"].copy()
        if "FSUP0" in calibrated:
            if "FSUP" in var_manager:
                var_manager.get("FSUP").value = calibrated["FSUP0"].copy()
        if "YH0" in calibrated:
            if "YH" in var_manager:
                var_manager.get("YH").value = np.array([calibrated["YH0"]])


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
    ) -> list[SymbolicEquation]:
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
        # Initialize with small positive values to avoid log(0) errors
        # These will be overwritten by calibration data
        yg_val = np.array([1.0])
        variables["YG"] = Variable(
            name="YG",
            value=yg_val,
            lower=0.0,
            description="Government revenue",
        )

        xg_vals = np.ones((n_comm,)) * 0.1
        variables["XG"] = Variable(
            name="XG",
            value=xg_vals,
            domains=("I",),
            lower=0.0,
            description="Government consumption",
        )

        equations = []

        # Government revenue: YG = sum_i tau_m[i] * PM[i] * QM[i]
        class GovernmentRevenueEq(SymbolicEquation):
            name: str = "Government_Revenue"
            domains: tuple = ()  # Scalar
            description: str = "Government tax revenue"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for government revenue."""
                YG = getattr(pyomo_model, "YG")
                tau_m = getattr(pyomo_model, "tau_m")
                PM = getattr(pyomo_model, "PM")
                QM = getattr(pyomo_model, "QM")

                I_set = pyomo_model.I
                revenue = sum(tau_m[i] * PM[i] * QM[i] for i in I_set)

                return YG == revenue

        # Government budget: YG = sum_i PA[i] * XG[i]
        class GovernmentBudgetEq(SymbolicEquation):
            name: str = "Government_Budget"
            domains: tuple = ()  # Scalar
            description: str = "Government budget balance"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for government budget."""
                YG = getattr(pyomo_model, "YG")
                PA = getattr(pyomo_model, "PA")
                XG = getattr(pyomo_model, "XG")

                I_set = pyomo_model.I
                expenditure = sum(PA[i] * XG[i] for i in I_set)

                return YG == expenditure

        equations.append(GovernmentRevenueEq())
        equations.append(GovernmentBudgetEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.INSTITUTIONS]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for government."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Get imports from trade block
            trade_params = data.get_block_params("Armington")
            if "QM0" in trade_params:
                QM0 = trade_params["QM0"]
                # Get prices - try LES blocks or default to 1.0
                les_params = data.get_block_params("LES_hrp")
                if "PA0" in les_params:
                    PA0 = les_params["PA0"]
                else:
                    PA0 = np.ones(n_comm)

                # Extract government consumption from SAM
                # Government pays to commodities (AGR, IND, AGR_2, etc.)
                XG0 = np.zeros(n_comm)  # Default to zeros
                sam_data = None
                if hasattr(data, "sam") and data.sam is not None:
                    sam_data = data.sam.data

                if sam_data is not None:
                    if "GVT" in sam_data.index:
                        gov_accounts = [
                            "AGR",
                            "IND",
                            "AGR_2",
                            "OTHIND",
                            "FOOD",
                            "SER",
                            "ADM",
                        ]
                        XG0_list = []
                        for acc in gov_accounts[:n_comm]:
                            if acc in sam_data.columns:
                                val = sam_data.loc["GVT", acc]
                                XG0_list.append(abs(float(val)) if val != 0 else 0.0)
                            else:
                                XG0_list.append(0.0)
                        XG0 = np.array(XG0_list[:n_comm])
                    # else: GVT not found in SAM
                # else: sam_data is None

                # Government expenditure
                YG0 = (PA0 * XG0).sum()

                # Calibrate tax rates (simplified - assume only tariffs)
                tau_m = np.zeros(n_comm)  # Could be calibrated from SAM tax rows
                tau_p = np.zeros(n_comm)
            else:
                XG0 = np.zeros(n_comm)
                YG0 = 0.0
                tau_m = np.zeros(n_comm)
                tau_p = np.zeros(n_comm)
        else:  # dummy mode
            XG0 = self._get_dummy_value("XG0", (n_comm,), 0.1)
            PA0 = self._get_dummy_value("PA0", (n_comm,), 1.0)
            YG0 = (PA0 * XG0).sum()
            tau_m = np.zeros(n_comm)
            tau_p = np.zeros(n_comm)

        return {
            "XG0": XG0,
            "YG0": YG0,
            "tau_m": tau_m,
            "tau_p": tau_p,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "XG0" in calibrated:
            if "XG" in var_manager:
                var_manager.get("XG").value = calibrated["XG0"].copy()
        if "YG0" in calibrated:
            if "YG" in var_manager:
                var_manager.get("YG").value = np.array([calibrated["YG0"]])


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
    ) -> list[SymbolicEquation]:
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
        # Initialize with small positive values to avoid log(0) errors
        # These will be overwritten by calibration data
        fsav_val = np.array([0.1])
        variables["FSAV"] = Variable(
            name="FSAV",
            value=fsav_val,
            lower=float("-inf"),  # FSAV can be negative (trade deficit)
            description="Foreign savings",
        )

        qm_vals = np.ones((n_comm,)) * 0.1
        variables["QM"] = Variable(
            name="QM",
            value=qm_vals,
            domains=("I",),
            lower=0.0,
            description="Imports",
        )

        qe_vals = np.ones((n_comm,)) * 0.1
        variables["QE"] = Variable(
            name="QE",
            value=qe_vals,
            domains=("I",),
            lower=0.0,
            description="Exports",
        )

        equations = []

        # Trade balance: FSAV = sum_i (pwe[i] * QE[i] - pwm[i] * QM[i])
        class TradeBalanceEq(SymbolicEquation):
            name: str = "Trade_Balance"
            domains: tuple = ()  # Scalar
            description: str = "Foreign trade balance"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for trade balance."""
                FSAV = getattr(pyomo_model, "FSAV")
                pwe = getattr(pyomo_model, "pwe")
                pwm = getattr(pyomo_model, "pwm")
                QE = getattr(pyomo_model, "QE")
                QM = getattr(pyomo_model, "QM")

                I_set = pyomo_model.I
                balance = sum(pwe[i] * QE[i] - pwm[i] * QM[i] for i in I_set)

                return FSAV == balance

        equations.append(TradeBalanceEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.INSTITUTIONS]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for rest of world."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Get trade data from trade blocks
            armington_params = data.get_block_params("Armington")
            cet_params = data.get_block_params("CET_Exports")

            if "QM0" in armington_params:
                QM0 = armington_params["QM0"]
            else:
                QM0 = np.zeros(n_comm)

            if "XE0" in cet_params:
                QE0 = cet_params["XE0"]
            else:
                QE0 = np.zeros(n_comm)

            # World prices (normalized to 1)
            pwm = np.ones(n_comm)
            pwe = np.ones(n_comm)

            # Trade balance
            FSAV0 = (pwe * QE0 - pwm * QM0).sum()

        else:  # dummy mode
            QM0 = self._get_dummy_value("QM0", (n_comm,), 0.3)
            QE0 = self._get_dummy_value("QE0", (n_comm,), 0.2)
            pwm = np.ones(n_comm)
            pwe = np.ones(n_comm)
            FSAV0 = (pwe * QE0 - pwm * QM0).sum()

        return {
            "QM0": QM0,
            "QE0": QE0,
            "pwm": pwm,
            "pwe": pwe,
            "FSAV0": FSAV0,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QM0" in calibrated:
            if "QM" in var_manager:
                var_manager.get("QM").value = calibrated["QM0"].copy()
        if "QE0" in calibrated:
            if "QE" in var_manager:
                var_manager.get("QE").value = calibrated["QE0"].copy()
        if "FSAV0" in calibrated:
            if "FSAV" in var_manager:
                var_manager.get("FSAV").value = np.array([calibrated["FSAV0"]])
