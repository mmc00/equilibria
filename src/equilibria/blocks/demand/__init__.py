"""Demand blocks for CGE models.

This module provides consumer demand-related equation blocks including:
- LES (Linear Expenditure System)
- Cobb-Douglas demand
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
    ) -> list[SymbolicEquation]:
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

        equations = []

        # LES demand equation: QD[i] = gamma[i] + (beta[i] / PA[i]) * (Y - sum_j PA[j] * gamma[j])
        class LESDemandEq(SymbolicEquation):
            name: str = "LES_Demand"
            domains: tuple = ("I",)
            description: str = "LES demand function"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for LES demand."""
                from pyomo.environ import log, summation

                i = indices[0]

                QD = getattr(pyomo_model, "QD")
                PA = getattr(pyomo_model, "PA")
                Y = getattr(pyomo_model, "Y")
                gamma = getattr(pyomo_model, "gamma")
                beta = getattr(pyomo_model, "beta")

                # Calculate subsistence expenditure
                I_set = pyomo_model.I
                subsistence_exp = sum(PA[j] * gamma[j] for j in I_set)

                # QD[i] = gamma[i] + (beta[i] / PA[i]) * (Y - subsistence_exp)
                lhs = QD[i]
                rhs = gamma[i] + (beta[i] / PA[i]) * (Y - subsistence_exp)

                return lhs == rhs

        # Budget constraint: sum_i PA[i] * QD[i] = Y
        class LESBudgetEq(SymbolicEquation):
            name: str = "LES_Budget"
            domains: tuple = ()  # Scalar
            description: str = "LES budget constraint"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for budget constraint."""
                PA = getattr(pyomo_model, "PA")
                QD = getattr(pyomo_model, "QD")
                Y = getattr(pyomo_model, "Y")

                I_set = pyomo_model.I
                total_exp = sum(PA[i] * QD[i] for i in I_set)

                return total_exp == Y

        equations.append(LESDemandEq())
        equations.append(LESBudgetEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.DEMAND]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for LES consumer."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Extract consumption from SAM (household columns)
            QD0 = data.get_matrix("I", "H").sum(axis=1)  # Sum over households

            # Get prices from trade block
            trade_params = data.get_block_params("Armington")
            if "PA0" in trade_params:
                PA0 = trade_params["PA0"]
            else:
                PA0 = np.ones(n_comm)

            # Calculate expenditure
            expenditure = QD0 * PA0
            Y0 = expenditure.sum()

            # LES parameters (simplified calibration)
            gamma = QD0 * 0.3  # 30% subsistence
            beta = expenditure / Y0

        else:  # dummy mode
            QD0 = self._get_dummy_value("QD0", (n_comm,), 1.0)
            PA0 = self._get_dummy_value("PA0", (n_comm,), 1.0)
            Y0 = (QD0 * PA0).sum()
            gamma = QD0 * 0.3
            beta = np.ones(n_comm) / n_comm

        return {
            "QD0": QD0,
            "PA0": PA0,
            "Y0": Y0,
            "gamma": gamma,
            "beta": beta,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QD0" in calibrated:
            if "QD" in var_manager:
                var_manager.get("QD").value = calibrated["QD0"].copy()
        if "Y0" in calibrated:
            if "Y" in var_manager:
                var_manager.get("Y").value = np.array([calibrated["Y0"]])


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
    ) -> list[SymbolicEquation]:
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

        equations = []

        # Cobb-Douglas demand equation: QD[i] = (alpha[i] * Y) / PA[i]
        class CDDemandEq(SymbolicEquation):
            name: str = "CD_Demand"
            domains: tuple = ("I",)
            description: str = "Cobb-Douglas demand function"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CD demand."""
                from pyomo.environ import log

                i = indices[0]

                QD = getattr(pyomo_model, "QD")
                PA = getattr(pyomo_model, "PA")
                Y = getattr(pyomo_model, "Y")
                alpha = getattr(pyomo_model, "alpha")

                # Log-linearized: log(QD[i]) = log(alpha[i]) + log(Y) - log(PA[i])
                lhs = log(QD[i])
                rhs = log(alpha[i]) + log(Y) - log(PA[i])

                return lhs == rhs

        # Budget constraint: sum_i PA[i] * QD[i] = Y
        class CDBudgetEq(SymbolicEquation):
            name: str = "CD_Budget"
            domains: tuple = ()  # Scalar
            description: str = "Cobb-Douglas budget constraint"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for budget constraint."""
                PA = getattr(pyomo_model, "PA")
                QD = getattr(pyomo_model, "QD")
                Y = getattr(pyomo_model, "Y")

                I_set = pyomo_model.I
                total_exp = sum(PA[i] * QD[i] for i in I_set)

                return total_exp == Y

        equations.append(CDDemandEq())
        equations.append(CDBudgetEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.DEMAND]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for Cobb-Douglas consumer."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Extract consumption from SAM (household columns)
            QD0 = data.get_matrix("I", "H").sum(axis=1)  # Sum over households

            # Get prices from trade block
            trade_params = data.get_block_params("Armington")
            if "PA0" in trade_params:
                PA0 = trade_params["PA0"]
            else:
                PA0 = np.ones(n_comm)

            # Calculate expenditure shares
            expenditure = QD0 * PA0
            Y0 = expenditure.sum()
            alpha = expenditure / Y0

        else:  # dummy mode
            QD0 = self._get_dummy_value("QD0", (n_comm,), 1.0)
            PA0 = self._get_dummy_value("PA0", (n_comm,), 1.0)
            Y0 = (QD0 * PA0).sum()
            alpha = np.ones(n_comm) / n_comm

        return {
            "QD0": QD0,
            "PA0": PA0,
            "Y0": Y0,
            "alpha": alpha,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QD0" in calibrated:
            if "QD" in var_manager:
                var_manager.get("QD").value = calibrated["QD0"].copy()
        if "Y0" in calibrated:
            if "Y" in var_manager:
                var_manager.get("Y").value = np.array([calibrated["Y0"]])
