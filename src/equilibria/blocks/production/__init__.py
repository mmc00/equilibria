"""Production blocks for CGE models with equations.

This module provides production-related equation blocks including:
- CES value-added aggregation
- Leontief intermediate inputs
- CET transformation
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


class CESValueAdded(Block):
    """CES value-added production block."""

    sigma: float = Field(default=0.8, gt=0)
    name: str = Field(default="CES_VA")
    description: str = Field(default="CES value-added production")

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["J", "F"]
        self.parameters = {
            "sigma_VA": ParameterSpec(
                name="sigma_VA", domains=("J",), default=self.sigma
            ),
            "beta_VA": ParameterSpec(name="beta_VA", domains=("F", "J")),
            "B_VA": ParameterSpec(name="B_VA", domains=("J",)),
        }
        self.variables = {
            "VA": VariableSpec(name="VA", domains=("J",), lower=0.0),
            "FD": VariableSpec(name="FD", domains=("F", "J"), lower=0.0),
            "PVA": VariableSpec(name="PVA", domains=("J",), lower=0.0),
            "WF": VariableSpec(name="WF", domains=("F",), lower=0.0),
        }

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        sectors = set_manager.get("J")
        factors = set_manager.get("F")
        n_sectors, n_factors = len(sectors), len(factors)

        # Initialize parameters and variables
        parameters["sigma_VA"] = Parameter(
            name="sigma_VA", value=np.full((n_sectors,), self.sigma), domains=("J",)
        )
        parameters["beta_VA"] = Parameter(
            name="beta_VA",
            value=np.ones((n_factors, n_sectors)) / n_factors,
            domains=("F", "J"),
        )
        parameters["B_VA"] = Parameter(
            name="B_VA", value=np.ones((n_sectors,)), domains=("J",)
        )

        variables["VA"] = Variable(
            name="VA", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )
        variables["FD"] = Variable(
            name="FD",
            value=np.ones((n_factors, n_sectors)),
            domains=("F", "J"),
            lower=0.0,
        )
        variables["PVA"] = Variable(
            name="PVA", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )
        variables["WF"] = Variable(
            name="WF", value=np.ones((n_factors,)), domains=("F",), lower=0.0
        )

        equations = []

        # Simple equation classes that define constraints
        class CESAggregationEq(SymbolicEquation):
            name: str = "CES_Aggregation"
            domains: tuple = ("J",)
            description: str = "CES aggregation: VA = B * (sum beta * FD^rho)^(1/rho)"

            def define(self, set_manager, variables, parameters):
                """Legacy closure-based definition."""
                constraints = {}
                sectors = set_manager.get("J")
                factors = set_manager.get("F")

                for j_idx, j in enumerate(sectors):
                    sigma = parameters["sigma_VA"].value[j_idx]
                    beta = parameters["beta_VA"].value[:, j_idx]
                    B = parameters["B_VA"].value[j_idx]
                    VA = variables["VA"].value[j_idx]
                    FD = variables["FD"].value[:, j_idx]

                    rho = (sigma - 1) / sigma if sigma != 1 else 0.0

                    if abs(rho) > 1e-10:
                        ces_sum = sum(
                            beta[f_idx] * (FD[f_idx] ** rho)
                            for f_idx in range(len(factors))
                        )
                        ces_value = B * (ces_sum ** (1.0 / rho))
                    else:
                        # Cobb-Douglas
                        ces_value = B * np.prod(
                            [FD[f_idx] ** beta[f_idx] for f_idx in range(len(factors))]
                        )

                    constraints[(j,)] = lambda va=VA, ces=ces_value: va - ces

                return constraints

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CES aggregation."""
                from pyomo.environ import log, exp, summation

                j = indices[0]

                # Get Pyomo variables and parameters
                VA = getattr(pyomo_model, "VA")
                FD = getattr(pyomo_model, "FD")
                B_VA = getattr(pyomo_model, "B_VA")
                beta_VA = getattr(pyomo_model, "beta_VA")
                sigma_VA = getattr(pyomo_model, "sigma_VA")

                # For simplicity, use log-linearized Cobb-Douglas form
                # log(VA[j]) = log(B_VA[j]) + sum_f(beta_VA[f,j] * log(FD[f,j]))
                # This is valid when sigma = 1 (Cobb-Douglas)

                # Build the expression
                lhs = log(VA[j])
                rhs = log(B_VA[j])

                # Sum over factors
                F_set = pyomo_model.F
                for f in F_set:
                    rhs = rhs + beta_VA[f, j] * log(FD[f, j])

                return lhs == rhs

        class CESFOCEq(SymbolicEquation):
            name: str = "CES_FOC"
            domains: tuple = ("F", "J")
            description: str = "FOC: WF = PVA * dVA/dFD"

            def define(self, set_manager, variables, parameters):
                """Legacy closure-based definition."""
                constraints = {}
                sectors = set_manager.get("J")
                factors = set_manager.get("F")

                for j_idx, j in enumerate(sectors):
                    sigma = parameters["sigma_VA"].value[j_idx]
                    beta = parameters["beta_VA"].value[:, j_idx]
                    PVA = variables["PVA"].value[j_idx]
                    VA = variables["VA"].value[j_idx]
                    FD = variables["FD"].value[:, j_idx]
                    rho = (sigma - 1) / sigma if sigma != 1 else 0.0

                    for f_idx, f in enumerate(factors):
                        WF = variables["WF"].value[f_idx]
                        if FD[f_idx] > 1e-10:
                            mp = beta[f_idx] * (VA / FD[f_idx]) ** (1 - rho)
                        else:
                            mp = 0.0
                        constraints[(f, j)] = (
                            lambda wf=WF, pva=PVA, mp=mp: wf - pva * mp
                        )

                return constraints

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CES FOC."""
                from pyomo.environ import log

                f, j = indices

                # Get Pyomo variables and parameters
                WF = getattr(pyomo_model, "WF")
                PVA = getattr(pyomo_model, "PVA")
                VA = getattr(pyomo_model, "VA")
                FD = getattr(pyomo_model, "FD")
                beta_VA = getattr(pyomo_model, "beta_VA")
                sigma_VA = getattr(pyomo_model, "sigma_VA")

                # For Cobb-Douglas (sigma = 1):
                # WF[f] = PVA[j] * beta_VA[f,j] * VA[j] / FD[f,j]
                # log(WF[f]) = log(PVA[j]) + log(beta_VA[f,j]) + log(VA[j]) - log(FD[f,j])

                lhs = log(WF[f])
                rhs = log(PVA[j]) + log(beta_VA[f, j]) + log(VA[j]) - log(FD[f, j])

                return lhs == rhs

        equations.append(CESAggregationEq())
        equations.append(CESFOCEq())
        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.PRODUCTION]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for CES value-added."""
        sectors = set_manager.get("J")
        factors = set_manager.get("F")
        n_sectors, n_factors = len(sectors), len(factors)

        if mode == "sam":
            # Extract factor payments from SAM
            FD0 = data.get_matrix("F", "J")  # shape: (n_factors, n_sectors)

            # Calculate value added per sector
            VA0 = FD0.sum(axis=0)

            # Calculate CES shares
            beta_VA = self._compute_shares(FD0, axis=0)

            # Efficiency parameter (assume 1.0 for now, could be calibrated)
            B_VA = np.ones(n_sectors)

        else:  # dummy mode
            # Use dummy defaults or uniform values
            FD0 = self._get_dummy_value("FD0", (n_factors, n_sectors), 1.0)
            # VA0 must be consistent with FD0: VA0[j] = sum_f FD0[f,j]
            VA0 = FD0.sum(axis=0)
            beta_VA = self._get_dummy_value(
                "beta_VA", (n_factors, n_sectors), 1.0 / n_factors
            )
            B_VA = np.ones(n_sectors)

        return {
            "FD0": FD0,
            "VA0": VA0,
            "beta_VA": beta_VA,
            "B_VA": B_VA,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        # Initialize from "0" parameters
        if "FD0" in calibrated and "FD" in var_manager:
            var_manager.get("FD").value = calibrated["FD0"].copy()
        if "VA0" in calibrated and "VA" in var_manager:
            var_manager.get("VA").value = calibrated["VA0"].copy()


class LeontiefIntermediate(Block):
    """Leontief intermediate input block."""

    name: str = Field(default="Leontief_INT")
    description: str = Field(default="Leontief intermediate inputs")

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["J", "I"]
        self.parameters = {"a_io": ParameterSpec(name="a_io", domains=("I", "J"))}
        self.variables = {
            "XST": VariableSpec(name="XST", domains=("I", "J"), lower=0.0),
            "Z": VariableSpec(name="Z", domains=("J",), lower=0.0),
        }

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        sectors = set_manager.get("J")
        commodities = set_manager.get("I")
        n_sectors, n_comm = len(sectors), len(commodities)

        parameters["a_io"] = Parameter(
            name="a_io", value=np.full((n_comm, n_sectors), 0.1), domains=("I", "J")
        )
        variables["XST"] = Variable(
            name="XST",
            value=np.ones((n_comm, n_sectors)),
            domains=("I", "J"),
            lower=0.0,
        )
        variables["Z"] = Variable(
            name="Z", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )

        equations = []

        class IntermediateDemandEq(SymbolicEquation):
            name: str = "Intermediate_Demand"
            domains: tuple = ("I", "J")
            description: str = "XST[i,j] = a_io[i,j] * Z[j]"

            def define(self, set_manager, variables, parameters):
                """Legacy closure-based definition."""
                constraints = {}
                sectors = set_manager.get("J")
                commodities = set_manager.get("I")

                for j_idx, j in enumerate(sectors):
                    Z = variables["Z"].value[j_idx]
                    for i_idx, i in enumerate(commodities):
                        a_io = parameters["a_io"].value[i_idx, j_idx]
                        XST = variables["XST"].value[i_idx, j_idx]
                        constraints[(i, j)] = lambda x=XST, a=a_io, z=Z: x - a * z

                return constraints

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for intermediate demand."""
                i, j = indices

                # Get Pyomo variables and parameters
                XST = getattr(pyomo_model, "XST")
                Z = getattr(pyomo_model, "Z")
                a_io = getattr(pyomo_model, "a_io")

                # Linear constraint: XST[i,j] - a_io[i,j] * Z[j] = 0
                return XST[i, j] == a_io[i, j] * Z[j]

        equations.append(IntermediateDemandEq())
        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.PRODUCTION]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for Leontief intermediate inputs."""
        sectors = set_manager.get("J")
        commodities = set_manager.get("I")
        n_sectors, n_comm = len(sectors), len(commodities)

        if mode == "sam":
            # Extract intermediate inputs from SAM
            XST0 = data.get_matrix("I", "J")  # shape: (n_comm, n_sectors)

            # Calculate total output per sector (from production block)
            prod_params = data.get_block_params("CES_VA")
            if "VA0" in prod_params:
                # Z0 = VA0 + intermediate (simplified - assumes no taxes/subsidies)
                Z0 = prod_params["VA0"] + XST0.sum(axis=0)
            else:
                Z0 = XST0.sum(axis=0)

            # Calculate IO coefficients: a_io[i,j] = XST[i,j] / Z[j]
            a_io = self._compute_shares(
                XST0, axis=1
            )  # Normalize by column (sector output)

        else:  # dummy mode
            # Use dummy defaults or uniform values
            XST0 = self._get_dummy_value("XST0", (n_comm, n_sectors), 0.1)
            Z0 = self._get_dummy_value("Z0", (n_sectors,), 1.0)
            a_io = np.full((n_comm, n_sectors), 0.1)

        return {
            "XST0": XST0,
            "Z0": Z0,
            "a_io": a_io,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "XST0" in calibrated:
            if "XST" in var_manager:
                var_manager.get("XST").value = calibrated["XST0"].copy()
        if "Z0" in calibrated:
            if "Z" in var_manager:
                var_manager.get("Z").value = calibrated["Z0"].copy()


class CETTransformation(Block):
    """CET output transformation block."""

    omega: float = Field(default=2.0, gt=0)
    name: str = Field(default="CET")
    description: str = Field(default="CET output transformation")

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["J"]
        self.parameters = {
            "omega_CET": ParameterSpec(
                name="omega_CET", domains=("J",), default=self.omega
            ),
            "gamma_D": ParameterSpec(name="gamma_D", domains=("J",)),
            "gamma_E": ParameterSpec(name="gamma_E", domains=("J",)),
            "B_CET": ParameterSpec(name="B_CET", domains=("J",)),
        }
        self.variables = {
            "XD": VariableSpec(name="XD", domains=("J",), lower=0.0),
            "XE": VariableSpec(name="XE", domains=("J",), lower=0.0),
            "PD": VariableSpec(name="PD", domains=("J",), lower=0.0),
            "PE": VariableSpec(name="PE", domains=("J",), lower=0.0),
        }

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        sectors = set_manager.get("J")
        n_sectors = len(sectors)

        parameters["omega_CET"] = Parameter(
            name="omega_CET", value=np.full((n_sectors,), self.omega), domains=("J",)
        )
        parameters["gamma_D"] = Parameter(
            name="gamma_D", value=np.full((n_sectors,), 0.5), domains=("J",)
        )
        parameters["gamma_E"] = Parameter(
            name="gamma_E", value=np.full((n_sectors,), 0.5), domains=("J",)
        )
        parameters["B_CET"] = Parameter(
            name="B_CET", value=np.ones((n_sectors,)), domains=("J",)
        )

        variables["XD"] = Variable(
            name="XD", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )
        variables["XE"] = Variable(
            name="XE", value=np.ones((n_sectors,)) * 0.3, domains=("J",), lower=0.0
        )
        variables["PD"] = Variable(
            name="PD", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )
        variables["PE"] = Variable(
            name="PE", value=np.ones((n_sectors,)), domains=("J",), lower=0.0
        )

        equations = []

        class CETAggregationEq(SymbolicEquation):
            name: str = "CET_Aggregation"
            domains: tuple = ("J",)
            description: str = "CET transformation equation"

            def define(self, set_manager, variables, parameters):
                """Legacy closure-based definition."""
                constraints = {}
                sectors = set_manager.get("J")

                for j_idx, j in enumerate(sectors):
                    omega = parameters["omega_CET"].value[j_idx]
                    gamma_D = parameters["gamma_D"].value[j_idx]
                    gamma_E = parameters["gamma_E"].value[j_idx]
                    B = parameters["B_CET"].value[j_idx]
                    XD = variables["XD"].value[j_idx]
                    XE = variables["XE"].value[j_idx]

                    rho = (omega + 1) / omega if omega != 0 else 1.0

                    if abs(rho) > 1e-10:
                        cet_sum = gamma_D * (XD**rho) + gamma_E * (XE**rho)
                        Z = B * (cet_sum ** (1.0 / rho))
                    else:
                        Z = B * (XD**gamma_D) * (XE**gamma_E)

                    constraints[(j,)] = lambda z=XD + XE, cet=Z: z - cet

                return constraints

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CET aggregation."""
                from pyomo.environ import log

                j = indices[0]

                XD = getattr(pyomo_model, "XD")
                XE = getattr(pyomo_model, "XE")
                B_CET = getattr(pyomo_model, "B_CET")
                gamma_D = getattr(pyomo_model, "gamma_D")
                gamma_E = getattr(pyomo_model, "gamma_E")

                # Simplified: log(XD + XE) = log(B_CET) + gamma_D * log(XD) + gamma_E * log(XE)
                lhs = log(XD[j] + XE[j])
                rhs = log(B_CET[j]) + gamma_D[j] * log(XD[j]) + gamma_E[j] * log(XE[j])

                return lhs == rhs

        class CETFOCEq(SymbolicEquation):
            name: str = "CET_FOC"
            domains: tuple = ("J",)
            description: str = "CET first-order condition"

            def define(self, set_manager, variables, parameters):
                """Legacy closure-based definition."""
                constraints = {}
                sectors = set_manager.get("J")

                for j_idx, j in enumerate(sectors):
                    omega = parameters["omega_CET"].value[j_idx]
                    gamma_D = parameters["gamma_D"].value[j_idx]
                    gamma_E = parameters["gamma_E"].value[j_idx]
                    XD = variables["XD"].value[j_idx]
                    XE = variables["XE"].value[j_idx]
                    PD = variables["PD"].value[j_idx]
                    PE = variables["PE"].value[j_idx]

                    rho = (omega + 1) / omega if omega != 0 else 1.0

                    if XE > 1e-10 and gamma_E > 1e-10:
                        price_ratio = (gamma_D / gamma_E) * ((XD / XE) ** (rho - 1))
                    else:
                        price_ratio = 1.0

                    if PE > 1e-10:
                        constraints[(j,)] = (
                            lambda pd=PD, pe=PE, pr=price_ratio: pd / pe - pr
                        )
                    else:
                        constraints[(j,)] = lambda: 0.0

                return constraints

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CET FOC."""
                from pyomo.environ import log

                j = indices[0]

                PD = getattr(pyomo_model, "PD")
                PE = getattr(pyomo_model, "PE")
                XD = getattr(pyomo_model, "XD")
                XE = getattr(pyomo_model, "XE")
                gamma_D = getattr(pyomo_model, "gamma_D")
                gamma_E = getattr(pyomo_model, "gamma_E")

                # log(PD/PE) = log(gamma_D/gamma_E) + (rho-1) * log(XD/XE)
                # Simplified: log(PD) - log(PE) = log(gamma_D) - log(gamma_E) + log(XD) - log(XE)
                lhs = log(PD[j]) - log(PE[j])
                rhs = log(gamma_D[j]) - log(gamma_E[j]) + log(XD[j]) - log(XE[j])

                return lhs == rhs

        equations.append(CETAggregationEq())
        equations.append(CETFOCEq())
        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.TRADE]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for CET transformation."""
        sectors = set_manager.get("J")
        n_sectors = len(sectors)

        if mode == "sam":
            # Extract domestic sales and exports from SAM
            # Domestic sales: from sectors to commodities
            XD0 = data.get_matrix("J", "I").sum(axis=1)  # Sum over commodities

            # Exports: from specific production accounts to ROW
            # SAM structure: exports are in AGR_2, OTHIND_1, FOOD_1, SER_2
            try:
                export_accounts = ["AGR_2", "OTHIND_1", "FOOD_1", "SER_2", "ADM"]
                XE0_list = []
                for acc in export_accounts:
                    if acc in data.sam.data.index and "ROW" in data.sam.data.columns:
                        val = data.sam.data.loc[acc, "ROW"]
                        XE0_list.append(float(val) if val != 0 else 0.0)
                    else:
                        XE0_list.append(0.0)
                XE0 = np.array(XE0_list)
            except Exception:
                XE0 = np.zeros(n_sectors)

            # Total output
            Z0 = XD0 + XE0

            # Calculate CET shares (with protection against division by zero)
            # Use np.where to only divide where Z0 > 0 (GAMS: gamma_D(j)$Z0(j) = ...)
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_D = np.where(Z0 > 0, XD0 / Z0, 0.0)
                gamma_E = np.where(Z0 > 0, XE0 / Z0, 0.0)

            # Efficiency parameter
            B_CET = np.ones(n_sectors)

        else:  # dummy mode
            XD0 = self._get_dummy_value("XD0", (n_sectors,), 0.7)
            XE0 = self._get_dummy_value("XE0", (n_sectors,), 0.3)
            Z0 = XD0 + XE0
            # Calculate CET shares (with protection against division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_D = np.where(Z0 > 0, XD0 / Z0, 0.0)
                gamma_E = np.where(Z0 > 0, XE0 / Z0, 0.0)
            B_CET = np.ones(n_sectors)

        return {
            "XD0": XD0,
            "XE0": XE0,
            "Z0": Z0,
            "gamma_D": gamma_D,
            "gamma_E": gamma_E,
            "B_CET": B_CET,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "XD0" in calibrated:
            if "XD" in var_manager:
                var_manager.get("XD").value = calibrated["XD0"].copy()
        if "XE0" in calibrated:
            if "XE" in var_manager:
                var_manager.get("XE").value = calibrated["XE0"].copy()
