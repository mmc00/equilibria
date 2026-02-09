"""Trade blocks for CGE models.

This module provides trade-related equation blocks including:
- Armington CES import aggregation
- CET export transformation
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks.base import Block, ParameterSpec, VariableSpec
from equilibria.core.calibration_phase import CalibrationPhase
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.parameters import Parameter
from equilibria.core.sets import SetManager
from equilibria.core.variables import Variable


class ArmingtonCES(Block):
    """Armington CES import aggregation block.

    Implements Armington aggregation of domestic and imported goods
    using CES specification. Consumers view domestic and imported
    varieties as imperfect substitutes.

    Required sets:
        - I: Commodities

    Equations:
        1. Armington aggregation: QA[i] = A_Ar[i] * (alpha_D[i]*QD[i]^(-rho) + alpha_M[i]*QM[i]^(-rho))^(-1/rho)
        2. FOC domestic: PD[i]/PA[i] = (alpha_D[i]) * (QA[i]/QD[i])^(rho+1)
        3. FOC imports: PM[i]/PA[i] = (alpha_M[i]) * (QA[i]/QM[i])^(rho+1)

    Attributes:
        sigma_m: Elasticity of substitution between domestic and imports (default: 1.5)
        name: Block name (default: "Armington")
    """

    sigma_m: float = Field(
        default=1.5, gt=0, description="Armington elasticity of substitution"
    )
    name: str = Field(default="Armington", description="Block name")
    description: str = Field(
        default="Armington CES import aggregation", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]

        self.parameters = {
            "sigma_Ar": ParameterSpec(
                name="sigma_Ar",
                domains=("I",),
                default=self.sigma_m,
                description="Armington elasticity by commodity",
            ),
            "alpha_D": ParameterSpec(
                name="alpha_D",
                domains=("I",),
                description="Domestic share parameter",
            ),
            "alpha_M": ParameterSpec(
                name="alpha_M",
                domains=("I",),
                description="Import share parameter",
            ),
            "A_Ar": ParameterSpec(
                name="A_Ar",
                domains=("I",),
                description="Armington efficiency parameter",
            ),
        }

        self.variables = {
            "QA": VariableSpec(
                name="QA",
                domains=("I",),
                lower=0.0,
                description="Armington aggregate supply",
            ),
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Domestic good demand",
            ),
            "QM": VariableSpec(
                name="QM",
                domains=("I",),
                lower=0.0,
                description="Import demand",
            ),
            "PA": VariableSpec(
                name="PA",
                domains=("I",),
                lower=0.0,
                description="Armington aggregate price",
            ),
            "PD": VariableSpec(
                name="PD",
                domains=("I",),
                lower=0.0,
                description="Domestic price",
            ),
            "PM": VariableSpec(
                name="PM",
                domains=("I",),
                lower=0.0,
                description="Import price (CIF)",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the Armington block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Parameters
        sigma_vals = np.full((n_comm,), self.sigma_m)
        parameters["sigma_Ar"] = Parameter(
            name="sigma_Ar",
            value=sigma_vals,
            domains=("I",),
            description="Armington elasticity",
        )

        alpha_d_vals = np.full((n_comm,), 0.7)
        parameters["alpha_D"] = Parameter(
            name="alpha_D",
            value=alpha_d_vals,
            domains=("I",),
            description="Domestic share",
        )

        alpha_m_vals = np.full((n_comm,), 0.3)
        parameters["alpha_M"] = Parameter(
            name="alpha_M",
            value=alpha_m_vals,
            domains=("I",),
            description="Import share",
        )

        a_ar_vals = np.ones((n_comm,))
        parameters["A_Ar"] = Parameter(
            name="A_Ar",
            value=a_ar_vals,
            domains=("I",),
            description="Armington efficiency",
        )

        # Variables
        qa_vals = np.ones((n_comm,))
        variables["QA"] = Variable(
            name="QA",
            value=qa_vals,
            domains=("I",),
            lower=0.0,
            description="Armington aggregate",
        )

        qd_vals = np.ones((n_comm,)) * 0.7
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Domestic demand",
        )

        qm_vals = np.ones((n_comm,)) * 0.3
        variables["QM"] = Variable(
            name="QM",
            value=qm_vals,
            domains=("I",),
            lower=0.0,
            description="Import demand",
        )

        pa_vals = np.ones((n_comm,))
        variables["PA"] = Variable(
            name="PA",
            value=pa_vals,
            domains=("I",),
            lower=0.0,
            description="Armington price",
        )

        pd_vals = np.ones((n_comm,))
        variables["PD"] = Variable(
            name="PD",
            value=pd_vals,
            domains=("I",),
            lower=0.0,
            description="Domestic price",
        )

        pm_vals = np.ones((n_comm,))
        variables["PM"] = Variable(
            name="PM",
            value=pm_vals,
            domains=("I",),
            lower=0.0,
            description="Import price",
        )

        equations = []

        # Armington aggregation equation: QA[i] = A_Ar[i] * (alpha_D[i]*QD[i]^rho + alpha_M[i]*QM[i]^rho)^(1/rho)
        class ArmingtonAggregationEq(SymbolicEquation):
            name: str = "Armington_Aggregation"
            domains: tuple = ("I",)
            description: str = "Armington CES aggregation"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for Armington aggregation."""
                from pyomo.environ import log

                i = indices[0]

                QA = getattr(pyomo_model, "QA")
                QD = getattr(pyomo_model, "QD")
                QM = getattr(pyomo_model, "QM")
                A_Ar = getattr(pyomo_model, "A_Ar")
                alpha_D = getattr(pyomo_model, "alpha_D")
                alpha_M = getattr(pyomo_model, "alpha_M")
                sigma_Ar = getattr(pyomo_model, "sigma_Ar")

                # Log-linearized form for Cobb-Douglas (sigma = 1)
                # log(QA[i]) = log(A_Ar[i]) + alpha_D[i]*log(QD[i]) + alpha_M[i]*log(QM[i])
                lhs = log(QA[i])
                rhs = log(A_Ar[i]) + alpha_D[i] * log(QD[i]) + alpha_M[i] * log(QM[i])

                return lhs == rhs

        # FOC for domestic goods
        class ArmingtonFOCDomesticEq(SymbolicEquation):
            name: str = "Armington_FOC_Domestic"
            domains: tuple = ("I",)
            description: str = "FOC for domestic goods demand"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for domestic FOC."""
                from pyomo.environ import log

                i = indices[0]

                PD = getattr(pyomo_model, "PD")
                PA = getattr(pyomo_model, "PA")
                QA = getattr(pyomo_model, "QA")
                QD = getattr(pyomo_model, "QD")
                alpha_D = getattr(pyomo_model, "alpha_D")

                # PD[i]/PA[i] = alpha_D[i] * QA[i]/QD[i]
                # log(PD[i]) - log(PA[i]) = log(alpha_D[i]) + log(QA[i]) - log(QD[i])
                lhs = log(PD[i]) - log(PA[i])
                rhs = log(alpha_D[i]) + log(QA[i]) - log(QD[i])

                return lhs == rhs

        # FOC for imports
        class ArmingtonFOCImportEq(SymbolicEquation):
            name: str = "Armington_FOC_Import"
            domains: tuple = ("I",)
            description: str = "FOC for import demand"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for import FOC."""
                from pyomo.environ import log

                i = indices[0]

                PM = getattr(pyomo_model, "PM")
                PA = getattr(pyomo_model, "PA")
                QA = getattr(pyomo_model, "QA")
                QM = getattr(pyomo_model, "QM")
                alpha_M = getattr(pyomo_model, "alpha_M")

                # PM[i]/PA[i] = alpha_M[i] * QA[i]/QM[i]
                lhs = log(PM[i]) - log(PA[i])
                rhs = log(alpha_M[i]) + log(QA[i]) - log(QM[i])

                return lhs == rhs

        equations.append(ArmingtonAggregationEq())
        equations.append(ArmingtonFOCDomesticEq())
        equations.append(ArmingtonFOCImportEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.TRADE]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for Armington aggregation."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Extract domestic use and imports from SAM
            QD0 = data.get_matrix("I", "J").sum(axis=1)  # Sum over sectors

            # Imports: from ROW to specific consumption accounts
            # SAM structure: imports are in AGR_1, FOOD, OTHIND, SER_1
            try:
                import_accounts = ["AGR_1", "FOOD", "OTHIND", "SER_1", "ADM"]
                QM0_list = []
                for acc in import_accounts:
                    if "ROW" in data.sam.data.index and acc in data.sam.data.columns:
                        val = data.sam.data.loc["ROW", acc]
                        QM0_list.append(float(val) if val != 0 else 0.0)
                    else:
                        QM0_list.append(0.0)
                QM0 = np.array(QM0_list)
            except Exception:
                QM0 = np.zeros(n_comm)

            # Calculate Armington aggregate
            QA0 = QD0 + QM0

            # Calculate Armington shares (with protection against division by zero)
            # Use np.where to only divide where QA0 > 0 (GAMS: alpha_D(i)$QA0(i) = ...)
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha_D = np.where(QA0 > 0, QD0 / QA0, 0.0)
                alpha_M = np.where(QA0 > 0, QM0 / QA0, 0.0)
            A_Ar = np.ones(n_comm)

        else:  # dummy mode
            QD0 = self._get_dummy_value("QD0", (n_comm,), 0.7)
            QM0 = self._get_dummy_value("QM0", (n_comm,), 0.3)
            QA0 = QD0 + QM0
            # Calculate Armington shares (with protection against division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha_D = np.where(QA0 > 0, QD0 / QA0, 0.0)
                alpha_M = np.where(QA0 > 0, QM0 / QA0, 0.0)
            A_Ar = np.ones(n_comm)

        return {
            "QD0": QD0,
            "QM0": QM0,
            "QA0": QA0,
            "alpha_D": alpha_D,
            "alpha_M": alpha_M,
            "A_Ar": A_Ar,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QD0" in calibrated:
            if "QD" in var_manager:
                var_manager.get("QD").value = calibrated["QD0"].copy()
        if "QM0" in calibrated:
            if "QM" in var_manager:
                var_manager.get("QM").value = calibrated["QM0"].copy()
        if "QA0" in calibrated:
            if "QA" in var_manager:
                var_manager.get("QA").value = calibrated["QA0"].copy()


class CETExports(Block):
    """CET export supply block.

    Implements Constant Elasticity of Transformation between domestic
    sales and exports. Producers can transform output between domestic
    and export markets.

    Required sets:
        - J: Production sectors (or I: Commodities)

    Equations:
        1. CET: Z[j] = B_X[j] * (xi_D[j]*XD[j]^rho + xi_E[j]*XE[j]^rho)^(1/rho)
        2. FOC: PE[j]/PD[j] = (xi_E[j]/xi_D[j]) * (XE[j]/XD[j])^(rho-1)

    Attributes:
        sigma_e: Elasticity of transformation (default: 2.0)
        name: Block name (default: "CET_Exports")
    """

    sigma_e: float = Field(
        default=2.0, gt=0, description="CET elasticity of transformation"
    )
    name: str = Field(default="CET_Exports", description="Block name")
    description: str = Field(
        default="CET export transformation", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["J"]

        self.parameters = {
            "sigma_X": ParameterSpec(
                name="sigma_X",
                domains=("J",),
                default=self.sigma_e,
                description="CET elasticity by sector",
            ),
            "xi_D": ParameterSpec(
                name="xi_D",
                domains=("J",),
                description="Domestic transformation share",
            ),
            "xi_E": ParameterSpec(
                name="xi_E",
                domains=("J",),
                description="Export transformation share",
            ),
            "B_X": ParameterSpec(
                name="B_X",
                domains=("J",),
                description="CET efficiency parameter",
            ),
        }

        self.variables = {
            "Z": VariableSpec(
                name="Z",
                domains=("J",),
                lower=0.0,
                description="Total output",
            ),
            "XD": VariableSpec(
                name="XD",
                domains=("J",),
                lower=0.0,
                description="Domestic sales",
            ),
            "XE": VariableSpec(
                name="XE",
                domains=("J",),
                lower=0.0,
                description="Export supply",
            ),
            "PD": VariableSpec(
                name="PD",
                domains=("J",),
                lower=0.0,
                description="Domestic price",
            ),
            "PE": VariableSpec(
                name="PE",
                domains=("J",),
                lower=0.0,
                description="Export price (FOB)",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the CET exports block."""
        sectors = set_manager.get("J")
        n_sectors = len(sectors)

        # Parameters
        sigma_x_vals = np.full((n_sectors,), self.sigma_e)
        parameters["sigma_X"] = Parameter(
            name="sigma_X",
            value=sigma_x_vals,
            domains=("J",),
            description="CET elasticity",
        )

        xi_d_vals = np.full((n_sectors,), 0.7)
        parameters["xi_D"] = Parameter(
            name="xi_D",
            value=xi_d_vals,
            domains=("J",),
            description="Domestic share",
        )

        xi_e_vals = np.full((n_sectors,), 0.3)
        parameters["xi_E"] = Parameter(
            name="xi_E",
            value=xi_e_vals,
            domains=("J",),
            description="Export share",
        )

        b_x_vals = np.ones((n_sectors,))
        parameters["B_X"] = Parameter(
            name="B_X",
            value=b_x_vals,
            domains=("J",),
            description="CET efficiency",
        )

        # Variables
        z_vals = np.ones((n_sectors,))
        variables["Z"] = Variable(
            name="Z",
            value=z_vals,
            domains=("J",),
            lower=0.0,
            description="Total output",
        )

        xd_vals = np.ones((n_sectors,)) * 0.7
        variables["XD"] = Variable(
            name="XD",
            value=xd_vals,
            domains=("J",),
            lower=0.0,
            description="Domestic sales",
        )

        xe_vals = np.ones((n_sectors,)) * 0.3
        variables["XE"] = Variable(
            name="XE",
            value=xe_vals,
            domains=("J",),
            lower=0.0,
            description="Export supply",
        )

        pd_vals = np.ones((n_sectors,))
        variables["PD"] = Variable(
            name="PD",
            value=pd_vals,
            domains=("J",),
            lower=0.0,
            description="Domestic price",
        )

        pe_vals = np.ones((n_sectors,))
        variables["PE"] = Variable(
            name="PE",
            value=pe_vals,
            domains=("J",),
            lower=0.0,
            description="Export price",
        )

        equations = []

        # CET transformation equation: Z[j] = B_X[j] * (xi_D[j]*XD[j]^rho + xi_E[j]*XE[j]^rho)^(1/rho)
        class CETTransformationEq(SymbolicEquation):
            name: str = "CET_Transformation"
            domains: tuple = ("J",)
            description: str = "CET output transformation"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CET transformation."""
                from pyomo.environ import log

                j = indices[0]

                Z = getattr(pyomo_model, "Z")
                XD = getattr(pyomo_model, "XD")
                XE = getattr(pyomo_model, "XE")
                B_X = getattr(pyomo_model, "B_X")
                xi_D = getattr(pyomo_model, "xi_D")
                xi_E = getattr(pyomo_model, "xi_E")

                # Log-linearized form
                # log(Z[j]) = log(B_X[j]) + xi_D[j]*log(XD[j]) + xi_E[j]*log(XE[j])
                lhs = log(Z[j])
                rhs = log(B_X[j]) + xi_D[j] * log(XD[j]) + xi_E[j] * log(XE[j])

                return lhs == rhs

        # FOC for export transformation
        class CETFOCEq(SymbolicEquation):
            name: str = "CET_FOC"
            domains: tuple = ("J",)
            description: str = "FOC for CET transformation"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for CET FOC."""
                from pyomo.environ import log

                j = indices[0]

                PE = getattr(pyomo_model, "PE")
                PD = getattr(pyomo_model, "PD")
                XE = getattr(pyomo_model, "XE")
                XD = getattr(pyomo_model, "XD")
                xi_E = getattr(pyomo_model, "xi_E")
                xi_D = getattr(pyomo_model, "xi_D")

                # PE[j]/PD[j] = (xi_E[j]/xi_D[j]) * (XE[j]/XD[j])
                lhs = log(PE[j]) - log(PD[j])
                rhs = log(xi_E[j]) - log(xi_D[j]) + log(XE[j]) - log(XD[j])

                return lhs == rhs

        equations.append(CETTransformationEq())
        equations.append(CETFOCEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.TRADE]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for CET exports."""
        sectors = set_manager.get("J")
        n_sectors = len(sectors)

        if mode == "sam":
            # Get data from production block
            prod_params = data.get_block_params("CET")
            if "XE0" in prod_params:
                XE0 = prod_params["XE0"]
                XD0 = prod_params["XD0"]
            else:
                try:
                    XE0 = data.get_matrix("J", "ROW").flatten()  # Exports to ROW
                except Exception:
                    XE0 = np.zeros(n_sectors)
                XD0 = data.get_matrix("J", "I").sum(axis=1)

            # Calculate shares
            Z0 = XD0 + XE0
            # Use np.where to only divide where Z0 > 0 (GAMS: xi_D(j)$Z0(j) = ...)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi_D = np.where(Z0 > 0, XD0 / Z0, 0.0)
                xi_E = np.where(Z0 > 0, XE0 / Z0, 0.0)
            B_X = np.ones(n_sectors)

        else:  # dummy mode
            XE0 = self._get_dummy_value("XE0", (n_sectors,), 0.3)
            XD0 = self._get_dummy_value("XD0", (n_sectors,), 0.7)
            Z0 = XD0 + XE0
            # Use np.where to only divide where Z0 > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                xi_D = np.where(Z0 > 0, XD0 / Z0, 0.0)
                xi_E = np.where(Z0 > 0, XE0 / Z0, 0.0)
            B_X = np.ones(n_sectors)

        return {
            "XE0": XE0,
            "XD0": XD0,
            "Z0": Z0,
            "xi_D": xi_D,
            "xi_E": xi_E,
            "B_X": B_X,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "XE0" in calibrated:
            if "XE" in var_manager:
                var_manager.get("XE").value = calibrated["XE0"].copy()
        if "XD0" in calibrated:
            if "XD" in var_manager:
                var_manager.get("XD").value = calibrated["XD0"].copy()
