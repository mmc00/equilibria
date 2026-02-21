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


class PEPTradeFlowInit(Block):
    """PEP trade-flow blockwise initializer/validator (EQ57-EQ64).

    This block is meant for blockwise level initialization workflows where
    variable levels are updated from calibrated PEP parameters and immediately
    validated against trade equations.
    """

    name: str = Field(default="PEP_TradeFlow_Init", description="Block name")
    description: str = Field(
        default="PEP blockwise trade-flow initialization and validation",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I", "J"]
        self.parameters = {
            "QO0": ParameterSpec(name="QO0", domains=("I",), description="Benchmark Q"),
            "DDO0": ParameterSpec(name="DDO0", domains=("I",), description="Benchmark DD"),
            "IMO0": ParameterSpec(name="IMO0", domains=("I",), description="Benchmark IM"),
            "EXDO0": ParameterSpec(name="EXDO0", domains=("I",), description="Benchmark EXD"),
            "PCO0": ParameterSpec(name="PCO0", domains=("I",), description="Benchmark PC"),
            "PDO0": ParameterSpec(name="PDO0", domains=("I",), description="Benchmark PD"),
            "PMO0": ParameterSpec(name="PMO0", domains=("I",), description="Benchmark PM"),
            "MRGNO0": ParameterSpec(name="MRGNO0", domains=("I",), description="Benchmark MRGN"),
            "DITO0": ParameterSpec(name="DITO0", domains=("I",), description="Benchmark DIT"),
        }
        self.variables = {
            "Q": VariableSpec(name="Q", domains=("I",), lower=0.0, description="Composite demand"),
            "DD": VariableSpec(name="DD", domains=("I",), lower=0.0, description="Domestic demand"),
            "IM": VariableSpec(name="IM", domains=("I",), lower=0.0, description="Imports"),
            "EXD": VariableSpec(name="EXD", domains=("I",), lower=0.0, description="Export demand"),
            "PC": VariableSpec(name="PC", domains=("I",), lower=0.0, description="Composite price"),
            "PD": VariableSpec(name="PD", domains=("I",), lower=0.0, description="Domestic price"),
            "PM": VariableSpec(name="PM", domains=("I",), lower=0.0, description="Import price"),
            "MRGN": VariableSpec(name="MRGN", domains=("I",), lower=0.0, description="Margins"),
            "DIT": VariableSpec(name="DIT", domains=("I",), lower=0.0, description="Intermediate demand"),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """No symbolic equations here; this block provides init/validation hooks."""
        _ = (set_manager, parameters, variables)
        return []

    @staticmethod
    def _first_map(source: dict[str, Any], *names: str) -> dict[Any, float]:
        for name in names:
            obj = source.get(name)
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _ensure_map(source: dict[str, Any], name: str) -> dict[Any, float]:
        obj = source.get(name)
        if isinstance(obj, dict):
            return obj
        out: dict[Any, float] = {}
        source[name] = out
        return out

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        """Initialize trade-flow levels from calibrated benchmark maps."""
        _ = mode
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        qo = self._first_map(parameters, "QO0", "QO")
        ddo = self._first_map(parameters, "DDO0", "DDO")
        imo = self._first_map(parameters, "IMO0", "IMO")
        exdo = self._first_map(parameters, "EXDO0", "EXDO")
        pco = self._first_map(parameters, "PCO0", "PCO")
        pdo = self._first_map(parameters, "PDO0", "PDO")
        pmo = self._first_map(parameters, "PMO0", "PMO")
        mrgno = self._first_map(parameters, "MRGNO0", "MRGNO")
        dito = self._first_map(parameters, "DITO0", "DITO")
        tmrg = self._first_map(parameters, "tmrg")
        tmrg_x = self._first_map(parameters, "tmrg_X")

        q = self._ensure_map(variables, "Q")
        dd = self._ensure_map(variables, "DD")
        im = self._ensure_map(variables, "IM")
        exd = self._ensure_map(variables, "EXD")
        pc = self._ensure_map(variables, "PC")
        pd = self._ensure_map(variables, "PD")
        pm = self._ensure_map(variables, "PM")
        mrgn = self._ensure_map(variables, "MRGN")
        dit = self._ensure_map(variables, "DIT")
        di = self._first_map(variables, "DI")

        for i in I:
            q[i] = float(qo.get(i, q.get(i, 0.0)))
            dd[i] = float(ddo.get(i, dd.get(i, 0.0)))
            im[i] = float(imo.get(i, im.get(i, 0.0)))
            exd[i] = float(exdo.get(i, exd.get(i, 0.0)))
            pc[i] = float(pco.get(i, pc.get(i, 1.0)))
            pd[i] = float(pdo.get(i, pd.get(i, 1.0)))
            pm[i] = float(pmo.get(i, pm.get(i, 1.0)))

            if i in dito:
                dit[i] = float(dito.get(i, 0.0))
            else:
                dit[i] = float(sum(di.get((i, j), 0.0) for j in J))

            if i in mrgno:
                mrgn[i] = float(mrgno.get(i, 0.0))
            else:
                m = 0.0
                for ij in I:
                    t = float(tmrg.get((i, ij), 0.0))
                    if abs(ddo.get(ij, 0.0)) > 1e-12:
                        m += t * dd.get(ij, 0.0)
                    if abs(imo.get(ij, 0.0)) > 1e-12:
                        m += t * im.get(ij, 0.0)
                    if abs(exdo.get(ij, 0.0)) > 1e-12:
                        m += float(tmrg_x.get((i, ij), 0.0)) * exd.get(ij, 0.0)
                mrgn[i] = m

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        """Validate EQ57-EQ64 residuals for initialized trade-flow levels."""
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        q = self._first_map(variables, "Q")
        dd = self._first_map(variables, "DD")
        im = self._first_map(variables, "IM")
        exd = self._first_map(variables, "EXD")
        xs = self._first_map(variables, "XS")
        xst = self._first_map(variables, "XST")
        ds = self._first_map(variables, "DS")
        ex = self._first_map(variables, "EX")
        p = self._first_map(variables, "P")
        pt = self._first_map(variables, "PT")
        pe = self._first_map(variables, "PE")
        pl = self._first_map(variables, "PL")
        pd = self._first_map(variables, "PD")
        pm = self._first_map(variables, "PM")
        pwm = self._first_map(variables, "PWM")
        pe_fob = self._first_map(variables, "PE_FOB")
        mrgn = self._first_map(variables, "MRGN")

        ddo = self._first_map(parameters, "DDO0", "DDO")
        imo = self._first_map(parameters, "IMO0", "IMO")
        exdo0 = self._first_map(parameters, "EXDO0")
        exdo = self._first_map(parameters, "EXDO", "EXDO0")
        xso = self._first_map(parameters, "XSO0", "XSO")
        xsto = self._first_map(parameters, "XSTO0", "XSTO")
        dso = self._first_map(parameters, "DSO0", "DSO")
        exo = self._first_map(parameters, "EXO0", "EXO")
        tmrg = self._first_map(parameters, "tmrg")
        tmrg_x = self._first_map(parameters, "tmrg_X")
        rho_xt = self._first_map(parameters, "rho_XT")
        sigma_xt = self._first_map(parameters, "sigma_XT")
        beta_xt = self._first_map(parameters, "beta_XT")
        b_xt = self._first_map(parameters, "B_XT")
        rho_x = self._first_map(parameters, "rho_X")
        beta_x = self._first_map(parameters, "beta_X")
        sigma_x = self._first_map(parameters, "sigma_X")
        b_x = self._first_map(parameters, "B_X")
        rho_m = self._first_map(parameters, "rho_M")
        beta_m = self._first_map(parameters, "beta_M")
        b_m = self._first_map(parameters, "B_M")
        sigma_m = self._first_map(parameters, "sigma_M")
        sigma_xd = self._first_map(parameters, "sigma_XD")
        pwx = self._first_map(parameters, "PWX")
        e = float(parameters.get("e", 1.0))

        residuals: dict[str, float] = {}

        for i in I:
            rhs = 0.0
            for ij in I:
                t = float(tmrg.get((i, ij), 0.0))
                if abs(ddo.get(ij, 0.0)) > 1e-12:
                    rhs += t * dd.get(ij, 0.0)
                if abs(imo.get(ij, 0.0)) > 1e-12:
                    rhs += t * im.get(ij, 0.0)
                if abs(exdo0.get(ij, 0.0)) > 1e-12:
                    rhs += float(tmrg_x.get((i, ij), 0.0)) * exd.get(ij, 0.0)
            residuals[f"EQ57_{i}"] = mrgn.get(i, 0.0) - rhs

        for j in J:
            rho = float(rho_xt.get(j, 1.0))
            b = float(b_xt.get(j, 1.0))
            if abs(rho) <= 1e-12:
                continue
            term = 0.0
            for i in I:
                if abs(xso.get((j, i), 0.0)) > 1e-12:
                    term += float(beta_xt.get((j, i), 0.0)) * (xs.get((j, i), 0.0) ** rho)
            expected = b * (term ** (1.0 / rho)) if term > 0.0 else 0.0
            residuals[f"EQ58_{j}"] = xst.get(j, 0.0) - expected

        for j in J:
            sig = float(sigma_xt.get(j, 2.0))
            b = float(b_xt.get(j, 1.0))
            pt_j = float(pt.get(j, 0.0))
            xst_j = float(xst.get(j, 0.0))
            if abs(sig) <= 1e-12 or b <= 0.0 or pt_j <= 0.0:
                continue
            for i in I:
                if abs(xso.get((j, i), 0.0)) <= 1e-12:
                    continue
                if abs(xso.get((j, i), 0.0) - xsto.get(j, 0.0)) <= 1e-12:
                    continue
                beta = float(beta_xt.get((j, i), 0.0))
                p_ji = float(p.get((j, i), 0.0))
                if beta <= 0.0 or p_ji <= 0.0 or xst_j <= 0.0:
                    continue
                expected = xst_j / (b ** (1.0 + sig)) * ((p_ji / (beta * pt_j)) ** sig)
                residuals[f"EQ59_{j}_{i}"] = xs.get((j, i), 0.0) - expected

        for j in J:
            for i in I:
                if abs(xso.get((j, i), 0.0)) <= 1e-12:
                    continue
                rho = float(rho_x.get((j, i), 1.0))
                b = float(b_x.get((j, i), 1.0))
                beta = float(beta_x.get((j, i), 0.5))
                if abs(rho) <= 1e-12:
                    continue
                term = 0.0
                if abs(exo.get((j, i), 0.0)) > 1e-12:
                    term += beta * (ex.get((j, i), 0.0) ** rho)
                if abs(dso.get((j, i), 0.0)) > 1e-12:
                    term += (1.0 - beta) * (ds.get((j, i), 0.0) ** rho)
                expected = b * (term ** (1.0 / rho)) if term > 0.0 else 0.0
                residuals[f"EQ60_{j}_{i}"] = xs.get((j, i), 0.0) - expected

        for j in J:
            for i in I:
                if abs(exo.get((j, i), 0.0)) <= 1e-12 or abs(dso.get((j, i), 0.0)) <= 1e-12:
                    continue
                beta = float(beta_x.get((j, i), 0.0))
                sig = float(sigma_x.get((j, i), 2.0))
                pe_i = float(pe.get(i, 0.0))
                pl_i = float(pl.get(i, 0.0))
                ds_ji = float(ds.get((j, i), 0.0))
                if beta <= 0.0 or beta >= 1.0 or pe_i <= 0.0 or pl_i <= 0.0:
                    continue
                expected = (((1.0 - beta) / beta) * (pe_i / pl_i)) ** sig * ds_ji
                residuals[f"EQ61_{j}_{i}"] = ex.get((j, i), 0.0) - expected

        for i in I:
            if abs(exdo0.get(i, 0.0)) <= 1e-12:
                continue
            sig = float(sigma_xd.get(i, 1.0))
            pe_fob_i = float(pe_fob.get(i, 0.0))
            if pe_fob_i <= 0.0:
                continue
            pwx_i = float(pwx.get(i, pwm.get(i, 1.0)))
            expected = float(exdo.get(i, 0.0)) * ((e * pwx_i) / pe_fob_i) ** sig
            residuals[f"EQ62_{i}"] = exd.get(i, 0.0) - expected

        for i in I:
            rho = float(rho_m.get(i, -0.5))
            b = float(b_m.get(i, 1.0))
            beta = float(beta_m.get(i, 0.5))
            if abs(rho) <= 1e-12:
                continue
            term = 0.0
            if abs(imo.get(i, 0.0)) > 1e-12:
                term += beta * (im.get(i, 0.0) ** (-rho))
            if abs(ddo.get(i, 0.0)) > 1e-12:
                term += (1.0 - beta) * (dd.get(i, 0.0) ** (-rho))
            expected = b * (term ** (-1.0 / rho)) if term > 0.0 else 0.0
            residuals[f"EQ63_{i}"] = q.get(i, 0.0) - expected

        for i in I:
            if abs(imo.get(i, 0.0)) <= 1e-12 or abs(ddo.get(i, 0.0)) <= 1e-12:
                continue
            beta = float(beta_m.get(i, 0.0))
            sig = float(sigma_m.get(i, 2.0))
            pd_i = float(pd.get(i, 0.0))
            pm_i = float(pm.get(i, 0.0))
            dd_i = float(dd.get(i, 0.0))
            if beta <= 0.0 or beta >= 1.0 or pd_i <= 0.0 or pm_i <= 0.0:
                continue
            expected = ((beta / (1.0 - beta)) * (pd_i / pm_i)) ** sig * dd_i
            residuals[f"EQ64_{i}"] = im.get(i, 0.0) - expected

        return residuals


class PEPTradeTransformationInit(Block):
    """PEP trade transformation blockwise initializer/validator (EQ58-EQ59)."""

    name: str = Field(default="PEP_TradeTransformation_Init", description="Block name")
    description: str = Field(
        default="PEP blockwise trade transformation initialization and validation",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["I", "J"]

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        _ = (set_manager, parameters, variables)
        return []

    @staticmethod
    def _first_map(source: dict[str, Any], *names: str) -> dict[Any, float]:
        for name in names:
            obj = source.get(name)
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _ensure_map(source: dict[str, Any], name: str) -> dict[Any, float]:
        obj = source.get(name)
        if isinstance(obj, dict):
            return obj
        out: dict[Any, float] = {}
        source[name] = out
        return out

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        _ = mode
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        xso = self._first_map(parameters, "XSO0", "XSO")
        rho_xt = self._first_map(parameters, "rho_XT")
        beta_xt = self._first_map(parameters, "beta_XT")
        b_xt = self._first_map(parameters, "B_XT")

        xs = self._ensure_map(variables, "XS")
        xst = self._ensure_map(variables, "XST")

        for j in J:
            rho = float(rho_xt.get(j, 1.0))
            b = float(b_xt.get(j, 1.0))
            if abs(rho) <= 1e-12 or b <= 0.0:
                continue
            s = 0.0
            for i in I:
                if abs(xso.get((j, i), 0.0)) > 1e-12:
                    s += float(beta_xt.get((j, i), 0.0)) * (float(xs.get((j, i), 0.0)) ** rho)
            if s > 0.0:
                xst[j] = b * (s ** (1.0 / rho))

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        xso = self._first_map(parameters, "XSO0", "XSO")
        xsto = self._first_map(parameters, "XSTO0", "XSTO")
        rho_xt = self._first_map(parameters, "rho_XT")
        beta_xt = self._first_map(parameters, "beta_XT")
        b_xt = self._first_map(parameters, "B_XT")
        sigma_xt = self._first_map(parameters, "sigma_XT")

        xs = self._first_map(variables, "XS")
        xst = self._first_map(variables, "XST")
        pt = self._first_map(variables, "PT")
        p = self._first_map(variables, "P")

        residuals: dict[str, float] = {}
        for j in J:
            rho = float(rho_xt.get(j, 1.0))
            b = float(b_xt.get(j, 1.0))
            if abs(rho) > 1e-12:
                s = 0.0
                for i in I:
                    if abs(xso.get((j, i), 0.0)) > 1e-12:
                        s += float(beta_xt.get((j, i), 0.0)) * (float(xs.get((j, i), 0.0)) ** rho)
                expected_xst = b * (s ** (1.0 / rho)) if s > 0.0 else 0.0
                residuals[f"EQ58_{j}"] = float(xst.get(j, 0.0)) - expected_xst

            sig = float(sigma_xt.get(j, 2.0))
            pt_j = float(pt.get(j, 0.0))
            xst_j = float(xst.get(j, 0.0))
            if abs(sig) <= 1e-12 or b <= 0.0 or pt_j <= 0.0 or xst_j <= 0.0:
                continue
            for i in I:
                if abs(xso.get((j, i), 0.0)) <= 1e-12:
                    continue
                if abs(xso.get((j, i), 0.0) - xsto.get(j, 0.0)) <= 1e-12:
                    continue
                beta = float(beta_xt.get((j, i), 0.0))
                p_ji = float(p.get((j, i), 0.0))
                if beta <= 0.0 or p_ji <= 0.0:
                    continue
                expected_xs = xst_j / (b ** (1.0 + sig)) * ((p_ji / (beta * pt_j)) ** sig)
                residuals[f"EQ59_{j}_{i}"] = float(xs.get((j, i), 0.0)) - expected_xs

        return residuals


class PEPCommodityBalanceInit(Block):
    """PEP commodity-balance blockwise initializer/validator.

    Reconciles commodity quantities by coupling:
    - EQ57 (margins),
    - EQ63 (Armington CES quantity),
    - EQ79 (composite value identity),
    - EQ84 (composite-good market clearing).
    """

    name: str = Field(default="PEP_CommodityBalance_Init", description="Block name")
    description: str = Field(
        default="PEP blockwise commodity balance initialization and validation",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["I", "J"]

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        _ = (set_manager, parameters, variables)
        return []

    @staticmethod
    def _first_map(source: dict[str, Any], *names: str) -> dict[Any, float]:
        for name in names:
            obj = source.get(name)
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _ensure_map(source: dict[str, Any], name: str) -> dict[Any, float]:
        obj = source.get(name)
        if isinstance(obj, dict):
            return obj
        out: dict[Any, float] = {}
        source[name] = out
        return out

    @staticmethod
    def _ces_q(dd: float, im: float, beta: float, rho: float, b_m: float) -> float:
        if b_m <= 0.0:
            return 0.0
        term = 0.0
        if im > 0.0:
            term += beta * (im ** (-rho))
        if dd > 0.0:
            term += (1.0 - beta) * (dd ** (-rho))
        if term <= 0.0 or abs(rho) <= 1e-12:
            return 0.0
        return b_m * (term ** (-1.0 / rho))

    def _solve_dd_im(
        self,
        *,
        q_target: float,
        pc: float,
        pd: float,
        pm: float,
        beta: float,
        rho: float,
        b_m: float,
        dd0: float,
        im0: float,
    ) -> tuple[float, float]:
        if q_target <= 0.0:
            return 0.0, 0.0
        if pc <= 0.0:
            return max(dd0, 0.0), max(im0, 0.0)

        budget = pc * q_target
        eps = 1e-10

        # One-source cases.
        if pm <= 0.0 and pd > 0.0:
            return budget / pd, 0.0
        if pd <= 0.0 and pm > 0.0:
            return 0.0, budget / pm
        if pd <= 0.0 or pm <= 0.0:
            return max(dd0, 0.0), max(im0, 0.0)

        dd_min = eps
        dd_max = max(eps, budget / pd - eps)
        if dd_max <= dd_min:
            dd = max(eps, budget / pd)
            im = max(eps, (budget - pd * dd) / pm)
            return dd, im

        target = q_target

        def f(dd: float) -> float:
            im = (budget - pd * dd) / pm
            if im <= 0.0 or dd <= 0.0:
                return float("inf")
            return self._ces_q(dd, im, beta, rho, b_m) - target

        fa = f(dd_min)
        fb = f(dd_max)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            # Fallback proportional split
            share = max(0.0, min(1.0, dd0 / (dd0 + im0) if (dd0 + im0) > 1e-12 else 0.5))
            dd = share * budget / pd
            im = (budget - pd * dd) / pm
            return max(dd, 0.0), max(im, 0.0)

        # If bracket does not straddle zero, fallback to endpoint with lower abs error.
        if fa * fb > 0.0:
            cand = [(dd_min, abs(fa)), (dd_max, abs(fb))]
            dd = min(cand, key=lambda t: t[1])[0]
            im = (budget - pd * dd) / pm
            return max(dd, 0.0), max(im, 0.0)

        a, b = dd_min, dd_max
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = f(m)
            if not np.isfinite(fm):
                a = m
                continue
            if abs(fm) < 1e-10:
                a = b = m
                break
            if fa * fm <= 0.0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm

        dd = 0.5 * (a + b)
        im = (budget - pd * dd) / pm
        return max(dd, 0.0), max(im, 0.0)

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        _ = mode
        I = tuple(set_manager.get("I"))
        H = tuple(set_manager.get("H")) if "H" in set_manager else tuple()

        ddo0 = self._first_map(parameters, "DDO0", "DDO")
        imo0 = self._first_map(parameters, "IMO0", "IMO")
        exdo0 = self._first_map(parameters, "EXDO0", "EXDO")
        tmrg = self._first_map(parameters, "tmrg")
        tmrg_x = self._first_map(parameters, "tmrg_X")
        rho_m = self._first_map(parameters, "rho_M")
        beta_m = self._first_map(parameters, "beta_M")
        b_m = self._first_map(parameters, "B_M")

        q = self._ensure_map(variables, "Q")
        dd = self._ensure_map(variables, "DD")
        im = self._ensure_map(variables, "IM")
        exd = self._ensure_map(variables, "EXD")
        pc = self._first_map(variables, "PC")
        pd = self._first_map(variables, "PD")
        pm = self._first_map(variables, "PM")
        mrgn = self._ensure_map(variables, "MRGN")
        c = self._first_map(variables, "C")
        cg = self._first_map(variables, "CG")
        inv = self._first_map(variables, "INV")
        vstk = self._first_map(variables, "VSTK")
        dit = self._first_map(variables, "DIT")

        # Fixed-point iterations because MRGN depends on DD/IM/EXD and Q depends on MRGN.
        for _ in range(4):
            for i in I:
                m = 0.0
                for ij in I:
                    t = float(tmrg.get((i, ij), 0.0))
                    if abs(ddo0.get(ij, 0.0)) > 1e-12:
                        m += t * float(dd.get(ij, 0.0))
                    if abs(imo0.get(ij, 0.0)) > 1e-12:
                        m += t * float(im.get(ij, 0.0))
                    if abs(exdo0.get(ij, 0.0)) > 1e-12:
                        m += float(tmrg_x.get((i, ij), 0.0)) * float(exd.get(ij, 0.0))
                mrgn[i] = m

            for i in I:
                base_dem = (
                    sum(float(c.get((i, h), 0.0)) for h in H)
                    + float(cg.get(i, 0.0))
                    + float(inv.get(i, 0.0))
                    + float(vstk.get(i, 0.0))
                    + float(dit.get(i, 0.0))
                )
                q_target = max(0.0, base_dem + float(mrgn.get(i, 0.0)))

                has_dd = abs(ddo0.get(i, 0.0)) > 1e-12
                has_im = abs(imo0.get(i, 0.0)) > 1e-12
                if not has_dd and not has_im:
                    q[i] = q_target
                    dd[i] = 0.0
                    im[i] = 0.0
                    continue
                if has_dd and not has_im:
                    q[i] = q_target
                    dd[i] = (float(pc.get(i, 0.0)) * q_target / float(pd.get(i, 1.0))) if abs(float(pd.get(i, 1.0))) > 1e-12 else float(dd.get(i, 0.0))
                    im[i] = 0.0
                    continue
                if has_im and not has_dd:
                    q[i] = q_target
                    im[i] = (float(pc.get(i, 0.0)) * q_target / float(pm.get(i, 1.0))) if abs(float(pm.get(i, 1.0))) > 1e-12 else float(im.get(i, 0.0))
                    dd[i] = 0.0
                    continue

                dd_i, im_i = self._solve_dd_im(
                    q_target=q_target,
                    pc=float(pc.get(i, 0.0)),
                    pd=float(pd.get(i, 0.0)),
                    pm=float(pm.get(i, 0.0)),
                    beta=float(beta_m.get(i, 0.5)),
                    rho=float(rho_m.get(i, -0.5)),
                    b_m=float(b_m.get(i, 1.0)),
                    dd0=float(dd.get(i, 0.0)),
                    im0=float(im.get(i, 0.0)),
                )
                dd[i] = dd_i
                im[i] = im_i
                q[i] = q_target

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        I = tuple(set_manager.get("I"))
        H = tuple(set_manager.get("H")) if "H" in set_manager else tuple()

        ddo0 = self._first_map(parameters, "DDO0", "DDO")
        imo0 = self._first_map(parameters, "IMO0", "IMO")
        exdo0 = self._first_map(parameters, "EXDO0", "EXDO")
        tmrg = self._first_map(parameters, "tmrg")
        tmrg_x = self._first_map(parameters, "tmrg_X")
        rho_m = self._first_map(parameters, "rho_M")
        beta_m = self._first_map(parameters, "beta_M")
        b_m = self._first_map(parameters, "B_M")

        q = self._first_map(variables, "Q")
        dd = self._first_map(variables, "DD")
        im = self._first_map(variables, "IM")
        exd = self._first_map(variables, "EXD")
        pc = self._first_map(variables, "PC")
        pd = self._first_map(variables, "PD")
        pm = self._first_map(variables, "PM")
        mrgn = self._first_map(variables, "MRGN")
        c = self._first_map(variables, "C")
        cg = self._first_map(variables, "CG")
        inv = self._first_map(variables, "INV")
        vstk = self._first_map(variables, "VSTK")
        dit = self._first_map(variables, "DIT")

        residuals: dict[str, float] = {}
        for i in I:
            rhs57 = 0.0
            for ij in I:
                t = float(tmrg.get((i, ij), 0.0))
                if abs(ddo0.get(ij, 0.0)) > 1e-12:
                    rhs57 += t * float(dd.get(ij, 0.0))
                if abs(imo0.get(ij, 0.0)) > 1e-12:
                    rhs57 += t * float(im.get(ij, 0.0))
                if abs(exdo0.get(ij, 0.0)) > 1e-12:
                    rhs57 += float(tmrg_x.get((i, ij), 0.0)) * float(exd.get(ij, 0.0))
            residuals[f"EQ57_{i}"] = float(mrgn.get(i, 0.0)) - rhs57

            lhs79 = float(pc.get(i, 0.0)) * float(q.get(i, 0.0))
            rhs79 = 0.0
            if abs(imo0.get(i, 0.0)) > 1e-12:
                rhs79 += float(pm.get(i, 0.0)) * float(im.get(i, 0.0))
            if abs(ddo0.get(i, 0.0)) > 1e-12:
                rhs79 += float(pd.get(i, 0.0)) * float(dd.get(i, 0.0))
            residuals[f"EQ79_{i}"] = lhs79 - rhs79

            rho = float(rho_m.get(i, -0.5))
            if abs(rho) > 1e-12:
                term = 0.0
                if abs(imo0.get(i, 0.0)) > 1e-12:
                    term += float(beta_m.get(i, 0.5)) * (float(im.get(i, 0.0)) ** (-rho))
                if abs(ddo0.get(i, 0.0)) > 1e-12:
                    term += (1.0 - float(beta_m.get(i, 0.5))) * (float(dd.get(i, 0.0)) ** (-rho))
                expected_q = float(b_m.get(i, 1.0)) * (term ** (-1.0 / rho)) if term > 0.0 else 0.0
                residuals[f"EQ63_{i}"] = float(q.get(i, 0.0)) - expected_q

            expected_q84 = (
                sum(float(c.get((i, h), 0.0)) for h in H)
                + float(cg.get(i, 0.0))
                + float(inv.get(i, 0.0))
                + float(vstk.get(i, 0.0))
                + float(dit.get(i, 0.0))
                + float(mrgn.get(i, 0.0))
            )
            residuals[f"EQ84_{i}"] = float(q.get(i, 0.0)) - expected_q84

        return residuals


class PEPTradeMarketClearingInit(Block):
    """PEP trade market-clearing blockwise initializer/validator (EQ64/EQ88)."""

    name: str = Field(default="PEP_TradeMarketClearing_Init", description="Block name")
    description: str = Field(
        default="PEP blockwise trade market-clearing initialization and validation",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["I", "J"]

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        _ = (set_manager, parameters, variables)
        return []

    @staticmethod
    def _first_map(source: dict[str, Any], *names: str) -> dict[Any, float]:
        for name in names:
            obj = source.get(name)
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _ensure_map(source: dict[str, Any], name: str) -> dict[Any, float]:
        obj = source.get(name)
        if isinstance(obj, dict):
            return obj
        out: dict[Any, float] = {}
        source[name] = out
        return out

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        _ = mode
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        ddo0 = self._first_map(parameters, "DDO0", "DDO")
        imo0 = self._first_map(parameters, "IMO0", "IMO")
        dso0 = self._first_map(parameters, "DSO0", "DSO")
        beta_m = self._first_map(parameters, "beta_M")
        sigma_m = self._first_map(parameters, "sigma_M")
        alpha = float(parameters.get("trade_market_alpha", 1.0))
        alpha = max(0.0, min(1.0, alpha))

        ds = self._first_map(variables, "DS")
        dd = self._ensure_map(variables, "DD")
        im = self._ensure_map(variables, "IM")
        pd = self._first_map(variables, "PD")
        pm = self._first_map(variables, "PM")

        for i in I:
            if abs(ddo0.get(i, 0.0)) > 1e-12:
                dd_new = sum(float(ds.get((j, i), 0.0)) for j in J if abs(dso0.get((j, i), 0.0)) > 1e-12)
                dd[i] = (1.0 - alpha) * float(dd.get(i, 0.0)) + alpha * dd_new

            if abs(imo0.get(i, 0.0)) > 1e-12 and abs(ddo0.get(i, 0.0)) > 1e-12:
                beta = float(beta_m.get(i, 0.0))
                sig = float(sigma_m.get(i, 2.0))
                pd_i = float(pd.get(i, 0.0))
                pm_i = float(pm.get(i, 0.0))
                if 0.0 < beta < 1.0 and pd_i > 0.0 and pm_i > 0.0:
                    im_new = ((beta / (1.0 - beta)) * (pd_i / pm_i)) ** sig * float(dd.get(i, 0.0))
                    im[i] = (1.0 - alpha) * float(im.get(i, 0.0)) + alpha * im_new

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        I = tuple(set_manager.get("I"))
        J = tuple(set_manager.get("J"))

        ddo0 = self._first_map(parameters, "DDO0", "DDO")
        imo0 = self._first_map(parameters, "IMO0", "IMO")
        dso0 = self._first_map(parameters, "DSO0", "DSO")
        beta_m = self._first_map(parameters, "beta_M")
        sigma_m = self._first_map(parameters, "sigma_M")

        ds = self._first_map(variables, "DS")
        dd = self._first_map(variables, "DD")
        im = self._first_map(variables, "IM")
        pd = self._first_map(variables, "PD")
        pm = self._first_map(variables, "PM")

        residuals: dict[str, float] = {}
        for i in I:
            if abs(imo0.get(i, 0.0)) > 1e-12 and abs(ddo0.get(i, 0.0)) > 1e-12:
                beta = float(beta_m.get(i, 0.0))
                sig = float(sigma_m.get(i, 2.0))
                pd_i = float(pd.get(i, 0.0))
                pm_i = float(pm.get(i, 0.0))
                if 0.0 < beta < 1.0 and pd_i > 0.0 and pm_i > 0.0:
                    expected_im = ((beta / (1.0 - beta)) * (pd_i / pm_i)) ** sig * float(dd.get(i, 0.0))
                    residuals[f"EQ64_{i}"] = float(im.get(i, 0.0)) - expected_im

            if abs(ddo0.get(i, 0.0)) > 1e-12:
                supply = sum(float(ds.get((j, i), 0.0)) for j in J if abs(dso0.get((j, i), 0.0)) > 1e-12)
                residuals[f"EQ88_{i}"] = supply - float(dd.get(i, 0.0))

        return residuals
