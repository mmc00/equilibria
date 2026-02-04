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
from equilibria.core.equations import Equation
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
    ) -> list[Equation]:
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

        return []


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
    ) -> list[Equation]:
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

        return []
