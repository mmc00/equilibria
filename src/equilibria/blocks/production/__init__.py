"""Production blocks for CGE models.

This module provides production-related equation blocks including:
- CES value-added aggregation
- Leontief intermediate inputs
- CET transformation
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


class CESValueAdded(Block):
    """CES value-added production block.

    Implements Constant Elasticity of Substitution (CES) aggregation
    of value-added inputs (typically labor and capital) with
    first-order conditions for factor demands.

    Required sets:
        - J: Production sectors
        - I: Factor inputs (e.g., labor, capital)

    Equations:
        1. CES aggregation: VA[j] = B_VA[j] * (sum_i beta_VA[i,j] * FD[i,j]^(-rho)) ^ (-1/rho)
        2. FOC labor: WC[i] = PVA[j] * dVA/dFD[i,j]
        3. FOC capital: Similar to labor

    Attributes:
        sigma: Elasticity of substitution (default: 0.8)
        name: Block name (default: "CES_VA")
    """

    sigma: float = Field(default=0.8, gt=0, description="Elasticity of substitution")
    name: str = Field(default="CES_VA", description="Block name")
    description: str = Field(
        default="CES value-added production", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        # Set required sets
        self.required_sets = ["J", "I"]

        # Define parameters
        self.parameters = {
            "sigma_VA": ParameterSpec(
                name="sigma_VA",
                domains=("J",),
                default=self.sigma,
                description="CES elasticity of substitution by sector",
            ),
            "beta_VA": ParameterSpec(
                name="beta_VA",
                domains=("I", "J"),
                description="CES share parameter",
            ),
            "B_VA": ParameterSpec(
                name="B_VA",
                domains=("J",),
                description="CES efficiency parameter",
            ),
        }

        # Define variables
        self.variables = {
            "VA": VariableSpec(
                name="VA",
                domains=("J",),
                lower=0.0,
                description="Value added by sector",
            ),
            "FD": VariableSpec(
                name="FD",
                domains=("I", "J"),
                lower=0.0,
                description="Factor demand by factor and sector",
            ),
            "PVA": VariableSpec(
                name="PVA",
                domains=("J",),
                lower=0.0,
                description="Price of value added",
            ),
            "WF": VariableSpec(
                name="WF",
                domains=("I",),
                lower=0.0,
                description="Factor price (wage/rental rate)",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the CES VA block.

        Creates parameters, variables, and equations for CES production.
        """
        # Get sets
        sectors = set_manager.get("J")
        factors = set_manager.get("I")

        # Create parameters with initial values
        n_sectors = len(sectors)
        n_factors = len(factors)

        # sigma_VA - elasticity by sector
        sigma_va_vals = np.full((n_sectors,), self.sigma)
        parameters["sigma_VA"] = Parameter(
            name="sigma_VA",
            value=sigma_va_vals,
            domains=("J",),
            description="CES elasticity by sector",
        )

        # beta_VA - share parameters (initialize equally)
        beta_va_vals = np.ones((n_factors, n_sectors)) / n_factors
        parameters["beta_VA"] = Parameter(
            name="beta_VA",
            value=beta_va_vals,
            domains=("I", "J"),
            description="CES share parameters",
        )

        # B_VA - efficiency parameter
        b_va_vals = np.ones((n_sectors,))
        parameters["B_VA"] = Parameter(
            name="B_VA",
            value=b_va_vals,
            domains=("J",),
            description="CES efficiency parameter",
        )

        # Create variables
        va_vals = np.ones((n_sectors,))
        variables["VA"] = Variable(
            name="VA",
            value=va_vals,
            domains=("J",),
            lower=0.0,
            description="Value added by sector",
        )

        fd_vals = np.ones((n_factors, n_sectors))
        variables["FD"] = Variable(
            name="FD",
            value=fd_vals,
            domains=("I", "J"),
            lower=0.0,
            description="Factor demand",
        )

        pva_vals = np.ones((n_sectors,))
        variables["PVA"] = Variable(
            name="PVA",
            value=pva_vals,
            domains=("J",),
            lower=0.0,
            description="Price of value added",
        )

        wf_vals = np.ones((n_factors,))
        variables["WF"] = Variable(
            name="WF",
            value=wf_vals,
            domains=("I",),
            lower=0.0,
            description="Factor prices",
        )

        # Return equations (placeholder - actual equations would be defined here)
        # For now, return empty list - equations will be added when we implement
        # the actual equation definitions
        return []


class LeontiefIntermediate(Block):
    """Leontief intermediate input block.

    Implements fixed-proportion (Leontief) intermediate input demands
    based on input-output coefficients.

    Required sets:
        - J: Production sectors
        - I: Commodities (intermediate inputs)

    Equations:
        1. Intermediate demand: XST[i,j] = a[i,j] * Z[j]
        where a[i,j] are fixed IO coefficients

    Attributes:
        name: Block name (default: "Leontief_INT")
    """

    name: str = Field(default="Leontief_INT", description="Block name")
    description: str = Field(
        default="Leontief intermediate inputs", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["J", "I"]

        self.parameters = {
            "a_io": ParameterSpec(
                name="a_io",
                domains=("I", "J"),
                description="Input-output coefficient",
            ),
        }

        self.variables = {
            "XST": VariableSpec(
                name="XST",
                domains=("I", "J"),
                lower=0.0,
                description="Intermediate demand",
            ),
            "Z": VariableSpec(
                name="Z",
                domains=("J",),
                lower=0.0,
                description="Sectoral output",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the Leontief intermediate block."""
        sectors = set_manager.get("J")
        commodities = set_manager.get("I")

        n_sectors = len(sectors)
        n_commodities = len(commodities)

        # IO coefficients (initialize to small values)
        a_io_vals = np.full((n_commodities, n_sectors), 0.1)
        parameters["a_io"] = Parameter(
            name="a_io",
            value=a_io_vals,
            domains=("I", "J"),
            description="Input-output coefficients",
        )

        # Variables
        xst_vals = np.ones((n_commodities, n_sectors))
        variables["XST"] = Variable(
            name="XST",
            value=xst_vals,
            domains=("I", "J"),
            lower=0.0,
            description="Intermediate demand",
        )

        z_vals = np.ones((n_sectors,))
        variables["Z"] = Variable(
            name="Z",
            value=z_vals,
            domains=("J",),
            lower=0.0,
            description="Sectoral output",
        )

        return []


class CETTransformation(Block):
    """CET (Constant Elasticity of Transformation) block.

    Implements output transformation between domestic and export markets
    using CET specification.

    Required sets:
        - J: Production sectors

    Equations:
        1. CET transformation: Z[j] = B_CET[j] * (gamma_D[j]*XD[j]^rho + gamma_E[j]*XE[j]^rho)^(1/rho)
        2. FOC domestic: PD[j]/PE[j] = (gamma_D[j]/gamma_E[j]) * (XD[j]/XE[j])^(rho-1)

    Attributes:
        omega: Elasticity of transformation (default: 2.0)
        name: Block name (default: "CET")
    """

    omega: float = Field(default=2.0, gt=0, description="Elasticity of transformation")
    name: str = Field(default="CET", description="Block name")
    description: str = Field(
        default="CET output transformation", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["J"]

        self.parameters = {
            "omega_CET": ParameterSpec(
                name="omega_CET",
                domains=("J",),
                default=self.omega,
                description="CET elasticity of transformation",
            ),
            "gamma_D": ParameterSpec(
                name="gamma_D",
                domains=("J",),
                description="CET domestic share parameter",
            ),
            "gamma_E": ParameterSpec(
                name="gamma_E",
                domains=("J",),
                description="CET export share parameter",
            ),
            "B_CET": ParameterSpec(
                name="B_CET",
                domains=("J",),
                description="CET efficiency parameter",
            ),
        }

        self.variables = {
            "XD": VariableSpec(
                name="XD",
                domains=("J",),
                lower=0.0,
                description="Domestic output supply",
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
                description="Export price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[Equation]:
        """Set up the CET transformation block."""
        sectors = set_manager.get("J")
        n_sectors = len(sectors)

        # Parameters
        omega_vals = np.full((n_sectors,), self.omega)
        parameters["omega_CET"] = Parameter(
            name="omega_CET",
            value=omega_vals,
            domains=("J",),
            description="CET elasticity",
        )

        gamma_d_vals = np.full((n_sectors,), 0.5)
        parameters["gamma_D"] = Parameter(
            name="gamma_D",
            value=gamma_d_vals,
            domains=("J",),
            description="Domestic share parameter",
        )

        gamma_e_vals = np.full((n_sectors,), 0.5)
        parameters["gamma_E"] = Parameter(
            name="gamma_E",
            value=gamma_e_vals,
            domains=("J",),
            description="Export share parameter",
        )

        b_cet_vals = np.ones((n_sectors,))
        parameters["B_CET"] = Parameter(
            name="B_CET",
            value=b_cet_vals,
            domains=("J",),
            description="CET efficiency parameter",
        )

        # Variables
        xd_vals = np.ones((n_sectors,))
        variables["XD"] = Variable(
            name="XD",
            value=xd_vals,
            domains=("J",),
            lower=0.0,
            description="Domestic output",
        )

        xe_vals = np.ones((n_sectors,)) * 0.3
        variables["XE"] = Variable(
            name="XE",
            value=xe_vals,
            domains=("J",),
            lower=0.0,
            description="Export output",
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
