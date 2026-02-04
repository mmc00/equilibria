"""CES (Constant Elasticity of Substitution) calibration.

Calibrates CES production function parameters from SAM data.
"""

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.babel import SAM
from equilibria.calibration.base import CalibrationResult, Calibrator
from equilibria.model import Model


class CESCalibrator(Calibrator):
    """Calibrator for CES production functions.

    Computes CES share parameters and efficiency parameters
    from SAM data and user-provided elasticities.

    The CES function is:
    VA[j] = B[j] * (sum_i beta[i,j] * FD[i,j]^(-rho[j]))^(-1/rho[j])

    Where:
    - rho[j] = (sigma[j] - 1) / sigma[j]
    - beta[i,j] = (WF[i] * FD[i,j]) / PVA[j] / VA[j]
    - B[j] = VA[j] / (sum_i beta[i,j] * FD[i,j]^(-rho[j]))^(-1/rho[j])

    Attributes:
        default_sigma: Default elasticity if not provided
    """

    name: str = Field(default="CES", description="Calibrator name")
    description: str = Field(
        default="CES production function calibration", description="Description"
    )
    default_sigma: float = Field(
        default=0.8, gt=0, description="Default elasticity of substitution"
    )

    def calibrate(
        self,
        model: Model,
        sam: SAM,
        elasticities: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """Calibrate CES parameters from SAM.

        Args:
            model: Model with CES blocks
            sam: SAM data
            elasticities: Optional sector-specific elasticities

        Returns:
            CalibrationResult with beta_VA, B_VA, sigma_VA
        """
        result = CalibrationResult()

        # Validate SAM
        self.validate_sam(sam)

        # Get sets from model
        sectors = list(model.set_manager.get("J").iter_elements())
        factors = list(model.set_manager.get("I").iter_elements())

        n_sectors = len(sectors)
        n_factors = len(factors)

        result.add_message(
            f"Calibrating CES for {n_sectors} sectors, {n_factors} factors"
        )

        # Initialize parameter arrays
        sigma_va = np.zeros(n_sectors)
        beta_va = np.zeros((n_factors, n_sectors))
        b_va = np.zeros(n_sectors)

        # Get elasticities
        for j_idx, sector in enumerate(sectors):
            if elasticities and sector in elasticities:
                sigma_va[j_idx] = elasticities[sector]
            else:
                sigma_va[j_idx] = self.default_sigma
                result.add_warning(f"Using default sigma for sector {sector}")

        # Compute rho from sigma
        rho = (sigma_va - 1.0) / sigma_va

        # Calibrate each sector
        for j_idx, sector in enumerate(sectors):
            # Get value added for this sector
            # VA = total payments to factors
            va = 0.0
            for f_idx, factor in enumerate(factors):
                # Get factor payment from SAM
                # Factor payments are in the factor rows, sector columns
                factor_payment = self.get_sam_value(sam, factor, sector)
                va += factor_payment

            if va <= 0:
                result.add_warning(f"Zero value added for sector {sector}")
                continue

            # Compute share parameters
            # beta[i,j] = (factor payment) / VA
            for f_idx, factor in enumerate(factors):
                factor_payment = self.get_sam_value(sam, factor, sector)
                beta_va[f_idx, j_idx] = factor_payment / va

            # Normalize shares to sum to 1
            share_sum = np.sum(beta_va[:, j_idx])
            if share_sum > 0:
                beta_va[:, j_idx] /= share_sum

            # Compute efficiency parameter B
            # At calibration point, CES should reproduce VA
            # B = VA / (sum_i beta[i] * FD[i]^(-rho))^(-1/rho)

            # Compute denominator
            denom_sum = 0.0
            for f_idx, factor in enumerate(factors):
                factor_payment = self.get_sam_value(sam, factor, sector)
                if factor_payment > 0 and beta_va[f_idx, j_idx] > 0:
                    # FD is factor demand = factor payment / factor price
                    # At base year, assume factor price = 1
                    fd = factor_payment
                    denom_sum += beta_va[f_idx, j_idx] * (fd ** (-rho[j_idx]))

            if denom_sum > 0 and rho[j_idx] != 0:
                b_va[j_idx] = va / (denom_sum ** (-1.0 / rho[j_idx]))
            else:
                b_va[j_idx] = 1.0

        # Store results
        result.parameters["sigma_VA"] = sigma_va
        result.parameters["beta_VA"] = beta_va
        result.parameters["B_VA"] = b_va

        result.statistics["n_sectors"] = n_sectors
        result.statistics["n_factors"] = n_factors
        result.statistics["avg_sigma"] = float(np.mean(sigma_va))

        result.add_message("CES calibration completed successfully")

        return result

    def get_info(self) -> dict[str, Any]:
        """Get calibrator info."""
        return {
            "name": self.name,
            "description": self.description,
            "default_sigma": self.default_sigma,
            "calibrated_parameters": ["sigma_VA", "beta_VA", "B_VA"],
        }
