"""Leontief calibration for intermediate inputs.

Calibrates input-output coefficients from SAM data.
"""

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.babel import SAM
from equilibria.calibration.base import CalibrationResult, Calibrator
from equilibria.model import Model


class LeontiefCalibrator(Calibrator):
    """Calibrator for Leontief intermediate inputs.

    Computes input-output coefficients from SAM data.

    The Leontief function is:
    XST[i,j] = a_io[i,j] * Z[j]

    Where:
    - XST[i,j] = intermediate demand for commodity i by sector j
    - Z[j] = total output of sector j
    - a_io[i,j] = input-output coefficient

    The coefficient is computed as:
    a_io[i,j] = XST[i,j] / Z[j]

    Attributes:
        min_coefficient: Minimum threshold for IO coefficients
    """

    name: str = Field(default="Leontief", description="Calibrator name")
    description: str = Field(
        default="Leontief intermediate input calibration", description="Description"
    )
    min_coefficient: float = Field(
        default=1e-10, ge=0, description="Minimum IO coefficient threshold"
    )

    def calibrate(
        self,
        model: Model,
        sam: SAM,
        elasticities: dict[str, float] | None = None,
    ) -> CalibrationResult:
        """Calibrate Leontief IO coefficients from SAM.

        Args:
            model: Model with Leontief blocks
            sam: SAM data
            elasticities: Not used for Leontief (no elasticities)

        Returns:
            CalibrationResult with a_io coefficients
        """
        result = CalibrationResult()

        # Validate SAM
        self.validate_sam(sam)

        # Get sets from model
        commodities = list(model.set_manager.get("I").iter_elements())
        sectors = list(model.set_manager.get("J").iter_elements())

        n_comm = len(commodities)
        n_sectors = len(sectors)

        result.add_message(
            f"Calibrating Leontief for {n_comm} commodities, {n_sectors} sectors"
        )

        # Initialize IO coefficient matrix
        a_io = np.zeros((n_comm, n_sectors))

        # Compute IO coefficients
        for j_idx, sector in enumerate(sectors):
            # Compute total output for this sector
            # Z[j] = sum of all outputs from sector j
            total_output = 0.0

            # Intermediate demand (from other sectors)
            for i_idx, comm in enumerate(commodities):
                intermediate = self.get_sam_value(sam, comm, sector)
                total_output += intermediate

            # Final demand (households, government, exports, etc.)
            for account in sam.data.index:
                if account not in commodities:
                    final_demand = self.get_sam_value(sam, account, sector)
                    total_output += final_demand

            if total_output <= 0:
                result.add_warning(f"Zero output for sector {sector}")
                continue

            # Compute IO coefficients
            for i_idx, comm in enumerate(commodities):
                intermediate = self.get_sam_value(sam, comm, sector)

                # a_io[i,j] = intermediate demand / total output
                if intermediate > 0:
                    coeff = intermediate / total_output
                    a_io[i_idx, j_idx] = max(coeff, self.min_coefficient)

        # Store results
        result.parameters["a_io"] = a_io

        result.statistics["n_commodities"] = n_comm
        result.statistics["n_sectors"] = n_sectors
        result.statistics["nonzero_coeffs"] = int(np.sum(a_io > 0))
        result.statistics["total_coeffs"] = n_comm * n_sectors

        result.add_message("Leontief calibration completed successfully")

        return result

    def get_info(self) -> dict[str, Any]:
        """Get calibrator info."""
        return {
            "name": self.name,
            "description": self.description,
            "min_coefficient": self.min_coefficient,
            "calibrated_parameters": ["a_io"],
        }
