"""Balancing primitives for SAM matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from equilibria.sam_tools.enums import RASMode


class RASBalanceResult(BaseModel):
    """Result payload for one RAS balancing run."""

    matrix: pd.DataFrame
    target_totals: np.ndarray
    max_diff_before: float
    max_diff_after: float
    iterations: int
    converged: bool
    ras_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RASBalancer(BaseModel):
    """RAS balancer with interchangeable target modes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_ras_type(self, ras_type: str | None) -> RASMode:
        """Return the normalized ``RASMode`` for a user-provided mode string."""
        return RASMode.from_alias(ras_type)

    def _build_targets(
        self,
        row_totals: np.ndarray,
        col_totals: np.ndarray,
        ras_type: RASMode,
    ) -> np.ndarray:
        """Build per-account target totals according to the selected RAS mode."""
        if ras_type == RASMode.ARITHMETIC:
            return 0.5 * (row_totals + col_totals)
        if ras_type == RASMode.ROW:
            return row_totals.copy()
        if ras_type == RASMode.COLUMN:
            return col_totals.copy()

        raw = np.sqrt(np.maximum(row_totals, 0.0) * np.maximum(col_totals, 0.0))
        raw_sum = float(raw.sum())
        total = float(row_totals.sum())
        if raw_sum <= 0.0 or total <= 0.0:
            return raw
        return raw * (total / raw_sum)

    def balance_dataframe(
        self,
        matrix_df: pd.DataFrame,
        *,
        ras_type: str = RASMode.ARITHMETIC.value,
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> RASBalanceResult:
        """Run iterative RAS scaling and return the balanced matrix plus diagnostics."""
        sam = matrix_df.to_numpy(copy=True, dtype=float)
        if sam.ndim != 2 or sam.shape[0] != sam.shape[1]:
            raise ValueError("RAS balancing requires a square matrix")

        row_totals = sam.sum(axis=1)
        col_totals = sam.sum(axis=0)
        mode = self.resolve_ras_type(ras_type)
        target = self._build_targets(row_totals, col_totals, mode)

        max_diff_before = float(np.max(np.abs(row_totals - col_totals))) if sam.size else 0.0
        max_diff_after = max_diff_before
        converged = max_diff_before <= tolerance
        iterations = 0

        if not converged:
            for step in range(1, max_iterations + 1):
                current_rows = sam.sum(axis=1)
                for i in range(sam.shape[0]):
                    if target[i] <= 0.0:
                        sam[i, :] = 0.0
                    elif current_rows[i] > 0.0:
                        sam[i, :] *= target[i] / current_rows[i]

                current_cols = sam.sum(axis=0)
                for j in range(sam.shape[1]):
                    if target[j] <= 0.0:
                        sam[:, j] = 0.0
                    elif current_cols[j] > 0.0:
                        sam[:, j] *= target[j] / current_cols[j]

                max_diff_after = float(np.max(np.abs(sam.sum(axis=1) - sam.sum(axis=0))))
                iterations = step
                if max_diff_after <= tolerance:
                    converged = True
                    break

        return RASBalanceResult(
            matrix=pd.DataFrame(sam, index=matrix_df.index, columns=matrix_df.columns),
            target_totals=target,
            max_diff_before=max_diff_before,
            max_diff_after=max_diff_after,
            iterations=iterations,
            converged=converged,
            ras_type=mode.value,
        )
