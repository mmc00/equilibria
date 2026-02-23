"""Balancing primitives for SAM matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

RASType = Literal["arithmetic", "geometric", "row", "column"]


@dataclass(frozen=True)
class RASBalanceResult:
    """Result payload for one RAS balancing run."""

    matrix: pd.DataFrame
    target_totals: np.ndarray
    max_diff_before: float
    max_diff_after: float
    iterations: int
    converged: bool
    ras_type: str


class RASBalancer:
    """RAS balancer with interchangeable target modes."""

    _ALIASES: dict[str, RASType] = {
        "arithmetic": "arithmetic",
        "mean": "arithmetic",
        "avg": "arithmetic",
        "arithmetic_mean": "arithmetic",
        "symmetric": "arithmetic",
        "geometric": "geometric",
        "geomean": "geometric",
        "geometric_mean": "geometric",
        "row": "row",
        "rows": "row",
        "column": "column",
        "col": "column",
        "cols": "column",
        "columns": "column",
    }

    def resolve_ras_type(self, ras_type: str | None) -> RASType:
        key = str(ras_type or "arithmetic").strip().lower()
        if key not in self._ALIASES:
            allowed = sorted(set(self._ALIASES.values()))
            raise ValueError(f"Unsupported ras_type '{ras_type}'. Allowed: {allowed}")
        return self._ALIASES[key]

    def _build_targets(
        self,
        row_totals: np.ndarray,
        col_totals: np.ndarray,
        ras_type: RASType,
    ) -> np.ndarray:
        if ras_type == "arithmetic":
            return 0.5 * (row_totals + col_totals)
        if ras_type == "row":
            return row_totals.copy()
        if ras_type == "column":
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
        ras_type: str = "arithmetic",
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> RASBalanceResult:
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
            ras_type=mode,
        )
