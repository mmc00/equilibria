from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.ieem_to_pep_transformations import balance_state_ras
from equilibria.sam_tools.models import SAMTransformState


def _sample_matrix() -> pd.DataFrame:
    values = np.array(
        [
            [2.0, 3.0, 1.0],
            [4.0, 5.0, 2.0],
            [1.0, 2.0, 6.0],
        ],
        dtype=float,
    )
    labels = ["a", "b", "c"]
    return pd.DataFrame(values, index=labels, columns=labels)


def test_ras_balancer_modes_converge() -> None:
    balancer = RASBalancer()
    matrix = _sample_matrix()

    for ras_type in ("arithmetic", "geometric", "row", "column", "geomean"):
        result = balancer.balance_dataframe(
            matrix,
            ras_type=ras_type,
            tolerance=1e-10,
            max_iterations=1000,
        )
        assert result.converged
        assert result.max_diff_after <= 1e-8

        rows = result.matrix.sum(axis=1).to_numpy(dtype=float)
        cols = result.matrix.sum(axis=0).to_numpy(dtype=float)
        assert np.allclose(rows, cols, atol=1e-8)


def test_ras_balancer_row_and_column_targets() -> None:
    balancer = RASBalancer()
    matrix = _sample_matrix()
    row_totals = matrix.sum(axis=1).to_numpy(dtype=float)
    col_totals = matrix.sum(axis=0).to_numpy(dtype=float)

    result_row = balancer.balance_dataframe(
        matrix,
        ras_type="row",
        tolerance=1e-10,
        max_iterations=1000,
    )
    row_rows = result_row.matrix.sum(axis=1).to_numpy(dtype=float)
    row_cols = result_row.matrix.sum(axis=0).to_numpy(dtype=float)
    assert np.allclose(row_rows, row_totals, atol=1e-6)
    assert np.allclose(row_cols, row_totals, atol=1e-6)

    result_col = balancer.balance_dataframe(
        matrix,
        ras_type="column",
        tolerance=1e-10,
        max_iterations=1000,
    )
    col_rows = result_col.matrix.sum(axis=1).to_numpy(dtype=float)
    col_cols = result_col.matrix.sum(axis=0).to_numpy(dtype=float)
    assert np.allclose(col_rows, col_totals, atol=1e-6)
    assert np.allclose(col_cols, col_totals, atol=1e-6)


def test_balance_state_ras_uses_ras_type() -> None:
    matrix = np.array(
        [
            [2.0, 3.0, 1.0],
            [4.0, 5.0, 2.0],
            [1.0, 2.0, 6.0],
        ],
        dtype=float,
    )
    keys = [("RAW", "a"), ("RAW", "b"), ("RAW", "c")]
    state = SAMTransformState(
        matrix=matrix,
        row_keys=keys.copy(),
        col_keys=keys.copy(),
        source_path=Path("<test>"),
        source_format="raw",
    )

    details = balance_state_ras(
        state,
        {"ras_type": "column", "tol": 1e-10, "max_iter": 1000},
    )
    assert details["ras_type"] == "column"
    assert details["converged"]
    assert details["max_diff_after"] <= 1e-8
