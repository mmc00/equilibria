from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from equilibria.sam_tools.models import Sam, SAMWorkflowConfig


def _sample_dataframe() -> pd.DataFrame:
    keys = [("I", "agr"), ("I", "ser")]
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    return pd.DataFrame(matrix, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))


def test_sam_requires_square_and_matching_accounts() -> None:
    df = _sample_dataframe()
    sam = Sam(dataframe=df)
    assert sam.row_keys == sam.col_keys
    assert np.allclose(sam.matrix, df.to_numpy(dtype=float))
    with np.testing.assert_raises(ValueError):
        Sam(dataframe=pd.DataFrame(np.ones((2, 3), dtype=float)))
    wrong_cols = df.copy()
    wrong_cols.columns = pd.MultiIndex.from_tuples([("A", "one"), ("B", "two")])
    with np.testing.assert_raises(ValueError):
        Sam(dataframe=wrong_cols)


def test_sam_update_matrix_enforces_shape() -> None:
    df = _sample_dataframe()
    sam = Sam(dataframe=df)
    new_matrix = np.array([[5.0, 6.0], [7.0, 8.0]])
    sam.update_matrix(new_matrix)
    assert np.allclose(sam.matrix, new_matrix)
    with np.testing.assert_raises(ValueError):
        sam.update_matrix(np.ones((3, 3), dtype=float))


def test_sam_aggregate_and_balance(tmp_path: Path) -> None:
    keys = [("RAW", "a"), ("RAW", "b")]
    matrix = np.array([[10.0, 5.0], [2.0, 8.0]], dtype=float)
    df = pd.DataFrame(matrix, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))
    sam = Sam(dataframe=df)

    mapping_path = tmp_path / "mapping.xlsx"
    mapping_df = pd.DataFrame(
        [
            {"original": "a", "aggregated": "TOTAL"},
            {"original": "b", "aggregated": "TOTAL"},
        ]
    )
    with pd.ExcelWriter(mapping_path, engine="openpyxl") as writer:
        mapping_df.to_excel(writer, sheet_name="mapping", index=False)

    sam.aggregate(mapping_path)
    assert {(cat, elem) for cat, elem in sam.row_keys} == {(cat, elem) for cat, elem in sam.col_keys}
    assert (sam.matrix.shape[0], sam.matrix.shape[1]) == (1, 1)

    # Unbalanced matrix -> run RAS and expect closure
    sam = Sam(dataframe=df)
    result = sam.balance_ras(ras_type="arithmetic", tolerance=1e-8, max_iterations=50)
    assert result.converged
    diff = np.abs(sam.matrix.sum(axis=0) - sam.matrix.sum(axis=1))
    assert np.all(diff <= 1e-6)


def test_sam_workflow_config_dataclass_small_fixture() -> None:
    assert issubclass(SAMWorkflowConfig, BaseModel)
    cfg = SAMWorkflowConfig(
        name="unit",
        country="cri",
        input_path=Path("/tmp/in.xlsx"),
        input_format="excel",
        output_path=Path("/tmp/out.gdx"),
        output_format="gdx",
        input_options={"sheet_name": "MCS2016"},
        transforms=[{"op": "scale_all", "factor": 0.001}],
        report_path=Path("/tmp/report.json"),
        output_symbol="SAM",
    )

    assert cfg.name == "unit"
    assert cfg.country == "cri"
    assert cfg.input_format == "excel"
    assert cfg.output_format == "gdx"
    assert cfg.input_options["sheet_name"] == "MCS2016"
    assert cfg.transforms[0]["op"] == "scale_all"
    assert cfg.output_symbol == "SAM"
