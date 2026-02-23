from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from equilibria.sam_tools.ieem_raw_excel import IEEMRawSAM
from equilibria.sam_tools.models import Sam


def _write_ieem_raw_excel(path: Path, matrix: np.ndarray, sheet_name: str = "MCS2016") -> None:
    groups = [
        ("actividades productivas", "act_agr"),
        ("Bienes y servicios", "com_agr"),
        ("Margenes", "marg"),
        ("Factores", "usk"),
        ("Hogares", "hh"),
        ("Empresas", "firm"),
        ("Gobierno", "gvt"),
        ("Resto del mundo", "row"),
        ("Ahorro", "s_hh"),
        ("Inversión", "inv"),
    ]
    if matrix.shape != (len(groups), len(groups)):
        raise ValueError("IEEM raw test matrix has invalid shape")

    n_rows = 4 + len(groups) + 5
    n_cols = 3 + len(groups) + 5
    raw = np.full((n_rows, n_cols), np.nan, dtype=object)
    start_row = 4

    for i, (group_name, label) in enumerate(groups):
        row = start_row + i
        raw[row, 1] = group_name
        raw[row, 2] = label

    for i in range(len(groups)):
        for j in range(len(groups)):
            raw[start_row + i, 3 + j] = float(matrix[i, j])

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(raw).to_excel(writer, sheet_name=sheet_name, index=False, header=False)


def _write_ieem_mapping(path: Path) -> None:
    mapping = pd.DataFrame(
        [
            {"original": "act_agr", "aggregated": "A-AGR"},
            {"original": "com_agr", "aggregated": "C-AGR"},
            {"original": "marg", "aggregated": "MARG"},
            {"original": "usk", "aggregated": "USK"},
            {"original": "hh", "aggregated": "HRP"},
            {"original": "firm", "aggregated": "FIRM"},
            {"original": "gvt", "aggregated": "GVT"},
            {"original": "row", "aggregated": "ROW"},
            {"original": "s_hh", "aggregated": "S-HH"},
            {"original": "inv", "aggregated": "INV"},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        mapping.to_excel(writer, sheet_name="mapping", index=False)


def test_ieem_raw_sam_init_requires_square_matrix() -> None:
    assert issubclass(IEEMRawSAM, Sam)
    with np.testing.assert_raises(ValueError):
        IEEMRawSAM(dataframe=pd.DataFrame(np.ones((2, 3), dtype=float)))


def test_sam_raw_pipeline_methods_with_small_fixture(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.xlsx"
    mapping_file = tmp_path / "mapping.xlsx"

    matrix = np.zeros((10, 10), dtype=float)
    matrix[0, 1] = 100.0
    matrix[1, 7] = 30.0
    matrix[1, 9] = 15.0
    matrix[8, 4] = 20.0
    _write_ieem_raw_excel(raw_file, matrix)
    _write_ieem_mapping(mapping_file)

    sam = IEEMRawSAM.from_ieem_excel(raw_file, sheet_name="MCS2016")
    assert sam.matrix.shape == (10, 10)

    sam.aggregate(mapping_file)
    aggregated_df = sam.to_dataframe()
    row_labels = {elem.lower() for _, elem in aggregated_df.index}
    col_labels = {elem.lower() for _, elem in aggregated_df.columns}
    assert "a-agr" in row_labels
    assert "c-agr" in col_labels

    before = float(np.max(np.abs(sam.matrix.sum(axis=1) - sam.matrix.sum(axis=0))))
    sam.balance_ras(ras_type="geometric", tolerance=1e-10, max_iterations=1000)
    after = float(np.max(np.abs(sam.matrix.sum(axis=1) - sam.matrix.sum(axis=0))))
    assert after <= before
    assert after <= 1e-8

    state = sam.to_raw_state()
    assert state.matrix.shape == sam.matrix.shape
    assert all(cat == "RAW" for cat, _ in state.row_keys)
    assert all(cat == "RAW" for cat, _ in state.col_keys)
