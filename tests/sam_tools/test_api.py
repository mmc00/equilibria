from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from equilibria.sam_tools.api import run_ieem_to_pep
from equilibria.sam_tools.models import Sam
from tests.sam_tools.fixtures import IEEM_RAW_GROUPS, write_sample_ieem_raw_excel


def _write_mapping_template(path: Path) -> None:
    labels = [label for _, label in IEEM_RAW_GROUPS]
    aggregated = [
        "A-AGR",
        "C-AGR",
        "MARG",
        "USK",
        "HRP",
        "FIRM",
        "GVT",
        "ROW",
        "S-HH",
        "INV",
    ]
    mapping_df = pd.DataFrame({"original": labels, "aggregated": aggregated})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        mapping_df.to_excel(writer, sheet_name="mapping", index=False)


def test_run_ieem_to_pep_returns_sam_and_steps(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.xlsx"
    mapping_path = tmp_path / "mapping.xlsx"
    output_path = tmp_path / "sam_pep.xlsx"
    report_path = tmp_path / "report.json"

    matrix = np.zeros((len(IEEM_RAW_GROUPS), len(IEEM_RAW_GROUPS)), dtype=float)
    matrix[0, 1] = 10.0  # A-AGR -> C-AGR
    matrix[1, 7] = 10.0  # C-AGR -> ROW (exports)
    matrix[3, 1] = 5.0   # USK -> C-AGR
    matrix[2, 1] = 2.0   # MARG -> C-AGR
    write_sample_ieem_raw_excel(raw_path, matrix)
    _write_mapping_template(mapping_path)

    result = run_ieem_to_pep(
        raw_path,
        mapping_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert isinstance(result.sam, Sam)
    assert output_path.exists()
    assert report_path.exists()
    assert result.output_path == output_path.resolve()
    assert result.report_path == report_path.resolve()

    step_names = [step["step"] for step in result.steps]
    assert step_names == [
        "aggregate_mapping",
        "balance_ras",
        "normalize_pep_accounts",
        "create_x_block",
        "convert_exports",
        "align_ti",
        "move_k",
        "move_l",
        "move_margins",
        "move_tx",
    ]
    assert ("X", "agr") in result.sam.row_keys

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["sheet_name"] == "MCS2016"
    assert len(payload["steps"]) == 10
