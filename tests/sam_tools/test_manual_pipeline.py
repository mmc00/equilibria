from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equilibria.sam_tools.manual_pipeline import run_from_excel
from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.sam_transforms import create_x_block_on_sam, convert_exports_to_x_on_sam
from tests.sam_tools.fixtures import IEEM_RAW_GROUPS, write_sample_ieem_raw_excel


def test_manual_pipeline_runs_on_fixture() -> None:
    raw_path = Path("tmp_ieem_test/raw.xlsx")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(IEEM_RAW_GROUPS), len(IEEM_RAW_GROUPS)), dtype=float)
    write_sample_ieem_raw_excel(raw_path, matrix)

    summary = run_from_excel(raw_path, sheet_name="MCS2016")

    step_names = [step["step"] for step in summary.steps]
    assert "normalize_accounts" in step_names
    assert step_names[-1] == "balance_ras"
    assert summary.balance_stats["max_row_col_diff"] == 0.0
    assert summary.balance_stats["rows"] == summary.balance_stats["columns"]
    assert summary.total_flow == 0.0


def test_create_x_block_adds_export_accounts() -> None:
    keys = [("I", "agr"), ("I", "ser")]
    matrix = np.zeros((2, 2), dtype=float)
    df = pd.DataFrame(matrix, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))
    sam = Sam(dataframe=df)

    result = create_x_block_on_sam(sam)
    assert ("X", "agr") in sam.row_keys
    assert ("X", "ser") in sam.row_keys
    assert result["commodities"] == 2
    assert result["added_x_accounts"] == 2


def test_convert_exports_moves_value_from_i_to_x() -> None:
    keys = [("I", "agr"), ("X", "agr"), ("J", "agr"), ("AG", "row")]
    n = len(keys)
    matrix = np.zeros((n, n), dtype=float)
    df = pd.DataFrame(matrix, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))
    df.loc[("I", "agr"), ("AG", "row")] = 20.0
    df.loc[("J", "agr"), ("I", "agr")] = 20.0

    sam = Sam(dataframe=df)
    result = convert_exports_to_x_on_sam(sam)

    assert result["converted_commodities"] == 1
    assert np.isclose(sam.to_dataframe().loc[("I", "agr"), ("AG", "row")], 0.0)
    assert np.isclose(sam.to_dataframe().loc[("X", "agr"), ("AG", "row")], 20.0)
    assert np.isclose(sam.to_dataframe().loc[("J", "agr"), ("I", "agr")], 0.0)
    assert np.isclose(sam.to_dataframe().loc[("J", "agr"), ("X", "agr")], 20.0)
