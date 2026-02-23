from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from equilibria.babel.gdx.reader import read_gdx
from equilibria.sam_tools.pipeline import run_sam_transform_workflow
from equilibria.templates.pep_sam_compat import load_sam_grid


def _write_canonical_excel(
    path: Path,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
    matrix: np.ndarray,
) -> None:
    grid = np.full((len(row_keys) + 2, len(col_keys) + 2), "", dtype=object)

    for j, (cat, elem) in enumerate(col_keys):
        grid[0, 2 + j] = cat
        grid[1, 2 + j] = elem

    for i, (cat, elem) in enumerate(row_keys):
        grid[2 + i, 0] = cat
        grid[2 + i, 1] = elem
        for j in range(len(col_keys)):
            grid[2 + i, 2 + j] = float(matrix[i, j])

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(grid).to_excel(writer, sheet_name="SAM", index=False, header=False)


def _cell(
    matrix: np.ndarray,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
    row_key: tuple[str, str],
    col_key: tuple[str, str],
) -> float:
    ri = row_keys.index(row_key)
    ci = col_keys.index(col_key)
    return float(matrix[ri, ci])


def test_yaml_pipeline_scale_and_shift(tmp_path: Path) -> None:
    input_sam = tmp_path / "input.xlsx"
    output_sam = tmp_path / "output.xlsx"
    report_json = tmp_path / "report.json"
    config_file = tmp_path / "workflow.yaml"

    row_keys = [("AG", "tx"), ("AG", "ti"), ("I", "ser")]
    col_keys = [("I", "agr"), ("I", "ser"), ("AG", "row")]
    matrix = np.array(
        [
            [10.0, 20.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    _write_canonical_excel(input_sam, row_keys, col_keys, matrix)

    config = {
        "metadata": {"name": "unit_scale_shift", "country": "tst"},
        "input": {"path": str(input_sam), "format": "excel"},
        "output": {"path": str(output_sam), "format": "excel"},
        "report_path": str(report_json),
        "transforms": [
            {"op": "scale_all", "factor": 2.0},
            {"op": "scale_slice", "row": "AG.tx", "col": "I.*", "factor": 0.5},
            {
                "op": "shift_row_slice",
                "source_row": "AG.tx",
                "target_row": "AG.ti",
                "col": "I.*",
                "share": 1.0,
            },
        ],
    }
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_sam_transform_workflow(config_file)
    out_grid = load_sam_grid(output_sam)

    assert report["summary"]["steps"] == 3
    assert report_json.exists()

    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "tx"), ("I", "agr")) == 0.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "tx"), ("I", "ser")) == 0.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "ti"), ("I", "agr")) == 12.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "ti"), ("I", "ser")) == 24.0


def test_yaml_pipeline_writes_gdx(tmp_path: Path) -> None:
    input_sam = tmp_path / "input.xlsx"
    output_sam = tmp_path / "output.gdx"
    config_file = tmp_path / "workflow.yaml"

    row_keys = [("AG", "tx")]
    col_keys = [("I", "agr")]
    matrix = np.array([[3.5]], dtype=float)
    _write_canonical_excel(input_sam, row_keys, col_keys, matrix)

    config = {
        "metadata": {"name": "unit_to_gdx"},
        "input": {"path": str(input_sam), "format": "excel"},
        "output": {"path": str(output_sam), "format": "gdx", "symbol": "SAM"},
        "transforms": [],
    }
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_sam_transform_workflow(config_file)

    gdx = read_gdx(output_sam)
    assert len(gdx["symbols"]) == 1
    assert gdx["symbols"][0]["name"] == "SAM"
    assert gdx["symbols"][0]["dimension"] == 4
    assert gdx["symbols"][0]["records"] == 1


def test_yaml_pipeline_disaggregated_pep_structural_moves(tmp_path: Path) -> None:
    input_sam = tmp_path / "input.xlsx"
    output_sam = tmp_path / "output.xlsx"
    config_file = tmp_path / "workflow.yaml"

    row_keys = [
        ("K", "cap"),
        ("J", "agr"),
        ("AG", "ti"),
        ("AG", "tx"),
        ("MARG", "MARG"),
        ("I", "ser"),
    ]
    col_keys = [("I", "agr")]
    matrix = np.array(
        [
            [5.0],
            [1.0],
            [4.0],
            [2.0],
            [3.0],
            [0.0],
        ],
        dtype=float,
    )
    _write_canonical_excel(input_sam, row_keys, col_keys, matrix)

    config = {
        "metadata": {"name": "unit_pep_moves"},
        "input": {"path": str(input_sam), "format": "excel"},
        "output": {"path": str(output_sam), "format": "excel"},
        "transforms": [
            {"op": "move_k_to_ji", "commodity_to_sector": {"agr": "agr"}},
            {"op": "move_l_to_ji", "commodity_to_sector": {"agr": "agr"}},
            {"op": "move_margin_to_i_margin", "margin_commodity": "ser"},
            {"op": "move_tx_to_ti_on_i"},
        ],
    }
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_sam_transform_workflow(config_file)

    out_grid = load_sam_grid(output_sam)
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("K", "cap"), ("I", "agr")) == 0.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "tx"), ("I", "agr")) == 0.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("MARG", "MARG"), ("I", "agr")) == 0.0

    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("J", "agr"), ("I", "agr")) == 6.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("AG", "ti"), ("I", "agr")) == 6.0
    assert _cell(out_grid.matrix, out_grid.row_keys, out_grid.col_keys, ("I", "ser"), ("I", "agr")) == 3.0


def test_yaml_pipeline_composite_matches_disaggregated(tmp_path: Path) -> None:
    input_sam = tmp_path / "input.xlsx"
    output_composite = tmp_path / "output_composite.xlsx"
    output_disaggregated = tmp_path / "output_disaggregated.xlsx"
    cfg_composite = tmp_path / "composite.yaml"
    cfg_disaggregated = tmp_path / "disaggregated.yaml"

    row_keys = [
        ("K", "cap"),
        ("L", "usk"),
        ("J", "agr"),
        ("AG", "ti"),
        ("AG", "tx"),
        ("MARG", "MARG"),
        ("I", "ser"),
    ]
    col_keys = [
        ("I", "agr"),
        ("I", "ser"),
        ("AG", "row"),
        ("AG", "gvt"),
        ("J", "agr"),
        ("MARG", "MARG"),
        ("OTH", "INV"),
    ]
    matrix = np.array(
        [
            [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    _write_canonical_excel(input_sam, row_keys, col_keys, matrix)

    composite_cfg = {
        "metadata": {"name": "composite"},
        "input": {"path": str(input_sam), "format": "excel"},
        "output": {"path": str(output_composite), "format": "excel"},
        "transforms": [
            {
                "op": "pep_structural_moves",
                "margin_commodity": "ser",
                "commodity_to_sector": {"agr": "agr", "ser": "agr"},
            }
        ],
    }
    cfg_composite.write_text(yaml.safe_dump(composite_cfg), encoding="utf-8")

    disaggregated_cfg = {
        "metadata": {"name": "disaggregated"},
        "input": {"path": str(input_sam), "format": "excel"},
        "output": {"path": str(output_disaggregated), "format": "excel"},
        "transforms": [
            {
                "op": "move_k_to_ji",
                "commodity_to_sector": {"agr": "agr", "ser": "agr"},
            },
            {
                "op": "move_l_to_ji",
                "commodity_to_sector": {"agr": "agr", "ser": "agr"},
            },
            {"op": "move_margin_to_i_margin", "margin_commodity": "ser"},
            {"op": "move_tx_to_ti_on_i"},
        ],
    }
    cfg_disaggregated.write_text(yaml.safe_dump(disaggregated_cfg), encoding="utf-8")

    run_sam_transform_workflow(cfg_composite)
    run_sam_transform_workflow(cfg_disaggregated)

    composite_out = load_sam_grid(output_composite)
    disaggregated_out = load_sam_grid(output_disaggregated)
    assert np.allclose(composite_out.matrix, disaggregated_out.matrix)
