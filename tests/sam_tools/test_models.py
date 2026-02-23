from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from equilibria.sam_tools.models import SAMTransformState, SAMWorkflowConfig


def test_sam_transform_state_dataclass_small_fixture() -> None:
    assert issubclass(SAMTransformState, BaseModel)
    state = SAMTransformState(
        matrix=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        row_keys=[("I", "agr"), ("I", "ser")],
        col_keys=[("AG", "gvt"), ("AG", "row")],
        source_path=Path("/tmp/source.xlsx"),
        source_format="excel",
        raw_df=None,
        data_start_row=None,
        data_start_col=None,
    )

    assert state.matrix.shape == (2, 2)
    assert state.row_keys[0] == ("I", "agr")
    assert state.col_keys[1] == ("AG", "row")
    assert state.source_format == "excel"


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
