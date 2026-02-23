"""Data models for SAM workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SAMTransformState(BaseModel):
    """In-memory SAM representation used during transformations."""

    matrix: np.ndarray
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]
    source_path: Path
    source_format: str
    raw_df: pd.DataFrame | None = None
    data_start_row: int | None = None
    data_start_col: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_matrix_shape(self) -> SAMTransformState:
        rows, cols = self.matrix.shape
        if rows != len(self.row_keys):
            raise ValueError(
                f"matrix rows ({rows}) do not match row_keys ({len(self.row_keys)})"
            )
        if cols != len(self.col_keys):
            raise ValueError(
                f"matrix cols ({cols}) do not match col_keys ({len(self.col_keys)})"
            )
        return self


class SAMWorkflowConfig(BaseModel):
    """Resolved workflow config from YAML."""

    name: str
    country: str | None
    input_path: Path
    input_format: str
    output_path: Path
    output_format: str
    input_options: dict[str, Any] = Field(default_factory=dict)
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    report_path: Path | None
    output_symbol: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
