"""Data models for SAM workflow execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SAMTransformState:
    """In-memory SAM representation used during transformations."""

    matrix: np.ndarray
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]
    source_path: Path
    source_format: str
    raw_df: pd.DataFrame | None = None
    data_start_row: int | None = None
    data_start_col: int | None = None


@dataclass
class SAMWorkflowConfig:
    """Resolved workflow config from YAML."""

    name: str
    country: str | None
    input_path: Path
    input_format: str
    output_path: Path
    output_format: str
    transforms: list[dict[str, Any]]
    report_path: Path | None
    output_symbol: str
