"""IEEM raw SAM support and PEP preprocessing operations for SAM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from equilibria.sam_tools.aggregation import build_multiindex_labels
from equilibria.sam_tools.models import SAM, SAMTransformState

IEEM_GROUP_LABELS: dict[str, str] = {
    "actividades productivas": "activities",
    "bienes y servicios": "commodities",
    "margenes": "margins",
    "factores": "factors",
    "hogares": "households",
    "empresas": "enterprises",
    "gobierno": "government",
    "resto del mundo": "row",
    "ahorro": "savings",
    "inversion": "investment",
    "inversión": "investment",
}


@dataclass(frozen=True)
class _IEEMGroup:
    name: str
    start_row: int
    labels: list[str]


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _norm_text_lower(value: Any) -> str:
    return _norm_text(value).lower()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _detect_groups(raw_df: pd.DataFrame) -> list[_IEEMGroup]:
    if raw_df.shape[1] < 3:
        raise ValueError("IEEM raw SAM must have at least 3 columns")

    group_positions: dict[str, int] = {}
    for i, val in enumerate(raw_df.iloc[:, 1]):
        if pd.isna(val):
            continue
        text = _norm_text_lower(val)
        for label, name in IEEM_GROUP_LABELS.items():
            if label in text and name not in group_positions:
                group_positions[name] = i
                break

    if not group_positions:
        raise ValueError(
            "Could not detect IEEM account groups in column 2. "
            "Verify sheet/options for raw IEEM input."
        )

    sorted_groups = sorted(group_positions.items(), key=lambda item: item[1])
    groups: list[_IEEMGroup] = []
    for idx, (group_name, start_row) in enumerate(sorted_groups):
        end_row = (
            sorted_groups[idx + 1][1]
            if idx + 1 < len(sorted_groups)
            else len(raw_df)
        )
        labels: list[str] = []
        for row in range(start_row, end_row):
            val = raw_df.iat[row, 2]
            if pd.isna(val):
                continue
            label = _norm_text(val)
            if not label or _norm_text_lower(label) == "total":
                continue
            labels.append(label)
        if labels:
            groups.append(_IEEMGroup(name=group_name, start_row=start_row, labels=labels))

    if not groups:
        raise ValueError("Detected IEEM groups but no account labels were extracted")
    return groups


def _extract_raw_sam_matrix(raw_df: pd.DataFrame, groups: list[_IEEMGroup]) -> pd.DataFrame:
    data_start_col = 3
    group_sizes = [len(g.labels) for g in groups]
    n_accounts = int(sum(group_sizes))
    matrix = np.zeros((n_accounts, n_accounts), dtype=float)

    labels: list[str] = []
    for group in groups:
        labels.extend(group.labels)

    row_offset = 0
    for group_r in groups:
        for i, _ in enumerate(group_r.labels):
            row_idx = group_r.start_row + i
            col_offset = 0
            for g_idx, group_c in enumerate(groups):
                for j, _ in enumerate(group_c.labels):
                    col_idx = data_start_col + sum(group_sizes[:g_idx]) + j
                    if row_idx >= raw_df.shape[0] or col_idx >= raw_df.shape[1]:
                        continue
                    value = raw_df.iat[row_idx, col_idx]
                    if pd.isna(value):
                        continue
                    if _is_numeric(value):
                        matrix[row_offset + i, col_offset + j] = float(value)
                col_offset += len(group_c.labels)
        row_offset += len(group_r.labels)

    return pd.DataFrame(matrix, index=labels, columns=labels, dtype=float)


def _parse_ieem_raw_matrix(input_path: Path, sheet_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    groups = _detect_groups(raw_df)
    matrix_df = _extract_raw_sam_matrix(raw_df, groups)
    labels = [_norm_text(label) for label in matrix_df.index]
    return raw_df, matrix_df, labels


def _state_to_dataframe(state: SAMTransformState, expected_category: str) -> pd.DataFrame:
    if not state.row_keys or not state.col_keys:
        raise ValueError("State has no support keys")
    if any(cat != expected_category for cat, _ in state.row_keys):
        raise ValueError(f"State rows must be category '{expected_category}' for this operation")
    if any(cat != expected_category for cat, _ in state.col_keys):
        raise ValueError(f"State cols must be category '{expected_category}' for this operation")
    labels_r = [elem for _, elem in state.row_keys]
    labels_c = [elem for _, elem in state.col_keys]
    return pd.DataFrame(state.matrix.copy(), index=labels_r, columns=labels_c, dtype=float)


def _replace_state_from_dataframe(state: SAMTransformState, df: pd.DataFrame) -> None:
    labels = [_norm_text(label) for label in df.index]
    state.matrix = df.to_numpy(dtype=float)
    state.row_keys = [("RAW", label) for label in labels]
    state.col_keys = [("RAW", label) for label in labels]


def load_ieem_raw_excel_state(
    input_path: Path,
    *,
    sheet_name: str = "MCS2016",
) -> SAMTransformState:
    """Read an IEEM raw Excel SAM and return a RAW state."""
    raw_df, matrix_df, labels = _parse_ieem_raw_matrix(input_path, sheet_name)
    multi_index, keys = build_multiindex_labels(labels, category="RAW")
    matrix_clean = matrix_df.to_numpy(dtype=float)
    sam = SAM(dataframe=pd.DataFrame(matrix_clean, index=multi_index, columns=multi_index))

    return SAMTransformState(
        sam=sam,
        row_keys=keys,
        col_keys=keys,
        source_path=input_path,
        source_format="ieem_raw_excel",
        raw_df=None,
        data_start_row=None,
        data_start_col=None,
    )


class IEEMRawSAM(SAM):
    """SAM helper especializado para cargar y transformar tablas IEEM sin procesar."""

    @classmethod
    def from_ieem_excel(cls, path: Path, sheet_name: str = "MCS2016") -> IEEMRawSAM:
        raw_df, matrix_df, labels = _parse_ieem_raw_matrix(path, sheet_name)
        multi_index, _ = build_multiindex_labels(labels, category="RAW")
        return cls(dataframe=pd.DataFrame(matrix_df.to_numpy(dtype=float), index=multi_index, columns=multi_index))

    def aggregate(self, mapping_path: Path) -> IEEMRawSAM:
        super().aggregate(mapping_path)
        return self

    def balance_ras(
        self,
        *,
        ras_type: str = "arithmetic",
        tolerance: float = 1e-9,
        max_iterations: int = 200,
    ) -> IEEMRawSAM:
        super().balance_ras(
            ras_type=ras_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return self

    def to_raw_state(
        self,
        *,
        source_path: Path | None = None,
        source_format: str = "raw",
    ) -> SAMTransformState:
        keys = [(cat, elem) for cat, elem in self.row_keys]
        return SAMTransformState(
            sam=self,
            row_keys=keys,
            col_keys=keys,
            source_path=source_path or Path("<memory>"),
            source_format=source_format,
        )


def aggregate_state_with_mapping(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    mapping_path = op.get("mapping_path")
    if not mapping_path:
        raise ValueError("aggregate_mapping requires mapping_path")
    before_shape = list(state.matrix.shape)
    state.sam.aggregate(Path(mapping_path))
    state.refresh_keys()
    return {
        "mapping_path": str(mapping_path),
        "shape_before": before_shape,
        "shape_after": list(state.matrix.shape),
    }
