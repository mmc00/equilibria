"""IEEM raw SAM support and PEP preprocessing operations for SAM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.models import SAMTransformState

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


def _load_mapping(mapping_path: Path) -> tuple[dict[str, str], list[str]]:
    mapping_df = pd.read_excel(mapping_path, sheet_name="mapping")
    required = {"original", "aggregated"}
    if not required.issubset(set(mapping_df.columns)):
        raise ValueError(f"Mapping file missing required columns: {sorted(required)}")

    mapping: dict[str, str] = {}
    ordered_aggregated: list[str] = []
    seen: set[str] = set()
    for _, row in mapping_df.iterrows():
        original = _norm_text(row["original"])
        aggregated = _norm_text(row["aggregated"])
        if not original or not aggregated:
            continue
        mapping[_norm_text_lower(original)] = aggregated
        if aggregated not in seen:
            seen.add(aggregated)
            ordered_aggregated.append(aggregated)

    if not mapping:
        raise ValueError(f"Mapping file has no usable rows: {mapping_path}")
    return mapping, ordered_aggregated


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


def _aggregate_with_mapping(
    matrix_df: pd.DataFrame,
    mapping: dict[str, str],
    ordered_aggregated: list[str],
) -> pd.DataFrame:
    def mapped(label: Any) -> str:
        label_txt = _norm_text(label)
        return mapping.get(_norm_text_lower(label_txt), label_txt)

    renamed = matrix_df.copy()
    renamed.index = [mapped(idx) for idx in matrix_df.index]
    renamed.columns = [mapped(col) for col in matrix_df.columns]

    aggregated = renamed.groupby(level=0).sum()
    aggregated = aggregated.T.groupby(level=0).sum().T

    ordered = [lab for lab in ordered_aggregated if lab in aggregated.index]
    for lab in aggregated.index:
        if lab not in ordered:
            ordered.append(lab)
    return aggregated.reindex(index=ordered, columns=ordered, fill_value=0.0)


def load_ieem_raw_excel_state(
    input_path: Path,
    *,
    sheet_name: str = "MCS2016",
) -> SAMTransformState:
    """Read an IEEM raw Excel SAM and return a RAW state."""
    raw_df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    groups = _detect_groups(raw_df)
    matrix_df = _extract_raw_sam_matrix(raw_df, groups)

    labels = [_norm_text(label) for label in matrix_df.index]
    matrix = matrix_df.to_numpy(dtype=float)
    return SAMTransformState(
        matrix=matrix,
        row_keys=[("RAW", label) for label in labels],
        col_keys=[("RAW", label) for label in labels],
        source_path=input_path,
        source_format="ieem_raw_excel",
        raw_df=None,
        data_start_row=None,
        data_start_col=None,
    )


class SAM:
    """Minimal SAM container for raw-IEEM preprocessing before PEP transforms."""

    def __init__(self, matrix: pd.DataFrame):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("SAM must be square")
        self.matrix = matrix.copy()
        self._balancer = RASBalancer()

    @classmethod
    def from_ieem_excel(cls, path: Path, sheet_name: str = "MCS2016") -> SAM:
        state = load_ieem_raw_excel_state(path, sheet_name=sheet_name)
        matrix_df = _state_to_dataframe(state, expected_category="RAW")
        return cls(matrix_df)

    def aggregate(self, mapping_path: Path) -> SAM:
        mapping, ordered = _load_mapping(mapping_path)
        self.matrix = _aggregate_with_mapping(self.matrix, mapping, ordered)
        return self

    def balance_ras(
        self,
        *,
        ras_type: str = "arithmetic",
        tolerance: float = 1e-9,
        max_iterations: int = 200,
    ) -> SAM:
        result = self._balancer.balance_dataframe(
            self.matrix,
            ras_type=ras_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        self.matrix = result.matrix
        return self

    def to_raw_state(self) -> SAMTransformState:
        labels = [_norm_text(label) for label in self.matrix.index]
        return SAMTransformState(
            matrix=self.matrix.to_numpy(dtype=float),
            row_keys=[("RAW", label) for label in labels],
            col_keys=[("RAW", label) for label in labels],
            source_path=Path("<memory>"),
            source_format="raw",
        )


def aggregate_state_with_mapping(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    mapping_path = op.get("mapping_path")
    if not mapping_path:
        raise ValueError("aggregate_mapping requires mapping_path")
    mapping, ordered = _load_mapping(Path(mapping_path))
    before_shape = list(state.matrix.shape)
    df = _state_to_dataframe(state, expected_category="RAW")
    aggregated = _aggregate_with_mapping(df, mapping, ordered)
    _replace_state_from_dataframe(state, aggregated)
    return {
        "mapping_path": str(mapping_path),
        "shape_before": before_shape,
        "shape_after": list(state.matrix.shape),
    }
