"""IEEM raw SAM support and PEP preprocessing operations for SAM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from equilibria.sam_tools.aggregation import build_multiindex_labels
from equilibria.sam_tools.models import Sam, SamTable

DEFAULT_IEEM_GROUP_ORDER: tuple[str, ...] = (
    "activities",
    "commodities",
    "margins",
    "factors",
    "households",
    "enterprises",
    "government",
    "row",
    "savings",
    "investment",
)

DEFAULT_IEEM_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "activities": ("actividades productivas",),
    "commodities": ("bienes y servicios",),
    "margins": ("margenes",),
    "factors": ("factores",),
    "households": ("hogares",),
    "enterprises": ("empresas",),
    "government": ("gobierno",),
    "row": ("resto del mundo",),
    "savings": ("ahorro",),
    "investment": ("inversion", "inversión"),
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


def _build_alias_lookup(group_aliases: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_group, aliases in group_aliases.items():
        for alias in aliases:
            key = _norm_text_lower(alias)
            if key in lookup and lookup[key] != canonical_group:
                raise ValueError(f"Alias '{alias}' maps to multiple groups")
            lookup[key] = canonical_group
    return lookup


def _groups_from_spec(
    raw_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_aliases: Mapping[str, Sequence[str]],
    group_col: int,
    label_col: int,
) -> list[_IEEMGroup]:
    if raw_df.shape[1] <= max(group_col, label_col):
        raise ValueError("IEEM raw SAM does not contain required group/label columns")

    alias_lookup = _build_alias_lookup(group_aliases)
    group_positions: dict[str, int] = {}
    for i, val in enumerate(raw_df.iloc[:, group_col]):
        if pd.isna(val):
            continue
        key = _norm_text_lower(val)
        canonical = alias_lookup.get(key)
        if canonical is None:
            continue
        if canonical not in group_positions:
            group_positions[canonical] = i

    missing = [group for group in group_order if group not in group_positions]
    if missing:
        raise ValueError(
            "Missing required IEEM groups in input sheet: "
            + ", ".join(missing)
        )

    positions = [group_positions[group] for group in group_order]
    if positions != sorted(positions):
        raise ValueError(
            "IEEM groups found but out of expected order. "
            f"Expected order: {list(group_order)}"
        )

    groups: list[_IEEMGroup] = []
    for idx, group in enumerate(group_order):
        start_row = group_positions[group]
        end_row = group_positions[group_order[idx + 1]] if idx + 1 < len(group_order) else len(raw_df)
        labels: list[str] = []
        for row in range(start_row, end_row):
            val = raw_df.iat[row, label_col]
            if pd.isna(val):
                continue
            label = _norm_text(val)
            if not label or _norm_text_lower(label) == "total":
                continue
            labels.append(label)
        if not labels:
            raise ValueError(f"Group '{group}' has no account labels in IEEM input")
        groups.append(_IEEMGroup(name=group, start_row=start_row, labels=labels))
    return groups


def _extract_raw_sam_matrix(
    raw_df: pd.DataFrame,
    groups: list[_IEEMGroup],
    *,
    data_start_col: int,
) -> pd.DataFrame:
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


def _parse_ieem_raw_matrix(
    input_path: Path,
    sheet_name: str,
    *,
    group_order: Sequence[str] = DEFAULT_IEEM_GROUP_ORDER,
    group_aliases: Mapping[str, Sequence[str]] = DEFAULT_IEEM_GROUP_ALIASES,
    group_col: int = 1,
    label_col: int = 2,
    data_start_col: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    groups = _groups_from_spec(
        raw_df,
        group_order=group_order,
        group_aliases=group_aliases,
        group_col=group_col,
        label_col=label_col,
    )
    matrix_df = _extract_raw_sam_matrix(raw_df, groups, data_start_col=data_start_col)
    labels = [_norm_text(label) for label in matrix_df.index]
    return raw_df, matrix_df, labels


def load_ieem_raw_excel_table(
    input_path: Path,
    *,
    sheet_name: str = "MCS2016",
    group_order: Sequence[str] | None = None,
    group_aliases: Mapping[str, Sequence[str]] | None = None,
    group_col: int | None = None,
    label_col: int | None = None,
    data_start_col: int | None = None,
) -> SamTable:
    """Load an IEEM raw workbook as ``SamTable`` using explicit parse spec."""
    parse_kwargs: dict[str, Any] = {}
    if group_order is not None:
        parse_kwargs["group_order"] = group_order
    if group_aliases is not None:
        parse_kwargs["group_aliases"] = group_aliases
    if group_col is not None:
        parse_kwargs["group_col"] = group_col
    if label_col is not None:
        parse_kwargs["label_col"] = label_col
    if data_start_col is not None:
        parse_kwargs["data_start_col"] = data_start_col

    sam = IEEMRawSAM.from_ieem_excel(
        path=input_path,
        sheet_name=sheet_name,
        **parse_kwargs,
    )
    return sam.to_table(source_path=input_path, source_format="ieem_raw_excel")


class IEEMRawSAM(Sam):
    """SAM helper especializado para cargar y transformar tablas IEEM sin procesar."""

    @classmethod
    def from_ieem_excel(
        cls,
        path: Path,
        sheet_name: str = "MCS2016",
        *,
        group_order: Sequence[str] | None = None,
        group_aliases: Mapping[str, Sequence[str]] | None = None,
        group_col: int | None = None,
        label_col: int | None = None,
        data_start_col: int | None = None,
    ) -> IEEMRawSAM:
        parse_kwargs: dict[str, Any] = {}
        if group_order is not None:
            parse_kwargs["group_order"] = group_order
        if group_aliases is not None:
            parse_kwargs["group_aliases"] = group_aliases
        if group_col is not None:
            parse_kwargs["group_col"] = group_col
        if label_col is not None:
            parse_kwargs["label_col"] = label_col
        if data_start_col is not None:
            parse_kwargs["data_start_col"] = data_start_col

        _raw_df, matrix_df, labels = _parse_ieem_raw_matrix(
            path,
            sheet_name,
            **parse_kwargs,
        )
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

    def to_table(
        self,
        *,
        source_path: Path | None = None,
        source_format: str = "raw",
    ) -> SamTable:
        return SamTable(
            sam=self,
            source_path=source_path or Path("<memory>"),
            source_format=source_format,
        )


def aggregate_table_with_mapping(table: SamTable, op: dict[str, Any]) -> dict[str, Any]:
    mapping_path = op.get("mapping_path")
    if not mapping_path:
        raise ValueError("aggregate_mapping requires mapping_path")
    before_shape = list(table.matrix.shape)
    table.sam.aggregate(Path(mapping_path))
    return {
        "mapping_path": str(mapping_path),
        "shape_before": before_shape,
        "shape_after": list(table.matrix.shape),
    }
