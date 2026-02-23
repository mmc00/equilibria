from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _norm_text(value: str | Iterable[str]) -> str:
    if isinstance(value, tuple):
        candidate = value[-1]
        value = str(candidate)
    return " ".join(str(value).strip().split())


def _norm_text_lower(value: str | Iterable[str]) -> str:
    return _norm_text(value).lower()


def load_mapping(mapping_path: Path) -> Tuple[dict[str, str], list[str]]:
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


def aggregate_dataframe(
    matrix_df: pd.DataFrame,
    mapping: dict[str, str],
    ordered_aggregated: list[str],
) -> pd.DataFrame:
    def mapped(label: str) -> str:
        normalized = _norm_text(label)
        return mapping.get(_norm_text_lower(normalized), normalized)

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


def build_multiindex_labels(labels: Iterable[str], category: str = "RAW") -> Tuple[pd.MultiIndex, list[tuple[str, str]]]:
    normalized = [(_norm_text(category), _norm_text(label)) for label in labels]
    return pd.MultiIndex.from_tuples(normalized), normalized
