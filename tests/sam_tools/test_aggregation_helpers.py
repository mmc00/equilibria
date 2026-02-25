from __future__ import annotations

from pathlib import Path

import pandas as pd

from equilibria.sam_tools.aggregation import (
    aggregate_dataframe,
    build_multiindex_labels,
    load_mapping,
)


def test_build_multiindex_labels_creates_multindex() -> None:
    labels = ["a", "b", "c"]
    multi_index, tuples = build_multiindex_labels(labels, category="RAW")
    assert multi_index.nlevels == 2
    assert tuples == [("RAW", "a"), ("RAW", "b"), ("RAW", "c")]


def test_aggregate_dataframe_groups_by_mapping() -> None:
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    mapping = {"a": "total", "b": "total"}
    ordered = ["total"]
    aggregated = aggregate_dataframe(df, mapping, ordered)
    assert aggregated.shape == (1, 1)
    assert aggregated.index[0] == "total"
    assert aggregated.iloc[0, 0] == 10.0


def test_load_mapping_requires_columns(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.xlsx"
    pd.DataFrame({"original": ["A"], "aggregated": ["TOTAL"]}).to_excel(
        mapping_path, sheet_name="mapping", index=False
    )
    mapping, ordered = load_mapping(mapping_path)
    assert mapping["a"] == "TOTAL"
    assert ordered == ["TOTAL"]
