"""GTAP equilibrium snapshots derived from COMP outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _build_key(
    region: str, sector: str | float, qualifier: str | float
) -> tuple[str, ...]:
    parts: list[str] = [region]
    if isinstance(sector, str) and sector:
        parts.append(sector)
    if isinstance(qualifier, str) and qualifier:
        parts.append(qualifier)
    return tuple(parts)


@dataclass
class GTAPEquilibriumSnapshot:
    """Snapshot of equilibrium values extracted from COMP outputs."""

    values: dict[str, dict[tuple[str, ...], float]]

    @classmethod
    def from_csv(cls, csv_path: Path, year: int = 1) -> GTAPEquilibriumSnapshot:
        df = pd.read_csv(csv_path)
        if year is not None:
            df = df[df["Year"] == year]

        entries: dict[str, dict[tuple[str, ...], float]] = defaultdict(dict)
        for row in df.itertuples(index=False):
            region = str(getattr(row, "Region", ""))
            if not region:
                continue
            key = _build_key(
                region, getattr(row, "Sector", ""), getattr(row, "Qualifier", "")
            )
            entries[row.Variable][key] = float(row.Value)

        return cls(dict(entries))

    def get(self, variable: str) -> dict[tuple[str, ...], float]:
        return self.values.get(variable, {})
