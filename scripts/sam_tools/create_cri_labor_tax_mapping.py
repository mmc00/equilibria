#!/usr/bin/env python3
"""Create a CRI mapping variant where social contributions map to taxes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


CONTRIBUTION_LABELS = [
    "Contribución seguro social- trabajo con calificación baja",
    "Contribución seguro social- trabajo con calificación media",
    "Constribución seguro social- trabajo con calificacion alta",
]


def parse_args() -> argparse.Namespace:
    cge_babel_root = os.environ.get("CGE_BABEL_ROOT")
    if cge_babel_root:
        root = Path(cge_babel_root) / "sam" / "cri" / "2016" / "output"
        default_input = root / "mapping_template.xlsx"
        default_output = root / "mapping_template_contrib_to_ti.xlsx"
    else:
        default_input = Path("output/mapping_template.xlsx")
        default_output = Path("output/mapping_template_contrib_to_ti.xlsx")

    parser = argparse.ArgumentParser(
        description="Build mapping variant moving social contributions to a tax account",
    )
    parser.add_argument(
        "--input-mapping",
        type=Path,
        default=default_input,
        help="Source mapping_template.xlsx path",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=default_output,
        help="Output mapping path",
    )
    parser.add_argument(
        "--target-tax",
        choices=["TI", "TD"],
        default="TI",
        help="Tax bucket for contribution accounts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_mapping.exists():
        raise FileNotFoundError(f"Input mapping not found: {args.input_mapping}")

    df = pd.read_excel(args.input_mapping, sheet_name="mapping")
    if not {"original", "aggregated", "group"}.issubset(df.columns):
        raise ValueError("Mapping sheet must contain columns: original, aggregated, group")

    before = df[df["original"].isin(CONTRIBUTION_LABELS)][["original", "aggregated", "group"]].copy()
    if len(before) != len(CONTRIBUTION_LABELS):
        found = before["original"].tolist()
        missing = [x for x in CONTRIBUTION_LABELS if x not in found]
        raise ValueError(f"Could not find all contribution labels in mapping. Missing: {missing}")

    mask = df["original"].isin(CONTRIBUTION_LABELS)
    df.loc[mask, "aggregated"] = args.target_tax
    df.loc[mask, "group"] = "factors"

    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_mapping, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="mapping", index=False)

    after = df[df["original"].isin(CONTRIBUTION_LABELS)][["original", "aggregated", "group"]]

    print(f"Input mapping: {args.input_mapping}")
    print(f"Output mapping: {args.output_mapping}")
    print(f"Target tax bucket: {args.target_tax}")
    print("")
    print("Changed rows:")
    for _, row in before.iterrows():
        orig = row["original"]
        old = row["aggregated"]
        new = after.loc[after["original"] == orig, "aggregated"].iloc[0]
        print(f"- {orig}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
