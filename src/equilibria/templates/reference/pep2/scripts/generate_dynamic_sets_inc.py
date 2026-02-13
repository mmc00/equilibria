#!/usr/bin/env python3
"""Generate dynamic GAMS set declarations for PEP2 from SAM Excel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _norm(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.lower()


def _unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_sets_from_sam(sam_xlsx: Path) -> dict[str, list[str]]:
    df = pd.read_excel(sam_xlsx, sheet_name="SAM", header=None)

    # Detect PEP SAM start (same logic as loader): first row with row category "L".
    data_start_row = 0
    for idx in range(len(df)):
        if _norm(df.iloc[idx, 0]) == "l":
            data_start_row = idx
            break
    data_start_row = max(data_start_row, 2)

    col_cat_row = data_start_row - 2
    col_elem_row = data_start_row - 1
    data_start_col = 2  # rdim=2 => first data col is C

    col_cats = [_norm(v) for v in df.iloc[col_cat_row, data_start_col:].tolist()]
    col_elems = [_norm(v) for v in df.iloc[col_elem_row, data_start_col:].tolist()]

    row_cats = [_norm(v) for v in df.iloc[data_start_row:, 0].tolist()]
    row_elems = [_norm(v) for v in df.iloc[data_start_row:, 1].tolist()]

    j = _unique([e for c, e in zip(row_cats, row_elems, strict=False) if c == "j"])
    i = _unique([e for c, e in zip(row_cats, row_elems, strict=False) if c == "i"])
    l = _unique([e for c, e in zip(row_cats, row_elems, strict=False) if c == "l"])
    k = _unique([e for c, e in zip(row_cats, row_elems, strict=False) if c == "k"])

    ag_raw = _unique([e for c, e in zip(row_cats, row_elems, strict=False) if c == "ag"])
    non_agent_accounts = {"td", "ti", "tm", "usk", "sk", "cap", "land"}
    ag = [a for a in ag_raw if a not in non_agent_accounts]

    h = [a for a in ag if a not in {"firm", "gvt", "row"}]
    f = [a for a in ag if a not in set(h) and a not in {"gvt", "row"}]
    agng = [a for a in ag if a != "gvt"]
    agd = [a for a in ag if a != "row"]
    i1 = [x for x in i if x != "agr"]

    return {
        "J": j,
        "I": i,
        "I1": i1,
        "L": l,
        "K": k,
        "AG": ag,
        "AGNG": agng,
        "AGD": agd,
        "H": h,
        "F": f,
    }


def _emit_set_block(name: str, desc: str, members: list[str], domain: str | None = None) -> str:
    dom = f"({domain})" if domain else ""
    lines = [f"{name}{dom} {desc}", "/"]
    for m in members:
        lines.append(f" {m}")
    lines.append("/")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: generate_dynamic_sets_inc.py <SAM.xlsx> <output.inc>", file=sys.stderr)
        return 2

    sam_xlsx = Path(sys.argv[1]).resolve()
    out_inc = Path(sys.argv[2]).resolve()

    if not sam_xlsx.exists():
        print(f"SAM Excel file not found: {sam_xlsx}", file=sys.stderr)
        return 1

    sets = _read_sets_from_sam(sam_xlsx)

    text = "\n".join(
        [
            "SET",
            _emit_set_block("J", "All industries dynamic", sets["J"]),
            "",
            _emit_set_block("I", "All commodities dynamic", sets["I"]),
            "",
            _emit_set_block("I1", "All commodities except agriculture dynamic", sets["I1"], "I"),
            "",
            _emit_set_block("L", "Labor categories dynamic", sets["L"]),
            "",
            _emit_set_block("K", "Capital categories dynamic", sets["K"]),
            "",
            _emit_set_block("AG", "All agents dynamic", sets["AG"]),
            "",
            _emit_set_block("AGNG", "Non governmental agents dynamic", sets["AGNG"], "AG"),
            "",
            _emit_set_block("AGD", "Domestic agents dynamic", sets["AGD"], "AG"),
            "",
            _emit_set_block("H", "Households dynamic", sets["H"], "AG"),
            "",
            _emit_set_block("F", "Firms dynamic", sets["F"], "AG"),
            ";",
            "",
        ]
    )
    out_inc.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
