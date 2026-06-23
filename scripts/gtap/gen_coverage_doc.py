"""Generate docs/gtap7_coverage_matrix.md from coverage_matrix.ROWS.

Run:  uv run python scripts/gtap/gen_coverage_doc.py
The output is a committed golden file; test_coverage_doc_sync enforces that the
committed file equals render() (CI fails on drift).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))

from coverage_matrix import nl_rows, altertax_rows  # noqa: E402

DOC_PATH = ROOT / "docs/gtap7_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->"
)


def _fmt_gap_min(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _row_cells(r) -> list[str]:
    ifsub = "—" if r.ifsub is None else str(r.ifsub)
    phases = ",".join(r.phases)
    return [
        r.dataset, r.kind, ifsub, phases,
        _fmt_gap_min(r.gap_min), r.gap_note, r.ci_status, r.ref,
    ]


_HEADER = ["dataset", "kind", "ifsub", "phases", "gap_min", "gap_note", "ci_status", "ref"]


def _table(rows) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |",
             "|" + "|".join("---" for _ in _HEADER) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_row_cells(r)) + " |")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "# GTAP 7 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        "`gap_min` is the conservative floor the tests assert; `gap_note` is the "
        "measured snapshot. `ci_status`: `ci` runs on ubuntu without a solver, "
        "`local` needs PATH+GAMS (run by hand), `blocked` has an unsound reference.",
        "",
        "## Single-period (`.nl` coefficient gate, CI, no solver)",
        "",
        _table(nl_rows()),
        "",
        "## Altertax multi-period (solver gate, local-only)",
        "",
        _table(altertax_rows()),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
