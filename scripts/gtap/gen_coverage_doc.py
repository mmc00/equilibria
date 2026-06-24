"""Generate docs/site/guide/gtap7_coverage_matrix.md from coverage_matrix.ROWS.

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

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->"
)


def _fmt_gap(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _row_cells(r) -> list[str]:
    ifsub = "—" if r.ifsub is None else str(r.ifsub)
    phases = ",".join(r.phases)
    gempack = r.note_gempack or "—"
    return [
        r.dataset, r.kind, r.solver, ifsub, phases,
        _fmt_gap(r.gap_min), r.gap_note, _fmt_gap(r.gap_gempack), gempack,
        r.ci_status, r.ref,
    ]


_HEADER = ["dataset", "kind", "solver", "ifsub", "phases",
           "gap_min", "gap_note", "gempack_min", "gempack_note", "ci_status", "ref"]


def _table(rows) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |",
             "|" + "|".join("---" for _ in _HEADER) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_row_cells(r)) + " |")
    return "\n".join(lines)


def _progress_section() -> str:
    from coverage_matrix import progress
    p = progress()
    return "\n".join([
        "## Progreso global",
        "",
        f"- total: {p['total']}",
        f"- done (≥99% / 0-diff): {p['done']}",
        f"- partial: {p['partial']}",
        f"- blocked: {p['blocked']}",
    ])


def _blocks_section() -> str:
    from coverage_matrix import BLOCKS
    lines = ["## Modularización (refactor a bloques — F3)", "",
             "| bloque | estado |", "|---|---|"]
    for b in BLOCKS:
        lines.append(f"| {b.name} | {b.status} |")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "# GTAP 7 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        "`gap_min` is the conservative floor the tests assert vs GAMS; "
        "`gempack_min` is the floor vs GEMPACK/RunGTAP; the `_note` columns are "
        "measured snapshots. `solver`: `mcp` (PATH) or `nlp` (walras/ifMCP=0). "
        "`ci_status`: `ci` runs on ubuntu without a solver, `local` needs "
        "PATH+GAMS (run by hand), `blocked` has an unsound reference.",
        "",
        _progress_section(),
        "",
        "## Single-period (`.nl` coefficient gate, CI, no solver)",
        "",
        _table(nl_rows()),
        "",
        "## Altertax multi-period (solver gate, local-only)",
        "",
        _table(altertax_rows()),
        "",
        _blocks_section(),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
