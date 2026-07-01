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

from coverage_matrix import nl_rows, altertax_rows, gtap_solve_rows  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix.md"

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
        "measured snapshot (shock match% @ tol1% for the solver gates). `ci_status`: "
        "`ci` runs on ubuntu without a solver, `local` needs PATH+GAMS (run by hand), "
        "`blocked` has an unsound reference.",
        "",
        "## `.nl` coefficient gate (CI, no solver)",
        "",
        "Diffs Python vs GAMS Jacobian coefficients. Phases are base+shock, plus "
        "`check` (the CD multi-period step) where a `gams_check.nl` fixture exists "
        "(3x3/5x5/10x7). Contract: 0 coefficient diffs. ifSUB does not apply.",
        "",
        _table(nl_rows()),
        "",
        "## Altertax multi-period SOLVE gate (PATH, local-only)",
        "",
        "Builds + seeds + solves base→check→shock in altertax-CD mode; asserts "
        "3×code=1 and shock match ≥ gap_min, per ifSUB.",
        "",
        _table(altertax_rows()),
        "",
        "## Pure-gtap (real-CES) multi-period SOLVE gate (PATH, local-only)",
        "",
        "The non-altertax real-CES model solved base→check→shock in `mode=\"gtap\"`, "
        "per ifSUB, vs the GAMS LOCAL `out_gtap_shock_ifsub{0,1}.gdx`. Only gtap7_3x3 "
        "has these fixtures today. ifSUB=1's remaining gap is an open export-side link.",
        "",
        _table(gtap_solve_rows()),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
