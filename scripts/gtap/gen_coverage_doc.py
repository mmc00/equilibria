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

from coverage_matrix import (  # noqa: E402
    nl_rows, altertax_rows, gtap_solve_rows, nlp_rows, mcp_rows,
)

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


_NLP_HEADER = ["dataset", "ifsub", "base ≥", "check ≥", "shock ≥", "ref"]


def _nlp_table(rows) -> str:
    """Per-stage floor table for the NLP gate. Columns are the CONTRACT floors the
    test asserts per stage (base/check/shock); the measured match% is produced by
    running the test, not stored here."""
    lines = ["| " + " | ".join(_NLP_HEADER) + " |",
             "|" + "|".join("---" for _ in _NLP_HEADER) + "|"]
    for r in rows:
        floors = dict(r.stage_floors)
        lines.append("| " + " | ".join([
            r.dataset, str(r.ifsub),
            f"{floors['base']:g}", f"{floors['check']:g}", f"{floors['shock']:g}",
            r.ref,
        ]) + " |")
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
        "per ifSUB, vs the GAMS LOCAL `out_gtap_shock_ifsub{0,1}.gdx` (gtap7_3x3 local, "
        "gtap7_5x5 via NEOS). All four cases are at parity: the sluggish factor price "
        "pft was wrongly FIXED by fix_sluggish_pft (via a nonexistent xftflag Param), "
        "freezing it against the tariff shock. Freeing it (eq_xfteq.active guard) + "
        "removing the 3x3-hardcoded eq_pfyeq trim (the matcher squares automatically) "
        "closed gtap7_5x5 ifSUB=0 64.87→100% and improved every case (commit f570e32).",
        "",
        _table(gtap_solve_rows()),
        "",
        "## NLP-vs-NLP fidelity gate (IPOPT both sides, local-only)",
        "",
        "Python is solved as an NLP (`EQUILIBRIA_GTAP_SOLVE_NLP=1`, maximize walras) "
        "against the GAMS `ifMCP=0` NLP reference. Same IPOPT on both sides, so the "
        "solver's equality tolerance cancels and the cell-by-cell match reflects "
        "**model fidelity**, not solver noise. Unlike the other gates this reports a "
        "floor **per stage** (base/check/shock): `test_gtap7_nlp_parity.py` runs the "
        "real solve, measures match% @ tol1% and the return code, and asserts "
        "`match ≥ floor` and `code == 1` for every stage. The measured snapshot is "
        "**not** stored in the matrix (it would be a dead copy) — regenerate the rich "
        "view with `scripts/gtap/gen_nlp_matrix_page.py`, which re-runs the "
        "measurement. See [the live matrix](../_static/gtap7_nlp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        "100% across every stage, both ifSUB, after the Jacobian pre-scale skip "
        "(commit e4c40d7 — GAMS solves the raw model; the Python-only pre-scale steered "
        "IPOPT to a wrong basin on the 5×5 shock, 59.56% → 100%).",
        "",
        _nlp_table([r for r in nlp_rows() if r.mode == "pure"]),
        "",
        "### Altertax (CD)",
        "",
        "Base is exact (100%). The check/shock floors are lower because the altertax "
        "NLP references are themselves mis-converged — IPOPT stops at \"Locally "
        "Optimal\" and the ref violates its own `eq_pxeq` in the ag sector. Where a "
        "cleanly-converged MCP reference exists (3×3 ifSUB=1) the same Python solve "
        "reaches 99.93%; the path to 99% for the rest is MCP references, not a code "
        "change. Every stage still converges (`code == 1`).",
        "",
        _nlp_table([r for r in nlp_rows() if r.mode == "altertax"]),
        "",
        "## MCP fidelity gate (PATH both sides, local-only)",
        "",
        "Python is solved via PATH (nonlinear-full MCP) against the cleanly-converged "
        "**NEOS** MCP reference (regenerated 2026-07-17, subsidy-aware, `eq_pxeq` clean). "
        "Same per-stage contract as the NLP gate; `test_gtap7_mcp_parity.py` runs the "
        "real PATH solve, measures match%/code, and asserts `match ≥ floor` and "
        "`code == 1` for every stage. With clean refs the match is 99%+ everywhere "
        "(base/check exact, shock ≥99.3 except 15×10's known eq_paa Armington micro-cell "
        "family ~95%) — the ~89–97 the NLP gate reads is the mis-converged NLP ref, NOT "
        "the model. See [the live matrix](../_static/gtap7_mcp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        "100% across every stage on 3×3/5×5/10×7 (both ifSUB) against the NEOS-regenerated "
        "`out_gtap_shock_ifsub{N}.gdx` refs; 15×10 shock ~95% is the same eq_paa Armington "
        "micro-cell family. This is the symmetric counterpart of the pure-gtap NLP gate — "
        "the earlier ~63% was a subsidy-blind (`ytax[ft]=0`) reference, not the model.",
        "",
        _nlp_table([r for r in mcp_rows() if r.mode == "pure"]),
        "",
        "### Altertax (CD)",
        "",
        "Base/check exact, shock ≥99.3 (15×10 ~95% eq_paa). The cleanly-converged MCP "
        "references make this the daily fidelity gate — where the NLP gate's mis-converged "
        "ref capped the match at 89–97%, PATH against a clean MCP ref reaches 99%+.",
        "",
        _nlp_table([r for r in mcp_rows() if r.mode == "altertax"]),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
