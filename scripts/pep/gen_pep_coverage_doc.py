"""Generate docs/site/guide/pep_coverage_matrix.md from pep_coverage_matrix.ROWS.

Run:  uv run python scripts/pep/gen_pep_coverage_doc.py
The output is a committed golden file; test_pep_coverage_doc_in_sync enforces that the
committed file equals render() (CI fails on drift). Mirrors gen_coverage_doc.py (GTAP).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pep"))
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))

from pep_coverage_matrix import (  # noqa: E402
    DATASET, nlp_rows, mcp_rows, mirror_rows, variant_rows,
)
import matrix_html as mx  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/pep_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/pep/pep_coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/pep/gen_pep_coverage_doc.py -->"
)


def _match_tone(match_note: str) -> str:
    return "good" if match_note.startswith("100") else "warn"


def _pep_table(rows) -> str:
    headers = ["Scenario · form", "Cells", "Match", "Gate", "Reference"]
    body = [
        [
            mx.label(r.scenario, r.form),
            str(r.cells),
            mx.cell(
                mx.num(r.match_note, _match_tone(r.match_note)),
                mx.chip("local", "neutral"),
            ),
            mx.ref(r.gate),
            mx.ref(r.ref),
        ]
        for r in rows
    ]
    return mx.tablecard(headers, body)


_LEGEND = mx.legend(
    '<span class="mx-li"><b>match</b> is the measured snapshot; the named '
    "pytest gate re-measures and enforces it</span>"
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " every row needs PATH/IPOPT + GAMS refs — no solver-free CI gate "
    "(the PEP reference is a solved GDX, not a .nl dump)</span>"
)


def render() -> str:
    parts = [
        "# PEP-1-1 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        f"Coverage for the `{DATASET}` dataset — the PEP-1-1 v2.1 CGE model "
        "ported to Pyomo (`equilibria.templates.pep_pyomo`). Every row "
        "compares the Pyomo solve against a GAMS reference solved by the "
        "**same engine** (IPOPT vs IPOPT, or PATH vs PATH), so the solver's "
        "equality tolerance cancels and the cell-by-cell match reflects "
        "**model fidelity**, not solver noise.",
        "",
        mx.raw(_LEGEND),
        "",
        "## NLP vs NLP",
        "",
        "The Pyomo model solved as an NLP (IPOPT on the raw model, "
        "`nlp_scaling_method=none`, faithful to GAMS's raw solve) against the "
        "GAMS CNS reference `Results.gdx`. The benchmark BASE reproduces the "
        "SAM, so the seeded point is the calibration answer. Run: "
        "`phase1_nlp.py --model pep --dataset pep2 --period BASE`.",
        "",
        mx.raw(_pep_table(nlp_rows())),
        "",
        "## MCP vs MCP",
        "",
        "The Pyomo model solved as a complementarity problem via PATH against "
        "the **GAMS-native** MCP reference (`PEP-1-1_v2_1_mcp_solve.gms`: "
        "`MODEL /ALL/` + `SOLVE USING MCP`, so GAMS infers the "
        "equation↔variable pairing). The `sim1` row is the reference "
        "counterfactual — a 25% export-tax cut (`ttix.fx=ttixO*0.75`) — "
        "applied faithfully in Python by scaling the `ttixO` benchmark before "
        "build. Run: `phase1_nlp.py --model pep --dataset pep2 --form mcp`.",
        "",
        mx.raw(_pep_table(mcp_rows())),
        "",
        "## NLP↔MCP mirror",
        "",
        "The two Pyomo forms, solved from the same feasible benchmark seed, "
        "land on the identical point (LEON, the form-defining Walras slack, "
        "is excluded). The one historical gap — `PD['othind']`, filled from "
        "its `*O` benchmark (1.132) rather than a blind 1.0 — closed the "
        "mirror to a clean 100%.",
        "",
        mx.raw(_pep_table(mirror_rows())),
        "",
        "## objdef variant",
        "",
        "The `objdef` variant adds a dummy objective (`OBJDEF: OBJ==0`, "
        "minimize OBJ) — the `SOLVE NLP MINIMIZING OBJ` lineage. A constant "
        "objective cannot move the equilibrium, so objdef-NLP lands on the "
        "exact base-NLP point; in the MCP form OBJ is not declared, keeping "
        "the system square.",
        "",
        mx.raw(_pep_table(variant_rows())),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
