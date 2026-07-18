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

from pep_coverage_matrix import (  # noqa: E402
    DATASET, nlp_rows, mcp_rows, mirror_rows, variant_rows,
)

DOC_PATH = ROOT / "docs/site/guide/pep_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/pep/pep_coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/pep/gen_pep_coverage_doc.py -->"
)

_HEADER = ["scenario", "form", "cells", "match", "gate", "ci_status", "ref"]


def _row_cells(r) -> list[str]:
    return [r.scenario, r.form, str(r.cells), r.match_note, "`" + r.gate + "`",
            r.ci_status, r.ref]


def _table(rows) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |",
             "|" + "|".join("---" for _ in _HEADER) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_row_cells(r)) + " |")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "# PEP-1-1 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        f"Coverage for the `{DATASET}` dataset — the PEP-1-1 v2.1 CGE model ported to "
        "Pyomo (`equilibria.templates.pep_pyomo`). Every row compares the Pyomo solve "
        "against a GAMS reference solved by the **same engine** (IPOPT vs IPOPT, or "
        "PATH vs PATH), so the solver's equality tolerance cancels and the cell-by-cell "
        "match reflects **model fidelity**, not solver noise. `match` is the measured "
        "snapshot; `gate` is the pytest that re-measures and enforces it. `ci_status` is "
        "`local` for every row (each needs PATH+GAMS) — PEP has no solver-free CI gate "
        "because its reference is a solved GDX, not a `.nl` coefficient dump.",
        "",
        "## NLP form vs GAMS CNS",
        "",
        "The Pyomo model solved as an NLP (IPOPT on the raw model, "
        "`nlp_scaling_method=none`, faithful to GAMS's raw solve) against the GAMS CNS "
        "reference `Results.gdx`. The benchmark BASE reproduces the SAM, so the seeded "
        "point is the calibration answer (the solver early-exits there, mirroring the "
        "original cyipopt solver and GAMS CNS). Run the gate with "
        "`phase1_nlp.py --model pep --dataset pep2 --period BASE`.",
        "",
        _table(nlp_rows()),
        "",
        "## MCP form vs GAMS-native MCP",
        "",
        "The Pyomo model solved as a complementarity problem via PATH against the "
        "**GAMS-native** MCP reference (`PEP-1-1_v2_1_mcp_solve.gms`: `MODEL /ALL/` + "
        "`SOLVE USING MCP`, so GAMS infers the equation↔variable pairing). This is the "
        "first PEP-MCP. The `sim1` row is the reference counterfactual — a 25% export-tax "
        "cut (`ttix.fx=ttixO*0.75`) — applied faithfully in Python by scaling the `ttixO` "
        "benchmark before build; both engines move GDP_BP 46707→46748.2. Run the base "
        "gate with `phase1_nlp.py --model pep --dataset pep2 --form mcp`.",
        "",
        _table(mcp_rows()),
        "",
        "## NLP↔MCP mirror",
        "",
        "The two Pyomo forms, solved from the same feasible benchmark seed, land on the "
        "identical point (LEON, the form-defining Walras slack, is excluded). Closing the "
        "one historical gap — `PD['othind']`, the price of a zero-domestic-demand good, "
        "now filled from its `*O` benchmark (1.132) in both forms rather than a blind 1.0 "
        "— lifted the mirror to a clean 100%.",
        "",
        _table(mirror_rows()),
        "",
        "## objdef variant",
        "",
        "The `objdef` variant adds a dummy objective (`OBJDEF: OBJ==0`, minimize OBJ) — "
        "the `SOLVE NLP MINIMIZING OBJ` lineage. A constant objective cannot move the "
        "equilibrium, so objdef-NLP lands on the exact base-NLP point. In the MCP form the "
        "OBJ variable is NOT declared (its OBJDEF equation is NLP-only; declaring it would "
        "leave an unpaired free variable and break squareness), so objdef+MCP stays square "
        "and solves.",
        "",
        _table(variant_rows()),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
