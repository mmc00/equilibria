"""Generate docs/site/guide/gtap7_coverage_matrix.md from coverage_matrix.ROWS.

Run:  uv run python scripts/gtap/gen_coverage_doc.py
The output is a committed golden file; test_coverage_doc_in_sync enforces that the
committed file equals render() (CI fails on drift).

The page renders ONLY the two same-engine fidelity gates (NLP vs NLP, MCP vs MCP)
in the artifact card format. The .nl coefficient gate and the legacy PATH SOLVE
gates (kind altertax/gtap_solve) remain in coverage_matrix.ROWS and keep driving
their pytest gates — they are just no longer rendered as tables here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))

from coverage_matrix import nlp_rows, mcp_rows  # noqa: E402
import matrix_html as mx  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->"
)

_LEGEND = mx.legend(
    '<span class="mx-li">Each cell is the per-stage <b>floor the pytest gate '
    "asserts</b> (measured match% @ tol 1% must be ≥ floor):"
    '<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥99'
    '<span class="mx-swatch" style="background:var(--mx-warn)"></span>95–99'
    '<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;95</span>'
    '<span class="mx-li"><b>convergence</b> chip: '
    + mx.chip("✓ code 1", "good") + " required "
    + mx.chip("code ≤ 2", "warn") + " audited near-miss</span>"
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " gates need PATH/IPOPT + GAMS refs — run by hand, not in CI</span>"
)


def _stage_cell(r, stage: str) -> str:
    floors = dict(r.stage_floors)
    f = floors[stage]
    conv = mx.chip("code ≤ 2", "warn") if stage in r.code2_ok else mx.chip("✓ code 1", "good")
    return mx.cell(mx.num(f"≥ {f:g}", mx.floor_tone(f)), conv)


def _gate_table(rows) -> str:
    headers = ["Dataset · ifSUB", "Base", "Check", "Shock", "Reference"]
    body = [
        [
            mx.label(r.dataset, f"ifSUB={r.ifsub}"),
            _stage_cell(r, "base"),
            _stage_cell(r, "check"),
            _stage_cell(r, "shock"),
            mx.ref(r.ref),
        ]
        for r in rows
    ]
    return mx.tablecard(headers, body)


def render() -> str:
    # This is the "against GAMS" page — only reference="gams" rows belong here.
    # reference="gempack" rows are a sibling page (different fixtures + pp metric).
    nlp = [r for r in nlp_rows() if r.reference == "gams"]
    mcp = [r for r in mcp_rows() if r.reference == "gams"]
    parts = [
        "# GTAP 7 Parity Coverage Matrix — against GAMS",
        "",
        _BANNER,
        "",
        "This page is the **GAMS reference** of the GTAP 7 model; sibling pages "
        "under \"Validation & parity\" hold the same matrix against other "
        "references (e.g. GEMPACK/RunGTAP) as those references and rows land. "
        "Same-engine parity between the Python `equilibria` GTAP Standard 7 "
        "implementation and GAMS: **NLP vs NLP** (IPOPT both sides) and "
        "**MCP vs MCP** (PATH both sides), so the solver's equality tolerance "
        "cancels and the cell-by-cell match reflects **model fidelity**, not "
        "solver noise. Cells show the conservative per-stage **floor** each "
        "pytest gate asserts (the gate runs the real solve, measures match% "
        "@ tol 1% and the return code, and requires `match ≥ floor` and "
        "`code == 1`); the measured snapshot is not stored here — the live "
        "views re-run the measurement. A separate solver-free `.nl` "
        "coefficient gate (0 Jacobian diffs vs GAMS, "
        "`test_gtap7_nl_parity.py`) runs in CI as the structural canary.",
        "",
        mx.raw(_LEGEND),
        "",
        "## NLP vs NLP",
        "",
        "Python solved as an NLP (`EQUILIBRIA_GTAP_SOLVE_NLP=1`, maximize "
        "walras) against the GAMS `ifMCP=0` NLP reference — same IPOPT both "
        "sides. Gate: `test_gtap7_nlp_parity.py` (local; needs IPOPT + the "
        "committed GAMS refs). Measured view: "
        "[live NLP matrix](../_static/gtap7_nlp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        mx.raw(_gate_table([r for r in nlp if r.mode == "pure"])),
        "",
        mx.raw(mx.note(
            "<b>100% across every stage, both ifSUB</b>, after the Jacobian "
            "pre-scale skip (commit e4c40d7) — GAMS solves the raw model; the "
            "Python-only pre-scale steered IPOPT to a wrong basin on the 5×5 "
            "shock (59.56% → 100%)."
        )),
        "",
        "### Altertax (CD)",
        "",
        mx.raw(_gate_table([r for r in nlp if r.mode == "altertax"])),
        "",
        mx.raw(mx.note(
            "<b>The check/shock ceiling here is the reference, not the "
            "model.</b> The altertax NLP references are mis-converged (IPOPT "
            "\"Locally Optimal\", the ref violates its own eq_pxeq in the ag "
            "sector). Where a cleanly-converged MCP reference exists the same "
            "Python solve reaches 99.9% — see the MCP gate below."
        )),
        "",
        "## MCP vs MCP",
        "",
        "Python solved via PATH (nonlinear-full MCP) against the "
        "cleanly-converged **NEOS** MCP references — same PATH both sides. "
        "Gate: `test_gtap7_mcp_parity.py` (local; needs PATH + the committed "
        "refs). This is the daily fidelity gate. Measured view: "
        "[live MCP matrix](../_static/gtap7_mcp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        mx.raw(_gate_table([r for r in mcp if r.mode == "pure"])),
        "",
        "### Altertax (CD)",
        "",
        mx.raw(_gate_table([r for r in mcp if r.mode == "altertax"])),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
