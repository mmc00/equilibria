"""Generate docs/site/guide/gtap7_coverage_matrix_gempack.md from the gempack rows.

Run:  uv run python scripts/gtap/gen_gempack_doc.py
The output is a committed golden file; test_coverage_doc_in_sync enforces that the
committed file equals render() (CI fails on drift).

This is the "against GEMPACK" sibling of the against-GAMS coverage page. GEMPACK
(RunGTAP) is a Gragg-LINEARIZED solver, so — unlike the levels-vs-levels GAMS page —
the comparison is QUANTITY-vs-quantity in PERCENTAGE POINTS: for each dataset the
gate solves Python, reads GEMPACK's SL4 quantity %-changes (qfd→xd, qxs→xw, qo→xp),
and measures the fraction of cells whose |Δ| ≤ 1 percentage point. The cell shows
the conservative floor that fraction must clear (measured @ runtime, not stored).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))

from coverage_matrix import rows_for  # noqa: E402
import matrix_html as mx  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix_gempack.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_gempack_doc.py -->"
)

_LEGEND = mx.legend(
    '<span class="mx-li">Each cell is the <b>floor the pytest gate asserts</b>: the '
    "min fraction of quantity cells whose |Δ| ≤ <b>1 percentage point</b> vs GEMPACK "
    "(measured @ runtime, set ~5pp below the observed value):"
    '<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥65'
    '<span class="mx-swatch" style="background:var(--mx-warn)"></span>45–65'
    '<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;45</span>'
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " needs PATH/IPOPT + the RunGTAP SL4 dump — run by hand, not in CI</span>"
)


def _tone(f: float) -> str:
    return "good" if f >= 65 else ("warn" if f >= 45 else "bad")


def _table(rows) -> str:
    headers = ["Dataset · ifSUB", "Within 1pp (floor)", "GEMPACK ref"]
    body = [
        [
            mx.label(r.dataset, f"ifSUB={r.ifsub}"),
            mx.cell(mx.num(f"≥ {dict(r.stage_floors)['shock']:g}%", _tone(dict(r.stage_floors)["shock"]))),
            mx.ref(r.ref),
        ]
        for r in rows
    ]
    return mx.tablecard(headers, body)


def render() -> str:
    rows = rows_for("gtap7", "gempack", kind="mcp")
    parts = [
        "# GTAP 7 Parity Coverage Matrix — against GEMPACK",
        "",
        _BANNER,
        "",
        "This is the **GEMPACK reference** of the GTAP 7 model — the sibling of the "
        "[against-GAMS page](gtap7_coverage_matrix.md). GEMPACK (RunGTAP) solves the "
        "same model with a **Gragg-linearized** method (Euler + Richardson "
        "extrapolation), so its native output is **percentage changes**, not levels "
        "(Horridge & Pearson, *Solution Software for CGE Modeling*, COPS G-214, 2011, "
        "§4.1/4.2). The comparison is therefore **quantity-vs-quantity in percentage "
        "points**: the gate solves Python, reads GEMPACK's SL4 quantity %-changes "
        "(`qfd`→`xd`, `qxs`→`xw`, `qo`→`xp`), and measures the fraction of cells whose "
        "**|Δ| ≤ 1 percentage point**. The median |Δ| is ~0.4–1.2pp; the residual is "
        "the structural linearized↔levels gap — **identical Python↔GAMS** (both are "
        "levels solvers), so it is not a fidelity defect. The floor **decays with "
        "dataset size** as that gap accumulates over more cells.",
        "",
        mx.raw(_LEGEND),
        "",
        "## Quantity-vs-quantity (percentage points)",
        "",
        "Python post-shock quantity %-change vs the GEMPACK SL4 solution, cell-by-cell "
        "(`qfd`/`qxs`/`qo` across commodity × activity × region). Single-shock solve — "
        "only the shock stage maps.",
        "",
        _table(rows),
        "",
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
