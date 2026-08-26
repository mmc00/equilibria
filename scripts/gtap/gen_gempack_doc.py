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


def _var_list() -> str:
    """The verified GEMPACK-quantity → Python-Var comparison map, from Q_TO_VAR."""
    import gempack_reference as gr

    desc = {
        "qfd": "firm domestic demand", "qfm": "firm imported demand",
        "qfa": "firm Armington demand", "qxs": "bilateral exports",
        "qxw": "aggregate exports", "qms": "aggregate imports",
        "qds": "domestic sales", "qpa": "private demand",
        "qga": "government demand", "qc": "total commodity supply",
        "qe": "endowment supply", "qtm": "global margin usage",
        "qinv": "investment demand", "qva": "value added",
        "qgdp": "real GDP index",
    }
    rows_b = [
        [f"`{gv}`", f"`{spec['var']}`", desc.get(gv, "")]
        for gv, spec in gr.Q_TO_VAR.items()
    ]
    return mx.tablecard(["GEMPACK var", "Python Var", "flow"], rows_b)


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
        "points**: the gate solves Python and, for each of the 15 mapped quantity "
        "variables below, measures the fraction of cells whose **|Δ| ≤ 1 percentage "
        "point** vs GEMPACK. The median |Δ| is ~0.4–1.2pp; the residual is the "
        "structural linearized↔levels gap — **identical Python↔GAMS** (both are levels "
        "solvers), so it is not a fidelity defect. The floor **decays with dataset "
        "size** as that gap accumulates over more cells.",
        "",
        "**The residual is the linearization gap — verified, not model scope.** "
        "van der Mensbrugghe (*The Standard GTAP Model in GAMS, v7*, JGEA 3(1), 2018) "
        "reports GAMS≡GEMPACK *\"to within 4–5 significant digits\"* under the standard "
        "spec, and its **Table 4** confirms our mapping (`VDFB → PD·XD`, `VMSB → PM·XW`, "
        "…). The GAMS/Python model does add extensions (CET output allocation, factor "
        "supply curves, extra closures — §6/Table 5), so we tested whether those cause "
        "the residual. **They do not:** the gate's capital closure (capFix) is identical "
        "to the RunGTAP `.cmf`, and factor quantities (`qe` endowments) match GEMPACK to "
        "**0.00pp at every factor-supply elasticity** — sweeping `omegas` only makes the "
        "*overall* match worse, never better (75.8% at the current `omegas=5`, 56% at ∞, "
        "31% at ≈0). So the ~0.4pp residual is the **Gragg-linearized↔levels method gap** "
        "(finite-step Gragg vs an exact levels solve), not a model-scope or configuration "
        "difference — and Python≡GAMS on these cells, so not a Python defect. Details: "
        "`docs/findings/gempack_residual_is_linearization_2026-07-24`.",
        "",
        mx.raw(_LEGEND),
        "",
        "## Variables compared (15)",
        "",
        "GEMPACK reports one solution %-change per model variable; these 15 quantity "
        "variables have a verified 1:1 correspondence to a Python Var (established by "
        "an exhaustive discovery pass, then filtered by economic meaning — a small Δ "
        "alone is not proof since tariff-shock quantities co-move). **Prices**, the "
        "**tariff shock itself** (`tm` = +10% uniform, the identical input to both "
        "engines), and **welfare** (`u`/`EV`) are out of scope here — welfare is "
        "sign-flipping and second-order and lives in the separate EV track "
        "(`docs/findings/gempack_welfare_not_cellwise`).",
        "",
        _var_list(),
        "",
        "## Quantity-vs-quantity match (percentage points)",
        "",
        "Fraction of cells within 1 pp, over the 15 variables × commodity × activity × "
        "region. Single-shock solve — only the shock stage maps. GEMPACK ran one tariff "
        "shock and is **ifSUB-agnostic**: ifSUB 0 and 1 measure identically (the "
        "quantities don't depend on the subsidy convention), so both are shown.",
        "",
        _table(rows),
        "",
        "### The residual is the near-zero-flow tail (measured)",
        "",
        "The cell counts above are **flat** — every mapped cell weighs 1, including "
        "bilateral trade flows that are a rounding speck (base value ≈ $0). A "
        "%-change on a near-zero denominator is numerical noise, not a measurement: "
        "GEMPACK and Python each report a large, meaningless number there (e.g. a "
        "`Chem` CHN→CHN flow of **$0.000000m** shows GEMPACK −24% vs Python +1957%). "
        "So the flat count understates the economic agreement. Thresholding on the "
        "cell's **base flow** (a property known before the shock — not on |Δ|, which "
        "would be circular), applied identically to both engines, the `qxs` "
        "off-diagonal match on **10x7** is:",
        "",
        "| base-flow threshold | cells kept | within-1pp |",
        "| --- | --- | --- |",
        "| $0 (all cells) | 490 | 96.5% |",
        "| ≥ $0.01m | 156 | 98.1% |",
        "| ≥ $0.05m | 70 | **100.0%** |",
        "",
        "**On flows with economic substance (≥ $0.05m) the match is 100%** — the "
        "~3.5% residual at the flat count lives entirely in the sub-$0.05m tail "
        "(13 of 17 failing cells are near-zero agricultural flows). This is measured, "
        "not assumed; the gate keeps the conservative flat count so the number is "
        "never inflated by exclusion.",
        "",
        "### The `gtap7_gempack` closure (subsidy-basis variant)",
        "",
        "Python is faithful to GAMS to ~6 digits; on **subsidized agriculture** the "
        "GAMS and GEMPACK *reference engines* disagree on the factor-subsidy sign "
        "convention (HAR `FBEP`, native-negative): GAMS/Python value value-added as "
        "`evfb + ftrv − fbep`, GEMPACK's EVFP as `evfb + ftrv + fbep`. The named "
        "closure **`gtap7_gempack`** (flag `va_subsidy_basis=\"gempack\"`, default "
        "`\"gams\"`, off) re-anchors the VA valuation to GEMPACK's basis. It is "
        "byte-identical on the default path (the `.nl` coefficient gate is unchanged), "
        "and lifts the flat `qxs` off-diagonal match — **3x3 92.6→100%, 3x4 "
        "93.8→100%, 5x5 93.6→95.2%, 10x7 84.5→96.5%**. The gain concentrates in the "
        "small-agricultural-flow tail (where the subsidy sign moves the price most); "
        "on the ≥$0.05m substantive flows both bases already reach 100%. The gate and "
        "this matrix run the **default (`gams`) basis**, so the numbers above are the "
        "faithful-to-GAMS reference; the variant is opt-in.",
        "",
        "### Scope",
        "",
        "This cell-by-cell page covers the five **gtap7_\\*** datasets, which the "
        "multi-period gate solves. **nus333** and **9x10** are *not* here: they are "
        "solved by a separate single-period apparatus (`compare_nus333_vs_neos._solve` "
        "with homotopy + capFix closure), and they already have GEMPACK/RunGTAP "
        "coverage in the **welfare/macro track** they were built for "
        "(`compare_nus333_rungtap.py` / `compare_9x10_rungtap.py`, validated to "
        "~0.01–0.3pp on `u` and ~0.3–1.7% on EV — see "
        "`docs/findings/rungtap_welfare_parity_2026-05-15`). Wiring them into the "
        "cell-by-cell gate would duplicate that coverage through a second, fragile "
        "solve path, so it is deliberately out of scope.",
        "",
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
