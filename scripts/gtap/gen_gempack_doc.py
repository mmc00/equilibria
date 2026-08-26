"""Generate docs/site/guide/gtap7_coverage_matrix_gempack.md from the gempack rows.

Run:  uv run python scripts/gtap/gen_gempack_doc.py
The output is a committed golden file; test_coverage_doc_in_sync enforces that the
committed file equals render() (CI fails on drift).

This is the "against GEMPACK" sibling of the against-GAMS coverage page. GEMPACK
(RunGTAP) is a Gragg-LINEARIZED solver, so — unlike the levels-vs-levels GAMS page —
the comparison is QUANTITY-vs-quantity in PERCENTAGE POINTS: solve Python, read
GEMPACK's SL4 quantity %-changes (qfd→xd, qxs→xw, qo→xp), and count the fraction of
cells within 1 percentage point. The page focuses on `qxs`, shown per dataset for
both closures (gams / gtap7_gempack) and both metrics (all cells / flows ≥ $0.05m).
The measured numbers in _MEASURED are cell-by-cell runtime measurements, not stored
gate floors; re-measure and update them when the model or fixtures change.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))

import matrix_html as mx  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix_gempack.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_gempack_doc.py -->"
)

_LEGEND = mx.legend(
    '<span class="mx-li"><b>All cells</b> counts every mapped `qxs` cell flat; '
    "<b>flows ≥ $0.05m</b> drops bilateral trade that is a rounding speck "
    "(near-zero base value, where a %-change is ÷0 noise). Green ≥95, amber 80–95, "
    "red &lt;80:"
    '<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥95'
    '<span class="mx-swatch" style="background:var(--mx-warn)"></span>80–95'
    '<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;80</span>'
    '<span class="mx-li">'
    + mx.chip("local", "neutral")
    + " needs PATH/IPOPT + the RunGTAP SL4 dump — run by hand, not in CI</span>"
)


def _tone(f: float) -> str:
    return "good" if f >= 95 else ("warn" if f >= 80 else "bad")


# Measured qxs off-diagonal within-1pp vs the capFix sl4dump fixtures, on main
# (base_calibrated=True, capFix/NLP for 10x7). Per (dataset): (all-cells%, ≥$0.05m%).
# Two bases: "gams" (default, faithful to GAMS) and "gempack" (the gtap7_gempack
# closure). 15x10 is @slow (NLP ~20min) and measured separately — left None here.
_MEASURED = {
    "gtap7_3x3": {"gams": (92.6, 95.5), "gempack": (100.0, 100.0)},
    "gtap7_3x4": {"gams": (93.8, 95.5), "gempack": (100.0, 100.0)},
    "gtap7_5x5": {"gams": (93.6, 100.0), "gempack": (95.2, 100.0)},
    "gtap7_10x7": {"gams": (84.5, 100.0), "gempack": (92.9, 100.0)},
}


def _match_table(basis: str) -> str:
    """qxs within-1pp per dataset for one basis: all-cells and flows ≥ $0.05m."""
    headers = [
        "Dataset",
        "Within 1pp · all cells",
        "Within 1pp · flows ≥ $0.05m",
    ]
    body = []
    for ds in ("gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7"):
        m = _MEASURED[ds][basis]
        body.append(
            [
                mx.label(ds),
                mx.cell(mx.num(f"{m[0]:g}%", _tone(m[0]))),
                mx.cell(mx.num(f"{m[1]:g}%", _tone(m[1]))),
            ]
        )
    return mx.tablecard(headers, body)


def _var_list() -> str:
    """The verified GEMPACK-quantity → Python-Var comparison map, from Q_TO_VAR."""
    import gempack_reference as gr

    desc = {
        "qfd": "firm domestic demand",
        "qfm": "firm imported demand",
        "qfa": "firm Armington demand",
        "qxs": "bilateral exports",
        "qxw": "aggregate exports",
        "qms": "aggregate imports",
        "qds": "domestic sales",
        "qpa": "private demand",
        "qga": "government demand",
        "qc": "total commodity supply",
        "qe": "endowment supply",
        "qtm": "global margin usage",
        "qinv": "investment demand",
        "qva": "value added",
        "qgdp": "real GDP index",
    }
    rows_b = [
        [f"`{gv}`", f"`{spec['var']}`", desc.get(gv, "")]
        for gv, spec in gr.Q_TO_VAR.items()
    ]
    return mx.tablecard(["GEMPACK var", "Python Var", "flow"], rows_b)


def render() -> str:
    parts = [
        "# GTAP 7 Parity Coverage Matrix — against GEMPACK",
        "",
        _BANNER,
        "",
        # --- 1. What this page is -------------------------------------------
        "The **GEMPACK reference** of the GTAP 7 model — the sibling of the "
        "[against-GAMS page](gtap7_coverage_matrix.md). GEMPACK (RunGTAP) solves the "
        "same model with a **Gragg-linearized** method, so its native output is "
        "**percentage changes**, not levels. The comparison is therefore "
        "**quantity-vs-quantity in percentage points**: solve Python, read GEMPACK's "
        "SL4 quantity %-changes, and count the fraction of cells within **1 "
        "percentage point**. This page focuses on **`qxs`** (bilateral exports), the "
        "variable where the two engines differ most.",
        "",
        # --- 2. The match table (both closures, both metrics) ---------------
        "## The match",
        "",
        "Fraction of `qxs` off-diagonal cells within 1pp of GEMPACK, per dataset. "
        "Two columns: **all cells** (flat count) and **flows ≥ $0.05m** (dropping "
        "bilateral trade that is a rounding speck, where a %-change on a near-zero "
        "base is ÷0 noise). Shown for both the default GAMS-faithful closure and the "
        "GEMPACK-faithful `gtap7_gempack` closure.",
        "",
        mx.raw(_LEGEND),
        "",
        "**Default closure — `gams` basis** (faithful to GAMS; what the CI gate runs):",
        "",
        _match_table("gams"),
        "",
        "**`gtap7_gempack` closure — `gempack` basis** (re-anchored to GEMPACK):",
        "",
        _match_table("gempack"),
        "",
        "The story both tables tell: **on flows with economic substance (≥ $0.05m) "
        "the match reaches ~95–100% under either closure**, and 100% under "
        "`gtap7_gempack`. The lower *all-cells* number is the near-zero-flow tail "
        "(next section), not a real disagreement. (`15x10` is `@slow` — NLP ~20min — "
        "and validated by hand, not shown here.)",
        "",
        # --- 3. How to invoke each closure ----------------------------------
        "## Choosing a closure",
        "",
        "Both closures are built through the same factory; pass the name to "
        "`_closure_template_data` and validate:",
        "",
        "```python",
        "from equilibria.templates.gtap.gtap_contract import (",
        "    GTAPClosureConfig, _closure_template_data,",
        ")",
        "",
        "# Default — faithful to GAMS (va_subsidy_basis='gams'). The standard closure.",
        "closure = GTAPClosureConfig.model_validate(",
        '    _closure_template_data("gtap_standard")',
        ")",
        "",
        "# GEMPACK-faithful — re-anchors VA valuation to GEMPACK's EVFP subsidy basis",
        "# (va_subsidy_basis='gempack'). Use when GEMPACK is the reference.",
        "closure = GTAPClosureConfig.model_validate(",
        '    _closure_template_data("gtap7_gempack")',
        ")",
        "```",
        "",
        "Equivalently, set the flag directly: "
        '`GTAPClosureConfig(..., va_subsidy_basis="gempack")`. The default is '
        '`"gams"`, so an unqualified closure is always the GAMS-faithful one.',
        "",
        # --- 4. Why the all-cells number is lower (the tail) ----------------
        "## Why the *all-cells* number is lower: the near-zero-flow tail",
        "",
        "The flat count weighs every cell equally, including bilateral flows that are "
        "a rounding speck (base value ≈ $0). A %-change on a near-zero denominator is "
        "numerical noise, not a measurement — e.g. a `Chem` CHN→CHN flow of "
        "**$0.000000m** shows GEMPACK −24% vs Python +1957%, both meaningless. "
        "Thresholding on the cell's **base flow** (a property known before the shock, "
        "not on |Δ| — which would be circular) removes them. On **10x7** (`gams` "
        "basis):",
        "",
        "| base-flow threshold | cells kept | within-1pp |",
        "| --- | --- | --- |",
        "| $0 (all cells) | 490 | 84.5% |",
        "| ≥ $0.01m | 157 | ~98% |",
        "| ≥ $0.05m | 70 | **100.0%** |",
        "",
        "All the failing cells are sub-$0.05m ghost flows (median base $0.00m). The "
        "gate keeps the conservative all-cells count so the number is never inflated "
        "by exclusion; this section is what that count is *made of*. Note the flat "
        "count is mildly solve-path sensitive (the perf stack — settle_only / "
        "MUMPS-reuse / cuDSS — can nudge a ghost cell across the ±1pp line) while the "
        "≥$0.05m match stays put.",
        "",
        # --- 5. What the residual is (and is not) ---------------------------
        "## What the residual is — and is not",
        "",
        "Python is faithful to GAMS to ~6 digits; van der Mensbrugghe "
        "(*The Standard GTAP Model in GAMS, v7*, JGEA 3(1), 2018) reports "
        'GAMS≡GEMPACK *"to within 4–5 significant digits"*. So the residual is a '
        "**GAMS-vs-GEMPACK** difference, not a Python defect. It is **not**:",
        "",
        "- **linearization** — GEMPACK's own step-grid converges internally to "
        "~0.002pp (s4 vs s64), ~300× smaller than any observed gap;",
        "- **model scope** — factor quantities (`qe`) match GEMPACK to 0.00pp at "
        "every factor-supply elasticity; sweeping `omegas` never improves the match;",
        "- **a closure mismatch** — the gate's `capFix` solve is compared only "
        "against `_capfix` fixtures (crossing capFix/capFlex would be a measurement "
        "error, not a model defect).",
        "",
        "What it **is**: on **subsidized agriculture**, GAMS and GEMPACK disagree on "
        "the factor-subsidy sign convention (HAR `FBEP`, native-negative). GAMS/Python "
        "value value-added as `evfb + ftrv − fbep`; GEMPACK's EVFP as "
        "`evfb + ftrv + fbep`. That shifts the VA-vs-intermediate weight on "
        "subsidized crops, moving the domestic price and — through the import nest — "
        "`qxs`. The `gtap7_gempack` closure adopts GEMPACK's convention, which is why "
        "its column above is higher on the small agricultural flows. Details: "
        "`docs/findings/gempack_residual_is_linearization_2026-07-24`.",
        "",
        # --- 6. The variable map --------------------------------------------
        "## Variables compared (15)",
        "",
        "`qxs` is the focus above, but the gate maps 15 quantity variables (verified "
        "1:1 GEMPACK→Python correspondence). **Prices**, the **tariff shock** itself "
        "(`tm` = +10% uniform), and **welfare** (`u`/`EV`) are out of scope here — "
        "welfare lives in the separate EV track "
        "(`docs/findings/gempack_welfare_not_cellwise`).",
        "",
        _var_list(),
        "",
        # --- 7. Scope -------------------------------------------------------
        "## Scope",
        "",
        "This page covers the five **gtap7_\\*** datasets the multi-period gate "
        "solves. **nus333** and **9x10** are elsewhere: they use a separate "
        "single-period apparatus and already have GEMPACK/RunGTAP coverage in the "
        "welfare/macro track they were built for (`compare_nus333_rungtap.py` / "
        "`compare_9x10_rungtap.py`, ~0.01–0.3pp on `u` — see "
        "`docs/findings/rungtap_welfare_parity_2026-05-15`).",
        "",
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
