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
    '<span class="mx-li"><b>Measured</b> is the fraction of quantity cells whose '
    "|Δ| ≤ <b>1 percentage point</b> vs GEMPACK (capFlex, verified 2026-08-19); "
    "<b>Floor</b> is the conservative threshold the pytest gate asserts (~5pp under "
    "measured, for solver-noise margin):"
    '<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥65'
    '<span class="mx-swatch" style="background:var(--mx-warn)"></span>45–65'
    '<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;45</span>'
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " needs PATH/IPOPT + the RunGTAP SL4 dump — run by hand, not in CI</span>"
)


def _tone(f: float) -> str:
    return "good" if f >= 65 else ("warn" if f >= 45 else "bad")


# Measured within-1pp @ capFlex vs the default sl4dump fixtures (verified 2026-08-19,
# base_calibrated=True; median |Δpp| in parens). The FLOOR column is this minus a
# ~5pp solver-noise margin. Keep in sync with the gate's own measured comments in
# test_gtap7_gempack_parity.py.
_MEASURED = {
    "gtap7_3x3": (99.5, 0.036),
    "gtap7_3x4": (99.2, 0.043),
    "gtap7_5x5": (97.1, 0.035),
    "gtap7_10x7": (97.5, 0.032),
    "gtap7_15x10": (92.3, 0.085),
}


def _table(rows) -> str:
    headers = ["Dataset · ifSUB", "Measured within 1pp", "Floor (gate asserts)", "GEMPACK ref"]
    body = []
    for r in rows:
        meas = _MEASURED.get(r.dataset)
        meas_cell = (
            mx.cell(mx.num(f"{meas[0]:g}% (med {meas[1]:g}pp)", _tone(meas[0])))
            if meas
            else mx.cell("—")
        )
        body.append(
            [
                mx.label(r.dataset, f"ifSUB={r.ifsub}"),
                meas_cell,
                mx.cell(mx.num(f"≥ {dict(r.stage_floors)['shock']:g}%", _tone(dict(r.stage_floors)["shock"]))),
                mx.ref(r.ref),
            ]
        )
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
        "point** vs GEMPACK. The measured match is **92–99.5% within 1pp** (median "
        "|Δ| ~0.03–0.2pp) across the five datasets with the gate's default **capFlex** "
        "closure; the per-dataset numbers below are the **conservative floors** the "
        "pytest gate asserts (set ~5pp under the measured value to absorb solver "
        "noise), not the match itself. The small residual is **identical "
        "Python↔GAMS** (both are levels solvers), so it is not a Python fidelity "
        "defect — see the ⚠️ note below for what it is (and is not).",
        "",
        "> **⚠️ The residual is NOT linearization, and the closure is capFlex, not "
        "capFix.** Two claims in the paragraph below were corrected (2026-08-19): "
        "(1) The gate's default closure is **capFlex** (RORDELTA=1, returns equalize), "
        "which matches the default `sl4dump_*_tm10.har` fixtures — measured 3x3 99.5%, "
        "5x5 97.1%, 10x7 97.5%, 15x10 92.3% within 1pp. The `capFix` closure (and its "
        "`_capfix` fixtures) is the *other* option, selected via `savf_flag`. Comparing "
        "a capFix solve against a capFlex fixture (or vice versa) crosses closures and "
        "drops the match to ~70–80% — a measurement error, not a model defect. "
        "(2) The residual is **not** the Gragg-linearization gap: GEMPACK's own "
        "step-grid (`sl4dump_*_tm10_s{4,8,16,32,64}.har`) converges internally to "
        "**~0.002pp** (s4 vs s64), ~300× smaller than any observed gap, so "
        "linearization cannot explain it. Details: "
        "`docs/findings/gempack_residual_is_linearization_2026-07-24` (corrected).",
        "",
        "**The residual is not model scope — extensions and factor elasticities ruled "
        "out.** "
        "van der Mensbrugghe (*The Standard GTAP Model in GAMS, v7*, JGEA 3(1), 2018) "
        "reports GAMS≡GEMPACK *\"to within 4–5 significant digits\"* under the standard "
        "spec, and its **Table 4** confirms our mapping (`VDFB → PD·XD`, `VMSB → PM·XW`, "
        "…). The GAMS/Python model does add extensions (CET output allocation, factor "
        "supply curves, extra closures — §6/Table 5), so we tested whether those cause "
        "the residual. **They do not:** the gate's capFlex capital closure matches the "
        "RunGTAP `.cmf` (the capFix option matches the RunGTAP swap `dpsave=del_tbalry`), "
        "and factor quantities (`qe` endowments) match GEMPACK to "
        "**0.00pp at every factor-supply elasticity** — sweeping `omegas` only makes the "
        "*overall* match worse, never better. So the small residual is **not** a "
        "model-scope or configuration difference — and Python≡GAMS on these cells, so "
        "not a Python defect. What it **is** (a base-seed/denominator effect, not "
        "linearization) is covered in the ⚠️ note above and in "
        "`docs/findings/gempack_residual_is_linearization_2026-07-24` (corrected).",
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
