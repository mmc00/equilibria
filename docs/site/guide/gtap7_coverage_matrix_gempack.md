# GTAP 7 Parity Coverage Matrix — against GEMPACK

<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/gtap/gen_gempack_doc.py -->

The **GEMPACK reference** of the GTAP 7 model — the sibling of the [against-GAMS page](gtap7_coverage_matrix.md). GEMPACK (RunGTAP) solves the same model with a **Gragg-linearized** method, so its native output is **percentage changes**, not levels. The comparison is therefore **quantity-vs-quantity in percentage points**: solve Python, read GEMPACK's SL4 quantity %-changes, and count the fraction of cells within **1 percentage point**. This page focuses on **`qxs`** (bilateral exports), the variable where the two engines differ most.

## The match

Fraction of `qxs` off-diagonal cells within 1pp of GEMPACK, per dataset. Two columns: **all cells** (flat count) and **flows ≥ $0.05m** (dropping bilateral trade that is a rounding speck, where a %-change on a near-zero base is ÷0 noise). Shown for both the default GAMS-faithful closure and the GEMPACK-faithful `gtap7_gempack` closure.

```{raw} html
<div class="mx-legend"><span class="mx-li"><b>All cells</b> counts every mapped `qxs` cell flat; <b>flows ≥ $0.05m</b> drops bilateral trade that is a rounding speck (near-zero base value, where a %-change is ÷0 noise). Green ≥95, amber 80–95, red &lt;80:<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥95<span class="mx-swatch" style="background:var(--mx-warn)"></span>80–95<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;80</span><span class="mx-li"><span class="mx-chip mx-neutral">local</span> needs PATH/IPOPT + the RunGTAP SL4 dump — run by hand, not in CI</span></div>
```

**Default closure — `gams` basis** (faithful to GAMS; what the CI gate runs):

<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Dataset</th><th>Within 1pp · all cells</th><th>Within 1pp · flows ≥ $0.05m</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x3</span></td><td><div class="mx-cell"><span class="mx-num mx-warn">92.6%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">95.5%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x4</span></td><td><div class="mx-cell"><span class="mx-num mx-warn">93.8%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">95.5%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_5x5</span></td><td><div class="mx-cell"><span class="mx-num mx-warn">93.6%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_10x7</span></td><td><div class="mx-cell"><span class="mx-num mx-warn">84.5%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr></tbody></table></div></div>

**`gtap7_gempack` closure — `gempack` basis** (re-anchored to GEMPACK):

<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Dataset</th><th>Within 1pp · all cells</th><th>Within 1pp · flows ≥ $0.05m</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x3</span></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_3x4</span></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_5x5</span></td><td><div class="mx-cell"><span class="mx-num mx-good">95.2%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr><tr><td class="mx-lbl"><span class="mx-ds">gtap7_10x7</span></td><td><div class="mx-cell"><span class="mx-num mx-warn">92.9%</span></div></td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span></div></td></tr></tbody></table></div></div>

The story both tables tell: **on flows with economic substance (≥ $0.05m) the match reaches ~95–100% under either closure**, and 100% under `gtap7_gempack`. The lower *all-cells* number is the near-zero-flow tail (next section), not a real disagreement. (`15x10` is `@slow` — NLP ~20min — and validated by hand, not shown here.)

## Choosing a closure

Both closures are built through the same factory; pass the name to `_closure_template_data` and validate:

```python
from equilibria.templates.gtap.gtap_contract import (
    GTAPClosureConfig, _closure_template_data,
)

# Default — faithful to GAMS (va_subsidy_basis='gams'). The standard closure.
closure = GTAPClosureConfig.model_validate(
    _closure_template_data("gtap_standard")
)

# GEMPACK-faithful — re-anchors VA valuation to GEMPACK's EVFP subsidy basis
# (va_subsidy_basis='gempack'). Use when GEMPACK is the reference.
closure = GTAPClosureConfig.model_validate(
    _closure_template_data("gtap7_gempack")
)
```

Equivalently, set the flag directly: `GTAPClosureConfig(..., va_subsidy_basis="gempack")`. The default is `"gams"`, so an unqualified closure is always the GAMS-faithful one.

## Why the *all-cells* number is lower: the near-zero-flow tail

The flat count weighs every cell equally, including bilateral flows that are a rounding speck (base value ≈ $0). A %-change on a near-zero denominator is numerical noise, not a measurement — e.g. a `Chem` CHN→CHN flow of **$0.000000m** shows GEMPACK −24% vs Python +1957%, both meaningless. Thresholding on the cell's **base flow** (a property known before the shock, not on |Δ| — which would be circular) removes them. On **10x7** (`gams` basis):

| base-flow threshold | cells kept | within-1pp |
| --- | --- | --- |
| $0 (all cells) | 490 | 84.5% |
| ≥ $0.01m | 157 | ~98% |
| ≥ $0.05m | 70 | **100.0%** |

All the failing cells are sub-$0.05m ghost flows (median base $0.00m). The gate keeps the conservative all-cells count so the number is never inflated by exclusion; this section is what that count is *made of*. Note the flat count is mildly solve-path sensitive (the perf stack — settle_only / MUMPS-reuse / cuDSS — can nudge a ghost cell across the ±1pp line) while the ≥$0.05m match stays put.

## What the residual is — and is not

Python is faithful to GAMS to ~6 digits; van der Mensbrugghe (*The Standard GTAP Model in GAMS, v7*, JGEA 3(1), 2018) reports GAMS≡GEMPACK *"to within 4–5 significant digits"*. So the residual is a **GAMS-vs-GEMPACK** difference, not a Python defect. It is **not**:

- **linearization** — GEMPACK's own step-grid converges internally to ~0.002pp (s4 vs s64), ~300× smaller than any observed gap;
- **model scope** — factor quantities (`qe`) match GEMPACK to 0.00pp at every factor-supply elasticity; sweeping `omegas` never improves the match;
- **a closure mismatch** — the gate's `capFix` solve is compared only against `_capfix` fixtures (crossing capFix/capFlex would be a measurement error, not a model defect).

What it **is**: on **subsidized agriculture**, GAMS and GEMPACK disagree on the factor-subsidy sign convention (HAR `FBEP`, native-negative). GAMS/Python value value-added as `evfb + ftrv − fbep`; GEMPACK's EVFP as `evfb + ftrv + fbep`. That shifts the VA-vs-intermediate weight on subsidized crops, moving the domestic price and — through the import nest — `qxs`. The `gtap7_gempack` closure adopts GEMPACK's convention, which is why its column above is higher on the small agricultural flows. Details: `docs/findings/gempack_residual_is_linearization_2026-07-24`.

## Variables compared (15)

`qxs` is the focus above, but the gate maps 15 quantity variables (verified 1:1 GEMPACK→Python correspondence). **Prices**, the **tariff shock** itself (`tm` = +10% uniform), and **welfare** (`u`/`EV`) are out of scope here — welfare lives in the separate EV track (`docs/findings/gempack_welfare_not_cellwise`).

<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">GEMPACK var</th><th>Python Var</th><th>flow</th></tr></thead><tbody><tr><td class="mx-lbl">`qfd`</td><td>`xda`</td><td>firm domestic demand</td></tr><tr><td class="mx-lbl">`qfm`</td><td>`xma`</td><td>firm imported demand</td></tr><tr><td class="mx-lbl">`qfa`</td><td>`xaa`</td><td>firm Armington demand</td></tr><tr><td class="mx-lbl">`qxs`</td><td>`xw`</td><td>bilateral exports</td></tr><tr><td class="mx-lbl">`qxw`</td><td>`xet`</td><td>aggregate exports</td></tr><tr><td class="mx-lbl">`qms`</td><td>`xmt`</td><td>aggregate imports</td></tr><tr><td class="mx-lbl">`qds`</td><td>`xd`</td><td>domestic sales</td></tr><tr><td class="mx-lbl">`qpa`</td><td>`xc`</td><td>private demand</td></tr><tr><td class="mx-lbl">`qga`</td><td>`xg`</td><td>government demand</td></tr><tr><td class="mx-lbl">`qc`</td><td>`xs`</td><td>total commodity supply</td></tr><tr><td class="mx-lbl">`qe`</td><td>`xft`</td><td>endowment supply</td></tr><tr><td class="mx-lbl">`qtm`</td><td>`xtmg`</td><td>global margin usage</td></tr><tr><td class="mx-lbl">`qinv`</td><td>`xiagg`</td><td>investment demand</td></tr><tr><td class="mx-lbl">`qva`</td><td>`xp`</td><td>value added</td></tr><tr><td class="mx-lbl">`qgdp`</td><td>`rgdpmp`</td><td>real GDP index</td></tr></tbody></table></div></div>

## Scope

This page covers the five **gtap7_\*** datasets the multi-period gate solves. **nus333** and **9x10** are elsewhere: they use a separate single-period apparatus and already have GEMPACK/RunGTAP coverage in the welfare/macro track they were built for (`compare_nus333_rungtap.py` / `compare_9x10_rungtap.py`, ~0.01–0.3pp on `u` — see `docs/findings/rungtap_welfare_parity_2026-05-15`).

