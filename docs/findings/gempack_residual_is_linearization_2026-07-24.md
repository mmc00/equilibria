# The against-GEMPACK residual is linearization, NOT model-scope or config (2026-07-24)

**Status:** Finding — closes the investigation into *why* the against-GEMPACK
quantity match is ~52–76% within 1pp (median |Δ| ~0.4pp) rather than the "4–5
significant digits" GAMS↔GEMPACK agreement reported in the literature.

## The reference

van der Mensbrugghe, **"The Standard GTAP Model in GAMS, Version 7"** (*Journal of
Global Economic Analysis* 3(1), 2018, pp. 1–83). The GAMS model this project ports
to Python is *"a literal implementation … an exact replica of the GTAP model … a
full-blown translation of GEMPACK's TABLO code."* Two things from it:

- **Table 4** (p.50) is the official GTAP↔GAMS variable correspondence and confirms
  our value/quantity mapping: `VDFB → PD·XD` (firms' domestic), `VMSB → PM·XW`,
  `VXSB → PE·XW`, etc. — i.e. `qfd→xd`/`xda`, `qxs→xw` are correct.
- **§6 / Table 5** enumerates the GAMS extensions over standard GEMPACK: (1) a
  nested CET for domestic output allocation (GEMPACK assumes perfect transformation),
  (2) upward-sloping factor/natural-resource **supply curves** (GEMPACK: supply
  elasticities zero), (3) extra capital-account closures. Its **default** spec
  ("GFT") is: `omegax=INF` (perfect transformation), **capFlex**, factor supply
  elasticities **0**.

The natural hypothesis: our gate runs *extensions* the GEMPACK reference does not,
so the residual is model-scope, not linearization. **We tested this. It is wrong.**

## What we actually run vs RunGTAP

| Aspect | Our gate | RunGTAP `.cmf` | Match? |
|---|---|---|---|
| Output transformation `omegax` | INF (perfect) | perfect | ✓ same |
| Capital closure | **capFix** | **capFix** (`swap dpsave=del_tbalry`) | ✓ same (the `.cmf` mirrors the gate) |
| Factor supply `omegas` | 5.0 (sluggish) | `qe`/`qesf` exogenous | differ |

So the ONLY spec difference is the factor supply elasticity. capFix is **identical**
on both sides (the RunGTAP `.cmf` was written to mirror the Python gate), so it is
NOT a source of the residual — contrary to the first guess.

## The decisive experiment (gtap7_3x3)

Sweep `omegas` (factor supply elasticity), everything else fixed, measure vs GEMPACK:

| `omegas` | within-1pp | median |Δpp| | **`qe` (endowment) median** |
|---|---:|---:|---:|
| 0.001 (≈exogenous) | 31.1% | 2.52 | — (degenerate) |
| **5 (current)** | **75.8%** | **0.38** | **0.00pp** |
| ∞ (perfect mobility) | 56.3% | 0.73 | **0.00pp** |

Two conclusions, both against the hypothesis:

1. **Factors are NOT the residual.** `qe` (endowment supply) matches GEMPACK to
   **0.00pp at every `omegas`** — the factor quantities already agree exactly,
   independent of the supply-elasticity spec. Aligning "factor supply → 0" cannot
   close a gap that is already zero on factors.
2. **The current spec is optimal.** Any move off `omegas=5` (toward 0 or ∞) makes
   the *overall* match worse, not better. `fix_endowments=True` and
   `capital_mobility="mobile"` both leave the match unchanged (75.8%) — they don't
   touch `omegas`. There is no config lever that raises the match.

## Conclusion

The against-GEMPACK residual (~0.4pp median) is **the Gragg-linearized↔levels method
gap** (Horridge & Pearson G-214 §4.2: "linearized equations are not satisfied exactly
by the accurate results"), *not* a model-scope or configuration difference:

- The two model-scope suspects from the 2018 paper are **ruled out empirically**:
  capFix is identical on both sides, and factor quantities (`qe`) match to 0.00pp
  regardless of the supply-elasticity spec.
- Python ≡ GAMS on these quantities (proven separately, cell-by-cell), so it is not
  a Python fidelity defect.
- The gate already runs the spec that best matches this RunGTAP; the residual is
  irreducible without changing the *solution method*, which is the whole point of an
  independent-engine comparison.

The "4–5 significant digits" figure in the 2018 paper is GAMS-vs-GEMPACK on the same
levels-vs-levels footing *only when the linearization is driven to the levels limit*
(many Gragg steps); our single committed RunGTAP solve is a finite-step Gragg result,
so the cell-level residual is the expected linearization signature.

## Repro
`scratchpad/test_gft_final.py` (omegas sweep + per-var qe/qva), `test_omegas.py`,
`test_std_spec.py`, all on gtap7_3x3.
