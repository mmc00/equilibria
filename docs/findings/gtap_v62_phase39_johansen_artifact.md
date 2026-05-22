# GTAP v6.2 Phase 3.9 — The 9pp gap was a GEMPACK linearization artifact

**Date:** 2026-05-21
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.8 (SAM pre-balance via residual bake-in)

## TL;DR

The ~9pp gap reported in Phase 3.8 against the GEMPACK ``Exp1a-upd.har``
reference is **mostly a GEMPACK Johansen-1-step linearization artifact**,
not a Python calibration bug.

BOOK3X3's ``Exp1a.exp`` ships with ``Method = Johansen; Steps = 1``, the
crudest available stepping scheme. It approximates the shock response by
a single linear Newton step at the SAM, which under-estimates the
non-linear amplification for shocks of ~10% magnitude with
``esubm[food] ≈ 4.64``.

Re-running the same shock with ``Method = Gragg; Steps = 2 4 6``
(Richardson extrapolation) produces a much larger non-linear response,
and the Python ↔ GEMPACK parity gap collapses to **~3pp** on the
bilateral US→EU food trade flows.

## Numbers

10% tariff cut on `tms[food,USA,EU]`, three reference references:

| Cell | Johansen-1 | Gragg 2-4-6 | Python (Phase 3.8 + IPOPT) |
|------|------------|-------------|----------------------------|
| VIMS USA→EU | +31.54% | **+38.17%** | +35.89% |
| VIWS USA→EU | +41.54% | **+53.52%** | +50.99% |
| VXMD USA→EU | +41.55% | **+53.55%** | +50.99% |
| VDPM food EU | -0.264% | -0.335% | -0.190% |
| VIPM food EU | +1.848% | +2.278% | +2.417% |

Gap vs **Johansen-1**: VIWS/VXMD +9.4pp (the "9pp" we had been chasing).
Gap vs **Gragg-multistep** (the non-linearly-corrected GEMPACK answer):
**VIWS/VXMD only −2.5pp** — Python *under*-shoots by 2-3pp, not over.

## Why Johansen-1 mis-fires

GEMPACK's TAB equations are linearized (percent-change form). The
Johansen-1 solver applies the full ``Shock = -10`` as a single linear
step:

```
d(qxs) = J(x_0) · d(shock) · 100%   (single step at SAM)
```

For a CES bottom Armington with σ = 4.64, the level response is

```
qxs_new / qxs_0 = (pim_new/pim_0)^σ · (pms_0/pms_new)^σ · (qim_new/qim_0)
```

The exponential is convex; for a 10% pms drop the linear truncation
under-estimates the response by:

```
e^(0.464) ≈ 1.59 vs. 1 + 0.464 = 1.464   →  Johansen-1 misses ~13%
```

Multi-step Richardson extrapolation re-evaluates the Jacobian at
intermediate points and recovers the non-linear pickup.

## What was added

- `scripts/gtap_v62/run_gempack_exp1a_multistep.py` — minimal driver
  that stages a workdir, writes a CMF with
  ``Method = Gragg; Steps = 2 4 6``, runs ``gtap.exe`` and converts
  the ``.sl4`` to ``.har``. Output:
  ``runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB246/Exp1a_GB246-upd.har``.

- `validate_v62_parity.py` now reads the Gragg reference if present
  and prints a second comparison block alongside the Johansen-1
  table. The JSON report has both `comparisons_johansen1` and
  `comparisons_gragg246` keys.

## What's needed to close the residual ~3pp

The remaining ~2.5pp Python under-shoot on bilateral US→EU trade
(VIMS/VIWS/VXMD) likely comes from a combination of:

1. **Gragg 2-4-6 is not fully converged either.** GEMPACK's
   recommended setting for accurate non-linear response is either
   `automatic accuracy = yes` (adaptive substeps) or
   `Steps = 4 8 12`. Worth re-running to see whether the GEMPACK
   reference itself shifts another ~1pp upward.

2. **Margin layer dynamics.** The current model uses additive
   `pmcif = ps + pwmg`. Phase 3.6 changed eq_pmcif to use `ps`
   (basic supply price), but GEMPACK's FOBCIF identity is
   share-weighted (`pcif = FOBSHR·pfob + TRNSHR·ptrans`). For
   benchmark these are equivalent; for shocks they differ if
   pwmg/ptrans moves significantly. Worth a fresh derivation to
   check.

3. **Armington upper-nest substitution into composite imports
   (`qim`).** The composite import quantity rises by +4.7% in our
   solve. If `esubd` interpretation differs between v6.2 GEMPACK
   and our calibration, the qim response could be off and propagate
   through to qxs.

The first hypothesis is the cheapest to test — re-run Gragg with
more steps and see if the GEMPACK reference shifts.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"

# 1. Build the multi-step GEMPACK reference (one-time, ~5s):
python scripts/gtap_v62/run_gempack_exp1a_multistep.py `
    --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB246

# 2. Run Python and compare against both references:
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

The validator prints two side-by-side comparison tables: against
GEMPACK Johansen-1 (the BOOK3X3 default) and against GEMPACK
Gragg 2-4-6 (closer to the levels equilibrium).

## Lesson learned

When validating a levels-MCP CGE against GEMPACK, **always compare
against a multi-step GEMPACK reference**. Johansen-1 (the BOOK3X3
default) is fine for tutorial purposes but understates non-linear
amplifications for shocks above ~5%. The naive comparison made our
Python implementation look 13–19pp wrong (Phase 3.7) and then 9pp
wrong (Phase 3.8). With the right reference, the actual remaining
gap is 2-3pp.
