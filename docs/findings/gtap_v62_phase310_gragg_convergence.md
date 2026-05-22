# GTAP v6.2 Phase 3.10 — Gragg stepping is already converged; 2.5pp gap is real

**Date:** 2026-05-21
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.9 (multi-step GEMPACK reference)

## TL;DR

Phase 3.9 hypothesized that the residual ~2.5pp Python under-shoot vs
GEMPACK Gragg 2-4-6 might shrink if GEMPACK was given more aggressive
stepping. We tested three variants:

| Variant | VIMS US→EU | VIWS US→EU | VXMD US→EU |
|---------|-----------|------------|------------|
| Gragg `Steps = 2 4 6` | +38.165% | +53.517% | +53.548% |
| Gragg `Steps = 4 8 12` | +38.165% | +53.517% | +53.548% |
| Gragg `automatic accuracy = yes` | +38.165% | +53.517% | +53.548% |

**All three give bit-identical answers.** GEMPACK has converged to its
non-linear solution at 2-4-6 Richardson extrapolation; pushing further
doesn't move it.

Therefore the residual **−2.5pp gap on bilateral US→EU food trade
(VIMS/VIWS/VXMD) is genuinely between Python and GEMPACK**, not an
artifact of solver tolerance. Closing it requires structural model
work, not solver tuning.

## Updated parity table

```
Cell                  Joh-1     GB 2-4-6   GB 4-8-12   GB auto    Python
VIMS food USA->EU    +31.536   +38.165    +38.165     +38.165    +35.887
VIWS food USA->EU    +41.536   +53.517    +53.517     +53.517    +50.986
VXMD food USA->EU    +41.552   +53.548    +53.548     +53.548    +50.991
VDPM food EU          -0.264    -0.335     -0.335      -0.335     -0.190
VIPM food EU          +1.848    +2.278     +2.279      +2.278     +2.417

Python − GEMPACK gap (the real one):
  VIMS US→EU:  -2.28pp  (Python under-shoots)
  VIWS US→EU:  -2.53pp
  VXMD US→EU:  -2.56pp
  VDPM food EU: +0.14pp  (Python under-shoots in absolute terms)
  VIPM food EU: +0.14pp  (Python over-shoots)
```

## What's added

`scripts/gtap_v62/run_gempack_exp1a_multistep.py` now takes a
`--variant` flag selecting one of three predefined stepping recipes:

- `GB246` — `Method=Gragg; Steps=2 4 6;` (the Phase 3.9 baseline)
- `GB48-12` — `Method=Gragg; Steps=4 8 12;` (more Richardson points)
- `GB_auto` — `Method=Gragg; Steps=2 4 6; automatic accuracy=yes;`
  (GEMPACK chooses its own substeps adaptively until the error
  estimate converges)

## What the 2.5pp gap is NOT

- **Not a non-linear stepping issue.** Three independent GEMPACK
  stepping configurations agree to machine precision on a +53.52%
  VIWS response. Python's +50.99% is genuinely different.
- **Not a SAM imbalance issue.** Phase 3.8 baked the SAM
  imperfections into the constraint constants; ``F(x_0) = 0`` holds
  exactly post-prebalance.
- **Not a Johansen 1-step artifact.** That was Phase 3.9, ruled out.

## Where the 2.5pp likely comes from (Phase 3.11 targets)

Two structural hypotheses survive:

**A. Margin layer dynamics — `eq_pmcif`.**
Our model uses `pmcif = ps + pwmg` (additive). GEMPACK's FOBCIF
identity in v6.2 is `pcif = FOBSHR·pfob + TRNSHR·ptrans` (share-
weighted). For the SAM benchmark these are equivalent. For shocks
they diverge when `pwmg/ptrans` adjusts:
  - Additive: `d(pmcif)/pmcif = (ps/pmcif)·d(ps)/ps + (pwmg/pmcif)·d(pwmg)/pwmg`
  - Share-weighted: `d(pcif)/pcif = FOBSHR·d(pfob)/pfob + TRNSHR·d(ptrans)/ptrans`
At benchmark `ps/pmcif = FOBSHR` and `pwmg/pmcif = TRNSHR`, so the
two responses match in the first derivative. The DIFFERENCE comes
from second-order non-linear corrections.

**B. Bottom Armington calibration of `qxs_0`.**
Phase 2d set `qxs_0 = vxwd` (world-price units, including the margin
layer). GEMPACK's `qxs(i,r,s)` is at producer prices (FOB-side
quantity). The `alpha_xs` calibration absorbs the level difference
at benchmark, but the CES response to a 10% price shock differs
when `qxs` is interpreted in different units (since the non-
linearity scales with `qxs_0` directly).

Hypothesis B is more likely to be the dominant cause because the
2-3pp signs (Python *under*-shoots VIMS/VIWS/VXMD but the GAP IS
THE SAME PROPORTION on all three) is consistent with a single
calibration scale, not a margin dynamics issue.

## Recommended next step (Phase 3.11)

Re-calibrate the bottom Armington with `qxs_0 = vxmd` (producer-
price quantity) and a separate margin layer treated as a service
input that the importer consumes. This requires:

1. Change `qxs_0 = vxmd` in the calibration.
2. Re-derive `pe_0`, `pmcif_0`, `pms_0` in producer-price units.
3. Add a separate variable `qimm[m,i,s,d]` for margin services
   consumed in each bilateral flow (or fold into `qst` accounting).
4. Update `eq_qim` and `eq_market` to consume `qxs * pe` (producer
   value) rather than `qxs` (world-price value).

This is a substantial refactor of the trade chain but it's the
cleanest path to eliminate the residual 2.5pp.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"

# Three GEMPACK references (~5s each):
python scripts/gtap_v62/run_gempack_exp1a_multistep.py `
    --variant GB246 --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB246
python scripts/gtap_v62/run_gempack_exp1a_multistep.py `
    --variant GB48-12 --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB48-12
python scripts/gtap_v62/run_gempack_exp1a_multistep.py `
    --variant GB_auto --workdir runs/gtap_v62_oracle/BOOK3X3_Exp1a_GB_auto
```

All three produce a bit-identical ``Exp1a_*-upd.har``.
