# GTAP v6.2 Phase 3.15 — Data loader audit + diagonal trade discovery

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.14 (PATH closure audit)

## TL;DR

The user asked: "is the data loader missing a field?" The audit
confirmed the loader correctly reads all V*/E* SAM headers, but
**uncovered a real semantic difference between Python and GEMPACK**:

- GEMPACK's bottom Armington formula sums over `k ∈ REG` (ALL
  regions, including the importer itself).
- Our Python explicitly excludes `s == d` via `if s == d: continue`
  in every bottom-Armington equation.

For BOOK3X3, the **diagonal trade** in the SAM is huge — ~1,003,769
in VXMD (31% of all trade), all concentrated in the aggregated
"ROW" region. Excluding it means our `qim[food, ROW]` composite is
missing 31% of its true value.

A test "include diagonal everywhere" patch was implemented and
tested — it made parity WORSE (under-shoots Gragg-multi by ~7pp vs
the prior ~3pp). The naive inclusion is wrong; GEMPACK's handling
has additional subtleties. **Patch reverted.**

The data loader itself is correct; the diagonal handling is a
semantic model issue, not a missing-field issue.

## What was confirmed about the data loader

BOOK3X3's `basedata.har` headers (24 total):
```
EVFA, EVOA, VDFM, VDFA, VIFM, VIFA, VDPM, VDPA, VIPM, VIPA,
VDGM, VDGA, VIGM, VIGA, VXMD, VXWD, VIMS, VIWS, VST, VTWR,
VFM, VKB, VDEP, SAVE
+ metadata: DPSM, DVER
```

The `GTAPv62Parameters.load_from_har()` reads ALL of them. The
metadata headers (DPSM = data prep status, DVER = data version)
are correctly not loaded as value flows.

`Default.prm` headers:
```
ESBD, ESBM, ESBT, ESBV, ETRE, INCP, SUBP, RFLX, SLUG
+ RDLT (RORDELTA — dynamic closure flag, unloaded but not needed
        for the static Exp1a test)
+ metadata: DVER, XXCD, XXCR, XXHS
```

`ESBT` (top CES) = 0 across all sectors — this is the canonical
v6.2 convention (Leontief top nest, no substitution between VA and
intermediates). The model's `_eps_sigma()` check correctly handles
σ=0 by switching to the linear/Leontief branch.

## The diagonal trade discovery

Cell-by-cell SAM identity audit confirmed:

```
VIWS(i,s,d) = VXWD(i,s,d) + sum_m VTWR(m,i,s,d)    ✓ machine epsilon
VXMD - VXWD = export tax wedge (txs)               ✓ matches calibration
```

BUT: BOOK3X3 has **non-zero diagonal entries** (s == d) for the
ROW region:

```
                       VXMD       VIMS       VTWR (sum_m)
(food, ROW→ROW)        77,794     118,099    7,750
(mnfcs, ROW→ROW)       765,480    914,030    58,089
(svces, ROW→ROW)       160,494    160,888    0
```

Economic interpretation: in aggregated SAMs, "ROW" combines many
real countries; trade BETWEEN those subregions is captured as
self-trade. GEMPACK keeps these flows in its bilateral Armington
nest; our Python skips them.

## GEMPACK gtap.tab convention (lines 1767, 1798)

```
pim(i,s) = sum(k,REG, MSHRS(i,k,s) * [pms(i,k,s) - ams(i,k,s)]);
qxs(i,r,s) = qim(i,s) - ESUBM(i) * [pms(i,r,s) - ams(i,r,s) - pim(i,s)];
```

`sum(k,REG, ...)` iterates ALL regions in REG — including k=s.
Self-imports are explicit sources in the CES composite.

## Test: include diagonal everywhere (REJECTED)

Implemented patch to remove `if s == d: continue` filters from:
- `eq_pe`, `eq_pwmg`, `eq_pmcif`, `eq_pms`, `eq_pim`, `eq_qxs`
  (bilateral price + demand equations)
- `eq_qtm` (margin demand)
- `eq_market` (commodity market clearing)
- `eq_qim` (composite import identity in `_make_square`)
- Calibration helpers (`vom`, `vim`, `vimw`, `pim_0`, alpha_xs)

Parity result (5 IPOPT runs):

```
Run  VIWS Python    vs Gragg (53.52)    vs Johansen-1 (41.54)
1    +45.42%        -8.1pp              +3.9pp
2    -99.97%        catastrophe          —
3    +44.18%        -9.3pp              +2.6pp
4    +44.17%        -9.3pp              +2.6pp
5    +44.88%        -8.6pp              +3.3pp
```

Verdict: parity vs Gragg-multi DEGRADES from -3pp (Phase 3.8) to
-8.6pp (Phase 3.15 patch). The naive "include s==d" change is the
wrong fix — GEMPACK's handling is more nuanced than a uniform
sum-over-all-REG.

Possible nuances we haven't replicated:
- `amgm[m,i,r,r]` (margin shares for self-trade) may be 0 in
  GEMPACK even when VTWR[m,i,r,r] is non-zero (treating intra-
  region transport differently).
- The cost share `MSHRS` for self-imports may be calibrated from
  a separate denominator that downweights it.
- VDPM/VDFM (domestic absorption) may already implicitly include
  what we'd otherwise call "self-imports" in some agent flows.

This requires careful study of GEMPACK's TABLO source + the
gtap62map.cmf aggregation step that produced BOOK3X3. Out of scope
for a quick patch.

## What the user's question taught us

> "spechás que algún campo no se está cargando correctamente"

The data loader is fine. But the question prompted a deeper audit
that revealed:

1. The diagonal trade ~1M is REAL in the SAM (not a bug we created)
2. Our model explicitly excludes it (`if s == d: continue` filters)
3. GEMPACK INCLUDES it in its bottom Armington
4. The fix isn't trivial — uniform inclusion makes parity worse

This is a meaningful structural finding even though Phase 3.15
didn't produce a working fix.

## Status of v6.2 parity (unchanged from Phase 3.13)

```
Cell             Gragg-multi    Python (IPOPT Mode B)    Gap
VIMS US→EU       +38.165%       +38.98%                  +0.82pp
VIWS US→EU       +53.517%       +54.42%                  +0.91pp
VXMD US→EU       +53.548%       +54.43%                  +0.88pp
VDPM food EU     -0.335%        -0.19%                   +0.14pp
VIPM food EU     +2.278%        +2.43%                   +0.15pp
```

Mode B is reached ~60% of the time; Mode A is the IPOPT-locally-
infeasible attractor that gives spurious answers.

## Phase 3.16 candidates

1. **Diagonal trade — proper fix**: audit gtap62map.cmf to see
   how BOOK3X3 was aggregated. If the aggregation produces real
   intra-region trade flows, we need to add domestic-self-import
   variables in the Python model. If the diagonals are just
   accounting artifacts, we need to ZERO them out in the SAM as a
   preprocessing step.

2. **PATH closure refactor (Phase 3.14 Option A)**: audit the 27
   bipartite-unmatched cells (pf, pwmg[svces], qfe in CGDS, etc.)
   and add the missing GEMPACK-equivalent equations.

3. **Accept current ~1pp Mode B parity** as the v6.2 research
   milestone, document the limitations, and move on. This branch
   has produced 14 commits worth of structural understanding that
   would inform a future "v6.2 v2" attempt.
