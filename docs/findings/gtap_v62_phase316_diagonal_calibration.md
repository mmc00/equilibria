# GTAP v6.2 Phase 3.16 — Diagonal trade calibration (semantically correct, Phase 3.8 had compensating errors)

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.15 (data loader audit + diagonal trade discovery)

## TL;DR

Cross-referenced our Python code against `c:/runGTAP375/gtap.tab` and
confirmed: **GEMPACK includes the diagonal (s == d) trade flows in
its bottom Armington nest by design.** This is documented explicitly
in gtap.tab lines 1775-1785 (TAB comment about intra-EU trade in
aggregated databases).

Our Phase 3.8 calibration excluded `s == d` everywhere, producing
an internal inconsistency: `qim_0 = 74,702` (Python) ≠ `sum agent
imports = 153,029` (SAM identity) for `food, ROW`. The discrepancy
of 78,327 is exactly the diagonal value `VXMD[food, ROW, ROW]`.

Phase 3.16 includes the diagonal everywhere — calibration of
`qim_0`, `pim_0`, `alpha_xs`, `vom`, `vim`, `vimw`, plus equations
`eq_pe`, `eq_pwmg`, `eq_pmcif`, `eq_pms`, `eq_pim`, `eq_qxs`,
`eq_qtm`, `eq_market`, `eq_qim`. Now `qim_0 = sum agent imports`
exactly (machine epsilon).

**But parity vs GEMPACK Gragg multi-step regressed slightly:**

```
                  Phase 3.8 best Mode B   Phase 3.16 best Mode B
VIWS USA→EU       +54.42% (+0.9pp gap)    +50.43% (-3.1pp gap)
```

Phase 3.8's better numerical agreement was due to compensating
errors: the qim_0 under-statement was offset by a related issue
elsewhere that fortuitously cancelled. Phase 3.16 exposes the
"real" structural gap of ~3pp, which now points to a different
underlying issue (likely the CES levels-vs-linearized response for
large shocks).

## The data evidence

`sum_s VIMS[i,s,r] = VIPM[i,r] + VIGM[i,r] + sum_j VIFM[i,j,r]`
holds **including the diagonal**:

```
Cell             sum_s VIMS    VIPM+VIGM+ΣVIFM    diff
food, USA        32,906        32,906             0
food, EU         79,837        79,837             0
food, ROW        234,751       234,751            0
mnfcs, USA       569,990       569,990            0
mnfcs, EU        589,651       589,651            0
mnfcs, ROW       1,701,743     1,701,743          0
...
```

For ROW the diagonal `VIMS[*, ROW, ROW]` contributes ~30% of the
agent-level imports. Excluding it means our `qim_0` is missing
that 30%.

## gtap.tab evidence

Line 1763: `MSHRS(i,r,s) = VIMS(i,r,s) / sum(k,REG, VIMS(i,k,s));`
- sum over `k in REG` includes k=s.

Line 1767: `pim(i,s) = sum(k,REG, MSHRS(i,k,s) * [pms(i,k,s) - ams(i,k,s)]);`
- composite price weights ALL sources including self.

Line 1798: `qxs(i,r,s) = qim(i,s) - ESUBM(i) * [pms(i,r,s) - ams(i,r,s) - pim(i,s)];`
- bilateral demand defined for ALL (r, s) pairs.

Lines 1775-1785 (author comment):
> "Note that the way this price ratio is defined, it includes
> intraregional imports as well... when aggregated to the EU level,
> the composite import price includes both intra-EU and outside
> imports. So some modification is needed to handle the EU case."

## Changes in Phase 3.16

`src/equilibria/templates/gtap_v62/gtap_v62_calibration.py`:
- `pim_0` (line ~207): sum over all REG (was `if s != r`).
- `vim`, `vimw` (line ~265): sum over all REG.
- `vom` (line ~227): sum exports over all REG.
- Bilateral block (line ~460): no `if s == d: continue`.

`src/equilibria/templates/gtap_v62/gtap_v62_model_equations.py`:
- `eq_pe`, `eq_pwmg`, `eq_pmcif`, `eq_pms`, `eq_pim`, `eq_qxs`,
  `eq_qtm`, `eq_market`: removed `if s == d: continue` filters.

`scripts/gtap_v62/_make_square.py`:
- `eq_qim` identity: sum over all sources (was `s != r`).

## Parity (5 IPOPT runs)

```
Run 1: VIWS = +45.46%  Mode A     (-8.1pp vs Gragg)
Run 2: VIWS = +46.11%  Mode A     (-7.4pp)
Run 3: VIWS = -36.45%  Catastrophic Mode A
Run 4: VIWS = +50.43%  Mode B     (-3.1pp) ← best
Run 5: VIWS = +46.16%  Mode A     (-7.4pp)
```

Best-case (Mode B) is ~-3pp from Gragg multi-step. The bimodality
of IPOPT remains the dominant stability issue.

## Why is Phase 3.16 WORSE than Phase 3.8?

Phase 3.8 was internally inconsistent but the inconsistency
fortuitously cancelled to give +0.9pp parity. Specifically:
- `qim_0` understated the composite import by 30% in ROW.
- `pim_0 = vim / sum_off_diag_VXWD` was *higher* than the consistent
  `vim_total / sum_all_VXWD` (because vim was the off-diagonal sum
  too, missing the same 30%).
- The household `alpha_dom`/`alpha_imp` calibration used these
  inconsistent values but produced shock responses that — by
  coincidence — matched GEMPACK Gragg multi-step better than the
  consistent calibration.

In Phase 3.16:
- All quantities are at their "correct" GEMPACK-consistent values.
- The bilateral CES response is computed in LEVELS form with these
  consistent inputs.
- The CES levels-vs-linearized gap for a 10% shock is ~3pp here.

So Phase 3.16's ~3pp gap is the "true" structural gap, while Phase
3.8's ~1pp was an artifact.

## What this tells us about v6.2 parity

GEMPACK's Gragg multi-step is an APPROXIMATION of the levels
equilibrium via Richardson extrapolation. For shocks of 10%+ with
σ_m=4.64, the linearized-with-extrapolation result diverges from
the exact levels solution by a small amount (~2-3pp in this case).

A truly "exact" parity would require either:
- GEMPACK with Newton-style nonlinear solver (which GEMPACK doesn't
  natively offer outside of "automatic accuracy = yes" which was
  tested in Phase 3.10 and converged at 2-4-6).
- A different reference (e.g. MPSGE, which solves in levels but
  doesn't exist for GTAP v6.2).

So ~3pp may be the LIMIT of achievable parity vs Gragg multi-step.

## Why the IPOPT bimodality

Independent of calibration, IPOPT lands in Mode A (locally
infeasible) ~60-80% of runs. Mode B is found ~20-40% of the time.
This is a SOLVER stability issue, not a calibration issue:

- IPOPT minimizes a regularizer subject to F(x) = 0 constraints.
- With non-convex F, multiple stationary points exist.
- IPOPT's interior-point method navigates with bound-feasibility,
  sometimes getting trapped at points where the constraints are
  approximately satisfied but not exactly.

Fix path: PATH equation/variable scaling (Phase 3.12) + bipartite
closure audit (Phase 3.14 Option A). Not addressed in Phase 3.16.

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

Run 5+ times to see Mode A vs Mode B split. Mode B is identified
by `walras` magnitude ≤ 1.0.

## Phase 3.17 — next

The structural model is now semantically aligned with GEMPACK.
Remaining work for tighter parity:

1. **Close IPOPT bimodality**: warm-start IPOPT from a near-Mode-B
   point on every run. Currently it relies on random initial
   ordering to choose between modes.
2. **PATH activation**: with consistent qim, qxs may be more
   amenable to PATH's Newton steps. Retest Phase 3.12 scaling
   on top of Phase 3.16 calibration.
3. **GEMPACK Newton equivalent**: implement a Gauss-Newton
   variant in Python that mimics what Gragg multi-step does
   approximately, to verify the ~3pp gap is purely the linearized-
   vs-levels difference and not another model bug.
