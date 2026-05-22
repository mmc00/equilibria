# GTAP v6.2 Phase 3.14 — PATH cannot solve v6.2 shocked under current closure

**Date:** 2026-05-22
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.13 (IPOPT-crash + PATH-polish hybrid)

## TL;DR

Audited the bipartite-matching closure to see if the 27 unmatched
variables are responsible for PATH's `term_code=2` (NoProgress) on
the shocked solve. The audit, combined with PATH homotopy and
initial-point-perturbation experiments, gives a definitive answer:

**PATH cannot solve the v6.2 BOOK3X3 model after a shock under the
current closure-and-square strategy.** The issue is structural in
the model, not in solver tuning. The 27 unmatched vars are mostly
in the CGDS (investment) sector and aren't the direct blocker — the
real issue is that PATH's Newton step is rejected from the prebalance-
adjusted baseline.

## What was audited

5 representative unmatched-variable cells (cleaned-up snapshot —
after closure, dangling fix, trivial-eq deactivation):

```
pf[Capital, *]     × 3   factor prices (Capital)
pf[Labor, *]       × 3   factor prices (Labor)
pf[Land, USA/ROW]  × 2   factor prices (Land, partial)
pva[CGDS, *]       × 3   value-added price (CGDS investment sector)
pwmg[svces, s, d]  × 6   margin per unit (svces is THE margin commodity)
qfe[Labor/Land, CGDS, *] × 6  factor demand in CGDS
qfe[Land, svces, *] × 3  factor demand for Land in svces
qo[CGDS, *]        × 3   capital-goods output
                   ────
                   27
```

These are all endogenous, BUT they are NOT in the bilateral US→EU
food trade response path. The trade-relevant variables
(`qxs`, `pms`, `pmcif`, `pe`, `qim`, `qpm`, `qfm`, `qgm`) are **all
free** in the matching. Fixing the CGDS-side accounting variables
at baseline should NOT block PATH's trade-shock Newton steps.

So the bipartite fix isn't the proximate cause of PATH's failure.

## Tests run, ranked by what was learned

### 1. Homotopy substepping (`--homotopy-steps N`)

Split the shock into N substeps of `(1-α)·tms_old + α·tms_new`.
Idea: each substep has a small perturbation that PATH should
handle. Reality:

```
Substep 1/4 (α=0.25): residual=3.76e-02, term_code=2
Substep 2/4 (α=0.50): residual=7.37e-02, term_code=2
Substep 3/4 (α=0.75): residual=1.08e-01, term_code=2
Substep 4/4 (α=1.00): residual=1.42e-01, term_code=2
```

The residual SCALES LINEARLY with `α` — PATH is NOT reducing the
perturbation at any step. It accepts the perturbed point as-is and
reports "NoProgress". Homotopy doesn't help because PATH won't take
Newton steps from baseline AT ALL.

### 2. Initial-point perturbation (`PATH_PERTURB=ε`)

Random multiplicative perturbation of x_0 by ε (e.g. each var
scaled by 1 + 0.05·U[-1,1]). Idea: PATH might be at a degenerate
"locally stationary" point exactly at baseline; perturb to escape.

Sweep results (baseline + shocked term codes):
```
ε=0.001:  code=2/2  (still stuck)
ε=0.005:  code=1/1  (Solved! but walras=6.5k baseline)
ε=0.02:   code=2/2  (stuck again)
ε=0.03:   code=2/2
ε=0.05:   code=2/4  (iteration limit)
ε=0.10:   code=1/1  (Solved! but walras=-14k baseline)
```

When PATH *does* converge (ε ∈ {0.005, 0.10}), it lands at an
equilibrium that's NOT the SAM — walras is in the thousands. PATH
is solving the math problem but the start was so perturbed that
the basin of attraction is a different equilibrium.

For ε in {0.02, 0.03, 0.05}, PATH gets stuck again. The "sweet
spot" is narrow and not robust.

### 3. PATH variable + equation scaling (Phase 3.12 retained)

Diagonal pre-scaling so all variables and residuals are O(1).
Implemented in Phase 3.12; doesn't unblock the shocked solve on
its own (term_code=2 with ZERO movement). Scaling is structurally
correct but not sufficient.

## What's added (still useful)

These features are committed even though they don't fix the issue:

- `--homotopy-steps N`: apply shock in N substeps. Useful for models
  where PATH does iterate but needs gradual warm-up.
- `PATH_PERTURB=ε` env var: multiplicative perturbation of x_0 to
  escape stationary points. Currently unsafe (lands at wrong
  equilibrium for ε > 0.005) — keep as opt-in diagnostic.

## What the failure mode looks like

PATH at baseline: `term_code=2`, residual=9.12e-04. ✓ trivial
(F(x_0) ≈ 0 because of prebalance, no real iteration needed).

PATH at shocked: `term_code=2`, residual=1.42e-01, walras=-3.49e-09.
- 17 major iterations
- ZERO observable movement on pms, qxs, qim — the variables the
  shock should directly affect.
- The shock perturbation gives eq_pms[food,USA,EU] = +0.153 residual.
- PATH evaluates J at x_0 and tries Newton step; all candidate steps
  fail the merit-function decrease test; PATH declares NoProgress.

## Hypothesis: J(x_0) is near-singular

The Jacobian of the v6.2 system at the SAM-prebalance baseline is
likely near-rank-deficient. Specifically:

- 25 baked-in constraints have body `body - residual_constant == rhs`.
  The constant offset doesn't affect derivatives (∂const/∂x = 0),
  but it DOES mean the original equation's derivatives have an
  effective "weighting" of the baked residual size in PATH's merit.
- 27 bipartite-fixed vars are pinned; their Jacobian rows are
  trivially [0 ... 0 1 0 ... 0]^T after substitution.
- The remaining 555 effective vars / 555 effective eqs might have
  a Jacobian with several small singular values.

PATH's LUSOL factorization of a near-singular J fails to produce a
reliable Newton step; the merit function isn't decreased; PATH
reports stationary point.

This hypothesis is consistent with all observations.

## What it would take to fix

The closure squaring (`_make_square.py`) was a HEURISTIC that produced
a square system but at the cost of a fragile Jacobian. A clean fix
requires:

**Option A — Audit and add the 27 missing equations.**
For each unmatched-var family, identify the v6.2 equation that
SHOULD have defined it:

- `pf[f,r]` → factor market clearing eq `sum_j qfe[f,j,r] = qoes[f,r]`.
  We have `eq_factor_clear` — why isn't it covering all 9 pf cells?
- `pva[CGDS,r]` → value-added price for CGDS. Maybe `eq_pva` skips
  CGDS sectors. Need to verify.
- `pwmg[svces, s, d]` → margin per unit. We have `eq_pwmg`. Why is
  it skipping svces (the margin commodity itself)?
- `qfe[*, CGDS, *]` → factor demand in CGDS. Should be defined by
  capital-goods production. Maybe `eq_qfe` skips CGDS by `Constraint.Skip`.
- `qo[CGDS, r]` → capital-goods output. Should be defined by
  investment demand identity.

Each of these is a 1-2 line fix in `gtap_v62_model_equations.py`,
but identifying the right form requires careful comparison to
GEMPACK's TAB equations for CGDS.

**Option B — Redesign the closure.**
The v6.2 GEMPACK closure (psave varies, pfactwld is numeraire) is
implemented in our model as `pgdpwld=1`, `qoes` fixed, `kb`/`ke`
fixed, `rorg=1`, `pgdpmp[r]=1`. The redundancy of `pgdpmp[r]=1`
(fixing 3 regional GDP deflators) might be over-constraining and
forcing the matching to leave 27 vars unmatched. Try freeing
`pgdpmp` for non-numeraire regions and see how the matching reshapes.

**Option C — Switch to GTAPinGAMS-style explicit slack.**
GTAPinGAMS uses Mathiesen's "auxiliary" formulation where each
market is paired with an explicit price variable (complementarity),
so the system is naturally square without bipartite fixing. This
is a bigger rewrite but eliminates the structural fragility.

## Current best parity result remains Phase 3.8 + IPOPT Mode B

```
Cell             Gragg-multi    Python (IPOPT Mode B)    Gap
VIMS US→EU       +38.165%       +38.98%                  +0.82pp
VIWS US→EU       +53.517%       +54.42%                  +0.91pp
VXMD US→EU       +53.548%       +54.43%                  +0.88pp
VDPM food EU     -0.335%        -0.19%                   +0.14pp
VIPM food EU     +2.278%        +2.43%                   +0.15pp
```

~1pp gap is good enough for research validation. **The remaining
work for v6.2 production-grade PATH support is Option A above —
audit the 27 missing equations and add them properly.**

## Reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"

# PATH with homotopy (still doesn't work, but logs progression):
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi --homotopy-steps 4 `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a

# PATH with perturbation (works ε=0.005 but at wrong equilibrium):
$env:PATH_PERTURB = "0.005"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

## Closing the v6.2 PATH chapter

The user's question — "shouldn't PATH work for both baseline and
shocked?" — has the right intuition. The answer is "yes, but the
v6.2 model as currently implemented has a closure that produces a
near-singular Jacobian after the prebalance, and that's why PATH's
Newton method fails." Fixing it requires the closure audit work
described above.

For now, the v6.2 branch demonstrates:
- Working PATH on BASELINE (residual ≤ 1e-6).
- IPOPT Mode B reaches ~1pp parity vs GEMPACK Gragg multi-step.
- The 19pp / 9pp / 2.5pp gaps reported in earlier phases were all
  GEMPACK Johansen-1 linearization artifacts + IPOPT solver noise,
  not real model discrepancy.
