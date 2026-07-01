# Plan: pft anchor surgery in the solver

## EXECUTED 2026-06-28 — RESULT: fix as specified WORSENS 3x3, reverted.

Gated the altertax skip behind `EQUILIBRIA_GTAP_HOLDFIX_PFT` (deactivate eq_pfteq +
fix pft at init≈1.0). Check converged CLEAN (code=1, residual 6e-11) — squareness was
correct, the flag worked technically. **But match% fell 72.99% → 60.68% (−12pp).**

Root cause of the worsening — the §5 OPEN QUESTION resolved the wrong way: **GAMS pft in
the CHECK is NOT 1.0**. Measured GAMS pft check = 0.94–1.04 per cell (the anchors move
3–5% in the check). So "prior period = benchmark, pft=1.0" was a FALSE assumption.
Fixing pft=init=1.0 flat pins it at the WRONG value → worse equilibrium.

The MECHANISM is right (deactivate eq_pfteq + fix pft, 1:1 self-squaring, converges);
the VALUE is wrong. The faithful holdfix value = Python's PRIOR-STAGE RESOLVED pft — but
the altertax pipeline runs NO full base solve (diff_altertax.py:386), so there is no
resolved prior pft≠1.0 to hold to. That is the real hole: GAMS gets pft≈0.97-1.03 in the
check because it solves base→check sequentially and the prior period is a SOLVE, not the
flat benchmark. Python's prior is the flat benchmark (pft=1.0), so holding it is wrong.

CONCLUSION: do NOT pursue the pft-fix-at-init route (reverted). The next faithful angle
is to give Python a resolved prior period (a real base solve under altertax CD) so the
holdfix value is 0.97-1.03 like GAMS — a bigger change than this skip-flag. Until then,
leaving pft FREE + eq_pfteq active (root-form, 72.99%) is the best faithful result.

---

# (original SPEC below)


## TL;DR — the surgery is smaller than expected

The solver **already has** a "pfteq free-row + fix pft" block (run_gtap.py:2138-2170)
that does EXACTLY the faithful fix — deactivate `eq_pfteq` + fix `pft` at init (=1.0),
1:1 by construction. It is **explicitly skipped for the altertax closure** (line 2156-2157
`if _is_altertax_closure: continue`). That skip is the design decision my measurement
showed produces the damage (pft lands ~0.55 instead of ~1.0).

So the candidate surgery is NOT "inject the 25-var sequence holdfix" (my 3 manual
attempts failed on squareness). It is: **lift / narrow the altertax skip in the existing
block** so pft gets the free-row + fix treatment. This is inherently square (the block
deactivates the paired row for every pft it fixes) and lands pft on the value I measured
converges clean (code=1).

---

## 1. Where conditional-fixing runs, and the insertion gap

`_run_path_capi_nonlinear_full` (run_gtap.py:1964) sequence:

| Line | Step |
|------|------|
| 2006-2008 | `GTAPSolver(...)` + `apply_closure(closure_config)` |
| 2009 | `apply_conditional_fixing()` ← fixes the ~18 extra anchor cells (off-diagonal pa, zero-flow trade) |
| 2015 | `apply_aggressive_fixing_for_mcp()` ← fixes structural unmatched vars |
| 2031-2051 | altertax: deactivate `eq_xft` for mobile factors |
| 2064 | `apply_squareness_patches(...)` |
| 2076-2120 | residual-region yi fix + eq_walras deactivate |
| **2138-2170** | **pfteq free-row block — SKIPPED for altertax (2156-2157)** ← the surgery site |
| 2175-2179 | `apply_solution_hint(solution_hint)` (warm-start) |
| 2234-2265 | price/income lower bounds |
| **2298-2308** | collect `constraints` (active) + `free_variables` (not fixed) ← **the squareness count** |
| 2316-2365 | Hopcroft-Karp structural matching (forced GAMS pairs) |
| 2367 | **if `len(constraints) != len(free_variables)` → return failed, residual=inf** |
| 2597 | `solve_nonlinear_mcp(...)` |

**The gap** = anywhere between 2009 (conditional-fixing done) and 2298 (count taken).
The existing pft block at 2138 already sits in that gap, AFTER conditional-fixing — so
its fixing count is STABLE (this is exactly what my manual diff_altertax injection got
wrong: it ran BEFORE conditional-fixing, so 18 anchor cells got double-counted).

---

## 2. How fixed/free state is inspected at the surgery point

At line 2298-2308 the solver itself reads the definitive state:
```python
constraints = sorted(model.component_data_objects(Constraint, active=True, ...))
free_variables = sorted(v for v in model.component_data_objects(Var, active=True) if not v.fixed)
```
- a row counts iff `.active` (deactivating drops it)
- a var counts iff `not .fixed` (fixing drops it)
- squareness gate at 2316/2367: `len(constraints) == len(free_variables)`

So to verify squareness of any holdfix, count active `eq_*` rows vs non-fixed var cells
AT THIS POINT (after conditional-fixing + aggressive + squareness patches), NOT at build
time. The existing pft block is self-squaring: for each `pft[r,f]` it fixes, it
deactivates the SAME-indexed `eq_pfteq[r,f]` (lines 2159+2163) → +0 to the gap.

---

## 3. Source of prior-period values (confirmed, not assumed)

The check solve is called (diff_altertax.py:676) with `solution_hint=warm_b` where
`warm_b = GTAPVariableSnapshot.from_python_model(m_chk)` (line 602). m_chk is built with
`t0_snapshot=m_b` (line 447), and **m_b is the benchmark model with getData init values**
(diff_altertax.py:428: "pf=1/(1-kappa), pft=1, xf=evfb*(1-kappa)"). GAMS altertax does
NOT run a full base solve (line 386) — the prior period IS the benchmark.

**Therefore the prior-period value of pft is its init = 1.0**, available directly as
`model.pft[r,f].value` at the surgery point (verified: all pft init = 1.0). The existing
block at 2160 already reads exactly this: `_pft_val = float(_pft_vd.value) ... or 1.0`.
**No GAMS values needed — fixing pft at its own init is the faithful holdfix** (=GAMS
var.fx(tsim-1)=var.l(tsim-1) with the prior period = benchmark).

For the broader 25-var set (pf/xf/pa/...), the source is the same: their init values in
m_chk (from t0_snapshot=m_b). But the 25-var holdfix is NOT recommended (see §5).

---

## 4. var↔row pairing for 1:1 squareness (against the real post-fixing state)

The existing pft block is already 1:1 by construction. For the record, the broader
anchor set's pairing (verified post-closure earlier this session):

| GAMS var | Python var | paired row | cells (post-closure) | 1:1? |
|----------|-----------|------------|----------------------|------|
| pf | pf | eq_pfeq | 36 ↔ 36 | OK |
| xf | xf | eq_xfeq | 36 ↔ 36 | OK |
| pa | pa | eq_paa | 57 ↔ 57 (build) | **off by ~18 at solve** (conditional-fixing fixes off-diagonal pa) |
| pe | pe | eq_peeq | 27 ↔ 27 | OK |
| pmcif | pmcif | eq_pmcifeq | 27 ↔ 27 | OK |
| pm | pm | eq_pmeq | 27 ↔ 27 | OK |
| pabs | pabs | eq_pabs | 3 ↔ 3 | OK |
| pfact | pfact | eq_pfact | 3 ↔ 3 | OK |
| pwfact | pwfact | eq_pwfact | 1 ↔ 1 (scalar) | OK |
| **pft** | **pft** | **eq_pfteq** | **15 ↔ 15** | **OK (self-squaring block)** |

The `pa` row is exactly why the 25-var injection broke: `apply_conditional_fixing`
fixes the off-diagonal pa cells, so a build-time count of 57 active eq_paa rows is wrong
by ~18 at solve time. **The pft-only fix avoids this entirely** — pft has no
conditional-fixing interference (all 15 cells free until this block).

---

## 5. Over-restriction risk: 3x3 vs 5x5 (and how GAMS avoids it)

**omegaf = 1.0 for ALL factors in BOTH 3x3 and 5x5** (measured). So the CET nest is
finite-elastic (not Leontief, not perfect-mobility) identically in both. The pfteq
equation is `pft^2 = Σ gf·pfy^2` in both. There is NO size-dependent binding difference
in omegaf — the anchor is structurally the same.

GAMS avoids over-restriction NOT by size but by the holdfixed=1 mechanism: it freezes
pft at the PRIOR period (benchmark =1.0) in EVERY period, then the free-row pfteq is just
verified ex-post. It never "over-restricts 3x3" because fixing pft=1.0 IS consistent with
the benchmark in every dataset (pft=1 at the calibration point by construction).

**Risk assessment for the pft-only surgery:**
- 3x3 currently converges (code=1, 72.99%) with pft FREE + eq_pfteq active (the skip).
  My measurement showed fixing pft=GAMS (≈1.0) + deactivating eq_pfteq ALSO converges
  clean (code=1, residual 1.77e-11). So **fixing pft=1.0 does NOT break 3x3** — both
  branches converge; the fixed one lands on the GAMS value.
- **OPEN QUESTION the spec cannot answer without running:** does fixing pft=init=1.0
  (not pft=GAMS) land on the same clean solve? init=1.0 ≈ GAMS≈1.0, so very likely, but
  the benchmark init may differ from the GAMS check value by the same ~residual the rest
  of the gap rides. This is the ONE thing to measure first when we execute.
- Larger datasets (10x7, 15x10): the memory says the anchor "binds in 5x5+". If pft=1.0
  is the benchmark value and GAMS holds it there, fixing it should help, not hurt — but
  this MUST be verified per-dataset, not assumed (the memory's "binds in 5x5+" warning).

**Conclusion on over-restriction:** the pft-only fix is LOW risk for 3x3 (measured: both
branches converge) and is the faithful GAMS mechanism. The 25-var holdfix is HIGH risk
(squareness fragility + over-restriction of vars that conditional-fixing already touches)
and is NOT recommended.

---

## Proposed surgery (for approval — NOT yet applied)

**Minimal, lowest-risk:** narrow the altertax skip at run_gtap.py:2156-2157 so the
existing pfteq free-row block runs for altertax too:
```python
# CURRENT (line 2156-2157):
if _is_altertax_closure:
    continue  # altertax: pfteq stays active, pft stays free
# PROPOSED: remove the skip (or gate it behind a flag) so altertax ALSO gets
#   _eq_pfteq_vd.deactivate() + _pft_vd.fix(init≈1.0)
```
This deactivates eq_pfteq + fixes pft at init, 1:1, self-squaring, at a point AFTER
conditional-fixing — exactly where my manual injection should have been.

**Execution order when approved:**
1. First MEASURE: gate the change behind a flag (`--holdfix-pft` or env), run 3x3,
   confirm code=1 AND match% ≥ 73% (the §5 open question: does init=1.0 ≈ GAMS work).
2. If 3x3 holds/improves → run 5x5, 10x7 (the "binds in 5x5+" datasets).
3. Only if all converge + improve → make it the default for altertax (remove the skip).
4. Regression gate (nl-parity 5/5) at every step.

**Do NOT** remove the skip unconditionally without step 1 — the §5 open question
(init=1.0 vs GAMS pft) is unmeasured.
