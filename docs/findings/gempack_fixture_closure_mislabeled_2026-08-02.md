# The GEMPACK fixture closure was mislabeled — the GAMS↔GEMPACK "gap" is a closure mismatch

**Date:** 2026-08-02
**Status:** root cause found; fixture generator fixed (fixtures pending regen in GEMPACK env)

## TL;DR

The long-standing ~0.35pp land-price gap (ours/GAMS −3.03% vs GEMPACK −2.68%) and the
"only 96.3%" global match against the GEMPACK fixture were **NOT** a linearization artifact
and **NOT** a defect in our model. The reference fixture `sl4dump_gtap7_3x3_tm10` was
generated with a **non-standard capital-account closure** (`swap dpsave = del_tbalry`) that
its `.cmf` mislabeled "capFix". Our `savf_flag=capFix` — which is the *correct*, standard
closure that the Julia GTAPv7 model uses to reproduce GEMPACK to 5 significant figures — was
being measured against a fixture built with a *different* closure. `run_gempack_matrix.py`
has been corrected to emit the standard closure.

## Evidence chain

1. **Not linearization.** GEMPACK land EU_28/Food is FLAT across step counts
   (s4=−2.6812, s8/s16/s32/s64=−2.6811, converged), and the gap is already present in the
   near-linear limit (0.1% shock: ours −0.3288/% vs GEMPACK −0.3075/%). Linearization error
   would vanish with more steps and at tiny shocks; it doesn't. So it's a first-order
   (base-derivative) difference, not a Gragg/step artifact.

2. **The fixture's rore is EQUALIZED** (−3.4485 all regions) and its **qsave is EXTREME**
   (EU −10%, stable s4–s64). Our capFix gives differing rore and flat qsave (EU +0.3%). These
   are different closures' signatures.

3. **The `.cmf` swaps `dpsave = del_tbalry`** (`run_gempack_matrix.py:make_cmf`): it frees the
   saving DISTRIBUTION (dpsave) and fixes the trade balance — a closure that redistributes
   savings and equalizes returns. Its comment claims "capFix mirroring equilibria's Python
   gate", but capFix (`savf = pigbl·savf_bar`) holds the saving share FIXED. Mislabeled.

4. **Julia settles it.** The Julia GTAPv7 model (levels + Ipopt, reproduces GEMPACK to 5 sig
   figs) uses the **standard** closure: `σyp`/`σyg` (saving/consumption shares) are EXOGENOUS
   (`generate_starting_values.jl:41` fixed list), i.e. the saving share is FIXED — exactly
   like our capFix. Julia's qsave is MODERATE (−0.5 to −2.6), NOT the fixture's −10%. So the
   FIXTURE is the outlier, not Julia and not our capFix.

5. **GEMPACK's own standard `.cmf`** (`gtapv7.cmf`, "Standard GTAP closure") exogenises
   `psaveslack pfactwld cgdslack tradslack au dppriv dpgov dpsave` with **no swaps** — dpsave
   fixed, del_tbalry endogenous. That is the canonical closure, and it matches capFix.

## What we tried (and what it proved)

- **capFlex** (returns equalize, `risk·rore=rorg`): reproduces the fixture's qinv (EU −5.34
  vs −5.80) but global match DROPS to 71.1% — it fits the fixture's investment side but not
  the rest.
- **capFixDp** (betaS endogenous = the fixture's dpsave swap): converges code=1 but the free
  betaS runs away (qsave EU −141%), match 34%. Freeing the saving distribution alone is NOT
  the fixture's closure — that swap also fixes the real `del_tbalry` (trade balance as % of
  world income), an equation our levels model has no explicit form for. (Kept, xfail-marked,
  as a documented negative result.)
- **capFix** (default): 96.3% global match — the best, and the correct standard closure.

## The fix

`scripts/gtap/run_gempack_matrix.py:make_cmf` now emits the **standard GTAP closure** (dpsave
exogenous, no `dpsave↔del_tbalry` swap), verbatim from GEMPACK's own `gtapv7.cmf`. Regenerated
fixtures (pending a run in the GEMPACK/Windows environment) should let our capFix match GEMPACK
like Julia does — near ~99%, not 96.3%.

**Our capFix is the correct closure.** The 3.7% "gap" was measuring the right closure against a
fixture built with the wrong one.

## Files

- `scripts/gtap/run_gempack_matrix.py` — `make_cmf` corrected (standard closure, no swap).
- `src/equilibria/blocks/gtap/demand_utility.py` — capFlex + capFixDp closures added (selectable).
- Attempts log: `dev-tools/equilibria-tools/plans/2026-07-31-gams-gempack-gap-attempts-log.md`.
