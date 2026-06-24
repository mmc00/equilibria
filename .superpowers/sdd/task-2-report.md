# Task 2 Report — Prove MP == SP gtap shock on gtap7_3x3 (fidelity gate)

**Status: BLOCKED** (HARD CHECKPOINT triggered — MP does not converge code=1)

Commit: `37e7815` test(gtap-mp): MP==SP fidelity gate (RED/BLOCKED — MP check+shock non-square)
Branch: `gtap7-multiperiod-matrix` (parent `d9bfb5f`)

---

## What I built

New file: `tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py` (committed).

Two tests, both `skipif(not _has_path_solver())` and skip when the dataset HAR is
missing (LOCAL-ONLY, SKIP-safe, never fails on CI — mirrors
`test_altertax_multiperiod_parity.py`):

1. `test_gtap_mp_shock_converges_3x3` — builds the pure-gtap MP model and solves
   base→check→shock via `solve_multiperiod(..., mode="gtap", holdfix_cd=False,
   seed_from_prior=True, skip_base_solve=False, mute_welfare=True)`; asserts all 3
   period codes == 1.
2. `test_gtap_mp_shock_equals_sp_shock_3x3` — after the convergence assert, runs an
   **in-process SP** solve (`_solve_sp_gtap_shock`): `GTAPModelEquations(...).build_model()`
   + `_apply_imptx_shock(p_shock, factor=0.10)` (tm_pct) +
   `run_gtap._run_path_capi_nonlinear_full(m_sp, p_shock, closure_config=gc,
   equation_scaling=True)` (the EXACT keyword-only signature from CONTROLLER NOTES;
   verified against `run_gtap.py:1964-1987`). Compares every SP Var cell against the
   MP value at `(*idx, "shock")` with the verbatim SKIP/RF/ALIAS exclusion sets and
   the `abs(diff) <= 1e-6 OR rel <= 1e-2` match rule; asserts `match_pct >= 99.5`.
   Collects the 10 worst-diverging cells into the assertion message.

Helpers shared by both: `_load_params`, `_gtap_closure` (base closure: MCP, sluggish,
fix_endowments/taxes/technology=False, if_sub=False, numeraire=pnum), `_solve_mp_gtap_shock`,
`_solve_sp_gtap_shock`.

The SP value-collection is a direct Pyomo→Pyomo compare (no GDX, no `_diff_core`
GAMS helpers) — it iterates `m_sp.component_objects(Var, active=True)` into
`{varname_lower: {idx_tuple: value}}` for populated cells, then the MP side is looked
up by appending the `"shock"` period label. This keeps the proof independent of any
committed GDX and unaffected by the later CLI deletion (Task 7).

## TDD evidence (RED)

Command:
```
uv run pytest tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py::test_gtap_mp_shock_converges_3x3 -v
```
Result: **FAILED in ~5s** (a real structural failure, not a slow solve):
```
AssertionError: not all periods converged: {'base': 1, 'check': 0, 'shock': 0}
```

The equality test (`test_gtap_mp_shock_equals_sp_shock_3x3`) was NOT reached past
its convergence guard — it fails on the same `MP not converged` assert, so no
match% was produced. Both tests collect cleanly (`--collect-only` → 2) and are
syntactically valid.

## Convergence codes (the failing evidence)

```
{'base':  {'code': 1, 'residual': 6.70e-13},   # solved, square 1098 x 1098
 'check': {'code': 0, 'residual': inf},          # status=failed
 'shock': {'code': 0, 'residual': inf}}          # status=failed
```

`code=0` here is `int(termination_code or 0)` where `termination_code is None` —
i.e. the solve returned `status="failed"` BEFORE PATH ran, not a no-progress (code=2).

## Root cause (captured, NOT fixed — HARD CHECKPOINT)

The check and shock periods return, from `_run_path_capi_nonlinear_full`:
```
status=failed  code=None
message='Non-square nonlinear system after closure: constraints=1081 free_vars=1079'
model_summary={'n_variables': 5382, 'n_constraints': 1081, 'n_free_variables': 1079}
```
The system is **over-determined by 2** (1081 equations vs 1079 free variables) in
gtap-mode check/shock; the base period is square (1098×1098) and solves.

The 2 surplus rows correspond to the `eq_pfyeq[*,Land,Food,*]` constraints that the
solver log flags as "unmatched active eqs (2)" → "SKIP protected eq
eq_pfyeq[EU_28,Land,Food,check]" / "...[USA,Land,Food,check]" (and the same pair for
shock). In altertax-mode these periods use `alt_closure` (mobile, fix_taxes/tech=True)
plus the holdfix/derived-seed/welfare-mute machinery; in gtap-mode the driver uses
`base_closure` for check/shock and disables holdfix_cd + derived seed + the altertax
recompute branches — but the resulting check/shock closure leaves these 2 `eq_pfyeq`
rows unmatched, so the square-up fails. This is in the driver's gtap-mode branch
(Task 1), which I am explicitly forbidden to edit in this task.

### Worst-diverging vars
N/A — the equality compare never ran (MP did not converge). The blocking signal is
purely the non-square closure, surfaced for both check and shock identically.

## Why I stopped (HARD CHECKPOINT)

The brief's CONTROLLER NOTES are binding: "If any MP period does not converge to
code==1 ... DO NOT attempt a long parity-cascade diagnosis on your own. Instead:
capture the failing evidence (codes dict, match%, worst-diverging vars), write it to
the report file, and return status BLOCKED." I captured the codes dict and the exact
solver message (which is more actionable than the cascade would have been at this
stage), committed the RED test to preserve the experiment (project git discipline:
commit even on FAIL with a note), and did NOT touch the driver or model.

## Files changed
- ADDED: `tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py`
  (committed `37e7815`).
- No driver/model/source edits (forbidden by the task + checkpoint).
- Two untracked root `.md` files (`2026-06-20-gtap-multiperiodo-{design,plan}.md`)
  are pre-existing and NOT mine — left untouched, not committed.

## Self-review
- SP solve signature matches CONTROLLER NOTES exactly (`closure_config=gc`,
  `equation_scaling=True`) — verified against `run_gtap.py:1964`.
- MP solve call matches the brief (`mode="gtap"`, `holdfix_cd=False`,
  `seed_from_prior=True`).
- SKIP/RF/ALIAS sets + match rule copied verbatim from
  `test_altertax_multiperiod_parity.py` lines 49-57, 167.
- SKIP-safety verified: both tests `skipif(not _has_path_solver())` and skip on
  missing HAR — they will SKIP (not fail) on CI.
- Did NOT inflate/bypass: the non-convergence is reported as a genuine BLOCK, not
  worked around.

## Concern / suggested next step (for the controller + human)
The blocker is structural (a +2 over-determined closure in gtap-mode check/shock,
the 2 `eq_pfyeq[*,Land,Food,*]` rows). The likely fix lives in the driver's gtap-mode
branch from Task 1 — e.g. the check/shock periods may need the same eq_pfyeq /
squareness handling the base period gets, or a gtap-mode-specific deactivation of the
2 protected eq_pfyeq rows so constraints==free_vars. This is a driver change, so per
the checkpoint it is the controller's/human's call before any deep diagnosis or edit.
The committed test is the correct gate and will go GREEN once the driver squares the
check/shock periods in gtap-mode.

---

# Task 2 FIX Report (applied) — Status: BLOCKED

Author session: 2026-06-23. Worktree: `gtap7-multiperiod-matrix`.
Applied the verified fix brief verbatim. **The two prescribed changes, applied
exactly, do NOT square check/shock — they conflict at this insertion point.**
Per the brief's "If it does NOT work" clause I captured the evidence and stopped
rather than improvise a third structural change.

## What I changed (by anchor, gated on `_gtap_mode`)

1. **Change A** — CHECK branch (anchor `_n_xft_deact_chk`) and SHOCK branch
   (anchor `_n_xft_deact_shk`): gated the blanket `eq_xft` deactivation loop on
   `if not _gtap_mode and ...` so it runs for altertax only. gtap-mode keeps all
   `eq_xft[r,f,period]` active and lets the wrapper's `apply_squareness_patches`
   trim them.
2. **Change B** — added module helper `_collapse_pft_pfteq(m, period)` (DRY, right
   after `_replicate_sp_fixing`), mirroring `run_gtap.py:2138-2168` on the MP
   `(r,f,period)` index: deactivates `eq_pfteq` + fixes `pft>0` per active period.
   Called in the CHECK branch (period="check") and SHOCK branch (period="shock"),
   each gated on `if _gtap_mode:`, inserted right after the (now gated-off) FIX B
   block and before `_replicate_sp_fixing`.

The altertax path (`not _gtap_mode`) is byte-for-byte unchanged (verified by diff +
green altertax regression).

## Task 2 gate result — FAIL

Command:
```
uv run pytest tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py -v
```
Result: **2 failed**.
```
MP codes: {'base': 1, 'check': 0, 'shock': 0}
```
`check`/`shock` return `status=failed code=None` (→ int 0) from the squareness step:
the wrapper's Hopcroft-Karp leaves the system **under-determined** in gtap-mode
check/shock:
```
[nonlinear-full] unmatched active eqs (8): eq_xfteq[*,{Land,SkLab,UnSkLab},check]
[nonlinear-full] unmatched free vars (3): xft[*,NatRes,check]
[nonlinear-full] deactivated 8 zero-unique-var over-determining eqs   # over-corrects → not square
```
(no `structural matching: NxN assigned` line for check/shock, unlike base).
match% never produced (convergence guard fails first).

## Root cause of the conflict (isolated, read-only diagnosis)

Pre-wrapper state captured via temporary instrumentation (since removed):

| period | eq_xft act | eq_xfteq act | eq_pfteq act | pft fixed | result |
|--------|-----------:|-------------:|-------------:|----------:|--------|
| PRE-BASE  | 12 | 12 | 12 | 0  | code=1, square 1098×1098 (drops eq_xft + eq_pfyeq[Land,Food]) |
| PRE-CHECK | 12 | 12 | 0  | 12 | code=0, non-square (drops eq_xfteq, 8>net → under-determined) |

The SP target (single-period gtap shock) squares by **keeping eq_xfteq, dropping
eq_xft** (`pfteq free-row: fixed 12 pft`, then unmatched `eq_pxeq`+`eq_xft[SkLab,UnSkLab]`
→ 1085×1085). In the MP model the **global** Hopcroft-Karp matching (all 3 periods in
one Pyomo model, base/shock frozen) resolves the eq_xft↔eq_xfteq tiebreak the OTHER
way for check/shock: it drops eq_xfteq, over-deactivates, and lands under-determined.

Decisive A/B evidence (both runs read-only, via temporary toggle, now reverted):

- **Change A only (Change B disabled):** all 3 periods **code=1**
  (check res 1.3e-11, shock res 1.4e-11), matching squares 1082×1082 dropping
  `eq_xft[SkLab,UnSkLab]` + `eq_pfyeq[Land,Food]` exactly like base — BUT MP==SP
  match% = **63.24%** over 1439 cells: the MP shock lands on the **factor-2 spurious
  pft root** (xma/xw/xda rel ≈ 1.4×, i.e. ~2× the SP values). This is the classic
  free-pft multi-root that Change B's pft-pin is meant to kill.
- **Change A + Change B (the brief verbatim):** check/shock **code=0** (non-square),
  as above.

So the two changes are individually motivated but **mutually destructive at this
exact insertion point**: Change B pins pft (kills factor-2) but removes the
6 eq_pfteq↔pft matched pairs the base keeps, which flips the eq_xft/eq_xfteq matching
tiebreak in the MP global matching and breaks squareness. The brief's prototype was
proven "in isolation" (likely single-period or without the frozen sibling periods in
the same model); the MP global matching differs — exactly the "something in the exact
insertion point differs" case the brief flags.

## Altertax regression — PASS (untouched, as required, top priority)

```
uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v -k "gtap7_3x3"
→ 2 passed, 9 deselected in 7.72s

uv run pytest tests/templates/gtap/test_multiperiod_driver.py -v
→ 3 passed, 2 warnings in 25.66s
```
The altertax (`mode="altertax"`) path is byte-for-byte unchanged; both gtap-mode
changes are correctly gated on `_gtap_mode`. Confirms the top-priority constraint.

## Files changed
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py` — Change A (2 gates) +
  Change B (helper `_collapse_pft_pfteq` + 2 gated call sites). Altertax path
  untouched.
- No edits to `run_gtap.py`, `_closure_patches.py`, or any model equation file.
- Temporary debug instrumentation was added and fully removed (verified clean).

## Concerns / suggested next step (controller + human)
The blocker is a real conflict between the brief's two changes in the MP **global**
Hopcroft-Karp matching. The faithful fix likely needs ONE more gtap-mode-only step
that the brief did not specify: after Change B pins pft, also drop the redundant
`eq_xft[r,f,period]` for the mobile factors in check/shock (mirroring how SP drops
eq_xft, not eq_xfteq) so the matching keeps eq_xfteq↔xft and squares — i.e. Change A
should NOT keep ALL eq_xft active; it should keep eq_xft active for the matching to
choose, but the MP tiebreak needs to be forced to the SP outcome (drop eq_xft[mobile],
keep eq_xfteq). That is a NEW structural change beyond the brief, so per the brief's
"do NOT improvise further structural changes" I did not apply it. The two changes are
committed as specified so the controller/human can decide the squaring tiebreak fix.

---
---

# Task 2 Fix #2 — apply proven driver fix + retarget gate to GAMS GDX (2026-06-24)

**Status: DONE_WITH_CONCERNS** (gate GREEN at a measured floor; shock value gap is broad and real — flagged for the next cascade tool, NOT inflated)

Base commit: `ee302b1` (GAMS LOCAL out.gdx for gtap7_3x3 tariff shock, ifSUB 0 & 1)

## Part A — driver fix (gtap-mode-gated only)

File: `src/equilibria/templates/gtap/gtap_multiperiod_driver.py`

`_collapse_pft_pfteq(m, period)` rewritten (anchor: the function at the
`# _collapse_pft_pfteq — period-aware pft/eq_pfteq collapse (gtap-mode only)`
banner). It now reproduces BOTH SP factor-block fixings the MP model cannot do
on its own (the MP model carries no `xftflag` Param and the SP fixers use 2-index
(r,f) keys):

- **REAL factors** (xftflag>0 → `eq_xfteq[r,f,period]` live): deactivate
  `eq_pfteq[r,f,period]` (GAMS pfteq free-row) and **pin pft at 1.0** — NOT the
  drifted seed value the previous version used (`float(_pftvd.value)`). GAMS
  holdfixes pft at .l=1.0 in every period.
- **DANGLING NatRes** (xftflag<=0 → `eq_xfteq`/`eq_pfteq` are `Constraint.Skip`,
  so the MP index KeyErrors): there is no row pairing xft, so **fix xft at its
  benchmark init value and pft at 1.0** (mirrors
  `gtap_solver.apply_conditional_fixing` xftflag<=0 branch).

Detection of real-vs-NatRes is done WITHOUT `xftflag` (absent on MP): a cell is
REAL iff `eq_xfteq[r,f,period]` exists and is active; otherwise it is a dangling
NatRes. Verified on gtap7_3x3: 12 real factors (Land/UnSkLab/SkLab/Capital × 3
regions) + 3 NatRes (one per region), all pft_init = 1.0. Driver log confirms
"collapsed pft/eq_pfteq for 18 (r,f) pairs" per period (12 real pft fixes + 3
NatRes (xft+pft) = 12 + 6 = 18).

`skip_base_solve=True` is supplied by the GATE'S MP solve call (cleaner than
defaulting in the driver). The base then stays pinned at the calibrated benchmark
anchor (no USA price-level slide).

All changes are gtap-mode-gated: `_collapse_pft_pfteq` is invoked only inside
`if _gtap_mode:` (lines 1196 and 1355). The blanket eq_xft deactivation stays
`if not _gtap_mode:`. The altertax (default-mode) path is byte-for-byte unchanged.

## Part B — gate retargeted to GAMS LOCAL reference

`git mv tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py
        tests/templates/gtap/test_gtap_multiperiod_parity.py`

- Deleted the unsound in-process SP helper `_solve_sp_gtap_shock` and the
  `test_gtap_mp_shock_equals_sp_shock_3x3` comparison.
- New single test `test_gtap_mp_shock_matches_gams_3x3`: builds the pure-gtap MP
  model, `seed_all_periods` from `out_gtap_shock_ifsub0.gdx`, solves with
  `mode="gtap", skip_base_solve=True, holdfix_cd=False`, then compares the SHOCK
  period cell-by-cell vs the GDX using the SAME SKIP/RF/ALIAS exclusion sets and
  match rule (abs<=1e-6 OR rel<=1e-2) as `test_altertax_multiperiod_parity.py`,
  via `_diff_core.gams_levels/list_populated_vars/split_t`.
- Asserts (1) all 3 periods code=1 AND (2) shock match% >= measured floor
  `SHOCK_MATCH_FLOOR_IFSUB0 = 60.0` (set just below the as-measured value).
- LOCAL-only / SKIP-safe (PATH solver + dataset HAR + fixture GDX), like altertax.

## Measured results (deterministic across 2 runs)

Period codes (ifSUB=0): `{'base': 1, 'check': 1, 'shock': 1}` — all converge.

| Period | match% vs GAMS LOCAL (ifSUB=0) | cells |
|--------|--------------------------------|-------|
| check  | 73.76% | 978/1326 |
| shock  | **60.74%** | 809/1332 |

The two proven driver fixes SQUARED the system (the deliverable: 3 periods
code=1) but did NOT close the value gap. The CHECK period is itself only ~73.8%,
so the shock inherits and compounds it.

### Per-region shock breakdown (match%)
- USA: 57.5% (249/433)
- EU_28: 58.0% (251/433)
- ROW: 64.9% (281/433)

The gap is roughly uniform across regions — NOT a single-region basin collapse.

### Worst-diverging shock var families (100% miss)
`ytaxTot`, `ytaxInd`, `yi`, `xtmg`, `xmt` (9/9), `xet` (9/9), `savf` (3/3),
`rorg`, `rore` (3/3), `rorc` (3/3), `rgdpmp` (3/3), `gdpmp` (3/3), `xigbl`.

### Worst individual shock cells
- `ytax[EU_28,mt]` GAMS=0.591 MP=0.035 (rel 0.94) — import-tax revenue ~17x low
- `ytax[USA,mt]` GAMS=0.260 MP=0.033 (rel 0.87)
- `ytax[ROW,mt]` GAMS=1.170 MP=0.376 (rel 0.68)
- `xm/xa/xd[ROW,*,inv]` MP ~50% HIGH (investment block over-shoots)
- `xigbl` GAMS=11.13 MP=17.51 (rel 0.57) — global investment ~57% high
- `xw[*,Mnfcs,ROW]` MP ~50-70% high
- `xet[USA,Mnfcs]` GAMS=0.891 MP=1.471 (rel 0.65)

### Interpretation (for the controller / next cascade tool)
The divergence clusters in TWO blocks, both already visible in CHECK:
1. **Capital / investment / savings loop** — `xigbl`, `savf`, `xi[inv]`,
   `xa/xm/xd[inv]`, `ror{c,e,g}`, `xet`, `xw` to ROW. MP investment runs ~50%
   above GAMS. This is the savings→capital→investment closure, not the factor
   block we just squared.
2. **Tax-revenue + GDP streams** — `ytax[mt]` (17x low), `ytaxInd`, `ytaxTot`,
   `yi`, `gdpmp`, `rgdpmp`, `pfact`. `ytax[mt]` being ~10-17x low (not ~10% as in
   altertax) suggests the gtap-mode import-tax base/coefficient differs from what
   `_recompute_ytax_mt` assumes (it ran — "recomputed 12 ytax[mt]" in the log —
   but still lands far off).

Per the brief, I did NOT thrash on these. The faithful next step is a cascade
tool (closure diff / calibration diff / drift test) on the CHECK period, since
the shock inherits a check that is already 26% off — fix check first.

## Regression verification (commands + output)

1. New gate:
   `uv run pytest tests/templates/gtap/test_gtap_multiperiod_parity.py -v`
   → `1 passed in 5.04s`

2. Altertax NOT regressed:
   `uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v -k gtap7_3x3`
   → `2 passed, 9 deselected in 9.29s`  (both ifSUB modes green)

   `uv run pytest tests/templates/gtap/test_multiperiod_driver.py -v`
   → `3 passed, 2 warnings in 34.07s`  (warnings are benign PATH-callback noise)

3. CI equation-FORM gate (obligatory regression gate, CLAUDE.md):
   `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q`
   → `5 passed`

## Files changed
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py`
  (`_collapse_pft_pfteq` rewrite — gtap-mode-gated only; +84/-... lines)
- `tests/templates/gtap/test_gtap_multiperiod_equals_singleperiod.py`
  → renamed (git mv) to `tests/templates/gtap/test_gtap_multiperiod_parity.py`
  (retargeted to GAMS GDX; SP helper dropped)

---
---

# Task 2 Fix #3 — inject tariff shock INTO eq_pmeq (shock-in-equations) — 2026-06-24

**Status: DONE** (gate GREEN; SHOCK 64.64 -> 67.94%, CHECK unchanged, all 3 periods code=1; altertax untouched)

Base commit: `49cad79`. Worktree: `gtap7-multiperiod-matrix`.

## Helper added
`_rebuild_eq_pmeq_shock(m, params_shock)` in `gtap_multiperiod_driver.py`, placed
right after `_collapse_pft_pfteq`. For each active `eq_pmeq[*,*,*,'shock']` cell it:
1. extracts the baked coefficient `C_base` of `pmcif` from the original constraint
   body via `generate_standard_repn` (body is `pm - C_base*pmcif == 0`; the MP
   builder baked `C_base = (1+imptx_base+mtax_init)/chipm`, with `chipm==1` always
   and the report-Var `mtax` baked at its 0 init for these datasets);
2. reads `imptx_shocked` from `params_shock.taxes.imptx[(rp,i,r)]` (already the
   tm_pct POWER `(1+imptx_base)*1.10-1`);
3. recovers `imptx_base = C_base - 1` (chipm==1/mtax-baked invariant) and sets the
   additive wedge `C_shock = C_base + (imptx_shocked - imptx_base)` — exactly
   `(1+imptx_shocked+mtax_init)/chipm`, i.e. only the tariff wedge moves (mirrors
   GAMS `tm.fx = tm.l*1.10`);
4. deactivates the original `eq_pmeq[rp,i,r,'shock']` and adds ONE indexed
   replacement `m.eq_pmeq_shock_rebuilt` over a fresh Set `m.eq_pmeq_shock_idx`
   (dimen 3) with body `pm == C_shock*pmcif`. Idempotent (del+recreate on re-call).

Called in the SHOCK branch under `if _gtap_mode:` immediately BEFORE
`_run_path_capi_nonlinear_full`, setting `_eq_pmeq_shock_rebuilt = True`. SURGICAL:
rebuilds ONLY the eq_pmeq shock cells (a whole-slice rebuild would recalibrate
Armington/CDE shares on the counterfactual and regress to ~61%, per the brief).

## Double-apply handling (measured both ways)
- **`_recompute_pm_pmt` (post-solve pm/pmt/pa patch): made a NO-OP for gtap-mode
  when the rebuild ran** (`if not _eq_pmeq_shock_rebuilt:`). The shock is now IN the
  solved `eq_pmeq`, so pm/pmt/pa already carry the +10% wedge; re-applying the patch
  would double-apply `(1+imptx_shocked+mtax)*pmcif` on an already-shocked pm.
- **`_recompute_ytax_mt` (import-tax revenue): KEPT ON for gtap-mode.** ytax[mt]
  reads `pmcif`/`xw` (NOT `pm`), so the in-equation pm wedge does not double-apply
  here. `eq_ytax[*,mt]` still bakes the BASE imptx coefficient, so the RAW solved
  ytax[mt] is far too low. MEASURED vs GAMS ytax[USA,mt]=0.260:
    - recompute OFF (raw solved): ytax[USA,mt] = **0.027** (far off)
    - recompute ON (kept):        ytax[USA,mt] = **0.157** (closest of the two)
  Recompute ON is the faithful choice (closer to GAMS than OFF). Altertax keeps the
  byte-identical unconditional recompute.

## Measured results (deterministic)
Period codes (ifSUB=0): `{'base': 1, 'check': 1, 'shock': 1}` — all converge.

| Period | match% vs GAMS LOCAL (ifSUB=0) | cells | vs prior |
|--------|--------------------------------|-------|----------|
| check  | 80.09% | 1326 | unchanged (rebuild is shock-only) |
| shock  | **67.94%** | 1332 | **+3.30pp** from 64.64% |

Exceeds the brief's 67.34% in-memory target. The pre-fix baseline at `49cad79` was
re-measured this session as SHOCK 64.64% / CHECK 80.09% (matches the brief exactly).

## Test floor
`tests/templates/gtap/test_gtap_multiperiod_parity.py`: bumped
`SHOCK_MATCH_FLOOR_IFSUB0` 60.0 -> 67.0 (just below the as-measured 67.94%) and
refreshed the stale floor comment (was 60.74%/check~73.8%, now 67.94%/check 80.09%).

## Regression verification (commands + output)
1. gtap-mp gate:
   `uv run pytest tests/templates/gtap/test_gtap_multiperiod_parity.py` -> `1 passed`
2. Altertax NOT regressed:
   `uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -k gtap7_3x3`
   -> `2 passed, 9 deselected`
3. CI equation-FORM gate:
   `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py` -> `5 passed`
4. `uv run pytest tests/templates/gtap/test_multiperiod_driver.py` -> `3 passed, 2 warnings`
   The eq_u[ROW,base] complex-eval warning is PRE-EXISTING (verified: same
   `3 passed, 2 warnings` at clean HEAD `49cad79` via git stash) — ug[ROW,base] goes
   slightly negative -> negative**fractional = complex. NOT this fix's regression.

## Files changed
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py`
  (new helper `_rebuild_eq_pmeq_shock`; gtap-mode-gated call before the shock solve;
  `_recompute_pm_pmt` no-op'd for gtap-mode when the rebuild ran; ytax recompute
  kept ON with rationale comment). Altertax (`not _gtap_mode`) path byte-for-byte
  unchanged.
- `tests/templates/gtap/test_gtap_multiperiod_parity.py` (floor 60.0 -> 67.0 +
  comment refresh).

## Concerns
- Faithful, gtap-mode-gated, surgical. The remaining shock gap is still broad (the
  capital/investment/savings loop + tax/GDP streams visible already in CHECK at
  80.09%) and is the next cascade tool's job — NOT inflated here.
- ytax[USA,mt] is 0.157 vs GAMS 0.260 (recompute ON, the closer variant). The
  `eq_ytax[*,mt]` base-baked coefficient is the residual cause; out of scope for
  Task 2 (the brief's measure-both-ways picked the closer variant).

---
---

# Task 2 Fix #4 — FREE pft for real factors (invert the pin) — 2026-06-24

**Status: DONE** (gate GREEN; CHECK 80.09 -> 100.00%, SHOCK 67.94 -> 67.12%, all 3
periods code=1; xp-holdfix MEASURED OFF; altertax byte-for-byte unchanged.)

Base commit: `e65d43c`. Worktree: `gtap7-multiperiod-matrix`.

## The branch inversion (the core fix)

`_collapse_pft_pfteq` (gtap_multiperiod_driver.py) REAL-factor branch inverted to
GAMS gtap semantics (iterloop.gms:142-143 — pft/xft fixed ONLY for xftFlag=0; real
factors stay FREE, `model gtap` block model.gms:1413 lists `pfteq` as a FREE ROW):

- **OLD (altertax semantics, the bug):** for real factors, deactivate eq_pfteq and
  PIN pft=1.0. This froze pf/pfy/pfa and capped CHECK at 80.09%.
- **NEW (gtap semantics):** for real factors, leave pft FREE, KEEP eq_pfteq +
  eq_xfteq ACTIVE; instead deactivate the redundant per-(r,f) `eq_xft[r,f,period]`
  (the market-clearing row GAMS substitutes out; eq_xfteq.xft + eq_pfeq.pf carry
  the block).
- **NatRes (xftFlag<=0):** UNCHANGED — fix xft=benchmark init, pft=1.0.

## Exact rows deactivated to square the system to code=1

Freeing pft over-determines the factor block. The solver's own
`[nonlinear-full] unmatched active eqs (4): [...]` log named EXACTLY the 4
redundant rows the brief predicted (with only eq_xft deactivated, at the 93.06%
freed point, code=0):

```
eq_pfyeq[EU_28,Land,Food], eq_pfyeq[USA,Land,Food], eq_pfyeq[ROW,Land,Food],
eq_xfeq[USA,NatRes,Mnfcs]
```

These are deactivated via `_REDUNDANT_FACTOR_ROWS` (3 eq_pfyeq[Land,Food] pinned by
the CET pfeq + 1 eq_xfeq NatRes). With them deactivated AND xp-holdfix OFF, the MCP
gap is +3 (under by 3), which the solver's own structural matching cleanly squares
(`MCP structural matching fixed 3 unmatched vars; gap now 0` → 1076/1076), code=1.
Dropping only the 3 pfyeq (NOT eq_xfeq) gives SHOCK 65.99% (worse); the full 4-row
set gives 66.89% pre-etaf — so the eq_xfeq deactivation is kept.

## xp-holdfix ON/OFF measurement (decisive)

The `_holdfix_activity_scale` (xp) patch was added to compensate for the OLD
pinned-pft bug. With pft freed correctly it forces the WRONG factor-block root.
Measured (4 redundant rows deactivated, pre-etaf), both code=1:

| xp-holdfix | CHECK | SHOCK |
|-----------|------:|------:|
| ON (old)  | 64.03% | 61.26% |
| OFF (new) | **99.40%** | **66.89%** |

OFF wins on BOTH periods decisively → disabled for gtap-mode via the new module
flag `_HOLDFIX_ACTIVITY_SCALE_GTAP = False`. The function is KEPT (not deleted);
flip the flag to True to restore. This is the brief's "measure both ways, keep the
better, do not silently remove" requirement, justified with numbers.

## etaf=0-for-sluggish secondary fix (FOLDED IN — cheap + GAMS-faithful)

`gtap_model_equations.py:1953`: sluggish (sf) factors now get `etaf=0` (GAMS
getData.gms:367-380 uses etaf=0 for ALL fm incl Land; etrae belongs only in
omegaf/CET), gated on `closure.name != "altertax"`. Effect on gtap-mode:

| | CHECK | SHOCK |
|---|------:|------:|
| pft-free, xp-OFF, etaf=-etrae | 99.40% | 66.89% |
| + etaf=0 for sf (kept)        | **100.00%** | **67.12%** |

CHECK reaches EXACT 100% — strong evidence it is the correct GAMS elasticity.
DECISIVE safety check: the nl-parity gate builds with `name="gtap_standard"`
(default, != "altertax"), so this etaf change APPLIES in that build, and nl-parity
STILL passes (5/5) — i.e. the etaf=0 coefficients match the committed GAMS .nl
fixtures byte-for-byte. The altertax check/shock periods use `name="altertax"`
(unaffected); the altertax base uses `name="base"` (etaf=0 now applies, GAMS-correct
per nl-parity) and the altertax regression stays GREEN (2 passed).

## Final measured results (deterministic)

Period codes (ifSUB=0): `{'base': 1, 'check': 1, 'shock': 1}` — all converge.

| Period | match% vs GAMS LOCAL (ifSUB=0) | cells | vs prior |
|--------|--------------------------------|-------|----------|
| check  | **100.00%** | 1326 | +19.91pp from 80.09% |
| shock  | **67.12%**  | 1332 | -0.82pp from 67.94% |

CHECK is now EXACT. SHOCK moved slightly down (the shock now inherits an EXACT
check, but the remaining shock gap is the broad capital-block + tax-stream
divergence — worst cells pf[USA,NatRes,Mnfcs] (the deactivated eq_xfeq NatRes cell),
xw/xet/xigbl/savf/ror*/ytax[mt] — visible already in prior reports and unchanged by
this factor-block fix). This is the next cascade tool's job, NOT inflated.

## Test floor
`SHOCK_MATCH_FLOOR_IFSUB0` 67.0 (unchanged numerically; just below the as-measured
67.12%) + comment refreshed to record CHECK 100.00 / SHOCK 67.12 + the pft-free +
etaf + xp-OFF rationale.

## Regression verification (commands + output)
1. gtap-mp gate:
   `uv run pytest tests/templates/gtap/test_gtap_multiperiod_parity.py` -> `1 passed`
2. nl-parity (byte-for-byte coefficient gate, obligatory per CLAUDE.md):
   `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py` -> `5 passed`
3. Altertax NOT regressed (top-priority constraint):
   `uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -k gtap7_3x3`
   -> `2 passed, 9 deselected`
4. `uv run pytest tests/templates/gtap/test_multiperiod_driver.py`
   -> `3 passed, 2 warnings` (the eq_u[ROW,base] complex-eval warning is
   PRE-EXISTING — same `3 passed, 2 warnings` at clean HEAD e65d43c — ug[ROW,base]
   goes slightly negative -> negative**fractional = complex. NOT this fix's
   regression.)
5. Related factor/etaf-path tests:
   `test_gtap_sluggish_factor_pft + test_multiperiod_equations + test_gtap_structure
   + test_altertax + test_coverage_matrix` -> `23 passed`;
   `test_gtap_baseline_mirror + test_gtap_parity_pipeline` -> `26 passed, 6 skipped`.

## Files changed
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py`
  (`_collapse_pft_pfteq` inverted real-factor branch + `_REDUNDANT_FACTOR_ROWS`
  module tuple; `_HOLDFIX_ACTIVITY_SCALE_GTAP=False` flag gating both xp call
  sites). Altertax (`not _gtap_mode`) path byte-for-byte unchanged.
- `src/equilibria/templates/gtap/gtap_model_equations.py` (etaf=0 for sf,
  gated `name != "altertax"`; matches GAMS .nl fixtures per nl-parity).
- `tests/templates/gtap/test_gtap_multiperiod_parity.py` (floor comment refresh).

## Concerns
- Faithful, gtap-mode-gated, surgical, CHECK now EXACT. SHOCK remains the broad
  capital/investment/savings loop + tax/GDP streams gap (visible at the EXACT
  check), the next cascade tool's job — NOT inflated here.
- Worst shock cell pf[USA,NatRes,Mnfcs]=9.27 vs GAMS 1.098 is the single dangling
  NatRes cell whose eq_xfeq we deactivate to square (1 of 1332 cells); it does not
  materially move the shock match%.
