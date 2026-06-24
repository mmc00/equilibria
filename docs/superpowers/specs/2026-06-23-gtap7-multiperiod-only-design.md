# GTAP 7 multi-period as the only path (deprecate single-period)

**Date:** 2026-06-23
**Branch:** `gtap7-multiperiod-matrix`
**Status:** design approved, ready for implementation plan

## Goal

Make the multi-period (`base → check → shock`) engine the **only** path for the
pure `gtap` model (tariff shock +10% on `tm`), exactly as it already is for
`altertax`, gated by `ifSUB` (0/1). Delete the single-period path. The coverage
matrix (`scripts/gtap/coverage_matrix.py`) loses its "single-period" section;
`gtap` rows become multi-period + ifSUB like the `altertax` rows.

## Guiding principle (the acceptance criterion)

For the **pure gtap model, multi-period MUST equal single-period**, and that
equality is the fidelity proof of the engine.

Why it holds: unlike altertax, gtap does **not** recalibrate (`betaCal`). So the
`check` period is idempotent by construction — it reproduces `base` with no
free-DOF drift (there is no forced-CD tautology to pin, hence `holdfix_cd=False`).
The shock period warm-started from `base` therefore equals the single-period
shock. Wrapping gtap in `base → check → shock` subsumes the single-period path.

**Acceptance: MP == SP == GAMS, all three identical.**
- `.nl` coefficient diff between MP slice and GAMS fixture = 0 per period.
- Solved-value match vs the GAMS GDX = 100% (not merely `>= gap_min`) for gtap.

If any dataset fails this equality, it is an MP-engine bug to close (via the
parity tool cascade, `equilibria-parity-debug`) **before** deleting the SP path.

## Architecture — four pieces (all existing patterns)

### 1. Driver: `gtap_multiperiod_driver.py` — add `mode`

Add `mode="gtap" | "altertax"` to `solve_multiperiod`. The altertax coupling is
concentrated in four points (verified by code read); the rest of the helpers
(`freeze_inactive_periods`, `_seed_period_from_prior`, `_replicate_sp_fixing`,
`freeze_period`) are generic and reused unchanged.

In `mode="gtap"`:
- use `params` directly — **skip** `apply_altertax_elasticities`;
- use `base_closure` (`fix_taxes=False`, `fix_technology=False`,
  `capital_mobility="sluggish"`) for **all three** periods (no `alt_closure`);
- `holdfix_cd=False` (no forced CD → no pva/pnd pin, `_holdfix_cd_nest` not called);
- keep the `tm_pct` 10% imptx shock for the shock period;
- skip / neutralize the altertax-specific post-solve corrections that assume the
  altertax param structure (revisit `_recompute_ytax_mt` / `_recompute_pm_pmt` /
  `_recompute_ifsub_report_vars` — they must still be correct for the pure shock,
  or be made conditional). This is the one area to validate carefully during
  implementation; the MP==SP equality is the check that tells us we got it right.

`mode="altertax"` keeps today's behaviour exactly (default stays altertax so the
existing altertax gate is untouched).

### 2. GAMS reference: multi-period `.gms` → NEOS CONVERT → `.nl` fixtures

Today `tariff_sim_fixed.gms` is single-period (one `Solve STD_GTAP using mcp`).
Write a `loop(tsim)` version (`base → check → shock`) for the pure gtap model —
the GAMS reference becomes multi-period too, symmetric with deprecating SP on the
Python side.

Generate the per-phase `.nl` via **NEOS CONVERT** (`nl_compare.py` already does
this — the expired local GAMS license is not needed; NEOS CONVERT runs the
conversion). These replace the committed single-period fixtures at
`tests/fixtures/gtap7/<dataset>/gams_{base,check,shock}.nl`.

### 3. CI gate: `test_gtap7_nl_parity.py`

Reframe the test to build the `.nl` from the **MP engine** (one slice per period)
and diff 0 against the regenerated MP fixtures. Stays a no-solver CI gate
(ubuntu-latest, `gtap7-nl-parity` job). The MP==SP equality (Section "Guiding
principle") is what guarantees a 0-diff against the same coefficient set.

### 4. Coverage matrix + doc

`scripts/gtap/coverage_matrix.py`:
- remove the single-period `gtap` rows (`ifsub=None`, `.nl`-only);
- gtap datasets become multi-period rows with `ifsub ∈ {0,1}` like altertax,
  carrying a real `gap_min` (100 / "0 diffs" semantics for the MP `.nl` CI rows,
  and a solver `gap_min` for the local solver rows where applicable);
- update `_validate()` invariants (the `ifsub is None ⟺ kind == "gtap"` rule and
  the `gap_min is None ⟺ nl-only gtap7_*` rule both change — re-derive them).

Regenerate `docs/site/guide/gtap7_coverage_matrix.md` via
`scripts/gtap/gen_coverage_doc.py`; keep the `test_coverage_doc_in_sync` golden test.

## Deprecation — what gets deleted (in safe order)

**Order: MP working + proven (MP == SP == GAMS) FIRST, then delete SP.** Never
remove the safety net before the replacement is green.

To remove once MP is proven:
- `run_gtap.py`: `validate-shock`, `_run_homotopy_shocked`, the single-period
  `_apply_shock_to_params` solve path, and the one-shot SP solve entry points;
- `tests/templates/gtap/test_altertax.py` single-period cases and any test that
  exercises the SP solve;
- `src/equilibria/templates/reference/gtap/tariff_sim_fixed.gms` (single `Solve`)
  → replaced by the `loop(tsim)` `.gms`;
- SP references in `CLAUDE.md`, `GTAP_VALIDATION_STATUS.md`,
  `docs/site/benchmarks.md`.

Open question for the plan: whether to keep `homotopy_shock.py` as a debug tool
(it is already flagged STALE in memory `project_gtap7_3x3_homotopy_deadend`).
Default: delete unless it earns its keep as a diagnostic.

## Risk & verification

Primary risk: the MP does **not** exactly equal SP on some dataset (an MP-engine
bias that altertax's `gap_min` tolerates but gtap, demanding 0/100%, exposes).

Verification = the guiding principle: run MP and SP of the same dataset, require
`.nl` diff 0 and solved-value match == 100% vs the GAMS GDX. If it fails, it is an
MP bug to close with the parity cascade **before** deleting SP.

Start with the **smallest dataset** (`nus333` / `gtap7_3x3`) to prove the equality
before scaling up (per the project rule: always iterate on the smallest dataset
first).

## Out of scope

- No change to `altertax` behaviour or its fixtures.
- No new economics / equation changes (this is a path-unification + deprecation,
  not a model change). Any equation touch that surfaces is a separate bug to gate
  via the `.nl` parity test.
- `gtap7_20x41` stays `blocked` (NEOS reference Infeasible).
