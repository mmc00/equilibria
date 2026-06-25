# Design: `probe.py --params` — Param/calibration diff (5th cascade tool)

**Date:** 2026-06-13
**Status:** Approved (pending spec review)
**Author:** debug session, gtap7_3x3 altertax shock

## Motivation

The four parity-debug tools (`.nl` diff, residual test, check-warmstart, closure
diff) share a blind spot: **calibrated constants whose value depends on how the
model is BUILT** — `pf0`, `xf0`, `base_rgdpmp`, `base_mqgdp`, `p_gf`, `betap`, etc.
These are Pyomo `Param`s baked in at build time (e.g. depending on whether
`t0_snapshot` is passed). The blind spot:

- **`.nl` diff** sees the equation but treats the constant as a literal — it can't
  question where the constant came from.
- **residual test** seeds the GAMS point and checks equations; it takes the
  constants as given (the GAMS point satisfies them either way).
- **check-warmstart** inspects seeded *variables*, not build-time *Params*.
- **closure diff** sees the fixed/free partition, not Param values.

This is exactly the gtap7_3x3 shock bug (2026-06-13): with `t0_snapshot`, the
Fisher-index anchors (`pf0`, `base_rgdpmp`) come from the check; without it, from
the calibrated benchmark — and the shock converges to a different equilibrium. All
four tools reported "fine"; only a manual two-sided closure diff (run late, after
the user pushed for it) exposed it.

## Goal

Add `probe.py --params`: compare **all Pyomo `Param`s of the built model**
cell-by-cell against the corresponding GAMS GDX symbols, and report which diverge.
Plus `--params-compare-builds`: detect which Params change between two build modes
(with/without `t0_snapshot`), auto-discovering the construction-dependent universe.

## Non-goals

- Not a solver/equilibrium check (that's the residual test).
- Not a fixed/free partition check (that's the closure diff).
- No curated allow-list of Params: iterate ALL of them; transparency about coverage
  replaces curation.

## Architecture

A `--params` query in `scripts/parity/probe.py`, with the comparison logic in a new
pure module `scripts/parity/_probe_params.py`. Reuses the existing build path (model
construction via `GTAPParityAdapter`, no solve needed — Params are set at build) and
`_diff_core.gams_levels` for reading GAMS.

```
probe.py --dataset gtap7_3x3 --scenario altertax_check \
         --params --gdx-ref output/.../out_local.gdx [--top 20] [--family pf0,base_]
         [--params-compare-builds]
```

### `--params` flow

1. Build the model (same path as other queries; no solve).
2. Iterate all active `Param` components → `{param_name: {idx: value}}` (mutable and
   immutable alike — both are build-time constants).
3. For each Param, resolve the GAMS GDX symbol: try the direct name, then an optional
   alias map for cross-period/renamed cases (e.g. `pf0 → pf@base`, `base_rgdpmp →
   rgdpmp@base`, `betap → betaP`). Strip GAMS prefixes (`a_/c_/f_/r_`) and the period
   dimension, same as the snapshot.
4. Compare cell-by-cell with `--tol-rel` (default 1e-3).
5. Classify each Param into three groups:
   - **DIVERGE**: in GAMS, ≥1 cell differs → sorted by `max_rel`, with the worst cell.
   - **ok**: in GAMS, all cells match → counted.
   - **no-GAMS-match**: no GAMS symbol → listed separately (reveals coverage).

### `--params-compare-builds` flow

1. Build the model TWICE — mode A (as the scenario builds it) and mode B (toggling
   `t0_snapshot`: pass `None` if the scenario passes a snapshot, or vice-versa).
2. Diff the Param dicts: report Params whose value changes between builds, with the
   worst-changing cell.
3. This needs no GAMS; it auto-detects the construction-dependent universe. For the
   shock bug it would flag `pf0`, `xf0`, `base_rgdpmp`, `p_gf`, `betap` as
   build-dependent — the regression signature of this session's bug.

## Output

Plain-text table (same style as `diff_altertax`):

```
param            cells  match  diverge  max_rel    worst cell
pf0                 45      9       36   4.66e-01   (EU_28,UnSkLab,Mnfcs) py=2.13 gams=1.45
...
coverage: 41 params verifiable vs GAMS, 18 with no GAMS counterpart
```

Each run prints whether it was a `[cache hit]` build or fresh (consistent with the
other probe queries).

## Types & errors

- Scalar Params (`mqfactw_bb`), 1-D (`betap[r]`), 3-D (`pf0[r,f,a]`) all handled via
  the snapshot's index normalization (tuple, prefix strip, period drop).
- GDX symbol absent → "no-GAMS-match" (no crash).
- Python cell with no GAMS counterpart → counted non-verifiable, not "diverge".
- Param with zero cells → skipped.
- Alias map is a small dict in `_probe_params.py`; entries use `name@period` to point
  at a GAMS symbol from a different period (the `pf0 → pf@base` case).

## Testing

`tests/parity/test_probe.py` (extends existing):
- `--params` on gtap7_3x3: the three groups (diverge/ok/no-match) are populated.
- A known-correct Param (`kappaf`, verified = GAMS this session) lands in "ok".
- A known-divergent Param lands in "DIVERGE" with its worst cell.
- `--params-compare-builds`: builds twice, detects `pf0`/`base_rgdpmp` change between
  with/without `t0_snapshot` — the regression test for this session's shock bug.
- Gates on reference-GDX presence (skip if absent, like the other adapter tests).

## Skill registration

Update `equilibria-parity-debug` skill:
- Add cascade row #4 (or 5th counting check-warmstart as 0): **Param/calibration
  diff** — `probe.py --params` — Sees: build-time calibrated constants (pf0,
  base_rgdpmp, p_gf, betap) the other tools take as literals — Use when: the solver
  converges to a valid equilibrium but the LEVELS differ and the other four tools all
  say "fine".
- Document the blind spot with the gtap7_3x3 shock precedent (Fisher anchors
  dependent on `t0_snapshot`).
- Add the lesson: run the closure diff comparing BOTH sides (Python vs GAMS fixed
  set), not just Python — a one-sided closure check misses what GAMS fixes that
  Python doesn't.

## Open risks

- The alias map needs maintenance as new construction-dependent Params appear; keep it
  small and documented. Unmapped Params surface in "no-GAMS-match", so they're visible
  rather than silently skipped.
- `--params-compare-builds` assumes toggling `t0_snapshot` is the relevant build axis;
  if other build flags (`is_counterfactual`, `if_sub`) also matter, the flag can be
  generalized later to take an explicit build-knob (v2).
