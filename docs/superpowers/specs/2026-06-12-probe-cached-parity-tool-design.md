# Design: `probe.py` — cached, parametrized parity probe

**Date:** 2026-06-12
**Status:** Approved (pending spec review)
**Author:** debug session, gtap7_3x3 altertax parity

## Motivation

During the gtap7_3x3 altertax debug, iteration was slow because each hypothesis
required:

1. **Rebuild + solve from scratch** (~30–60s) for every probe, with ~20 lines of
   solver log noise before any result.
2. **Hand-writing ~6 near-identical `/tmp/probe_*.py` scripts**, each re-importing,
   re-loading params, re-seeding.
3. A **residual test that gave a false lead** (165/907 cells seeded → 107 spurious
   residuals from init-stuck vars) until `apply_solution_hint` was used.
4. **Manual A/B between commits** (checkout 3 files, run, restore — broke a stash once).
5. **Undetected collateral damage**: a faithful pfa/af fix silently broke the ROW
   residual-region base solve; not noticed until ~4 probes later.

The four existing skill tools (check-warmstart, value/residual diff, closure diff,
.nl diff) are good at *locating* a divergence. They do not help with *fast
hypothesis iteration*, *attribution* (did I break this or was it already broken?),
or *collateral-damage detection*. `probe.py` targets exactly those gaps.

## Goals

- Sub-2s repeat queries on a model whose solve is cached.
- Reusable flags instead of bespoke scripts.
- Safe-by-construction against stale code: the model is always *built* with current
  code; only the expensive *solve point* is reused, and it is invalidated when the
  equation code changes.
- Built-in attribution via `--compare-ref <commit>`.
- A robust seed-from-GAMS residual test that reports coverage and refuses to give a
  false lead.

## Non-goals

- Not a live REPL/persistent process (rejected: editing equation code mid-session
  would make an in-memory cached model lie).
- Not a replacement for the four-tool cascade — it complements them.
- No own model constructor: it reuses `GTAPParityAdapter` (the same path as
  `triage.py`) to avoid duplicating build/solve logic and the two-solver drift.

## Architecture

A script `scripts/parity/probe.py`, registered alongside `triage.py`, invoked with
flags:

```
probe.py --template gtap --dataset gtap7_3x3 --scenario altertax_check \
         --show pf,pfa,pi,xiagg --region ROW
```

Flow per invocation:

1. Compute a **cache key** = `sha256(dataset + scenario + closure_name +
   contents_of_key_files)[:16]`.
2. **Build** the model via `GTAPParityAdapter` (always — ~3–5s — so it reflects
   current code). For `altertax_check`/`altertax_shock` this uses the adapter's
   existing build path with the correct altertax closure and residual region.
3. **Solve point:** if `~/.cache/equilibria_probe/<key>.pkl` exists → inject cached
   solution values into the freshly built model (skips the ~30–55s solve). Else →
   solve, extract the solution, write the pickle.
4. Run the requested query against the model (now carrying solved values) and print
   a plain-text table.

### Why build-always, cache-solve-only

The pickle stores **values**, not the live Pyomo object (Pyomo objects don't pickle
cleanly and are fragile across versions). Cached payload:

- `solution`: `{var_name: {idx: value}}`
- `residual_meta` (optional): not cached; residuals are evaluated live on the built
  model after value injection, so any equation-form change is reflected.

Because the model is rebuilt every run from current source, equation edits cannot
produce stale results. The cache only short-circuits the solver, and the cache key
includes the equation source hash, so an equation edit forces a re-solve anyway.

## Cache & invalidation

**Key files hashed** (the ones actually edited during parity work):

- `src/equilibria/templates/gtap/gtap_model_equations.py`
- `src/equilibria/templates/gtap/altertax/parameter_overrides.py`
- `src/equilibria/templates/gtap/altertax/calibration_sequence.py`
- `src/equilibria/templates/gtap/altertax/postmodel.py`
- `src/equilibria/templates/gtap/gtap_parameters.py`

Key = `sha256(dataset + scenario + closure_name + concat(file_contents))[:16]`.

Cache location: `~/.cache/equilibria_probe/<key>.pkl`.

Escape hatches: `--no-cache` (force re-solve, still writes cache), `--clear-cache`
(delete all probe caches). Corrupt/unreadable pickle → ignored, re-solved,
overwritten.

## Queries (v1)

### `--show <vars> [--region R] [--index ...]`
Print `var | idx | value` for the named variables. Optional region/index filters.
Replaces most of the session's `probe_*.py`.

### `--residuals [--top N] [--family F]`
Evaluate `|body − target|` for all active constraints at the cached point, sorted
descending. `--family eq_regy` filters to one family. Replaces the fast path of
check-warmstart / check-solution.

### `--seed-gams <period> [--residuals | --show ...]`
Seed the model with the GAMS point for `<period>` (base/check/shock) via
`apply_solution_hint` (correct GAMS→Pyomo name mapping incl. derived vars). Print
**seed coverage** (`907/1119 cells, 81%`) and **abort with a warning if <95%**,
listing which variable families failed to seed. Then run the requested sub-query
(`--residuals` or `--show`) at the GAMS point. This is the robust residual test that
was missing.

### `--compare-ref <commit> [same query flags]`
Automatic A/B attribution:

1. Run the query on HEAD (current state).
2. Create a temp detached worktree at `<commit>` (`git worktree add --detach <tmp>
   <commit>`); run `probe.py` there as a subprocess with the same flags, reusing the
   current worktree's `.venv`.
3. Clean up the temp worktree (`git worktree remove --force`) in a `try/finally`.
4. Print side-by-side diff: `var/idx | HEAD | <commit> | Δ`.

`try/finally` guarantees no leftover worktree (the stash-breakage problem).

## Output & error handling

- Plain-text aligned tables (consistent with `triage.py`).
- Each query prints a header with the cache key and `[cache hit]` or `[solved +
  cached <N>s]`, so freshness is always visible.
- Solve non-convergence → print partial state + warning, do not crash (so a
  degenerate point like ROW can be inspected).
- `--seed-gams` coverage <95% → abort with per-family detail.
- `--compare-ref` invalid commit / hung worktree → `try/finally` cleanup + clear
  error.

## Reuse

Reuses, without duplicating:

- `parity/_adapter_protocol.py` — `AdapterRegistry`, adapter build/solve.
- `parity/_triage_steps.py` — residual evaluation, seed coverage logic.
- `gtap/_diff_core.py` — GAMS levels reader for `--seed-gams`.

## Testing

`tests/parity/test_probe.py`:

- `--show` on gtap7_3x3 returns non-trivial values.
- Cache hit: a second identical run is <2s (no re-solve).
- Invalidation: touching a key file forces a re-solve (key changes).
- `--seed-gams` reports coverage and aborts below threshold on a deliberately
  partial seed.
- Gates on fixture presence (skip if reference GDX absent, like the other adapters).

## Open risks

- `apply_solution_hint` coverage for some derived vars may be <100% even when
  correct; the 95% abort threshold may need tuning per dataset (make it a
  `--seed-threshold` flag, default 0.95).
- `--compare-ref` requires the target commit's code to be buildable with the current
  venv; if dependencies changed across the commit, the subprocess build may fail —
  reported clearly rather than silently.
