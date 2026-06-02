# GTAP v62 Multi-Period MCP — Design

**Date:** 2026-06-02
**Branch:** `gtap/v62-rollback` (target merge branch: `gtap/v62-multiperiod`)
**Status:** Design — awaiting user approval before plan/implementation
**Author:** Claude (collaborator)
**Motivation commit:** `e14afd3` (Phase 3.36 — 15x10 base parity achieved; check/shock still drift)

## 1. Problem statement

The Python GTAP template (`src/equilibria/templates/gtap/`) achieves 100% parity with NEOS at the **base period** for the 15x10 USA-VegFruit bilateral dataset. However at the **check period** (no shock, identical params, just `regY.fx` released and `regYeq` activated), Python's prices inflate ~30%:

| var | Python check | NEOS check | gap |
|-----|-------------:|-----------:|----:|
| yc[USA] | 15.24 | 11.83 | +29% |
| regy[USA] | 20.19 | 15.67 | +29% |
| pft[USA,UnSkLab] | 1.371 | 1.011 | +36% |
| pft[USA,Land] | 0.360 | 0.879 | −59% |
| pabs[USA] | 1.241 | 0.988 | +25% |

The root cause was identified via `iterloop.gms:151-182`: NEOS solves periods **sequentially**, and when solving period `tsim`, it fixes ~18 lagged variables (`pf, xf, pa, xa, pe, pefob, pmcif, pm, xw, ptmg, psave, pi, uh, pabs, pmuv, pfact, pwfact, gdpmp, rgdpmp, pgdpmp`) of period `tsim-1` to their resolved values. Python currently solves each period as an **independent** MCP with only initial-value inheritance via `t0_snapshot` — losing the structural lag anchors.

Disproven hypotheses (see memory file `project_gtap_neos_check_rebase.md`):
- Warm-start path dependence (no effect)
- 2-stage solve / conditional betap calibration (no effect)
- Folding `rtf` into `dt` (double-counts)
- Configuring `rmuv`/`imuv` Tornqvist basket alone (Fisher self-consistent at inflated equilibrium)
- Fixing `pft` at base values directly (yields `regy ≈ NEOS` but breaks 11 equations via over-determination)

## 2. Goal

Refactor `GTAPModelEquations` and the solve pipeline to a **t-indexed architecture with sequential per-period solves**, replicating the NEOS `solveloop.gms` + `iterloop.gms` pattern with full intertemporal dynamic support (`ifDyn=1` equivalent).

### Success criteria

| Metric | Target |
|---|---|
| 9x10 base parity vs NEOS | 100% (unchanged from commit `28a9b93`) |
| NUS333 base+shock parity vs NEOS/GAMS local | 100% (unchanged) |
| 15x10 base parity vs NEOS | 100% (unchanged from `e14afd3`) |
| 15x10 check parity vs NEOS — `yc[USA]` | `11.83 ± 0.01` |
| 15x10 check parity vs NEOS — `regy[USA]` | `15.67 ± 0.01` |
| 15x10 check parity vs NEOS — `pmuv` | `0.985 ± 0.001` |
| 15x10 check parity vs NEOS — `pft[USA,UnSkLab]` | `1.011 ± 0.005` |
| 15x10 shock parity vs NEOS — `yc[USA]` | `11.83 ± 0.01` (USA-VegFruit +10%) |
| Single-period wall-time regression | ≤ 1.1× baseline |
| 3-period sequential wall-time | ≤ 4× single-period |

## 3. Non-goals

- No change to economic equations (the 200 `eq_*` rules) other than adding the `t` index.
- No change to `gtap_solver.py` closure-application logic.
- No change to `GTAPParameters` data structure (taxes/benchmark remain time-invariant unless explicitly shocked per period).
- No exotic intertemporal dynamics beyond what `iterloop.gms` defines (e.g., no overlapping-generations, no perfect-foresight optimization across periods).
- No change to CLI public interfaces of `run_gtap.py` (`validate-shock`, etc.) beyond adding optional `--t-set` flag.

## 4. Architecture

### 4.1 High-level

```
GTAPModelEquations
├── build_model(t_set: tuple[str,...] = ("base","check","shock"))
│   ├── Adds Sets: model.t, model.t0, model.ts
│   ├── Adds dim `t` (last position) to all endogenous Vars
│   ├── Adds dim `t` (last position) to policy Params (imptx, exptx, prdtx, fcttx, kappaf)
│   ├── Equations rule-defined per (..., t) with $ts(t)/$t0(t) guards via Constraint.Skip
│   └── Returns ONE ConcreteModel containing all periods
│
└── solve_sequential(model, params, closure)  [new, in run_gtap.py]
    ├── For each tsim ∈ t_set (in order):
    │   ├── apply_iterloop_fixings(model, tsim, ...)         [new module]
    │   ├── _set_active_t_slice(model, tsim)
    │   ├── _run_path_capi_nonlinear_full(..., active_t=tsim)
    │   └── snapshot resolved values (already in model.var[..., tsim].value)
    └── Returns {tsim: {residual, code, ...}}
```

### 4.2 Sets

| Set | Definition | Mirrors GAMS |
|---|---|---|
| `model.t` | ordered Set `("base","check","shock")` (or user-specified) | `set t / base, check, shock /` |
| `model.t0` | Set within `t`, contains only first element | `set t0(t) / base /` |
| `model.ts` | Set within `t`, contains non-first elements (sim periods) | `set ts(t) / check, shock /` |

### 4.3 Variable indexing pattern

Convention: `t` is the LAST index. Examples:

| Variable | Before | After |
|---|---|---|
| `pf` | `Var(model.r, model.f, model.a, ...)` | `Var(model.r, model.f, model.a, model.t, ...)` |
| `regy` | `Var(model.r, ...)` | `Var(model.r, model.t, ...)` |
| `pmuv` | `Var(within=NonNegativeReals, ...)` (scalar) | `Var(model.t, within=NonNegativeReals, ...)` |
| `walras` | `Var(...)` (scalar) | `Var(model.t, ...)` |

**Invariants kept time-independent** (no `t` dim): calibration params (`σ`, `ε`, `αd`, `αm`, `kappaf` when not shocked, `phiP`, `betap`, `betag`, `betas`, `phi`).

**Policy params get `t` dim**: `imptx`, `exptx`, `prdtx`, `fcttx`, `fctts`, `dtxshft`, `mtxshft`, `rtxshft`. These can be shocked per period.

### 4.4 Equation indexing pattern

Each `eq_X` constraint definition gains `model.t` as last index, and its rule signature gains `t` as last arg. Guards using GAMS `$ts(t)` translate to `Constraint.Skip`:

```python
def eq_regy_rule(model, r, t):
    if t not in model.ts:
        return Constraint.Skip          # mirrors GAMS $ts(t)
    return model.regy[r, t] == model.facty[r, t] + model.ytax_ind[r, t]

model.eq_regy = Constraint(model.r, model.t, rule=eq_regy_rule)
```

### 4.5 Lag operator `t-1`

Helper in `gtap_model_equations.py`:

```python
def prev_t(t_val: str, t_set: tuple[str, ...]) -> str | None:
    idx = t_set.index(t_val)
    return t_set[idx - 1] if idx > 0 else None
```

Used in equations that GAMS expresses with `tsim-1`, e.g. kstock evolution:

```python
def eq_kstock_rule(m, r, t):
    tp = prev_t(t, t_set)
    if tp is None:
        return Constraint.Skip
    return m.kstock[r, t] == m.kstock[r, tp] * (1 - m.depr[r, tp]) + m.netInv[r, tp]
```

For `eq_pwfact` (Tornqvist Fisher world factor index) the reference shifts from `t0` to `t-1` when in dynamic mode — this is the structural anchor that keeps `pft ≈ 1` at check.

### 4.6 New module: `gtap_iterloop.py`

Path: `src/equilibria/templates/gtap/gtap_iterloop.py`
Estimated size: ~300 lines.
Public API:

```python
def apply_iterloop_fixings(
    model: ConcreteModel,
    tsim: str,
    *,
    t_set: tuple[str, ...],
    sets: GTAPSets,
    params: GTAPParameters,
    flags: dict[str, dict],
    first_year: str = "base",
) -> None:
    """Mirror of iterloop.gms — fix vars/params for period tsim BEFORE solving."""
    _fix_tax_instruments(model, tsim, params)              # iterloop.gms L23-37
    _fix_trade_margins(model, tsim)                         # L57
    _set_price_lower_bounds(model, tsim, t_set)             # L61-88
    _fix_inactive_flows(model, tsim, t_set, flags)          # L92-147 (loop t0)
    _fix_lagged_state(model, tsim, t_set, first_year)       # L151-182 (critical block)
```

The `LAGGED_VARS` constant in `_fix_lagged_state` matches the 18-variable list from `iterloop.gms:151-182`:

```python
LAGGED_VARS = [
    ("axp", ("r","a")),
    ("lambdand", ("r","a")), ("lambdava", ("r","a")),
    ("lambdaio", ("r","i","a")), ("lambdaf", ("r","f","a")),
    ("pf", ("r","f","a")), ("xf", ("r","f","a")),
    ("pa", ("r","i","aa")), ("xa", ("r","i","aa")),
    ("pe", ("r","i","rp")), ("pefob", ("r","i","rp")),
    ("pmcif", ("r","i","rp")), ("pm", ("r","i","rp")),
    ("xw", ("r","i","rp")),
    ("ptmg", ("m",)),
    ("psave", ("r",)), ("pi", ("r",)),
    ("uh", ("r","h")),
    ("pabs", ("r",)),
    ("pmuv", ()), ("pwfact", ()),
    ("pfact", ("r",)),
    ("gdpmp", ("r",)), ("rgdpmp", ("r",)), ("pgdpmp", ("r",)),
]
```

### 4.7 New entry point: `solve_sequential`

Lives in `scripts/gtap/run_gtap.py`:

```python
def solve_sequential(
    model, params, *, closure_config, t_set, equation_scaling=True, tol=1e-8
) -> dict[str, dict]:
    results = {}
    flags = _build_flags_dict(params)
    for tsim in t_set:
        logger.info(f"[t={tsim}] starting")
        apply_iterloop_fixings(
            model, tsim,
            t_set=t_set, sets=params.sets, params=params,
            flags=flags, first_year=t_set[0],
        )
        _set_active_t_slice(model, tsim)
        r = _run_path_capi_nonlinear_full(
            model, params,
            closure_config=closure_config,
            equation_scaling=equation_scaling,
            path_capi_convergence_tol=tol,
            active_t=tsim,
        )
        logger.info(f"[t={tsim}] res={r['residual']:.4e} code={r['code']}")
        results[tsim] = r
        if r["code"] != 1:
            logger.error(f"[t={tsim}] failed to converge — aborting sequence")
            break
    return results
```

`_set_active_t_slice(model, tsim)`: deactivates constraints whose last index ≠ tsim; reactivates those = tsim. Combined with the `.fix()` calls in `_fix_lagged_state`, the PATH MCP becomes square over only the slice's free variables.

### 4.8 Runner integration

`_run_path_capi_nonlinear_full` gains kwarg `active_t: str | None = None`. When set:
- Structural matching filters pairs to (var, eq) with last index == `active_t`
- Jacobian sparsity logging reports slice only
- Scaling and PATH call operate on the slice's free vars

When `active_t=None`, full-model legacy behavior (preserves the current path for `t_set=("base",)`).

### 4.9 Shock application

Shocks applied as Param overrides at `t=shock` before `solve_sequential`. Example (`--shock-mode tm_pct`, USA-VegFruit +10%):

```python
for (r, i, rp), cur in params.taxes.imptx.items():
    if i == "VegFruit" and r == "USA" and r != rp:
        new_val = (1.0 + cur) * 1.10 - 1.0
        params.taxes.imptx[(r, i, rp, "shock")] = new_val
        # base/check inherit cur via default
```

This requires policy Params to be initialized over `model.r × ... × model.t` with the original (time-invariant) values, and overrides applied per-period.

## 5. CLI / public interface changes

- **`GTAPModelEquations.__init__`** gains kwarg `t_set: tuple[str,...] = ("base","check","shock")`. Default is multi-period for 15x10; can be overridden to `("base",)` to recover single-period legacy behavior.
- **`run_gtap.py` CLI** gains `--t-set base,check,shock` flag on `validate-shock` and `validate-gams-parity` commands (default = multi-period).
- **`t0_snapshot` parameter** of `GTAPModelEquations` becomes DEPRECATED (warning on use). Inheritance is now structural via `_fix_lagged_state`.
- **`_run_homotopy_shocked`** becomes DEPRECATED. Homotopy is no longer needed — sequential solve provides natural warm-start.

## 6. Backwards compatibility

Three guarantees:

1. **`t_set=("base",)` is bit-exact equivalent to current single-period path.** Validated via Fase 1 regression (see §9).
2. **9x10 / NUS333 datasets default to single-period (`t_set=("base",)`)** during Fase 1-2 to avoid risk to existing parity. Migrated to multi-period in Fase 3 only after 15x10 check parity confirmed.
3. **`GTAPParameters` API unchanged.** Multi-period Param indexing is internal to model construction; user-facing dict access patterns (`params.taxes.imptx[(r,i,rp)]`) unchanged when querying time-invariant values.

## 7. Error handling

| Condition | Behavior |
|---|---|
| `t_set[0] != "base"` | `ValueError` in `__init__` |
| `len(t_set) == 0` | `ValueError` in `__init__` |
| Lagged var has no value (None/NaN) in `tprev` | `RuntimeError("lagged var X[idx, tprev] not solved — did previous period fail?")` |
| Period `tsim` fails to converge | Log error, abort remaining periods, return partial results dict |
| Shock applied to period not in `t_set` | `KeyError` with explicit message |

## 8. Testing

New file: `tests/gtap/test_multiperiod.py`

1. **`test_single_period_equivalence`** — `t_set=("base",)` produces bit-identical numbers to legacy path on 9x10 and NUS333.
2. **`test_check_no_shock_matches_base`** — `t_set=("base","check")` without shocks produces `var[..., "check"] == var[..., "base"]` for all endogenous vars (within solver tolerance).
3. **`test_15x10_check_parity`** — `yc[USA, "check"] == NEOS 11.83 ± 0.01`; same for `regy`, `pmuv`, `pft[USA,UnSkLab]`.
4. **`test_15x10_shock_parity`** — USA-VegFruit +10% applied at `t=shock`; `yc[USA, "shock"] == NEOS shock value ± 0.01`.
5. **`test_lagged_var_fixing`** — after `apply_iterloop_fixings(model, "check", ...)`, all vars in `LAGGED_VARS` are `.fixed == True` for `t="base"`.
6. **`test_shock_param_isolation`** — applying shock at `t=shock` does NOT modify `param[..., "base"]` or `param[..., "check"]`.

Existing tests (`test_gtap_parity_pipeline.py`, `test_full_model.py`, etc.) must continue passing unchanged with default `t_set`.

## 9. Validation plan (4 phases)

### Phase 0 — Baseline snapshot

```bash
.venv/bin/python scripts/gtap/validate_gams_parity.py --dataset 9x10
.venv/bin/python scripts/gtap/bench_nus333_dual.py
.venv/bin/python scripts/gtap/diff_9x10_full.py > /tmp/baseline_9x10.txt
.venv/bin/python scripts/gtap/diff_nus333_full.py > /tmp/baseline_nus333.txt
```
Commit baseline snapshots before any refactor work begins.

### Phase 1 — Refactor with default `t_set=("base",)`

Migrate all 116 Vars and 200 Constraints to t-indexed form. Default kept as single-period. After each milestone commit:
- 9x10 validate_gams_parity → still 100%
- NUS333 diff_full → 0 cells diverge

If regression detected, revert before proceeding.

### Phase 2 — Activate multi-period for 15x10 only

`t_set=("base","check","shock")` enabled when dataset is 15x10. Implement `gtap_iterloop.py` and `solve_sequential`. Validate:
- 15x10 check parity criteria (see §2 success criteria)
- 15x10 shock parity criteria
- 9x10 / NUS333 untouched (still single-period default)

### Phase 3 — Migrate 9x10 / NUS333 to multi-period opt-in

Only after Phase 2 fully green. Validate that `t_set=("base","check","shock")` on 9x10 ALSO matches NEOS check/shock for 9x10. Merge to `main` only when all 6 success criteria green.

## 10. Rollback plan

If Phase 2 fails (e.g., `pft` still inflates at check after t-indexing + iterloop):

**Option A — full rollback:** `git reset --hard e14afd3`; fall back to the "sequential-with-lag-anchors" approach from the original brainstorming (smaller refactor, no t-indexing).

**Option B — debug:** instrument `_fix_lagged_state` to log which vars remain `.fixed == False` after iterloop; identify gaps vs NEOS's `iterloop.gms` reference; patch.

The refactor is designed so partial rollback is straightforward: `gtap_iterloop.py` is isolated; t-indexing modifications to `gtap_model_equations.py` are mechanical and revertible.

## 11. Git discipline (per `CLAUDE.md`)

Branch: `gtap/v62-multiperiod` cut from `gtap/v62-rollback`. Commit per milestone with message format:

```
gtap(v62): Phase 4.N — <description>

Result: <metric impacted>
Residual base: <value>
Test status: 9x10 ✓, NUS333 ✓, 15x10 base ✓, 15x10 check ?
```

Merge to `main` via PR only when all success criteria in §2 are green.

## 12. Estimated effort

- Phase 0 (baseline): 1 day
- Phase 1 (t-indexing 116 vars + 200 eqs): 2 weeks
- Phase 2 (iterloop + solve_sequential + 15x10 validation): 1 week
- Phase 3 (migrate 9x10/NUS333 + cell-by-cell validation): 1 week

**Total: ~3-4 weeks.**

Effort dominated by Phase 1 (mechanical but voluminous) and the careful Phase 1 regression validation per-commit. Phase 2 is the highest-uncertainty phase because it's where the actual physics (NEOS check parity) is validated.

## 13. Open questions

None at design time. All clarifying questions resolved during brainstorming.
