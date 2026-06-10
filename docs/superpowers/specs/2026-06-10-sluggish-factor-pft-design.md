# Design: sluggish factor aggregate price (`pft[r,sf]`) market clearing

**Date:** 2026-06-10  
**Branch:** gtap-parity-decision  
**Status:** approved

## Problem

`pft[r,sf]` (aggregate price for sluggish factors Land/NatRes) has no active equation in Python.
PATH leaves it at its initialization value (1.0) throughout the solve.

In GAMS, `pft[r,sf]` is determined structurally: `xfteq` and `pfteq` apply to `fm` (mobile
factors), and for `sf` (sluggish), the MCP complementarity between `xft[r,sf]` and `pft[r,sf]`
anchors the price via the supply-demand balance. Python lacks the equivalent constraint pair.

**Symptom:** `pft[Land]` and `pft[NatRes]` diverge ~2–5 cells across gtap7_3x3 and gtap7_5x5,
accounting for the residual ~0.1% gap after all other fixes.

**nus333 is unaffected** because it has no sluggish factors.

---

## Design

Four targeted changes, all in `gtap_model_equations.py`. No changes to `gtap_solver.py` or
`run_gtap.py` unless post-validation reveals a DOF count regression.

### 1. Calibration — extend `aft`/`etaf` to `sf`

**File:** `src/equilibria/templates/gtap/gtap_model_equations.py`  
**Location:** loop at line ~1844 (`for factor in self.sets.f`)

Change the guard from `if factor in self.sets.mf` to:

```python
if factor in self.sets.mf or factor in self.sets.sf:
```

This computes `aft[(r,sf)]` (benchmark aggregate supply) and `etaf[(r,sf)]` (supply elasticity
from `_lookup_etrae`) for sluggish factors using the same logic already used for mobile factors.

With standard GTAP data (`etrae` not set for Land/NatRes), `_lookup_etrae` returns 0.0 →
inelastic supply, identical to benchmark. With GFTLnd-style data (`etrae` set), the elasticity
is picked up automatically — no extra code path needed.

**Impact on existing datasets:** none. `aft[(r,mf)]` values are unchanged; only new `sf` entries
are added (previously defaulted to 0.0).

### 2. Flags — activate `xftflag` for `sf`

**File:** `src/equilibria/templates/gtap/gtap_model_equations.py`  
**Location:** line ~1523

Change:

```python
xftflag_data[(r, f)] = 1.0 if (any_flow and f in self.sets.mf) else 0.0
```

to:

```python
xftflag_data[(r, f)] = 1.0 if (any_flow and f in (self.sets.mf | self.sets.sf)) else 0.0
```

Mirrors GAMS `cal.gms:661`: `xftFlag(r,fm)$xft.l(r,fm,t0) = 1`, where `fm` covers all
endowment factors with active aggregate supply.

### 3. Equations — remove `mf`-only guard in `eq_xft` and `eq_xfteq`

**File:** `src/equilibria/templates/gtap/gtap_model_equations.py`  
**Location:** `eq_xft_rule` (~line 4759) and `eq_xfteq_rule` (~line 4787)

In both rules, replace:

```python
if f not in self.sets.mf:
    return Constraint.Skip
```

with nothing — let `xftflag` be the sole gate (the flag check that follows already handles
inactive factors). Both constraints are now active for any `f` (mobile or sluggish) where
`xftflag[(r,f)] = 1`.

**What this adds for `sf`:**

- `eq_xft[r,sf]`: market clearing — `xft[r,sf] = Σ_a xf[r,sf,a] / xscale[r,a]`
- `eq_xfteq[r,sf]`: supply curve — `xft[r,sf] = aft[r,sf] * (pft[r,sf] / pabs[r])^etaf[r,sf]`

Together these two equations determine `xft[r,sf]` and `pft[r,sf]` jointly, exactly as GAMS
does via the MCP complementarity pair.

With `etaf=0`: `eq_xfteq` pins `xft[r,sf] = aft[r,sf]` (fixed supply); `eq_xft` then pins
`pft[r,sf]` to the price that clears the market.

### 4. `eq_pfeq` — no change needed

The `fnm` branch (line ~4844) correctly computes `xf[r,sf,a]` as a function of `pfy[r,sf,a]`
and `pabs[r]`. It does not reference `pft[r,sf]`, which is correct — GAMS `pfeq` for `fnm`
also does not reference `pft`. The two new equations in section 3 anchor `pft[r,sf]` without
touching `eq_pfeq`.

---

## DOF accounting

Each `(r,sf)` pair with `xftflag=1` gains:
- **+2 equations:** `eq_xft[r,sf]` + `eq_xfteq[r,sf]`
- **+1 variable implicitly freed:** `xft[r,sf]` (was unconstrained before)
- **`pft[r,sf]`:** already a free Var, now anchored by the second equation

Net DOF change: 0 per `(r,sf)` pair. `apply_aggressive_fixing_for_mcp` gap should remain
unchanged. Verify post-implementation via the logged `MCP gap before aggressive fixing` line.

---

## Validation plan

1. Commit all changes (clean working directory per CLAUDE.md discipline).
2. Run `diff_gtap7_local.py` on gtap7_3x3 → expect `pft[Land/NatRes]` cells to match → 100%.
3. Run `diff_gtap7_local.py` on gtap7_5x5 → same expectation.
4. Run diff on nus333 and 9x10 → expect no regression (100% / 99.96%).
5. Check solver log: `MCP gap before aggressive fixing` must remain 0 (or same value as before).
6. Record SHA of each run in commit message per project convention.

---

## Files touched

| File | Change |
|------|--------|
| `src/equilibria/templates/gtap/gtap_model_equations.py` | 3 edits (~5 lines each) |
| `src/equilibria/templates/gtap/gtap_solver.py` | none (unless DOF regression found) |
| `scripts/gtap/run_gtap.py` | none |
