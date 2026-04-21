# GTAP MCP Structural Gap Note (2026-04-10)

## Context
- Model: `gtap_standard7_9x10` in `equilibria` with PATH-CAPI nonlinear callbacks.
- State before this change: MCP closure used heuristic aggressive fixing (mostly `xda`) to close a +130 equation gap.

## Structural finding
- I ran a bipartite matching on the active Jacobian incidence (`constraints` vs `free variables`).
- Result at pre-aggressive stage:
  - `constraints=15998`
  - `free variables=16128`
  - `gap=130`
  - unmatched variables are predominantly `pf` (plus small macro leftovers).
- With the new matching-based aggressive pass, the fixed set is:
  - `pf`: 128
  - `yi`: 1
  - `chif`: 1

## Code change
- File: `src/equilibria/templates/gtap/gtap_solver.py`
- `apply_aggressive_fixing_for_mcp()` now:
  1. builds structural incidence from active constraints,
  2. runs Hopcroft-Karp matching,
  3. fixes unmatched free variables first,
  4. falls back to previous heuristic only if needed.

## Solve results after change
- `major_iteration_limit=10`:
  - output: `output/gtap_path_capi_nonlinear_status_mi10_structural.json`
  - residual: `2.012326778519535`
  - previous comparable run: `2.0440419264463032`
- `major_iteration_limit=21`:
  - output: `output/gtap_path_capi_nonlinear_status_mi21_structural.json`
  - residual: `2.0115303381655503`
  - previous comparable run: `2.0426555231541697`

## Remaining parity risk vs GAMS
- The factor-price closure still differs from GAMS because the Python `eq_pfeq` is aggregate (`r,f`) while GAMS uses activity-level `pfeq(r,fp,a)`.
- This likely explains why the unmatched structural set is mostly in `pf` and should be the next parity task.
