# A warm AlterTax run on the family route as fast as GEMPACK

## Why

On the Windows laptop (i7-1365U), a warm 20x41 AlterTax run on the family route takes
2.77-2.89 min at best (`perf/family-warm-windows`, down from 3.04 on main), while
GEMPACK's `gtapv7.exe` solves the same exercise in 2.12 min when the two run
interleaved on a quiet machine. What remains is not slow Python: the family route does
work that GEMPACK does not do.

## What changes

1. **The squared system on disk.** The cell index, the adjacency, the squared rows and
   columns and the compiled family systems of each period are kept next to the family
   records, under the same key, so a warm process loads them instead of building them.
2. **No second solve of the base.** When the settled base is known to replicate the data,
   the check period is that point, not another Newton solve. The family route accepts
   `solve_check=False` for AlterTax.
3. **The welfare decomposition from the shock solve.** The decomposition is integrated
   along the sub-steps of the shock solve, as GEMPACK accumulates its subtotals inside
   its multi-step solution, instead of solving the path again for K=2 and K=4.
4. **Timing like for like.** The bench reports the run with and without `compare_all`
   (the gate against the reference, which GEMPACK does not do), and times GEMPACK and
   equilibria interleaved in one session.

Items 1 and 4 change no number. Items 2 and 3 change what the route computes and keep
every AlterTax gate against the GEMPACK references (ADR-0008).

## Issues

To open, one per item above.

## Impact (areas, ADRs)

`templates/gtap` (the recipe, the AlterTax exercise) and `solver`
(`src/equilibria/family/`, `src/equilibria/backends/`, which has no baseline yet);
`parity` for item 4 (`compare_all`). Relies on ADR-0008 (the reference decides) and
ADR-0011 (one driver fed by a recipe). Item 3 may need an ADR: it changes how the
welfare decomposition is computed on the family route.
