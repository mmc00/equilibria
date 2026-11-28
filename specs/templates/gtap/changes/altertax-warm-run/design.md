# A warm AlterTax run on the family route as fast as GEMPACK design

## Context

One warm 20x41 run of `scripts/family/altertax_rungtap_measure.py` (family cache full,
MUMPS, `OMP_NUM_THREADS=1`), about 170 s on a quiet laptop, timed with
`equilibria.family.phases` and wall clocks around the main calls:

| Part | s | GEMPACK |
|---|---|---|
| Python, JAX and equilibria imports | ~16 | loads `gtapv7.exe` |
| Setup: cache keys and loads, `mp_build`, masks, the first `square` of check, system builds | ~40 | "reads, formulas etc: less than one second" |
| `settle_check` and `check`: two Newton solves of the base (2 iterations each) | ~35 | no solve of the base |
| `shock`: one Newton solve (4 iterations) | ~20 | Gragg 8-16-32 steps, 56 LU factorizations |
| Decomposition path: K=2 and K=4, four more Newton solves, headers at each point, Richardson | ~45 | subtotals summed inside the same steps |
| `compare_all` | ~15 | none |

GEMPACK runs with 10 OpenMP threads; the conda `mumps-seq` build is sequential, and 10
threads for MUMPS and MKL did not change the times. McAfee scans every file the runs
write and moves the times of both tools by up to 70 %: comparisons are interleaved.

Tried and dropped: the settle cache (`EQUILIBRIA_FAMILY_SETTLE_CACHE=1` with
`EQUILIBRIA_SEED_CACHE_DISABLE=0`; reading the 64 MB seed costs what the settle solve
costs) and a fixed K=4 path (auto takes the Richardson extrapolation of K=2 and K=4, so
a fixed K=4 gives other numbers).

### The floor is the computation, not the setup

The same AlterTax run three times in one Python process (imports, caches and compiled
programs already in memory from the second run on): the exercise (params, solve and the
decomposition path, `seconds.exercise` of the measure script) took 168.6, 148.5 and
143.6 s, `compare_all` 21.2, 14.9 and 18.4 s. GEMPACK takes about 127 s with everything
included. So items 1 and 4 below help but cannot win alone; the gap is in the
computation itself: two Newton solves of the base, one of the shock, and the path
(four Newton steps plus the headers and changes at each point, which `012b0983` cut
from about 57 to 35 s of Python).

Winning needs a change of numerical method, in this order of size:

- A GEMPACK-like shock: Johansen steps with Gragg extrapolation and the decomposition
  accumulated inside the same steps (the recipe has `shock_method="johansen"`; the
  family route refuses it today). This needs its own ADR.
- Item 3 below within the current Newton method: at most one solve less (~20 s).

## Approach

1. Persist, per family cache key and period, what `_period_system` builds and does not
   depend on the levels: `CellIndex`, `ArrayIncidence`, the squared ids and the kept
   family systems. Load them when the key and the code digest match.
2. Let the family route copy the settled base into the check period when the recipe has
   `base_calibrated=True` and `solve_check=False`, with the residual at that point
   checked against the Newton tolerance (refuse, as today, when it does not hold).
3. Solve the shock path in K sub-steps once (K from the auto rule) and take both the
   shock point and the decomposition from those sub-steps.
4. Add a `--no-compare` timing to the bench and a script that interleaves GEMPACK and
   equilibria runs.

## Alternatives considered

- More Python polishing: after `perf/family-warm-windows`, what is left is under the
  noise of the machine (about 10 s against a ±25 % spread).
- A multithreaded MUMPS: worth a measurement, but threads did not help the BLAS part.

## Verification

- Items 1 and 4: the MAR-246 golden and the family tests unchanged, bit for bit.
- Items 2 and 3: every AlterTax gate against the GEMPACK references of
  `data/gtap7/*/gempack_altertax-*` and `tests/family/test_altertax_*`.
- Time: GEMPACK and equilibria interleaved, at least three rounds, on the laptop.
