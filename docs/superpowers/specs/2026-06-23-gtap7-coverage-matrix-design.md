# GTAP 7 Coverage Matrix — single source of truth

## Goal

Make one declarative table the **single source of truth** for GTAP 7 parity
coverage — across `dataset × kind (gtap | altertax) × ifSUB × phase (base/check/shock)`
— and wire it so that:

1. the **tests** read it to parametrize themselves and to assert per-row gap
   thresholds (the matrix becomes an *executable contract*), and
2. the **documentation** Markdown table is *generated* from it and kept in sync
   by a CI test (golden-file pattern), so doc and reality can never drift.

Today coverage is implicit: the `.nl` gate discovers datasets by scanning fixture
files on disk, the altertax gate hardcodes a single global `98%` threshold, and
the human-readable status lives in prose in `CLAUDE.md` that drifts from the code.
This replaces all three with one table.

## Non-goals

- Not changing any model equations, parameters, or builders. `_parity_datasets.py`
  (the *builder* registry) is untouched — this is purely a *coverage* layer on top.
- Not generating new GAMS fixtures (15x10/3x4 `gams_check.nl`, 20x41). Those remain
  out-of-CI rows in the matrix with honest `ci_status`. Generating them is a
  separate, later task.
- Not making the altertax *solver* gate run in CI. It stays local-only (no
  self-hosted runner); the matrix records that honestly as `ci_status="local"`.

## Architecture

Three components, one data flow:

```
scripts/gtap/coverage_matrix.py          ← SINGLE SOURCE (rows + helpers)
        │
        ├──→ tests/templates/gtap/test_gtap7_nl_parity.py        (reads rows: which phases per dataset)
        ├──→ tests/templates/gtap/test_altertax_multiperiod_parity.py  (reads rows: per-row gap_min)
        └──→ scripts/gtap/gen_coverage_doc.py ──→ docs/gtap7_coverage_matrix.md
                     ▲                                   │
                     └────── test_coverage_doc_sync ─────┘  (CI: regenerate == committed, else FAIL)
```

- **`coverage_matrix.py`** — the data: a list of `Row` records + small query
  helpers. No model building, no I/O beyond importing the dataclass.
- **`gen_coverage_doc.py`** — reads the matrix, emits the Markdown table to
  `docs/gtap7_coverage_matrix.md`. Deterministic (stable row order).
- **Tests** — consume the matrix (details below).

## The row schema

```python
@dataclass(frozen=True)
class Row:
    dataset: str          # "gtap7_3x3", "nus333", "9x10", ...
    kind: str             # "gtap" (single-period) | "altertax" (multi-period)
    ifsub: int | None     # None for gtap; 0 or 1 for altertax
    phases: tuple[str, ...]   # phases with coverage, e.g. ("base","check","shock")
    gap_min: float        # CONTRACT: minimum real-cell match % the test asserts
    gap_note: str         # human snapshot of the *measured* value, e.g. "100%", "~99%"
    ci_status: str        # "ci" | "local" | "blocked"
    ref: str              # provenance of the number (fixture filename or note)
```

**`gap_min` vs `gap_note` — the critical distinction.** `gap_min` is a *conservative
floor* the test enforces; `gap_note` is the *measured* value for humans.
`gap_min` MUST sit safely below the measured value so basin/boundary micro-noise
cannot cause a spurious failure. Rule of thumb: `gap_min = floor(measured) - margin`
(e.g. measured 100% → `gap_min=99.5`; measured ~99% → `gap_min=98.0`). Never set
`gap_min` equal to a measured 100%.

**`ci_status` semantics:**
- `"ci"` — runs on ubuntu CI without a solver (the `.nl` gate rows).
- `"local"` — runs only locally (needs PATH+GAMS); the altertax solver rows. CI
  skips them (no solver), so the test self-skips; honest, not hidden.
- `"blocked"` — cannot be verified because the *reference* is unsound (20x41 NEOS
  came back Infeasible). Documented in matrix + doc; the test `skip`s with the reason.

## Initial matrix content (honest snapshot)

Single-period `.nl` gate (CI, no solver):

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| nus333 | gtap | None | base,shock | 99.5 | 100% (NEOS+GAMS) | ci | nus333 NEOS |
| 9x10 | gtap | None | base,shock | 99.5 | 100% (NEOS) | ci | job 18737509 |
| gtap7_3x3 | gtap | None | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_5x5 | gtap | None | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_10x7 | gtap | None | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_15x10 | gtap | None | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |
| gtap7_3x4 | gtap | None | base,shock | — | 0 diffs .nl | ci | gams_base/shock.nl |

Altertax multi-period (solver gate, local-only) — both ifSUB modes:

| dataset | kind | ifsub | phases | gap_min | gap_note | ci_status | ref |
|---|---|---|---|---|---|---|---|
| gtap7_3x3 | altertax | 0 | base,check,shock | 98.0 | ~99% | local | out_altertax_ifsub0.gdx |
| gtap7_3x3 | altertax | 1 | base,check,shock | 98.0 | ~99% | local | out_altertax_ifsub1.gdx |
| gtap7_5x5 | altertax | 0 | base,check,shock | 99.5 | 100% | local | out_altertax_ifsub0.gdx |
| gtap7_5x5 | altertax | 1 | base,check,shock | 99.5 | 100% | local | out_altertax_ifsub1.gdx |
| gtap7_10x7 | altertax | 0 | base,check,shock | 98.0 | ~99% | local | out_altertax_ifsub0.gdx |
| gtap7_10x7 | altertax | 1 | base,check,shock | 98.0 | ~99% | local | out_altertax_ifsub1.gdx |
| gtap7_15x10 | altertax | 0 | base,check,shock | 99.0 | 99.30% | local | out_altertax_ifsub0.gdx |
| gtap7_15x10 | altertax | 1 | base,check,shock | 99.0 | 99.30% | local | out_altertax_ifsub1.gdx |
| gtap7_3x4 | altertax | 0 | base,check,shock | 99.0 | 99.61% | local | out_altertax_ifsub0.gdx |
| gtap7_3x4 | altertax | 1 | base,check,shock | 99.0 | 99.56% | local | out_altertax_ifsub1.gdx |
| gtap7_20x41 | altertax | 0 | base | — | blocked | blocked | NEOS ref Infeasible |

Notes:
- `gap_min` applies to rows whose contract is a *percentage match* (the altertax
  solver rows, and nus333/9x10 which have a real 100% solve). Pure `.nl`-gate rows
  (the gtap7_* single-period rows) have NO percentage contract — their contract is
  "0 non-structural-FP coefficient diffs", enforced by the family-diff logic — so
  `gap_min=None` for those. nus333/9x10 are `kind="gtap"` but DO carry a `gap_min`
  (they have a measured solve %); thus `gap_min` is keyed to *whether a % is
  measured*, NOT to `kind`. The schema invariant is therefore: `gap_min is None`
  for the gtap7_* `.nl`-only rows; non-None where a percentage is measured.
- The `check` phase only appears for datasets that have a committed `gams_check.nl`
  (3x3/5x5/10x7). 15x10/3x4 list `check` in their altertax phases (the solver does
  run all 3) but their `.nl`-gate row stays base,shock until a check fixture exists.
- The altertax-`gap_min` floor for 3x3/10x7 is 98.0 (measured ~99%, margin ~1pt);
  5x5 is 99.5 (measured 100%); 15x10/3x4 is 99.0 (measured 99.3–99.6%).

## How each test consumes the matrix

### `test_gtap7_nl_parity.py` (CI, no solver)
- Parametrize over matrix rows where `kind=="gtap"` and `ci_status=="ci"`.
- For each, diff the phases in `row.phases` (so adding `check` to a row's phases
  + the fixture turns it on — no test edit). Keep the existing per-phase fixture
  existence skip.
- Assertion unchanged in nature (0 non-structural-FP coefficient diffs per phase).

### `test_altertax_multiperiod_parity.py` (local-only)
- Parametrize over matrix rows where `kind=="altertax"` and `ci_status!="blocked"`.
- Replace the global `MATCH_THRESHOLD = 98.0` with `row.gap_min` — assert
  `match_pct >= row.gap_min`. The matrix is now the contract.
- `ci_status=="blocked"` rows are skipped with `row.ref` as the reason.
- Existing skips (no PATH solver, missing fixture, missing HAR) stay.

### `test_coverage_doc_sync.py` (CI, new)
- Regenerate the doc from the matrix in-memory and compare to the committed
  `docs/gtap7_coverage_matrix.md`. On mismatch, FAIL with a diff and the message
  "run `uv run python scripts/gtap/gen_coverage_doc.py` and commit". No solver, runs
  on ubuntu.

## The generated doc

`docs/gtap7_coverage_matrix.md` — header + the two tables above, rendered from the
matrix in stable order, with a top banner: "GENERATED FROM
scripts/gtap/coverage_matrix.py — do not edit by hand; run gen_coverage_doc.py".
Linked from `CLAUDE.md` (replacing the hand-maintained status table) and from
`docs/site/benchmarks.md`.

## Error handling / edge cases

- Matrix row references a dataset with no fixture on disk → test `skip`s with a
  clear message (mirrors today's behavior), never a hard fail.
- A `local` row collected in CI (no PATH) → self-skips (as today).
- Doc out of sync → `test_coverage_doc_sync` fails with the regenerate instruction.
- `gap_min` is keyed to whether a percentage is measured (see Notes above), not to
  `kind`: the gtap7_* `.nl`-only rows use `gap_min=None`; nus333/9x10 (measured
  solve %) and all altertax rows carry a non-None `gap_min`. A `.nl`-only row with a
  non-None `gap_min`, or a measured-% row with `gap_min=None`, is a matrix authoring
  error caught by the import-time self-validation in `coverage_matrix.py`.

## Testing strategy

- `coverage_matrix.py` ships with a tiny self-test (schema invariants: `ifsub` is
  None iff `kind=="gtap"`; `ci_status` in the allowed set; `gap_min` is None exactly
  for the `.nl`-only gtap7_* rows and non-None for measured-% rows).
- `gen_coverage_doc.py` is covered by `test_coverage_doc_sync` (golden file).
- The two parity gates are covered by their own existing runs; this spec only
  changes *how they get their parameters/thresholds*, verified by running them.

## Why this design

- **One table, enforced.** The contract (`gap_min`) and the doc both derive from
  the same rows; a CI sync test makes drift impossible.
- **Honest.** `ci_status` shows exactly what runs in CI vs local vs blocked — no
  green-washing by omission.
- **Low-coupling.** Builders (`_parity_datasets.py`) stay separate; the coverage
  layer is additive and removable.
- **Extensible.** Adding a dataset/mode = one `Row`; regenerate the doc; done.
