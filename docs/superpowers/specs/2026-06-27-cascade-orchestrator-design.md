# Cascade Orchestrator — Phase 2 Design

**Date:** 2026-06-27
**Branch:** built on `sphenoid-soy` (Phase 1, commit `606a576`)
**Status:** design approved; implementation pending

## Where this sits

The parity-debug cascade is a sequence of diagnostic tools, each seeing one layer
of a Python↔GAMS gap and blind to the others (see `CLAUDE.md`, "cascade de tools",
and the `equilibria-parity-debug` skill). Closing a gap by hand means running them
in order and reading each JSON result.

- **Phase 1 (COMPLETE, `sphenoid-soy` @ `606a576`):** the six cascade tools that
  emit machine-readable output — `probe.py`, `drift_test.py`, `nl_compare.py`,
  `diff_mcp_pairing.py`, `validate_reference.py`, `diff_calibration.py` — were
  standardized to print **only one strict-JSON object to stdout** (no text mode,
  no `--json` flag). All six already import the shared contract `_parity_json.py`
  and route through `run_tool`/`emit`/`make_violation`; all solver/log/progress
  goes to stderr. **This is the real precondition for Phase 2 and it is satisfied:
  6/6 tools emit pure JSON.** There is no remaining tool-wiring work before the
  orchestrator.
- **Phase 2 (THIS DESIGN):** an orchestrator that runs the cascade automatically
  for a dataset across periods, stops at the first layer that explains the gap,
  and adds the one layer that has no JSON tool yet — the KKT/marginals layer.

The orchestrator is **not** built until this design is approved and the spec is
reviewed; this document is the design.

## Scope (MVP)

- **Datasets:** iterate and validate on **`gtap7_3x3`** (smallest dataset with a
  durable local reference; the historical free-DOF `pva` case). Other datasets are
  a later concern (see Known debts).
- **Periods:** `check` and `shock` (the gtap7 altertax axis).
- **Layers:** the **six JSON-emitting tools** plus the new **KKT/marginals** layer.
  The non-JSON cascade layers (`triage` warm-start/value-residual, `diff_closure`,
  `diff_equation_form`, `diff_holdfixed`, `diff_tautology`) are **out of MVP
  scope**, documented here as future work — they require human reading or do not
  yet emit a binary signal.

## Architecture

### Module

`scripts/gtap/cascade_orchestrator.py` — one unit, one purpose: orchestrate. It
(1) resolves per-dataset config, (2) per period runs layers in diagnostic order
until the first real `dirty`, (3) aggregates and emits. **It reimplements no
parity logic** — each layer's logic lives in its tool; the orchestrator invokes,
reads `status`, and builds the report.

### Invocation: subprocess, not import

Each tool already guarantees exactly one strict-JSON line on stdout (via
`run_tool`/`stdout_to_stderr`) and an exit code (0=clean / 1=dirty / 2=error) that
mirrors `status`. The orchestrator runs each as a subprocess
(`subprocess.run([".venv/bin/python", <tool>, ...], capture_output=True,
text=True, timeout=...)`), parses `stdout` as JSON, and reads `returncode`.

Rationale: importing the tools would break isolation — they share
`sys.modules["run_gtap"]`, swap global `sys.stdout`, and run heavy Pyomo solves. A
subprocess that blows up does not take down the orchestrator, and its noisy output
is already captured to stderr by the Phase-1 contract.

### CLI

```
cascade_orchestrator.py --dataset gtap7_3x3 --periods check shock
    [--no-stop]        # run all layers without pruning (debug mode)
    [--fire-neos]      # fallback: fire NEOS if no local ref (NOT default)
    [--tool-timeout S] # kill a hung tool
```

## Config resolution (pure functions)

The orchestrator does **not** hardcode scenario/gdx-ref. It reads them from the
parity adapter, the single source of truth:

- **period → scenario** is derived per dataset family. `gtap7_*` has the altertax
  two-period axis: `check → altertax_check`, `shock → altertax_shock`. The
  bundle family (`9x10`/`nus333`) has `base → baseline`, `shock → shock_tm10`
  (no `check`).
- **reference GDX** prefers the **durable backup** at
  `/Users/marmol/proyectos2/equilibria_refs/<dataset>_altertax_cd/out_altertax_ifsub0.gdx`,
  with the adapter's gitignored `output/...out_local.gdx` as fallback.

### Non-silent GDX fallback (REQUIRED)

A silent fallback to a stale/wrong GDX would produce false dirty/clean across all
six layers — the same failure class as nl_compare's vacuity. Therefore the
resolver:

1. **Validates** the chosen GDX exists and is non-empty (`stat().st_size > 0`)
   before any layer runs against it.
2. On fallback, records **what ref and why** in a provenance record that travels
   to the report. "Couldn't use the good ref" must never look like "used the good
   ref."
3. If **no** usable ref exists, the orchestrator **aborts before building any
   GDX-dependent command** — it never emits a `--gdx None`.

### Period intersection (REQUIRED)

Before iterating, intersect `--periods` with the periods the dataset actually has.
A `gtap7_*` dataset drops `base`; a bundle dataset drops `check`. The dropped
periods are reported, not silently ignored. Additionally, a layer is skipped for a
period its tool rejects (e.g. `drift_test` has no `base`).

### `LAYER_SPECS` — single source of truth for invocation

A table, one entry per layer, each with a **pure** builder
`(dataset, period, gdx) -> argv`. Validated against each tool's verified
`argparse` (`--help`), not against prose. Fields: `name`, `tool` (script path),
`periods` (which periods the tool accepts), `seeds` (does it seed-at-GAMS and
solve — the layers a `no_convergence` upstream would make measure noise).

Verified invocations (gtap7_3x3 / shock), in diagnostic order:

| Layer | Builder output |
|-------|----------------|
| `mcp_pairing` (static) | `diff_mcp_pairing.py --dataset gtap7_3x3 --period shock --apply-closure` |
| `nl_compare` (static) | `nl_compare.py --dataset gtap7_3x3 --phase shock --skip-gams` |
| `calibration` (static) | `diff_calibration.py --dataset gtap7_3x3 --gdx <ref> --period check` |
| `validate_reference` (static) | `validate_reference.py --dataset gtap7_3x3 --period shock --gdx <ref>` |
| `probe_seed` (seed+solve) | `probe.py --template gtap --dataset gtap7_3x3 --scenario altertax_shock --seed-gams shock --gdx-ref <ref> --residuals` |
| `drift_test` (seed+solve) | `drift_test.py --dataset gtap7_3x3 --gdx <ref> --period shock` |
| `kkt_marginals` (in-process) | reads `.m` from `<ref>` via `reader.read_equation_values` |

Note: `calibration` is a **benchmark-input** layer (period-agnostic); it reads the
benchmark period (`check` for gtap7) regardless of the shock axis — that is how
the tool is meant to be used.

## Control flow — six branches, none maps to `clean`

The loop classifies each layer result by `(status, error_kind)` and confirms with
the subprocess exit code. **Governing principle (written verbatim into the
report's semantics): no error branch maps to `clean`. `clean` means "measured and
found nothing", never "could not measure."**

```
clean
    → record clean ("measured, found nothing"). CONTINUE to next layer.

dirty
    → record as the EXPLANATION of the gap (first real dirty of the period).
      STOP this period's sweep (unless --no-stop, which continues for a full diagnosis).

("error", "no_common_constraints")        # nl_compare vacuity, issue #23
    → NOT a signal. Record the layer as "vacuous / did-not-opine" — a VISIBLE,
      first-class report state ("the algebraic layer did not opine for lack of
      common constraints"), never omitted. CONTINUE. Does not stop, is not an
      explanation, is NOT conflated with clean.

("error", "no_convergence")
    → upstream cause: the model did not converge at the seeded point. The
      seed→solve layers after it (probe_seed, drift_test) would measure NOISE.
      Record as "not-measurable (upstream)". STOP this period's sweep. Not clean:
      it did not measure.

("error", "exception")
    → the TOOL crashed (traceback captured in meta), but the next layer can still
      be valid — they are orthogonal. Record as "tool broken" (with error_kind and
      message). CONTINUE. "Redundant" = the exit code (2) AND status="error"
      confirm each other, so a tool that crashes weirdly is still classified broken
      and does not leave us blind. Action: record-and-continue.

("error", <any other: gdx_not_found, OR an UNRECOGNIZED error_kind>)   # safety net
    → the unknown is BLOCKING, never "continue as if nothing happened". Record as
      "blocking error (<kind>)". STOP this period's sweep and report it. An
      error_kind the orchestrator cannot interpret is treated as
      not-measurable/blocking by safe default — never degraded to clean or to
      silent-continue. This is what stops a future tool's new error_kind from
      slipping through silently as clean.
```

Each error maps to exactly one of: **dirty** (explains, stops the period),
**stop** (not-measurable upstream: `no_convergence`, or unknown/`gdx_not_found` as
blocking), **continue-but-visible** (did-not-opine: nl_compare's
`no_common_constraints` / #23, noisy in the report, never omitted), or
**continue-but-record** (tool crashed: `exception`). The false-green that issue #23
documents reappears if nl_compare's vacuity is omitted or painted clean — hence it
is a distinct, visible state in the final report.

## KKT / marginals layer (the new one)

In-process, runs last. It reads the equation marginals (`.m`, the KKT dual /
complementarity slack) from the **existing local reference GDX** — no NEOS round
trip in the default path.

- **Reader:** `src/equilibria/babel/gdx/reader.py::read_equation_values`
  (pure-Python, license-independent; extracts the `marginal` field). This is the
  primary path and stays primary.
- **Implementation gate (RESOLVED 2026-06-27):** the spec required a smoke test
  that this reader parses `out_altertax_ifsub0.gdx` BEFORE building the KKT logic.
  **The gate fired and FAILED**, which is exactly what it was for. Findings, with
  `gdxdump` (the repo's trusted GDX reader) as oracle:
  - The ref GDX *does* contain equations and their `.M` marginals — `gdxdump
    Symb=arenteq` shows per-record `'USA'.'check'.M 0.157…` etc., with the period
    (`check`/`shock`) as a key dimension.
  - But `reader.py`'s pure-Python symbol-type detection is a **byte-level
    heuristic** (`_read_symbol_table` guesses type from `type_flag` magic bytes it
    itself marks ambiguous) that **loses sync** on the GTAP GDX symbol table:
    `read_gdx` returns garbage types (150, 248, …) and **0 equation symbols**, so
    `read_equation_values` finds nothing.
  - **Decision (user, with full scope visible):** FIX `reader.py` so it classifies
    Var/Equ correctly for GTAP GDX, keeping it the primary license-independent path
    — rather than fall back to `gdxdump`. This hardens a **shared** module
    (30+ consumers across PEP calibration, GTAP parameters, SAM tools, QA), so the
    fix is gated by a no-regression discipline (below), not a loose patch.
  - **Do not** build the KKT complementarity logic until the repaired reader
    returns verified equation marginals from that GDX, with the existing reader
    test suite still green.
- **Reader-fix discipline (REQUIRED, see plan):**
  1. Establish the green baseline: `tests/babel/gdx/test_reader*.py` +
     `test_multidim_sets.py` + `test_decoder_csv_validation.py` (81 tests, green
     today). The 2 failures in `test_writer.py` are pre-existing and unrelated
     (writer lacks Equation support) — out of scope.
  2. Characterize the bug with `gdxdump` as the source of truth for which symbol is
     Set/Param/Var/Equ.
  3. TDD: a new failing test asserting `read_gdx(out_altertax_ifsub0.gdx)`
     classifies the known equation families (e.g. `arenteq`, `apeeq`) as equations
     and `read_equation_values` returns their marginals.
  4. Fix `_read_symbol_table` minimally; the new test passes AND the full reader
     suite stays green.
- **Decision rule (dirty/clean):** for each MCP equation, check GAMS's
  complementarity (`marginal != 0 ⇒ binding`; `marginal == 0 ⇒ slack`) against the
  state Python gives the equation's paired variable (fixed/free, binding/slack, via
  the adapter's closure). **dirty** = an equation GAMS reports binding (`m != 0`)
  that Python leaves free/non-binding, or vice versa — the KKT signal no other
  layer sees. **clean** = the complementarity states agree.

### NEOS as fallback only

The canonical `gtap7_3x3` reference (`out_altertax_ifsub0.gdx`) is generated
**locally with GAMS v53**, not via NEOS. NEOS is the expensive async fallback
(`submit_altertax_neos.py`: xmlrpc + `while True: sleep(15)` poll + manual save of
the out.gdx) for datasets that exceed the v53 community license (2500 rows).
Therefore NEOS is **not** fired by default; only behind `--fire-neos` when a
dataset has no local ref.

## Output

Respecting the Phase-1 contract:

- **stdout** = one aggregate JSON summary:
  `{dataset, ref, kkt_reader, periods: {<period>: {first_dirty_layer, layers:
  [{tool, status, error_kind, headline}]}}, verdict}`.
- **stderr** = a human-readable tree: period → layer with status + headline,
  layers collapsed after the first dirty.

### Provenance in the report (REQUIRED)

Two runtime decisions change the meaning of the result: which GDX was used
(durable vs adapter-fallback) and which reader path the KKT layer took
(pure-python vs gdxdump-text). Both are recorded **in the final report, alongside
the result** — not lost in stderr. One line:

```
ref: <path> [durable|adapter-fallback], kkt_reader: [pure-python|gdxdump-text]
```

If three months from now a run produces a strange `dirty`, the first question is
"against which GDX, read with which reader?" — and that answer must be grabbed
next to the result, the same non-silent principle already applied to the GDX
fallback.

## Testing

- **Pure-function unit tests** (no solve): the resolver (durable vs fallback vs
  missing; period intersection and drops) and `LAYER_SPECS` builders (argv matches
  the verified `--help` for each tool/period; `--gdx None` is never produced).
- **Dry-run** (`--dry-run`, no subprocess): prints the exact commands per
  (dataset, period) — used during design to validate `build_cmd` against the
  verified argparse.
- **Reader fix (gates KKT)**: existing `tests/babel/gdx/` reader suite green as
  baseline; new test asserts `read_gdx(out_altertax_ifsub0.gdx)` classifies known
  equation families and `read_equation_values` returns non-empty marginals; reader
  suite still green after the fix.
- **End-to-end on gtap7_3x3** (check + shock): one full orchestrator run; assert
  the report shape and that the four-way error classification holds.
- **Regression gate (mandatory before any merge that touches equations/params/data
  loading):** `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v`. The
  orchestrator touches none of those, but the gate is the project rule.

## Known debts (registered so they are not lost)

1. **Regression gate passes vacuously on `gtap7_3x3` (GitHub issue #23).** The
   GAMS fixtures lack `.row`/`.col`, so constraints become anonymous `c[idx]` → 0
   common constraints → 0 failures → green-by-no-comparison. The orchestrator must
   surface nl_compare's `no_common_constraints` as a visible "did-not-opine" state
   (see Control flow) so this vacuity is never read as clean.
2. **Reference resolution for `9x10`/`nus333` is `future`.** Those datasets have no
   durable `*_altertax_cd/` CD reference; they use the bundled `basedata-*.gdx`
   plus a NEOS `out.gdx`, a different resolution path. The MVP iterates on
   `gtap7_3x3` only; the bundle-family branch of the resolver is scoped as future
   work, not half-wired.

## Out of scope (MVP)

- The non-JSON cascade layers (`triage` 0/1, `diff_closure` 2, `diff_equation_form`
  5, `diff_holdfixed` 8, `diff_tautology` 9). Documented as Phase 3.
- Firing NEOS by default (fallback only, behind `--fire-neos`).
- `9x10`/`nus333` reference resolution (debt #2).
