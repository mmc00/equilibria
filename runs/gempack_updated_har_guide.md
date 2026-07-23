# Generating the RunGTAP `updated.har` refs on Windows (F5)

This guide takes the `gtap/validation-parity-pages` branch to a Windows machine
with RunGTAP, produces the real post-shock `updated.har` for each matrix dataset,
drops it where the parity gate expects it, and completes the `VAR_TO_HEADER` map
so the "against GEMPACK" cell-by-cell page fills with real data.

The Python side (reader → `gempack_levels` → SKIP-if-missing gate) already exists
and is unit-tested against a synthetic fixture. This guide is the Windows half,
and it is now **scripted** — one command produces every solvable dataset.

**Status (2026-07-22 run):** 7 of 8 matrix datasets produced a converged
`updated.har` and are committed under `tests/fixtures/gtap7_gempack/`:

| dataset | residual | max residual ratio | fixture |
|---|---|---|---|
| `nus333` | ROW | 1.07e-07 | `updated_nus333_tm10.har` |
| `9x10` | NAmerica | 1.19e-07 | `updated_9x10_tm10.har` |
| `gtap7_3x3` | ROW | 9.32e-08 | `updated_gtap7_3x3_tm10.har` |
| `gtap7_3x4` | ROW | 1.25e-07 | `updated_gtap7_3x4_tm10.har` |
| `gtap7_5x5` | ROW | 8.89e-08 | `updated_gtap7_5x5_tm10.har` |
| `gtap7_10x7` | ROW | 9.87e-08 | `updated_gtap7_10x7_tm10.har` |
| `gtap7_15x10` | ROW | 9.85e-08 | `updated_gtap7_15x10_tm10.har` |
| `gtap7_20x41` | — | — | **blocked** (see §2c) |

The shock and closure are identical across all datasets — `Shock tm = uniform 10`
on the tariff power (the welfare-parity-validated GEMPACK mirror of the Python
gate's uniform +10% `imptx` shock), GTAPv7 condensed closure, capFix
`swap dpsave(r)=del_tbalry(r)` for every NON-residual region.

---

## 0. Prerequisites (Windows machine)

- **RunGTAP** installed (the `.cmf` header points at `C:\runGTAP375\gtapv7`; adjust
  the `Auxiliary files = ...` line / `--gtapv7` flag if your install path differs).
- **GEMPACK** (ships with RunGTAP) so `gtapv7.exe` can solve.
- **Git** for Windows.
- **Python 3.11+** with `uv` (or the repo's `.venv`).

---

## 1. Bring the branch to Windows

```powershell
git clone https://github.com/mmc00/equilibria.git
cd equilibria
git fetch origin
git checkout gtap/validation-parity-pages
uv sync --dev
```

---

## 2. Produce `updated.har`

### 2a. The gtap7_* matrix datasets — one command

`scripts/gtap/run_gempack_matrix.py` generates a per-dataset `tm10.cmf`
(residual = last region = `ROW`, matching the Python gate's
`rr = list(sets.r)[-1]`), copies the dataset inputs into an isolated run folder,
solves with `gtapv7.exe`, and copies each converged `updated.har` into
`tests/fixtures/gtap7_gempack/`:

```powershell
uv run python scripts\gtap\run_gempack_matrix.py
# or, to (re)generate the .cmf without solving (e.g. on a non-Windows box):
uv run python scripts\gtap\run_gempack_matrix.py --no-solve
# custom solver path:
uv run python scripts\gtap\run_gempack_matrix.py --gtapv7 C:\path\to\gtapv7.exe
```

The generated `.cmf` are committed under `runs/gempack_matrix/<ds>/tm10.cmf`
(the experiment definitions); the solver artefacts and input copies in those
folders are git-ignored.

### 2b. nus333 & 9x10 — their committed `.cmf`

These two keep hand-tuned `.cmf` under `runs/<ds>_compare/rungtap/tm10.cmf`
(9x10 deliberately uses residual = **NAmerica**, not the last region — mirroring
`compare_9x10_rungtap.py`). Copy the inputs next to each `.cmf` and solve:

```powershell
copy src\equilibria\templates\reference\gtap\data\nus333\*.har  runs\nus333_compare\rungtap\
copy src\equilibria\templates\reference\gtap\data\nus333\default.prm runs\nus333_compare\rungtap\
cd runs\nus333_compare\rungtap ; gtapv7 -cmf tm10.cmf ; cd ..\..\..
copy runs\nus333_compare\rungtap\updated.har tests\fixtures\gtap7_gempack\updated_nus333_tm10.har
```

**9x10 needs one fix first.** Its reference `default.prm` was missing the
`EFLG` (`ENDOWFLAG(ENDW,ENDWT)`) header GTAPv7 requires, so the solve aborts with
*"no header EFLG"*. `scripts/gtap/reconstruct_9x10_eflg.py` rebuilds it
deterministically from 9x10's own `ENDM/ENDS/ENDF` mobility subsets (and asserts
the result is cell-identical to `gtap7_5x5`'s standard EFLG), appending it
byte-exact to the parm's header records:

```powershell
uv run python scripts\gtap\reconstruct_9x10_eflg.py          # patches the reference default.prm in place
uv run python scripts\gtap\reconstruct_9x10_eflg.py --check  # verify EFLG present (exit 1 if not)
```

The reference `default.prm` fix is already committed on this branch; run the
script only if you're regenerating from a clean checkout that predates it.

### 2c. Sanity-check the solve

Each run's `tm10.log` should report a max residual ratio ~1e-7 and *"completed
without error"* (Gragg 8/16/32, no closure errors). The welfare parity was
validated in `docs/findings/rungtap_welfare_parity_2026-05-15.md`.

**`gtap7_20x41` does not solve in GEMPACK:** it aborts with a `loge`-of-negative
arithmetic error in equation `E_u` (utility) for region `Caribbean` — the same
unsound base the coverage matrix already marks `blocked` (its GAMS reference
violates 37 of its own equations). No fixture is emitted for it.

---

## 3. The fixtures

Converged `updated.har` land in `tests/fixtures/gtap7_gempack/updated_<ds>_tm10.har`.

> These are the ONLY new binary artifacts. Do NOT overwrite
> `updated_synthetic.har` — that stays as the unit-test fixture.

---

## 4. Inspect the headers → complete `VAR_TO_HEADER`

This is the spec's open-risk step: the `VAR_TO_HEADER` map in
`scripts/gtap/gempack_reference.py` is seeded with ONE entry (`qfd → VDFB`). Do
NOT guess the rest — read the real header inventory and map only what genuinely
corresponds cell-by-cell.

```powershell
uv run python scripts\gtap\inspect_updated_har.py tests\fixtures\gtap7_gempack\updated_nus333_tm10.har
```

It prints every header: name, rank, shape, and set names (the updated.har carries
the standard GTAP value flows — `VDFB`, `VMFB`, `VDGB`, … — which are **values**,
so mapping to a Python Var level is a modelling judgment, not a rename). Cross-
reference against GTAP Standard 7's HAR conventions (the loading map in
`src/equilibria/templates/gtap/gtap_std7_mapping.py` is the starting point) and add
entries to `VAR_TO_HEADER`, e.g.:

```python
VAR_TO_HEADER = {
    "qfd": "VDFB",   # firm domestic purchases (already seeded)
    "qfm": "VMFB",   # firm imported purchases      ← only if the header exists
    # ... add ONLY headers that map to a comparable post-shock Var level.
}
```

Variables with no clean cell-by-cell header are **aggregate-only** — leave them
OUT; `gempack_levels` raises `KeyError` for them, which the gate treats as
"not comparable", never a fabricated match.

---

## 5. Add the `reference="gempack"` rows

In `scripts/gtap/coverage_matrix.py`, add rows for the datasets you produced,
pointing `ref` at the fixture filename and `reference="gempack"`:

```python
Row("nus333", "mcp", 1, ("base", "check", "shock"), None, "measured @ runtime",
    "local", "updated_nus333_tm10.har", stage_floors=_F(F, F, F), mode="pure",
    model="gtap7", reference="gempack"),
```

Set `stage_floors` conservatively BELOW the measured match% (step 6 tells you the
number) — same `TOL=1e-2` as GAMS; the linearized↔levels gap lives in the floor,
not a looser tolerance.

---

## 6. Run the gate → read the measured match → set the floor

```powershell
uv run pytest tests/templates/gtap/test_gtap7_gempack_parity.py -v -s
```

It now solves Python + measures cell-by-cell vs `updated.har` at 1% tolerance and
prints match% per stage. Set each row's `stage_floors` a few points below the
measured value, re-run until green.

Then regenerate the GEMPACK page + run the full mandatory sweep before pushing:

```powershell
uv run python scripts/gtap/gen_coverage_doc.py
uv run python scripts/gtap/run_parity_gates.py     # must be GREEN + writes stamp
```

---

## 7. Commit + push (back to the same branch / PR #34)

```powershell
git add tests/fixtures/gtap7_gempack/updated_*_tm10.har `
        runs/gempack_matrix/*/tm10.cmf `
        scripts/gtap/gempack_reference.py `
        scripts/gtap/coverage_matrix.py `
        docs/site/guide/
git commit -m "gtap(F5): real RunGTAP updated.har refs + VAR_TO_HEADER + gempack rows"
git push
```

---

## 8. Second pass — QUANTITY %-changes via the SL4 solution (recommended)

**Why:** `updated.har` holds VALUES only ($ SAM flows: `VDFB`, `VMSB`, …). A
value like `VDFB = pd·xd` folds the linearized↔levels structural gap into BOTH
price and quantity, so a value-vs-value cell match caps around ~66% at 1% tol
(measured on gtap7_3x3; **confirmed structural** — GAMS-levels vs GEMPACK gives
the SAME 66.67% on the identical cells, so it is not a Python defect, it is the
Gragg-linearized vs levels difference for a finite 10% shock). Comparing PURE
QUANTITIES (GEMPACK `qfd` vs Python `xd`) folds the gap in only once and is the
cleaner comparison. The quantity %-changes live in GEMPACK's **solution file**
`tm10.sl4`, which `updated.har` does not carry.

**No new `.cmf` directive needed** — the `Gragg` solve auto-writes `tm10.sl4`.
The runner now (a) keeps the `.sl4` instead of deleting it and (b) converts it to
a plain HAR with GEMPACK's `sltoht.exe` (same chain as
`runs/nus333_compare/rungtap/trajectory_runner.py`), committing it as a separate
fixture `tests/fixtures/gtap7_gempack/sl4dump_<ds>_tm10.har`:

```powershell
uv run python scripts\gtap\run_gempack_matrix.py --sltoht C:\GP\sltoht.exe
```

If `sltoht.exe` is elsewhere, pass its path; if omitted/not found, the SL4 export
is skipped and only the values `updated.har` is produced (backward compatible).

**Reading it in Python:** the SL4-as-HAR is readable by the existing `read_har`.
Its headers ARE numbered (`"0002"`, …), BUT `sltoht` preserves each variable's
GEMPACK name in the header `long_name` (`"qfd # demand for domestic commodity … #"`).
So map #1 below is **already solved by the file itself** — only the modeling map
(#2) remains hand-authored:

1. **GEMPACK q-name → SL4 numeric header id — SOLVED.** `gempack_reference.sl4_index()`
   parses `long_name` to build the name→id map (100% of headers, 256 unique vars
   on 3x3); `gempack_reference.sl4_levels(path, "qfd")` returns the %-change cells
   keyed by set elements. `inspect_updated_har.py` now prints the `long_name`
   column so numbered headers are legible at a glance. No hand-authored id table.
2. **GEMPACK q-name → Python Var** (the modeling map, still TODO): roughly
   `qfd`→`xf`/`xd` (firm domestic), `qxs`→`xw` (bilateral exports), `qpd`/`qpm`→`xaa`
   (private agent), `qgd`/`qgm`→`xaa` (gov agent), `qo`→activity output. **Author
   against the real inventory (now visible via the long_name column) — do NOT
   guess.** GTAPv7 splits domestic/imported (`qfd`/`qfm`) where Python may hold a
   composite; verify per variable.

Then the `against-GEMPACK` gate compares quantity-vs-quantity at 1% tol and the
per-page floor reflects the (smaller) structural residual.

### 8a. Post-sim LEVELS are available — but only via sltoht's TEXT modes

`sltoht`'s HAR/VAI export writes **only** the cumulative %-change column (that is
what `sl4dump_<ds>_tm10.har` holds; `SHL` and `SEP` do not change it — the log
still says *"Writing ONLY CUMULATIVE TOTALS"*). The pre-/post-simulation LEVELS
(solution headers `LEVB`/`LEVA`) are exposed **only in the TEXT modes** under
option `SHL`, where every variable prints four solution columns:

```
[ %-change , pre-sim level , post-sim level , change ]
qo, component 1:  -1.5210288   1579235.5   1555214.9   -24020.625
```

`scripts/gtap/export_sl4_levels.py` drives `sltoht -SHL -SIC`, parses the
POST-SIM LEVEL column and writes it back as `sl4levels_<ds>_tm10.har`:

```powershell
uv run python scripts\gtap\export_sl4_levels.py --sltoht C:\GP\sltoht.exe
```

Shapes / set-elements / numeric ids / long_names come from the matching
`sl4dump`, and every variable is CROSS-CHECKED (the text's %-change column must
equal the sl4dump values) before its level array is kept — 252/252 variables on
gtap7_3x3, 0 mismatches. **Component order is FORTRAN (column-major)**: verified
across all 252 variables (C-order agrees only for rank-1). Read either fixture
with `gempack_reference.sl4_levels(path, "qo")`.

### 8b. More Gragg steps do NOT close the gap (measured)

Re-solving gtap7_3x3 with `Subintervals = 10` instead of `1` (same
`Steps = 8 16 32`) changes `updated.har` by a **max relative 2.6e-6**. GEMPACK's
Gragg + Richardson extrapolation is ALREADY at levels accuracy, so raising the
step/subinterval count cannot move the ~66.67% value-vs-value match. That gap
lives in the value RECONSTRUCTION (`VDFB = pd·xd` folding the gap into price and
quantity), which is exactly why the quantity/levels path above is the fix — not
a finer solve.

**Status of the value path:** the `VDFB = pd·xd` value mapping is already verified
(66.67% @ 1% on gtap7_3x3, gap proven structural). It can ship as the FIRST
against-GEMPACK flow while the quantity SL4 path is built out — or the quantity
path can supersede it. That is the open design choice for the next session.

### 8c. Why %-change is the right comparison — Horridge & Pearson (G-214)

The design rests on GEMPACK's own authors. Horridge & Pearson, *"Solution Software
for CGE Modeling"* (COPS General Paper **G-214**, 2011):

- **§4.1 — GAMS = LEVELS strategy**: solve for the level `Y` directly; a solution is
  declared when the merit `FᵀF < 1e-6`. Python is the same (a levels MCP).
- **§4.2 — GEMPACK = CHANGE strategy**: start from a solution and impose the shock
  as a *linearized* system, made accurate by multi-step **Euler + Richardson
  extrapolation**; **§4.2.1**: "most of the change variables are expressed as
  **percentage changes**." So GEMPACK's native output is %-change, not a level.
- **§6.5 "All three give the same results"**: the same model in GEMPACK/GAMS/MPSGE
  yields the same numbers — mirroring our Python≡GAMS finding.
- **§6 (p.28)**: the linearized decomposition (−1.824) is *"not exactly equal"* to
  the levels result (−2.214) because *"the linearized equations … are not satisfied
  exactly by the accurate results."* **That is precisely our per-cell pp-residual**,
  named by GEMPACK's own author — structural, not a defect.

Consequence: comparing GEMPACK↔Python in **%-change / percentage points** (the form
both share) is the correct apples-to-apples. A "levels-vs-levels" GEMPACK comparison
does not exist as a distinct thing: GEMPACK levels are $millions, Python is a
benchmark-normalized index, and normalizing GEMPACK to its benchmark gives exactly
`1 + %change` (verified identical to 8 decimals). GAMS levels ARE directly
comparable to Python only because both share the same levels normalization.

---

## What this closes

- The "against GEMPACK" page fills with **real cell-by-cell** data for 7 datasets.
- The gate stops SKIPping for those rows (measures at 1% tol, floor-gated).
- One `run_gempack_matrix.py` invocation reproduces every solvable dataset
  (add `--sltoht` for the quantity SL4 export).

## Fidelity reminders

- Same 1% tolerance for GAMS and GEMPACK — the gap goes in the per-page floor.
- `VAR_TO_HEADER` grows ONLY from the real header inventory (step 4), never guesses.
- No fabricated cells: aggregate-only vars raise `KeyError` and are excluded.
- 9x10's EFLG is reconstructed from its OWN mobility subsets and cross-checked
  against a sibling dataset — a deterministic derivation, not invented data.
- `gtap7_20x41` stays blocked on both the GAMS and GEMPACK sides — not forced.
