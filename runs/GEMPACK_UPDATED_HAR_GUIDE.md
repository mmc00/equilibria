# Generating the RunGTAP `updated.har` refs on Windows (F5)

This guide takes the `gtap/validation-parity-pages` branch to a Windows machine
with RunGTAP, produces the real post-shock `updated.har` for each dataset, drops
it where the parity gate expects it, and completes the `VAR_TO_HEADER` map so the
"against GEMPACK" cell-by-cell page fills with real data.

The Python side (reader → `gempack_levels` → SKIP-if-missing gate) already exists
and is unit-tested against a synthetic fixture (PR #34). This guide is the missing
Windows half.

---

## 0. Prerequisites (Windows machine)

- **RunGTAP** installed (the `.cmf` header points at `C:\runGTAP375\gtapv7`; adjust
  `Auxiliary files = ...` if your install path differs).
- **GEMPACK** (ships with RunGTAP) so `gtapv7.exe` / the `.cmf` can solve.
- **Git** for Windows.
- **Python 3.11+** with `uv` (or the repo's `.venv`) to run the header-inspector
  and the parity gate.

---

## 1. Bring the branch to Windows

```powershell
git clone https://github.com/mmc00/equilibria.git
cd equilibria
git fetch origin
git checkout gtap/validation-parity-pages
```

(That branch is PR #34 — it has the GEMPACK reader/mapper/gate but no real
`updated.har` yet.)

Set up the Python env (once):

```powershell
uv sync --dev
```

---

## 2. Produce `updated.har` — one dataset at a time

The `.cmf` files ALREADY export the post-shock database — the line
`Updated file GTAPDATA = updated.har ;` in each `.cmf` writes it automatically.
**No `.cmf` edit is needed** to get `updated.har`.

The two ready `.cmf` files and their datasets:

| dataset | `.cmf` | input `.har` (identical to equilibria's) |
|---|---|---|
| `nus333` | `runs/nus333_compare/rungtap/tm10.cmf` | `src/equilibria/templates/reference/gtap/data/nus333/` |
| `9x10`   | `runs/9x10_compare/rungtap/tm10.cmf`   | `src/equilibria/templates/reference/gtap/data/9x10/` |

### 2a. Copy the input `.har` next to the `.cmf`

Each `.cmf` reads `sets.har` / `basedata.har` / `default.prm` from its own folder.
Copy the dataset's inputs there (RunGTAP resolves relative to the `.cmf`):

```powershell
# nus333
copy src\equilibria\templates\reference\gtap\data\nus333\sets.har     runs\nus333_compare\rungtap\
copy src\equilibria\templates\reference\gtap\data\nus333\basedata.har  runs\nus333_compare\rungtap\
copy src\equilibria\templates\reference\gtap\data\nus333\default.prm   runs\nus333_compare\rungtap\
```

(Confirm the input filenames match the dataset folder; if the parm file is named
differently there, rename the copy to `default.prm` to match the `.cmf`.)

### 2b. Run the `.cmf`

Open **RunGTAP** → load `runs\nus333_compare\rungtap\tm10.cmf` → **Solve**.
Or from the command line (GEMPACK):

```powershell
cd runs\nus333_compare\rungtap
gtapv7 -cmf tm10.cmf
```

On success the folder now contains `updated.har` (plus `summary.har`,
`decomp.har`, `volume.har`, `<name>.log`). The one we need is **`updated.har`**.

### 2c. Sanity-check the solve

Open `tm10.log` — confirm it converged (Gragg 8/16/32 steps, no closure errors).
This mirrors the `capFix` closure equilibria uses; the welfare parity was already
validated in `docs/findings/rungtap_welfare_parity_2026-05-15.md`.

Repeat 2a–2c for `9x10` using `runs\9x10_compare\rungtap\tm10.cmf`.

---

## 3. Drop the `updated.har` where the gate looks

The gate reads `tests/fixtures/gtap7_gempack/<row.ref>` (see
`tests/templates/gtap/test_gtap7_gempack_parity.py`). Use a per-dataset name so
both datasets coexist:

```powershell
copy runs\nus333_compare\rungtap\updated.har  tests\fixtures\gtap7_gempack\updated_nus333_tm10.har
copy runs\9x10_compare\rungtap\updated.har    tests\fixtures\gtap7_gempack\updated_9x10_tm10.har
```

> These are the ONLY new binary artifacts. Do NOT overwrite
> `updated_synthetic.har` — that stays as the unit-test fixture.

---

## 4. Inspect the headers → complete `VAR_TO_HEADER`

This is the spec's open-risk step: the `VAR_TO_HEADER` map in
`scripts/gtap/gempack_reference.py` is seeded with ONE entry (`qfd → VDFB`). Do
NOT guess the rest — read the real header inventory and map only what genuinely
corresponds cell-by-cell.

Run the inspector helper (added by this branch):

```powershell
uv run python scripts\gtap\inspect_updated_har.py tests\fixtures\gtap7_gempack\updated_nus333_tm10.har
```

It prints every header: name, rank, shape, and set names. Cross-reference against
GTAP Standard 7's HAR conventions (the loading map in
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
        scripts/gtap/gempack_reference.py `
        scripts/gtap/coverage_matrix.py `
        docs/site/guide/
git commit -m "gtap(F5): real RunGTAP updated.har refs + VAR_TO_HEADER + gempack rows"
git push
```

---

## What this closes

- The "against GEMPACK" page fills with **real cell-by-cell** data for nus333 + 9x10.
- The gate stops SKIPping for those rows (measures at 1% tol, floor-gated).
- Remaining datasets follow the same recipe; each is one more `.cmf` run + one row.

## Fidelity reminders

- Same 1% tolerance for GAMS and GEMPACK — the gap goes in the per-page floor.
- `VAR_TO_HEADER` grows ONLY from the real header inventory (step 4), never guesses.
- No fabricated cells: aggregate-only vars raise `KeyError` and are excluded.
