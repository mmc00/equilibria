# GTAP7 Coverage Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one declarative table (`coverage_matrix.py`) the single source of truth for GTAP7 parity coverage — driving both parity gates' parameters/thresholds and generating the documentation table, with a CI sync test preventing drift.

**Architecture:** A new `scripts/gtap/coverage_matrix.py` holds a frozen-dataclass list of coverage rows + query helpers + import-time schema validation. A generator `gen_coverage_doc.py` renders `docs/gtap7_coverage_matrix.md` from it. The two existing parity tests are re-wired to read rows from the matrix (the altertax test asserts per-row `gap_min` instead of a global 98%). A new golden-file sync test fails CI if the committed doc differs from the regenerated one.

**Tech Stack:** Python 3.11+, pytest, dataclasses. No new dependencies. Use `uv run python` / `uv run pytest` (never bare `python`).

## Global Constraints

- Do NOT modify model equations, parameters, or builders. `scripts/gtap/_parity_datasets.py` (the *builder* registry) is untouched.
- `gap_min` is a CONSERVATIVE floor below the measured value (margin ~1pt), never equal to a measured 100%. `gap_note` is the measured snapshot for humans.
- `gap_min is None` exactly for the `.nl`-only `gtap7_*` single-period rows; non-None for measured-% rows (nus333, 9x10, all altertax rows).
- `ci_status` ∈ {`"ci"`, `"local"`, `"blocked"`}. `ifsub is None` iff `kind == "gtap"`.
- Tests SKIP (never hard-fail) on missing fixture / missing HAR / missing PATH solver.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Branch: `debug-gtap7-check-income` (already merged to main as PR #20; this is follow-up work — commit directly here, do NOT create a new branch).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/gtap/coverage_matrix.py` (new) | The data: `Row` dataclass, `ROWS` list, query helpers, import-time `_validate()`. |
| `scripts/gtap/gen_coverage_doc.py` (new) | Render `docs/gtap7_coverage_matrix.md` from `ROWS`. CLI + importable `render()`. |
| `docs/gtap7_coverage_matrix.md` (new, generated) | Golden file, committed, visible on GitHub/RTD. |
| `tests/templates/gtap/test_coverage_matrix.py` (new) | Schema self-test + doc-sync golden test. |
| `tests/templates/gtap/test_altertax_multiperiod_parity.py` (modify) | Read `kind=="altertax"` rows; assert per-row `gap_min`; skip `blocked`. |
| `tests/templates/gtap/test_gtap7_nl_parity.py` (modify) | Parametrize from `kind=="gtap"`, `ci_status=="ci"` rows. |
| `CLAUDE.md`, `docs/site/benchmarks.md` (modify) | Link to the generated matrix doc. |

---

### Task 1: The coverage matrix source

**Files:**
- Create: `scripts/gtap/coverage_matrix.py`
- Test: `tests/templates/gtap/test_coverage_matrix.py`

**Interfaces:**
- Produces:
  - `Row` — frozen dataclass with fields `dataset: str`, `kind: str`, `ifsub: int | None`, `phases: tuple[str, ...]`, `gap_min: float | None`, `gap_note: str`, `ci_status: str`, `ref: str`.
  - `ROWS: list[Row]` — the full matrix (content below).
  - `nl_rows() -> list[Row]` — rows where `kind == "gtap"`.
  - `altertax_rows() -> list[Row]` — rows where `kind == "altertax"`.
  - `CI_STATUSES = {"ci", "local", "blocked"}`.

- [ ] **Step 1: Write the failing schema self-test**

Create `tests/templates/gtap/test_coverage_matrix.py`:

```python
"""Tests for the GTAP7 coverage matrix (single source of truth)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))


def test_matrix_schema_invariants():
    from coverage_matrix import ROWS, CI_STATUSES
    assert ROWS, "matrix must not be empty"
    for r in ROWS:
        # ifsub is None iff kind == "gtap"
        assert (r.ifsub is None) == (r.kind == "gtap"), r
        assert r.kind in ("gtap", "altertax"), r
        assert r.ci_status in CI_STATUSES, r
        assert r.phases, r
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            # gap_min is None exactly for the .nl-only gtap7_* single-period rows
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, r
            # a non-None gap_min must be a sane floor (never an exact 100)
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, r


def test_matrix_helpers_partition():
    from coverage_matrix import ROWS, nl_rows, altertax_rows
    assert set(nl_rows()) | set(altertax_rows()) == set(ROWS)
    assert all(r.kind == "gtap" for r in nl_rows())
    assert all(r.kind == "altertax" for r in altertax_rows())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/templates/gtap/test_coverage_matrix.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'coverage_matrix'`.

- [ ] **Step 3: Create the matrix source**

Create `scripts/gtap/coverage_matrix.py`:

```python
"""GTAP7 parity coverage matrix — the single source of truth.

One declarative table (`ROWS`) describes parity coverage across
dataset x kind (gtap|altertax) x ifSUB x phase (base/check/shock). It drives:
  * test_gtap7_nl_parity.py        (which gtap rows/phases run in CI)
  * test_altertax_multiperiod_parity.py  (per-row gap_min contract)
  * gen_coverage_doc.py            (the generated docs/gtap7_coverage_matrix.md)

`gap_min` is a CONSERVATIVE floor the tests assert (margin below the measured
value); `gap_note` is the measured snapshot for humans. `gap_min is None` only
for the .nl-only gtap7_* single-period rows (their contract is "0 coefficient
diffs", not a percentage). `ci_status` records honestly what runs where:
  * "ci"      runs on ubuntu CI without a solver (the .nl gate rows)
  * "local"   runs only locally (needs PATH+GAMS) — the altertax solver rows
  * "blocked" cannot be verified (reference unsound, e.g. 20x41 NEOS Infeasible)
"""
from __future__ import annotations

from dataclasses import dataclass

CI_STATUSES = {"ci", "local", "blocked"}


@dataclass(frozen=True)
class Row:
    dataset: str
    kind: str               # "gtap" | "altertax"
    ifsub: int | None       # None for gtap; 0 or 1 for altertax
    phases: tuple[str, ...]  # phases with coverage
    gap_min: float | None   # contract floor; None for .nl-only gtap7_* rows
    gap_note: str           # measured snapshot, e.g. "100%", "~99%"
    ci_status: str          # "ci" | "local" | "blocked"
    ref: str                # provenance


ROWS: list[Row] = [
    # --- single-period .nl gate (CI, no solver) ---
    Row("nus333", "gtap", None, ("base", "shock"), 99.5, "100% (NEOS+GAMS)", "ci", "nus333 NEOS"),
    Row("9x10", "gtap", None, ("base", "shock"), 99.5, "100% (NEOS)", "ci", "job 18737509"),
    Row("gtap7_3x3", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_5x5", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_10x7", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_15x10", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    Row("gtap7_3x4", "gtap", None, ("base", "shock"), None, "0 diffs .nl", "ci", "gams_base/shock.nl"),
    # --- altertax multi-period (solver gate, local-only), both ifSUB modes ---
    Row("gtap7_3x3", "altertax", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x3", "altertax", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_5x5", "altertax", 0, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_5x5", "altertax", 1, ("base", "check", "shock"), 99.5, "100%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_10x7", "altertax", 0, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_10x7", "altertax", 1, ("base", "check", "shock"), 98.0, "~99%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_15x10", "altertax", 0, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_15x10", "altertax", 1, ("base", "check", "shock"), 99.0, "99.30%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_3x4", "altertax", 0, ("base", "check", "shock"), 99.0, "99.61%", "local", "out_altertax_ifsub0.gdx"),
    Row("gtap7_3x4", "altertax", 1, ("base", "check", "shock"), 99.0, "99.56%", "local", "out_altertax_ifsub1.gdx"),
    Row("gtap7_20x41", "altertax", 0, ("base",), None, "blocked", "blocked", "NEOS ref Infeasible"),
]


def nl_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "gtap"]


def altertax_rows() -> list[Row]:
    return [r for r in ROWS if r.kind == "altertax"]


def _validate() -> None:
    """Import-time schema invariants — fail fast on a malformed matrix."""
    for r in ROWS:
        assert (r.ifsub is None) == (r.kind == "gtap"), f"ifsub/kind mismatch: {r}"
        assert r.kind in ("gtap", "altertax"), f"bad kind: {r}"
        assert r.ci_status in CI_STATUSES, f"bad ci_status: {r}"
        assert r.phases, f"empty phases: {r}"
        # gap_min invariants do NOT apply to blocked rows (never asserted).
        if r.ci_status != "blocked":
            nl_only = r.kind == "gtap" and r.dataset.startswith("gtap7_")
            assert (r.gap_min is None) == nl_only, f"gap_min/nl-only mismatch: {r}"
            if r.gap_min is not None:
                assert 0.0 < r.gap_min < 100.0, f"gap_min must be a floor <100: {r}"


_validate()
```

**Blocked-row rule (already applied above):** the `gtap7_20x41` blocked row has
`kind="altertax"` and `gap_min=None`. The `gap_min`/`nl_only` invariant does NOT
apply to blocked rows — they are guarded by `if r.ci_status != "blocked":` in both
`_validate()` (shown in Step 3) and the test (shown in Step 1). A blocked row's
`gap_min` is never read by any test (the altertax test skips blocked rows first), so
`None` is correct and intentional.

This rule is ALREADY reflected in Step 1's test (the gap_min block is wrapped in
`if r.ci_status != "blocked":`) and in Step 3's `_validate()` (same guard). No extra
action needed — this note just explains why the blocked row passes validation.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/templates/gtap/test_coverage_matrix.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/gtap/coverage_matrix.py tests/templates/gtap/test_coverage_matrix.py
git commit -m "feat(gtap): coverage matrix single source of truth (rows + schema self-test)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The doc generator + golden file

**Files:**
- Create: `scripts/gtap/gen_coverage_doc.py`
- Create: `docs/gtap7_coverage_matrix.md` (generated output, committed)

**Interfaces:**
- Consumes: `coverage_matrix.ROWS`, `nl_rows()`, `altertax_rows()` (Task 1).
- Produces:
  - `render() -> str` — returns the full Markdown document as a string (deterministic).
  - `DOC_PATH: Path` — `ROOT / "docs/gtap7_coverage_matrix.md"`.
  - CLI: running the module writes `render()` to `DOC_PATH`.

- [ ] **Step 1: Write the generator**

Create `scripts/gtap/gen_coverage_doc.py`:

```python
"""Generate docs/gtap7_coverage_matrix.md from coverage_matrix.ROWS.

Run:  uv run python scripts/gtap/gen_coverage_doc.py
The output is a committed golden file; test_coverage_doc_sync enforces that the
committed file equals render() (CI fails on drift).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))

from coverage_matrix import nl_rows, altertax_rows  # noqa: E402

DOC_PATH = ROOT / "docs/gtap7_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->"
)


def _fmt_gap_min(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _row_cells(r) -> list[str]:
    ifsub = "—" if r.ifsub is None else str(r.ifsub)
    phases = ",".join(r.phases)
    return [
        r.dataset, r.kind, ifsub, phases,
        _fmt_gap_min(r.gap_min), r.gap_note, r.ci_status, r.ref,
    ]


_HEADER = ["dataset", "kind", "ifsub", "phases", "gap_min", "gap_note", "ci_status", "ref"]


def _table(rows) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |",
             "|" + "|".join("---" for _ in _HEADER) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_row_cells(r)) + " |")
    return "\n".join(lines)


def render() -> str:
    parts = [
        "# GTAP 7 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        "`gap_min` is the conservative floor the tests assert; `gap_note` is the "
        "measured snapshot. `ci_status`: `ci` runs on ubuntu without a solver, "
        "`local` needs PATH+GAMS (run by hand), `blocked` has an unsound reference.",
        "",
        "## Single-period (`.nl` coefficient gate, CI, no solver)",
        "",
        _table(nl_rows()),
        "",
        "## Altertax multi-period (solver gate, local-only)",
        "",
        _table(altertax_rows()),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the doc**

Run: `uv run python scripts/gtap/gen_coverage_doc.py`
Expected: prints `wrote .../docs/gtap7_coverage_matrix.md`.

- [ ] **Step 3: Eyeball the output**

Run: `cat docs/gtap7_coverage_matrix.md`
Expected: two Markdown tables, the banner near the top, `nus333` first row of the
single-period table, `gtap7_20x41` last row of the altertax table.

- [ ] **Step 4: Commit**

```bash
git add scripts/gtap/gen_coverage_doc.py docs/gtap7_coverage_matrix.md
git commit -m "feat(gtap): generate coverage matrix doc from the source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: The doc-sync golden test

**Files:**
- Modify: `tests/templates/gtap/test_coverage_matrix.py` (append the sync test)

**Interfaces:**
- Consumes: `gen_coverage_doc.render()`, `gen_coverage_doc.DOC_PATH` (Task 2).

- [ ] **Step 1: Write the failing sync test**

Append to `tests/templates/gtap/test_coverage_matrix.py`:

```python
def test_coverage_doc_in_sync():
    """The committed doc must equal render() — regenerate + commit on drift."""
    import gen_coverage_doc
    committed = gen_coverage_doc.DOC_PATH.read_text(encoding="utf-8")
    assert committed == gen_coverage_doc.render(), (
        "docs/gtap7_coverage_matrix.md is stale — run "
        "`uv run python scripts/gtap/gen_coverage_doc.py` and commit."
    )
```

- [ ] **Step 2: Run it — it should PASS (doc was just generated in Task 2)**

Run: `uv run pytest tests/templates/gtap/test_coverage_matrix.py::test_coverage_doc_in_sync -q`
Expected: PASS.

- [ ] **Step 3: Prove it actually catches drift**

Temporarily append a line to the doc and confirm the test fails:

```bash
echo "stale" >> docs/gtap7_coverage_matrix.md
uv run pytest tests/templates/gtap/test_coverage_matrix.py::test_coverage_doc_in_sync -q || echo "FAILED AS EXPECTED"
git checkout docs/gtap7_coverage_matrix.md
```

Expected: the run FAILS (prints the stale message), then `git checkout` restores it.
Re-run the test: `uv run pytest tests/templates/gtap/test_coverage_matrix.py -q` → all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/templates/gtap/test_coverage_matrix.py
git commit -m "test(gtap): golden-file sync test for the coverage matrix doc

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Re-wire the altertax test to per-row gap_min

**Files:**
- Modify: `tests/templates/gtap/test_altertax_multiperiod_parity.py`

**Interfaces:**
- Consumes: `coverage_matrix.altertax_rows()` (Task 1). Each row gives `dataset`, `ifsub` (0/1), `gap_min`, `ci_status`.

Current state (verbatim): the module has `MATCH_THRESHOLD = 98.0` (line ~37),
parametrizes `if_sub ∈ {False, True}` and `dataset ∈ DATASETS` (a hardcoded list
`["gtap7_3x3","gtap7_5x5","gtap7_10x7"]`), and the body asserts
`match_pct >= MATCH_THRESHOLD` (line ~187). `sys.path` already includes
`scripts/gtap` (line ~34).

- [ ] **Step 1: Build parametrization from the matrix**

Replace the hardcoded `DATASETS` / `MATCH_THRESHOLD` and the two
`@pytest.mark.parametrize` decorators. Add near the top (after the existing
`sys.path.insert(0, str(ROOT / "scripts/gtap"))`):

```python
from coverage_matrix import altertax_rows  # noqa: E402

# (dataset, ifsub_int, gap_min, ci_status) per altertax matrix row
_ALTERTAX_CASES = [
    (r.dataset, r.ifsub, r.gap_min, r.ci_status) for r in altertax_rows()
]
```

- [ ] **Step 2: Replace the test signature + assertion**

Replace the decorated test function with a single matrix-driven parametrization:

```python
@pytest.mark.parametrize(
    "dataset,ifsub,gap_min,ci_status",
    _ALTERTAX_CASES,
    ids=[f"{r.dataset}-ifsub{r.ifsub}" for r in altertax_rows()],
)
def test_altertax_multiperiod_parity(dataset, ifsub, gap_min, ci_status):
    if ci_status == "blocked":
        pytest.skip(f"blocked reference: {dataset}")
    if not _has_path_solver():
        pytest.skip("path_capi_python (PATH solver) not available")
    if_sub = bool(ifsub)
    ref = _fixture_gdx(dataset, if_sub)
    if not ref.exists():
        pytest.skip(f"fixture GDX missing: {ref}")
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")

    codes, match_pct, tot = _solve_and_match(dataset, if_sub)

    mode = "ifSUB=1" if if_sub else "ifSUB=0"
    assert all(c == 1 for c in codes.values()), (
        f"[{dataset}/{mode}] not all periods converged: {codes}"
    )
    assert tot > 0, f"[{dataset}/{mode}] no comparable cells found"
    assert match_pct >= gap_min, (
        f"[{dataset}/{mode}] real-cell match {match_pct:.2f}% "
        f"< gap_min {gap_min}% (over {tot} cells)"
    )
```

Delete the now-unused `MATCH_THRESHOLD = 98.0` and the old hardcoded `DATASETS`
list. Keep `SKIP`, `RF`, `ALIAS`, `_has_path_solver`, `_fixture_gdx`,
`_solve_and_match` exactly as they are.

- [ ] **Step 3: Run the altertax gate (local, has PATH)**

Run: `uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -q`
Expected on this machine (PATH present): the rows with committed GDX fixtures
(3x3/5x5/10x7 × ifsub0/1 = 6 cases) PASS; 15x10/3x4 rows SKIP (no GDX fixture in
git); the 20x41 row SKIPs (blocked). So: `6 passed, 5 skipped` (4 missing-fixture +
1 blocked) — confirm the 6 that run all pass against their per-row `gap_min`.

- [ ] **Step 4: Confirm `-m gams` still collects nothing**

Run: `uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -m gams -q`
Expected: no tests collected (the module is intentionally unmarked / local-only).

- [ ] **Step 5: Commit**

```bash
git add tests/templates/gtap/test_altertax_multiperiod_parity.py
git commit -m "test(gtap-altertax): drive parametrization + gap_min from coverage matrix

Replaces the global 98% threshold with the per-row gap_min contract; blocked
rows (20x41) skip with their reason; 15x10/3x4 skip on missing GDX fixture.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Re-wire the .nl gate + link the doc

**Files:**
- Modify: `tests/templates/gtap/test_gtap7_nl_parity.py`
- Modify: `CLAUDE.md`
- Modify: `docs/site/benchmarks.md`

**Interfaces:**
- Consumes: `coverage_matrix.nl_rows()` (Task 1).

Current state (verbatim): `_available_datasets()` scans `FIXTURES_DIR` for dirs
having `gams_base.nl` + `gams_shock.nl` + `basedata.har`, returns names, and the
test is `@pytest.mark.parametrize("dataset", DATASETS)`. The body already builds
`phases = ["base","shock"]` and inserts `"check"` when `gams_check.nl` exists.

- [ ] **Step 1: Parametrize from the matrix (CI rows only), keep disk-fixture skip**

Replace `_available_datasets()` usage. The matrix is the authoritative list of CI
`.nl` datasets; the existing on-disk skip stays as a safety net. Change the
`DATASETS` definition and the parametrize to:

```python
from coverage_matrix import nl_rows  # noqa: E402

# Datasets whose .nl gate runs in CI, per the coverage matrix.
DATASETS = [r.dataset for r in nl_rows() if r.ci_status == "ci"]
```

Keep `_available_datasets()` defined (it documents the on-disk contract) but no
longer feed it into `DATASETS`. Inside the test body, keep the existing per-phase
fixture-existence logic unchanged, and add an early skip if the dataset's fixtures
are absent on this checkout:

```python
    fixture_dir = FIXTURES_DIR / dataset
    if not (fixture_dir / "gams_base.nl").exists():
        pytest.skip(f"no .nl fixtures for {dataset}")
```

(Place this right after `fixture_dir = FIXTURES_DIR / dataset`.)

- [ ] **Step 2: Run the .nl gate (CI path, no solver)**

Run: `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q`
Expected: the 5 `gtap7_*` datasets that have committed fixtures PASS (`5 passed`);
nus333/9x10 are `kind="gtap"` in the matrix but are NOT `.nl`-fixture datasets under
`tests/fixtures/gtap7/` — confirm they are excluded from this gate. (nus333/9x10
parity is covered by their own dedicated tests, not this `.nl` gate.)

IMPORTANT — reconcile with the matrix: the matrix lists nus333 and 9x10 as
`kind="gtap", ci_status="ci"`, but this `.nl` gate only has fixtures for the
`gtap7_*` datasets. To avoid the gate trying to build nus333/9x10 here, filter
`DATASETS` to dataset names starting with `"gtap7_"` as well:

```python
DATASETS = [r.dataset for r in nl_rows()
            if r.ci_status == "ci" and r.dataset.startswith("gtap7_")]
```

This keeps nus333/9x10 in the matrix/doc (they ARE covered, elsewhere) without
forcing this particular `.nl` gate to own them. Expected: `5 passed`.

- [ ] **Step 3: Link the generated doc from CLAUDE.md**

In `CLAUDE.md`, under the "Estado actual" / status section, add a one-line pointer
(do not duplicate the table — the generated doc is the source):

```markdown
> **Matriz de cobertura (fuente única):** ver [`docs/gtap7_coverage_matrix.md`](docs/gtap7_coverage_matrix.md), generada de `scripts/gtap/coverage_matrix.py`. NO editar a mano (CI `test_coverage_doc_in_sync` lo verifica).
```

- [ ] **Step 4: Link the generated doc from benchmarks.md**

In `docs/site/benchmarks.md`, add a short section near the top:

```markdown
## Coverage matrix

The authoritative parity-coverage matrix (dataset × kind × ifSUB × phase, with
per-row gap thresholds and CI status) is generated from
`scripts/gtap/coverage_matrix.py`: see
[GTAP 7 Parity Coverage Matrix](../gtap7_coverage_matrix.md).
```

(Adjust the relative path if `docs/site/benchmarks.md` resolves links differently —
verify by checking an existing relative link in that file first.)

- [ ] **Step 5: Run the full coverage-related suite once**

Run:
```bash
uv run pytest tests/templates/gtap/test_coverage_matrix.py tests/templates/gtap/test_gtap7_nl_parity.py -q
```
Expected: all PASS (matrix self-test + sync + 5 `.nl` datasets).

- [ ] **Step 6: Commit**

```bash
git add tests/templates/gtap/test_gtap7_nl_parity.py CLAUDE.md docs/site/benchmarks.md
git commit -m "test(gtap-nl): drive .nl gate datasets from coverage matrix + link doc

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

Run the whole coverage + parity surface and confirm green:

```bash
uv run pytest tests/templates/gtap/test_coverage_matrix.py \
              tests/templates/gtap/test_gtap7_nl_parity.py \
              tests/templates/gtap/test_altertax_multiperiod_parity.py -q
```
Expected (this machine, PATH present): matrix tests PASS, `.nl` gate `5 passed`,
altertax `6 passed, 5 skipped`. On ubuntu CI (no PATH): matrix + `.nl` gate run and
pass; altertax module self-skips entirely (no solver) — and is not collected by the
unit-tests file list anyway.

Confirm the doc is in sync and committed:
```bash
uv run python scripts/gtap/gen_coverage_doc.py && git diff --quiet docs/gtap7_coverage_matrix.md && echo "DOC IN SYNC"
```
