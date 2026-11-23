# Uniform shock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a shock say "x % on every cell of an instrument that can take it" (`uniform(x)`), and use it to solve Python at 3, 1, 0.3 and 0.1 % in the linearization study.

**Architecture:** `uniform(pct)` is a small frozen marker that replaces the cell map of one instrument in a shock. `apply_shock` (Pyomo and family) expands it on the built model with the same filter the default tariff uses, then runs its normal all-or-nothing path. Code that reads a shock cell by cell outside `apply_shock` refuses the marker with a clear error.

**Tech Stack:** Python, Pyomo, the family arrays backend, pytest.

**Spec:** `specs/templates/gtap/changes/uniform-shock/proposal.md` and its delta `specs/specs.md`.

## Global Constraints

- Repo text, names, comments and commits in English; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No shock still means the +10 % default; the default code path is not rewritten.
- `uniform` uses the instrument's GEMPACK kind (`gempack_kind`); for `imptx` that is the power 1+t.
- Oracle tests are committed before the code they validate (separate commit).
- A new test file needs `uv run python scripts/dev/plan_test_groups.py --refresh-files`.
- Numeric comparisons at 1e-12 relative, never bit for bit across machines.
- Local suite at most `-n 2`.

## Review Focus

- A shock mixing a `uniform` instrument with explicit cells of another (`{"imptx": uniform(3), "aft": {...}}`): both written. Test in Task 2.
- `uniform(-100)` or less: the existing `> -100 %` `ValueError`, nothing fixed. Test in Task 2.
- `uniform` reaching a cell-by-cell reader (`family/shock_session._write_levels`, `family/altertax/run.py:170`, `family/altertax/path.py:93-96`): `TypeError` naming `uniform`, not an `AttributeError` from `.items()`. Test in Task 3.
- `uniform(0)`: every cell equals its check, so `check_shock_value` raises as for an explicit 0 %. Test in Task 2.
- A non-number pct (`uniform("3")`): `TypeError` at construction. Test in Task 2.

---

### Task 1: Oracle (tests only, committed first)

**Files:**
- Create: `tests/templates/gtap/test_uniform_shock.py`
- Modify: `tests/family/test_shock_cell.py` (append one test)
- Modify: `tests/templates/gtap/test_gtap7_gempack_parity.py` (append one test)

**Interfaces:**
- Produces (expected by the tests, built in Tasks 2-4): `equilibria.templates.gtap.uniform`, `equilibria.templates.gtap.shocks.Uniform` (frozen dataclass, field `pct: float`), `_solve_shock(..., tariff_pct: float | None = None)`.

- [ ] **Step 1: Pyomo tests, no solver.** In `test_uniform_shock.py`, reuse the `built` / `m` fixture pattern of `tests/templates/gtap/test_shocks.py` (gtap7_3x3, ShockBlock restored after each test).
  - `test_uniform_10_equals_default_tariff`: `apply_shock(m, {"imptx": uniform(10)})` returns the same dict (same cells, values equal to 1e-12 rel) as `apply_default_tariff(m, gtap_mode=True)` on the restored model.
  - `test_uniform_3_is_power_times_1_03`: every returned cell value equals `(1 + chk) * 1.03 - 1` (chk = its `check` value), and the cell set equals the default tariff's.
  - `test_uniform_no_cell_raises`: with `monkeypatch.setattr(shocks, "_can_take_shock", lambda *a: False)`, `pytest.raises(ValueError, match="uniform")`, and no `imptx` shock cell changed.
  - `test_uniform_mixed_with_cells`: `{"imptx": uniform(3), "aft": {cell: 10.0}}` writes both (cell from `_cell(m, "aft")` as in `test_shocks.py`).
  - `test_uniform_minus_100_raises` and `test_uniform_zero_raises`: `ValueError`, nothing fixed.
  - `test_uniform_rejects_non_number`: `pytest.raises(TypeError)` on `uniform("3")`.
- [ ] **Step 2: Family test.** In `test_shock_cell.py`, `test_uniform_10_equals_default_tariff_arrays(case)`: on two copies of the case's arrays (as `test_rejections_raise_like_pyomo` gets one), `family.shock.apply_shock(mp_b, be, {"imptx": uniform(10)})` equals `apply_default_tariff(mp_a, be)` (dicts to 1e-12 rel; `imptx` value and fixed arrays equal).
- [ ] **Step 3: Solve test.** In `test_gtap7_gempack_parity.py`, `@pytest.mark.needs_path test_uniform_10_solves_like_default`: `_solve_shock("gtap7_3x3", 1, "capFlex", tariff_pct=10.0)` and `_solve_shock("gtap7_3x3", 1, "capFlex")` both code 1; every Var value of the shock period equal to 1e-12 rel (atol 1e-14).
- [ ] **Step 4: Run, expect failures.** `uv run pytest tests/templates/gtap/test_uniform_shock.py tests/family/test_shock_cell.py -k uniform -v`: ImportError on `uniform`. The solve test: TypeError on `tariff_pct`.
- [ ] **Step 5: Refresh groups and commit.** `uv run python scripts/dev/plan_test_groups.py --refresh-files`; commit `test(gtap): oracle for the uniform shock (MAR-227)`.

### Task 2: `uniform` in the Pyomo `apply_shock`

**Files:**
- Modify: `src/equilibria/templates/gtap/shocks.py`
- Modify: `src/equilibria/templates/gtap/__init__.py` (export `uniform`, next to `apply_shock` at line 77 and in `__all__`)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Uniform: pct: float` (`__post_init__` raises `TypeError` unless `pct` is an `int`/`float`, not `bool`); `def uniform(pct: float) -> Uniform`; `def expand_uniform(name: str, spec, cells_of: Callable[[str], Iterable[tuple]]) -> dict[tuple, float]` returning `spec` unchanged when it is not a `Uniform`, else `{cell: spec.pct for cell in cells_of(name)}`, raising `ValueError(f"uniform({pct}) on {name}: no cell can take a shock")` when empty. Task 3 reuses `Uniform` and `expand_uniform`.

- [ ] **Step 1:** Implement `Uniform`, `uniform`, `expand_uniform` in `shocks.py`; mention `uniform` in the module docstring next to the `apply_shock` example.
- [ ] **Step 2:** In `apply_shock`, before the validation loop, replace each instrument's spec with `expand_uniform(name, spec, cells_of)`, where `cells_of` yields the `shock` cells of `getattr(m, name)` that `_can_take_shock(m, name, cell)` accepts (the loop of `apply_default_tariff`). An unregistered instrument name must still raise as today, before expansion reads the Var. `levels=True` with a `Uniform` raises `ValueError` (a uniform is a percent).
- [ ] **Step 3:** `uv run pytest tests/templates/gtap/test_uniform_shock.py tests/templates/gtap/test_shocks.py -v`: all pass.
- [ ] **Step 4:** Commit `feat(gtap): uniform shock on an instrument, Pyomo route (MAR-227)`.

### Task 3: `uniform` in the family route, refusal elsewhere

**Files:**
- Modify: `src/equilibria/family/shock.py`
- Modify: `src/equilibria/family/shock_session.py:168-181`, `src/equilibria/family/altertax/run.py:170`, `src/equilibria/family/altertax/path.py:93-96`
- Test: `tests/family/test_shock_cell.py` (Task 1 test), plus one refusal test appended to `tests/templates/gtap/test_uniform_shock.py`

**Interfaces:**
- Consumes: `Uniform`, `expand_uniform` from Task 2.
- Produces: `def reject_uniform(shock: Mapping, where: str) -> None` in `templates/gtap/shocks.py`, raising `TypeError(f"{where} needs explicit cells; expand uniform(...) with apply_shock first")`.

- [ ] **Step 1:** In `family.shock.apply_shock`, expand with `expand_uniform` and `cells_of = lambda n: (c for c in period_table(mp.var(n), SHOCK) if can_take_shock(mp, sys, n, c))`.
- [ ] **Step 2:** Implement `reject_uniform`; call it at the top of the three cell-by-cell readers.
- [ ] **Step 3:** Append `test_uniform_refused_where_cells_are_read`: `_write_levels(mp, {"imptx": uniform(3)})` raises `TypeError` matching `uniform` (any `mp`; the guard runs first, so a `None` stand-in is enough if the guard precedes every use).
- [ ] **Step 4:** `uv run pytest tests/family/test_shock_cell.py tests/templates/gtap/test_uniform_shock.py -v`: pass.
- [ ] **Step 5:** Commit `feat(family): uniform shock on the array route (MAR-227)`.

### Task 4: `_solve_shock` takes the shock size

**Files:**
- Modify: `tests/templates/gtap/test_gtap7_gempack_parity.py:67-182`

**Interfaces:**
- Produces: `_solve_shock(dataset, ifsub, savf_flag="capFlex", solve_nlp=False, residual_region=None, backend=None, unit_sigma="gams", tariff_pct: float | None = None)`; `None` passes no shock (the default +10 %), a number passes `shock={"imptx": uniform(tariff_pct)}` to `Exercise`. Docstring says so.

- [ ] **Step 1:** Add the parameter and the `shock=` argument.
- [ ] **Step 2:** `uv run pytest tests/templates/gtap/test_gtap7_gempack_parity.py -k uniform_10 -v` with GAMS 53 and pymumps on the PATH (handoff 2026-11-20): PASS.
- [ ] **Step 3:** Commit `test(gtap): _solve_shock takes the tariff size (MAR-227)`.

### Task 5: The study solves every size, with a scaled tolerance

**Files:**
- Modify: `scripts/gtap/gen_linearization_study.py`
- Modify (regenerated): `docs/site/guide/gtap7_gempack_linearization_study.md`

**Interfaces:**
- Consumes: `_solve_shock(..., tariff_pct=...)` from Task 4.
- Produces: `_solve_at(dataset: str, shock_pct: str)` (cached, replaces `_solve_shock_10`; passes `tariff_pct=float(shock_pct)` except `None` for "10", so the 10 % column stays the gate's run); `_measure_within_pp(dataset, sl4, shock_pct) -> tuple[str, str]` = (within 1 pp, within `0.01 * float(shock_pct) / 10`).

- [ ] **Step 1:** Remove the `"— (py-shock pending)"` branch; the sweep table gets one cell per size showing both numbers (`"99% / 97%"`), with a header line explaining the scaled tolerance and why (at 0.1 % almost no cell moves 1 pp). Non-linearity uses the scaled numbers.
- [ ] **Step 2:** Update the page prose that says Python is solved at 10 % only.
- [ ] **Step 3:** Run `uv run python scripts/gtap/gen_linearization_study.py` (all 5 datasets x 5 sizes; ask the user first if it runs long). Expected: no "py-shock pending" in the page; any "— (err: ...)" cell is reported to the user, not hidden.
- [ ] **Step 4:** Commit `feat(gtap): linearization study solves Python at every shock size (MAR-227)`.

### Task 6: Sync the spec and close the change

**Files:**
- Modify: `specs/templates/gtap/specs/specs.md` (add the requirement exactly as the delta states it; `Last synced: <date>`)
- Move: `specs/templates/gtap/changes/uniform-shock/` → `specs/templates/gtap/changes/archive/<YYYY-MM-DD>-uniform-shock/`

- [ ] **Step 1:** Sync and archive.
- [ ] **Step 2:** `uv run pytest -m "not slow" -n 2` (about 40 min). Compare pass/fail AND skip lists with main; the known `test_cli` failure is environmental.
- [ ] **Step 3:** Commit `docs(specs): sync and archive uniform-shock (MAR-227)`.
