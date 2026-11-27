# port-to-jax: spec

## Goal

Take an equilibria CGE model (a blocks template, or an original GAMS/GEMPACK model, via stretch 1) that is not GTAP and make it run on the
JAX "family" backend with no Pyomo model, matching its GAMS or GEMPACK reference at a
tolerance the model owner sets after measuring. The skill also rescues a family run of
such a template that fails, is not square, or drifts from its reference. It reuses the
generic pieces that already exist (`build_arrays`, `GenericFamilyPeriodModel`,
`newton_family`) and never writes a new period model, array builder or Newton.

## Outputs

- The template with a `backend` field (`"pyomo"` stays the default) whose `build()` and
  `default_recipe()` use the generic family pieces.
- Family tests: residual at the calibration, `check` back to `base`, no Pyomo model built,
  runs that do not leak, and a cell-by-cell parity test against the reference.
- A measured table of the worst gap per backend and period (`measure_gams_gap.py` style)
  to hand the model owner, who sets the tolerance.
- The new test files listed in the CI groups.

## Tools

- `uv run pytest` on `tests/family/` (the simple_open files are the worked example).
- `scripts/simple_open/measure_gams_gap.py` (the template for measuring a gap).
- `scripts/dev/plan_test_groups.py --refresh-files` (restore what it drops) or lines added by hand, for the CI lists.
- Skills it points to: `equilibria-parity-debug`, `xmodel-parity` (parity gaps).

## Forbidden

- Writing a new period model, array builder or Newton solver for a model that blocks cover.
- Picking or loosening a tolerance, or editing the shared compare map yaml: measure and ask the owner.
- Falling back to a Pyomo model inside the family path, or asserting `"pyomo" not in sys.modules`.
- Touching the GTAP family route (`gtap_periods.py`, `run_setup.py`, `newton_gtap.py`).
- `git push`, opening a PR, running GAMS, or running the GTAP parity gates from this skill.
- `.venv/bin/python`, a bare `python`, or a path under `/Users/`.

## Agents

Claude Code, measured with Sonnet (`uv run python scripts/dev/run_skill_evals.py port-to-jax --agent claude --ablation`).
The skill is plain Markdown in the open Agent Skills format, so Codex reads it too; it has not been measured there.

The description keeps the old skill's trigger on an original GAMS/GEMPACK model: stretch 1 (original code to blocks) is still in the body, marked pending, and sends the agent to the parity skills until the GTAP dynamic port fills it.
