"""The shock in steps on the family route (MAR-251): a step that fails is reported as
the period's status. (That the steps give Pyomo's numbers, ME9B included, is
``tests/scripts/test_run_burfisher_family.py``.)"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo")

from equilibria._local_refs import nus333_dir  # noqa: E402
from tests.family._period_levels import solver_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _rb():
    spec = importlib.util.spec_from_file_location(
        "run_burfisher", ROOT / "scripts" / "gtap" / "run_burfisher.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_family_failed_step_reports_its_status(monkeypatch, tmp_path):
    """A solve that fails at the second step ends the shock with that status and the
    shocked cells back at the full shock (``strategies.in_steps``)."""
    from equilibria.templates.gtap import recipe, strategies

    rb = _rb()
    calls = {"n": 0}
    real = strategies.in_steps

    def failing_in_steps(lambdas):
        inner = real(lambdas)

        def strategy(model, solver):
            def flaky(m):
                calls["n"] += 1
                return "infeasible" if calls["n"] == 2 else solver(m)

            return inner(model, flaky)

        return strategy

    monkeypatch.setattr(recipe, "in_steps", failing_in_steps)
    with solver_env({"EQUILIBRIA_FAMILY_CACHE": str(tmp_path)}):
        ex = rb.exercise("ME9B", nus333_dir(), backend="family")
        out = ex.run()
        cells = strategies.shocked_cells(ex.model)
    assert out["shock"]["code"] == "infeasible"
    assert calls["n"] == 2
    assert cells, "the shocked cells are back at the full shock"
