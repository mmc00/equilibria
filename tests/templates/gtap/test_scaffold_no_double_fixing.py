"""apply_closure() already runs apply_conditional_fixing(); do not run it twice.

MEASURED on the 20x41 (Kaggle kernel gtap-scaffold-breakdown, 2026-09-03):

    apply_closure              17 calls   190.2s
    apply_conditional_fixing   34 calls   380.1s   <- exactly 2x
    aggressive_fixing          17 calls    31.2s

The 2x is structural, not noise: ``apply_closure`` calls ``apply_conditional_fixing``
whenever the closure carries ``apply_flag_fixing`` (gtap_solver.py:389-390), and
``_run_path_capi_nonlinear_full`` then called it again unconditionally. The shock
continuation invokes that function once per lambda sub-step, so the duplicate cost
multiplies by the number of sub-steps (~190s on a 20x41 run).

Removing the duplicate is only safe because the pass is IDEMPOTENT: it derives the
fixed set purely from ``params.benchmark`` (SAM flows), never from current model
state. These tests pin BOTH facts — the idempotence that makes the removal safe, and
the wiring that makes it unnecessary — so a future edit that breaks either one fails
here instead of silently changing the fixed set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASET = "gtap7_3x3"  # smallest bundled dataset — this is a wiring test, not a solve


def _fixed_names(model) -> set[str]:
    from pyomo.environ import Var

    return {v.name for v in model.component_data_objects(Var, active=True) if v.fixed}


@pytest.fixture(scope="module")
def model_and_params():
    """Build the smallest GTAP model available, or skip."""
    d = ROOT / "datasets" / DATASET
    if not (d / "basedata.har").exists():
        pytest.skip(f"dataset {DATASET} not present at {d}")

    scripts_gtap = ROOT / "scripts" / "gtap"
    if str(scripts_gtap) not in sys.path:
        sys.path.insert(0, str(scripts_gtap))

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )
    rr = list(p.sets.r)[-1]
    model, _mp, _fx = build_sparse_model_mp(
        p, p.sets, closure, rr, base_calibrated=False, ref_gdx=None
    )
    return model, p, closure


def test_conditional_fixing_is_idempotent(model_and_params):
    """Running the pass twice fixes exactly the same variables.

    This is the property that makes dropping the duplicate call safe. If a future
    change makes the pass depend on model state, this test fails and the removal in
    run_gtap.py must be revisited.
    """
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    model, params, closure = model_and_params
    solver = GTAPSolver(model, closure=closure, solver_name="path", params=params)

    solver.apply_conditional_fixing()
    after_first = _fixed_names(model)

    solver.apply_conditional_fixing()
    after_second = _fixed_names(model)

    assert after_first == after_second, (
        "apply_conditional_fixing is NOT idempotent: the second pass changed the "
        f"fixed set (+{len(after_second - after_first)} / "
        f"-{len(after_first - after_second)}). The duplicate-call removal in "
        "run_gtap.py:_run_path_capi_nonlinear_full is no longer safe."
    )
    assert after_first, "expected the pass to fix at least one variable"


def test_apply_closure_already_runs_conditional_fixing(model_and_params):
    """apply_closure() delegates to apply_conditional_fixing when the flag is set.

    Pins the wiring the optimization relies on. Uses the REAL model (a mock closure
    breaks apply_closure, which does attribute access with computed names), and only
    spies on the delegate so nothing about the real fixing changes.
    """
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    model, params, closure = model_and_params
    solver = GTAPSolver(model, closure=closure, solver_name="path", params=params)

    if not getattr(closure, "apply_flag_fixing", False):
        pytest.skip(
            "this closure does not carry apply_flag_fixing, so apply_closure is not "
            "expected to delegate; the duplicate call in run_gtap.py is then the only "
            "one and is correctly kept"
        )

    spy = MagicMock(return_value=0)
    solver.apply_conditional_fixing = spy  # instance attribute shadows the method
    solver.apply_closure(closure)

    assert spy.call_count == 1, (
        "apply_closure no longer calls apply_conditional_fixing exactly once "
        f"(called {spy.call_count}x). The duplicate-call removal in "
        "run_gtap.py:_run_path_capi_nonlinear_full assumed it does — revisit it."
    )
