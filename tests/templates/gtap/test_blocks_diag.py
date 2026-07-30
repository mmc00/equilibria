"""Diagnostic tools must surface a planted residual / form / domain mismatch.

Guards `scripts/gtap/blocks_diag.py`, the read-only diagnostics stood up
BEFORE the first B-block solve (F3 Task 2) so a failed solve is debuggable
directed, not blind. Solver-free, milliseconds — no GAMS/GDX involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "src"))


def _toy_model_one_violation():
    """Two constraints at a fixed point: eq_bad violated, eq_good satisfied."""
    from pyomo.environ import ConcreteModel, Constraint, Reals, Var

    m = ConcreteModel()
    m.x = Var(domain=Reals, initialize=5.0)
    m.y = Var(domain=Reals, initialize=1.0)
    # x - y == 0 at (5, 1) -> residual |5 - 1 - 0| = 4 (violated)
    m.eq_bad = Constraint(expr=m.x - m.y == 0)
    # x + y == 6 at (5, 1) -> residual |6 - 6| = 0 (satisfied)
    m.eq_good = Constraint(expr=m.x + m.y == 6)
    return m


def _toy_model(domain: str):
    """A one-var, one-constraint model whose var `x` has the given domain."""
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Reals,
        Var,
    )

    domains = {"Reals": Reals, "NonNegativeReals": NonNegativeReals}
    m = ConcreteModel()
    m.x = Var(domain=domains[domain], initialize=1.0)
    m.eq_x = Constraint(expr=m.x >= 0)
    return m


def test_residual_report_ranks_worst_first():
    from blocks_diag import residual_report

    m = _toy_model_one_violation()
    rep = residual_report(m, seed=None)
    assert rep[0][2] > rep[-1][2]  # sorted worst-first
    assert rep[0][0] == "eq_bad"  # the violated one is on top


def test_domain_bounds_diff_flags_mismatch():
    from blocks_diag import domain_bounds_diff

    a = _toy_model(domain="Reals")
    b = _toy_model(domain="NonNegativeReals")
    diffs = domain_bounds_diff(a, b)
    assert any(d[0] == "x" for d in diffs)


def test_domain_bounds_diff_no_mismatch_when_identical():
    from blocks_diag import domain_bounds_diff

    a = _toy_model(domain="Reals")
    b = _toy_model(domain="Reals")
    diffs = domain_bounds_diff(a, b)
    assert diffs == []


def test_form_diff_synthetic_expressions():
    """Synthetic two-expression compare (no real B-block exists until Task 4).

    Builds two tiny Pyomo Constraint components over the SAME vars: one index
    where the expressions match structurally, and one where they differ.
    """
    from blocks_diag import form_diff
    from pyomo.environ import ConcreteModel, Constraint, Set, Var

    m = ConcreteModel()
    m.I = Set(initialize=["same", "different"])
    m.x = Var(m.I, initialize=1.0)
    m.y = Var(m.I, initialize=2.0)

    block = {
        "same": m.x["same"] + m.y["same"] == 3,
        "different": m.x["different"] * 2 == 4,
    }
    monolith = {
        "same": m.x["same"] + m.y["same"] == 3,
        "different": m.x["different"] + m.y["different"] == 4,
    }

    diffs = form_diff(block, monolith)
    idx_diffed = {d[0] for d in diffs}
    assert "different" in idx_diffed
    assert "same" not in idx_diffed


def test_form_diff_cross_model_identical_reports_equal():
    """Cross-model widening: same algebra on TWO ConcreteModels → NO diff.

    The Task-4 form gate compares a block-built model against a separately-built
    monolith model. ``compare_expressions`` matches leaf components by object
    identity, so a structurally identical expression built on a DIFFERENT
    ConcreteModel returns False (no exception) — the string fallback in the old
    ``_exprs_equal`` never ran, and every cell of a byte-perfect port flagged a
    false diff. This test builds the SAME expression on two separate models and
    asserts ``form_diff`` reports them equal.
    """
    from blocks_diag import form_diff
    from pyomo.environ import ConcreteModel, Set, Var

    def _build():
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I, initialize=1.0)
        return {"cell": m.x[1] + 2 * m.x[2] == 0}

    block = _build()
    monolith = _build()  # a DIFFERENT ConcreteModel, identical algebra

    diffs = form_diff(block, monolith)
    assert diffs == []


def test_form_diff_same_model_different_coefficient_reports_diff():
    """Caveat guard: widening must NOT blind the tool.

    A genuine structural difference on the SAME model (``2*x`` vs ``3*x``) must
    STILL report a diff — the strings differ, so string-equality flags it. We
    widened "equal" to cross the model-identity boundary, not to swallow real
    algebraic re-shapes.
    """
    from blocks_diag import form_diff
    from pyomo.environ import ConcreteModel, Set, Var

    m = ConcreteModel()
    m.I = Set(initialize=["c"])
    m.x = Var(m.I, initialize=1.0)

    block = {"c": 2 * m.x["c"] == 0}
    monolith = {"c": 3 * m.x["c"] == 0}

    diffs = form_diff(block, monolith)
    assert {d[0] for d in diffs} == {"c"}
