"""summary() on a real solve: the README's nus333 example (bundled data, PATH)."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def solved():
    import sys

    from equilibria._local_refs import path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    from equilibria import load_bundled
    from equilibria.templates.gtap import apply_shock
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    params = load_bundled("gtap", "nus333")
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    m, _ = build_block_model(params, params.sets, closure, "ROW")
    apply_shock(m, {"imptx": {("USA", "MFG", "ROW"): 10.0}})
    results = solve_multiperiod(
        m, params, closure, mode="gtap", skip_base_solve=True, solve_check=True
    )
    assert results["shock"]["code"] == 1, results
    return m, results


def test_solved_shock_has_small_residuals_and_walras(solved):
    from equilibria.report import summary

    m, results = solved
    console = Console(file=io.StringIO(), width=100, record=True)
    s = summary(m, results, period="shock", console=console)

    assert s.code == 1
    assert s.n_equations > 500  # every equation of the period, not a handful
    assert s.unevaluated == 0
    assert s.max_residual < 1e-6
    assert s.walras is not None and abs(s.walras) < 1e-6
    assert "SOLVE COMPLETE" in console.export_text()


def test_a_perturbed_solution_shows_up_as_the_worst_equation(solved):
    """The residuals are recomputed from the model, not copied from the solver."""
    from equilibria.report import summary

    m, results = solved
    v = m.xp["USA", "MFG", "shock"]
    old = v.value
    try:
        v.set_value(old * 1.01)
        s = summary(m, results, period="shock", console=Console(file=io.StringIO()))
    finally:
        v.set_value(old)

    assert s.max_residual > 1e-4
    assert any("USA" in w.equation and "MFG" in w.equation for w in s.worst)


def test_an_earlier_period_is_refused_not_mislabelled(solved):
    """After the solve only the shock's equations are active; check must not
    silently report the shock's numbers."""
    from equilibria.report import summary

    m, results = solved
    with pytest.raises(ValueError, match="no active equations"):
        summary(m, results, period="check", console=Console(file=io.StringIO()))


def test_no_price_on_its_floor_after_a_normal_solve(solved):
    from equilibria.report import summary

    m, results = solved
    s = summary(m, results, period="shock", console=Console(file=io.StringIO()))

    assert s.on_floor == []


def test_a_price_pushed_onto_its_floor_is_flagged(solved):
    """The floor is the driver's own bound (max(1e-8, 1e-3*init)), not a test value."""
    from equilibria.report import summary

    m, results = solved
    v = m.pf["USA", "CAPITAL", "AGR", "shock"]
    assert v.lb is not None and v.lb > 0
    old = v.value
    try:
        v.set_value(v.lb)
        s = summary(m, results, period="shock", console=Console(file=io.StringIO()))
    finally:
        v.set_value(old)

    assert s.on_floor == ["pf[USA,CAPITAL,AGR,shock]"]
