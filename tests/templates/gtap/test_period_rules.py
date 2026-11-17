"""The GTAP period rules: constants and closures the steps share, outside Pyomo."""

from __future__ import annotations

import ast
import pathlib

import pytest

from equilibria.templates.gtap import gtap_multiperiod_driver as drv
from equilibria.templates.gtap import period_prep
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "equilibria"
BACKEND = ("pyomo", "equilibria.backends", "equilibria.solver")


def _imports(path: pathlib.Path) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


def test_period_rules_import_no_backend() -> None:
    bad = sorted(
        m
        for m in _imports(SRC / "templates" / "gtap" / "period_rules.py")
        if m.startswith(BACKEND)
        or m.endswith(("gtap_multiperiod_driver", "period_prep"))
    )
    assert bad == []


def test_old_names_are_the_rules() -> None:
    from equilibria.templates.gtap import period_rules as r

    assert period_prep.F35_DERIVED_DEMAND is r.F35_DERIVED_DEMAND
    assert period_prep._SHOCK_PNUM == r.SHOCK_PNUM == 1.5
    assert period_prep._RECALIBRATE_FROM is r.RECALIBRATE_FROM
    assert drv.GAMS_BOUNDED_VARS is r.GAMS_BOUNDED_VARS
    assert drv._WELFARE_LEAVES is r.WELFARE_LEAVES
    assert drv._WELFARE_SOLVE_ORDER is r.WELFARE_SOLVE_ORDER
    assert drv.JOHANSEN_STEP == r.JOHANSEN_STEP == 0.01


@pytest.mark.parametrize("if_sub", [False, True])
def test_period_closures_match_the_driver(if_sub: bool) -> None:
    from equilibria.templates.gtap.period_rules import period_closures

    user = GTAPClosureConfig(
        name="mine",
        closure_type="MCP",
        capital_mobility="mobile",
        if_sub=if_sub,
        numeraire="pnum",
    )
    base, alt = period_closures(user)
    assert (base.name, base.capital_mobility) == ("base", "sluggish")
    assert (base.fix_taxes, base.fix_technology, base.fix_endowments) == (
        False,
        False,
        False,
    )
    assert (alt.name, alt.capital_mobility) == ("altertax", "mobile")
    assert (alt.fix_taxes, alt.fix_technology, alt.fix_endowments) == (
        True,
        True,
        False,
    )
    for c in (base, alt):
        assert c.closure_type == "MCP"
        assert c.if_sub is if_sub
        assert c.numeraire == "pnum"


def test_period_closures_without_closure() -> None:
    from equilibria.templates.gtap.period_rules import period_closures

    base, alt = period_closures(None)
    assert base.if_sub is False and alt.if_sub is False
    assert base.numeraire == alt.numeraire == "pnum"
