"""Tests for solve_sequential — Option C (fresh model per period).

Each period gets its own ConcreteModel warm-started from the previous
period's solution, exactly like GAMS solveloop.gms.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))
sys.path.insert(0, str(Path('scripts/gtap').resolve()))

from pyomo.environ import value as pyo_value
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import default_gtap_contract

GDX = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()


@pytest.mark.slow
def test_solve_sequential_single_period():
    """t_set=('base',) → just like running the single-period solver."""
    from run_gtap import solve_sequential

    contract = default_gtap_contract()
    p = GTAPParameters()
    p.load_from_gdx(GDX)
    eq = GTAPModelEquations(p.sets, p, contract.closure, t_set=("base",))
    m = eq.build_model()
    results = solve_sequential(
        m, p, closure_config=contract.closure, t_set=("base",),
    )
    assert "base" in results
    assert results["base"]["code"] == 1
    assert results["base"]["residual"] < 1e-7
    base_model = results["_models"]["base"]
    assert abs(float(pyo_value(base_model.yc['USA', 'base'])) - 13.3326) < 0.01


@pytest.mark.slow
def test_solve_sequential_three_periods_no_shock_matches_base():
    """Three periods with no shock → all converge; check ~ NEOS check value."""
    from run_gtap import solve_sequential

    contract = default_gtap_contract()
    p = GTAPParameters()
    p.load_from_gdx(GDX)
    eq = GTAPModelEquations(
        p.sets, p, contract.closure, t_set=("base",),
    )
    m = eq.build_model()
    results = solve_sequential(
        m, p, closure_config=contract.closure,
        t_set=("base", "check", "shock"),
    )
    assert all(results[t]["code"] == 1 for t in ("base", "check", "shock")), (
        {t: results[t]["code"] for t in ("base", "check", "shock")}
    )
    check_model = results["_models"]["check"]
    yc_check = float(pyo_value(check_model.yc['USA', 'base']))
    assert abs(yc_check - 11.83) < 0.05, (
        f"yc[USA,check]={yc_check}, expected ~11.83"
    )
