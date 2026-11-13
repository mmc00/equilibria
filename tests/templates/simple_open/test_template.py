"""simple_open assembled from its blocks: square, zero at base, check returns base."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest
from tests.templates.simple_open._helpers import max_residual

from equilibria.templates.simple_open import SimpleOpenEconomy


def _equations(pm: pyo.ConcreteModel) -> list[str]:
    return [
        c.name.removesuffix("_con")
        for c in pm.component_objects(pyo.Constraint, active=True)
    ]


def _free_vars(pm: pyo.ConcreteModel) -> int:
    return sum(
        1
        for v in pm.component_objects(pyo.Var, active=True)
        for k in v
        if not v[k].fixed
    )


def _rows(pm: pyo.ConcreteModel) -> int:
    return sum(1 for c in pm.component_objects(pyo.Constraint, active=True) for _ in c)


def test_system_is_square() -> None:
    pm = SimpleOpenEconomy().build()
    assert _rows(pm) == _free_vars(pm)


def test_base_residual_is_zero() -> None:
    pm = SimpleOpenEconomy().build()
    assert max_residual(pm, _equations(pm)) <= 1e-10


def test_check_returns_base() -> None:
    template = SimpleOpenEconomy()
    pm = template.build()
    base = {
        (v.name, k): pyo.value(v[k])
        for v in pm.component_objects(pyo.Var, active=True)
        for k in v
    }
    for v in pm.component_objects(pyo.Var, active=True):
        for k in v:
            if v.name != "WALRAS" and not v[k].fixed:  # instruments stay put
                v[k].set_value(base[(v.name, k)] * 1.05)
    assert template.solve(pm) == "optimal"
    for (name, k), b in base.items():
        got = pyo.value(pm.find_component(name)[k])
        assert got == pytest.approx(b, rel=1e-8, abs=1e-8), (name, k)
    assert abs(pyo.value(pm.WALRAS)) <= 1e-8


def test_unknown_numeraire_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\('CAP', 'LAB'\)"):
        SimpleOpenEconomy(numeraire="XYZ").build()


def test_tax_rates_are_fixed_variables() -> None:
    template = SimpleOpenEconomy()
    pm = template.build()
    values = template.calibration().values
    for name in ("tau_d", "tau_z", "tau_m"):
        rate = pm.find_component(name)
        assert rate.ctype is pyo.Var, name
        assert all(rate[k].fixed for k in rate), name
        got = [pyo.value(rate[k]) for k in rate]
        assert got == pytest.approx(list(np.atleast_1d(values[name]))), name
