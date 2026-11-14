"""simple_open assembled from its blocks: square, zero at base, rates fixed.

That the check returns the base is tested through the driver (test_driver_run.py).
"""

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
    pm = SimpleOpenEconomy().period_model()
    assert _rows(pm) == _free_vars(pm)


def test_base_residual_is_zero() -> None:
    pm = SimpleOpenEconomy().period_model()
    assert max_residual(pm, _equations(pm)) <= 1e-10


def test_unknown_numeraire_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\('CAP', 'LAB'\)"):
        SimpleOpenEconomy(numeraire="XYZ").period_model()


def test_tax_rates_are_fixed_variables() -> None:
    template = SimpleOpenEconomy()
    pm = template.period_model()
    values = template.calibration().values
    for name in ("tau_d", "tau_z", "tau_m"):
        rate = pm.find_component(name)
        assert rate.ctype is pyo.Var, name
        assert all(rate[k].fixed for k in rate), name
        got = [pyo.value(rate[k]) for k in rate]
        assert got == pytest.approx(list(np.atleast_1d(values[name]))), name
