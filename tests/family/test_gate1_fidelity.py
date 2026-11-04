"""Gate 1: the family backend reproduces Pyomo's rows, nnz, F and J (eq_pndeq/eq_pvaeq/eq_facty).

Calibrated point and a perturbed point (every primal times 1 + 0.05*U(-1,1), seed 0),
on gtap7_3x3, gtap7_10x7 and gtap7_15x10. The two negative tests prove the comparison
can fail: dropping one cell or nudging one parameter by 1e-9 must trip it.
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from equilibria.family.core import FamilySystem  # noqa: E402
from tests.family._data import DATASETS  # noqa: E402
from tests.family._oracle import assert_same, oracle  # noqa: E402


@pytest.mark.parametrize("point", ["calibrated", "perturbed"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_matches_pyomo(dataset, point):
    o = oracle(dataset)
    layout, system = o.system()
    assert_same(system, layout, o, o.point(point))


def _with_family(system, name, **changes):
    fams = [
        dataclasses.replace(f, **changes) if f.name == name else f
        for f in system.families
    ]
    return FamilySystem(fams, n_cols=system.n_cols, fixed_values=system.fixed_values)


def test_drop_one_cell_fails():
    o = oracle("gtap7_3x3")
    layout, system = o.system()
    fam = next(f for f in system.families if f.name == "eq_facty")
    broken = _with_family(
        system,
        "eq_facty",
        term_rows=fam.term_rows[1:],
        term_cols=fam.term_cols[1:],
        term_args={k: v[1:] for k, v in fam.term_args.items()},
    )
    with pytest.raises(AssertionError, match="pattern differs"):
        assert_same(broken, layout, o, o.point("calibrated"))


def test_perturbed_param_fails():
    o = oracle("gtap7_3x3")
    layout, system = o.system()
    fam = next(f for f in system.families if f.name == "eq_pndeq")
    io = np.array(fam.term_args["io"], copy=True)
    io[0] *= 1.0 + 1e-9
    broken = _with_family(system, "eq_pndeq", term_args={**fam.term_args, "io": io})
    with pytest.raises(AssertionError):
        assert_same(broken, layout, o, o.point("perturbed"))


def test_ces_exponent_is_exercised():
    # eq_pndeq is linear on these datasets (sigmand = 0); eq_pvaeq carries the CES power.
    o = oracle("gtap7_15x10")
    _, system = o.system()
    e = np.asarray(
        next(f for f in system.families if f.name == "eq_pvaeq").row_args["e"]
    )
    assert (np.abs(e - 1.0) > 0.1).all() and (e < 0).any() and (e > 0).any()
