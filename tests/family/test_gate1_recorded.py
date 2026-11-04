"""Gate 1 for families READ FROM THE BLOCKS by FamilyBackend (no hand templates).

The recorded eq_pndeq / eq_pvaeq / eq_facty must match PyomoNLP exactly as the
hand-written Phase 0 templates do: same rows, same pattern and nnz, same F and J.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from tests.family._data import DATASETS  # noqa: E402
from tests.family._oracle import FAMILIES, assert_same, oracle  # noqa: E402


def _recorded(o):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.layout import VarLayout

    backend = FamilyBackend()
    backend.build(o.model, only=FAMILIES)
    values, free = o.state(backend.var_domains)
    layout = VarLayout.from_state(o.sets, backend.var_domains, values, free)
    return layout, backend.system(layout)


@pytest.mark.parametrize("point", ["calibrated", "perturbed"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_recorded_matches_pyomo(dataset, point):
    o = oracle(dataset)
    layout, system = _recorded(o)
    assert_same(system, layout, o, o.point(point))


def test_recorded_drop_one_cell_fails():
    import dataclasses

    from equilibria.family.core import FamilySystem

    o = oracle("gtap7_3x3")
    layout, system = _recorded(o)
    fams = list(system.families)
    k = next(i for i, f in enumerate(fams) if f.name == "eq_facty")
    g0 = fams[k].groups[-1]
    g0 = dataclasses.replace(
        g0,
        rows=g0.rows[1:],
        cols=g0.cols[1:],
        args={a: v[1:] for a, v in g0.args.items()},
    )
    fams[k] = dataclasses.replace(fams[k], groups=[*fams[k].groups[:-1], g0])
    broken = FamilySystem(fams, n_cols=system.n_cols, fixed_values=system.fixed_values)
    with pytest.raises(AssertionError):
        assert_same(broken, layout, o, o.point("calibrated"))


def test_recorded_groups_are_few():
    o = oracle("gtap7_15x10")
    _, system = _recorded(o)
    per_family = {f.name: len(f.groups) for f in system.families}
    assert set(per_family) == set(FAMILIES)
    assert max(per_family.values()) <= 4, per_family
