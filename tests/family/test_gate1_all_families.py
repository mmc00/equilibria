"""Gate 1 for EVERY SP family read from the blocks (comp-stat, ifSUB=0)."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from tests.family._data import DATASETS  # noqa: E402
from tests.family._oracle import assert_same, oracle  # noqa: E402


def _all_recorded(o, if_sub=False):
    """The whole family path, no Pyomo state: compose + composer steps on arrays,
    record the blocks, lay out the state."""
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.templates.gtap.gtap_block_model import compose_block_state
    from tests.family._oracle import comp_stat_closure

    model, state = compose_block_state(
        o.params, o.params.sets, comp_stat_closure(if_sub), o.residual_region
    )
    backend = FamilyBackend()
    backend.build(model, skip=state.ifsub_skipped)
    assert set(backend.records).isdisjoint(state.ifsub_skipped)
    layout = backend.layout(state)
    return layout, backend.system(layout)


@pytest.mark.parametrize("point", ["calibrated", "perturbed"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_all_families_match_pyomo(dataset, point):
    o = oracle(dataset)
    layout, system = _all_recorded(o)
    assert len(system.families) == 100
    assert_same(system, layout, o, o.point(point), families=None)


@pytest.mark.parametrize("point", ["calibrated", "perturbed"])
@pytest.mark.parametrize("dataset", ["gtap7_3x3", "gtap7_10x7"])
def test_all_families_match_pyomo_ifsub1(dataset, point):
    o = oracle(dataset, if_sub=True)
    layout, system = _all_recorded(o, if_sub=True)
    assert len(system.families) == 100 - 9
    assert_same(system, layout, o, o.point(point), families=None)
