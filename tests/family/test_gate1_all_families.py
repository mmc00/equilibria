"""Gate 1 for EVERY SP family read from the blocks (comp-stat, ifSUB=0)."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from tests.family._data import DATASETS  # noqa: E402
from tests.family._oracle import assert_same, oracle  # noqa: E402


def _fisher_snapshot_model(o):
    """A fresh composed Model carrying the composer's post-scaling Fisher snapshot.

    ``apply_fisher_snapshot_overwrite`` rebuilds eq_mfr_*/eq_mfw_*/eq_pfact/eq_pwfact in
    Pyomo after scaling (pf0 = pf.l, xf0 = xf.l, mqfact*_bb = sum pf0*xf0/xscale). The
    same snapshot is applied here to the CLOSURE block's Parameters, so the block rules
    produce those rows themselves.
    """
    import numpy as np

    from equilibria.templates.gtap.gtap_block_model import compose_block_sp
    from tests.family._oracle import comp_stat_closure

    model = compose_block_sp(
        o.params, o.params.sets, comp_stat_closure(), o.residual_region
    )[0]
    pm = model.parameter_manager
    values, _ = o.state({"pf": ("r", "f", "a"), "xf": ("r", "f", "a")})
    aa, a = o.sets["aa"], o.sets["a"]
    xscale = pm.get("xscale").value[:, [aa.index(x) for x in a]]
    ok = (xscale > 1e-12)[:, None, :]
    safe = np.where(xscale > 1e-12, xscale, 1.0)[:, None, :]
    contrib = np.where(ok, values["pf"] * values["xf"] / safe, 0.0)
    s_reg = contrib.sum(axis=(1, 2))
    pm.get("pf0").value = values["pf"]
    pm.get("xf0").value = values["xf"]
    pm.get("mqfactr_bb").value = np.where(s_reg > 0.0, s_reg, 1.0)
    w = s_reg.sum()
    pm.get("mqfactw_bb").value = np.array([w if w > 0.0 else 1.0])
    return model


def _all_recorded(o):
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.layout import VarLayout

    backend = FamilyBackend()
    backend.build(_fisher_snapshot_model(o))
    values, free = o.state(backend.var_domains)
    layout = VarLayout.from_state(o.sets, backend.var_domains, values, free)
    return layout, backend.system(layout)


@pytest.mark.parametrize("point", ["calibrated", "perturbed"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_all_families_match_pyomo(dataset, point):
    o = oracle(dataset)
    layout, system = _all_recorded(o)
    assert len(system.families) == 100
    assert_same(system, layout, o, o.point(point), families=None)
