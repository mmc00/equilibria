"""The two Phase 0 GTAP families build the expected rows and pattern from data alone."""

import numpy as np
import pytest

pytest.importorskip("jax")

from equilibria.blocks.gtap import _derived_params as dp  # noqa: E402
from equilibria.family.gtap_phase0 import PHASE0_VARS, build_phase0_system  # noqa: E402
from equilibria.family.layout import VarLayout  # noqa: E402
from equilibria.templates.gtap.gtap_block_model import _set_elems  # noqa: E402
from tests.family._data import load_params  # noqa: E402


def _all_free_layout(sets):
    shapes = {v: tuple(len(sets[d]) for d in doms) for v, doms in PHASE0_VARS.items()}
    return VarLayout.from_state(
        sets,
        PHASE0_VARS,
        values={v: np.ones(s) for v, s in shapes.items()},
        free={v: np.ones(s, bool) for v, s in shapes.items()},
    )


def test_shapes_3x3():
    p = load_params("gtap7_3x3")
    sets = _set_elems(p.sets)
    layout = _all_free_layout(sets)
    system = build_phase0_system(p, p.sets, layout, residual_region=sets["r"][-1])

    r, i, a, f = (sets[k] for k in "riaf")
    io = dp.to_array(dp.io_param_data(p, p.sets), [r, i, a], 0.0)
    xfflag = dp.to_array(dp.xfflag_data(p, p.sets), [r, f, a], 0.0)
    n_pnd = int((io > 0).any(axis=1).sum())

    by_family = {}
    for name, _ in system.rows:
        by_family[name] = by_family.get(name, 0) + 1
    assert by_family == {"eq_pndeq": n_pnd, "eq_facty": len(r)}
    assert system.nnz == n_pnd + int((io > 0).sum()) + 3 * len(r) + 2 * int(
        (xfflag > 0).sum()
    )
    assert len(layout.col_names) == system.n_cols
    assert layout.col_names[0] == f"pa[{r[0]},{i[0]},{sets['aa'][0]}]"
