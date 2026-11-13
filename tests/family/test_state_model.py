"""The composer's post-build steps, run on arrays, give Pyomo's exact state.

``build_block_single_period`` scales levels, snapshots the Fisher constants and fixes
cells on the Pyomo model. ``compose_block_state`` runs the SAME step functions on a
``StateModel`` over the equilibria Model's arrays. Every variable cell must end with the
same level, fixed flag and lower bound, and the CLOSURE snapshot Parameters with the
values Pyomo's rebuilt Fisher rows use.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip("pyomo")
pytest.importorskip("jax")  # tests.family._oracle imports the family route

from tests.family._data import DATASETS, load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402


def _pyomo_cell(comp, idx):
    key = None if not idx else idx[0] if len(idx) == 1 else idx
    return comp[key] if key in comp else None  # noqa: SIM401 — Pyomo components


def _state_mismatches(dataset, if_sub=False):
    from equilibria.templates.gtap.gtap_block_model import (
        _set_elems,
        build_block_single_period,
        compose_block_state,
    )

    p = load_params(dataset)
    rr = _set_elems(p.sets)["r"][-1]
    closure = comp_stat_closure(if_sub)
    pm = build_block_single_period(p, p.sets, closure, rr)
    _, state = compose_block_state(p, p.sets, closure, rr)

    mismatches = []
    for name, doms in state.var_domains.items():
        comp = getattr(pm, name)
        vals, fixed, lower = state.values[name], state.fixed[name], state.lower[name]
        for pos, idx in zip(
            itertools.product(*(range(len(state.sets[d])) for d in doms)),
            itertools.product(*(state.sets[d] for d in doms)),
            strict=True,
        ):
            cell = _pyomo_cell(comp, idx)
            if cell is None:
                continue
            want = (cell.value, cell.fixed, cell.lb)
            got = (vals[pos], bool(fixed[pos]), lower[pos])
            if (
                not np.isclose(
                    got[0], want[0] if want[0] is not None else 0.0, rtol=1e-15, atol=0
                )
                or got[1] != want[1]
                or not ((want[2] is None and got[2] == -np.inf) or got[2] == want[2])
            ):
                mismatches.append((f"{name}{list(idx)}", got, want))

    # Mutable Parameters the steps may set (e.g. xet_flag during scaling).
    from pyomo.environ import value

    n_mutable = 0
    for name, comp in state._components.items():
        if getattr(comp, "mutable", False):
            n_mutable += 1
            pc = getattr(pm, name)
            for flat, key in enumerate(comp.keys()):
                k = None if not comp.domains else key
                if k in pc and comp.data.flat[flat] != value(pc[k]):
                    mismatches.append(
                        (f"{name}[{key}]", comp.data.flat[flat], value(pc[k]))
                    )
    n_fixed = sum(int(f.sum()) for f in state.fixed.values())
    return mismatches, n_mutable, n_fixed


@pytest.mark.parametrize("dataset", DATASETS)
def test_state_matches_pyomo_after_composer_steps(dataset):
    mismatches, n_mutable, n_fixed = _state_mismatches(dataset)
    assert not mismatches, f"{len(mismatches)} cells differ, e.g. {mismatches[:6]}"
    assert n_mutable > 0 and n_fixed > 0


@pytest.mark.parametrize("dataset", DATASETS)
def test_state_matches_pyomo_ifsub1(dataset):
    mismatches, _, _ = _state_mismatches(dataset, if_sub=True)
    assert not mismatches, f"{len(mismatches)} cells differ, e.g. {mismatches[:6]}"


def test_state_check_can_fail(monkeypatch):
    import equilibria.templates.gtap.gtap_block_model as gbm

    calls = []
    real = gbm.apply_production_scaling

    def skip_on_state(m, ctx):
        # Scale the Pyomo model as usual; skip the StateModel one.
        if type(m).__name__ == "StateModel":
            calls.append(1)
            return None
        return real(m, ctx)

    monkeypatch.setattr(gbm, "apply_production_scaling", skip_on_state)
    mismatches, _, _ = _state_mismatches("gtap7_3x3")
    assert calls and mismatches
