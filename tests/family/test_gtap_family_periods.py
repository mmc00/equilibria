"""GTAPFamilyPeriodModel: the GTAP operations over the arrays, on gtap7_3x3.

Each operation is checked against the array function of the old family recipe it
stands for (``prep_recipe``), on a copy of the same ``prepare_run`` arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_prep_recipe import _params
from tests.family.test_squared_system import _route_closure

DS = "gtap7_3x3"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    from equilibria.family.driver import prepare_run

    load_params(DS)
    mp_env = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp_env.setenv(k, v)
    p = _params(DS)
    try:
        yield prepare_run(
            p,
            _route_closure(),
            list(p.sets.r)[-1],
            cache_root=tmp_path_factory.mktemp("records"),
            jax_cache=False,
        )
    finally:
        mp_env.undo()


def _model(prepared):
    from equilibria.family.gtap_periods import GTAPFamilyPeriodModel

    run, mp = prepared
    return GTAPFamilyPeriodModel(run, mp.copy())


def _same_arrays(a, b) -> None:
    assert a.vars.keys() == b.vars.keys()
    for name, v in a.vars.items():
        w = b.vars[name]
        for f in ("value", "fixed", "lb", "ub"):
            x, y = getattr(v, f), getattr(w, f)
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), (name, f)


def test_reference_matches_sp_ref(prepared) -> None:
    run, _ = prepared
    ref = _model(prepared).sp_reference(None, None, "check", fresh=True)
    for name, sp in run.ref.vars.items():
        want_fixed = {sp.keys[i]: float(sp.value[i]) for i in np.flatnonzero(sp.fixed)}
        assert ref.fixed.get(name, {}) == want_fixed, name
        for i, k in enumerate(sp.keys):
            lo = float(sp.lb[i]) if np.isfinite(sp.lb[i]) else None
            hi = float(sp.ub[i]) if np.isfinite(sp.ub[i]) else None
            if lo is None and hi is None:
                assert k not in ref.bounds.get(name, {}), (name, k)
            else:
                assert ref.bounds[name][k] == (lo, hi), (name, k)


def test_default_tariff_same_cells_as_prep_recipe(prepared) -> None:
    from equilibria.family.shock import apply_default_tariff

    run, mp = prepared
    m = _model(prepared)
    m.write_default_tariff(gtap_mode=True)
    old = mp.copy()
    apply_default_tariff(old, run.backend)
    _same_arrays(m.mp, old)
    assert m.shock_labels()


def test_user_shock_same_cells_as_prep_recipe(prepared) -> None:
    from equilibria.family.shock import apply_shock

    run, mp = prepared
    imptx = mp.vars["imptx"]
    cell = next(k[:-1] for k in imptx.keys if k[-1] == "shock")
    shock = {"imptx": {cell: 5.0}}
    m = _model(prepared)
    m.write_shock(shock)
    old = mp.copy()
    apply_shock(old, run.backend, shock)
    _same_arrays(m.mp, old)
    assert m.shock_labels() == [f"imptx{cell}"]


def test_residual_nan_until_solved(prepared) -> None:
    m = _model(prepared)
    assert math.isnan(m.residual("check"))
    m.stage("closure", "params", gtap_mode=True, period="check")
    m.record_solve({"termination_code": 1, "residual": 1e-12})
    assert m.residual("check") == 1e-12
    assert math.isnan(m.residual("shock"))


def test_exogenous_is_the_instruments_shock_cells(prepared) -> None:
    run, mp = prepared
    m = _model(prepared)
    exo = m.exogenous()
    assert exo and {name for name, _ in exo} <= set(mp.exogenous)
    assert m.exogenous_cells("qo", "check") == frozenset()


def test_altertax_refused(prepared) -> None:
    with pytest.raises(NotImplementedError, match="altertax"):
        _model(prepared).recalibrate_and_ava("check", "base", None)


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_replicate_reference_same_arrays_as_cell_by_cell(prepared, period) -> None:
    from equilibria.templates.gtap.period_model import apply_reference

    run, _ = prepared
    m, old = _model(prepared), _model(prepared)
    # cells already fixed keep their level: pre-fix, at another level, cells the
    # reference fixes, on both sides
    name, sp = next((n, s) for n, s in run.ref.vars.items() if s.fixed.sum() >= 3)
    body = {sp.keys[i] for i in np.flatnonzero(sp.fixed)[:3]}
    for x in (m, old):
        v = x.mp.vars[name]
        at = [i for i, k in enumerate(v.keys) if k[-1] == period and k[:-1] in body]
        assert at, name
        v.value[at], v.fixed[at] = 7.0, True
    m.replicate_reference(None, None, period, fresh=False)
    apply_reference(old, old.sp_reference(None, None, period, fresh=False), period)
    _same_arrays(m.mp, old.mp)
