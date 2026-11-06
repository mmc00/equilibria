"""``MPArrays.build`` (the multiperiod state from the composed state) equals the Pyomo model's.

Oracle: photo ``001_A_base`` of the saved Pyomo run, the whole multiperiod state at entry
of the first ``PeriodPreparer.prepare``. ``calibrate_base`` runs the settle on the model
``build_block_model`` just built, and its first ``prepare`` is that call: nothing touches
the state between the reflection (plus the Fisher rows) and the photo except
``solve_multiperiod`` stamping ``_residual_region`` / ``_gtap_mode`` (attributes, not
cells).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import _params
from tests.family.test_squared_system import _route_closure

RTOL = 1e-9
DATASETS = (
    "gtap7_3x3",
    "gtap7_10x7",
    pytest.param("gtap7_20x41", marks=pytest.mark.slow),
)


def _few(items, n: int = 5) -> str:
    items = sorted(map(str, items))
    return ", ".join(items[:n]) + (f" (+{len(items) - n})" if len(items) > n else "")


def diff_to_photo(mp, photo, rtol: float = RTOL) -> list[str]:
    """Differences between ``mp`` and a photo (empty = equal): names, keys, values within
    ``rtol`` (NaN equals NaN), fixed flags, bounds, row active flags."""
    msgs = []
    if set(mp.vars) != set(photo.vars):
        msgs.append(
            f"var components: missing {_few(set(photo.vars) - set(mp.vars))}; "
            f"extra {_few(set(mp.vars) - set(photo.vars))}"
        )
    if set(mp.rows) != set(photo.rows):
        msgs.append(
            f"row components: missing {_few(set(photo.rows) - set(mp.rows))}; "
            f"extra {_few(set(mp.rows) - set(photo.rows))}"
        )
    for name in sorted(set(mp.vars) & set(photo.vars)):
        v, c = mp.vars[name], photo.vars[name]
        if list(v.keys) == list(c.keys):
            sel = slice(None)
            got = v
        else:
            if set(v.keys) != set(c.keys) or len(v.keys) != len(c.keys):
                msgs.append(
                    f"{name}: keys missing {_few(set(c.keys) - set(v.keys))}; "
                    f"extra {_few(set(v.keys) - set(c.keys))}"
                )
                continue
            order = np.array([v.pos(k) for k in c.keys])
            sel, got = order, v
        gv, gf, gl, gu = (
            got.value[sel],
            got.fixed[sel],
            got.lb[sel],
            got.ub[sel],
        )
        close = (np.isnan(gv) & np.isnan(c.value)) | (
            (gv == c.value)
            | (np.abs(gv - c.value) <= rtol * np.maximum(1.0, np.abs(c.value)))
        )
        for what, bad in (
            ("value", ~close),
            ("fixed", gf != c.fixed),
            ("lb", gl != c.lb),
            ("ub", gu != c.ub),
        ):
            if bad.any():
                cells = [name + str(c.keys[i]) for i in np.flatnonzero(bad)]
                msgs.append(f"{what}: {_few(cells)}")
    for name in sorted(set(mp.rows) & set(photo.rows)):
        r, c = mp.rows[name], photo.rows[name]
        if set(r.keys) != set(c.keys) or len(r.keys) != len(c.keys):
            msgs.append(
                f"{name}: row keys missing {_few(set(c.keys) - set(r.keys))}; "
                f"extra {_few(set(r.keys) - set(c.keys))}"
            )
            continue
        where = {k: i for i, k in enumerate(r.keys)}
        act = np.array([r.active[where[k]] for k in c.keys])
        if (act != c.active).any():
            pairs = zip(c.keys, act, c.active, strict=True)
            bad = [name + str(k) for k, a, w in pairs if a != w]
            msgs.append(f"active: {_few(bad)}")
    return msgs


@pytest.fixture(scope="module", params=DATASETS)
def case(request, tmp_path_factory):
    from pathlib import Path

    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.prep_capture import load

    ds = request.param
    load_params(ds)  # skip when the dataset is missing
    p = _params(ds)
    closure = _route_closure()
    rr = list(p.sets.r)[-1]
    root = (
        tmp_path_factory.mktemp("records")
        if ds == "gtap7_3x3"
        else Path.home() / ".cache" / "equilibria" / "family"
    )
    backend, state = load_or_build(p, closure, rr, root)
    cross = cross_period.load_or_record(
        p, closure, rr, root, sets=backend.sets, var_domains=backend.var_domains
    )
    photo = load(capture_dir(ds) / "001_A_base.pkl")
    return {
        "ds": ds,
        "params": p,
        "closure": closure,
        "rr": rr,
        "backend": backend,
        "state": state,
        "cross": cross,
        "photo": photo,
    }


def test_build_equals_photo_001_a_base(case):
    from equilibria.family.mp_arrays import MPArrays

    mp = MPArrays.build(
        case["state"], case["backend"], case["cross"], params=case["params"]
    )
    assert diff_to_photo(mp, case["photo"]) == []


def test_build_attributes(case):
    from equilibria.blocks.gtap import _derived_params as dp
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS
    from equilibria.family.mp_arrays import MPArrays

    p, state = case["params"], case["state"]
    mp = MPArrays.build(state, case["backend"], case["cross"], params=p)
    assert set(SHOCK_INSTRUMENTS) <= mp.exogenous <= set(mp.vars)
    assert mp.endogenous_instrument_cells == {}
    assert mp.targets == {}
    assert mp.shock_rows == frozenset()
    assert mp.residual_region == case["rr"]
    assert mp.fctts == dp.fctts_data(p, p.sets)
    assert mp.gtap_mode is False
    assert mp.base_calibrated is False
    assert mp.settled_seed is None
    # The attributes travel with copies.
    cp = mp.copy()
    assert cp.exogenous == mp.exogenous and cp.residual_region == mp.residual_region
    assert cp.vars["pa"].value is not mp.vars["pa"].value


def test_build_refuses_overwrite_hooks(case, monkeypatch):
    from equilibria.blocks.gtap.overwrite import overwrite
    from equilibria.family.mp_arrays import MPArrays

    monkeypatch.setattr(overwrite, "active", lambda: True)
    with pytest.raises(NotImplementedError, match="overwrite"):
        MPArrays.build(
            case["state"], case["backend"], case["cross"], params=case["params"]
        )


def test_sp_reference_from_composed_state_is_bitwise_todays(case):
    from equilibria.family.prep_recipe import sp_reference_arrays

    p, closure, rr = case["params"], case["closure"], case["rr"]
    want = sp_reference_arrays(p, closure, rr)
    got = sp_reference_arrays(p, closure, rr, state=case["state"])
    assert got.exogenous == want.exogenous and got.endogenous == want.endogenous
    assert list(got.vars) == list(want.vars)
    for name, w in want.vars.items():
        g = got.vars[name]
        assert g.keys == w.keys
        for attr in ("value", "fixed", "lb", "ub"):
            a, b = getattr(g, attr), getattr(w, attr)
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), (name, attr)
