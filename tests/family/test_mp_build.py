"""``MPArrays.build`` (the multiperiod state from the composed state) equals the Pyomo model's.

Oracle: photo ``001_A_base`` of the saved Pyomo run (frozen), the whole multiperiod state
at entry of the first period prepare. ``calibrate_base`` runs the settle on the model
``build_block_model`` just built, and its first prepare is that call: nothing touched
the state between the reflection (plus the Fisher rows) and the photo except the old
driver stamping ``_residual_region`` / ``_gtap_mode`` (attributes, not cells).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir
from tests.family.test_sp_arrays import _params
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


def diff_to_photo(mp, photo, rtol: float = RTOL, lb_rtol: float = 0.0) -> list[str]:
    """Differences between ``mp`` and a photo (empty = equal): names, keys, values within
    ``rtol`` (NaN equals NaN), fixed flags, bounds (lower bounds within ``lb_rtol`` relative,
    exact by default), row active flags."""
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
            (
                "lb",
                (gl != c.lb)
                & ~(
                    np.isfinite(gl)
                    & np.isfinite(c.lb)
                    & (
                        np.abs(
                            np.where(
                                np.isfinite(gl) & np.isfinite(c.lb), gl - c.lb, 0.0
                            )
                        )
                        <= lb_rtol * np.abs(c.lb)
                    )
                ),
            ),
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


def _nus333_mp(tmp_path):
    """``prepare_run``'s arrays for nus333 under the TBL65B closure (no cache key
    under hooks, so nothing is shared across these tests)."""
    from equilibria.family.run_setup import prepare_run
    from tests.templates.gtap._nus333 import closure, nus333_params

    _, mp = prepare_run(
        nus333_params(), closure(), "ROW", cache_root=tmp_path, jax_cache=False
    )
    return mp


def _active(row) -> set:
    return {row.keys[i] for i in np.flatnonzero(row.active)}


def _fixed(var, cell: tuple) -> dict:
    return {k[-1]: bool(var.fixed[i]) for i, k in enumerate(var.keys) if k[:-1] == cell}


def test_build_keeps_hook_rows_in_the_shock_only(tmp_path):
    """TBL65B's hooks rule only in the shock, as on Pyomo (``shock_only``): the real
    wage row is active at the shock alone and ``aft[USA,LABOR]`` is free there only."""
    from typing import Any, cast

    from equilibria.blocks.gtap.overwrite import overwrite
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from tests.templates.gtap._nus333 import closure, nus333_params
    from tests.templates.gtap._unemployment import register_unemployment_hooks

    register_unemployment_hooks()
    try:
        mp = _nus333_mp(tmp_path)
        p = nus333_params()
        m, _ = cast(Any, build_block_model(p, p.sets, closure(), "ROW"))
    finally:
        overwrite.clear()
    assert _active(mp.rows["eq_wreal"]) == {("USA", "LABOR", "shock")}
    assert _active(mp.rows["eq_wreal"]) == {
        tuple(k) for k in m.eq_wreal if m.eq_wreal[k].active
    }
    want = {t: bool(m.aft["USA", "LABOR", t].fixed) for t in ("base", "check", "shock")}
    assert want == {"base": True, "check": True, "shock": False}
    assert _fixed(mp.vars["aft"], ("USA", "LABOR")) == want


def test_build_after_clear_has_no_hook_rows(tmp_path):
    """Hooks of one exercise do not reach the next build after ``overwrite.clear()``."""
    from equilibria.blocks.gtap.overwrite import overwrite
    from tests.templates.gtap._unemployment import register_unemployment_hooks

    register_unemployment_hooks()
    try:
        assert "eq_wreal" in _nus333_mp(tmp_path / "a").rows
    finally:
        overwrite.clear()
    mp = _nus333_mp(tmp_path / "b")
    assert "eq_wreal" not in mp.rows
    assert set(_fixed(mp.vars["aft"], ("USA", "LABOR")).values()) == {True}


def test_sp_reference_from_composed_state_is_bitwise_todays(case):
    from equilibria.family.sp_arrays import sp_reference_arrays

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


def test_activating_a_period_keeps_hook_rows_out_of_base_and_check(tmp_path):
    """Turning a period's rows on (``activate_rows``, the recipe's
    ``freeze_inactive``) never turns a hook row on in base or check: those cells do
    not exist, as Pyomo's ``shock_only`` deleted them."""
    from equilibria.blocks.gtap.overwrite import overwrite
    from equilibria.family.periods import FamilyPeriodModel
    from tests.templates.gtap._unemployment import register_unemployment_hooks

    register_unemployment_hooks()
    try:
        mp = _nus333_mp(tmp_path)
    finally:
        overwrite.clear()
    model = FamilyPeriodModel(mp)
    for period, want in (
        ("base", set()),
        ("check", set()),
        ("shock", {("USA", "LABOR", "shock")}),
    ):
        model.activate_rows(period)
        assert _active(mp.rows["eq_wreal"]) == want, period
        model.set_all_rows_active("eq_wreal", period, True)
        assert _active(mp.rows["eq_wreal"]) == want, period
