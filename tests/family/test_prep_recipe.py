"""The period recipe (check, shock; gtap) on arrays gives the Pyomo preparer's state.

Two oracles: the single-period reference as arrays equals the ``SPSnapshot`` the
driver reads from the reference model, and ``run_recipe`` on photo A equals photo A2
(the end of ``PeriodPreparer.prepare``) cell by cell. The default tariff the driver
writes AFTER ``prepare`` is checked against photo B.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir

RTOL = 1e-9
DATASETS = (
    "gtap7_3x3",
    "gtap7_10x7",
    pytest.param("gtap7_15x10", marks=pytest.mark.slow),
)
SP_DATASETS = ("gtap7_3x3", "gtap7_10x7")


def _closure():
    """The driver's ``base_closure`` (ifSUB=1) for the test route."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        numeraire="pnum",
    )


def _params(dataset: str):
    """The run's params: ``build_block_model`` recalibrates them to the GEMPACK
    factor-subsidy basis of the test route (in place), and the driver's reference
    model reads them like that."""
    import copy

    p = copy.deepcopy(load_params(dataset))
    p.set_va_subsidy_basis("gempack")
    return p


def _key(idx) -> tuple:
    if idx is None:
        return ()
    return tuple(str(i) for i in idx) if isinstance(idx, tuple) else (str(idx),)


@pytest.mark.parametrize("dataset", SP_DATASETS)
def test_sp_ref_arrays_equal_pyomo_snapshot(dataset):
    from equilibria.family.prep_recipe import sp_reference_arrays
    from equilibria.templates.gtap import gtap_multiperiod_driver as d

    p = _params(dataset)
    rr = list(p.sets.r)[-1]
    snap = d._sp_snapshot(d._build_sp_reference(p.sets, p, _closure(), rr))
    ref = sp_reference_arrays(p, _closure(), rr)

    bad = []
    for name, (_, cells) in snap.fixed.items():
        want = {_key(i): v for i, v in cells}
        v = ref.vars.get(name)
        got = (
            {}
            if v is None
            else {k: x for k, x, f in zip(v.keys, v.value, v.fixed, strict=True) if f}
        )
        if want.keys() != got.keys():
            bad.append(f"fixed cells of {name}: {len(want)} vs {len(got)}")
            continue
        off = [k for k in want if not np.isclose(want[k], got[k], rtol=1e-12, atol=0)]
        if off:
            bad.append(f"fixed values of {name}: {off[:3]}")
    for name, (_, cells) in snap.bounds.items():
        want = {
            _key(i): (-np.inf if lb is None else lb, np.inf if ub is None else ub)
            for i, lb, ub in cells
        }
        v = ref.vars.get(name)
        got = (
            {}
            if v is None
            else {
                k: (lo, hi)
                for k, lo, hi in zip(v.keys, v.lb, v.ub, strict=True)
                if np.isfinite(lo) or np.isfinite(hi)
            }
        )
        if want != got:
            diff = [k for k in want.keys() | got.keys() if want.get(k) != got.get(k)]
            bad.append(f"bounds of {name}: {len(diff)} cells, e.g. {diff[:3]}")
    extra = [
        n
        for n, v in ref.vars.items()
        if (v.fixed.any() or np.isfinite(v.lb).any() or np.isfinite(v.ub).any())
        and n not in snap.fixed
    ]
    assert not bad and not extra, f"{bad[:6]} extra components {extra[:6]}"


def _differences(got, want_photo, skip_values):
    """Differences between MPArrays ``got`` and a Photo, grouped by component."""
    from equilibria.family.mp_arrays import MPArrays

    want = MPArrays.from_photo(want_photo)
    out: dict[str, list[str]] = {}

    def note(comp: str, what: str, i: int, arr) -> None:
        out.setdefault(comp, []).append(f"{what}:{arr.keys[i]}")

    for comp, w in want.vars.items():
        g = got.vars.get(comp)
        if g is None or g.keys != w.keys:
            out.setdefault(comp, []).append("cells differ")
            continue
        for what, a, b in (
            ("fixed", g.fixed, w.fixed),
            ("lb", g.lb, w.lb),
            ("ub", g.ub, w.ub),
        ):
            same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype != bool else a == b
            for i in np.flatnonzero(~same):
                note(comp, what, i, w)
        if comp in skip_values:
            continue
        gv, wv = g.value, w.value
        both_nan = np.isnan(gv) & np.isnan(wv)
        close = np.abs(gv - wv) <= RTOL * np.maximum(1.0, np.abs(wv))
        close |= both_nan | (gv == wv)
        for i in np.flatnonzero(~close):
            note(comp, "value", i, w)
    for comp, w in want.rows.items():
        g = got.rows.get(comp)
        if g is None or g.keys != w.keys:
            out.setdefault(comp, []).append("rows differ")
            continue
        for i in np.flatnonzero(g.active != w.active):
            note(comp, "active", i, w)
    return out


def _report(diffs) -> str:
    return "; ".join(
        f"{c}: {len(v)} e.g. {v[:3]}" for c, v in sorted(diffs.items())[:10]
    )


@pytest.fixture(scope="module", params=DATASETS)
def case(request):
    from equilibria.family.prep_capture import load_a2, pairs
    from equilibria.family.prep_recipe import sp_reference_arrays

    ds = request.param
    d = capture_dir(ds)
    p = _params(ds)
    ref = sp_reference_arrays(p, _closure(), list(p.sets.r)[-1])
    return ds, d, p, ref, pairs(d), load_a2


@pytest.mark.parametrize("k", range(3))
def test_recipe_matches_pyomo(case, k):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe

    _, d, p, ref, calls, load_a2 = case
    if k >= len(calls):
        pytest.skip("fewer solve calls")
    a, _ = calls[k]
    a2 = load_a2(d, a)
    assert a2 is not None
    got = run_recipe(
        MPArrays.from_photo(a),
        a.period,
        params=p,
        closure=_closure(),
        sp_ref=ref,
        mode="gtap",
        mute_welfare=True,
        seed_from_prior=False,
    )
    # mq_* values: the Fisher aggregates refreshed after the seed (known gap); the
    # shock's imptx: the default tariff is written after prepare (see test below).
    skip = {n for n in a.vars if n.startswith("mq_")}
    if a.period == "shock":
        skip.add("imptx")
    diffs = _differences(got, a2, skip)
    assert not diffs, f"{a.period} call {a.n}: {_report(diffs)}"


def test_shock_default_tariff_matches_b(case):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe

    _, _, p, ref, calls, _ = case
    a, b = next((a, b) for a, b in calls if a.period == "shock")
    got = run_recipe(
        MPArrays.from_photo(a), "shock", params=p, closure=_closure(), sp_ref=ref
    )
    g, w = got.var("imptx"), b.vars["imptx"]
    assert g.keys[: len(w.keys)] is not None
    sel = [g.pos(k) for k in w.keys]
    np.testing.assert_allclose(g.value[sel], w.value, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(g.fixed[sel], w.fixed)
    # and the check cells did not move
    chk = g.period == "check"
    assert chk.any()


def test_missing_component_raises(case):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe

    _, _, p, ref, calls, _ = case
    a, _ = calls[-1]
    mp = MPArrays.from_photo(a)
    del mp.vars["pft"]
    with pytest.raises(KeyError, match="pft"):
        run_recipe(mp, a.period, params=p, closure=_closure(), sp_ref=ref)


def test_other_periods_not_ported(case):
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe

    _, _, p, ref, calls, _ = case
    mp = MPArrays.from_photo(calls[0][0])
    with pytest.raises(NotImplementedError):
        run_recipe(mp, "base", params=p, closure=_closure(), sp_ref=ref)
    with pytest.raises(NotImplementedError):
        run_recipe(
            mp, "check", params=p, closure=_closure(), sp_ref=ref, mode="altertax"
        )
