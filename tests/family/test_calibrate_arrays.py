"""``calibrate_base_arrays``: the settle solve on a copy of the arrays gives the dict
``FactorBlock.calibrate_base`` returns.

The oracle is the saved Pyomo run's ``settled.pkl`` (``tests/family/_prep.capture_dir``;
3x3 is captured with the same ``linsolve`` as the family run, since the stopping point of
the Newton loop depends on its trajectory). Gate: same keys, every value within 1e-9
(``|a - b| <= 1e-9 * max(1, |b|)``, NaN equals NaN) and the caller's arrays bitwise
unchanged. The iteration count is reported, never asserted.
"""

from __future__ import annotations

import importlib.util
import os
import pickle

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import _newton_mumps, capture_dir
from tests.family.test_prep_recipe import _params
from tests.family.test_squared_system import _route_closure

RTOL = 1e-9
DATASETS = ("gtap7_3x3", "gtap7_10x7")


def _linsolve(ds: str) -> str:
    """The saved references of the larger datasets were captured with MUMPS and a family
    run only matches a reference captured with its own linsolve; 3x3 is captured here."""
    if importlib.util.find_spec("mumps") is None:
        if ds != "gtap7_3x3":
            pytest.skip(f"{ds}: the saved reference needs pymumps")
        return "direct"
    return "mumps"


def diff_settled(got: dict, want: dict, rtol: float = RTOL) -> list[str]:
    """Differences between two ``{var: {body: value}}`` dicts (empty = equal)."""
    msgs = []
    if set(got) != set(want):
        msgs.append(
            f"vars: missing {sorted(set(want) - set(got))[:5]}; "
            f"extra {sorted(set(got) - set(want))[:5]}"
        )
    for name in sorted(set(got) & set(want)):
        g, w = got[name], want[name]
        if set(g) != set(w):
            msgs.append(
                f"{name}: keys missing {sorted(set(w) - set(g), key=str)[:3]}; "
                f"extra {sorted(set(g) - set(w), key=str)[:3]}"
            )
            continue
        bad = []
        for k, b in w.items():
            a = g[k]
            same = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= rtol * max(
                1.0, abs(b)
            )
            if not same:
                bad.append(f"{name}[{k}] {a} vs {b}")
        if bad:
            msgs.append(f"{len(bad)} cells differ: {bad[:3]}")
    return msgs


@pytest.fixture(scope="module", params=DATASETS)
def case(request, monkeypatch_module, tmp_path_factory):
    from equilibria.family.driver import prepare_run

    ds = request.param
    load_params(ds)  # skip when the dataset is missing
    linsolve = _linsolve(ds)
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch_module.setenv(k, v)
    d = capture_dir(ds, linsolve)
    root = (
        tmp_path_factory.mktemp("records")
        if ds == "gtap7_3x3"
        else None  # the user's cache: 10x7 records take minutes to build
    )
    p = _params(ds)
    run, mp = prepare_run(
        p, _route_closure(), list(p.sets.r)[-1], cache_root=root, jax_cache=False
    )
    return {
        "ds": ds,
        "run": run,
        "mp": mp,
        "want": pickle.loads((d / "settled.pkl").read_bytes()),
    }


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def _snapshot(mp):
    return (
        {
            n: (v.value.copy(), v.fixed.copy(), v.lb.copy(), v.ub.copy())
            for n, v in mp.vars.items()
        },
        {n: r.active.copy() for n, r in mp.rows.items()},
        mp.settled_seed,
        mp.base_calibrated,
    )


@pytest.fixture(scope="module")
def calibrated(case):
    from equilibria.family.driver import calibrate_base_arrays

    before = _snapshot(case["mp"])
    settled, results = calibrate_base_arrays(case["mp"], case["run"])
    return settled, results, before


def test_settled_equals_pyomo(case, calibrated):
    settled, results, _ = calibrated
    print(f"{case['ds']} settle: {results}")
    assert results["check"]["code"] == 1
    assert diff_settled(settled, case["want"]) == []


def test_settled_has_the_calibrate_base_shape(case, calibrated):
    settled, _, _ = calibrated
    want = case["want"]
    assert set(settled) == set(want)
    for name, cells in want.items():
        assert all(
            isinstance(k, tuple) == isinstance(next(iter(settled[name])), tuple)
            for k in cells
        )
        assert set(settled[name]) == set(cells)
    assert all(isinstance(v, float) for c in settled.values() for v in c.values())


def test_callers_arrays_are_untouched(case, calibrated):
    _, _, before = calibrated
    mp = case["mp"]
    vars_, rows, seed, base_cal = before
    for n, (value, fixed, lb, ub) in vars_.items():
        v = mp.vars[n]
        assert v.value.tobytes() == value.tobytes(), n
        assert (v.fixed == fixed).all() and (v.lb == lb).all() and (v.ub == ub).all()
    for n, active in rows.items():
        assert (mp.rows[n].active == active).all(), n
    assert mp.settled_seed is seed
    assert mp.base_calibrated is base_cal
