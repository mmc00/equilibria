"""A multi-instrument user shock gives the same result on the Pyomo and the family routes.

The shock (``imptx`` on two cells, ``aft``, ``pnum``, ``lambdava``) is applied with
``shocks.apply_shock`` to the Pyomo model before ``solve_multiperiod``, and passed as
``shock=`` to ``solve_gtap_family``. Every Var cell of the three periods agrees within
1e-9 (value, fixed flag and bounds as in ``diff_to_photo``), codes 1 on both.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pyomo")

from tests.family._prep import _newton_mumps  # noqa: E402


def _multishock(m) -> dict:
    """The shock of the test, from the cells of ``m`` that can take it."""
    from equilibria.templates.gtap.shocks import _can_take_shock

    def cells(name, positive=False):
        var = getattr(m, name, None)
        if var is None:
            return []
        return [
            tuple(k[:-1])
            for k in var
            if k[-1] == "shock"
            and (not positive or (var[k].value or 0) > 0)
            and _can_take_shock(m, name, tuple(k[:-1]))
        ]

    imptx, aft, lamb = cells("imptx"), cells("aft", True), cells("lambdava")
    shock: dict = {"imptx": {imptx[1]: 8.6637, imptx[-1]: 4.0}, "aft": {aft[0]: 10.0}}
    if _can_take_shock(m, "pnum", ()):
        shock["pnum"] = {(): 5.0}
    if lamb:
        shock["lambdava"] = {lamb[0]: 3.0}
    return shock


def _check(dataset: str, linsolve: str, monkeypatch, tmp_path) -> None:
    from equilibria.family import prep_capture
    from equilibria.family.driver import solve_gtap_family
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.shocks import apply_shock
    from tests.family._data import load_params
    from tests.family.test_mp_build import diff_to_photo
    from tests.family.test_squared_system import _route_closure
    from tests.templates.gtap.test_gtap7_gempack_parity import find_reference

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)

    def params():
        return copy.deepcopy(load_params(dataset))

    p = params()
    closure = _route_closure()
    rr = list(p.sets.r)[-1]
    ref = find_reference(
        dataset, "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    ) or find_reference(
        dataset, "gams", form="nlp", mode="gtap", ifsub=1, fbep="header"
    )
    gdx = None if ref is None else ref / "out.gdx"

    m, _ = build_block_model(p, p.sets, closure, rr, base_calibrated=True, ref_gdx=gdx)
    shock = _multishock(m)
    assert len(shock) >= 3  # several instruments, or the test says nothing
    apply_shock(m, shock)

    photo_dir = tmp_path / "photo"
    monkeypatch.setenv(prep_capture.ENV, str(photo_dir))
    prep_capture.reset()
    res = solve_multiperiod(
        m,
        p,
        closure,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
    )
    monkeypatch.delenv(prep_capture.ENV)
    assert {k: v.get("code") for k, v in res.items()} == {
        "base": 1,
        "check": 1,
        "shock": 1,
    }
    final = prep_capture.load(Path(photo_dir) / "final.pkl")

    fres = solve_gtap_family(
        params(),
        _route_closure(),
        rr,
        shock=shock,
        cache_root=tmp_path / "cache",
        jax_cache=False,
    )
    assert {k: v["code"] for k, v in fres.results.items()} == {
        "base": 1,
        "check": 1,
        "shock": 1,
    }
    assert diff_to_photo(fres.mp, final, 1e-9, lb_rtol=1e-9) == []


def test_multishock_matches_pyomo_3x3(monkeypatch, tmp_path):
    _check("gtap7_3x3", "direct", monkeypatch, tmp_path)


@pytest.mark.slow
def test_multishock_matches_pyomo_10x7(monkeypatch, tmp_path):
    if importlib.util.find_spec("mumps") is None:
        pytest.skip("pymumps not available")
    _check("gtap7_10x7", "mumps", monkeypatch, tmp_path)
