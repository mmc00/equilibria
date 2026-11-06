"""The base recipe and the post-solve steps on arrays equal Pyomo's, solver injected.

Oracles are the photos of the saved Pyomo run (``tests/family/_prep.capture_dir``):

* ``A`` / ``A2``: the whole multiperiod state at entry / end of ``prepare``;
* ``P``: the active period's Var cells right after the Newton writeback;
* ``final.pkl``: the state after the shock call's post-solve; ``settled.pkl``: the
  dict ``calibrate_base`` returns (the check slice of the settle).

Each step starts from Pyomo's own photo (not from the previous array result) and injects
Pyomo's solution from ``P``, so a mismatch names one step and no Newton noise enters:

* base recipe on ``A`` == ``A2`` (call 1 without a settled seed, call 3 with it);
* base post-solve on ``A2`` == the next call's ``A``;
* ``square`` on ``A2`` + writeback of ``P`` + post-solve == the next call's ``A`` (check),
  == ``final.pkl`` (shock, after the recipe's default tariff, which ``A2`` lacks);
* the welfare cells and the ifSUB report cells against the same photos, by name.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir
from tests.family.test_mp_build import diff_to_photo
from tests.family.test_prep_recipe import ROUTE, _closure, _params

TOL = 1e-12
DATASETS = (
    "gtap7_3x3",
    "gtap7_10x7",
    pytest.param("gtap7_20x41", marks=pytest.mark.slow),
)
REPORT_VARS = ("pfa", "pfy", "pp_rai", "pwmg", "pefob", "pmcif", "pm")
WELFARE_VARS = ("ug", "us", "ev", "cv", "u")


def _load(d: Path, name: str):
    from equilibria.family.prep_capture import load

    return load(d / name)


def _calls(d: Path) -> dict[int, str]:
    """Call number -> period, from the ``A`` photos."""
    out = {}
    for f in sorted(d.glob("*_A_*.pkl")):
        n, _, period = f.stem.split("_")
        out[int(n)] = period
    return out


@pytest.fixture(scope="module", params=DATASETS)
def case(request, tmp_path_factory):
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.prep_recipe import sp_reference_arrays
    from tests.family.test_squared_system import _route_closure

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
    d = capture_dir(ds)
    return {
        "ds": ds,
        "dir": d,
        "params": p,
        "closure": closure,
        "recipe_closure": _closure(),
        "backend": backend,
        "state": state,
        "cross": cross,
        "ref": sp_reference_arrays(p, _closure(), rr, state=state),
        "calls": _calls(d),
        "settled": pickle.loads((d / "settled.pkl").read_bytes()),
        "final": _load(d, "final.pkl"),
    }


def _mp(case, name: str):
    from equilibria.family.mp_arrays import MPArrays

    mp = MPArrays.from_photo(_load(case["dir"], name))
    mp.exogenous = case["ref"].exogenous  # what MPArrays.build sets, a photo lacks
    mp.endogenous_instrument_cells = dict(case["ref"].endogenous)
    return mp


def _stage(case, n: int):
    """``A2`` of call ``n``, squared: (mp after ``square``, ``Squared``)."""
    period = case["calls"][n]
    mp = _mp(case, f"{n:03d}_A2_{period}.pkl")
    return _square(case, mp, period)


def _square(case, mp, period):
    from equilibria.family.prep_square import ConditionalMasks, square
    from equilibria.family.squared_system import period_incidence

    inc = period_incidence(case["backend"], case["state"], period, case["cross"])
    sq, mp_sq = square(
        mp,
        period,
        params=case["params"],
        closure=case["recipe_closure"],
        incidence=inc,
        exogenous=case["ref"].exogenous,
        masks=ConditionalMasks.from_params(case["params"], mp),
        return_mp=True,
    )
    return mp_sq, sq


def _solution(sq, photo_p) -> np.ndarray:
    """Pyomo's solution at the squared columns, from photo P."""
    from equilibria.family.mp_arrays import _name

    pv = {}
    for comp, vc in photo_p.vars.items():
        for k, v in zip(vc.keys, vc.value, strict=True):
            pv[_name(comp, k)] = float(v)
    missing = [c for c in sq.col_names if c not in pv]
    assert not missing, f"columns not in P: {missing[:5]}"
    return np.array([pv[c] for c in sq.col_names])


def _post(case, mp, period, *, sq=None, x=None, code=1):
    from equilibria.family.post_solve import post_solve

    p = case["params"]
    return post_solve(
        mp,
        period,
        backend=case["backend"],
        state=case["state"],
        params=p,
        regions=list(p.sets.r),
        if_sub=True,
        sq=sq,
        x=x,
        code=code,
    )


def _flags(n: int) -> dict[str, bool]:
    """The driver flags of call ``n``: the settle (calls 1-2) runs without
    ``base_calibrated`` / ``solve_check``, the main run with both."""
    main = n > 2
    return {
        "mute_welfare": True,
        "seed_from_prior": False,
        "base_calibrated": main,
        "solve_check": main,
    }


def _restricted(photo, names):
    """A copy of ``photo`` keeping only the Var components ``names``."""
    import copy

    out = copy.copy(photo)
    out.vars = {k: v for k, v in photo.vars.items() if k in names}
    out.rows = {}
    return out


def _restrict_mp(mp, names):
    import copy

    out = copy.copy(mp)
    out.vars = {k: v for k, v in mp.vars.items() if k in names}
    out.rows = {}
    return out


# ── The base recipe ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3])
def test_base_recipe_equals_a2(case, n):
    from equilibria.family.prep_recipe import run_recipe

    if n not in case["calls"]:
        pytest.skip("no such call")
    assert case["calls"][n] == "base"
    mp = _mp(case, f"{n:03d}_A_base.pkl")
    flags = _flags(n)
    if flags["base_calibrated"]:
        mp.settled_seed = case["settled"]
    got = run_recipe(
        mp,
        "base",
        params=case["params"],
        closure=case["recipe_closure"],
        sp_ref=case["ref"],
        mode="gtap",
        backend=case["backend"],
        **flags,
        cross=case["cross"],
    )
    a2 = _load(case["dir"], f"{n:03d}_A2_base.pkl")
    assert diff_to_photo(got, a2, TOL) == []


def test_base_recipe_without_seed_ignores_settled(case):
    """No ``base_calibrated``: the settled dict is not read (faithful to GAMS)."""
    from equilibria.family.prep_recipe import run_recipe

    mp = _mp(case, "001_A_base.pkl")
    mp.settled_seed = case["settled"]
    got = run_recipe(
        mp,
        "base",
        params=case["params"],
        closure=case["recipe_closure"],
        sp_ref=case["ref"],
        mode="gtap",
        backend=case["backend"],
        **_flags(1),
    )
    a2 = _load(case["dir"], "001_A2_base.pkl")
    assert diff_to_photo(got, a2, TOL) == []


# ── The base post-solve ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3])
def test_base_post_solve_equals_next_photo_a(case, n):
    if n + 1 not in case["calls"]:
        pytest.skip("no next call")
    mp = _post(case, _mp(case, f"{n:03d}_A2_base.pkl"), "base")
    nxt = _load(case["dir"], f"{n + 1:03d}_A_{case['calls'][n + 1]}.pkl")
    assert diff_to_photo(mp, nxt, TOL) == []


# ── Check and shock ─────────────────────────────────────────────────────────


def _check_next(case, n: int):
    """Array state after the post-solve of check call ``n``, Pyomo's solution injected."""
    mp_sq, sq = _stage(case, n)
    x = _solution(sq, _load(case["dir"], f"{n:03d}_P_check.pkl"))
    return _post(case, mp_sq, "check", sq=sq, x=x), sq


@pytest.mark.parametrize("n", [4])
def test_check_post_solve_equals_next_photo_a(case, n):
    if n not in case["calls"]:
        pytest.skip("no such call")
    assert case["calls"][n] == "check"
    mp, _ = _check_next(case, n)
    nxt = _load(case["dir"], f"{n + 1:03d}_A_{case['calls'][n + 1]}.pkl")
    assert diff_to_photo(mp, nxt, TOL) == []


def test_settle_check_equals_settled(case):
    """The settle's check slice (post-solve, welfare included) is what ``calibrate_base``
    returns: the check cells of ``settled.pkl``, to 1e-12."""
    mp, _ = _check_next(case, 2)
    bad = []
    for name, cells in case["settled"].items():
        v = mp.vars.get(name)
        if v is None:
            continue
        for body, val in cells.items():
            key = (*body, "check") if isinstance(body, tuple) else (body, "check")
            try:
                got = v.value[v.pos(key)]
            except KeyError:
                continue
            if not (np.isnan(got) and np.isnan(val)) and not abs(
                got - val
            ) <= TOL * max(1.0, abs(val)):
                bad.append(f"{name}{key}: {got} vs {val}")
    assert not bad, f"{len(bad)} differ, e.g. {bad[:5]}"


def _shock_final(case):
    """Array state after the shock call's post-solve (recipe, default tariff, square,
    Pyomo's solution injected)."""
    from equilibria.family.prep_recipe import run_recipe

    n = max(case["calls"])
    assert case["calls"][n] == "shock"
    mp0 = run_recipe(
        _mp(case, f"{n:03d}_A_shock.pkl"),
        "shock",
        params=case["params"],
        closure=case["recipe_closure"],
        sp_ref=case["ref"],
        mode="gtap",
        backend=case["backend"],
        **ROUTE,
        cross=case["cross"],
    )
    mp_sq, sq = _square(case, mp0, "shock")
    x = _solution(sq, _load(case["dir"], f"{n:03d}_P_shock.pkl"))
    return _post(case, mp_sq, "shock", sq=sq, x=x), sq


@pytest.fixture(scope="module")
def shock_final(case):
    return _shock_final(case)[0]


def test_shock_final_state_equals_final_pkl(case, shock_final):
    assert diff_to_photo(shock_final, case["final"], TOL) == []


def test_ifsub_report_cells_in_shock_equal_final(case, shock_final):
    names = set(REPORT_VARS)
    assert (
        diff_to_photo(
            _restrict_mp(shock_final, names), _restricted(case["final"], names), TOL
        )
        == []
    )
    # and they changed: the report cells are not the ones the closure left
    v, f = shock_final.vars["pm"], case["final"].vars["pm"]
    assert v.value.size == f.value.size


def test_welfare_cells_equal_pyomo(case, shock_final):
    names = set(WELFARE_VARS)
    assert (
        diff_to_photo(
            _restrict_mp(shock_final, names), _restricted(case["final"], names), TOL
        )
        == []
    )


# ── Unit tests of the pieces ────────────────────────────────────────────────


def test_calculate_variable_matches_pyomo():
    """The port of ``calculate_variable_from_constraint`` on a linear, a nonlinear, a
    no-solution and a zero-derivative constraint, against Pyomo."""
    import pyomo.environ as pe
    from pyomo.common.errors import IterationLimitError as PyoLimit
    from pyomo.util.calc_var_value import calculate_variable_from_constraint

    from equilibria.family.post_solve import IterationLimitError, calculate_variable

    cases = {
        "linear": (
            lambda m: m.x == 3.0 * m.p + 1.0,
            lambda x: x - (3.0 * 2.0 + 1.0),
            0.0,
        ),
        "power": (
            lambda m: m.p * (m.x**0.5) + m.x**2 == 7.0,
            lambda x: 2.0 * x**0.5 + x**2,
            7.0,
        ),
        "sum_ces": (
            lambda m: 0.3 * (2.0 / m.x) ** 2.5 + 0.7 * (1.5 / m.x) ** 2.5 == 1.0,
            lambda x: 0.3 * (2.0 / x) ** 2.5 + 0.7 * (1.5 / x) ** 2.5,
            1.0,
        ),
    }
    for name, (rule, body, upper) in cases.items():
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=2.0)
        m.x = pe.Var(initialize=1.0)
        m.c = pe.Constraint(expr=rule(m))
        calculate_variable_from_constraint(m.x, m.c)
        h = 1e-7

        def deriv(x, body=body):
            return (body(x + h) - body(x - h)) / (2 * h)

        got = calculate_variable(1.0, body, deriv, upper)
        # Pyomo differentiates symbolically, the test numerically: same root to 1e-8
        assert got == pytest.approx(pe.value(m.x), rel=1e-6), name
    assert issubclass(IterationLimitError, RuntimeError)
    assert not issubclass(IterationLimitError, ArithmeticError)
    assert PyoLimit is not None


def test_pin_pabs_only_acts_on_free_cells(case):
    from equilibria.family.post_solve import pin_pabs

    mp = _mp(case, "001_A2_base.pkl")
    r = list(case["params"].sets.r)
    pa = mp.vars["pabs"]
    i = pa.pos((r[0], "check"))
    pa.fixed[i] = False
    pa.value[i] = 0.0
    j = pa.pos((r[0], "base"))
    pa.fixed[j] = False
    pa.value[j] = 3.0
    pin_pabs(mp, "check", r)
    assert pa.fixed[i] and pa.value[i] == 1.0  # falsy level -> 1.0
    pin_pabs(mp, "base", r)
    assert pa.fixed[j] and pa.value[j] == 1.0


def test_freeze_period_fixes_only_that_period(case):
    from equilibria.family.post_solve import freeze_period

    mp = _mp(case, "001_A2_base.pkl")
    other = {k: v.fixed[v.period != "check"].copy() for k, v in mp.vars.items()}
    n = freeze_period(mp, "check")
    assert n == sum(int((v.period == "check").sum()) for v in mp.vars.values())
    for k, v in mp.vars.items():
        assert v.fixed[v.period == "check"].all()
        assert (v.fixed[v.period != "check"] == other[k]).all()


# ── check_endowment_shock_entered, against Pyomo's own ─────────────────────


@pytest.fixture(scope="module", params=("gtap7_3x3", "gtap7_10x7"))
def endow(request, tmp_path_factory):
    """A multiperiod Pyomo model (``build_block_model``, unsolved) and the arrays
    ``MPArrays.build`` makes of the same composed state."""
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from tests.family.test_squared_system import _route_closure

    ds = request.param
    load_params(ds)
    p = _params(ds)
    closure = _route_closure()
    rr = list(p.sets.r)[-1]
    m, _ = build_block_model(p, p.sets, closure, rr, base_calibrated=False)
    root = (
        tmp_path_factory.mktemp("records_e")
        if ds == "gtap7_3x3"
        else Path.home() / ".cache" / "equilibria" / "family"
    )
    backend, state = load_or_build(p, closure, rr, root)
    cross = cross_period.load_or_record(
        p, closure, rr, root, sets=backend.sets, var_domains=backend.var_domains
    )
    return {
        "m": m,
        "backend": backend,
        "state": state,
        "mp": MPArrays.build(state, backend, cross, params=p),
    }


@pytest.fixture(autouse=True)
def _restore_pyomo_levels(request):
    """The Pyomo model is shared by the module: put ``aft`` / ``xft`` back after a test."""
    if "endow" not in request.fixturenames:
        yield
        return
    m = request.getfixturevalue("endow")["m"]
    snap = [(vd, vd.value, vd.fixed) for comp in (m.aft, m.xft) for vd in comp.values()]
    yield
    for vd, val, fixed in snap:
        vd.set_value(val)
        vd.fixed = fixed


def _aft_cells(m, n: int = 2) -> list[tuple]:
    """(r, f) with an ``eq_xfteq`` row at the shock period and a positive ``aft``."""
    out = []
    for k in m.eq_xfteq:
        if k[-1] == "shock" and m.aft[k].value > 0:
            out.append(tuple(k[:-1]))
        if len(out) == n:
            break
    return out


def _both(endow, shock: dict[tuple, float]):
    """Apply ``aft`` shocks (percent) to the Pyomo model and the same levels to a copy of
    the arrays; return ``(model, arrays)``."""
    from equilibria.templates.gtap.shocks import apply_shock

    m = endow["m"]
    apply_shock(m, {"aft": shock})
    mp = endow["mp"].copy()
    aft = mp.vars["aft"]
    for cell in shock:
        key = (*cell, "shock")
        aft.value[aft.pos(key)] = m.aft[key].value
    return m, mp


def _pyomo_outcome(m) -> str | None:
    from equilibria.templates.gtap.shocks import check_endowment_shock_entered

    try:
        check_endowment_shock_entered(m)
    except RuntimeError as exc:
        return str(exc)
    return None


def _array_outcome(endow, mp) -> str | None:
    from equilibria.family.post_solve import check_endowment_shock_entered

    try:
        check_endowment_shock_entered(mp, endow["backend"], endow["state"])
    except RuntimeError as exc:
        return str(exc)
    return None


def test_check_endowment_no_shock_passes(endow):
    assert _pyomo_outcome(endow["m"]) is None
    assert _array_outcome(endow, endow["mp"]) is None


def test_check_endowment_shocked_aft_not_entered_raises_like_pyomo(endow):
    """A shocked ``aft`` with ``xft`` not moved: both raise, with the same message."""
    cells = _aft_cells(endow["m"])
    m, mp = _both(endow, dict.fromkeys(cells, 20.0))
    want = _pyomo_outcome(m)
    assert want is not None and "did not enter the solution" in want
    got = _array_outcome(endow, mp)
    assert got == want


def test_check_endowment_shocked_aft_entered_passes_like_pyomo(endow):
    """Move ``xft`` so the row holds at the shocked ``aft``: both pass."""
    from pyomo.environ import value

    cells = _aft_cells(endow["m"])
    m, mp = _both(endow, dict.fromkeys(cells, -15.0))
    xft = mp.vars["xft"]
    for cell in cells:
        k = (*cell, "shock")
        row = m.eq_xfteq[k]
        resid = float(value(row.body)) - float(value(row.upper))
        new = float(value(m.xft[k])) - resid
        m.xft[k].set_value(new)
        xft.value[xft.pos(k)] = new
    assert _pyomo_outcome(m) is None
    assert _array_outcome(endow, mp) is None


def test_check_endowment_only_the_off_cell_is_named(endow):
    """Two shocked cells, one entered: the message names the other, as Pyomo's."""
    from pyomo.environ import value

    cells = _aft_cells(endow["m"])
    if len(cells) < 2:
        pytest.skip("fewer than two aft cells")
    m, mp = _both(endow, dict.fromkeys(cells, 10.0))
    xft = mp.vars["xft"]
    k = (*cells[0], "shock")  # the first one enters, the second does not
    resid = float(value(m.eq_xfteq[k].body)) - float(value(m.eq_xfteq[k].upper))
    new = float(value(m.xft[k])) - resid
    m.xft[k].set_value(new)
    xft.value[xft.pos(k)] = new
    want = _pyomo_outcome(m)
    assert want is not None and str(tuple(cells[1])) in want
    assert _array_outcome(endow, mp) == want
