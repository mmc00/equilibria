"""``can_take_shock`` / ``apply_shock`` on the arrays equal Pyomo's on the same model state.

The oracle is a Pyomo ``build_block_model`` built here, taken through the stages a user
and the driver see it in:

* ``user``: as built, where a user calls ``shocks.apply_shock`` (before ``solve_multiperiod``);
* ``prepared``: after ``PeriodPreparer.prepare(m, "shock")``, where the driver writes the
  default tariff.

Accept / reject of ``check_shock_cell`` is compared cell by cell (every ``imptx`` cell,
a sample of the cells of each other instrument, plus a cell that does not exist). A
multi-instrument user shock goes through both ``apply_shock`` and the instrument cells of
all three periods are compared bitwise, after the write and after ``prepare``. The default
tariff of the shock recipe is compared with the saved Pyomo run's ``final.pkl``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import capture_dir
from tests.family.test_prep_recipe import ROUTE, _closure, _params
from tests.family.test_squared_system import _route_closure

# (dataset, ifSUB): ifSUB=0 only on 3x3, where it accepts the instruments that
# ifSUB=1 rejects (fcttx, kappaf, exptx, prdtx_rai, lambdamg). Under ifSUB=1 the array
# route also accepts prdtx_rai (family/shock.py INLINE_EQS: M_PP puts it into eq_peq),
# which Pyomo rejects: the one documented divergence (``ARRAYS_ACCEPT_MORE``).
DATASETS = (
    ("gtap7_3x3", 1),
    ("gtap7_10x7", 1),
    pytest.param(("gtap7_15x10", 1), marks=pytest.mark.slow),
    ("gtap7_3x3", 0),
)


def _ids(param) -> str:
    ds, ifsub = param
    return ds if ifsub == 1 else f"{ds}-ifsub{ifsub}"


def _closures(ifsub: int):
    """The model closure (``_route_closure``) and the driver's ``_closure`` at ``ifsub``."""
    if ifsub == 1:
        return _route_closure(), _closure()
    off = {"if_sub": False}
    return (
        _route_closure().model_copy(update=off),
        _closure().model_copy(update=off),
    )


SAMPLE = 300
# Shock of the multi-instrument case: (instrument, GEMPACK percent, cells taken).
USER = (
    ("imptx", 8.6637, 4),
    ("aft", 10.0, 2),
    ("pnum", 5.0, 1),
    ("lambdava", 3.0, 3),
    ("kappaf", 2.0, 2),
    ("fcttx", 4.0, 2),
    ("exptx", 6.0, 2),
)


def _keys(v) -> list[tuple]:
    return [k[:-1] for k in v.keys if k[-1] == "shock"]


def _pyomo_cells(m, name: str) -> list[tuple]:
    """Shock-period cells of a Pyomo instrument Var, without the period."""
    out = []
    for k in getattr(m, name):
        kt = k if isinstance(k, tuple) else (k,)
        if kt[-1] == "shock":
            out.append(tuple(kt[:-1]))
    return out


def _snapshot(m) -> dict[str, dict[tuple, tuple[float, bool]]]:
    """Value and fixed flag of every cell of every instrument Var of ``m``."""
    from pyomo.environ import value

    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    out = {}
    for name in SHOCK_INSTRUMENTS:
        var = getattr(m, name, None)
        if var is None:
            continue
        cells = {}
        for k in var:
            kt = k if isinstance(k, tuple) else (k,)
            cells[kt] = (float(value(var[k])), bool(var[k].fixed))
        out[name] = cells
    return out


def _arrays_snapshot(mp) -> dict[str, dict[tuple, tuple[float, bool]]]:
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

    out = {}
    for name in SHOCK_INSTRUMENTS:
        v = mp.vars.get(name)
        if v is None:
            continue
        out[name] = {
            k: (float(x), bool(f))
            for k, x, f in zip(v.keys, v.value, v.fixed, strict=True)
        }
    return out


def _diff(a: dict, b: dict) -> list[str]:
    """Cells whose value (bitwise) or fixed flag differ, by instrument."""
    bad = []
    for name in sorted(set(a) | set(b)):
        ca, cb = a.get(name, {}), b.get(name, {})
        if ca.keys() != cb.keys():
            bad.append(f"{name}: cell sets differ")
            continue
        off = [
            k
            for k in ca
            if not (
                (
                    ca[k][0] == cb[k][0]
                    or (ca[k][0] != ca[k][0] and cb[k][0] != cb[k][0])
                )
                and ca[k][1] == cb[k][1]
            )
        ]
        if off:
            bad.append(
                f"{name}: {len(off)} cells, e.g. {off[:3]} {ca[off[0]]} {cb[off[0]]}"
            )
    return bad


@pytest.fixture(scope="module", params=DATASETS, ids=_ids)
def case(request, tmp_path_factory):
    from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS
    from equilibria.family import cross_period
    from equilibria.family.disk_cache import load_or_build
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.shock import apply_shock, can_take_shock
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.period_prep import PeriodPreparer
    from equilibria.templates.gtap.shocks import _can_take_shock
    from equilibria.templates.gtap.shocks import apply_shock as pyomo_apply

    ds, ifsub = request.param
    load_params(ds)  # skip when the dataset is missing
    p = _params(ds)
    closure, base_closure = _closures(ifsub)
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
    sp_ref = sp_reference_arrays(p, base_closure, rr, state=state)
    mp0 = MPArrays.build(state, backend, cross, params=p)
    m: Any = build_block_model(p, p.sets, closure, rr, base_calibrated=False)[0]

    rng = random.Random(0)
    # (instrument, cell) probes: all imptx cells, a sample of the rest, a missing cell.
    probes: dict[str, list[tuple]] = {}
    for name in SHOCK_INSTRUMENTS:
        if name not in mp0.vars:
            continue
        cells = _keys(mp0.vars[name])
        if name != "imptx" and len(cells) > SAMPLE:
            cells = rng.sample(cells, SAMPLE)
        probes[name] = cells
    probes["imptx"] = probes["imptx"] + [("NOWHERE", "NOGOOD", "NOWHERE")]

    def pyomo_table():
        return {
            n: {c: _can_take_shock(m, n, c) for c in cells}
            for n, cells in probes.items()
        }

    def array_table(mp):
        return {
            n: {c: can_take_shock(mp, backend, n, c) for c in cells}
            for n, cells in probes.items()
        }

    out = {
        "ds": ds,
        "ifsub": ifsub,
        "base_closure": base_closure,
        "params": p,
        "closure": closure,
        "rr": rr,
        "backend": backend,
        "state": state,
        "cross": cross,
        "sp_ref": sp_ref,
        "mp0": mp0,
        "probes": probes,
    }
    # stage 1: the model as a user finds it
    out["user_pyomo"] = pyomo_table()
    out["user_arrays"] = array_table(mp0)
    # the user shock, from cells both accept
    shock = {}
    for name, pct, n in USER:
        ok = [c for c in probes.get(name, []) if out["user_pyomo"][name].get(c)]
        if name == "pnum":
            ok = [()] if out["user_pyomo"].get(name, {}).get((), False) else []
        if ok:
            step = max(1, len(ok) // n)
            shock[name] = dict.fromkeys(ok[::step][:n], pct)
    out["shock"] = shock
    out["pyomo_levels"] = pyomo_apply(m, shock)
    out["array_levels"] = apply_shock(mp0.copy(), backend, shock)
    mp_sh = mp0.copy()
    apply_shock(mp_sh, backend, shock)
    out["user_snap_pyomo"] = _snapshot(m)
    out["user_snap_arrays"] = _arrays_snapshot(mp_sh)
    # stage 2: after the driver's prepare of the shock period
    prep = PeriodPreparer(
        mode="gtap",
        base_closure=base_closure,
        alt_closure=base_closure,
        residual_region=rr,
        **ROUTE,
    )
    prep.prepare(m, "shock", p)
    out["prep_pyomo"] = pyomo_table()
    out["prep_snap_pyomo"] = _snapshot(m)

    def recipe(shock_arg):
        return run_recipe(
            mp0,
            "shock",
            params=p,
            closure=base_closure,
            sp_ref=sp_ref,
            mode="gtap",
            backend=backend,
            shock=shock_arg,
            **ROUTE,
        )

    got = recipe(shock)
    out["prep_snap_arrays"] = _arrays_snapshot(got)
    plain = recipe(None)  # default tariff
    out["prep_arrays"] = array_table(plain)
    # stage 3: switch rows off / fix xft on both sides so that real cells get rejected
    stress = {}
    acc = out["user_pyomo"]
    for eq, name, step in (
        ("eq_xweq", "imptx", 7),
        ("eq_xfteq", "aft", 3),
        ("eq_va", "lambdava", 4),
    ):
        cells = [c for c in probes[name] if acc[name].get(c)][::step]
        stress[name] = cells
        row = plain.rows[eq]
        tab = {k[:-1]: i for i, k in enumerate(row.keys) if k[-1] == "shock"}
        for c in cells:
            getattr(m, eq)[(*c, "shock")].deactivate()
            row.active[tab[c]] = False
    xft = plain.vars["xft"]
    for c in [c for c in probes["aft"] if acc["aft"].get(c)][1::3]:
        m.xft[(*c, "shock")].fix()  # ty: ignore[not-subscriptable]
        xft.fixed[xft.pos((*c, "shock"))] = True
        stress["aft"].append(c)
    out["stress_cells"] = stress
    out["stress_pyomo"] = pyomo_table()
    out["stress_arrays"] = array_table(plain)
    return out


# Instruments the arrays accept under ifSUB=1 where Pyomo rejects wholesale (ALTERTAX).
ARRAYS_ACCEPT_MORE = frozenset({"prdtx_rai"})


def _compare_table(
    a: dict, b: dict, ifsub: int = 0
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    bad, counts = [], {}
    for name in a:
        if ifsub == 1 and name in ARRAYS_ACCEPT_MORE:
            # documented divergence: arrays may accept what Pyomo rejects, never the
            # reverse, and must accept something
            counts[name] = (sum(b[name].values()), len(b[name]) - sum(b[name].values()))
            bad += [
                f"{name}{c}: pyomo accepts, arrays reject"
                for c in a[name]
                if a[name][c] and not b[name][c]
            ]
            if not any(a[name].values()) and not any(b[name].values()):
                bad.append(f"{name}: arrays accept no cell under ifSUB=1")
            continue
        ok = sum(a[name].values())
        counts[name] = (ok, len(a[name]) - ok)
        bad += [
            f"{name}{c}: pyomo {a[name][c]} arrays {b[name][c]}"
            for c in a[name]
            if a[name][c] != b[name][c]
        ]
    return bad, counts


@pytest.mark.parametrize("stage", ["user", "prep"])
def test_accept_reject_equals_pyomo(case, stage, capsys):
    pyo = case[f"{stage}_pyomo"]
    arr = case["user_arrays" if stage == "user" else "prep_arrays"]
    bad, counts = _compare_table(pyo, arr, case["ifsub"])
    with capsys.disabled():
        print(
            f"\n[{case['ds']} ifsub={case['ifsub']} {stage}] accepted/rejected per instrument:",
            counts,
        )
    assert not bad, bad[:8]
    assert counts["imptx"][0] > 0, "imptx: nothing accepted, the comparison is vacuous"
    assert counts["imptx"][1] > 0, "imptx: nothing rejected (the missing cell)"


def test_arrays_accept_prdtx_rai_under_ifsub1_pyomo_rejects(case):
    """ALTERTAX shocks ``to`` = ``prdtx_rai`` under ifSUB=1: the arrays accept the cells
    whose ``eq_peq`` is live, Pyomo rejects them all (unchanged)."""
    if case["ifsub"] != 1:
        pytest.skip("ifSUB=0: prdtx_rai is live in both routes")
    assert not any(case["user_pyomo"]["prdtx_rai"].values())
    assert any(case["user_arrays"]["prdtx_rai"].values())


def test_ifsub0_accepts_and_rejects_the_ifsub_instruments(case):
    """Under ifSUB=0 the instruments ifSUB=1 rejects wholesale are live, with real
    rejected cells, and the arrays agree with Pyomo on every one of them."""
    if case["ifsub"] != 0:
        pytest.skip("ifSUB=1: Pyomo rejects these instruments outright")
    bad, counts = _compare_table(case["user_pyomo"], case["user_arrays"])
    assert not bad, bad[:8]
    for name in ("fcttx", "kappaf", "exptx", "prdtx_rai", "lambdamg"):
        assert counts[name][0] > 0, f"{name}: nothing accepted under ifSUB=0"
    for name in ("fcttx", "kappaf", "prdtx_rai", "lambdamg"):
        assert counts[name][1] > 0, f"{name}: nothing rejected under ifSUB=0"
    assert {"fcttx", "kappaf", "exptx"} <= set(case["shock"])


def test_accept_reject_equals_pyomo_with_rows_off(case, capsys):
    """Real cells rejected: rows of the inline path off, ``xft`` fixed, ``eq_va`` off."""
    bad, counts = _compare_table(
        case["stress_pyomo"], case["stress_arrays"], case["ifsub"]
    )
    with capsys.disabled():
        print(
            f"\n[{case['ds']} ifsub={case['ifsub']} stress] accepted/rejected:",
            {n: counts[n] for n in ("imptx", "aft", "lambdava")},
        )
    assert not bad, bad[:8]
    # under ifSUB=0 the eq_xweq rows do not read imptx: only the missing cell is rejected
    assert counts["imptx"][1] > (1 if case["ifsub"] == 1 else 0)
    assert counts["aft"][1] > 3 and counts["lambdava"][1] > 0


def test_user_shock_levels_equal_pyomo(case):
    assert case["shock"], "no instrument could take the multi-instrument shock"
    assert {"imptx", "aft", "pnum"} <= set(case["shock"])
    assert case["array_levels"] == case["pyomo_levels"]


def test_user_shock_cells_bitwise_equal_pyomo(case):
    bad = _diff(case["user_snap_pyomo"], case["user_snap_arrays"])
    assert not bad, bad


def test_user_shock_survives_prepare_like_pyomo(case):
    bad = _diff(case["prep_snap_pyomo"], case["prep_snap_arrays"])
    assert not bad, bad


def test_rejections_raise_like_pyomo(case):
    from equilibria.family.shock import apply_shock

    mp = case["mp0"].copy()
    be = case["backend"]
    first = next(iter(case["shock"]["imptx"]))
    for bad in (
        {"nope": {first: 1.0}},
        {"imptx": {("NOWHERE", "NOGOOD", "NOWHERE"): 1.0}},
        {"imptx": {first: -100.0}},
        {"imptx": {first: 0.0}},
    ):
        with pytest.raises(ValueError) as e:
            apply_shock(mp, be, bad)
        assert e.value.args  # message from the same checks as Pyomo's
    # all or nothing: a bad cell after a good one writes nothing
    before = mp.vars["imptx"].value.copy()
    with pytest.raises(ValueError):
        apply_shock(
            mp, be, {"imptx": {first: 5.0, ("NOWHERE", "NOGOOD", "NOWHERE"): 1.0}}
        )
    np.testing.assert_array_equal(mp.vars["imptx"].value, before)


def test_default_tariff_equals_final_pkl(case):
    """The recipe's default tariff fixes the cells and values of the Pyomo run."""
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import load, pairs
    from equilibria.family.prep_recipe import run_recipe

    d = capture_dir(case["ds"], ifsub=case["ifsub"])
    final = load(d / "final.pkl")
    a, _ = next((a, b) for a, b in pairs(d) if a.period == "shock")
    got = run_recipe(
        MPArrays.from_photo(a),
        "shock",
        params=case["params"],
        closure=case["base_closure"],
        sp_ref=case["sp_ref"],
        mode="gtap",
        backend=case["backend"],
        **ROUTE,
    )
    g = got.var("imptx")
    w = final.vars["imptx"]
    shock = np.array([k[-1] == "shock" for k in w.keys])
    check = {k[:-1]: i for i, k in enumerate(w.keys) if k[-1] == "check"}
    changed = {
        w.keys[i][:-1]
        for i in np.flatnonzero(shock)
        if w.value[i] != w.value[check[w.keys[i][:-1]]]
    }
    mine = {
        g.keys[i][:-1]
        for i in np.flatnonzero(g.period == "shock")
        if g.value[i] != g.value[g.pos((*g.keys[i][:-1], "check"))]
    }
    assert mine == changed and mine
    sel = [g.pos(k) for k in w.keys]
    np.testing.assert_allclose(g.value[sel], w.value, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(g.fixed[sel], w.fixed)
