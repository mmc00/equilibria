"""Path-integral welfare decomposition (Phase C, controller ruling 2026-11-09).

``decomp.py``'s one-step integral (``COEF(pre) * q%`` with ``pre`` the check) is exact
only to first order in the shock. For the 10 % RunGTAP references the shock is solved in
K equal sub-steps along GEMPACK's path (the power of the instrument compounds: each step
is ``(1 + p/100)**(1/K)``), each step's contributions use the previous step as ``pre``,
and the K-step sums are the decomposition. K doubles until the EV of every region changes
by less than ``TOL_CHANGE / 10`` between K and 2K; the reported decomposition is the
Richardson extrapolation ``2 D(K) - D(K/2)`` and the stop rule looks at every gated total."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.family.altertax import gates
from equilibria.family.altertax.decomp import DECOMP_HEADERS, decomp_headers
from equilibria.family.altertax.path import (
    PATH_TOL,
    one_step_dvol,
    pick_steps,
    richardson,
    shock_fraction,
)
from equilibria.family.altertax.shocks import load_tms_shock
from equilibria.family.altertax.tolerances import TOL_CHANGE
from tests.family._prep import ROOT

DATA = ROOT / "data/gtap7"
REF = "gempack_altertax-rungtap_tms-rate10_s8-16-32"


def folder(ds):
    return DATA / ds / REF


# --- pure ----------------------------------------------------------------------------


def test_path_tolerance_is_a_tenth_of_tol_change():
    assert PATH_TOL == TOL_CHANGE / 10


def test_shock_fraction_compounds_the_power():
    """GEMPACK's multistep path for a percent-change shock: K equal steps whose powers
    compound to the full shock (``(1 + p_t/100) = (1 + p/100)**t``)."""
    full = {"imptx": {("a", "b", "c"): 10.0, ("d", "e", "f"): -3.0}}
    half = shock_fraction(full, 0.5)
    for cell, p in full["imptx"].items():
        h = half["imptx"][cell]
        assert (1 + h / 100) ** 2 == pytest.approx(1 + p / 100, rel=1e-14)
    assert shock_fraction(full, 1.0) == full
    with pytest.raises(ValueError, match="fraction"):
        shock_fraction(full, 0.0)


def test_richardson_is_exact_for_a_first_order_path():
    """D(K) = a + b / K: R(K) = 2 D(K) - D(K/2) = a, header by header."""
    from equilibria.babel.har.symbols import HeaderArray

    a, b = np.array([100.0, -50.0]), np.array([4.0, 1.0])

    def ha(k):
        return {
            "A": HeaderArray(
                name="A", coeff_name="A", long_name="", array=a + b / k,
                set_names=["REG"], set_elements=[["x", "y"]],
            )
        }  # fmt: skip

    r = richardson(ha(8), ha(4))
    np.testing.assert_allclose(r["A"].array, a, rtol=1e-14)


def test_dvol_comes_from_the_one_step_decomposition():
    """``dvol`` is GEMPACK's ``BASE * cumulative %`` (a coefficient times a percent, not a
    change variable): the path keeps the one-step value in every COL header's ``dvol``
    column and A231's, and the summed/extrapolated value everywhere else (welcnt)."""
    from equilibria.babel.har.symbols import HeaderArray

    def ha(name, shape, cols, fill):
        arr = np.full(shape, fill, dtype=np.float64)
        return HeaderArray(
            name=name, coeff_name=name, long_name="", array=arr,
            set_names=["X"] * (len(shape) - 1) + [cols], set_elements=[],
        )  # fmt: skip

    path = {
        "A22P": ha("A22P", (2, 3, 4), "COL", 7.0),
        "A231": ha("A231", (2, 2, 2, 4, 2), "DIREC", 7.0),
        "A": ha("A", (2, 5), "COLUMN", 7.0),
    }
    one = {
        "A22P": ha("A22P", (2, 3, 4), "COL", 1.0),
        "A231": ha("A231", (2, 2, 2, 4, 2), "DIREC", 1.0),
        "A": ha("A", (2, 5), "COLUMN", 1.0),
    }
    out = one_step_dvol(path, one)
    a22p, a231 = np.asarray(out["A22P"].array), np.asarray(out["A231"].array)
    assert (a22p[..., 1] == 1.0).all() and (a22p[..., [0, 2, 3]] == 7.0).all()
    assert (a231[..., 1, :] == 1.0).all() and (a231[..., [0, 2, 3], :] == 7.0).all()
    assert (np.asarray(out["A"].array) == 7.0).all()


def _synthetic_steps(k):
    """``k`` per-step decompositions of every DECOMP header (sizes 2, set sizes of the
    fixed sets): step ``j`` holds ``j + 1`` everywhere, so a summed cell is
    ``k (k + 1) / 2`` and a first/last-step cell is 1 / ``k``."""
    from equilibria.babel.har.symbols import HeaderArray
    from equilibria.family.altertax.decomp import _META

    size = {"COL": 4, "COLUMN": 7, "PRICES": 3, "FORM": 2, "COLM": 3, "DIR": 2,
            "DIREC": 2, "TYPE": 2, "CTAX": 9, "ACTIVITY": 4, "TECHTYPE": 7}  # fmt: skip
    out = []
    for j in range(k):
        step = {}
        for name, coeff, long_name, names in _META:
            shape = tuple(size.get(s, 2) for s in names)
            step[name] = HeaderArray(
                name=name, coeff_name=coeff, long_name=long_name,
                array=np.full(shape, j + 1.0), set_names=list(names), set_elements=[],
            )  # fmt: skip
        out.append(step)
    return out


@pytest.mark.parametrize("name", ["A211", "A22P", "A231"])
def test_rate_columns_do_not_depend_on_k(name):
    """``taxrateb`` is the rate at the first step's ``pre`` and ``taxrateu`` at the last
    step's ``post`` wherever the COL axis sits: A231 ends in DIREC (COL is the axis
    before), and its rate cells were summed over the K steps (they grew with K)."""
    from equilibria.family.altertax.path import sum_steps

    for k in (2, 4):
        arr = np.asarray(sum_steps(_synthetic_steps(k))[name].array)
        ax = list(_synthetic_steps(1)[0][name].set_names).index("COL")
        col = np.moveaxis(arr, ax, 0)
        assert (col[2] == 1.0).all(), (k, "taxrateb")
        assert (col[3] == float(k)).all(), (k, "taxrateu")
        assert (col[0] == k * (k + 1) / 2).all(), (k, "welcnt is summed")


def test_pick_steps_stops_at_the_first_pair_when_r_is_exact():
    """D(K) = a + b / K gives R(K) = a at every K >= 2, so the first pair of R's is
    (R(2), R(4)): K = 4 where the EV-only rule needed K = 32."""
    a, b = {"ev": 100.0, "t": -50.0}, {"ev": 4.0, "t": 1.0}
    seen = []

    def totals_at(k):
        seen.append(k)
        return {n: a[n] + b[n] / k for n in a}

    k, hist = pick_steps(totals_at, tol=PATH_TOL, k_max=256)
    assert seen == [1, 2, 4] and k == 4
    assert hist[1]["change"] is None and hist[2]["change"] is None
    assert hist[4]["change"] < 1e-14


def test_pick_steps_keeps_doubling_while_a_non_ev_total_has_not_settled():
    """The EV is exact from K = 2 but another gated total carries a 1/K**2 term
    (R has a 1/K**2 error left): the rule waits for it, not only for the EV."""
    b = 5.0

    def totals_at(k):
        return {"ev": 100.0 + 4.0 / k, "other": 10.0 + b / k**2}

    k, hist = pick_steps(totals_at, tol=PATH_TOL, k_max=256)
    # R(K) = 10 - 2 b / K**2: the change K/2 -> K is 6 b / K**2 / 10 < 2e-3 -> K = 64
    assert k == 64 and hist[32]["change"] > PATH_TOL > hist[64]["change"]
    ev_only = pick_steps(lambda k: {"ev": 100.0 + 4.0 / k}, tol=PATH_TOL)[0]
    assert ev_only == 4 < k


def test_pick_steps_ignores_totals_below_the_noise_floor():
    def totals_at(k):
        return {"ev": 100.0, "tiny": 1e-4 * (1 + k**-0.5)}

    assert pick_steps(totals_at, floor_at=lambda k: 1e-2)[0] == 4
    with pytest.raises(RuntimeError, match="did not settle"):
        pick_steps(totals_at, k_max=16)


def test_pick_steps_refuses_to_stop_unconverged():
    with pytest.raises(RuntimeError, match="did not settle"):
        pick_steps(lambda k: {"t": 1.0 + 1.0 / k**0.5}, tol=PATH_TOL, k_max=8)


# --- solves ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from tests.family._prep import _newton_mumps

    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    yield tmp_path_factory
    mp.undo()


def _rungtap(env, ds, steps):
    pytest.importorskip("pyomo")
    from equilibria.family.altertax.run import solve_altertax

    return solve_altertax(
        DATA / ds,
        prm=folder(ds) / "parameters.prm",
        shock=load_tms_shock(folder(ds) / "tms_rate10.shk"),
        trade_balance="ratio",
        cache_root=env.mktemp(f"c_{ds}"),
        decomp_steps=steps,
    )


def test_one_step_default_keeps_no_path(altax_run):
    """The Burfisher command's default: one-step decomposition, no path solved."""
    assert altax_run.decomp_path is None


def test_path_matches_one_step_on_the_tiny_burfisher_shock(env):
    """On ALTAX's 0.0012 % output-tax shock the path effect is second order: the 4-step
    path integral's EV equals the one-step one (measured 1.8e-5) and every regional total
    is within PATH_TOL of it (measured 1.1e-3, A21 USA: the shocked tax's own revenue
    moves ~1 % along the path because the base rate is small). The path is the closer of
    the two to GEMPACK (worst total 4.0e-4 vs 1.5e-3, EV 4.2e-6 vs 2.3e-5) and passes
    the Burfisher gate."""
    pytest.importorskip("pyomo")
    from equilibria._local_refs import nus333_dir
    from equilibria.babel.har.reader import read_har
    from equilibria.family.altertax.run import solve_altertax

    run = solve_altertax(nus333_dir(), cache_root=env.mktemp("burf"), decomp_steps=4)
    path = run.decomp_path
    assert path.steps == 4 and set(path.headers) == set(DECOMP_HEADERS)
    one = decomp_headers(run)
    assert max(gates.ev_gaps(path.headers, one)) <= 1e-4
    gaps = gates.total_gaps(path.headers, one, gates.decomp_scale(one))
    assert gaps and max(r for *_, r in gaps.values()) <= PATH_TOL
    base = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"
    gem = read_har(str(base / "altax_decomp.har"))
    scale = gates.decomp_scale(gem)

    def worst(h):
        return max(r for *_, r in gates.total_gaps(h, gem, scale).values())

    assert worst(path.headers) < worst(one)
    assert max(gates.ev_gaps(path.headers, gem)) < max(gates.ev_gaps(one, gem))
    sl4 = gates.read_sl4(base / "altax.sl4.gz")
    rec = gates.report_decomp(path.headers, gem, gates.waivers_for(nus333_dir(), sl4))
    assert rec["pass"], rec["failing"]


def test_warm_start_from_a_solution_stays_there(env):
    """A sub-step starts from the previous one: solving the full shock again from its
    own solution reproduces it and needs (almost) no Newton iteration."""
    run = _rungtap(env, "nus333", 1)
    sess = run.result.session
    shock = load_tms_shock(folder("nus333") / "tms_rate10.shk")
    mp, res = sess.solve_shock(shock, start=run.result.mp)
    assert int(res["code"]) == 1
    assert int(res["nit"]) < int(run.result.results["shock"]["nit"])
    for name in ("xw", "regy", "pd", "savf"):
        a, b = mp.vars[name], run.result.mp.vars[name]
        sel = a.period == "shock"
        np.testing.assert_allclose(a.value[sel], b.value[sel], rtol=1e-7, atol=1e-9)


def test_k_converges_on_nus333_rungtap(env):
    """K doubles until every gated total of R(K/2) -> R(K) moves less than PATH_TOL; the
    reported decomposition is the Richardson one of K and K/2."""
    run = _rungtap(env, "nus333", "auto")
    path = run.decomp_path
    k, hist = path.steps, path.history
    assert k >= 4 and hist[k]["change"] < PATH_TOL
    assert hist[k // 2]["change"] is None or hist[k // 2]["change"] >= PATH_TOL
    assert path.richardson and path.steps_half == k // 2
    # the path's last point is the run's own shock: same EV basis as the other outputs
    assert path.points[-1] is run.result.mp


def _decomp_gate(env, ds):
    run = _rungtap(env, ds, "auto")
    cmp = gates.compare_all(run, folder(ds))
    rec = next(o for o in cmp["outputs"] if o["output"] == "decomp")
    assert rec["steps"] == run.decomp_path.steps
    assert rec["steps_half"] == run.decomp_path.steps // 2 and rec["richardson"]
    return rec


def test_decomp_gate_passes_on_nus333_rungtap(env):
    rec = _decomp_gate(env, "nus333")
    assert max(rec["ev_gap"]) <= TOL_CHANGE
    assert rec["pass"], rec["failing"]


def test_decomp_gate_passes_on_10x7_rungtap(env):
    """~40 s (K = 16: 15 sub-step solves on top of the run), so not marked slow."""
    rec = _decomp_gate(env, "gtap7_10x7")
    assert max(rec["ev_gap"]) <= TOL_CHANGE
    assert rec["pass"], rec["failing"]


def test_decomp_gate_passes_on_3x3_rungtap_with_one_step_dvol(env):
    """A22P dvol ROW is GEMPACK's -365.93 (one-step -365.94); the path integral of dvol
    gave -384.35, a different quantity (``V(t) dq`` instead of ``V0 q``)."""
    rec = _decomp_gate(env, "gtap7_3x3")
    assert max(rec["ev_gap"]) <= TOL_CHANGE
    assert rec["pass"], rec["failing"]


# --- Newton sub-steps ----------------------------------------------------------------

# largest relative change of a gated total between the Newton-step path and the
# full-solve path on gtap7_3x3: measured 2.95e-6 (K = 4, 2026-11-10); the bound is that
# measurement with a ~3x margin, far below PATH_TOL (1e-3)
NEWTON_VS_SOLVE_3X3 = 1e-5


@pytest.fixture(scope="module")
def run_3x3(env):
    return _rungtap(env, "gtap7_3x3", "auto")


def test_newton_path_matches_the_full_solve_path_on_3x3(run_3x3):
    """One Newton step per sub-step (the ratio closure in the linear system) in place
    of a full nonlinear solve with its ratio iterations: the reported decomposition is
    the full-solve one within NEWTON_VS_SOLVE_3X3 on every gated total, at the same K."""
    from equilibria.family.altertax.path import (
        gated_totals,
        noise_floor,
        solve_decomp_path,
        totals_change,
    )

    run = run_3x3
    newton = run.decomp_path
    assert newton.substep == "newton"
    shock = load_tms_shock(folder("gtap7_3x3") / "tms_rate10.shk")
    full = solve_decomp_path(run, shock, "auto", substep="solve")
    assert full.substep == "solve" and full.steps == newton.steps
    gap = totals_change(
        gated_totals(newton.headers),
        gated_totals(full.headers),
        noise_floor(full.headers),
    )
    assert gap <= NEWTON_VS_SOLVE_3X3


def test_newton_points_hold_the_ratio_closure(run_3x3):
    """``savf[r] / W`` of every ratio region equals the check's ratio at every Newton
    point (``dW = g . dx`` is in the linear system, savf follows it)."""
    from equilibria.family.driver import _world_income

    run = run_3x3
    path = run.decomp_path
    assert path.substep == "newton"
    mp0 = run.result.mp
    sv0 = mp0.vars["savf"]
    w_chk = _world_income(mp0, "check")
    regions = run.result.session.run.nominal_savf
    assert regions
    for mp in path.points[:-1]:
        w = _world_income(mp, "shock")
        sv = mp.vars["savf"]
        for r in regions:
            want = float(sv0.value[sv0.pos((r, "check"))]) / w_chk
            got = float(sv.value[sv.pos((r, "shock"))]) / w
            assert abs(got - want) <= 1e-10 * abs(want), (r, got, want)


def test_updated_system_equals_a_fresh_build_at_a_sub_step_point(run_3x3):
    """The Newton path builds the shock system once and writes each sub-step's shock and
    savf targets into its fixed slots: at a sub-step point (from the first point of the
    path towards the second, targets at that point's world income) its start point, F
    and J equal those of the system built from scratch (recipe, square, build)."""
    from equilibria.family.altertax.path import _NewtonSteps
    from equilibria.family.driver import _world_income

    run = run_3x3
    shock = load_tms_shock(folder("gtap7_3x3") / "tms_rate10.shk")
    k = run.decomp_path.steps
    steps = _NewtonSteps(run, shock)
    s1, s2 = shock_fraction(shock, 1 / k), shock_fraction(shock, 2 / k)
    start = steps.step(s1, None)
    w = _world_income(start, "shock")
    ss, x = steps.system_at(s2, start, w)
    fresh, sq_f, _ = run.result.session.shock_system(
        s2, start=start, savf_targets=steps.targets(w)
    )
    assert list(sq_f.col_names) == list(steps.sq.col_names)
    np.testing.assert_allclose(x, fresh.x0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ss.F(x), fresh.F(x), rtol=0, atol=1e-12)
    ja, jb = ss.J(x).tocsr(), fresh.J(x).tocsr()
    assert abs(ja - jb).max() <= 1e-12
