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
