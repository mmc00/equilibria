"""Activities and commodities with their own names (``9x10``: ``a_agricultur`` makes
``c_Crops`` and ``c_MeatLstk``): the output flags are keyed by activity, and the
square check refuses a system short of rows as loudly as one with too many."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _params(vom=None, vom_i=None):
    """One region, activity ``a_ag`` making ``c_crops`` and ``c_meat``, activity
    ``a_mfg`` making ``c_mfg``; ``vom_i`` is by commodity, as the loaders give it."""
    makb = {
        ("R", "a_ag", "c_crops"): 2.0,
        ("R", "a_ag", "c_meat"): 1.0,
        ("R", "a_mfg", "c_mfg"): 5.0,
    }
    if vom_i is None:
        vom_i = {("R", "c_crops"): 2.0, ("R", "c_meat"): 1.0, ("R", "c_mfg"): 5.0}
    bm = SimpleNamespace(vom=vom or {}, vom_i=vom_i, makb=makb)
    sets = SimpleNamespace(
        r=["R"], a=["a_ag", "a_mfg"], i=["c_crops", "c_meat", "c_mfg"]
    )
    return SimpleNamespace(benchmark=bm, sets=sets)


def test_output_flag_is_keyed_by_activity():
    """xFlag(r, a): every activity with a make flow, never a commodity."""
    from equilibria.family.prep_square import _output_flags

    x_flag, x_rai = _output_flags(_params())
    assert x_flag == {("R", "a_ag"), ("R", "a_mfg")}
    assert x_rai == {
        ("R", "a_ag", "c_crops"),
        ("R", "a_ag", "c_meat"),
        ("R", "a_mfg", "c_mfg"),
    }


def test_output_flag_ignores_a_commodity_keyed_vom():
    """A ``vom`` keyed by commodity (the HAR ``VOSB(COMM, REG)``) does not flag
    commodities as activities."""
    from equilibria.family.prep_square import _output_flags

    p = _params(vom={("R", "c_crops"): 2.0, ("R", "c_meat"): 1.0, ("R", "c_mfg"): 5.0})
    x_flag, _ = _output_flags(p)
    assert x_flag == {("R", "a_ag"), ("R", "a_mfg")}


def _mp(free: int, rows: int):
    v = SimpleNamespace(fixed=np.array([False] * free + [True]))
    r = SimpleNamespace(active=np.array([True] * rows + [False]))
    return SimpleNamespace(vars={"x": v}, rows={"eq": r})


def test_square_check_refuses_a_shortfall():
    """Fewer active rows than free variables after the squareness patches."""
    from equilibria.family.prep_square import _check_overdetermined

    with pytest.raises(ValueError, match="under-determined: 3 active rows for 5 free"):
        _check_overdetermined(_mp(free=5, rows=3))


def test_square_check_still_refuses_too_many_rows():
    from equilibria.family.prep_square import _check_overdetermined

    with pytest.raises(ValueError, match="over-determined"):
        _check_overdetermined(_mp(free=3, rows=5))
    _check_overdetermined(_mp(free=4, rows=4))


def test_har_sets_map_activities_to_their_commodities():
    """A HAR load fills the activity -> commodity maps from MAKB, not only when the
    names coincide: ``a_agricultur`` makes ``c_Crops`` and ``c_MeatLstk``."""
    from equilibria.family.altertax.run import load_altertax_params

    d = Path(__file__).resolve().parents[2] / "data" / "gtap7" / "9x10"
    if not (d / "basedata.har").exists():
        pytest.skip("9x10 data missing")
    p = load_altertax_params(
        d, d / "gempack_altertax-rungtap_tms-rate10_s8-16-32" / "parameters.prm"
    )
    s = p.sets
    assert s.activity_commodities["a_agricultur"] == ["c_Crops", "c_MeatLstk"]
    assert s.commodity_activities["c_MeatLstk"] == ["a_agricultur"]
    assert all(s.activity_commodities[a] for a in s.a)
    assert all(s.commodity_activities[i] for i in s.i)
    assert set(s.output_pairs) == {(a, i) for (_r, a, i) in p.benchmark.makb}


_9X10 = Path(__file__).resolve().parents[2] / "data" / "gtap7" / "9x10"
_REF = "gempack_altertax-rungtap_tms-rate10_s8-16-32"


def test_residual_region_comes_from_the_cmf_cgdslack():
    """9x10 has no ``ROW``: RunGTAP's closure leaves ``cgdslack("RestofWorld")``."""
    from equilibria.family.altertax.closure import residual_region_from_exp

    cmf = _9X10 / _REF / "run.cmf"
    if not cmf.exists():
        pytest.skip("9x10 reference missing")
    assert residual_region_from_exp(cmf) == "RestofWorld"


def test_solve_altertax_refuses_a_residual_region_outside_the_regions():
    """An unknown residual region would drop no ``eq_savf`` / ``eq_yi`` row."""
    from equilibria.family.altertax.run import load_altertax_params, solve_altertax

    if not (_9X10 / "basedata.har").exists():
        pytest.skip("9x10 data missing")
    prm = _9X10 / _REF / "parameters.prm"
    p = load_altertax_params(_9X10, prm)
    with pytest.raises(ValueError, match="residual region 'ROW' is not a region"):
        solve_altertax(_9X10, prm=prm, shock={}, params=p, residual_region="ROW")


def test_pyomo_conditional_fixing_keys_xflag_by_activity():
    """The Pyomo ``apply_conditional_fixing`` had the same commodity-keyed fallback:
    with activity names apart from the commodity names it fixed every xp / va / nd."""
    pyo = pytest.importorskip("pyomo.environ")
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    p = _params()
    m = pyo.ConcreteModel()
    m.ra = pyo.Set(initialize=[("R", "a_ag"), ("R", "a_mfg")], dimen=2)
    for name in ("xp", "va", "nd"):
        setattr(m, name, pyo.Var(m.ra, initialize=1.0))
    GTAPSolver(m, params=p).apply_conditional_fixing()
    fixed = [(n, k) for n in ("xp", "va", "nd") for k in m.ra if getattr(m, n)[k].fixed]
    assert fixed == []


def test_har_vom_is_keyed_by_activity():
    """``vom`` is (r, a); the HAR ``VOSB`` is by commodity, so it enters ``vom`` only
    through a one-to-one make structure (nus333), never under commodity keys."""
    from equilibria.family.altertax.run import load_altertax_params

    if not (_9X10 / "basedata.har").exists():
        pytest.skip("9x10 data missing")
    p = load_altertax_params(_9X10, _9X10 / _REF / "parameters.prm")
    acts = set(p.sets.a)
    assert all(a in acts for (_r, a) in p.benchmark.vom)
