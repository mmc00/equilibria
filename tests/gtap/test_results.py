"""``equilibria.gtap.results``: GEMPACK-named % changes, EV and a summary of a solved exercise.

The tables are ``gtap7_side.gempack_pct`` of the model levels (the rule AlterTax is
gated with against GEMPACK) and the EV is ``regional_ev`` (gated against GEMPACK's EV
on Burfisher): ``results`` adds no translation of its own.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from equilibria.blocks.gtap._derived_params import xscale_data
from equilibria.compare.gtap7_side import gempack_pct, model_levels
from equilibria.gtap import load, results
from equilibria.gtap._results import GTAPResults
from equilibria.templates.gtap.welfare_regional import regional_ev

ROUTE = ("USA", "MFG", "ROW")


@pytest.fixture(scope="module")
def ex():
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    e = Exercise(
        GTAP(load("nus333"), GTAPClosureConfig.gtap_default(), residual_region="ROW"),
        gtap_static("gtap", solve_check=True),
        shock={"imptx": {ROUTE: 10.0}},
    )
    codes = e.run()
    assert codes["shock"]["code"] == "optimal"
    return e


@pytest.fixture(scope="module")
def res(ex) -> GTAPResults:
    return results(ex)


def _want(ex):
    p = ex.template.params
    return gempack_pct(model_levels(ex.model.pyomo), dict(xscale_data(p, p.sets)))[0]


pytestmark = [pytest.mark.needs_path, pytest.mark.needs_asl]


def test_tables_equal_gempack_pct(ex, res) -> None:
    want = _want(ex)
    assert set(res.tables) == set(want)
    for var, cells in want.items():
        t = res.tables[var]
        assert list(t.columns) == ["pct"]
        if t.index.names == [None]:  # a scalar (pfactwld): one row
            got = {(): float(t["pct"].iloc[0])}
        else:
            got = {
                (k if isinstance(k, tuple) else (k,)): x for k, x in t["pct"].items()
            }
        assert got == cells, var


def test_table_index_uses_gempack_sets(res) -> None:
    assert list(res.tables["qxs"].index.names) == ["COMM", "REG", "REG"]
    assert list(res.tables["qo"].index.names) == ["ACTS", "REG"]
    assert res.tables["qxs"].loc[("MFG", "USA", "ROW"), "pct"] < 0  # the taxed route


def test_ev_equals_regional_ev(ex, res) -> None:
    assert res.ev.to_dict() == regional_ev(ex.model.pyomo, "shock", "check")
    assert res.ev.name == "EV"


def test_summary_has_every_region(ex, res) -> None:
    s = res.summary
    assert list(s.index) == list(ex.template.params.sets.r)
    assert list(s.columns) == ["EV", "qgdp", "exports", "imports"]
    assert s["qgdp"].to_dict() == res.tables["qgdp"]["pct"].to_dict()
    assert s["EV"].to_dict() == res.ev.to_dict()


def test_summary_trade_is_total_xw(ex, res) -> None:
    from pyomo.environ import value

    m = ex.model.pyomo

    def total(r, t, out):
        return sum(
            value(m.xw[(s, i, d, t)])
            for (s, i, d, tt) in m.xw
            if tt == t and (s if out else d) == r
        )

    for r in ex.template.params.sets.r:
        for col, out in (("exports", True), ("imports", False)):
            want = 100 * (total(r, "shock", out) / total(r, "check", out) - 1)
            assert res.summary.loc[r, col] == pytest.approx(want, rel=1e-12)


def test_repr_prints_the_summary(res) -> None:
    assert repr(res) == repr(res.summary)
    assert "<table" in res._repr_html_()


def test_before_run_raises() -> None:
    ex = SimpleNamespace(
        model=None, codes=None, template=SimpleNamespace(backend="pyomo")
    )
    with pytest.raises(RuntimeError, match="run the exercise first"):
        results(ex)


def test_run_that_raised_midway_raises(ex) -> None:
    """``run()`` leaves the model on ``ex`` before it solves: a run that raised has a
    model but no codes, and its levels are not a solution."""
    broken = SimpleNamespace(
        model=ex.model, codes=None, template=ex.template, recipe=ex.recipe
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        results(broken)


def test_exercise_keeps_its_codes(ex) -> None:
    assert ex.codes["shock"]["code"] == "optimal"


def test_post_not_optimal_warns(ex, caplog) -> None:
    codes = {**ex.codes, "shock": {"code": "infeasible", "residual": 1.0}}
    bad = SimpleNamespace(
        model=ex.model, codes=codes, template=ex.template, recipe=ex.recipe
    )
    with caplog.at_level(logging.WARNING, logger="equilibria.gtap._results"):
        results(bad)
    assert any("infeasible" in r.getMessage() for r in caplog.records)


@pytest.fixture(scope="module")
def johansen_ex():
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    e = Exercise(
        GTAP(load("nus333"), GTAPClosureConfig.gtap_default(), residual_region="ROW"),
        gtap_static("gtap", solve_check=True, shock_method="johansen"),
        shock={"imptx": {ROUTE: 10.0}},
    )
    e.run()
    return e


def test_johansen_uses_the_linear_translation(johansen_ex) -> None:
    """A Johansen solution is compared with GEMPACK through ``equilibria_side(...,
    linear=True)`` (``compare_gempack_levels.run``); ``results`` reads the method from
    the recipe and uses the same translation."""
    p = johansen_ex.template.params
    levels = model_levels(johansen_ex.model.pyomo)
    xs = dict(xscale_data(p, p.sets))
    want = gempack_pct(levels, xs, linear=True)[0]
    assert (
        want["pr"] != gempack_pct(levels, xs)[0]["pr"]
    )  # the test can tell them apart
    got = results(johansen_ex).tables["pr"]["pct"].to_dict()
    assert got == want["pr"]


def test_family_tables_without_ev_warn(ex, res, caplog) -> None:
    """On the family backend the tables come from ``family_levels`` and there is no EV
    (MAR-251): an empty Series and one WARNING, never another quantity as EV."""
    levels = model_levels(ex.model.pyomo)
    mp = SimpleNamespace(
        vars={
            n: SimpleNamespace(keys=list(c), value=list(c.values()))
            for n, c in levels.items()
        }
    )
    fam = SimpleNamespace(
        model=SimpleNamespace(mp=mp),
        codes=ex.codes,
        recipe=ex.recipe,
        template=SimpleNamespace(backend="family", params=ex.template.params),
    )
    with caplog.at_level(logging.WARNING, logger="equilibria.gtap._results"):
        r = results(fam)
    assert [x.levelno for x in caplog.records] == [logging.WARNING]
    assert r.ev.empty
    for var in res.tables:
        pd.testing.assert_frame_equal(r.tables[var], res.tables[var])
    assert r.summary["EV"].isna().all()
