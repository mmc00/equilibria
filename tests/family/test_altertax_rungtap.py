"""Phase C: RunGTAP's AlterTax references (10 % tariff-rate shock, del_tbalry closure)."""

from __future__ import annotations

import pytest

from equilibria.family.altertax import gates
from equilibria.family.altertax.run import load_altertax_params
from equilibria.family.altertax.shocks import load_tms_shock
from tests.family._prep import ROOT

DATA = ROOT / "data/gtap7"
REF = "gempack_altertax-rungtap_tms-rate10_s8-16-32"
DATASETS = ("nus333", "gtap7_10x7")


def folder(ds):
    return DATA / ds / REF


@pytest.mark.parametrize("ds", DATASETS)
def test_tms_shock_is_rate_times_1p1(ds):
    """The shk percent is of the tms POWER; imptx is a rate (GEMPACK kind ``power``), so
    ``shocked("power", rate, 1 + pct/100)`` must be ``1.1 * rate`` in every cell, and the
    cells are exactly the non-zero-rate flows (cell order (source, commodity, dest))."""
    from equilibria.blocks.gtap._derived_params import imptx_data
    from equilibria.templates.gtap.shocks import gempack_kind, shocked

    shock = load_tms_shock(folder(ds) / "tms_rate10.shk")
    assert list(shock) == ["imptx"] and gempack_kind("imptx") == "power"
    p = load_altertax_params(DATA / ds, folder(ds) / "parameters.prm")
    rate = imptx_data(p, p.sets)
    for cell, pct in shock["imptx"].items():
        assert shocked("power", rate[cell], 1 + pct / 100) == pytest.approx(
            1.1 * rate[cell], abs=1e-11
        )
    assert {k for k, r in rate.items() if r != 0.0} == set(shock["imptx"])


def test_tms_shock_cell_order_nus333():
    """File line order is (destination, commodity) with the source across."""
    cells = load_tms_shock(folder("nus333") / "tms_rate10.shk")["imptx"]
    # first file value 0.1539448261 is tms(AGR, ROW -> USA)
    assert cells[("ROW", "AGR", "USA")] == pytest.approx(0.1539448261)
    assert cells[("USA", "AGR", "ROW")] == pytest.approx(0.59784054369)


@pytest.mark.parametrize("ds", DATASETS)
def test_prm_outside_dataset_sets_mobility_and_leontief_investment(ds):
    """EFLG of the folder's parameters.prm decides the mobility (RunGTAP: all mobile);
    esubi = 0 and the unit elasticities stay exactly 1 (the 1.01 stand-in only on request)."""
    p = load_altertax_params(DATA / ds, folder(ds) / "parameters.prm")
    assert sorted(p.sets.mf) == sorted(p.sets.f) and p.sets.sf == []
    assert all(v == 0.0 for v in p.elasticities.esubi.values())
    assert any(v == 1.0 for v in p.elasticities.esubd.values())
    assert not any(v == 1.01 for v in p.elasticities.esubd.values())
    q = load_altertax_params(DATA / ds, folder(ds) / "parameters.prm", unit_sigma=1.01)
    assert not any(v == 1.0 for v in q.elasticities.esubd.values())


def test_closure_capital_mobility_does_not_override_eflg():
    """altertax_closure() says ``capital_mobility="sluggish"`` but nothing on the family
    route or in the blocks reads it: the mobility sets are the .prm's."""
    import re

    from equilibria.family.altertax.closure import altertax_closure

    assert altertax_closure().capital_mobility == "sluggish"
    pat = re.compile(r"\.capital_mobility\b|capital_mobility\s*==")
    readers = [
        str(f.relative_to(ROOT))
        for d in ("src/equilibria/family", "src/equilibria/blocks/gtap")
        for f in (ROOT / d).rglob("*.py")
        if pat.search(f.read_text())
    ]
    assert readers == []


@pytest.mark.parametrize("ds", DATASETS)
def test_burfisher_waivers_do_not_apply_to_rungtap_folders(ds):
    sl4 = gates.read_sl4(folder(ds) / "sl4dump.har")
    assert not gates.is_nus333_altax_reference(DATA / ds, sl4)
    assert gates.waivers_for(DATA / ds, sl4) is gates.NO_WAIVERS


@pytest.mark.parametrize("ds", DATASETS)
def test_sl4dump_har_is_read_as_percent_changes(ds):
    sl4 = gates.read_sl4(folder(ds) / "sl4dump.har")
    ch = sl4.changes
    assert {"qo", "pds", "tms", "del_tbalry", "ywld", "EV"} <= set(ch)
    assert ch["qo"].kind == "p" and ch["qo"].values.ndim == 2
    assert ch["ywld"].values.shape == ()
    assert ch["tms"].set_elements[0][0] in ("AGR", "Rice")


def test_levels_from_pct_is_exact_on_identical_changes():
    """A solution whose % changes ARE the reference's has zero level gap."""
    assert gates.level_gap(110.0, 100.0, 10.0) < 1e-15
    assert gates.level_gap(110.011, 100.0, 10.0) == pytest.approx(0.011 / 110.0)


def test_driver_rejects_unknown_trade_balance():
    from equilibria.family.driver import solve_gtap_family

    with pytest.raises(ValueError, match="nominal_trade_balance"):
        solve_gtap_family(object(), object(), "ROW", nominal_trade_balance="real")
    from equilibria.family.altertax.run import solve_altertax

    with pytest.raises(ValueError, match="trade_balance"):
        solve_altertax(DATA / "nus333", trade_balance="real")


# --- solves (nus333 ~10 s, 10x7 ~20 s) -------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from tests.family._prep import _newton_mumps

    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    yield tmp_path_factory
    mp.undo()


def _ratio_run(env, ds):
    pytest.importorskip("pyomo")
    from equilibria.family.altertax.run import solve_altertax

    prm = folder(ds) / "parameters.prm"
    return solve_altertax(
        DATA / ds,
        prm=prm,
        shock=load_tms_shock(folder(ds) / "tms_rate10.shk"),
        trade_balance="ratio",
        cache_root=env.mktemp(f"c_{ds}"),
    )


def _at(mp, var, key, period):
    v = mp.vars[var]
    return float(v.value[v.keys.index((*key, period))])


@pytest.mark.parametrize("ds", DATASETS)
def test_ratio_closure_holds_savf_over_world_income(env, ds):
    """GEMPACK del_tbalry(r) = 0 for r != residual: savf / world income is the check's."""
    run = _ratio_run(env, ds)
    assert run.codes == {"check": 1, "shock": 1}
    mp = run.result.mp
    regs = run.params.sets.r
    w = {p: sum(_at(mp, "regy", (r,), p) for r in regs) for p in ("check", "shock")}
    for r in regs[:-1]:
        assert _at(mp, "savf", (r,), "shock") / w["shock"] == pytest.approx(
            _at(mp, "savf", (r,), "check") / w["check"], rel=1e-10, abs=1e-14
        )
    eq = mp.rows["eq_savf"]
    assert not any(
        bool(eq.active[eq.keys.index((r, "shock"))]) for r in regs[:-1]
    )  # the real-balance row is off, the residual region keeps its own
    assert run.result.results["shock"]["ratio_iterations"] > 1


def test_ratio_levels_gate_on_nus333(env):
    """Levels of the family shock against sl4dump.har's changes (post = check * (1 +
    pct/100)) at TOL_LEVELS on nus333 (measured 6.7e-5; the nominal closure gives 3.8e-4)."""
    run = _ratio_run(env, "nus333")
    sl4 = gates.read_sl4(folder("nus333") / "sl4dump.har")
    rec = gates.report_levels_from_pct(run, sl4)
    assert rec["pass"] and rec["max"] <= gates.TOL_LEVELS and rec["cells"] > 100


def test_compare_all_runs_on_a_rungtap_folder(env):
    """The one entry point finds sl4dump.har for the levels and the pct gates, applies no
    waivers and does not compare long-name labels."""
    run = _ratio_run(env, "nus333")
    cmp = gates.compare_all(run, folder("nus333"))
    assert [o["output"] for o in cmp["outputs"]] == [
        "levels", "updated", "summary", "volume", "decomp", "pct",
    ]  # fmt: skip
    assert cmp["waivers"]["applied"] is False
    assert next(o for o in cmp["outputs"] if o["output"] == "levels")["pass"]
    upd = next(o for o in cmp["outputs"] if o["output"] == "updated")
    assert upd["structure_problems"] == []  # XXHS is history text: not compared


def test_solved_run_keeps_the_prm_mobility(env):
    """After a solve with altertax_closure() (capital_mobility='sluggish') the sets are the
    .prm's: every factor mobile, none sluggish."""
    run = _ratio_run(env, "nus333")
    p = load_altertax_params(DATA / "nus333", folder("nus333") / "parameters.prm")
    assert run.params.sets.mf == p.sets.mf == run.params.sets.f
    assert run.params.sets.sf == p.sets.sf == []


def _trade_prices(run, t):
    """GAMS M_PEFOB / M_PMCIF at period t from the live pe / ptmg / lambdamg."""
    from equilibria.family.cross_period import _ifsub_params

    mp, sets = run.result.mp, run.params.sets
    pr = {k: v[1] for k, v in _ifsub_params(run.params).items()}
    rs, cs, ms = list(sets.r), list(sets.i), list(sets.m)
    pefob, pmcif = {}, {}
    for a, e in enumerate(rs):
        for b, c in enumerate(cs):
            for k, d in enumerate(rs):
                pef = pr["efac"][a, b, k] * _at(mp, "pe", (e, c, d), t)
                pwmg = sum(
                    pr["amgm"][n, a, b, k]
                    * _at(mp, "ptmg", (mm,), t)
                    / (_at(mp, "lambdamg", (mm, e, c, d), t) + 1e-12)
                    for n, mm in enumerate(ms)
                )
                pefob[(e, c, d)] = pef
                pmcif[(e, c, d)] = pef + pwmg * pr["tmarg"][a, b, k]
    return pefob, pmcif


def _mqgdp_from_levels(run, r, tp, tq):
    """GAMS mqgdp(tp, tq) of region r (model.gms 1306) from the solved levels."""
    mp = run.result.mp
    pefob, pmcif = _trade_prices(run, tp)
    fd = ("hhd", "gov", "inv", "tmg")
    xa = mp.vars["xaa"]
    ab = sum(
        _at(mp, "pa", k[:3], tp) * float(xa.value[n])
        for n, k in enumerate(xa.keys)
        if k[0] == r and k[2] in fd and k[3] == tq
    )
    xw = mp.vars["xw"]
    tr = 0.0
    for n, k in enumerate(xw.keys):
        if k[3] != tq:
            continue
        if k[0] == r:
            tr += pefob[k[:3]] * float(xw.value[n])
        if k[2] == r:
            tr -= pmcif[k[:3]] * float(xw.value[n])
    return ab + tr


def test_fisher_gdp_cross_sums_use_live_trade_prices(env):
    """Under ifSUB the Fisher GDP aggregates value trade at M_PEFOB / M_PMCIF of the
    period (GAMS model.gms 1306), not at the frozen pefob / pmcif report Vars: with a
    tariff shock mq_gdp_sb (shock prices, base quantities) moved off base-price trade."""
    run = _ratio_run(env, "nus333")
    mp = run.result.mp
    pefob, pmcif = _trade_prices(run, "shock")  # the post-solve report vars agree
    for k in pefob:
        assert pefob[k] == pytest.approx(_at(mp, "pefob", k, "shock"), rel=1e-12)
        assert pmcif[k] == pytest.approx(_at(mp, "pmcif", k, "shock"), rel=1e-12)
    for r in run.params.sets.r:
        for slot, (tp, tq) in {
            "sb": ("shock", "base"),
            "bs": ("base", "shock"),
        }.items():
            assert _at(mp, f"mq_gdp_{slot}", (r,), "shock") == pytest.approx(
                _mqgdp_from_levels(run, r, tp, tq), rel=1e-10
            ), (r, slot)
