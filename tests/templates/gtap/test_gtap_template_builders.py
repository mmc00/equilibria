"""``GTAP(builder=...)``: the template builds the blocks model or one of its two
variants (``gtap_sparse``, ``gtap_loglevels``), so the variants run on the core
driver (PR 3a of gtap-on-core)."""

from __future__ import annotations

import math

import pytest

from equilibria.sources import dataset_dir, has_dataset

pytestmark = pytest.mark.skipif(
    not has_dataset("gtap7_3x3"), reason="gtap7_3x3 not present"
)
PERIODS = ("base", "check", "shock")


def _params():
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure(**kw):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        **kw,
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )


def _log_rows(m) -> int:
    from pyomo.environ import Constraint

    return sum(1 for c in m.component_data_objects(Constraint) if "log(" in str(c.body))


def test_unknown_builder_is_refused() -> None:
    from equilibria.templates.gtap.template import BUILDERS, GTAP

    with pytest.raises(ValueError, match="builder") as e:
        GTAP(_params(), _closure(), builder="dense")
    assert all(b in str(e.value) for b in BUILDERS)


@pytest.mark.parametrize("builder", ["sparse", "loglevels"])
def test_family_refuses_a_variant_builder(builder: str) -> None:
    from equilibria.templates.gtap.template import GTAP

    with pytest.raises(NotImplementedError, match=builder):
        GTAP(_params(), _closure(), builder=builder, backend="family")


def test_sparse_builder_builds_the_sparse_model() -> None:
    from equilibria.templates.gtap.template import GTAP

    t = GTAP(_params(), _closure(), builder="sparse")
    m = t.build(PERIODS).pyomo
    # measured on gtap7_3x3: 3 padding routes, 258 variables fixed near 0
    stats = t.fix_stats
    assert stats is not None
    assert stats["vars_fixed"] == 258
    assert stats["route_padding"] == 3
    assert _log_rows(m) == 2  # not the log-levels model


def test_loglevels_builder_builds_the_loglevels_model() -> None:
    from equilibria.templates.gtap.template import GTAP

    t = GTAP(_params(), _closure(), builder="loglevels")
    m = t.build(PERIODS).pyomo
    # measured on gtap7_3x3: 2186 log-wrapped rows (the blocks model has 2)
    assert _log_rows(m) == 2186
    assert t.fix_stats is None


def test_blocks_builder_is_the_default() -> None:
    from equilibria.templates.gtap.template import GTAP

    t = GTAP(_params(), _closure())
    m = t.build(PERIODS).pyomo
    assert _log_rows(m) == 2
    assert t.fix_stats is None


def _levels(m) -> dict[str, float]:
    from pyomo.environ import Var
    from pyomo.environ import value as V

    out = {}
    for v in m.component_objects(Var, active=True):
        for idx in v:
            x = v[idx].value
            out[f"{v.name}{idx}"] = math.nan if x is None else float(V(v[idx]))
    return out


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def _loglevels_reference():
    """The setup of the log-levels gate (tests/templates/gtap_loglevels): from a cold
    start PATH does not converge on the log form; seeded from the GAMS MCP solution
    and solved as an NLP it does. ``None`` when that reference is not recorded."""
    from equilibria.sources import find_reference

    ref = find_reference(
        "gtap7_3x3", "gams", form="mcp", mode="gtap", ifsub=0, fbep="header"
    )
    return None if ref is None else ref / "out.gdx"


@pytest.mark.parametrize("builder", ["sparse", "loglevels"])
def test_variant_solves_equal_the_old_driver(builder: str, monkeypatch) -> None:
    """The variant on the core equals the variant on the old driver, every cell
    (the same equality as test_gtap_on_core_solve for the blocks model)."""
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP
    from equilibria.templates.gtap_loglevels.multiperiod import (
        build_loglevels_model_mp,
    )
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    gdx = None
    if builder == "loglevels":
        gdx = _loglevels_reference()
        if gdx is None:
            pytest.skip("gtap7_3x3 GAMS MCP reference (fbep=header) not recorded")
        monkeypatch.setenv("EQUILIBRIA_GTAP_SOLVE_NLP", "1")

    def setup():
        p = _params()
        if gdx is None:
            return p, _closure()
        p.set_va_subsidy_basis("gempack")
        return p, _closure(va_subsidy_basis="gempack")

    p, cl = setup()
    rr = list(p.sets.r)[-1]
    if builder == "sparse":
        m_old = build_sparse_model_mp(p, p.sets, cl, rr)[0]
    else:
        m_old, mp = build_loglevels_model_mp(p, p.sets, cl, rr, ref_gdx=gdx)
        mp.seed_all_periods(m_old, gdx)
    old = solve_multiperiod(
        m_old,
        p,
        cl,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
    )
    p2, cl2 = setup()
    ex = Exercise(
        GTAP(p2, cl2, builder=builder, ref_gdx=gdx, seed_gdx=gdx), gtap_static()
    )
    res = ex.run()
    assert [old[k]["code"] for k in ("check", "shock")] == [1, 1]
    assert [res[k]["code"] for k in ("check", "shock")] == ["optimal", "optimal"]
    a, b = _levels(m_old), _levels(ex.model.pyomo)  # type: ignore[union-attr]
    assert a.keys() == b.keys()
    bad = [k for k in a if not _same(a[k], b[k])]
    assert bad == []
