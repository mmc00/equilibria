"""The GTAP template and its static recipe on the core driver."""

from __future__ import annotations

from typing import Any

import pytest

from equilibria.core.exercise import Exercise
from equilibria.core.recipe import solve_once
from equilibria.sources import dataset_dir
from equilibria.templates.gtap.recipe import gtap_static
from equilibria.templates.gtap.template import GTAP

DATASET = dataset_dir("gtap7_3x3")
needs_3x3 = pytest.mark.skipif(
    not DATASET.exists(), reason="gtap7_3x3 dataset not present"
)


def _params() -> Any:
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATASET / "basedata.har",
        sets_path=DATASET / "sets.har",
        default_path=DATASET / "default.prm",
        baserate_path=DATASET / "baserate.har",
    )
    return p


def _closure() -> Any:
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )


def test_static_periods() -> None:
    r = gtap_static()
    assert [(p.name, p.kind) for p in r.periods] == [
        ("base", "calibrate"),
        ("check", "replicate"),
        ("shock", "shock"),
    ]


def test_base_not_solved_by_default() -> None:
    r = gtap_static()
    assert r.plans["calibrate"].solver is None
    assert gtap_static(solve_base=True).plans["calibrate"].solver is not None


def test_settle_only_has_no_shock_period() -> None:
    assert gtap_static(settle_only=True).names == ("base", "check")


def test_copied_check_is_not_solved() -> None:
    assert gtap_static(base_calibrated=True).plans["replicate"].solver is None
    r = gtap_static(base_calibrated=True, solve_check=True)
    assert r.plans["replicate"].solver is not None


def test_strategy_follows_the_shock_method() -> None:
    assert gtap_static().plans["shock"].strategy is solve_once
    assert gtap_static(shock_steps=(0.5,)).plans["shock"].strategy is not solve_once
    assert (
        gtap_static(shock_method="johansen").plans["shock"].strategy is not solve_once
    )


def _bound(step: Any, name: str) -> Any:
    """What the step factory was called with (``stage_closure(which=...)``)."""
    import inspect

    return inspect.getclosurevars(step).nonlocals.get(name)


def _made_by(plan: Any, factory: str) -> list[Any]:
    """The steps of ``plan`` (before its solve) that ``steps.<factory>`` made."""
    return [s for s in plan.before if s.__qualname__.startswith(f"{factory}.")]


@pytest.mark.parametrize(
    ("mode", "period", "closure"),
    [
        ("gtap", "calibrate", "base"),
        ("gtap", "replicate", "base"),
        ("gtap", "shock", "base"),
        ("altertax", "calibrate", "base"),
        ("altertax", "replicate", "altertax"),
        ("altertax", "shock", "altertax"),
    ],
)
def test_each_period_stages_the_closure_of_its_mode(mode, period, closure) -> None:
    (stage,) = _made_by(gtap_static(mode).plans[period], "stage_closure")
    assert _bound(stage, "which") == closure


@pytest.mark.parametrize(
    ("kw", "prior"), [({}, "check"), ({"base_calibrated": True}, "base")]
)
def test_shock_seeds_from_the_check_unless_it_was_copied(kw, prior) -> None:
    r = gtap_static(**kw)
    seeds = [getattr(s, "seeds_from", None) for s in r.plans["shock"].before]
    assert prior in seeds
    if not kw:
        check = [getattr(s, "seeds_from", None) for s in r.plans["replicate"].before]
        assert "base" in check


def test_shock_always_builds_its_own_reference_model() -> None:
    for mode in ("gtap", "altertax"):
        plans = gtap_static(mode).plans
        assert [
            _bound(s, "fresh") for s in _made_by(plans["shock"], "replicate_reference")
        ] == [True]
        assert [
            _bound(s, "fresh")
            for s in _made_by(plans["replicate"], "replicate_reference")
        ] == [False]


def test_gtap_recipes_do_not_recalibrate_and_altertax_ones_do() -> None:
    from equilibria.templates.gtap import steps

    for kind in ("replicate", "shock"):
        assert steps.recalibrate_shares not in gtap_static().plans[kind].before
        assert steps.recalibrate_shares in gtap_static("altertax").plans[kind].before


def test_johansen_refuses_steps() -> None:
    with pytest.raises(ValueError, match="takes no shock_steps"):
        gtap_static(shock_method="johansen", shock_steps=(0.5,))


def test_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be 'altertax' or 'gtap'"):
        gtap_static(mode="pure")


def test_bad_shock_method() -> None:
    with pytest.raises(ValueError, match="shock_method must be"):
        gtap_static(shock_method="euler")


def test_bad_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        GTAP(None, None, backend="jax")


def test_bad_builder() -> None:
    """Only the ShockBlock builders: the monolith (no ShockBlock to write the shock
    into) never reaches a recipe."""
    with pytest.raises(ValueError, match="builder"):
        GTAP(None, None, builder="monolith")


def test_overwrite_argument_not_yet() -> None:
    with pytest.raises(NotImplementedError, match="@overwrite"):
        GTAP(None, None).build(("base", "check", "shock"), overwrite=[object()])


def test_periods_must_be_the_static_ones() -> None:
    with pytest.raises(ValueError, match="periods"):
        GTAP(None, None).build(("y2020", "y2021"))


def test_settle_only_refuses_shock() -> None:
    with pytest.raises(ValueError, match="no shock period"):
        Exercise(
            template=GTAP(None, None),
            recipe=gtap_static(settle_only=True),
            shock={"imptx": {("USA", "Mnfcs", "EU_28"): 5.0}},
        )


def test_default_recipe_takes_the_template_flag() -> None:
    t = GTAP(None, None, base_calibrated=True)
    assert t.default_recipe().plans["replicate"].solver is None


@needs_3x3
def test_recipe_refuses_mismatched_base_calibrated() -> None:
    calls: list[Any] = []

    def solver(m: Any) -> str:
        calls.append(m)
        return "optimal"

    ex = Exercise(
        template=GTAP(_params(), _closure(), base_calibrated=False),
        recipe=gtap_static(base_calibrated=True, solve_base=True, solver=solver),
    )
    with pytest.raises(ValueError, match="base_calibrated"):
        ex.run()
    assert calls == []


def test_gtap_run_scope_freezes_once_for_the_run() -> None:
    """The old driver froze the built model once over base, check and shock
    (the old multiperiod driver); the core route does the same through the template."""
    import gc

    from equilibria.templates.gtap.template import GTAP

    t = GTAP(params=None, closure=None)
    gc.unfreeze()  # an earlier test in the process may have left objects frozen
    with t.run_scope():
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0
