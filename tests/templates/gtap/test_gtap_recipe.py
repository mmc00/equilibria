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


def test_family_backend_not_yet() -> None:
    with pytest.raises(NotImplementedError, match="family"):
        GTAP(None, None, backend="family").build(("base", "check", "shock"))


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
    (``solve_multiperiod``); the core route does the same through the template."""
    import gc

    from equilibria.templates.gtap.template import GTAP

    t = GTAP(params=None, closure=None)
    gc.unfreeze()  # an earlier test in the process may have left objects frozen
    with t.run_scope():
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0
