"""GTAPPyomoPeriodModel: the GTAP multiperiod Pyomo model behind GTAPPeriodModel."""

from __future__ import annotations

from typing import Any

import pytest

from equilibria.sources import dataset_dir
from equilibria.templates.gtap.instruments import exogenous_test

DATASET = dataset_dir("gtap7_3x3")
pytestmark = pytest.mark.skipif(
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


@pytest.fixture(scope="module")
def built() -> tuple[Any, Any, Any]:
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.pyomo_periods import GTAPPyomoPeriodModel

    p = _params()
    m, _ = build_block_model(p, p.sets, _closure(), list(p.sets.r)[-1])
    return GTAPPyomoPeriodModel(m), m, p


def _exo_by_test(m: Any, period: str) -> set[tuple[str, tuple]]:
    out = set()
    for name in m._exogenous_instruments:
        test = exogenous_test(m, name)
        for idx in getattr(m, name):
            kt = idx if isinstance(idx, tuple) else (idx,)
            if kt[-1] == period and test(idx):
                out.add((name, kt[:-1]))
    return out


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_exogenous_cells_match_exogenous_test(built, period: str) -> None:
    gm, m, _ = built
    got = {
        (name, idx)
        for name in m._exogenous_instruments
        for idx in gm.exogenous_cells(name, period)
    }
    assert got == _exo_by_test(m, period)
    assert got  # the 3x3 model has instruments


def test_exogenous_is_the_shock_period(built) -> None:
    gm, m, _ = built
    assert gm.exogenous() == frozenset(_exo_by_test(m, "shock"))


def test_reference_matches_sp_snapshot(built) -> None:
    from equilibria.templates.gtap import gtap_multiperiod_driver as drv
    from equilibria.templates.gtap.period_rules import period_closures

    gm, m, p = built
    base, _ = period_closures(_closure())
    ref = gm.reference(base, p, "base", fresh=True)
    snap = drv._sp_snapshot(
        drv._build_sp_reference(p.sets, p, base, m._residual_region, model=m)
    )
    want_fixed = {
        name: {drv._mp_index(ix, k, "base")[:-1]: v for k, v in cells}
        for name, (ix, cells) in snap.fixed.items()
    }
    want_bounds = {
        name: {drv._mp_index(ix, k, "base")[:-1]: (lb, ub) for k, lb, ub in cells}
        for name, (ix, cells) in snap.bounds.items()
    }
    assert ref.fixed == want_fixed
    assert ref.bounds == want_bounds


def test_stage_sets_driver_attributes(built) -> None:
    from equilibria.templates.gtap.period_rules import period_closures

    gm, m, p = built
    base, _ = period_closures(_closure())
    gm.stage(base, p, gtap_mode=True)
    assert m._gtap_mode is True
    assert m._residual_region == list(p.sets.r)[-1]
    assert gm.staged.closure is base and gm.staged.params is p
    assert gm.staged.gtap_mode is True


def test_properties(built) -> None:
    gm, m, _ = built
    assert gm.residual_region == m._residual_region
    assert gm.base_calibrated is False
    assert gm.settled_seed is None


def test_path_options_argument_wins_over_env() -> None:
    from equilibria.solver.path_capi import _merged_path_options

    got = _merged_path_options("major_iteration_limit 3", env="major_iteration_limit 7")
    assert "major_iteration_limit 3" in got.splitlines()
    assert "major_iteration_limit 7" not in got


def test_path_options_none_reads_env() -> None:
    from equilibria.solver.path_capi import _merged_path_options

    got = _merged_path_options(None, env="major_iteration_limit 7").splitlines()
    assert "major_iteration_limit 7" in got
    assert "cumulative_iteration_limit 1000000" in got
    assert "convergence_tolerance 1e-08" in got
