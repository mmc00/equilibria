"""capFlex ``risk[r]`` from the calibrated data, as ``cal.gms:680-684`` (MAR-251).

Oracle: ``_capflex_risk_oracle`` (GAMS's own ``risk`` read with gdxdump, and today's
capFix twin)."""

from __future__ import annotations

import copy

import pytest
from tests.templates.gtap._capflex_risk_oracle import (
    GDX_REL,
    RISK_GDX,
    RISK_TWIN,
    TWIN_REL,
)

from equilibria.sources import dataset_dir, has_dataset


def _params(dataset: str):
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _rel(got: dict[str, float], want: dict[str, float]) -> float:
    assert set(got) == set(want)
    return max(abs(got[r] - want[r]) / abs(want[r]) for r in want)


@pytest.mark.parametrize("dataset", sorted(RISK_GDX))
def test_formula_matches_gams(dataset: str) -> None:
    if not has_dataset(dataset):
        pytest.skip(f"{dataset} not present")
    from equilibria.templates.gtap.capflex_risk import capflex_risk_from_data

    _, want = RISK_GDX[dataset]
    assert _rel(capflex_risk_from_data(_params(dataset)), want) < GDX_REL


@pytest.mark.parametrize("dataset", sorted(RISK_TWIN))
def test_formula_vs_todays_twin(dataset: str) -> None:
    """The move Pyomo makes when the twin goes: inside the twin's own tolerance."""
    if not has_dataset(dataset):
        pytest.skip(f"{dataset} not present")
    from equilibria.templates.gtap.capflex_risk import capflex_risk_from_data

    rel = _rel(capflex_risk_from_data(_params(dataset)), RISK_TWIN[dataset])
    print(f"{dataset}: formula vs twin max rel {rel:.3e}")
    assert rel < TWIN_REL


@pytest.mark.skipif(not has_dataset("nus333"), reason="nus333 not present")
def test_a_region_without_capital_return_raises() -> None:
    """rorc = arent - fdepr = 0 makes rore 0: an error naming the region, never 1."""
    from equilibria.templates.gtap.capflex_risk import capflex_risk_from_data

    p = copy.deepcopy(_params("nus333"))
    b = p.benchmark
    evos_usa = sum(v for k, v in b.evos.items() if "USA" in k and "CAPITAL" in k)
    b.vdep["USA"] = evos_usa
    with pytest.raises(ValueError, match="USA"):
        capflex_risk_from_data(p)


def _capflex_closure(if_sub: bool = False):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        savf_flag="capFlex",
        numeraire="pnum",
    )


@pytest.fixture
def no_twin(monkeypatch):
    """Any capFix twin-solve fails loudly: the test proves none is attempted."""
    from equilibria.core import exercise

    def _fail(self):
        raise AssertionError("a capFix twin-solve was attempted")

    monkeypatch.setattr(exercise.Exercise, "run", _fail)


@pytest.mark.skipif(not has_dataset("nus333"), reason="nus333 not present")
def test_pyomo_capflex_risk_is_the_formula(no_twin) -> None:
    from equilibria.templates.gtap.capflex_risk import capflex_risk_from_data
    from equilibria.templates.gtap.gtap_block_model import _calibrate_capflex_risk

    p = _params("nus333")
    risk = _calibrate_capflex_risk(p, p.sets, _capflex_closure(), "ROW", ref_gdx=None)
    assert risk == capflex_risk_from_data(p)


@pytest.mark.skipif(not has_dataset("nus333"), reason="nus333 not present")
def test_built_capflex_model_carries_the_formula_risk(no_twin) -> None:
    """Burfisher's case (no GDX): the risk the block reads is the formula, never 1.
    (The multi-period model folds ``risk`` to numbers, so it is read off ``params``.)"""
    from equilibria.templates.gtap.capflex_risk import capflex_risk_from_data
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _params("nus333")
    want = capflex_risk_from_data(p)
    build_block_model(p, p.sets, _capflex_closure(), "ROW")
    assert p._capflex_risk == want
