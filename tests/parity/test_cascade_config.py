import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import pytest
from cascade_config import (
    family,
    resolve_periods,
    resolve_ref_gdx,
    scenario_for,
)

pytestmark = pytest.mark.needs_path


def test_family_split():
    assert family("gtap7_3x3") == "gtap7"
    assert family("9x10") == "bundle"
    assert family("nus333") == "bundle"


def test_scenario_for_gtap7():
    assert scenario_for("gtap7_3x3", "check") == "altertax_check"
    assert scenario_for("gtap7_3x3", "shock") == "altertax_shock"


def test_resolve_periods_drops_base_for_gtap7():
    available, dropped = resolve_periods("gtap7_3x3", ["base", "check", "shock"])
    assert available == ["check", "shock"]
    assert dropped == ["base"]


def test_resolve_periods_drops_check_for_bundle():
    available, dropped = resolve_periods("9x10", ["base", "check", "shock"])
    assert available == ["base", "shock"]
    assert dropped == ["check"]


def test_resolve_ref_gdx_altertax_is_the_recorded_gams_reference():
    # the MCP gate's reference, not the old durable CD ref, whose point is no
    # longer a fixed point of the model (MAR-216)
    from equilibria.sources import reference

    res = resolve_ref_gdx("gtap7_3x3")
    want = reference(
        "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
    )
    assert res.usable is True
    assert res.source == "gams_reference"
    assert res.path == want / "out.gdx"


def test_resolve_ref_gdx_missing_is_not_usable():
    res = resolve_ref_gdx("gtap7_20x41")  # no GAMS MCP altertax ref
    assert res.usable is False
    assert res.source == "missing"
    assert res.path is None
    assert "NO usable" in res.note


def test_resolve_ref_gdx_gtap_is_the_recorded_gams_reference():
    from equilibria.sources import reference

    res = resolve_ref_gdx("gtap7_3x3", mode="gtap", ifsub=1)
    want = reference(
        "gtap7_3x3", "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    )
    assert res.usable is True
    assert res.source == "gams_reference"
    assert res.path == want / "out.gdx"


def test_resolve_ref_gdx_gtap_missing_is_not_usable():
    res = resolve_ref_gdx("gtap7_20x41", mode="gtap", ifsub=0)  # no GAMS MCP ref
    assert res.usable is False
    assert res.source == "missing"
    assert res.path is None
