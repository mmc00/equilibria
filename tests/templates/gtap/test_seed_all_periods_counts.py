"""seed_all_periods says how much of the GAMS point it placed, per period: the
cells the GDX exports for a Python variable and those that land on a cell of it
(MAR-219: the cascade's seed-at-GAMS layer reports this coverage)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


@pytest.mark.skipif(_FOLDER is None, reason="GAMS MCP altertax ref absent")
@pytest.mark.needs_gdxdump
def test_the_seed_counts_exported_and_placed_cells_per_period():
    spec = importlib.util.spec_from_file_location(
        "seed_and_solve", ROOT / "scripts/gtap/seed_and_solve.py"
    )
    assert spec is not None and spec.loader is not None
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)

    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS

    assert _FOLDER is not None
    p, gc, rr = ss._case("gtap7_3x3", False, "altertax")
    mp = GTAPBlockMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    counts = mp.seed_all_periods(m, _FOLDER / "out.gdx")
    assert set(counts) == set(PERIODS)
    for t in PERIODS:
        exported, placed = counts[t]
        assert 0 < placed <= exported
        assert placed / exported > 0.9
