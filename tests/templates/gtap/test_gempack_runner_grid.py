"""The GEMPACK runner must emit distinct .cmf content per (shock, steps) config
so the linearization-study grid does not overwrite itself."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))


def test_shock_pct_appears_in_cmf():
    from run_gempack_matrix import make_cmf

    regs = ["USA", "EU", "ROW"]
    c3 = make_cmf("gtap7_3x3", regs, shock_pct=3.0)
    assert "Shock tm = uniform 3 ;" in c3
    assert "uniform 3%" in c3


def test_steps_flag_drives_gragg_line():
    from run_gempack_matrix import make_cmf

    regs = ["USA", "EU", "ROW"]
    c = make_cmf("gtap7_3x3", regs, shock_pct=10.0, steps="4 8 16 32 64")
    assert "Steps  = 4 8 16 32 64 ;" in c
    assert "Steps  = 8 16 32 ;" not in c


def test_updated_name_is_config_specific():
    from run_gempack_matrix import config_tag, make_cmf

    regs = ["USA", "EU", "ROW"]
    tag = config_tag(3.0, "8 16 32")
    c = make_cmf("gtap7_3x3", regs, shock_pct=3.0, updated_name=f"updated_{tag}.har")
    assert f"Updated file GTAPDATA = updated_{tag}.har ;" in c


def test_config_tag_is_stable_and_distinct():
    from run_gempack_matrix import config_tag

    assert config_tag(10.0, "8 16 32") != config_tag(3.0, "8 16 32")
    assert config_tag(10.0, "8 16 32") != config_tag(10.0, "4 8 16 32 64")
    # deterministic
    assert config_tag(0.1, "8 16 32") == config_tag(0.1, "8 16 32")
    # filesystem-safe: no dots or spaces
    tag = config_tag(0.1, "4 8 16 32 64")
    assert "." not in tag
    assert " " not in tag
