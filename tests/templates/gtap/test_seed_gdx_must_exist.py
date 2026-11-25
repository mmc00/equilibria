"""seed_all_periods refuses a GDX it cannot read (MAR-163).

A missing path or a file gdxdump cannot read used to leave the model unseeded
and solve anyway, so a run seeded from the wrong place looked like a seeded one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def built():
    from tests.templates.gtap._period_snapshot import _load_params

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = _load_params()
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    return build_block_model(p, p.sets, gc, list(p.sets.r)[-1])


def test_a_missing_gdx_raises(built, tmp_path):
    m, mp = built
    with pytest.raises(FileNotFoundError, match="nope.gdx"):
        mp.seed_all_periods(m, tmp_path / "nope.gdx")


@pytest.mark.needs_gdxdump
def test_a_gdx_gdxdump_cannot_read_raises(built, tmp_path):
    from equilibria.babel.gdx.gdxdump import locate_gdxdump

    if locate_gdxdump() is None:
        pytest.skip("gdxdump not on the PATH (comes with GAMS)")
    m, mp = built
    bad = tmp_path / "bad.gdx"
    bad.write_text("not a gdx file")
    with pytest.raises(RuntimeError, match="bad.gdx"):
        mp.seed_all_periods(m, bad)
