"""capFlex investment closure (returns equalize, GEMPACK-matching) — tests.

Default savf_flag=capFix (GAMS comp.gms) stays byte-identical; capFlex (GAMS
GFTLnd.gms/gft.gms) is the returns-equalizing closure GEMPACK's land result uses.
"""

import contextlib
import io
from pathlib import Path

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

DS = Path("datasets/gtap7_3x3")


def _build(savf_flag):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DS / "basedata.har",
        sets_path=DS / "sets.har",
        default_path=DS / "default.prm",
        baserate_path=DS / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=False,
        savf_flag=savf_flag,
    )
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        m, mp = build_block_model(p, p.sets, gc, list(p.sets.r)[-1])
    return m


def _eq_savf_bodies(m):
    """Concatenate the string form of every active eq_savf cell (to detect the branch)."""
    return " ".join(str(m.eq_savf[k].body) for k in m.eq_savf if m.eq_savf[k].active)


def test_capfix_default_savf_uses_pigbl():
    # capFix (default): savf = pigbl * savf_bar (savf_bar folds to a literal; pigbl stays).
    m = _build("capFix")
    body = _eq_savf_bodies(m)
    assert "pigbl" in body, (
        "capFix eq_savf must reference pigbl (savf = pigbl*savf_bar)"
    )
    assert "rorg" not in body, "capFix eq_savf must NOT equalize returns"


def test_capflex_savf_equalizes_returns():
    # capFlex: risk[r]*rore[r] == rorg  (returns equalize). No savf_bar.
    m = _build("capFlex")
    body = _eq_savf_bodies(m)
    assert "rore" in body and "rorg" in body, (
        "capFlex eq_savf must equalize returns (risk*rore == rorg)"
    )
    assert "savf_bar" not in body, "capFlex must NOT use the capFix savf_bar form"
