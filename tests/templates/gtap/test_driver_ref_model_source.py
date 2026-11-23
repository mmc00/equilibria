"""F3, the central cut: a run must not build the MONOLITH.

The old driver built a `GTAPModelEquations(...).build_model()` in three places
(base / check / shock) and only read `.fixed`, `lb` and `ub` from it to replicate
the single-period fixing and bounds.

On 20x41 that is 3.4M cells built to read three attributes, and while it happens
the monolith is a RUNTIME dependency.

Measured (altertax closure) after d9e6d49, the monolith's SP against
`build_block_single_period`:

    gtap7_3x3    common 1,330   .fixed different 0
    gtap7_10x7   common 20,145  .fixed different 0

The cells only the monolith has (470 and 5,063) are the 30 dead shifters, in no
equation: there is nothing to replicate.

THE DEFAULT IS ALREADY BLOCKS. Two things blocked it, both measured and fixed:

- 33 duplicated Fisher rows that only survived on the SP route (cf2c1b0).
- The 9 ifSUB report vars were frozen at "their current value", which made the
  init difference between the routes DEFINITIVE (`pwmg` at 0.001 vs 1.0, a factor
  of 1000), with the 15x10 pure ifSUB=1 MCP gate at 87% (f654f63).

``EQUILIBRIA_GTAP_REF_MODEL=monolith`` makes the core build it: the test then fails
(checked, 3 builds).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


@pytest.mark.needs_path
def test_core_run_does_not_build_the_monolith(monkeypatch):
    """The contract: zero `GTAPModelEquations` during a whole core run (altertax,
    base, check and shock)."""
    from tests.templates.gtap._routes import solve

    from equilibria.templates.gtap import gtap_model_equations as ME
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
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

    built: list[int] = []
    orig = ME.GTAPModelEquations.__init__

    def counting(self, *a, **k):
        built.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(ME.GTAPModelEquations, "__init__", counting)
    res = solve("core", p, gc, rr, mode="altertax", skip_base_solve=True)

    assert set(res.codes) == {"base", "check", "shock"}
    assert built == [], (
        f"the run built {len(built)} monolith(s); the reference model comes "
        "from the blocks"
    )
