"""The rules every GTAP period follows, shared by the steps of any backend.

Constants and the period closures the multiperiod driver used inline, moved here
so the steps (``templates/gtap/steps.py``) read them without importing a backend.
The old names in ``period_prep`` and ``gtap_multiperiod_driver`` point here.
"""

from __future__ import annotations

from typing import Any

# F3.5: derived-demand variables the settled seed does not write. They follow
# from prices and incomes; seeding them at the check's value makes a small
# inconsistency. Letting them re-derive raised the match against GEMPACK on
# xg/xc from 78% to 100% (overall 94% -> 96%). The core (prices, factor
# quantities, output and trade) is seeded.
F35_DERIVED_DEMAND = frozenset(
    {
        "xg",
        "xc",
        "xg_agg",
        "xi",
        "xiagg",
        "yg",
        "yc",
        "zcons",
        "ug",
        "uh",
        "us",
        "u",
        "ev",
        "cv",
    }
)

# The period the altertax shares are recalibrated from. The shock always
# recalibrates from the check, even when it is seeded from the base (F3.5).
RECALIBRATE_FROM = {"check": "base", "shock": "check"}

# The numeraire anchor in the shock (GAMS fixes pnum in the period it solves).
SHOCK_PNUM = 1.5

# GAMS declares every variable free (model.gms:35 `Variables`) and in the solved
# period bounds only these prices/utilities (iterloop.gms:61-88, floor
# 0.001*level; pft commented out at :79). p = p_rai; pdp/pmp do not exist here.
GAMS_BOUNDED_VARS = frozenset(
    {
        "px", "pva", "pnd", "p_rai", "ps", "pdp", "pmp", "pa", "pmt", "pe",
        "pefob", "pmcif", "pm", "pet", "pd", "pwmg", "pf", "pfa",
        "uh", "ug", "us", "u", "pcons", "pg", "pi", "ptmg",
    }
)  # fmt: skip

# The inert welfare report tail: each row with the variable it defines. See
# ``gtap_multiperiod_driver._mute_welfare_tail`` for the measurement.
WELFARE_LEAVES = (
    ("eq_cv", "cv"),
    ("eq_ev", "ev"),
    ("eq_walras", "walras"),
    ("eq_u", "u"),
    ("eq_ug", "ug"),
    ("eq_us", "us"),
)

# The order the tail is solved after a period: u depends on ug/us (and the live
# uh), so it goes last.
WELFARE_SOLVE_ORDER = (
    ("eq_ug", "ug"),
    ("eq_us", "us"),
    ("eq_ev", "ev"),
    ("eq_cv", "cv"),
    ("eq_u", "u"),
)

# Steps of the Johansen derivative, as fractions of the shock. Measured on the
# Burfisher Johansen exercises (2026-11-03): at 0.001 the solver's own noise
# (~1e-10) breaks the slope of the tiny self-trade flows (qxs ~1e-5); at 0.03 the
# curvature of ME4A (land productivity -81%) does. 0.01 and 0.02, combined by
# Richardson, cancel the O(h^2) term and keep both away.
JOHANSEN_STEP = 0.01


def period_closures(closure: Any) -> tuple[Any, Any]:
    """The closures the periods are solved with: ``base`` (sluggish capital,
    taxes and technology free) and ``altertax`` (mobile capital, taxes and
    technology fixed). Both take ``if_sub`` and the numeraire from ``closure``
    (``None``: no ifSUB, ``pnum``)."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    if_sub = getattr(closure, "if_sub", False) if closure is not None else False
    numeraire = getattr(closure, "numeraire", "pnum") if closure is not None else "pnum"
    base = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        numeraire=numeraire,
    )
    alt = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=if_sub,
        numeraire=numeraire,
    )
    return base, alt
