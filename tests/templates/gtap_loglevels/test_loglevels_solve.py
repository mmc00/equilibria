"""Gate for the log-levels template: OUR levels model in log form solves and matches
GAMS on gtap7_3x3 — the same parity as the levels blocks (blocks/gtap), proving the
log re-expression is faithful. Reuses the levels test's seed/solve/measure verbatim,
only swapping the single-period source to the log-wrapped blocks.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASET = "gtap7_3x3"
RESIDUAL_REGION = "ROW"
GDX = ROOT / "tests" / "fixtures" / "gtap7" / DATASET / "out_gtap_shock_ifsub0.gdx"
FLOOR = 99.0

import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "tests" / "templates" / "gtap"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "scripts"))

from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS  # noqa: E402


def _load_params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    d = ROOT / "datasets" / DATASET
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure():
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


def _strip(x):
    """Drop the GAMS set-prefix (c_/a_/f_/r_) so GDX 'c_Food' matches model 'Food'."""
    return (
        x[2:]
        if (isinstance(x, str) and len(x) > 2 and x[1] == "_" and x[0] in "acfr")
        else x
    )


def _measure(m, period):
    """Cell-by-cell match% vs the GAMS GDX for one period, with the c_/a_ prefix
    stripped and a 1% band — the SAME criterion the levels-block gate uses (identical
    SKIP/RF exclusions from measure_nlp_vs_nlp), written locally to avoid import-path
    fragility. RF report vars (pm/pmcif/pfa/imptx/…) are EXCLUDED: GAMS computes them
    post-solve and they are not part of the compared equilibrium."""
    from _diff_core import (  # ty: ignore[unresolved-import]
        gams_levels,
        list_populated_vars,
        split_t,
    )
    from measure_nlp_vs_nlp import ALIAS, RF, SKIP  # ty: ignore[unresolved-import]
    from pyomo.environ import value

    tot = match = 0
    for vn in list_populated_vars(str(GDX)):
        if vn.lower() in SKIP or vn.lower() in RF:
            continue
        pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        try:
            g = gams_levels(str(GDX), vn)
        except Exception:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t != period:
                continue
            st = tuple(_strip(x) for x in body)
            try:
                mval = float(value(pv[(*st, period)]))
            except Exception:
                continue
            tot += 1
            if abs(mval - gval) <= 0.01 * max(abs(gval), 1e-6) + 0.01:
                match += 1
    return tot, (100.0 * match / tot if tot else 0.0)


@pytest.mark.slow
@pytest.mark.skipif(not GDX.exists(), reason="gtap7_3x3 reference GDX missing")
def test_loglevels_3x3_nlp_matches_gams():
    from equilibria.templates.gtap.gtap_block_model import solve_block_model
    from equilibria.templates.gtap_loglevels.multiperiod import build_loglevels_model_mp

    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
    try:
        p = _load_params()
        gc = _closure()
        m, mp = build_loglevels_model_mp(p, p.sets, gc, RESIDUAL_REGION)
        mp.seed_all_periods(m, GDX)
        res = solve_block_model(m, p, gc, ref_gdx=GDX, mode="gtap")
    finally:
        os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
    for per in PERIODS:
        assert res[per]["code"] == 1, (
            f"log-levels NLP {per} did not converge: {res[per]}"
        )
    for per in PERIODS:
        tot, pct = _measure(m, per)
        assert tot > 0, f"{per}: no comparable cells"
        assert pct >= FLOOR, f"log-levels NLP {per} match {pct:.4f} < {FLOOR}"
