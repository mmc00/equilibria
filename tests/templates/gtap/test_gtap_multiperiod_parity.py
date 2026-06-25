"""Fidelity gate: pure-gtap multi-period shock vs the GAMS LOCAL reference.

This is the Task 2 fidelity gate.  It builds the pure-gtap (NON-altertax)
multi-period model, seeds it from the committed GAMS GDX fixture
(out_gtap_shock_ifsub{0,1}.gdx — solved Variables with the base/check/shock
t-axis), solves base->check->shock via solve_multiperiod in gtap-mode, and
asserts:
  1. all 3 periods converge (termination code == 1), and
  2. the SHOCK-period real-cell match vs the GAMS GDX is >= a MEASURED floor.

It compares CHECK and SHOCK (not base: base is the calibration anchor; check is
the baseline equilibrium; the shock is relative to check).  The assert floor is a
value MEASURED this run and recorded in the commit message (the controller sets
the matrix gap_min from it) — NOT an aspirational target.

The earlier "MP == single-period" gate was retired: the in-process Python
single-period reference was itself code=2 (an unsound reference).  This gate
instead pins MP against the GAMS LOCAL solve, mirroring
test_altertax_multiperiod_parity.py (same _diff_core reader, same SKIP/RF/ALIAS
exclusion sets, same match rule abs<=1e-6 OR rel<=1e-2).

LOCAL-ONLY: needs the PATH solver (path_capi_python) + the dataset HAR + the
fixture GDX.  It SKIPS (never fails) when any is absent, so it is safe on CI (no
self-hosted runner / no solver there) — mirror of the altertax gate.  Run by hand:

    uv run pytest tests/templates/gtap/test_gtap_multiperiod_parity.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = ROOT / "datasets"
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

# Local PATH solver lives outside the venv; add its src dir so find_spec can
# locate it.  This does NOT affect CI (the dir won't exist there, find_spec will
# still return None, and the tests SKIP as intended).
_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

# Exclusion sets (denominator), identical to test_altertax_multiperiod_parity.py
# lines 49-57.
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
    "xwmg", "xmgm", "lambdamg", "imptx", "exptx",
}
ALIAS = {
    "xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
    "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind",
}

# Measured shock-period match% floor vs the GAMS LOCAL reference (ifSUB=0).
# AS MEASURED 2026-06-24 (deterministic): CHECK 100.00% / SHOCK 99.70% (1332 cells),
# after the 3-link coupled shock fix (gtap-mode-gated; altertax byte-identical):
#   (1) anchor the NatRes factor — holdfix xf[r,NatRes,a] + deactivate the redundant
#       vertical eq_pfeq (etaff=0 makes it pf-free), leaving the live eq_xfeq to pin
#       pf; and DROP the eq_xfeq[USA,NatRes,Mnfcs] entry from _REDUNDANT_FACTOR_ROWS
#       (it was the only row pinning pf there → pf exploded to 9.27 under the shock).
#   (2) un-skip the domestic-diagonal tariff in _apply_imptx_shock (GAMS shocks ALL
#       routes incl r==rp; Python skipped it) — gtap-mode-gated.
#   (3) rebuild eq_ytax[mt] in-equation with the SHOCKED imptx (the income leak):
#       eq_ytax[mt] bakes the BASE imptx at build, so the in-solve mt revenue missed
#       the shocked tariff stream → understated regY → demand collapse → Armington
#       overshoot.  The rebuild uses the col2=importer convention (= GAMS ytaxeq line
#       680 imptx(rp,i,r); verified = GAMS ytax[USA,mt]=0.26003 exact).  The post-
#       solve _recompute_ytax_mt is no-op'd for gtap-mode (it overwrote the correct
#       in-solve value with a col0-filtered 0.19892).
# This lifted SHOCK 67.12% -> 99.70% (CHECK stays 100.00%, EXACT baseline).  The
# remaining 0.30% is proven basin (EU/ROW Land sluggish single-jump; seed-at-GAMS
# holds 100%), NOT a 4th link.  All 3 periods code=1.  The floor is set just below
# the as-measured shock value so the gate is GREEN and the true number is captured
# for the coverage-matrix gap_min.  See the commit message + the report for details.
SHOCK_MATCH_FLOOR_IFSUB0 = 99.0


def _has_path_solver() -> bool:
    """The PATH MCP solver lives in a local package, absent on ubuntu-latest."""
    return importlib.util.find_spec("path_capi_python") is not None


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _fixture_gdx(dataset: str, if_sub: bool) -> Path:
    suffix = "ifsub1" if if_sub else "ifsub0"
    return FIXTURES_DIR / dataset / f"out_gtap_shock_{suffix}.gdx"


def _gtap_closure(if_sub: bool):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=if_sub, numeraire="pnum",
    )


def _solve_and_match(dataset: str, if_sub: bool, period: str = "shock"):
    """Build the pure-gtap MP model, seed from the GAMS GDX, solve, compare.

    Returns (codes_dict, match_pct, total, worst) for the requested period.
    """
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel,
        PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from pyomo.environ import value as V
    from _diff_core import gams_levels, list_populated_vars, split_t

    ref = _fixture_gdx(dataset, if_sub)
    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    gc = _gtap_closure(if_sub)
    mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, ref)

    # gtap-mode: skip_base_solve=True keeps the calibrated base anchor (pva=ps=pd=1)
    # instead of letting PATH slide the USA price level; holdfix_cd=False (the
    # gtap-mode driver path does not run the altertax CD-nest holdfix).
    res = solve_multiperiod(
        m, p, gc, ref_gdx=ref,
        skip_base_solve=True, mute_welfare=True,
        seed_from_prior=False, holdfix_cd=False, mode="gtap",
    )
    codes = {k: res[k]["code"] for k in res}

    tot = match = 0
    worst: list[tuple[float, str, tuple, float, float]] = []
    for vn in list_populated_vars(ref):
        if vn.lower() in SKIP or vn.lower() in RF:
            continue
        try:
            g = gams_levels(ref, vn)
        except Exception:
            continue
        pv = getattr(m, ALIAS.get(vn, vn), None) or getattr(m, vn.lower(), None)
        if pv is None:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t != period:
                continue
            st = tuple(_strip(x) for x in body)
            if not st:
                idx = (period,)
            elif len(st) == 1:
                idx = (st[0], period)
            else:
                idx = (*st, period)
            val = None
            for cand in [idx, (*body, period) if body else (period,)]:
                try:
                    val = float(V(pv[cand]))
                    break
                except Exception:
                    pass
            if val is None:
                continue
            tot += 1
            d_abs = abs(val - gval)
            rel = d_abs / abs(gval) if abs(gval) > 1e-12 else (
                0.0 if d_abs < 1e-6 else 9e9
            )
            if d_abs <= 1e-6 or rel <= 1e-2:
                match += 1
            else:
                worst.append((rel, vn, body, gval, val))

    match_pct = 100.0 * match / max(tot, 1)
    worst.sort(reverse=True)
    return codes, match_pct, tot, worst


@pytest.mark.skipif(not _has_path_solver(), reason="PATH solver unavailable")
def test_gtap_mp_shock_matches_gams_3x3():
    """Pure-gtap MP shock vs GAMS LOCAL (ifSUB=0): 3 periods code=1 + measured
    shock match% floor."""
    dataset = "gtap7_3x3"
    if_sub = False
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")
    ref = _fixture_gdx(dataset, if_sub)
    if not ref.exists():
        pytest.skip(f"fixture GDX missing: {ref}")

    codes, match_pct, tot, worst = _solve_and_match(dataset, if_sub, "shock")

    worst_str = "; ".join(
        f"{vn}{idx}: GAMS={gv:.5g} MP={mv:.5g} rel={r:.3g}"
        for r, vn, idx, gv, mv in worst[:10]
    )
    assert all(c == 1 for c in codes.values()), (
        f"[{dataset}/ifSUB=0] not all periods converged: {codes}"
    )
    assert tot > 0, f"[{dataset}/ifSUB=0] no comparable shock cells found"
    # MEASURED floor (recorded in commit message); assert at the measured value
    # so the gate is green AND the real number is captured for the matrix.
    assert match_pct >= SHOCK_MATCH_FLOOR_IFSUB0, (
        f"[{dataset}/ifSUB=0] shock match {match_pct:.2f}% < measured floor "
        f"{SHOCK_MATCH_FLOOR_IFSUB0}% (over {tot} cells). worst: {worst_str}"
    )
