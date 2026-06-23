"""GTAP altertax multi-period SOLVING parity gate (both ifSUB modes).

For each (dataset, ifSUB) this builds the multi-period model, seeds it from the
committed GAMS GDX fixture, solves base->check->shock via solve_multiperiod, and
asserts:
  1. all 3 periods converge (termination code == 1), and
  2. the shock-period real-cell match vs the GDX is >= 98%.

Unlike test_gtap7_nl_parity.py (a no-solve .nl coefficient diff), this catches
regressions that CONVERGE to wrong values (e.g. the save<0 bug that silently
dropped gtap7_3x4 to 94% while still reporting code=1).

The test SKIPS (not fails) when either the fixture GDX is missing or the local
PATH solver (path_capi_python) is unavailable -- so it self-skips on ubuntu-latest
and actually runs on the self-hosted gams-tests job.

Run:
    uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v
    uv run pytest tests/templates/gtap/test_altertax_multiperiod_parity.py -v \
        -k "gtap7_3x3 and ifsub1"
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7_altertax"
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATASETS = ["gtap7_3x3", "gtap7_5x5", "gtap7_10x7"]
MATCH_THRESHOLD = 98.0

# Exclusion sets (denominator), identical to the session measurement harness.
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
    "xwmg", "xmgm", "lambdamg", "imptx", "exptx",
}
ALIAS = {
    "xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
    "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind",
}

# Local PATH solver lives outside the venv; add its src dir so find_spec can
# locate it.  This does NOT affect CI (the dir won't exist there, find_spec
# will still return None, and the 6 cases will SKIP as intended).
_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))


def _has_path_solver() -> bool:
    """The PATH MCP solver lives in a local package, absent on ubuntu-latest."""
    return importlib.util.find_spec("path_capi_python") is not None


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _fixture_gdx(dataset: str, if_sub: bool) -> Path:
    suffix = "ifsub1" if if_sub else "ifsub0"
    return FIXTURES_DIR / dataset / f"out_altertax_{suffix}.gdx"


def _solve_and_match(dataset: str, if_sub: bool):
    """Build, seed, solve, compare. Returns (codes_dict, match_pct, total)."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
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
    pa = apply_altertax_elasticities(p, in_place=False)
    ac = GTAPClosureConfig(
        name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True,
        if_sub=if_sub, numeraire="pnum",
    )
    mp = GTAPMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, ref)

    res = solve_multiperiod(
        m, p, ac, ref_gdx=ref,
        skip_base_solve=True, mute_welfare=True,
        seed_from_prior=False, holdfix_cd=True,
    )
    codes = {k: res[k]["code"] for k in res}

    tot = match = 0
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
            if t != "shock":
                continue
            st = tuple(_strip(x) for x in body)
            if not st:
                idx = ("shock",)
            elif len(st) == 1:
                idx = (st[0], "shock")
            else:
                idx = (*st, "shock")
            val = None
            for cand in [idx, (*body, "shock") if body else ("shock",)]:
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

    match_pct = 100.0 * match / max(tot, 1)
    return codes, match_pct, tot


@pytest.mark.gams
@pytest.mark.parametrize("if_sub", [False, True], ids=["ifsub0", "ifsub1"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_altertax_multiperiod_parity(dataset: str, if_sub: bool) -> None:
    if not _has_path_solver():
        pytest.skip("path_capi_python (PATH solver) not available")
    ref = _fixture_gdx(dataset, if_sub)
    if not ref.exists():
        pytest.skip(f"fixture GDX missing: {ref}")
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing for {dataset}")

    codes, match_pct, tot = _solve_and_match(dataset, if_sub)

    mode = "ifSUB=1" if if_sub else "ifSUB=0"
    assert all(c == 1 for c in codes.values()), (
        f"[{dataset}/{mode}] not all periods converged: {codes}"
    )
    assert tot > 0, f"[{dataset}/{mode}] no comparable cells found"
    assert match_pct >= MATCH_THRESHOLD, (
        f"[{dataset}/{mode}] real-cell match {match_pct:.2f}% "
        f"< {MATCH_THRESHOLD}% (over {tot} cells)"
    )
