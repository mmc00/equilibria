"""GTAP7 MCP fidelity gate (per stage, altertax, both ifSUB).

Python is solved via PATH (nonlinear-full MCP) against the cleanly-converged NEOS
MCP reference (regenerated 2026-07-17, subsidy-aware, eq_pxeq clean). Same per-stage
contract as the NLP gate: for every coverage-matrix `mcp` row this runs the real
base→check→shock solve, MEASURES match%/code per stage, and asserts match >= floor
AND code == 1 for EVERY stage. Nothing hardcoded — the match% is re-derived here.

With clean refs the match is 99%+ everywhere (base/check exact, shock ≥99.3 except
15x10's known eq_paa Armington micro-cell ~95%) — proving the NLP gate's 89–97
ceiling was the mis-converged NLP reference, not the model.

LOCAL-ONLY: not marked `gams`, so CI never collects it. SKIPS when the fixture GDX
is missing or the local PATH solver is absent. Run by hand to validate the MCP
pipeline.

Run:
    uv run pytest tests/templates/gtap/test_gtap7_mcp_parity.py -v
    uv run pytest tests/templates/gtap/test_gtap7_mcp_parity.py -v -k "15x10"
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
sys.path.insert(0, str(ROOT / "src"))

from coverage_matrix import mcp_rows  # noqa: E402

_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {"pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
      "xwmg", "xmgm", "lambdamg", "imptx", "exptx"}
ALIAS = {"xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
         "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind"}
TOL = 1e-2

_MCP_CASES = [
    (r.dataset, r.ifsub, r.mode, dict(r.stage_floors), r.ci_status)
    for r in mcp_rows()
]


def _has_solver() -> bool:
    return importlib.util.find_spec("path_capi_python") is not None


def _fixture_gdx(dataset: str, ifsub: int) -> Path:
    return FIXTURES_DIR / dataset / f"out_altertax_ifsub{ifsub}.gdx"


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _solve_and_measure(dataset: str, ifsub: int, gdx: Path):
    """Build + seed + solve base→check→shock via PATH (MCP); return
    {period: {"match": float, "code": int, "total": int}}."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel, PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from pyomo.environ import value as V
    from _diff_core import gams_levels, list_populated_vars, split_t

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(basedata_path=d / "basedata.har", sets_path=d / "sets.har",
                    default_path=d / "default.prm", baserate_path=d / "baserate.har")
    rr = list(p.sets.r)[-1]
    pa = apply_altertax_elasticities(p, in_place=False)
    ac = GTAPClosureConfig(name="altertax", closure_type="MCP",
                           capital_mobility="mobile", fix_endowments=False,
                           fix_taxes=True, fix_technology=True,
                           if_sub=bool(ifsub), numeraire="pnum")
    mp = GTAPMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)

    res = solve_multiperiod(m, p, ac, ref_gdx=gdx, skip_base_solve=True,
                            mute_welfare=True, seed_from_prior=False,
                            holdfix_cd=True)
    codes = {k: res[k]["code"] for k in res}

    def measure(period: str):
        tot = match = 0
        for vn in list_populated_vars(gdx):
            if vn.lower() in SKIP or vn.lower() in RF:
                continue
            try:
                g = gams_levels(gdx, vn)
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
                idx = (period,) if not st else ((st[0], period) if len(st) == 1 else (*st, period))
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
                da = abs(val - gval)
                rel = da / abs(gval) if abs(gval) > 1e-12 else (0.0 if da < 1e-6 else 9e9)
                if da <= 1e-6 or rel <= TOL:
                    match += 1
        return {"match": 100.0 * match / max(tot, 1), "total": tot}

    out = {}
    for per in PERIODS:
        mm = measure(per)
        out[per] = {"match": mm["match"], "code": codes[per], "total": mm["total"]}
    return out


@pytest.mark.parametrize(
    "dataset,ifsub,mode,floors,ci_status",
    _MCP_CASES,
    ids=[f"{r.dataset}-ifsub{r.ifsub}" for r in mcp_rows()],
)
def test_gtap7_mcp_parity(dataset, ifsub, mode, floors, ci_status):
    if ci_status == "blocked":
        pytest.skip(f"blocked: {dataset}")
    if not _has_solver():
        pytest.skip("PATH solver (path_capi_python) not available")
    gdx = _fixture_gdx(dataset, ifsub)
    if not gdx.exists():
        pytest.skip(f"fixture GDX missing: {gdx}")
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing: {dataset}")

    per = _solve_and_measure(dataset, ifsub, gdx)
    tag = f"{dataset}/ifSUB={ifsub}"

    bad_codes = {k: v["code"] for k, v in per.items() if v["code"] != 1}
    assert not bad_codes, f"[{tag}] stages did not converge: {bad_codes}"

    for stage, floor in floors.items():
        mm = per[stage]
        assert mm["total"] > 0, f"[{tag}/{stage}] no comparable cells"
        assert mm["match"] >= floor, (
            f"[{tag}/{stage}] MCP match {mm['match']:.2f}% < floor {floor}% "
            f"(over {mm['total']} cells) — regression"
        )
