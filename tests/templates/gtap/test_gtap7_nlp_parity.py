"""GTAP7 NLP-vs-NLP fidelity gate (per stage, both modes, both ifSUB).

Python is solved as an NLP (EQUILIBRIA_GTAP_SOLVE_NLP=1, maximize walras) against
the GAMS ifMCP=0 NLP reference. Same IPOPT on both sides → the equality tolerance
cancels → the cell-by-cell match reflects MODEL fidelity, not solver noise.

For each coverage-matrix `nlp` row this runs the real base→check→shock solve, then
for EVERY stage asserts:
  1. the stage converged (return code == 1), and
  2. the stage's cell-by-cell match% @ tol1% >= its per-stage floor (stage_floors
     in scripts/gtap/coverage_matrix.py).

The floors are the versioned contract; the match% is MEASURED here at run time and
never hardcoded — that is what makes this a regression gate rather than a snapshot
comparison. If a code change slides any stage below its floor (or breaks
convergence), this fails.

LOCAL-ONLY: intentionally NOT marked `gams`, so CI never collects it. SKIPS (not
fails) when the fixture GDX is missing or the local PATH/NLP toolchain is absent,
so it is safe to run anywhere; it only does real work on a machine with the NLP
solve path + the dataset HAR. Run by hand to validate the NLP pipeline.

Run:
    uv run pytest tests/templates/gtap/test_gtap7_nlp_parity.py -v
    uv run pytest tests/templates/gtap/test_gtap7_nlp_parity.py -v -k "5x5 and pure"
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7_nlp"
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "src"))

from coverage_matrix import nlp_rows  # noqa: E402

# Local PATH/NLP toolchain lives outside the venv; add its src so find_spec locates
# it. Absent on CI → the cases SKIP.
_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

# Same exclusion sets as the session measurement harness (denominator).
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {
    "pfa",
    "pfy",
    "pm",
    "pmcif",
    "pefob",
    "pwmg",
    "pp",
    "pdp",
    "pmp",
    "xwmg",
    "xmgm",
    "lambdamg",
    "imptx",
    "exptx",
}
ALIAS = {
    "xa": "xaa",
    "xd": "xda",
    "xm": "xma",
    "pp": "pp_rai",
    "p": "p_rai",
    "ytaxInd": "ytax_ind",
    "ytaxind": "ytax_ind",
}
TOL = 1e-2  # match @ tol1%

# (dataset, ifsub, mode, stage_floors_dict, ci_status) per nlp matrix row.
_NLP_CASES = [
    (r.dataset, r.ifsub, r.mode, dict(r.stage_floors), r.ci_status) for r in nlp_rows()
]


def _has_solver() -> bool:
    return importlib.util.find_spec("path_capi_python") is not None


def _fixture_gdx(dataset: str, mode: str, ifsub: int) -> Path:
    return FIXTURES_DIR / f"{dataset}_{mode}_ifsub{ifsub}.gdx"


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def _solve_and_measure(dataset: str, ifsub: int, mode: str, gdx: Path):
    """Build + seed + solve base→check→shock as an NLP; return
    {period: {"match": float, "code": int, "total": int}}."""
    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
    from _diff_core import gams_levels, list_populated_vars, split_t
    from pyomo.environ import value as V

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        PERIODS,
        GTAPMultiPeriodModel,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]

    if mode == "altertax":
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities

        pa = apply_altertax_elasticities(p, in_place=False)
        ac = GTAPClosureConfig(
            name="altertax",
            closure_type="MCP",
            capital_mobility="mobile",
            fix_endowments=False,
            fix_taxes=True,
            fix_technology=True,
            if_sub=bool(ifsub),
            numeraire="pnum",
        )
        solve_mode = "altertax"
    else:  # pure real-CES
        pa = p
        ac = GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            fix_endowments=False,
            fix_taxes=False,
            fix_technology=False,
            if_sub=bool(ifsub),
            numeraire="pnum",
        )
        solve_mode = "gtap"

    mp = GTAPMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)

    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode=solve_mode,
    )
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
                idx = (
                    (period,)
                    if not st
                    else ((st[0], period) if len(st) == 1 else (*st, period))
                )
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
                rel = (
                    da / abs(gval) if abs(gval) > 1e-12 else (0.0 if da < 1e-6 else 9e9)
                )
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
    _NLP_CASES,
    ids=[f"{r.dataset}-{r.mode}-ifsub{r.ifsub}" for r in nlp_rows()],
)
def test_gtap7_nlp_parity(dataset, ifsub, mode, floors, ci_status):
    if ci_status == "blocked":
        pytest.skip(f"blocked: {dataset}")
    if not _has_solver():
        pytest.skip("NLP solve toolchain (path_capi_python) not available")
    gdx = _fixture_gdx(dataset, mode, ifsub)
    if not gdx.exists():
        pytest.skip(f"fixture GDX missing: {gdx}")
    if not (DATASETS_DIR / dataset / "basedata.har").exists():
        pytest.skip(f"dataset HAR missing: {dataset}")

    per = _solve_and_measure(dataset, ifsub, mode, gdx)
    tag = f"{dataset}/{mode}/ifSUB={ifsub}"

    # 1. every stage must converge (code == 1)
    bad_codes = {k: v["code"] for k, v in per.items() if v["code"] != 1}
    assert not bad_codes, f"[{tag}] stages did not converge: {bad_codes}"

    # 2. every stage's measured match% must clear its versioned floor
    for stage, floor in floors.items():
        m = per[stage]
        assert m["total"] > 0, f"[{tag}/{stage}] no comparable cells"
        assert m["match"] >= floor, (
            f"[{tag}/{stage}] NLP match {m['match']:.2f}% < floor {floor}% "
            f"(over {m['total']} cells) — regression"
        )
