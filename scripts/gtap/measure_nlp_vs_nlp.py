"""NLP-vs-NLP parity: solve the Python multi-period model as an NLP (maximize
walras via IPOPT, EQUILIBRIA_GTAP_SOLVE_NLP=1) and compare cell-by-cell against a
GAMS ifMCP=0 (NLP/IPOPT) reference GDX — for base, check AND shock. Both sides use
the same IPOPT, so IPOPT's ~0.9% equality slop cancels and the match isolates
MODEL differences. Emits one JSON line to stdout.

Usage:
    EQUILIBRIA_GTAP_SOLVE_NLP=1 uv run python scripts/gtap/measure_nlp_vs_nlp.py \
        --dataset gtap7_3x3 --ifsub 1 --gdx output/gtap7_3x3_pure_local_bundle/out_3x3_nlp.gdx
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

DATASETS_DIR = ROOT / "datasets"
SKIP = {"walras", "ev", "cv", "uh", "u", "ug", "us"}
RF = {"pfa", "pfy", "pm", "pmcif", "pefob", "pwmg", "pp", "pdp", "pmp",
      "xwmg", "xmgm", "lambdamg", "imptx", "exptx"}
ALIAS = {"xa": "xaa", "xd": "xda", "xm": "xma", "pp": "pp_rai", "p": "p_rai",
         "ytaxInd": "ytax_ind", "ytaxind": "ytax_ind"}
TOLS = [1e-3, 5e-3, 1e-2]


def _strip(s):
    if isinstance(s, str) and len(s) > 2 and s[1] == "_" and s[0] in "acfr":
        return s[2:]
    return s


def solve_and_measure(dataset: str, if_sub: bool, gdx: Path):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel, PERIODS
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from pyomo.environ import value as V
    from _diff_core import gams_levels, list_populated_vars, split_t

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(basedata_path=d / "basedata.har", sets_path=d / "sets.har",
                    default_path=d / "default.prm", baserate_path=d / "baserate.har")
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
                           fix_endowments=False, fix_taxes=False, fix_technology=False,
                           if_sub=if_sub, numeraire="pnum")
    mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)

    res = solve_multiperiod(m, p, gc, ref_gdx=gdx,
                            skip_base_solve=True, mute_welfare=True,
                            seed_from_prior=False, holdfix_cd=False, mode="gtap")
    codes = {k: res[k]["code"] for k in res}

    def measure(period: str):
        tot = 0
        match = {t: 0 for t in TOLS}
        worst = []
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
                rel = d_abs / abs(gval) if abs(gval) > 1e-12 else (0.0 if d_abs < 1e-6 else 9e9)
                for t in TOLS:
                    if d_abs <= 1e-6 or rel <= t:
                        match[t] += 1
                if not (d_abs <= 1e-6 or rel <= 1e-2):
                    worst.append((rel, vn, fk, val, gval))
        worst.sort(reverse=True)
        return {
            "total": tot,
            "match_pct": {"0.1%": round(100.0 * match[1e-3] / max(tot, 1), 2),
                          "0.5%": round(100.0 * match[5e-3] / max(tot, 1), 2),
                          "1%": round(100.0 * match[1e-2] / max(tot, 1), 2)},
            "worst_cells": [{"var": vn, "key": list(fk), "rel_pct": round(100 * rel, 3),
                             "py": round(pyv, 5), "gams": round(gv, 5)}
                            for rel, vn, fk, pyv, gv in worst[:6]],
        }

    return {"dataset": dataset, "ifsub": int(if_sub), "mode": "gtap-nlp",
            "codes": codes, "base": measure("base"),
            "check": measure("check"), "shock": measure("shock")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ifsub", type=int, required=True, choices=[0, 1])
    ap.add_argument("--gdx", required=True, type=Path, help="GAMS ifMCP=0 NLP reference GDX")
    args = ap.parse_args()
    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"  # force the NLP solve path
    try:
        print(json.dumps(solve_and_measure(args.dataset, bool(args.ifsub), args.gdx)))
        return 0
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
