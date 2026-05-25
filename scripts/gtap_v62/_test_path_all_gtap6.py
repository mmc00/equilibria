"""PATH sweep across all 5 gtap6 datasets with Phase 3.28 conditional fixing."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import value, Var

from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore


os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"


DATASETS = [
    {"name": "gtap6_3x3",  "dir": "datasets/gtap6_3x3",  "shock": ("Food", "USA", "EU_28")},
    {"name": "gtap6_5x5",  "dir": "datasets/gtap6_5x5",  "shock": ("Food", "USA", "EU_28")},
    {"name": "gtap6_10x7", "dir": "datasets/gtap6_10x7", "shock": ("FoodProc", "USA", "EU_28")},
    {"name": "gtap6_15x10","dir": "datasets/gtap6_15x10","shock": ("OtherFood", "USA", "EU_28")},
    # 20x41 separately given size — uncomment to include
    # {"name": "gtap6_20x41","dir": "datasets/gtap6_20x41","shock": ("FoodProd", "USA", "EU_28")},
]


def solve_dataset(spec):
    name = spec["name"]
    dir_ = Path(spec["dir"])
    shock = spec["shock"]

    print(f"\n{'='*72}\nPATH on {name}\n{'='*72}", flush=True)
    t0 = time.perf_counter()

    sets = GTAPv62Sets()
    sets.load_from_har(dir_ / "sets.har", default_path=dir_ / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=dir_ / "basedata.har", default_prm_path=dir_ / "default.prm", sets=sets)
    model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
    pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-3, params=params)
    cond = pipe["closure"].get("conditional_fixing", {})
    t1 = time.perf_counter()
    print(f"Build+closure: {t1-t0:.1f}s | cond_fixed={cond.get('n_fixed_total',0)} | "
          f"free={pipe['free_vars']} mismatch={pipe['mismatch']}", flush=True)

    # Capture baseline init.
    base_init = {(v.name, idx): value(v[idx])
                 for v in model.component_objects(Var, active=True)
                 for idx in v if not v[idx].fixed and v[idx].value is not None}

    # PATH BASELINE
    res_b = solve_v62_with_path_capi(
        model, output=False,
        license_string=os.environ["PATH_LICENSE_STRING"],
        path_lib=os.environ["PATH_CAPI_LIBPATH"],
        lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
        variable_scaling=True, equation_scaling=True, perturbation=0.0,
    )
    t2 = time.perf_counter()
    print(f"PATH BASELINE: {t2-t1:.1f}s | term_code={res_b.termination_code} "
          f"residual={res_b.residual:.2e} | major={res_b.major_iterations}", flush=True)

    base_vals = {(v.name, idx): value(v[idx])
                 for v in model.component_objects(Var, active=True)
                 for idx in v if not v[idx].fixed and v[idx].value is not None}

    # Apply shock.
    i_shk, s_shk, d_shk = shock
    old_tms = value(model.tms[i_shk, s_shk, d_shk])
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    model.tms[i_shk, s_shk, d_shk] = new_tms

    # PATH SHOCKED
    res_s = solve_v62_with_path_capi(
        model, output=False,
        license_string=os.environ["PATH_LICENSE_STRING"],
        path_lib=os.environ["PATH_CAPI_LIBPATH"],
        lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
        variable_scaling=True, equation_scaling=True, perturbation=0.0,
    )
    t3 = time.perf_counter()
    print(f"PATH SHOCKED:  {t3-t2:.1f}s | term_code={res_s.termination_code} "
          f"residual={res_s.residual:.2e} | major={res_s.major_iterations}", flush=True)

    pms_s = value(model.pms[i_shk, s_shk, d_shk]); pms_0 = base_vals[("pms", shock)]
    qxs_s = value(model.qxs[i_shk, s_shk, d_shk]); qxs_0 = base_vals[("qxs", shock)]
    pmcif_s = value(model.pmcif[i_shk, s_shk, d_shk]); pmcif_0 = base_vals[("pmcif", shock)]

    viws_pct = (pmcif_s*qxs_s)/(pmcif_0*qxs_0)*100 - 100 if pmcif_0*qxs_0 > 0 else 0
    vims_pct = (pms_s*qxs_s)/(pms_0*qxs_0)*100 - 100 if pms_0*qxs_0 > 0 else 0
    qxs_pct = qxs_s/qxs_0*100 - 100 if qxs_0 > 0 else 0

    return {
        "name": name, "free": pipe["free_vars"],
        "cond_fixed": cond.get("n_fixed_total", 0),
        "base_term": res_b.termination_code, "base_res": res_b.residual, "base_time": t2-t1,
        "shock_term": res_s.termination_code, "shock_res": res_s.residual, "shock_time": t3-t2,
        "VIWS_pct": viws_pct, "VIMS_pct": vims_pct, "qxs_pct": qxs_pct,
    }


def main():
    results = []
    for spec in DATASETS:
        try:
            r = solve_dataset(spec)
            results.append(r)
        except Exception as e:
            print(f"ERROR on {spec['name']}: {e}", flush=True)
            traceback.print_exc()
            results.append({"name": spec["name"], "error": str(e)})

    print(f"\n{'='*100}\nSUMMARY (PATH on gtap6 datasets, Phase 3.28 conditional fixing)\n{'='*100}", flush=True)
    print(f"{'Dataset':14s} {'Free':>7s} {'CondFix':>7s} "
          f"{'BaseTerm':>9s} {'BaseRes':>10s} {'ShockTerm':>10s} {'ShockRes':>10s} "
          f"{'VIWS%':>10s}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['name']:14s}  ERROR: {r['error']}", flush=True)
            continue
        print(f"{r['name']:14s} {r['free']:>7d} {r['cond_fixed']:>7d} "
              f"{r['base_term']:>9d} {r['base_res']:>10.2e} "
              f"{r['shock_term']:>10d} {r['shock_res']:>10.2e} "
              f"{r['VIWS_pct']:>+10.3f}", flush=True)


if __name__ == "__main__":
    main()
