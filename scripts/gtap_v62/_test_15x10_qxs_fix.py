"""Test Phase 3.37 — apply_v62_diagonal_redundancy_fix on gtap6_15x10.

Should detect and deactivate exactly the 13 eq_qxs(i, r, r) cells that
PATH's hook identified as structurally singular:
  - eq_qxs(VegFruit, r, r) for 7 regions
  - eq_qxs(Construction, r, r) for 8 regions
  (Total 15 cells if all dominant; PATH found 13 singular at run time
   because some basis updates resolved 2 of them; baseline should
   detect 15.)
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

VENDOR = ROOT / "vendor" / "umfpack"
if VENDOR.exists():
    os.add_dll_directory(str(VENDOR))

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"
os.environ["PATH_CAPI_OUTPUT_HOOK"] = "stderr"

os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 50 "
    "convergence_tolerance 1e-4 "
    "output yes "
)

from pyomo.environ import value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
print(f"\n=== gtap6_15x10 baseline with Phase 3.37 qxs diagonal fix ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)

dr = pipe['closure'].get('diagonal_redundancy', {})
print(f"\n  diagonal_redundancy: n_deactivated={dr.get('n_deactivated', 0)}", flush=True)
print(f"  by_family: {dr.get('by_family', {})}", flush=True)
print(f"  detail:", flush=True)
for d in dr.get('deactivated', []):
    print(f"    eq_qxs({d['i']:>15s}, {d['r']:>10s}, {d['r']:>10s})  diag_share={d['diag_share']:.4f}  vxmd_diag={d['vxmd_diag']:.2e}", flush=True)

print(f"\n  free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

t0 = time.perf_counter()
res = solve_v62_with_path_capi(
    model, output=True,
    license_string=os.environ["PATH_LICENSE_STRING"],
    path_lib=os.environ["PATH_CAPI_LIBPATH"],
    lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
    variable_scaling=True, equation_scaling=True, perturbation=0.0,
)
elapsed = time.perf_counter() - t0
print(f"\nPATH BASELINE: tc={res.termination_code} r={res.residual:.4e} "
      f"major={res.major_iterations} time={elapsed:.0f}s", flush=True)
