"""Test gtap6_15x10 PATH with extended major_iteration_limit + verbose."""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Extend PATH iteration budget and slightly relax convergence
os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 1000 "
    "minor_iteration_limit 1000 "
    "convergence_tolerance 1e-4 "
    "crash_method none "
)
os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

from pyomo.environ import value, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
SHOCK = ("OtherFood", "USA", "EU_28")

print(f"\n=== gtap6_15x10 baseline with extended PATH options ===", flush=True)
print(f"PATH options: {os.environ['PATH_CAPI_OPTIONS']}\n", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har", default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-3, params=params,
                          drop_dead_rows_threshold=1.0e-6)
print(f"Model: free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}\n", flush=True)

t0 = time.perf_counter()
res = solve_v62_with_path_capi(
    model, output=False,
    license_string=os.environ["PATH_LICENSE_STRING"],
    path_lib=os.environ["PATH_CAPI_LIBPATH"],
    lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
    variable_scaling=True, equation_scaling=True, perturbation=0.0,
)
elapsed = time.perf_counter() - t0
print(f"BASELINE result: tc={res.termination_code} r={res.residual:.4e}", flush=True)
print(f"  major={res.major_iterations} minor={res.minor_iterations}", flush=True)
print(f"  time={elapsed:.0f}s", flush=True)
