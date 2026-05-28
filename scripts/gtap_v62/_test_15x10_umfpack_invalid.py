"""Sanity test: point factorization_library_name to a nonexistent DLL.

If PATH actually tries to load UMFPACK, this should fail clearly.
If PATH silently falls back to LUSOL, this will produce the same
LUSOL result as a normal run (indicating UMFPACK was not actually
being used in the previous test).
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

# Intentionally wrong UMFPACK path
os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 50 "
    "convergence_tolerance 1e-4 "
    "factorization_method umfpack "
    "factorization_library_name C:/NONEXISTENT_FAKE_PATH/libumfpack.dll "
    "output yes "
)

from pyomo.environ import value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
print("\n=== gtap6_15x10 baseline: UMFPACK with INVALID library path ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)
print(f"  free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

t0 = time.perf_counter()
try:
    res = solve_v62_with_path_capi(
        model, output=True,
        license_string=os.environ["PATH_LICENSE_STRING"],
        path_lib=os.environ["PATH_CAPI_LIBPATH"],
        lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
        variable_scaling=True, equation_scaling=True, perturbation=0.0,
    )
    elapsed = time.perf_counter() - t0
    print(f"  PATH (UMFPACK fake): tc={res.termination_code} "
          f"r={res.residual:.4e} major={res.major_iterations} time={elapsed:.0f}s",
          flush=True)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
