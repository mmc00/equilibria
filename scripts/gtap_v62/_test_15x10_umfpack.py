"""Test PATH with UMFPACK as linear solver on gtap6_15x10 baseline.

Phase 3.37: built libumfpack.dll from SuiteSparse 7.12.2 source
(MinGW gcc 13.2 + scipy_openblas32 + custom BLAS shim DLL). Now we
flip ``factorization_method`` from the default ``lusol`` to ``umfpack``
and measure (a) wall time, (b) baseline residual, (c) whether the
lottery resolves the same way.

The expected improvement vs LUSOL:
* 3-8× speedup on the per-Newton linear solve (Davis 2004 benchmarks)
* Possibly different basin behavior (UMFPACK uses different pivoting)
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Stage UMFPACK DLLs FIRST so PATH (which uses LoadLibrary) finds them.
VENDOR = ROOT / "vendor" / "umfpack"
if VENDOR.exists():
    # Add to DLL search path BEFORE loading any GAMS/PATH code.
    os.add_dll_directory(str(VENDOR))
    # Also prepend to PATH so child loads (PATH's LoadLibrary) work.
    os.environ["PATH"] = str(VENDOR) + os.pathsep + os.environ.get("PATH", "")

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

# Switch PATH's internal linear solver to UMFPACK.
os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 200 "
    "minor_iteration_limit 1000 "
    "convergence_tolerance 1e-4 "
    "factorization_method umfpack "
    f"factorization_library_name {VENDOR / 'libumfpack.dll'} "
)

from pyomo.environ import value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
print(f"\n=== gtap6_15x10 baseline with UMFPACK linear solver ===", flush=True)
print(f"Vendor dir: {VENDOR}", flush=True)
print(f"PATH options: {os.environ['PATH_CAPI_OPTIONS']}", flush=True)

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
res = solve_v62_with_path_capi(
    model, output=True,  # verbose so we can see what PATH actually uses
    license_string=os.environ["PATH_LICENSE_STRING"],
    path_lib=os.environ["PATH_CAPI_LIBPATH"],
    lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
    variable_scaling=True, equation_scaling=True, perturbation=0.0,
)
elapsed = time.perf_counter() - t0
print(f"\n  PATH BASELINE (UMFPACK): tc={res.termination_code} "
      f"r={res.residual:.4e} major={res.major_iterations} time={elapsed:.0f}s",
      flush=True)
