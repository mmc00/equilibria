"""Run PATH with UMFPACK option + capture PATH's C-level stdout.

Redirects FD 1 (C-level stdout) to a file ONLY around the
solve_v62_with_path_capi call, then restores it. PATH writes via
the C runtime's fprintf(stdout,...) which uses FD 1, so the redirect
captures it; Pyomo printf uses Python's sys.stdout which is buffered
separately.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

VENDOR = ROOT / "vendor" / "umfpack"
if VENDOR.exists():
    os.add_dll_directory(str(VENDOR))
    os.environ["PATH"] = str(VENDOR) + os.pathsep + os.environ.get("PATH", "")

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 50 "
    "convergence_tolerance 1e-4 "
    "factorization_method umfpack "
    f"factorization_library_name {VENDOR / 'libumfpack.dll'} "
    "output yes "
    "output_factorization_information yes "
)

from pyomo.environ import value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
PATH_LOG = ROOT / "path_capi_stdout.log"

print(f"\n=== gtap6_15x10 baseline with UMFPACK + VERBOSE log ===", flush=True)
print(f"PATH options: {os.environ['PATH_CAPI_OPTIONS']}", flush=True)
print(f"PATH C-stdout log: {PATH_LOG}\n", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)
print(f"free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

# Redirect FD 1 just for the PATH solve
sys.stdout.flush()
saved_fd1 = os.dup(1)
log_fd = os.open(str(PATH_LOG), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
os.dup2(log_fd, 1)
os.close(log_fd)

t0 = time.perf_counter()
res = solve_v62_with_path_capi(
    model, output=True,
    license_string=os.environ["PATH_LICENSE_STRING"],
    path_lib=os.environ["PATH_CAPI_LIBPATH"],
    lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
    variable_scaling=True, equation_scaling=True, perturbation=0.0,
)
elapsed = time.perf_counter() - t0

# Restore FD 1
os.dup2(saved_fd1, 1)
os.close(saved_fd1)

print(f"\nPATH BASELINE: tc={res.termination_code} r={res.residual:.4e} "
      f"major={res.major_iterations} time={elapsed:.0f}s", flush=True)

print(f"\n--- First 100 lines of PATH C-stdout log ---", flush=True)
try:
    with open(PATH_LOG, 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-8', errors='replace')
    lines = text.splitlines()
    for i, line in enumerate(lines[:100]):
        print(f"  {i:3d} | {line}")
    if len(lines) > 100:
        print(f"  ... [total {len(lines)} lines, {len(raw)} bytes]")
except Exception as e:
    print(f"  Could not read log: {e}")

# Grep for UMFPACK/LUSOL keywords
print(f"\n--- Lines mentioning UMFPACK/LUSOL/factorization ---", flush=True)
for line in text.splitlines():
    if any(k in line.lower() for k in ['umfpack', 'lusol', 'factorization', 'library', 'load']):
        print(f"  >> {line}")
