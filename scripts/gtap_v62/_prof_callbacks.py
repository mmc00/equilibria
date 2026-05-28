"""Profile Pyomo's callback_f vs callback_jac evaluation cost on gtap6_15x10.

This tells us whether the wall-time bottleneck is in:
  * Pyomo expression evaluation (suggests NL writer / ASL is the next step)
  * Linear solve (suggests UMFPACK / MA48 etc. is the next step)
  * PATH internals (suggests we need a different MCP framework)
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

PATH_CAPI_SRC = Path("c:/Documentos/proyectos/path-capi-python/src")
sys.path.insert(0, str(PATH_CAPI_SRC))

from pyomo.environ import Var, Constraint
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore
from path_capi_python import PyomoMCPAdapter  # type: ignore

DATA = Path("datasets/gtap6_15x10")
print(f"\n=== gtap6_15x10 callback profiling ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)

free_vars = [v[idx] for v in model.component_objects(Var, active=True)
             for idx in v if not v[idx].fixed]
active_cons = [c[idx] for c in model.component_objects(Constraint, active=True)
               for idx in c if c[idx].active]
print(f"free={len(free_vars)} active_cons={len(active_cons)}", flush=True)

t0 = time.perf_counter()
adapter = PyomoMCPAdapter()
data = adapter.build_nonlinear_from_equality_constraints(
    model, constraints=active_cons, variables=free_vars,
    jacobian_eval_mode="reverse_numeric",
)
t_build = time.perf_counter() - t0
print(f"\nbuild_nonlinear_from_equality_constraints: {t_build:.1f}s", flush=True)
print(f"  variables: {len(data.variable_names)}", flush=True)
print(f"  jacobian nnz: {len(data.jacobian_structure.row_indices)}", flush=True)

x0 = list(data.x0)

print(f"\n--- Timing callback_f over 5 iterations ---", flush=True)
times_f = []
for i in range(5):
    t0 = time.perf_counter()
    f = data.callback_f(x0)
    times_f.append(time.perf_counter() - t0)
mean_f = sum(times_f) / len(times_f)
print(f"  callback_f: mean={mean_f:.2f}s  min={min(times_f):.2f}s  max={max(times_f):.2f}s", flush=True)

print(f"\n--- Timing callback_jac over 5 iterations ---", flush=True)
times_j = []
for i in range(5):
    t0 = time.perf_counter()
    j = data.callback_jac(x0)
    times_j.append(time.perf_counter() - t0)
mean_j = sum(times_j) / len(times_j)
print(f"  callback_jac: mean={mean_j:.2f}s  min={min(times_j):.2f}s  max={max(times_j):.2f}s", flush=True)

print(f"\n--- Summary ---", flush=True)
print(f"  Adapter build (one-time): {t_build:.1f}s", flush=True)
print(f"  Per-Newton-iter Pyomo cost: ~{mean_f + mean_j:.1f}s (F={mean_f:.1f}s + J={mean_j:.1f}s)", flush=True)
print(f"  At 13 Newton iters: ~{13*(mean_f + mean_j):.0f}s of Pyomo cost", flush=True)
print(f"  Observed wall time ~295s, so Pyomo fraction: ~{100*13*(mean_f+mean_j)/295:.0f}%", flush=True)
