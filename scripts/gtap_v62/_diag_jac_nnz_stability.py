"""Quick diagnostic: how stable is the Jacobian nnz across runs?

Builds the gtap6_15x10 model identically to our test scripts and
prints the Jacobian non-zero count. Repeated execution should
reveal whether nnz is deterministic across processes or varies
(hash seed effect).
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import _deterministic_startup  # noqa: F401  Phase 3.37

os.environ.setdefault("PATH_LICENSE_STRING", "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0")
os.environ.setdefault("PATH_CAPI_LIBPATH", "C:/GAMS/53/path52.dll")
os.environ.setdefault("PATH_CAPI_LIBLUSOL", "C:/GAMS/53/lusol.dll")

from pyomo.environ import Constraint, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

# Ensure path_capi_python is importable
PATH_CAPI_SRC = Path("c:/Documentos/proyectos/path-capi-python/src")
if PATH_CAPI_SRC.exists() and str(PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(PATH_CAPI_SRC))
from path_capi_python import PyomoMCPAdapter  # type: ignore

DATA = Path("datasets/gtap6_15x10")

hash_seed = os.environ.get("PYTHONHASHSEED", "<random>")
print(f"PYTHONHASHSEED = {hash_seed}", flush=True)
print(f"sys.flags.hash_randomization = {sys.flags.hash_randomization}", flush=True)

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

print(f"\nfree_vars = {len(free_vars)}  active_cons = {len(active_cons)}", flush=True)
print(f"baked = {pipe['prebalance']['n_baked']}", flush=True)

fixed_vars = [v[idx] for v in model.component_objects(Var, active=True)
              for idx in v if v[idx].fixed]
print(f"fixed_vars = {len(fixed_vars)}", flush=True)
fixed_by_family = {}
for fv in fixed_vars:
    name = fv.parent_component().name
    fixed_by_family[name] = fixed_by_family.get(name, 0) + 1
print(f"fixed_by_family (sorted): {sorted(fixed_by_family.items())}", flush=True)

# Build Jacobian using path-capi adapter (same path our tests use)
adapter = PyomoMCPAdapter()
data = adapter.build_nonlinear_from_equality_constraints(
    model,
    constraints=active_cons,
    variables=free_vars,
    jacobian_eval_mode="reverse_numeric",
)

print(f"\nJacobian nnz = {len(data.jacobian_structure.row_indices)}", flush=True)
print(f"variable_names hash = {hash(tuple(data.variable_names)) & 0xFFFFFFFF:08x}", flush=True)
print(f"x0[:5] = {[f'{v:.6e}' for v in data.x0[:5]]}", flush=True)
