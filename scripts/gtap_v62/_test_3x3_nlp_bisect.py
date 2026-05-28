"""Minimal bisect-friendly NLP test on gtap6_3x3.

Tries each call signature in turn so it works across Phase 3.26-3.37.
Prints walras at baseline + shocked + VIWS%.
"""
import os, sys, time, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import SolverFactory, Objective, minimize, value, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
import _make_square as _ms  # type: ignore

DATA = Path("datasets/gtap6_3x3")
SHOCK_KEY = ("Food", "USA", "EU_28")
IPOPT_PATH = ROOT / ".idaes-bin" / "ipopt.exe"

phase_tag = os.environ.get("PHASE_TAG", "?")
print(f"\n=== gtap6_3x3 NLP test [PHASE {phase_tag}] ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)

# Try to build with mode="nlp" (Phase 3.26+) or fallback to plain build
try:
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
except TypeError:
    model = GTAPv62ModelEquations(sets, params).build_model()

# Phase compatibility: use apply_v62_pipeline (3.26+) or call closure+bake directly (3.25)
if hasattr(_ms, "apply_v62_pipeline"):
    sig = inspect.signature(_ms.apply_v62_pipeline)
    kwargs = {"bake_tolerance": 1.0e-3}
    if "mode" in sig.parameters: kwargs["mode"] = "nlp"
    if "params" in sig.parameters: kwargs["params"] = params
    if "drop_dead_rows_threshold" in sig.parameters: kwargs["drop_dead_rows_threshold"] = 0.0
    if "conditional_fixing" in sig.parameters: kwargs["conditional_fixing"] = True
    print(f"using apply_v62_pipeline with kwargs: {list(kwargs.keys())}", flush=True)
    pipe = _ms.apply_v62_pipeline(model, **kwargs)
    cl = pipe.get('closure', {})
    pb = pipe.get('prebalance', {})
else:
    # Phase 3.25: closure + bake separately
    print(f"using closure+bake (Phase 3.25 style)", flush=True)
    cl = _ms.apply_v62_closure_and_square(model)
    pb = _ms.bake_baseline_residuals_as_slacks(model, tolerance=1.0e-3)

print(f"free={cl.get('free_vars', '?')} cons={cl.get('active_cons', '?')} "
      f"mismatch={cl.get('mismatch', '?')} baked={pb.get('n_baked', '?')}", flush=True)

init_values = {(v.name, idx): v[idx].value
               for v in model.component_objects(Var, active=True) for idx in v
               if not v[idx].fixed and v[idx].value is not None}

def _obj(m, anchor, weight=1.0e-6):
    return weight * sum(
        ((v[idx] - anchor.get((v.name, idx), 1.0))
         / max(abs(anchor.get((v.name, idx), 1.0)), 1.0)) ** 2
        for v in m.component_objects(Var, active=True) for idx in v
        if not v[idx].fixed
    )

model.obj = Objective(rule=lambda m: _obj(m, init_values), sense=minimize)

solver = SolverFactory("ipopt", executable=str(IPOPT_PATH))
solver.options.update({"max_iter": 5000, "tol": 1e-6, "linear_solver": "ma27", "print_level": 0})

# BASELINE
t0 = time.perf_counter()
res = solver.solve(model, tee=False)
walras_b = abs(value(model.walras)) if hasattr(model, "walras") else 0.0
print(f"\nBASELINE: term={res.solver.termination_condition} walras={walras_b:.4e} time={time.perf_counter()-t0:.1f}s", flush=True)
viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))

# SHOCKED
old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
model.tms[SHOCK_KEY] = new_tms
t0 = time.perf_counter()
res = solver.solve(model, tee=False)
walras_s = abs(value(model.walras)) if hasattr(model, "walras") else 0.0
viws_f = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))
viws_pct = 100.0 * (viws_f - viws_0) / viws_0
print(f"SHOCKED:  term={res.solver.termination_condition} walras={walras_s:.4e} time={time.perf_counter()-t0:.1f}s", flush=True)

print(f"\n[PHASE {phase_tag}] VIWS={viws_pct:+.4f}% (GEMPACK ref +62.359%) gap={viws_pct - 62.359:+.4f}pp", flush=True)
