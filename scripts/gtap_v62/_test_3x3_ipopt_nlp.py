"""Test IPOPT NLP on gtap6_3x3 baseline + shocked.

Phase 3.25 reported +62.3860% VIWS (gap 0.04% vs GEMPACK +62.3585%).
If current code reproduces that, NLP regression is scale-dependent.
If not, regression is fundamental (model equations changed).
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import (
    SolverFactory, Objective, minimize, value, Var, Constraint
)
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_3x3")
SHOCK_KEY = ("Food", "USA", "EU_28")
IPOPT_PATH = ROOT / ".idaes-bin" / "ipopt.exe"

print(f"\n=== gtap6_3x3 IPOPT NLP test (Phase 3.25 reproduction) ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-9, params=params,
                          drop_dead_rows_threshold=0.0)
closure = pipe['closure']
print(f"\nfree={closure['free_vars']} cons={closure['active_cons']} mismatch={closure['mismatch']} baked={pipe['prebalance']['n_baked']}", flush=True)

init_values = {(v.name, idx): v[idx].value
               for v in model.component_objects(Var, active=True) for idx in v
               if not v[idx].fixed and v[idx].value is not None}

def _obj(m, anchor, weight=1.0e-6):
    return weight * sum(
        ((v[idx] - anchor.get((v.name, idx), 1.0))
         / max(abs(anchor.get((v.name, idx), 1.0)), 1.0)) ** 2
        for v in m.component_objects(Var, active=True)
        for idx in v if not v[idx].fixed
    )

model.obj = Objective(rule=lambda m: _obj(m, init_values), sense=minimize)

solver = SolverFactory("ipopt", executable=str(IPOPT_PATH))
solver.options.update({
    "max_iter": 5000,
    "tol": 1e-6,
    "linear_solver": "ma27",
    "nlp_scaling_method": "gradient-based",
    "print_level": 0,
})

# BASELINE
t0 = time.perf_counter()
res = solver.solve(model, tee=False)
elapsed_baseline = time.perf_counter() - t0
walras_baseline = abs(value(model.walras))
print(f"\n--- BASELINE ---", flush=True)
print(f"  status={res.solver.status}  term={res.solver.termination_condition}", flush=True)
print(f"  walras={walras_baseline:.4e}  time={elapsed_baseline:.1f}s", flush=True)
viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
vims_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))
print(f"  VIWS_0 (qxs*pmcif, CIF) = {viws_0:.4e}", flush=True)
print(f"  VIMS_0 (qxs*pms, agent) = {vims_0:.4e}", flush=True)

# SHOCKED
old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
model.tms[SHOCK_KEY] = new_tms
print(f"\n--- SHOCKED (tms {old_tms:.4f} -> {new_tms:.4f}) ---", flush=True)
t0 = time.perf_counter()
res = solver.solve(model, tee=False)
elapsed_shocked = time.perf_counter() - t0
walras_shocked = abs(value(model.walras))
print(f"  status={res.solver.status}  term={res.solver.termination_condition}", flush=True)
print(f"  walras={walras_shocked:.4e}  time={elapsed_shocked:.1f}s", flush=True)

viws_final = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
vims_final = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))
viws_pct = 100.0 * (viws_final - viws_0) / viws_0
vims_pct = 100.0 * (vims_final - vims_0) / vims_0
print(f"\n=== RESULT (Phase 3.25 reproduction) ===", flush=True)
print(f"VIWS (qxs*pmcif): {viws_pct:+.4f}%   GEMPACK ref +62.3585% gap={viws_pct - 62.3585:+.4f}pp", flush=True)
print(f"VIMS (qxs*pms):   {vims_pct:+.4f}%   GEMPACK ref +46.1227% gap={vims_pct - 46.1227:+.4f}pp", flush=True)
