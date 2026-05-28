"""Test IPOPT NLP on gtap6_15x10 baseline + shocked.

Phase 3.25 documented IPOPT NLP solving this dataset with 0.63% gap to
GEMPACK (VIWS +66.77% vs +66.36%). This script reproduces that result
end-to-end, confirming IPOPT NLP as the production path for v6.2.
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

DATA = Path("datasets/gtap6_15x10")
SHOCK_KEY = ("OtherFood", "USA", "EU_28")
IPOPT_PATH = ROOT / ".idaes-bin" / "ipopt.exe"

print(f"\n=== gtap6_15x10 IPOPT NLP test ===", flush=True)
print(f"IPOPT: {IPOPT_PATH} exists={IPOPT_PATH.exists()}", flush=True)

# Load model in NLP mode (walras Var + objective added)
sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()

# NLP pipeline: closure + prebalance bake (no MCP-specific dead-row drop).
# Use bake_tolerance=1e-3 (the Phase 3.25 default) for NLP — 1e-6 is the
# MCP-specific tightening from Phase 3.36 that helps PATH but may
# over-bake for IPOPT.
# Phase 3.25 reproduction: disable conditional_fixing — that was added
# in Phase 3.28 for PATH and may over-fix vars that NLP needs free.
pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-3, params=params,
                          conditional_fixing=False,
                          drop_dead_rows_threshold=0.0)
closure = pipe['closure']
print(f"\nfree_vars   = {closure['free_vars']}", flush=True)
print(f"active_cons = {closure['active_cons']}", flush=True)
print(f"mismatch    = {closure['mismatch']}", flush=True)
print(f"baked       = {pipe['prebalance']['n_baked']}", flush=True)

# Capture init values for the regularizer objective (Phase 3.25 standard).
# Tiny-weight (1e-6) regularizer keeps all free vars near their baseline
# initial values — provides direction for IPOPT without distorting the
# equilibrium response.
init_values = {
    (v.name, idx): v[idx].value
    for v in model.component_objects(Var, active=True)
    for idx in v
    if not v[idx].fixed and v[idx].value is not None
}

def _obj(model, anchor, weight: float = 1.0e-6):
    return weight * sum(
        ((v[idx] - anchor.get((v.name, idx), 1.0))
         / max(abs(anchor.get((v.name, idx), 1.0)), 1.0)) ** 2
        for v in model.component_objects(Var, active=True)
        for idx in v
        if not v[idx].fixed
    )

model.obj = Objective(rule=lambda m: _obj(m, init_values), sense=minimize)

# Build IPOPT solver
solver = SolverFactory("ipopt", executable=str(IPOPT_PATH))
solver.options.update({
    "max_iter": 5000,
    "tol": 1e-6,
    "linear_solver": "ma27",  # Phase 3.25 used ma27 via IDAES HSL
    "nlp_scaling_method": "gradient-based",
    "print_level": 3,  # see iteration info
})

# --- BASELINE ---
print(f"\n--- BASELINE solve (IPOPT NLP) ---", flush=True)
t0 = time.perf_counter()
res = solver.solve(model, tee=True)
elapsed_baseline = time.perf_counter() - t0
walras_baseline = abs(value(model.walras)) if hasattr(model, "walras") else 0.0
print(f"  status={res.solver.status}  term_cond={res.solver.termination_condition}", flush=True)
print(f"  walras = {walras_baseline:.4e}", flush=True)
print(f"  time = {elapsed_baseline:.1f}s", flush=True)

# Snapshot baseline state for VIWS comparison
viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
print(f"  VIWS_0 (qxs*pms at SHOCK_KEY) = {viws_0:.4e}", flush=True)

# --- SHOCKED ---
print(f"\n--- SHOCKED solve (IPOPT NLP, full 10% tariff cut in one shot) ---", flush=True)
old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
model.tms[SHOCK_KEY] = new_tms
print(f"  tms[{SHOCK_KEY}]: {old_tms:.6f} -> {new_tms:.6f}", flush=True)

t0 = time.perf_counter()
res = solver.solve(model, tee=False)
elapsed_shocked = time.perf_counter() - t0
walras_shocked = abs(value(model.walras)) if hasattr(model, "walras") else 0.0
print(f"  status={res.solver.status}  term_cond={res.solver.termination_condition}", flush=True)
print(f"  walras = {walras_shocked:.4e}", flush=True)
print(f"  time = {elapsed_shocked:.1f}s", flush=True)

# Compute VIWS shift
viws_final = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
viws_pct = 100.0 * (viws_final - viws_0) / viws_0
print(f"\n=== RESULT ===", flush=True)
print(f"VIWS baseline:     {viws_0:.4e}", flush=True)
print(f"VIWS shocked:      {viws_final:.4e}", flush=True)
print(f"FINAL VIWS % change: {viws_pct:+.4f}%", flush=True)
print(f"GEMPACK ref:       +66.359%", flush=True)
print(f"Gap to GEMPACK:    {viws_pct - 66.359:+.4f} pp", flush=True)
print(f"Relative error:    {abs(viws_pct - 66.359) / 66.359 * 100:.3f}%", flush=True)
print(f"\nTotal time: {elapsed_baseline + elapsed_shocked:.0f}s ({(elapsed_baseline + elapsed_shocked)/60:.1f} min)", flush=True)
