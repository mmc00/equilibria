"""Test IPOPT NLP on gtap6_20x41 (~290K vars) with Phase 3.38 sav-Var fix.

This is the dataset that NO solver previously closed (PATH didn't
converge, IPOPT didn't converge). If the Phase 3.38 fix scales,
this should now produce sub-2% parity vs GEMPACK.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import SolverFactory, Objective, minimize, value, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_20x41")
# Phase 3.25 doc used shock: FoodProd/USA/EU_28
# We need to verify the actual commodity name in this dataset.
SHOCK_KEY = ("FoodProd", "USA", "EU_28")
IPOPT_PATH = ROOT / ".idaes-bin" / "ipopt.exe"

print(f"\n=== gtap6_20x41 IPOPT NLP test (Phase 3.38) ===", flush=True)
print(f"Shock target: tms{SHOCK_KEY}", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
print(f"Sectors |i|={len(sets.i)} regions |r|={len(sets.r)} factors |f|={len(sets.f)}", flush=True)
print(f"Available commodities: {list(sets.i)[:10]} ...", flush=True)

# Verify shock commodity exists
if SHOCK_KEY[0] not in sets.i:
    # Try alternatives
    food_alts = [c for c in sets.i if 'food' in c.lower() or 'proc' in c.lower()]
    print(f"WARNING: '{SHOCK_KEY[0]}' not in sets.i. Food-related alternatives: {food_alts}", flush=True)
    if food_alts:
        SHOCK_KEY = (food_alts[0], SHOCK_KEY[1], SHOCK_KEY[2])
        print(f"Using {SHOCK_KEY} instead", flush=True)

params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-3, params=params,
                          conditional_fixing=True,
                          drop_dead_rows_threshold=0.0)
closure = pipe['closure']
print(f"\nfree_vars   = {closure['free_vars']}", flush=True)
print(f"active_cons = {closure['active_cons']}", flush=True)
print(f"mismatch    = {closure['mismatch']}", flush=True)
print(f"baked       = {pipe['prebalance']['n_baked']}", flush=True)

# For 290K vars the regularizer-over-all-vars objective triggers Pyomo's
# NL writer stack overflow (return code 0xC0000005). Use a tiny objective
# touching only the walras variable; with mismatch=0 closure the model
# is already determined by the constraints — the objective just provides
# IPOPT a direction.
model.obj = Objective(expr=model.walras ** 2, sense=minimize)
solver = SolverFactory("ipopt", executable=str(IPOPT_PATH))
# MUMPS for 290K vars — MA27 is tuned for moderate-size (~50K) problems
# and degrades on very large sparse Jacobians. MUMPS handles >100K vars
# with better memory locality and supports multifrontal LU.
solver.options.update({
    "max_iter": 3000,
    "tol": 1e-6,
    "linear_solver": "mumps",
    "nlp_scaling_method": "gradient-based",
    "print_level": 3,  # iteration log so we can see progress
})

# --- BASELINE ---
print(f"\n--- BASELINE solve (IPOPT NLP, linear_solver=mumps) ---", flush=True)
t0 = time.perf_counter()
res = solver.solve(model, tee=True)
elapsed_b = time.perf_counter() - t0
walras_b = abs(value(model.walras))
print(f"  status={res.solver.status}  term={res.solver.termination_condition}", flush=True)
print(f"  walras = {walras_b:.4e}  time = {elapsed_b:.1f}s", flush=True)
viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
print(f"  VIWS_0 (qxs*pmcif) = {viws_0:.4e}", flush=True)

# --- SHOCKED ---
old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
model.tms[SHOCK_KEY] = new_tms
print(f"\n--- SHOCKED (tms {old_tms:.4f} -> {new_tms:.4f}) ---", flush=True)
t0 = time.perf_counter()
res = solver.solve(model, tee=False)
elapsed_s = time.perf_counter() - t0
walras_s = abs(value(model.walras))
print(f"  status={res.solver.status}  term={res.solver.termination_condition}", flush=True)
print(f"  walras = {walras_s:.4e}  time = {elapsed_s:.1f}s", flush=True)

viws_f = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
viws_pct = 100.0 * (viws_f - viws_0) / viws_0
print(f"\n=== RESULT ===", flush=True)
print(f"VIWS baseline:     {viws_0:.4e}", flush=True)
print(f"VIWS shocked:      {viws_f:.4e}", flush=True)
print(f"FINAL VIWS % change: {viws_pct:+.4f}%", flush=True)
print(f"GEMPACK ref:       +51.432% (Phase 3.25 doc, FoodProd/USA/EU_28)", flush=True)
print(f"Gap to GEMPACK:    {viws_pct - 51.432:+.4f} pp", flush=True)
print(f"Relative error:    {abs(viws_pct - 51.432) / 51.432 * 100:.3f}%", flush=True)
print(f"Total time: {elapsed_b + elapsed_s:.0f}s ({(elapsed_b + elapsed_s)/60:.1f} min)", flush=True)
