"""Decompose walras AFTER shock to find which income/spending term breaks.

Baseline walras converges to 0 with bake_tolerance=1e-9, so the model
is correctly calibrated. But after a 10% tariff shock, walras blows up
to ~2781 (gtap6_3x3) / ~9870 (gtap6_15x10). This script decomposes
each region's contribution to walras at the shocked solution.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import SolverFactory, Objective, minimize, value, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_3x3")
SHOCK_KEY = ("Food", "USA", "EU_28")
IPOPT_PATH = ROOT / ".idaes-bin" / "ipopt.exe"

print(f"\n=== gtap6_3x3 walras decomposition pre/post shock ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-9, params=params,
                          drop_dead_rows_threshold=0.0)

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

def dump_walras(label):
    print(f"\n--- {label}: walras decomposition ---", flush=True)
    print(f"{'region':<12s} {'y':>12s} {'yp':>12s} {'yg':>12s} {'sav':>12s} {'savf':>12s} {'(y-yp-yg-sav+savf)':>22s}", flush=True)
    total = 0.0
    for r in model.r:
        y_r = value(model.y[r])
        yp_r = value(model.yp[r])
        yg_r = value(model.yg[r])
        sav_r = value(model.sav[r]) if hasattr(model, "sav") else value(model.save_0[r])
        savf_r = value(model.savf[r])
        residual = y_r - yp_r - yg_r - sav_r + savf_r
        total += residual
        print(f"{str(r):<12s} {y_r:>12.4e} {yp_r:>12.4e} {yg_r:>12.4e} {sav_r:>12.4e} {savf_r:>12.4e} {residual:>+22.4e}", flush=True)
    walras_var = value(model.walras) if hasattr(model, "walras") else 0.0
    print(f"  sum = {total:+.4e}   walras Var = {walras_var:+.4e}", flush=True)
    print(f"  sum_savf = {sum(value(model.savf[r]) for r in model.r):+.4e}", flush=True)

# --- BASELINE solve ---
solver.solve(model, tee=False)
dump_walras("BASELINE")

# Snapshot baseline values for delta analysis
baseline_y  = {r: value(model.y[r])  for r in model.r}
baseline_yp = {r: value(model.yp[r]) for r in model.r}
baseline_yg = {r: value(model.yg[r]) for r in model.r}
baseline_savf = {r: value(model.savf[r]) for r in model.r}
baseline_sav = {r: value(model.sav[r]) if hasattr(model, "sav") else value(model.save_0[r]) for r in model.r}

# --- SHOCKED ---
old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
model.tms[SHOCK_KEY] = new_tms
print(f"\nApplying shock: tms[{SHOCK_KEY}] {old_tms:.4f} -> {new_tms:.4f}", flush=True)

solver.solve(model, tee=False)
dump_walras("SHOCKED")

# Delta analysis: how much did each term shift?
print(f"\n--- DELTA (shocked - baseline) ---", flush=True)
print(f"{'region':<12s} {'dy':>14s} {'dyp':>14s} {'dyg':>14s} {'dsav':>14s} {'dsavf':>14s} {'expect=dy-...':>20s}", flush=True)
for r in model.r:
    dy = value(model.y[r]) - baseline_y[r]
    dyp = value(model.yp[r]) - baseline_yp[r]
    dyg = value(model.yg[r]) - baseline_yg[r]
    sav_curr = value(model.sav[r]) if hasattr(model, "sav") else value(model.save_0[r])
    dsav = sav_curr - baseline_sav[r]
    dsavf = value(model.savf[r]) - baseline_savf[r]
    expected = dy - dyp - dyg - dsav + dsavf
    print(f"{str(r):<12s} {dy:>+14.4e} {dyp:>+14.4e} {dyg:>+14.4e} {dsav:>+14.4e} {dsavf:>+14.4e} {expected:>+20.4e}", flush=True)
