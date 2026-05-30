"""Sweep IPOPT NLP across gtap6_3x3, 5x5, 10x7, 15x10 (skip 20x41 — separate).

Run after Phase 3.38 fix to confirm parity scales sub-2% across dataset sizes.
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

IPOPT_PATH = Path(os.environ.get("EQUILIBRIA_IPOPT", str(ROOT / ".idaes-bin" / "ipopt")))

DATASETS = [
    {"name": "gtap6_3x3",   "shock": ("Food",       "USA", "EU_28"), "gempack_viws": 62.359},
    {"name": "gtap6_5x5",   "shock": ("Food",       "USA", "EU_28"), "gempack_viws": 64.553},
    {"name": "gtap6_10x7",  "shock": ("FoodProc",   "USA", "EU_28"), "gempack_viws": 64.391},
    {"name": "gtap6_15x10", "shock": ("OtherFood",  "USA", "EU_28"), "gempack_viws": 66.359},
]

results = []
for spec in DATASETS:
    DATA = Path(f"datasets/{spec['name']}")
    if not DATA.exists():
        print(f"\nSKIP {spec['name']}: directory not found", flush=True)
        continue
    SHOCK_KEY = spec['shock']
    print(f"\n{'='*70}\n{spec['name']} | shock={SHOCK_KEY} | GEMPACK ref +{spec['gempack_viws']:.3f}%\n{'='*70}", flush=True)

    t_load = time.perf_counter()
    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
    pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-3, params=params,
                              drop_dead_rows_threshold=0.0)
    print(f"load+build+closure: {time.perf_counter()-t_load:.1f}s | free={pipe['closure']['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

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

    t0 = time.perf_counter()
    res_b = solver.solve(model, tee=False)
    elapsed_b = time.perf_counter() - t0
    walras_b = abs(value(model.walras))
    viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
    print(f"BASELINE: term={res_b.solver.termination_condition} walras={walras_b:.2e} time={elapsed_b:.1f}s", flush=True)

    old_tms = float(value(model.tms[SHOCK_KEY]))
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    model.tms[SHOCK_KEY] = new_tms
    t0 = time.perf_counter()
    res_s = solver.solve(model, tee=False)
    elapsed_s = time.perf_counter() - t0
    walras_s = abs(value(model.walras))
    viws_f = float(value(model.qxs[SHOCK_KEY]) * value(model.pmcif[SHOCK_KEY]))
    viws_pct = 100.0 * (viws_f - viws_0) / viws_0
    gap_pp = viws_pct - spec['gempack_viws']
    rel_pct = abs(gap_pp) / spec['gempack_viws'] * 100
    print(f"SHOCKED:  term={res_s.solver.termination_condition} walras={walras_s:.2e} time={elapsed_s:.1f}s", flush=True)
    print(f"VIWS = {viws_pct:+.4f}%  vs GEMPACK +{spec['gempack_viws']:.3f}%  gap={gap_pp:+.3f}pp ({rel_pct:.2f}%)", flush=True)

    results.append({
        "name": spec['name'], "viws_pct": viws_pct, "gempack": spec['gempack_viws'],
        "gap_pp": gap_pp, "rel_pct": rel_pct, "walras_b": walras_b, "walras_s": walras_s,
        "time_b": elapsed_b, "time_s": elapsed_s,
    })

# Summary
print(f"\n\n{'='*70}\nSUMMARY (Phase 3.38)\n{'='*70}", flush=True)
print(f"{'dataset':<14s} {'VIWS%':>10s} {'GEMPACK%':>10s} {'gap_pp':>8s} {'rel%':>7s} {'walras_s':>11s} {'time(s)':>8s}", flush=True)
for r in results:
    print(f"{r['name']:<14s} {r['viws_pct']:>+10.4f} {r['gempack']:>+10.3f} {r['gap_pp']:>+8.3f} {r['rel_pct']:>6.2f}% {r['walras_s']:>11.2e} {r['time_b']+r['time_s']:>8.1f}", flush=True)
