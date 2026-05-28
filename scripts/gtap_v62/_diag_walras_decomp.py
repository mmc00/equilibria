"""Decompose walras = sum_r [y - yp - yg - save_0 + savf] per region.

Identifies which region(s) contribute to the baseline walras imbalance
of ~2.6 (gtap6_3x3) / ~8.5 (gtap6_15x10). Walras should be 0 at
benchmark for a balanced SAM.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.environ import value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)

for dataset in ["gtap6_3x3", "gtap6_15x10"]:
    DATA = Path(f"datasets/{dataset}")
    if not DATA.exists():
        continue
    print(f"\n=== {dataset} walras decomposition ===", flush=True)

    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()

    total_walras = 0.0
    print(f"{'region':<12s} {'y_0':>12s} {'yp_0':>12s} {'yg_0':>12s} {'save_0':>12s} {'savf_0':>12s} {'residual':>12s}", flush=True)
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", flush=True)
    for r in model.r:
        y_r = value(model.y[r])
        yp_r = value(model.yp[r])
        yg_r = value(model.yg[r])
        save_r = value(model.save_0[r])
        savf_r = value(model.savf[r])
        residual = y_r - yp_r - yg_r - save_r + savf_r
        total_walras += residual
        print(f"{str(r):<12s} {y_r:>12.4e} {yp_r:>12.4e} {yg_r:>12.4e} {save_r:>12.4e} {savf_r:>12.4e} {residual:>+12.4e}", flush=True)

    print(f"\nTotal walras (baseline): {total_walras:+.4e}", flush=True)
    print(f"Total absolute imbalance: {sum(abs(value(model.y[r]) - value(model.yp[r]) - value(model.yg[r]) - value(model.save_0[r]) + value(model.savf[r])) for r in model.r):.4e}", flush=True)
    # Also report sum of savf — should be exactly 0 (rest-of-world adds up)
    print(f"sum_r savf_0 = {sum(value(model.savf[r]) for r in model.r):+.4e}  (should be 0)", flush=True)
