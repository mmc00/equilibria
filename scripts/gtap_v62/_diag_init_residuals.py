"""Find which constraints have non-zero residual at the INIT point of NLP mode.

After bake_baseline_residuals_as_slacks, all *baked* constraints should have
residual ≈ 0. But IPOPT moves walras away from 0, meaning SOMETHING is
unbalanced at init. Identify the unbalanced families.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from collections import defaultdict
from pyomo.environ import Constraint, value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_3x3")
print(f"\n=== gtap6_3x3 init-point residual breakdown (NLP mode) ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="nlp").build_model()
pipe = apply_v62_pipeline(model, mode="nlp", bake_tolerance=1.0e-3, params=params,
                          drop_dead_rows_threshold=0.0)

family = defaultdict(lambda: {"max": 0.0, "sum_sq": 0.0, "n": 0, "top": []})
for c in model.component_objects(Constraint, active=True):
    name = c.name
    for idx in c:
        if not c[idx].active:
            continue
        try:
            b = value(c[idx].body)
        except Exception:
            continue
        if b is None:
            continue
        upper = c[idx].upper
        lower = c[idx].lower
        u = value(upper) if upper is not None else None
        l = value(lower) if lower is not None else None
        if u is not None and l is not None and u == l:
            r_val = b - u
        elif u is not None and b > u:
            r_val = b - u
        elif l is not None and b < l:
            r_val = b - l
        else:
            r_val = 0.0
        a = abs(r_val)
        f = family[name]
        f["sum_sq"] += a * a
        f["n"] += 1
        if a > f["max"]:
            f["max"] = a
        f["top"].append((a, idx, r_val))

rows = sorted(family.items(), key=lambda kv: -kv[1]["max"])
print(f"\n{'Family':30s} {'max|resid|':>14s} {'L2 norm':>14s} {'n':>6s}", flush=True)
for name, f in rows[:20]:
    if f["max"] < 1e-10:
        continue
    l2 = f["sum_sq"] ** 0.5
    print(f"{name:30s} {f['max']:14.4e} {l2:14.4e} {f['n']:6d}", flush=True)

print(f"\nTop offenders per family:")
for name, f in rows[:5]:
    if f["max"] < 1e-10:
        continue
    print(f"\n{name}:")
    top = sorted(f["top"], key=lambda t: -t[0])[:5]
    for a, idx, r_val in top:
        print(f"  {idx}: residual = {r_val:+.4e}")
