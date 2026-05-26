"""Diagnose where 15x10 baseline residual lives by equation family.

CORRECTED: residual for a constraint is body - upper (or lower - body),
NOT body. For `expr == 1.0`, body = expr; residual = expr - 1.0.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"

from collections import defaultdict
from pyomo.environ import Constraint, value
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_15x10")

print(f"\n=== gtap6_15x10 baseline residual breakdown (correct math) ===", flush=True)
sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har", default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-3, params=params,
                          drop_dead_rows_threshold=1.0e-6)
print(f"  free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

family = defaultdict(lambda: {"max": 0.0, "sum": 0.0, "n": 0, "top": []})
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
        # Compute residual: violation from the equality / inequality bound.
        upper = c[idx].upper
        lower = c[idx].lower
        u = value(upper) if upper is not None else None
        l = value(lower) if lower is not None else None
        if u is not None and l is not None and u == l:
            # equality
            r_val = b - u
        elif u is not None and b > u:
            r_val = b - u
        elif l is not None and b < l:
            r_val = b - l
        else:
            r_val = 0.0
        a = abs(r_val)
        f = family[name]
        f["sum"] += a
        f["n"] += 1
        if a > f["max"]:
            f["max"] = a
        f["top"].append((a, idx, r_val))

rows = sorted(family.items(), key=lambda kv: -kv[1]["max"])
print(f"\n{'Family':30s} {'max|resid|':>14s} {'sum|resid|':>14s} {'n':>6s}", flush=True)
for name, f in rows[:25]:
    if f["max"] < 1e-12 and f["n"] > 0:
        continue  # skip clean families
    print(f"{name:30s} {f['max']:14.4e} {f['sum']:14.4e} {f['n']:6d}", flush=True)

print("\nTop 3 offenders per top-7 family:")
for name, f in rows[:7]:
    if f["max"] < 1e-12:
        continue
    print(f"\n{name}:")
    top = sorted(f["top"], key=lambda t: -t[0])[:3]
    for a, idx, r_val in top:
        print(f"  {idx}: residual = {r_val:+.4e}")
