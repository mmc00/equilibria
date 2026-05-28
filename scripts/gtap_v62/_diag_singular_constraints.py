"""Map PATH's singular constraint indices f[XXXX] to Pyomo constraint names.

PATH's hook (Phase 3.36 output_hook) reported 13 structurally singular
constraints during gtap6_15x10 crash factorization:

  Group 1 (~23070-23160): f[23070], f[23081], f[23092], f[23114],
                          f[23125], f[23136], f[23158]
  Group 2 (~24170-24260): f[24170], f[24181], f[24192], f[24214],
                          f[24225], f[24236], f[24247], f[24258]

This script reproduces the exact enumeration PATH sees (active_cons
in component_objects order) and prints constraint name + index for
each of the singular f[i].
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Path/license env not needed; we only build the model and enumerate
from pyomo.environ import Var, Constraint
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline  # type: ignore

DATA = Path("datasets/gtap6_15x10")

SINGULAR_INDICES = [
    # 1-based as PATH reports them; we'll subtract 1 for Python indexing
    23070, 23081, 23092, 23114, 23125, 23136, 23158,
    24170, 24181, 24192, 24214, 24225, 24236, 24247, 24258,
]

print("\n=== Mapping PATH singular f[i] -> Pyomo constraint name ===", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)

# Enumerate active constraints in declaration order — PATH's f index
active_cons = []
for c in model.component_objects(Constraint, active=True):
    for idx in c:
        if c[idx].active:
            active_cons.append((c, idx))

print(f"\nTotal active constraints: {len(active_cons)}", flush=True)

# Build a histogram of where each constraint family starts/ends
family_ranges = {}
for k, (c, idx) in enumerate(active_cons, start=1):  # PATH uses 1-based
    name = c.name
    family_ranges.setdefault(name, [k, k])
    family_ranges[name][1] = k

print(f"\nConstraint family ranges (1-based, inclusive):", flush=True)
for name, (lo, hi) in sorted(family_ranges.items(), key=lambda x: x[1][0]):
    n = hi - lo + 1
    print(f"  {name:30s}  f[{lo:6d}..{hi:6d}]  ({n} cells)")

print(f"\nSingular constraints reported by PATH:", flush=True)
for idx_1based in SINGULAR_INDICES:
    py_idx = idx_1based - 1  # convert to 0-based
    if 0 <= py_idx < len(active_cons):
        c, idx = active_cons[py_idx]
        print(f"  f[{idx_1based:6d}] = {c.name}{idx}", flush=True)
    else:
        print(f"  f[{idx_1based:6d}] = OUT OF RANGE (max={len(active_cons)})", flush=True)
