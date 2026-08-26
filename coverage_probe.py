"""Coverage probe: translate EVERY constraint family in a real GTAP model to JAX and report
which families translate + match Pyomo, and which hit an unsupported node (so the spec knows
the exact vocabulary gap). Not a pytest — a diagnostic script."""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "gtap"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from equilibria.templates.gtap_jax.expr_to_jax import translate
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.environ import value

DS = None
for cand in ("gtap7_3x3","gtap6_10x7","gtap7_10x7"):
    if (Path("datasets")/cand).exists(): DS=cand; break
print("dataset:", DS)
d = Path("datasets")/DS
p = GTAPParameters()
p.load_from_har(basedata_path=d/"basedata.har",sets_path=d/"sets.har",default_path=d/"default.prm",baserate_path=d/"baserate.har")
rr = list(p.sets.r)[-1]
ac = GTAPClosureConfig(name="base",closure_type="MCP",capital_mobility="sluggish",fix_endowments=False,fix_taxes=False,fix_technology=False,if_sub=False,savf_flag="capFix",numeraire="pnum")
m,mp,_ = build_sparse_model_mp(p,p.sets,ac,rr,base_calibrated=False,ref_gdx=None)

vars_list = [v for v in m.component_data_objects(Var, active=True)]
var_index = {id(v): i for i,v in enumerate(vars_list)}
z = jnp.asarray([value(v) if v.value is not None else 1.0 for v in vars_list], dtype=float)

# group constraints by family (name prefix before '[')
fam_ok = defaultdict(int); fam_fail = defaultdict(int); fam_mismatch = defaultdict(int)
fail_reasons = defaultdict(set)
per_family_one = {}
for c in m.component_data_objects(Constraint, active=True):
    fam = str(c.name).split("[")[0]
    per_family_one.setdefault(fam, c)

for fam, c in per_family_one.items():
    try:
        f = translate(c.body, var_index)
        jr = float(f(z)); pr = value(c.body)
        if abs(pr-jr) < 1e-7 or abs(pr-jr) < 1e-7*max(1,abs(pr)):
            fam_ok[fam]+=1
        else:
            fam_mismatch[fam]+=1; fail_reasons[fam].add(f"mismatch pyo={pr:.3e} jax={jr:.3e}")
    except Exception as e:
        fam_fail[fam]+=1; fail_reasons[fam].add(f"{type(e).__name__}: {str(e)[:60]}")

fams = sorted(per_family_one)
print(f"\n=== {len(fams)} families probed (one representative cell each) ===")
ok = [f for f in fams if f in fam_ok]
mis = [f for f in fams if f in fam_mismatch]
bad = [f for f in fams if f in fam_fail]
print(f"OK (translate+match): {len(ok)}")
print(f"MISMATCH (translate, wrong value): {len(mis)}")
print(f"FAILED (unsupported node): {len(bad)}")
print("\n--- FAILED families + reason (the vocabulary gap for the spec) ---")
for f in bad: print(f"  {f}: {list(fail_reasons[f])[:1]}")
print("\n--- MISMATCH families ---")
for f in mis: print(f"  {f}: {list(fail_reasons[f])[:1]}")
print("\n--- OK families ---")
print("  " + ", ".join(ok))
