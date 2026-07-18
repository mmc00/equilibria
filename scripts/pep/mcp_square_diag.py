import sys; sys.path.insert(0,'src')
from pathlib import Path
from collections import Counter
from pyomo.environ import Var, Constraint
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
root=Path("src/equilibria/templates/reference/pep2")
st=PEPModelCalibrator(sam_file=root/"data/SAM-V2_0.gdx",val_par_file=root/"data/VAL_PAR.xlsx").calibrate()
m=build_pep_model(st, variant="base", form="mcp")
# free vars per parent component
freev=Counter(); 
for v in m.component_data_objects(Var,active=True):
    if not v.fixed: freev[v.parent_component().name]+=1
cons=Counter()
for c in m.component_data_objects(Constraint,active=True):
    cons[c.parent_component().name]+=1
tv=sum(freev.values()); tc=sum(cons.values())
print(f"free vars={tv} constraints={tc} diff={tv-tc}")
print("\n=== free-var counts per variable (top 40) ===")
for nm,n in freev.most_common(40):
    print(f"  {nm}: {n}")
print("\n=== constraint counts per equation family ===")
print(" ", dict(cons.most_common(20)))
