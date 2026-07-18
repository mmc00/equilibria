import sys; sys.path.insert(0,'src')
from pathlib import Path
from collections import Counter
from pyomo.environ import Var, Constraint
from pyomo.core.expr.visitor import identify_variables
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
root=Path("src/equilibria/templates/reference/pep2")
st=PEPModelCalibrator(sam_file=root/"data/SAM-V2_0.gdx",val_par_file=root/"data/VAL_PAR.xlsx").calibrate()
m=build_pep_model(st, variant="base", form="mcp")
# free vars
free=set(id(v) for v in m.component_data_objects(Var,active=True) if not v.fixed)
free_by={id(v):v for v in m.component_data_objects(Var,active=True) if not v.fixed}
# vars appearing in some constraint
in_con=set()
for c in m.component_data_objects(Constraint,active=True):
    for v in identify_variables(c.body):
        in_con.add(id(v))
# also constraints > vars means over-determined; but we want vars with NO constraint (orphans)
orphan=[free_by[i] for i in free if i not in in_con]
print("free vars with NO constraint (orphans):", len(orphan))
oc=Counter(v.parent_component().name for v in orphan)
print("  by name:", dict(oc))
# and the reverse: is any constraint referencing only fixed vars (redundant)?
