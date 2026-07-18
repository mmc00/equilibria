import sys, json; sys.path.insert(0,'src')
from pathlib import Path
from collections import Counter
from pyomo.environ import Constraint
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
root=Path('src/equilibria/templates/reference/pep2')
st=PEPModelCalibrator(sam_file=root/'data/SAM-V2_0.gdx',val_par_file=root/'data/VAL_PAR.xlsx').calibrate()
m=build_pep_model(st,variant='base',form='mcp')
mine=Counter(cc.parent_component().name.upper() for cc in m.component_data_objects(Constraint,active=True))
gams={k.upper():v for k,v in json.load(open("/private/tmp/claude-501/-Users-marmol--superset-worktrees-b14cb643-ee65-449d-b3f0-be8003b60783-scratched-stag/93588e4b-b814-4740-a289-a468ed3a55bc/scratchpad/gams_inst.json")).items()}
allk=set(mine)|set(gams)
print("family | GAMS | mine | diff")
for k in sorted(allk, key=lambda x:(x!='WALRAS', int(x[2:]) if x[2:].isdigit() else 0)):
    g=gams.get(k,0); mm=mine.get(k,0)
    if g!=mm: print(f"  {k}: GAMS={g} mine={mm} Δ={mm-g}")
print("TOTAL: GAMS",sum(gams.values()),"mine",sum(mine.values()))
