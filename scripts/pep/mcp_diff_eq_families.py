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
if len(sys.argv) < 2:
    raise SystemExit(
        "uso: mcp_diff_eq_families.py <gams_inst.json>\n"
        "  el JSON lleva {familia_de_ecuacion: cardinalidad} del modelo GAMS,\n"
        "  tal como lo emite un `gams ... --instance` o gdxdump sobre el .lst"
    )
gams={k.upper():v for k,v in json.load(open(sys.argv[1])).items()}
allk=set(mine)|set(gams)
print("family | GAMS | mine | diff")
for k in sorted(allk, key=lambda x:(x!='WALRAS', int(x[2:]) if x[2:].isdigit() else 0)):
    g=gams.get(k,0); mm=mine.get(k,0)
    if g!=mm: print(f"  {k}: GAMS={g} mine={mm} Δ={mm-g}")
print("TOTAL: GAMS",sum(gams.values()),"mine",sum(mine.values()))
