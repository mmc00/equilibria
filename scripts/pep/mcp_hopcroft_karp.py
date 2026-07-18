import sys, logging; logging.disable(logging.WARNING)
sys.path.insert(0,'src'); sys.path.insert(0,'/Users/marmol/proyectos/path-capi-python/src')
from pathlib import Path
from pyomo.environ import Var, Constraint, SolverFactory, value
from pyomo.repn import generate_standard_repn
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
from equilibria.templates.pep_pyomo.pep_pyomo_solver import _ensure_path_lib, _max_residual
_ensure_path_lib(); import path_capi_python
root=Path('src/equilibria/templates/reference/pep2')
st=PEPModelCalibrator(sam_file=root/'data/SAM-V2_0.gdx',val_par_file=root/'data/VAL_PAR.xlsx').calibrate()
m=build_pep_model(st,variant='base',form='mcp')
cons=list(m.component_data_objects(Constraint,active=True))
# adjacency: constraint i → free vars appearing in it
adj=[]
vid={}   # var id -> stable index
def vidx(v):
    if id(v) not in vid: vid[id(v)]=len(vid)
    return vid[id(v)]
varobj={}
for c in cons:
    repn=generate_standard_repn(c.body,quadratic=False)
    vs=[v for v in list(repn.linear_vars or ())+list(getattr(repn,'nonlinear_vars',None) or ()) if not v.fixed]
    row=[]
    for v in vs:
        j=vidx(v); varobj[j]=v; row.append(j)
    adj.append(row)
nV=len(vid)
# Hopcroft-Karp / Kuhn augmenting-path matching (constraints → vars)
matchV=[-1]*nV   # var j -> constraint i
def aug(i,seen):
    for j in adj[i]:
        if not seen[j]:
            seen[j]=True
            if matchV[j]==-1 or aug(matchV[j],seen):
                matchV[j]=i; return True
    return False
matchC=[-1]*len(cons)
for i in range(len(cons)):
    seen=[False]*nV
    if aug(i,seen): pass
for j,i in enumerate(matchV):
    if i>=0: matchC[i]=j
nmatched=sum(1 for x in matchC if x>=0)
print(f"HK matched {nmatched}/{len(cons)} constraints;  free vars {nV}")
if nmatched==len(cons)==nV:
    paired=[varobj[matchC[i]] for i in range(len(cons))]
    seed=_max_residual(m)
    opt=SolverFactory("path_capi_bridge")
    res=opt.solve(m, load_solutions=True, variables=paired)
    print(f"PATH tc={res.solver.termination_condition} seed={seed:.2e} solved={_max_residual(m):.2e} GDP_BP={value(m.GDP_BP,exception=False)}")
else:
    un=[i for i in range(len(cons)) if matchC[i]<0]
    print("unmatched constraints:", [cons[i].name for i in un][:12])
