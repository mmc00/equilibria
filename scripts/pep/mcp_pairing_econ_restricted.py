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
# economic family→var(s) the eq may pair with (its determined var + close alternatives)
PAIR={'eq1':['VA'],'eq2':['CI'],'eq3':['VA','LDC','KDC'],'eq4':['LDC','KDC'],'eq5':['KDC','LDC'],'eq6':['LD'],
 'eq7':['KDC'],'eq8':['KD'],'eq9':['DI'],'eq10':['YH'],'eq11':['YHL'],'eq12':['YHK'],'eq13':['YHTR'],'eq14':['YDH'],
 'eq15':['CTH'],'eq16':['SH'],'eq17':['YF'],'eq18':['YFK'],'eq19':['YFTR'],'eq20':['YDF'],'eq21':['SF'],'eq22':['YG'],
 'eq23':['YGK'],'eq24':['TDHT'],'eq25':['TDFT'],'eq26':['TPRODN'],'eq27':['TIWT'],'eq28':['TIKT'],'eq29':['TIPT'],
 'eq30':['TPRCTS'],'eq31':['TICT'],'eq32':['TIMT'],'eq33':['TIXT'],'eq34':['YGTR'],'eq35':['TDH'],'eq36':['TDF'],
 'eq37':['TIW'],'eq38':['TIK'],'eq39':['TIP'],'eq40':['TIC'],'eq41':['TIM'],'eq42':['TIX'],'eq43':['SG'],'eq44':['YROW'],
 'eq45':['SROW'],'eq46':['CAB','SROW'],'eq47':['TR'],'eq48':['TR'],'eq49':['TR'],'eq50':['TR'],'eq51':['TR'],'eq52':['C'],
 'eq53':['GFCF'],'eq54':['INV'],'eq55':['CG'],'eq56':['DIT'],'eq57':['MRGN'],'eq58':['XST'],'eq59':['XS'],'eq60':['DS','XS'],
 'eq61':['EX'],'eq62':['EXD'],'eq63':['Q'],'eq64':['IM'],'eq65':['PP'],'eq66':['PT'],'eq67':['PCI'],'eq68':['PVA'],
 'eq70':['WTI'],'eq72':['RTI'],'eq73':['R'],'eq74':['P'],'eq75':['P','PL'],'eq76':['PE'],'eq77':['PD'],'eq78':['PM'],
 'eq79':['PC'],'eq80':['PIXGDP'],'eq81':['PIXCON'],'eq82':['PIXINV'],'eq83':['PIXGVT'],'eq84':['DD','Q'],'eq85':['W','LS'],
 'eq86':['RK','KS'],'eq87':['IT'],'eq88':['WC','DD'],'eq89':['RC','EXD'],'eq90':['GDP_BP'],'eq91':['GDP_MP'],'eq92':['GDP_IB'],
 'eq93':['GDP_FD'],'eq94':['CTH_REAL'],'eq95':['G_REAL'],'eq96':['GDP_BP_REAL'],'eq97':['GDP_MP_REAL'],'eq98':['GFCF_REAL'],
 'walras':['LEON']}
cons=list(m.component_data_objects(Constraint,active=True))
vid={}; varobj={}
def vidx(v):
    if id(v) not in vid: vid[id(v)]=len(vid); varobj[vid[id(v)]]=v
    return vid[id(v)]
adj=[]
for c in cons:
    fam=c.parent_component().name
    allowed=set(PAIR.get(fam,[]))
    repn=generate_standard_repn(c.body,quadratic=False)
    vs=[v for v in list(repn.linear_vars or ())+list(getattr(repn,'nonlinear_vars',None) or ())
        if not v.fixed and v.parent_component().name in allowed]
    if not vs:  # fallback: any free var in body
        vs=[v for v in list(repn.linear_vars or ())+list(getattr(repn,'nonlinear_vars',None) or ()) if not v.fixed]
    adj.append([vidx(v) for v in vs])
nV=len(vid); matchV=[-1]*nV
def aug(i,seen):
    for j in adj[i]:
        if not seen[j]:
            seen[j]=True
            if matchV[j]==-1 or aug(matchV[j],seen):
                matchV[j]=i; return True
    return False
for i in range(len(cons)):
    aug(i,[False]*nV)
matchC=[-1]*len(cons)
for j,i in enumerate(matchV):
    if i>=0: matchC[i]=j
nm=sum(1 for x in matchC if x>=0)
print(f"econ-restricted HK matched {nm}/{len(cons)}")
if nm==len(cons):
    paired=[varobj[matchC[i]] for i in range(len(cons))]
    seed=_max_residual(m); opt=SolverFactory("path_capi_bridge")
    res=opt.solve(m, load_solutions=True, variables=paired)
    print(f"PATH tc={res.solver.termination_condition} seed={seed:.2e} solved={_max_residual(m):.2e} GDP_BP={value(m.GDP_BP,exception=False)}")
else:
    print("unmatched:", [cons[i].name for i in range(len(cons)) if matchC[i]<0][:12])
