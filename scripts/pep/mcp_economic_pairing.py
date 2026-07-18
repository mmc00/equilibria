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

# ECONOMIC pairing: constraint cn (family fam) ⊥ the var it determines. Family→var-name map
# (from the GAMS-MCP derivation). For each constraint, pick the paired var at the SAME index.
PAIR={ 'eq1':'PP','eq2':'PCI','eq3':'VA','eq4':'KDC','eq5':'LDC','eq6':'LD','eq7':'KD','eq8':'RTI','eq9':'DI',
 'eq10':'YH','eq11':'YHL','eq12':'YHK','eq13':'YHTR','eq14':'YDH','eq15':'CTH','eq16':'SH',
 'eq17':'YF','eq18':'YFK','eq19':'YFTR','eq20':'YDF','eq21':'SF','eq22':'YG','eq23':'YGK',
 'eq24':'TDHT','eq25':'TDFT','eq26':'TPRODN','eq27':'TIWT','eq28':'TIKT','eq29':'TIPT','eq30':'TPRCTS',
 'eq31':'TICT','eq32':'TIMT','eq33':'TIXT','eq34':'YGTR','eq35':'TDH','eq36':'TDF','eq37':'TIW','eq38':'TIK',
 'eq39':'TIP','eq40':'TIC','eq41':'TIM','eq42':'TIX','eq43':'SG','eq44':'YROW','eq45':'SROW','eq46':'CAB',
 'eq47':'TR','eq48':'YHTR','eq49':'TR','eq50':'TR','eq51':'TR','eq52':'C','eq53':'GFCF','eq54':'INV',
 'eq55':'CG','eq56':'DIT','eq57':'MRGN','eq58':'XST','eq59':'XS','eq60':'DS','eq61':'EX','eq62':'EXD',
 'eq63':'Q','eq64':'IM','eq65':'PVA','eq66':'PT','eq67':'CI','eq68':'RC','eq70':'WTI','eq72':'R','eq73':'RK',
 'eq74':'P','eq75':'PL','eq76':'PE','eq77':'PD','eq78':'PM','eq79':'PC','eq80':'PIXGDP','eq81':'PIXCON',
 'eq82':'PIXINV','eq83':'PIXGVT','eq84':'DD','eq85':'W','eq86':'RK','eq87':'IT','eq88':'WC','eq89':'RC',
 'eq90':'GDP_BP','eq91':'GDP_MP','eq92':'GDP_IB','eq93':'GDP_FD','eq94':'CTH_REAL','eq95':'G_REAL',
 'eq96':'GDP_BP_REAL','eq97':'GDP_MP_REAL','eq98':'GFCF_REAL','walras':'LEON' }

cons=list(m.component_data_objects(Constraint,active=True))
paired=[]; used=set(); unmatched=0
for c in cons:
    fam=c.parent_component().name
    varname=PAIR.get(fam)
    idx=c.index()
    v=None
    if varname:
        comp=getattr(m,varname,None)
        if comp is not None:
            # try the constraint's own index, else scan the var's free cells for an unused one
            try:
                cand=comp[idx] if idx is not None else comp
                if not cand.fixed and id(cand) not in used: v=cand
            except Exception: pass
            if v is None:
                for k in comp:
                    cc=comp[k]
                    if not cc.fixed and id(cc) not in used: v=cc; break
    if v is None:
        # fallback: any free var in the constraint not yet used
        repn=generate_standard_repn(c.body,quadratic=False)
        for vv in list(repn.linear_vars or ())+list(getattr(repn,'nonlinear_vars',None) or ()):
            if not vv.fixed and id(vv) not in used: v=vv; break
    if v is None: unmatched+=1; continue
    used.add(id(v)); paired.append(v)
print(f"paired {len(paired)} constraints, unmatched {unmatched}, cons {len(cons)}")
seed=_max_residual(m)
opt=SolverFactory("path_capi_bridge")
res=opt.solve(m, load_solutions=True, variables=paired)
print(f"PATH tc={res.solver.termination_condition} seed_resid={seed:.2e} solved_resid={_max_residual(m):.2e} GDP_BP={value(m.GDP_BP,exception=False)}")
