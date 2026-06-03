"""Snapshot 15x10 USA base values for regression testing."""
import contextlib, json, os, sys
from pathlib import Path
os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))
sys.path.insert(0, str(Path('scripts/gtap').resolve()))

from pyomo.environ import value as pyo_value
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import default_gtap_contract
from run_gtap import _run_path_capi_nonlinear_full

GDX = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()
contract = default_gtap_contract()
with contextlib.redirect_stdout(sys.stderr):
    p = GTAPParameters(); p.load_from_gdx(GDX)
    eq = GTAPModelEquations(p.sets, p, contract.closure)
    m = eq.build_model()
    r = _run_path_capi_nonlinear_full(m, p, closure_config=contract.closure,
                                       equation_scaling=True, path_capi_convergence_tol=1e-8)
out = {
    'residual': r['residual'],
    'usa': {
        'yc': float(pyo_value(m.yc['USA'])),
        'yg': float(pyo_value(m.yg['USA'])),
        'rsav': float(pyo_value(m.rsav['USA'])),
        'regy': float(pyo_value(m.regy['USA'])),
        'facty': float(pyo_value(m.facty['USA'])),
        'ytax_ind': float(pyo_value(m.ytax_ind['USA'])),
        'pft_UnSkLab': float(pyo_value(m.pft['USA','UnSkLab','base'])),
        'pft_Land': float(pyo_value(m.pft['USA','Land','base'])),
        'pft_Capital': float(pyo_value(m.pft['USA','Capital','base'])),
        'pabs': float(pyo_value(m.pabs['USA'])),
    },
}
print(json.dumps(out, indent=2))
