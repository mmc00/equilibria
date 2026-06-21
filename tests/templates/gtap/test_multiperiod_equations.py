# tests/templates/gtap/test_multiperiod_equations.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from test_multiperiod_sets import _load_3x3_params
from pyomo.environ import value as pv


def test_intra_eq_matches_single_period_at_base():
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel
    from equilibria.templates.gtap import GTAPModelEquations
    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
    sp = GTAPModelEquations(p.sets, p, None, residual_region=rr).build_model()
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    mp.build_equations_intra(m, "base")
    # eq_facty[USA] in multi (t=base) == eq_facty[USA] in single-period
    sp_body = pv(sp.eq_facty["USA"].body)
    mp_body = pv(m.eq_facty["USA", "base"].body)
    assert abs(sp_body - mp_body) < 1e-9
