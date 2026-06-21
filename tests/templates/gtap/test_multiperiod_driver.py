# tests/templates/gtap/test_multiperiod_driver.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from test_multiperiod_sets import _load_3x3_params


def test_driver_runs_three_periods():
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)
    res = solve_multiperiod(m, p, None)
    assert set(res) == {"base", "check", "shock"}
    assert all("code" in res[t] for t in res)
