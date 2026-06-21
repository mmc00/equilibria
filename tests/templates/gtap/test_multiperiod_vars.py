# tests/templates/gtap/test_multiperiod_vars.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from test_multiperiod_sets import _load_3x3_params  # reuse loader


def test_vars_have_t_dim():
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel
    from pyomo.environ import Var
    p = _load_3x3_params()
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=list(p.sets.r)[-1])
    m = mp.build_sets(); mp.build_vars(m)
    # pva existe y tiene t como última dimensión
    assert hasattr(m, "pva")
    idx = next(iter(m.pva))
    assert idx[-1] in ("base", "check", "shock")
    # cuenta: pva single-period tiene |r|*|a| celdas → multi tiene *3
    n_sp = len(p.sets.r) * len(p.sets.a)
    assert len(m.pva) == n_sp * 3
