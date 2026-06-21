# tests/templates/gtap/test_multiperiod_sets.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

def _load_3x3_params():
    from equilibria.templates.gtap import GTAPParameters
    d = ROOT / "datasets/gtap7_3x3"
    p = GTAPParameters()
    p.load_from_har(basedata_path=d/"basedata.har", sets_path=d/"sets.har",
                    default_path=d/"default.prm", baserate_path=d/"baserate.har")
    return p

def test_t_axis_sets():
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel
    p = _load_3x3_params()
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=list(p.sets.r)[-1])
    m = mp.build_sets()
    assert list(m.t) == ["base", "check", "shock"]
    assert list(m.t0) == ["base"]
    assert "USA" in list(m.r)  # reusa los sets actuales
