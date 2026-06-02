"""Phase 1 scaffolding tests — t_set, Sets, prev_t helper."""
import os, sys
from pathlib import Path
os.environ['EQUILIBRIA_GTAP_RRES'] = 'USA'
sys.path.insert(0, str(Path('src').resolve()))

from pyomo.environ import Set
from equilibria.templates.gtap.gtap_parameters import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations, prev_t
from equilibria.templates.gtap.gtap_contract import default_gtap_contract

GDX_15X10 = Path('datasets/gtap7_15x10/GDX/basedata.gdx').resolve()


def _build(t_set):
    contract = default_gtap_contract()
    p = GTAPParameters(); p.load_from_gdx(GDX_15X10)
    eq = GTAPModelEquations(p.sets, p, contract.closure, t_set=t_set)
    return eq.build_model()


def test_default_t_set_is_base_only():
    m = _build(("base",))
    assert list(m.t) == ["base"]
    assert list(m.t0) == ["base"]
    assert list(m.ts) == []


def test_three_period_t_set():
    m = _build(("base", "check", "shock"))
    assert list(m.t) == ["base", "check", "shock"]
    assert list(m.t0) == ["base"]
    assert list(m.ts) == ["check", "shock"]


def test_prev_t_helper():
    t_set = ("base", "check", "shock")
    assert prev_t("base", t_set) is None
    assert prev_t("check", t_set) == "base"
    assert prev_t("shock", t_set) == "check"


def test_t_set_validation():
    import pytest
    contract = default_gtap_contract()
    p = GTAPParameters(); p.load_from_gdx(GDX_15X10)
    with pytest.raises(ValueError, match="t_set"):
        GTAPModelEquations(p.sets, p, contract.closure, t_set=())
    with pytest.raises(ValueError, match="base"):
        GTAPModelEquations(p.sets, p, contract.closure, t_set=("shock", "check"))
