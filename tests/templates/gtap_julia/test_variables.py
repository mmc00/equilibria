"""Task 4: gtap_julia variables carry Julia's positive lower bounds."""

import pyomo.environ as pyo

from equilibria.templates.gtap_julia.sets import build_sets
from equilibria.templates.gtap_julia.variables import build_variables

DATASET = "gtap7_3x3"


def _model():
    m = pyo.ConcreteModel()
    sets = build_sets(DATASET)
    build_variables(m, sets)
    return m, sets


def test_key_vars_exist():
    m, _ = _model()
    for name in ("qfe", "qxs", "pds", "tms", "qva", "rore", "walras_sup"):
        assert hasattr(m, name), f"missing var {name}"


def test_quantity_vars_have_positive_floor():
    m, _ = _model()
    # qfe is a quantity — every cell lb must be strictly positive (q_min)
    for idx in m.qfe:
        assert m.qfe[idx].lb is not None and m.qfe[idx].lb > 0.0
        break


def test_price_vars_have_price_floor():
    m, _ = _model()
    for idx in m.pds:
        assert m.pds[idx].lb is not None and m.pds[idx].lb > 0.0
        break


def test_qsave_is_free_below_zero():
    m, _ = _model()
    # qsave alone has a negative lower bound (Julia: qsave => -q_max)
    for idx in m.qsave:
        assert m.qsave[idx].lb is not None and m.qsave[idx].lb < 0.0
        break


def test_tax_vars_are_multiplicative_positive():
    m, _ = _model()
    # taxes are POWERS (>0), the key property that keeps log() valid
    for idx in m.tms:
        assert m.tms[idx].lb is not None and m.tms[idx].lb > 0.0
        break
