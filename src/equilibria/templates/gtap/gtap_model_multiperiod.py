"""GTAP altertax MULTI-PERIODO builder. NO toca el single-period gtap_model_equations.
Arma un ConcreteModel con eje t={base,check,shock} presente (estructura del loop(tsim)
de GAMS) para que los enlaces Fisher inter-temporales sean filas del Jacobiano."""
from __future__ import annotations
from typing import Any, Optional
from pyomo.environ import ConcreteModel, Set
from .gtap_model_equations import GTAPModelEquations

PERIODS = ("base", "check", "shock")

class GTAPMultiPeriodModel:
    def __init__(self, sets, params, closure=None, residual_region: Optional[str] = None):
        self.sets = sets
        self.params = params
        self.closure = closure
        self.residual_region = residual_region
        # builder single-period reusable para sets/vars/eqs base
        self._sp = GTAPModelEquations(sets, params, closure, residual_region=residual_region)

    def build_sets(self) -> ConcreteModel:
        m = ConcreteModel()
        self._sp._add_sets(m)              # r,a,i,f,... actuales
        m.t = Set(initialize=list(PERIODS), ordered=True)
        m.t0 = Set(initialize=["base"], ordered=True)
        return m
