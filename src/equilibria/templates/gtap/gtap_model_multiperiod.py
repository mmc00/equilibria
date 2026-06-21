"""GTAP altertax MULTI-PERIODO builder. NO toca el single-period gtap_model_equations.
Arma un ConcreteModel con eje t={base,check,shock} presente (estructura del loop(tsim)
de GAMS) para que los enlaces Fisher inter-temporales sean filas del Jacobiano."""
from __future__ import annotations
from typing import Any, Optional
from pyomo.environ import ConcreteModel, Set
from .gtap_model_equations import GTAPModelEquations

PERIODS = ("base", "check", "shock")


def _astuple(k):
    """Normalize an index key to a tuple (handles scalar keys)."""
    return k if isinstance(k, tuple) else (k,)


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

    def build_vars(self, m: ConcreteModel) -> None:
        """Reflect every Var family from the single-period model, adding a t dimension.

        For each Var v in the single-period model:
          - Indexed Var:   new index = (*original_key, t)  for each t in PERIODS
          - Scalar Var:    new index = (t,)                for each t in PERIODS

        Init: all periods get the single-period benchmark value (base=check=shock).
        The within/domain is preserved from the single-period VarData.
        """
        from pyomo.environ import Var, NonNegativeReals, Reals

        # Build a temporary single-period model to read Var families from
        sp_model = GTAPModelEquations(
            self.sets, self.params, self.closure,
            residual_region=self.residual_region,
        ).build_model()

        periods = list(PERIODS)

        for v in sp_model.component_objects(Var, active=True):
            name = v.name

            # Determine domain from first VarData element
            first_key = next(iter(v)) if v.is_indexed() else None
            first_data = v[first_key] if v.is_indexed() else v[None]
            try:
                domain = first_data.domain
            except Exception:
                domain = NonNegativeReals

            if v.is_indexed():
                # Build explicit list of (orig_key..., t) tuples
                new_index = [
                    (*_astuple(k), t)
                    for k in v.index_set()
                    for t in periods
                ]

                # Capture values from single-period: key → float
                sp_vals = {}
                for k in v.index_set():
                    try:
                        sp_vals[_astuple(k)] = float(v[k].value)
                    except (TypeError, ValueError):
                        sp_vals[_astuple(k)] = 1.0

                def _mk_init(vals_dict):
                    def _init(_m, *key):
                        *orig, _t = key
                        return vals_dict.get(tuple(orig), 1.0)
                    return _init

                init_fn = _mk_init(sp_vals)
            else:
                # Scalar Var → indexed by t only
                new_index = [(t,) for t in periods]

                try:
                    sp_val = float(first_data.value)
                except (TypeError, ValueError):
                    sp_val = 0.0

                def _mk_scalar_init(val):
                    def _init(_m, t):
                        return val
                    return _init

                init_fn = _mk_scalar_init(sp_val)

            doc = v.doc if hasattr(v, "doc") and v.doc else ""
            setattr(m, name, Var(new_index, within=domain, initialize=init_fn, doc=doc))
