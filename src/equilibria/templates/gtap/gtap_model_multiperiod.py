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
        from pyomo.environ import Var, NonNegativeReals

        # Build a temporary single-period model to read Var families from.
        # Reuse the builder stored in __init__ (same immutable sets/params);
        # build_model() runs apply_production_scaling + _align_xi_xaa_post_scaling
        # so the reflected init values are the benchmark-consistent post-scaling
        # values the solver actually warm-starts from.
        sp_model = self._sp.build_model()

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
                    sp_val = 1.0

                def _mk_scalar_init(val):
                    def _init(_m, t):
                        return val
                    return _init

                init_fn = _mk_scalar_init(sp_val)

            doc = v.doc if hasattr(v, "doc") and v.doc else ""
            setattr(m, name, Var(new_index, within=domain, initialize=init_fn, doc=doc))

    def build_equations_intra(self, m: ConcreteModel, period: str) -> None:
        """Replicate all single-period Constraint families onto m, indexed by (orig_key..., period).

        Strategy: build a fresh single-period model, then for each Constraint family,
        substitute every sp_var[k] reference in the body with m_var[(*k, period)]
        using ExpressionReplacementVisitor's substitute dict (keyed by id).

        The resulting constraints on m are named identically to the single-period ones
        (e.g. eq_facty) but indexed by (original_key..., period), so
        m.eq_facty["USA", "base"] holds the same algebraic body as sp.eq_facty["USA"]
        evaluated at the multi-period vars for t=base.
        """
        from pyomo.environ import Constraint, Var
        from pyomo.core.expr.visitor import ExpressionReplacementVisitor

        # Build a fresh single-period model to get its Constraints and Vars.
        sp = GTAPModelEquations(
            self.sets, self.params, self.closure,
            residual_region=self.residual_region,
        ).build_model()

        # Build substitute dict: id(sp_var[k]) -> m_var[(*k, period)]
        # for every VarData in the single-period model.
        substitute: dict = {}
        for v in sp.component_objects(Var, active=True):
            vname = v.name
            mp_var = getattr(m, vname)
            if v.is_indexed():
                for k in v.index_set():
                    sp_vd = v[k]
                    kt = (*_astuple(k), period)
                    substitute[id(sp_vd)] = mp_var[kt]
            else:
                # scalar Var — sp uses v[None], mp is indexed by (period,)
                sp_vd = v[None]
                substitute[id(sp_vd)] = mp_var[(period,)]

        visitor = ExpressionReplacementVisitor(substitute=substitute)

        for con in sp.component_objects(Constraint, active=True):
            cname = con.name
            if con.is_indexed():
                # Build explicit index list: (orig_key..., period) for each key
                new_index = [(*_astuple(k), period) for k in con]

                # Capture (body, lower, upper) per original key to avoid closure issues
                con_data: dict = {}
                for k in con:
                    cd = con[k]
                    new_body = visitor.walk_expression(cd.body)
                    con_data[k] = (new_body, cd.lower, cd.upper)

                def _make_rule(data_dict):
                    def _rule(_m, *key):
                        orig_key = key[:-1]  # strip trailing period
                        orig_key = orig_key[0] if len(orig_key) == 1 else orig_key
                        body, lb, ub = data_dict[orig_key]
                        if lb is not None and ub is not None and lb == ub:
                            # equality constraint: body == value
                            return body == lb
                        if lb is not None and ub is not None:
                            # ranged constraint — return (lb, body, ub) tuple
                            return (lb, body, ub)
                        if lb is not None:
                            return body >= lb
                        if ub is not None:
                            return body <= ub
                        return Constraint.Skip
                    return _rule

                setattr(m, cname, Constraint(new_index, rule=_make_rule(con_data)))
            else:
                # Scalar constraint — key is None in sp; mp gets key (period,)
                cd = con[None]
                new_body = visitor.walk_expression(cd.body)
                lb, ub = cd.lower, cd.upper

                def _make_scalar_rule(body, lb, ub):
                    def _rule(_m, t):
                        if lb is not None and ub is not None and lb == ub:
                            return body == lb
                        if lb is not None and ub is not None:
                            return (lb, body, ub)
                        if lb is not None:
                            return body >= lb
                        if ub is not None:
                            return body <= ub
                        return Constraint.Skip
                    return _rule

                setattr(
                    m, cname,
                    Constraint([(period,)], rule=_make_scalar_rule(new_body, lb, ub)),
                )
