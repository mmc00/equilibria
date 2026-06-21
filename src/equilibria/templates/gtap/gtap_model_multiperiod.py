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

    def build_equations_fisher(self, m: ConcreteModel) -> None:
        """Inter-temporal Fisher GDP index as Jacobian rows.

        Declares eq_rgdpmp[r,t] and eq_pgdpmp[r,t] with cross-period prices×quantities.
        These replace the reflected (intra-period) versions created by build_equations_intra,
        which used snapshot constants rather than live Vars from another period.

        Fisher cross-product:
          mqgdp(tp,tq,r) = Σ_{i,fd} pa[r,i,fd,tp]·xaa[r,i,fd,tq]
                         + Σ_{i,rp}( pefob[r,i,rp,tp]·xw[r,i,rp,tq]
                                    − pmcif[rp,i,r,tp]·xw[rp,i,r,tq] )
        fd = {hhd, gov, inv, tmg}  — MUST include tmg (verified: reproduces GAMS exactly)

        eq_rgdpmp[r,base]:  rgdpmp[r,base] == gdpmp[r,base]          (anchor)
        eq_rgdpmp[r,t≠base]: rgdpmp[r,t] == rgdpmp[r,base]
                              · sqrt( (gdpmp[r,t]/gdpmp[r,base]) · (mqgdp(base,t,r)/mqgdp(t,base,r)) )
          with smooth positive guard on the sqrt argument.

        eq_pgdpmp[r,t]:  pgdpmp[r,t] · rgdpmp[r,t] == gdpmp[r,t]
        """
        from pyomo.environ import Constraint, sqrt as _pyo_sqrt
        from .gtap_model_equations import (
            GTAP_HOUSEHOLD_AGENT as H,
            GTAP_GOVERNMENT_AGENT as G,
            GTAP_INVESTMENT_AGENT as I,
            GTAP_MARGIN_AGENT as MG,
        )

        fd = (H, G, I, MG)

        def _mqgdp(tp: str, tq: str, r: str):
            """Fisher cross-product of absorption + net exports."""
            # Final-demand absorption: Σ_{i,fd} pa[r,i,fd,tp] · xaa[r,i,fd,tq]
            ab = sum(
                m.pa[r, i, a, tp] * m.xaa[r, i, a, tq]
                for i in m.i
                for a in fd
            )
            # Net export value: Σ_{i,rp} ( pefob[r,i,rp,tp]·xw[r,i,rp,tq]
            #                              − pmcif[rp,i,r,tp]·xw[rp,i,r,tq] )
            tr = sum(
                m.pefob[r, i, rp, tp] * m.xw[r, i, rp, tq]
                - m.pmcif[rp, i, r, tp] * m.xw[rp, i, r, tq]
                for i in m.i
                for rp in m.r
            )
            return ab + tr

        # Remove intra-period eq_rgdpmp / eq_pgdpmp (created by build_equations_intra)
        # so we don't have duplicate bindings on rgdpmp/pgdpmp.
        for cname in ("eq_rgdpmp", "eq_pgdpmp"):
            comp = getattr(m, cname, None)
            if comp is not None:
                m.del_component(comp)

        def _rgdpmp_rule(_m, r, t):
            if t == "base":
                # At benchmark, real GDP equals nominal GDP (pgdpmp=1 by construction).
                return m.rgdpmp[r, "base"] == m.gdpmp[r, "base"]
            # Fisher chain-volume index relative to base:
            #   rgdpmp[t] = rgdpmp[base] · √( (gdpmp[t]/gdpmp[base]) · (mqgdp(base,t)/mqgdp(t,base)) )
            # Smooth positive guard on the sqrt argument to keep PATH evaluable during
            # iterations where the trade balance might transiently go negative:
            #   arg_pos = (arg + √(arg²+ε))/2  →  ≈ arg for arg≫√ε, ≈ 0⁺ for arg≤0, C¹-smooth.
            _mq_base_t = _mqgdp("base", t, r)   # price=base, qty=current
            _mq_t_base = _mqgdp(t, "base", r)   # price=current, qty=base
            _arg = (m.gdpmp[r, t] / m.gdpmp[r, "base"]) * (_mq_base_t / (_mq_t_base + 1e-12))
            _arg_pos = (_arg + _pyo_sqrt(_arg * _arg + 1e-8)) * 0.5
            return m.rgdpmp[r, t] == m.rgdpmp[r, "base"] * _pyo_sqrt(_arg_pos + 1e-12)

        # Use the full (r, t) index set — all regions × all periods
        all_rt = [(r, t) for r in m.r for t in m.t]
        m.eq_rgdpmp = Constraint(all_rt, rule=_rgdpmp_rule)

        def _pgdpmp_rule(_m, r, t):
            return m.pgdpmp[r, t] * m.rgdpmp[r, t] == m.gdpmp[r, t]

        m.eq_pgdpmp = Constraint(all_rt, rule=_pgdpmp_rule)

    def seed_all_periods(self, m: ConcreteModel, gdx_path) -> None:
        """Seed var[...,t] from a GAMS altertax GDX for t ∈ {base, check, shock}.

        Handles the GAMS → Python mapping:
          - Symbol aliases: xa→xaa, pa→pa, xw→xw, pefob→pefob, pmcif→pmcif, etc.
          - Prefix stripping on index keys: c_Food→Food, a_Food→Food (but hhd/gov/inv/tmg untouched)
          - Multi-period GDX has trailing 't' dimension matching PERIODS

        Only unfixed VarData entries are seeded (fixed vars keep their value).
        """
        import subprocess
        import csv as _csv
        from pathlib import Path as _Path

        gdx_path = _Path(gdx_path)
        GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
        T_LABELS = {"base", "check", "shock"}

        # GAMS symbol → Python Var name on m
        _ALIAS = {
            "xa": "xaa",
            "pp": "pp_rai",
            "p": "p_rai",
            # Add others as needed; most names are identical
        }

        def _strip(k: str) -> str:
            """Strip GAMS prefix a_/c_/f_/r_ from set elements."""
            if isinstance(k, str) and len(k) > 2 and k[1] == "_" and k[0] in "acfr":
                return k[2:]
            return k

        def _dump_sym(sym: str) -> dict:
            res = subprocess.run(
                [GDXDUMP, str(gdx_path), "Format=csv", f"Symb={sym}"],
                capture_output=True, text=True, check=False,
            )
            if res.returncode != 0 or not res.stdout.strip():
                return {}
            out: dict = {}
            reader = _csv.reader(res.stdout.splitlines())
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                *keys, val = row
                keys = tuple(k.strip('"') for k in keys)
                try:
                    out[keys] = float(val)
                except ValueError:
                    pass
            return out

        def _list_var_symbols() -> list:
            res = subprocess.run(
                [GDXDUMP, str(gdx_path), "Symbols"],
                capture_output=True, text=True, check=False,
            )
            names = []
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) < 5 or parts[3] != "Var":
                    continue
                try:
                    if int(parts[4]) > 0:
                        names.append(parts[1])
                except ValueError:
                    pass
            return names

        for gams_name in _list_var_symbols():
            py_name = _ALIAS.get(gams_name, gams_name)
            # Try to get the Var from m (case-sensitive, then lowercase)
            py_var = getattr(m, py_name, None) or getattr(m, py_name.lower(), None)
            if py_var is None:
                continue
            # Only seed indexed Vars that have a time axis
            if not hasattr(py_var, "index_set"):
                continue

            data = _dump_sym(gams_name)
            if not data:
                continue

            for gk, gval in data.items():
                # Last key must be a period label
                if not (gk and gk[-1] in T_LABELS):
                    continue
                t = gk[-1]
                # Strip prefixes from non-period keys
                stripped = tuple(_strip(x) for x in gk[:-1]) + (t,)
                try:
                    vd = py_var[stripped]
                    if not vd.fixed:
                        vd.set_value(float(gval))
                except (KeyError, TypeError, ValueError):
                    pass
