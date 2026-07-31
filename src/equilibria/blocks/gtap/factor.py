"""GTAP FACTOR block.

Ports the monolith's factor-market equations VERBATIM from
``gtap_model_equations.py`` (6634-6918):

  eq_xft (6638), eq_xfeq (6650), eq_xfteq (6670), eq_pfeq (6710),
  eq_pfteq (6788), eq_pfaeq (6832), eq_pfyeq (6841), eq_kstock (6896)

NOTE — deferred eq_pfact: the monolith DEFINES ``eq_pfact_rule`` here (6867) but
stores it (``self._defer_eq_pfact``) and REGISTERS the constraint later, in the
CLOSURE range (7831), because it needs pf0/xf0/mqfactr_bb which are snapshotted
after production scaling. To match the monolith's registration order the
eq_pfact CONSTRAINT lives in ``closure.py`` (unit 7), not here.

ifSUB (global switch): ``_m_pfa`` (5560) / ``_m_pfy`` (5567) return the PLAIN VAR
under ``if_sub=False`` (the comp-stat oracle these gates compare against). This
block ports the ``if_sub=False`` form (``_m_pfa`` -> pfa, ``_m_pfy`` -> pfy). Under
``if_sub=True`` the composer (Task 5) deactivates eq_pfaeq/eq_pfyeq and fixes
pfa/pfy to the macro value (monolith post-block 7970-8036). See
``blocks/gtap/__init__.py``.

FIDELITY: equation bodies are the monolith's, transcribed (same Skip guards, same
``value(model.<param>[...])`` inlining, same omegaf mf/sf/fnm branches, same
cross-multiplied CET form, same sqrt/single-root pfteq). Params are built via
``_derived_params`` verbatim transcriptions so every ``model.<param>[...]``
reference and inlined float matches the oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import value

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3


def _price_floor(init: float) -> float:
    """Monolith two-pass price floor: max(1e-8, 1e-3*init) for init>0 (5295-5356)."""
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class FactorBlock(Block):
    """GTAP factor markets (supply, demand, CET nest, factor prices, capital)."""

    name: str = "GTAP_FACTOR"
    description: str = "GTAP factor supply/demand CET nest + factor prices + capital"
    sets: Any = None
    params: Any = None
    # ifSUB toggle: under if_sub the factor report eqs (eq_pfaeq/eq_pfyeq) are
    # deactivated by the composer and the M_PFA/M_PFY wedges are substituted INLINE
    # in the consuming eqs (eq_xfeq/eq_pfeq), so pfa/pfy stay coupled to pf.
    if_sub: bool = False

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "f", "a"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        facs = list(set_manager.get("f"))
        acts = list(set_manager.get("a"))
        aa = list(set_manager.get("aa"))
        p = self.params
        s = self.sets
        _byname = {"r": regions, "f": facs, "a": acts, "aa": aa}

        # ------------------------------------------------------------------
        # Parameters (monolith _add_parameters recipes)
        # ------------------------------------------------------------------
        def _param(name, data, doms, default=0.0, mutable=False):
            parameters[name] = Parameter(
                name=name,
                value=dp.to_array(data, [_byname[d] for d in doms], default),
                domains=tuple(doms),
                mutable=mutable,
            )

        _param("xfflag", dp.xfflag_data(p, s), ("r", "f", "a"))
        _param("xftflag", dp.xftflag_data(p, s), ("r", "f"))
        # gf_share: base build (t0_snapshot is None) -> normalize from SAM.
        _param("gf_share", dp.gf_share_data(p, s), ("r", "f", "a"))
        _param("af_param", dp.af_param_data(p, s), ("r", "f", "a"))
        # aft/etaf: closure_name default gtap_standard -> etaf=0 (gate oracle).
        _param("aft", dp.aft_data(p, s), ("r", "f"))
        _param("etaf", dp.etaf_data(p, s), ("r", "f"))
        # fcttx/fctts: mutable in the monolith (5077-5094) and referenced UNWRAPPED
        # in eq_pfaeq (6835) -> they must stay symbolic (mutable) so the expression
        # prints fctts[...]/fcttx[...] not the folded literal. See bridge mutable=.
        _param("fcttx", dp.fcttx_data(p, s), ("r", "f", "a"), mutable=True)
        _param("fctts", dp.fctts_data(p, s), ("r", "f", "a"), mutable=True)
        # xscale shared with PRODUCTION_SUPPLY (dedup by name); declared here so a
        # standalone factor build resolves model.xscale[r,a] in eq_xft/eq_pfeq.
        _param("xscale", dp.xscale_data(p, s), ("r", "aa"), default=1.0)

        # ------------------------------------------------------------------
        # Variables OWNED by this unit (monolith 4349-4395, 4798-4826).
        # ------------------------------------------------------------------
        def _q(name, doms, init, lower=0.0):
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=lower,
                upper=float("inf"),
            )

        def _price(name, doms, init):
            lo = np.vectorize(_price_floor)(init)
            variables[name] = Variable(
                name=name,
                value=init,
                domains=tuple(doms),
                domain="NonNegativeReals",
                lower=lo,
                upper=float("inf"),
            )

        nr, nf, na = len(regions), len(facs), len(acts)
        # xft: bounds=(1e-8,None) declared (monolith 4349-4354), NOT in price list
        # (the runtime floor sweep skips it) -> stays (1e-8, None).
        xft_init = np.array(
            [[max(self._xft_init(r, f), 0.0) for f in facs] for r in regions]
        )
        pft_init = np.array([[self._pft_init(r, f) for f in facs] for r in regions])
        xf_init = np.array(
            [
                [[max(self._vfm_init(r, f, a), 0.0) for a in acts] for f in facs]
                for r in regions
            ]
        )
        pf_init = np.array(
            [[[self._pf_init(r, f, a) for a in acts] for f in facs] for r in regions]
        )
        pfa_init = np.array(
            [[[self._pfa_init(r, f, a) for a in acts] for f in facs] for r in regions]
        )
        pfy_init = np.array(
            [[[self._pfy_init(r, f, a) for a in acts] for f in facs] for r in regions]
        )
        # xft: monolith 4349 declares Var(bounds=(1e-8,None)) with NO within= ->
        # default domain is Reals (the (1e-8,None) bound is the only constraint).
        variables["xft"] = Variable(
            name="xft",
            value=xft_init,
            domains=("r", "f"),
            domain="Reals",
            lower=1e-8,
            upper=float("inf"),
        )
        _price("pft", ("r", "f"), pft_init)
        _q("xf", ("r", "f", "a"), xf_init)
        _price("pf", ("r", "f", "a"), pf_init)
        _price("pfa", ("r", "f", "a"), pfa_init)
        _price("pfy", ("r", "f", "a"), pfy_init)
        # pwfact scalar (monolith 4395): price var, init 1.0 -> floor 1e-3. Shared
        # with CLOSURE (eq_pwfact) — dedup by name, first (this) registration wins.
        variables["pwfact"] = Variable(
            name="pwfact",
            value=np.array([1.0]),
            domains=(),
            domain="NonNegativeReals",
            lower=1e-3,
            upper=float("inf"),
        )
        # Capital block (monolith 4798-4826). pfact = price var (floor 1e-3);
        # kstock/kapEnd = level vars with the 1e-8 floor; arent = NonNeg no floor
        # (not in any floor list); rorc/rore = within=Reals FREE.
        variables["pfact"] = Variable(
            name="pfact",
            value=np.ones(nr),
            domains=("r",),
            domain="NonNegativeReals",
            lower=1e-3,
            upper=float("inf"),
        )
        kstock_init = np.array([max(self._kstock_init(r), 0.0) for r in regions])
        kapend_init = np.array([max(self._kapend_init(r), 0.0) for r in regions])
        variables["kstock"] = Variable(
            name="kstock",
            value=kstock_init,
            domains=("r",),
            domain="NonNegativeReals",
            lower=1e-8,
            upper=float("inf"),
        )
        variables["kapEnd"] = Variable(
            name="kapEnd",
            value=kapend_init,
            domains=("r",),
            domain="NonNegativeReals",
            lower=np.vectorize(_price_floor)(kapend_init),
            upper=float("inf"),
        )
        variables["arent"] = Variable(
            name="arent",
            value=np.full(nr, 0.05),
            domains=("r",),
            domain="NonNegativeReals",
            lower=0.0,
            upper=float("inf"),
        )
        variables["rorc"] = Variable(
            name="rorc",
            value=np.full(nr, 0.05),
            domains=("r",),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        variables["rore"] = Variable(
            name="rore",
            value=np.full(nr, 0.05),
            domains=("r",),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )

        # ------------------------------------------------------------------
        # Inline-python accessors (read exactly as the monolith rules do).
        # ------------------------------------------------------------------
        el = p.elasticities
        taxes = p.taxes
        mf = set(s.mf)
        sf = set(s.sf)
        bm = p.benchmark

        def _get_sigmav(r, a):
            return el.sigmav.get((r, a), 1.0)

        def _lambdaf(r, f, a):
            return p.shifts.lambdaf.get((r, f, a), 1.0)

        def _omegaf(region, factor):
            omega = el.omegaf.get((region, factor))
            if omega is not None:
                return float(omega)
            if factor in mf:
                return float("inf")
            etrae = el.etrae.get(factor, float("inf"))
            if etrae == float("inf"):
                return float("inf")
            return -float(etrae)

        def _etaff(region, factor, activity):
            return float(el.etaff.get((region, factor, activity), 0.0))

        vkb = bm.vkb
        if_sub = self.if_sub

        def _kappaf(r, f, a):
            kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
            if kappa == 0.0:
                kappa = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
            return kappa

        def _m_pfa(model, r, f, a):
            """GAMS $macro M_PFA (8022): pf·(1+fctts+fcttx) inline under ifSUB (so
            pfa stays coupled to pf when eq_pfaeq is deactivated), else the plain
            report var pfa. Mirrors this block's own eq_pfaeq body."""
            if if_sub:
                return model.pf[r, f, a] * (
                    1.0 + model.fctts[r, f, a] + model.fcttx[r, f, a]
                )
            return model.pfa[r, f, a]

        def _m_pfy(model, r, f, a):
            """GAMS $macro M_PFY (8023): pf·(1-kappaf) inline under ifSUB, else pfy."""
            if if_sub:
                return model.pf[r, f, a] * (1.0 - _kappaf(r, f, a))
            return model.pfy[r, f, a]

        equations: list[SymbolicEquation] = []

        # ---------------- eq_xft (monolith 6638) ----------------
        class EqXft(SymbolicEquation):
            name: str = "eq_xft"
            domains: tuple = ("r", "f")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f = indices
                if value(model.xftflag[r, f]) <= 0.0:
                    return None
                return model.xft[r, f] == sum(
                    model.xf[r, f, a] / model.xscale[r, a] for a in model.a
                )

        equations.append(EqXft())

        # ---------------- eq_xfeq (monolith 6650) ----------------
        class EqXfeq(SymbolicEquation):
            name: str = "eq_xfeq"
            domains: tuple = ("r", "f", "a")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f, a = indices
                af_val = (
                    value(model.af_param[r, f, a])
                    if hasattr(model, "af_param")
                    else value(model.af_share[r, f, a])
                )
                if af_val <= 0.0:
                    return None
                if value(model.xfflag[r, f, a]) <= 0.0:
                    return None
                # M_PFA: plain pfa under if_sub=False; pf·(1+fctts+fcttx) inline
                # under if_sub=True (pfa coupled to pf when eq_pfaeq is deactivated).
                ratio = model.pva[r, a] / _m_pfa(model, r, f, a)
                sigmav = _get_sigmav(r, a)
                lambdaf = _lambdaf(r, f, a)
                return model.xf[r, f, a] == af_val * model.va[
                    r, a
                ] * ratio**sigmav * lambdaf ** (sigmav - 1)

        equations.append(EqXfeq())

        # ---------------- eq_xfteq (monolith 6670) ----------------
        class EqXfteq(SymbolicEquation):
            name: str = "eq_xfteq"
            domains: tuple = ("r", "f")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f = indices
                if value(model.xftflag[r, f]) <= 0.0:
                    return None
                benchmark_supply = float(value(model.aft[r, f]))
                if benchmark_supply <= 0:
                    return None
                elasticity = float(value(model.etaf[r, f]))
                return (
                    model.xft[r, f]
                    == model.aft[r, f]
                    * (model.pft[r, f] / (model.pabs[r] + 1e-12)) ** elasticity
                )

        equations.append(EqXfteq())

        # ---------------- eq_pfeq (monolith 6710) ----------------
        class EqPfeq(SymbolicEquation):
            name: str = "eq_pfeq"
            domains: tuple = ("r", "f", "a")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f, a = indices
                if value(model.xfflag[r, f, a]) <= 0.0:
                    return None
                gf = float(value(model.gf_share[r, f, a]))
                if gf <= 0.0:
                    return None
                # M_PFY: plain pfy under if_sub=False; pf·(1-kappaf) inline under
                # if_sub=True (pfy coupled to pf when eq_pfyeq is deactivated).
                pfy = _m_pfy(model, r, f, a)
                if f in mf:
                    omegaf = _omegaf(r, f)
                    if omegaf == float("inf"):
                        return pfy == model.pft[r, f]
                    kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
                    if kappa == 0.0:
                        kappa = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
                    denom = (
                        model.xscale[r, a] * model.gf_share[r, f, a] * model.xft[r, f]
                    )
                    pf_term = (model.pf[r, f, a] * (1.0 - kappa)) ** omegaf
                    pft_term = model.pft[r, f] ** omegaf
                    return pf_term * denom == pft_term * model.xf[r, f, a]
                if f in sf:
                    omegaf = _omegaf(r, f)
                    if omegaf == float("inf"):
                        return pfy == model.pft[r, f]
                    kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
                    if kappa == 0.0:
                        kappa = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
                    denom = (
                        model.xscale[r, a] * model.gf_share[r, f, a] * model.xft[r, f]
                    )
                    pf_term = (model.pf[r, f, a] * (1.0 - kappa)) ** omegaf
                    pft_term = model.pft[r, f] ** omegaf
                    return pf_term * denom == pft_term * model.xf[r, f, a]
                etaff = _etaff(r, f, a)
                return model.xf[r, f, a] == (
                    model.xscale[r, a]
                    * model.gf_share[r, f, a]
                    * (pfy / model.pabs[r]) ** etaff
                )

        equations.append(EqPfeq())

        # ---------------- eq_pfteq (monolith 6788) ----------------
        class EqPfteq(SymbolicEquation):
            name: str = "eq_pfteq"
            domains: tuple = ("r", "f")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f = indices
                if value(model.xftflag[r, f]) <= 0.0:
                    return None
                omegaf_val = _omegaf(r, f)
                if omegaf_val == float("inf"):
                    xs_sum = sum(
                        model.xf[r, f, a] / model.xscale[r, a]
                        for a in model.a
                        if value(model.xfflag[r, f, a]) > 0.0
                        and value(model.xscale[r, a]) > 1e-12
                    )
                    return model.xft[r, f] == xs_sum
                expo = 1.0 + omegaf_val
                if abs(expo) < 1e-10:
                    return None
                # M_PFY inline under if_sub=True (pfy coupled to pf; eq_pfyeq off).
                pfy_sum = sum(
                    model.gf_share[r, f, a] * _m_pfy(model, r, f, a) ** expo
                    for a in model.a
                    if value(model.xfflag[r, f, a]) > 0.0
                    and float(value(model.gf_share[r, f, a])) > 0.0
                )
                return model.pft[r, f] == pfy_sum ** (1.0 / expo)

        equations.append(EqPfteq())

        # ---------------- eq_pfaeq (monolith 6832) ----------------
        class EqPfaeq(SymbolicEquation):
            name: str = "eq_pfaeq"
            domains: tuple = ("r", "f", "a")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f, a = indices
                if value(model.xfflag[r, f, a]) <= 0.0:
                    return None
                wedge = model.fctts[r, f, a] + model.fcttx[r, f, a]
                return model.pfa[r, f, a] == model.pf[r, f, a] * (1.0 + wedge)

        equations.append(EqPfaeq())

        # ---------------- eq_pfyeq (monolith 6841) ----------------
        class EqPfyeq(SymbolicEquation):
            name: str = "eq_pfyeq"
            domains: tuple = ("r", "f", "a")

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                r, f, a = indices
                if value(model.xfflag[r, f, a]) <= 0.0:
                    return None
                kappa = float(taxes.kappaf_activity.get((r, f, a), 0.0))
                if kappa == 0.0:
                    kappa = float(taxes.kappaf.get((r, f), 0.0))
                return model.pfy[r, f, a] == model.pf[r, f, a] * (1.0 - kappa)

        equations.append(EqPfyeq())

        # ---------------- eq_kstock (monolith 6896) ----------------
        class EqKstock(SymbolicEquation):
            name: str = "eq_kstock"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                capital_factors = [
                    f
                    for f in model.f
                    if str(f).lower() in ("capital", "cap", "k", "kap")
                ]
                if not capital_factors:
                    return None
                xft_bench_total = sum(
                    float(value(model.aft[r, f])) for f in capital_factors
                )
                vkb_val = vkb.get(r)
                if vkb_val is None:
                    vkb_val = vkb.get((r,), 0.0)
                kstock_bench = float(vkb_val or 0.0)
                if kstock_bench <= 0.0 or xft_bench_total <= 0.0:
                    return None
                krat = xft_bench_total / kstock_bench
                return krat * model.kstock[r] == sum(
                    model.xft[r, f] for f in capital_factors
                )

        equations.append(EqKstock())

        return equations

    # ------------------------------------------------------------------
    # F3.5 base calibration — adopt the settled (check) point as base
    # ------------------------------------------------------------------
    def calibrate_base(self, params, sets, closure, residual_region, ref_gdx=None):
        """Base calibration (F3.5): return the SETTLED check-period point.

        Runs the settle solve ONCE (the base->check->shock stack the gates use,
        reused as a calibration tool) and returns the CHECK-period point — the
        settled equilibrium where the specific factors have re-settled. The
        composer uses it as the base seed so the shock runs base->shock (no
        check) and the specific-factor response matches GEMPACK. The GAMS-faithful
        default never calls this.

        Returns ``{var_name: {body: value}}`` for the check slice, where ``body``
        is the index tuple without the trailing period (a bare scalar for 1-D).
        """
        from pyomo.environ import Var
        from pyomo.environ import value as _V

        from equilibria.templates.gtap.gtap_block_model import (
            build_block_model,
            solve_block_model,
        )

        m, mp = build_block_model(params, sets, closure, residual_region)
        if ref_gdx is not None:
            mp.seed_all_periods(m, ref_gdx)
        solve_block_model(m, params, closure, ref_gdx, mode="gtap")

        settled: dict[str, dict] = {}
        for v in m.component_objects(Var, active=True):
            for idx in v:
                if not (isinstance(idx, tuple) and idx and idx[-1] == "check"):
                    continue
                try:
                    val = float(_V(v[idx]))
                except (TypeError, ValueError):
                    continue
                body = idx[:-1]
                body = body[0] if len(body) == 1 else body
                settled.setdefault(v.name, {})[body] = val
        return settled

    # ------------------------------------------------------------------
    # init helpers (monolith get_*_init logic — the price-floor-bearing ones)
    # ------------------------------------------------------------------
    def _pf_init(self, r: str, f: str, a: str) -> float:
        """get_pf_init: 1/(1-kappa) (benchmark branch) — monolith 2858-2860."""
        kappa = self.params.taxes.kappaf_activity.get((r, f, a), 0.0)
        return max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)

    def _pfa_init(self, r: str, f: str, a: str) -> float:
        """get_pfa_init: pf*(1+rtf) — monolith 2862-2867."""
        pf_val = self._pf_init(r, f, a)
        fcttx = float(self.params.taxes.rtf.get((r, f, a), 0.0) or 0.0)
        return max(pf_val * (1.0 + fcttx), 1e-8)

    def _pfy_init(self, r: str, f: str, a: str) -> float:
        """get_pfy_init: pf*(1-kappa) — monolith 2869-2872."""
        pf_val = self._pf_init(r, f, a)
        kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
        return max(pf_val * max(1.0 - kappa, 1e-8), 1e-8)

    def _vfm_init(self, r: str, f: str, a: str) -> float:
        """get_vfm_init: evfb/pf (benchmark branch) — monolith 2838-2849."""
        bm = self.params.benchmark
        vfm_val = max(
            float(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)) or 0.0), 0.0
        )
        pf_val = max(self._pf_init(r, f, a), 1e-12)
        return max(vfm_val / pf_val, 0.0)

    def _xft_init(self, r: str, f: str) -> float:
        """get_factor_supply_init: sum_a pfy*xf — monolith 3386-3393."""
        return max(
            sum(self._pfy_init(r, f, a) * self._vfm_init(r, f, a) for a in self.sets.a),
            0.0,
        )

    def _pft_init(self, r: str, f: str) -> float:
        """get_pft_init: 1.0 if supply>0 else 1e-8 — monolith 3395-3399."""
        supply = self._xft_init(r, f)
        return 1e-8 if supply <= 0.0 else 1.0

    def _kstock_init(self, r: str) -> float:
        """get_kstock_init: vkb (benchmark branch) — monolith 3401+."""
        raw_vkb = self.params.benchmark.vkb
        vkb_val = raw_vkb.get(r)
        if vkb_val is None:
            vkb_val = raw_vkb.get((r,), 0.0)
        return float(vkb_val or 0.0)

    def _kapend_init(self, r: str) -> float:
        """get_kapend_init: (1-depr)*vkb + xi_bench — monolith 3415-3420."""
        vkb = self._kstock_init(r)
        vdep = float(self.params.benchmark.vdep.get(r, 0.0))
        depr = (vdep / vkb) if vkb > 0.0 else 0.0
        xi_bench = sum(self._vim_init(r, i) for i in self.sets.i)
        return max((1.0 - depr) * vkb + xi_bench, 1e-8)

    def _vim_init(self, r: str, i: str) -> float:
        """get_vim_init: total investment Armington demand — for xi_bench."""
        val, _, _ = self.params.benchmark.get_investment_demand(r, i)
        return max(float(val or 0.0), 1e-8)
