"""GTAP CLOSURE block (LAST unit — market clearing / numeraire / Walras).

Ports the monolith's closure equations VERBATIM from
``gtap_model_equations.py`` (7742-8036):

  eq_pfact (7867, the DEFERRED rule from FACTOR — registered HERE at monolith
            7831, matching registration order), eq_pwfact (7834), eq_pnum (7944),
  eq_walras (7954). eq_pmuv (7917) is conditional on rmuv&imuv (see below).

DEFERRED eq_pfact / eq_pwfact (COMPOSER CARRY, __init__.py checklist item 3):
both reference the SNAPSHOT Params pf0/xf0/mqfactr_bb/mqfactw_bb (monolith
7762-7827) computed from the model's pf/xf/xscale Var levels AFTER
apply_production_scaling. On the base build pf0=pf.l≈1.04 (matches FACTOR's
_pf_init exactly) but xf0=xf.l is the SCALED xf (benchmark / xscale) — the block
cannot reproduce it from self.params (it needs the scaled model). So the block
declares pf0/xf0/mqfactr_bb/mqfactw_bb as Params seeded from the BENCHMARK, and
builds eq_pfact/eq_pwfact against them; the composer OVERWRITES those Params from
the scaled model post-registration. On the gate the xf0/mqfactr_bb coefficient
therefore differs from the oracle by the xscale factor — a STRUCTURAL composer
carry (the skeleton is byte-identical; only the composer-owned folded numeric
differs). See _FORM_CARRY_EQS[(ClosureBlock, eq_pfact/eq_pwfact)] in the test.

pmuv Var-or-Param switch (monolith 5096-5118): only when ``closure.rmuv`` &
``closure.imuv`` are both non-empty is ``pmuv`` a Var (bnd 0.001,None) and
``eq_pmuv`` registered. Otherwise (the gate oracle: rmuv/imuv empty) ``pmuv`` is
a mutable Param frozen at 1.0 and eq_pmuv is NOT created. The block mirrors the
gate-oracle form (pmuv = Param, no eq_pmuv); the composer flips it under the
switch (__init__.py checklist item 4).

pwfact is shared with FACTOR (unit 3) — dedup by name (FACTOR's Var wins).
The ifSUB post-block (7970-8036) is the COMPOSER's (__init__.py checklist item 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import value

from equilibria.blocks.gtap._backend_math import sqrt

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

_FLOOR = 1e-8
_REL = 1e-3


def _price_floor(init: float) -> float:
    """Monolith relative floor: max(1e-8, 1e-3*init) for init>0 (5298-5379)."""
    if init is None or init <= 0.0:
        return _FLOOR
    return max(_FLOOR, _REL * float(init))


class ClosureBlock(Block):
    """GTAP closure: factor-price Fisher index, numeraire, Walras check."""

    name: str = "GTAP_CLOSURE"
    description: str = "GTAP factor-price Fisher index (pfact/pwfact), pnum, walras"
    sets: Any = None
    params: Any = None
    residual_region: str = "NAmerica"

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        facs = list(set_manager.get("f"))
        acts = list(set_manager.get("a"))
        aa = list(set_manager.get("aa"))
        p = self.params
        s = self.sets
        _byname = {"r": regions, "f": facs, "a": acts, "aa": aa}
        nr = len(regions)

        # -------- snapshot Params (COMPOSER CARRY, seeded from benchmark) --------
        # pf0 = _pf_init (matches oracle post-scaling pf.l exactly); xf0 = _vfm_init
        # (the UN-scaled benchmark xf — the composer overwrites with the SCALED xf.l).
        # mqfactr_bb / mqfactw_bb = sum pf0*xf0/xscale (calibration constants,
        # composer re-computes from the scaled snapshot). All non-mutable (fold to a
        # literal in eq_pfact/eq_pwfact), matching the oracle (7781-7827 mutable=False).
        xscale = dp.xscale_data(p, s)

        pf0: dict[tuple, float] = {}
        xf0: dict[tuple, float] = {}
        for r in regions:
            for f in facs:
                for a in acts:
                    pf0[(r, f, a)] = self._pf_init(r, f, a)
                    xf0[(r, f, a)] = max(self._vfm_init(r, f, a), 0.0)

        mqfactr_bb_data: dict[tuple, float] = {}
        mqfactw = 0.0
        for r in regions:
            s_reg = 0.0
            for f in facs:
                for a in acts:
                    xs = xscale.get((r, a), 1.0)
                    if xs <= 1e-12:
                        continue
                    term = pf0[(r, f, a)] * xf0[(r, f, a)] / xs
                    s_reg += term
                    mqfactw += term
            mqfactr_bb_data[(r,)] = s_reg if s_reg > 0.0 else 1.0
        mqfactw_bb = mqfactw if mqfactw > 0.0 else 1.0

        parameters["pf0"] = Parameter(
            name="pf0",
            value=dp.to_array(pf0, [regions, facs, acts], 0.0),
            domains=("r", "f", "a"),
        )
        parameters["xf0"] = Parameter(
            name="xf0",
            value=dp.to_array(xf0, [regions, facs, acts], 0.0),
            domains=("r", "f", "a"),
        )
        parameters["mqfactr_bb"] = Parameter(
            name="mqfactr_bb",
            value=dp.to_array(mqfactr_bb_data, [regions], 1.0),
            domains=("r",),
        )
        parameters["mqfactw_bb"] = Parameter(
            name="mqfactw_bb", value=np.array([mqfactw_bb]), domains=()
        )
        # xscale shared with PRODUCTION/FACTOR (dedup); needed by eq_pfact/eq_pwfact.
        parameters["xscale"] = Parameter(
            name="xscale",
            value=dp.to_array(xscale, [regions, aa], 1.0),
            domains=("r", "aa"),
        )

        # -------- Variables OWNED by this unit (monolith 4785, 4969, 5106-5118) --
        # pnum: price floor 1e-3 (init 1.0). walras: within=Reals FREE.
        variables["pnum"] = Variable(
            name="pnum",
            value=np.array([1.0]),
            domains=(),
            domain="NonNegativeReals",
            lower=1e-3,
            upper=float("inf"),
        )
        variables["walras"] = Variable(
            name="walras",
            value=np.array([0.0]),
            domains=(),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        # pwfact scalar shared with FACTOR (dedup by name — FACTOR's registration
        # wins in the composed model). Declared here so a standalone closure build
        # resolves model.pwfact; price floor 1e-3 (init 1.0), monolith 4395.
        variables["pwfact"] = Variable(
            name="pwfact",
            value=np.array([1.0]),
            domains=(),
            domain="NonNegativeReals",
            lower=1e-3,
            upper=float("inf"),
        )
        # Named aggregates for the world Fisher index (eq_pwfact).
        #
        # eq_pwfact sums over every (r, f, a) — 701 variables in one row at 10x7,
        # 1,501 at 15x10 — and wraps the result in a division under a square root.
        # Symbolic differentiation of that single row is superlinear in its width:
        # measured in isolation, 161 variables take 66s and 701 never finish, while
        # the model's other 12,502 rows declare in 0.15s combined.
        #
        # Naming each sum makes the wide part three separate rows, which are linear
        # or quadratic and never reach the symbolic differentiator, leaving a
        # four-variable nonlinear row. The arithmetic is unchanged.
        # Seeded at the benchmark value of the sum they stand for, not at 1.0: the
        # solve is warm-started from the benchmark, and an auxiliary left at 1.0
        # would start its row ~3,000 away from feasible and move the solver off the
        # seeded point. mqfactw is that same sum over the benchmark (computed above),
        # which is what all three aggregates equal at the benchmark.
        _agg_seed = float(mqfactw) if mqfactw > 0.0 else 1.0
        for _agg in ("mfw_bs", "mfw_sb", "mfw_ss"):
            variables[_agg] = Variable(
                name=_agg,
                value=np.array([_agg_seed]),
                domains=(),
                domain="Reals",
                lower=-float("inf"),
                upper=float("inf"),
            )

        # The same three aggregates per region, for eq_pfact — the regional twin of
        # eq_pwfact, with the same square root over a ratio of wide sums.
        _mqfactr_seed = np.array(
            [float(mqfactr_bb_data.get((rr,), 1.0)) for rr in regions], dtype=float
        )
        for _agg in ("mfr_bs", "mfr_sb", "mfr_ss"):
            variables[_agg] = Variable(
                name=_agg,
                value=_mqfactr_seed.copy(),
                domains=("r",),
                domain="Reals",
                lower=-float("inf"),
                upper=float("inf"),
            )
        # pmuv: rmuv/imuv empty on the gate oracle -> a mutable Param frozen at 1.0
        # (monolith 5113-5118); eq_pmuv is NOT registered. The composer flips it to a
        # Var (bnd 0.001,None) + registers eq_pmuv under the rmuv&imuv switch.
        parameters["pmuv"] = Parameter(
            name="pmuv", value=np.array([1.0]), domains=(), mutable=True
        )

        equations: list[SymbolicEquation] = []

        # The regional aggregates, one row each — wide but linear or quadratic, so
        # they never reach the symbolic differentiator (same split as eq_pwfact).
        class EqMfrBs(SymbolicEquation):
            name: str = "eq_mfr_bs"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.mfr_bs[r] == sum(
                    model.pf0[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12
                )

        equations.append(EqMfrBs())

        class EqMfrSb(SymbolicEquation):
            name: str = "eq_mfr_sb"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.mfr_sb[r] == sum(
                    model.pf[r, f, a] * model.xf0[r, f, a] / model.xscale[r, a]
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12 and model.xf0[r, f, a] > 0.0
                )

        equations.append(EqMfrSb())

        class EqMfrSs(SymbolicEquation):
            name: str = "eq_mfr_ss"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.mfr_ss[r] == sum(
                    model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12
                )

        equations.append(EqMfrSs())

        # ---------------- eq_pfact (monolith 7867; registered here per 7831) -----
        # The Fisher index itself, now over four variables per region.
        class EqPfact(SymbolicEquation):
            name: str = "eq_pfact"
            domains: tuple = ("r",)

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                (r,) = indices
                return model.pfact[r] == sqrt(
                    (model.mfr_sb[r] / model.mqfactr_bb[r])
                    * (model.mfr_ss[r] / (model.mfr_bs[r] + 1e-12))
                )

        equations.append(EqPfact())

        # ---------------- eq_pwfact (monolith 7834) ----------------
        # The three aggregates, each its own row. Wide but linear or quadratic, so
        # they never reach the symbolic differentiator.
        class EqMfwBs(SymbolicEquation):
            name: str = "eq_mfw_bs"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                return model.mfw_bs == sum(
                    model.pf0[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                    for r in model.r
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12
                )

        equations.append(EqMfwBs())

        class EqMfwSb(SymbolicEquation):
            name: str = "eq_mfw_sb"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                return model.mfw_sb == sum(
                    model.pf[r, f, a] * model.xf0[r, f, a] / model.xscale[r, a]
                    for r in model.r
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12 and model.xf0[r, f, a] > 0.0
                )

        equations.append(EqMfwSb())

        class EqMfwSs(SymbolicEquation):
            name: str = "eq_mfw_ss"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                return model.mfw_ss == sum(
                    model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                    for r in model.r
                    for f in model.f
                    for a in model.a
                    if value(model.xscale[r, a]) > 1e-12
                )

        equations.append(EqMfwSs())

        # ---------------- eq_pwfact (monolith 7834) ----------------
        # The Fisher index itself, now over four variables instead of 701.
        class EqPwfact(SymbolicEquation):
            name: str = "eq_pwfact"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                return model.pwfact == sqrt(
                    (model.mfw_sb / model.mqfactw_bb)
                    * (model.mfw_ss / (model.mfw_bs + 1e-12))
                )

        equations.append(EqPwfact())

        # ---------------- eq_pnum (monolith 7944) ----------------
        class EqPnum(SymbolicEquation):
            name: str = "eq_pnum"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                return model.pnum == model.pwfact

        equations.append(EqPnum())

        # ---------------- eq_walras (monolith 7954) ----------------
        residual_region = self.residual_region
        residual_regions = tuple(r for r in regions if str(r) == residual_region)

        class EqWalras(SymbolicEquation):
            name: str = "eq_walras"
            domains: tuple = ()

            def build_expression(self, pyomo_model, indices):
                model = pyomo_model
                target_regions = (
                    residual_regions if residual_regions else tuple(model.r)
                )
                return model.walras == sum(
                    model.yi[r]
                    - (
                        model.pi[r] * model.depr[r] * model.kstock[r]
                        + model.rsav[r]
                        + model.savf[r]
                    )
                    for r in target_regions
                )

        equations.append(EqWalras())

        _ = nr  # (region count; kept for parity with sibling blocks)
        return equations

    # ------------------------------------------------------------------
    # init helpers (pf0/xf0 seed — reproduce FACTOR's get_pf_init/get_vfm_init)
    # ------------------------------------------------------------------
    def _pf_init(self, r: str, f: str, a: str) -> float:
        """get_pf_init: 1/(1-kappa) (benchmark branch) — monolith 2858-2860."""
        kappa = self.params.taxes.kappaf_activity.get((r, f, a), 0.0)
        return max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)

    def _vfm_init(self, r: str, f: str, a: str) -> float:
        """get_vfm_init: evfb/pf (benchmark branch) — monolith 2838-2849."""
        bm = self.params.benchmark
        vfm_val = max(
            float(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)) or 0.0), 0.0
        )
        pf_val = max(self._pf_init(r, f, a), 1e-12)
        return max(vfm_val / pf_val, 0.0)
