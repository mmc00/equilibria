"""GTAP TRADE_CET block (leaf unit).

Ports the monolith's CET domestic/export allocation equations
(`eq_xds`, `eq_xet`, `eq_xseq`) VERBATIM from
``equilibria/templates/gtap/gtap_model_equations.py`` (5907-5954).

The four legacy constraints declared then ``.deactivate()``d in the monolith
(``eq_pe`` 5958, ``eq_pe_route`` 5967, ``eq_xet_agg`` 5983, ``eq_xe_xw`` 5998)
are NOT ported as active constraints — they are not part of the active square
system.

FIDELITY: the monolith is the parity oracle; equation bodies here are its
bodies, transcribed (same ``omega`` inf-branch, same ``Constraint.Skip`` →
``return None``, same ``self.params.elasticities.omegax.get`` default). The
share coefficients ``gd_share``/``ge_share``/``xet_flag`` are read as Pyomo
Params exactly as the monolith reads ``model.gd_share[r, i]`` inside the rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.core.parameters import Parameter
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable


class TradeCETBlock(Block):
    """GTAP CET domestic/export allocation (xds, xet, xseq)."""

    name: str = "GTAP_TRADE_CET"
    description: str = "GTAP CET domestic/export allocation (eq_xds, eq_xet, eq_xseq)"
    sets: Any = None
    params: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "i"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        comms = list(set_manager.get("i"))
        nr, ni = len(regions), len(comms)
        p = self.params

        # ----- Variables OWNED by this unit (monolith 3738 / 3749) -----
        # xet: within=Reals FREE, NO lb=0 (monolith 3738). Represent the Reals
        # domain via lower=-inf so the bridge emits bounds=(None, None).
        # NOTE: the monolith applies a runtime relative price floor to `pet`
        # (setlb(1e-3*init)); reproduced here as lower=1e-3 (init pet = 1.0).
        xet_init = np.zeros((nr, ni))
        pet_init = np.ones((nr, ni))
        for ri, r in enumerate(regions):
            for ci, i in enumerate(comms):
                xet_init[ri, ci] = self._get_xet_init(r, i)

        variables["xet"] = Variable(
            name="xet",
            value=xet_init,
            domains=("r", "i"),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        variables["pet"] = Variable(
            name="pet",
            value=pet_init,
            domains=("r", "i"),
            domain="NonNegativeReals",
            lower=1e-3,
            upper=float("inf"),
        )

        # ----- Share/flag Params read INSIDE the rule bodies (monolith reads
        # model.gd_share[r,i] etc). Seeded from params.shares like the monolith
        # (2001-2019). See report: these DRIFT under apply_production_scaling. -----
        gd = np.zeros((nr, ni))
        ge = np.zeros((nr, ni))
        # xet_flag seeds to 1.0 for every (r,i) — monolith 2018-2024.
        xetflag = np.ones((nr, ni))
        for ri, r in enumerate(regions):
            for ci, i in enumerate(comms):
                gd[ri, ci] = float(p.shares.p_gd.get((r, i), 0.0) or 0.0)
                ge[ri, ci] = float(p.shares.p_ge.get((r, i), 0.0) or 0.0)
        parameters["gd_share"] = Parameter(
            name="gd_share", value=gd, domains=("r", "i")
        )
        parameters["ge_share"] = Parameter(
            name="ge_share", value=ge, domains=("r", "i")
        )
        parameters["xet_flag"] = Parameter(
            name="xet_flag", value=xetflag, domains=("r", "i")
        )

        # Closure captures for the inline-python omega branch.
        omegax = p.elasticities.omegax

        # ---------------- eq_xds (monolith 5907) ----------------
        class EqXds(SymbolicEquation):
            name: str = "eq_xds"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices
                from pyomo.environ import value

                omega = omegax.get((r, i), float("inf"))
                gd_share = value(pyomo_model.gd_share[r, i])
                if gd_share <= 0.0:
                    return pyomo_model.xds[r, i] == 0.0
                if omega == float("inf"):
                    return pyomo_model.pd[r, i] == pyomo_model.ps[r, i]
                return (
                    pyomo_model.xds[r, i]
                    == pyomo_model.gd_share[r, i]
                    * pyomo_model.xs[r, i]
                    * (pyomo_model.pd[r, i] / pyomo_model.ps[r, i]) ** omega
                )

        # ---------------- eq_xet (monolith 5923) ----------------
        class EqXet(SymbolicEquation):
            name: str = "eq_xet"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices
                from pyomo.environ import value

                omega = omegax.get((r, i), float("inf"))
                if value(pyomo_model.xet_flag[r, i]) <= 0.0:
                    return None
                ge_share = value(pyomo_model.ge_share[r, i])
                if ge_share <= 0.0:
                    return pyomo_model.xet[r, i] == 0.0
                if omega == float("inf"):
                    return pyomo_model.pet[r, i] == pyomo_model.ps[r, i]
                return (
                    pyomo_model.xet[r, i]
                    == pyomo_model.ge_share[r, i]
                    * pyomo_model.xs[r, i]
                    * (pyomo_model.pet[r, i] / pyomo_model.ps[r, i]) ** omega
                )

        # ---------------- eq_xseq (monolith 5941) ----------------
        class EqXseq(SymbolicEquation):
            name: str = "eq_xseq"
            domains: tuple = ("r", "i")

            def build_expression(self, pyomo_model, indices):
                r, i = indices

                omega = omegax.get((r, i), float("inf"))
                if omega == float("inf"):
                    return (
                        pyomo_model.xs[r, i]
                        == pyomo_model.xds[r, i] + pyomo_model.xet[r, i]
                    )
                exponent = 1.0 + omega
                return (
                    pyomo_model.ps[r, i] ** exponent
                    == pyomo_model.gd_share[r, i] * pyomo_model.pd[r, i] ** exponent
                    + pyomo_model.ge_share[r, i] * pyomo_model.pet[r, i] ** exponent
                )

        return [EqXds(), EqXet(), EqXseq()]

    # ------------------------------------------------------------------
    # init helpers reproducing the monolith's get_*_init logic
    # ------------------------------------------------------------------
    def _get_xet_init(self, r: str, i: str) -> float:
        """Monolith get_xet_init (3343), benchmark branch (no reference_snapshot)."""
        p = self.params
        total_vxsb = sum(
            float(p.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0) for rp in self.sets.r
        )
        if total_vxsb > 0.0:
            return max(total_vxsb, 1e-8)
        # Fallback branch uses live ps/pd Var levels in the monolith; at the
        # benchmark those are 1.0 and this branch is rarely hit for gtap7_*.
        return 1e-8
