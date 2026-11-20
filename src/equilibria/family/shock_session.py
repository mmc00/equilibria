"""The shock period of a finished core run, solved or staged again at other shocks.

``solve_altertax(decomp_steps=...)`` integrates the welfare decomposition along the
shock path: the shock period at fractions of the shock, with the same base and check.
A :class:`ShockSession` holds the arrays as they stood after the check (the run's own
shock period started from them) and flies the recipe's shock period on a copy of
them, through the same steps and solver as the run (``core.driver.fly_period``). It
replaces the old family driver's ``FamilySession`` (PR 3d of gtap-on-core): same
three methods, same returns.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from equilibria.core.driver import fly_period
from equilibria.core.recipe import KindPlan, Recipe, StepContext
from equilibria.family.gtap_periods import GTAPFamilyPeriodModel
from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.period_newton import _period_system
from equilibria.family.run_setup import FamilyRun

SHOCK = "shock"


@dataclass
class ShockSession:
    """``template`` and ``recipe``: the run's (``GTAP``, ``gtap_static``); ``run`` its
    ``FamilyRun``; ``after_check`` the arrays after the check (a copy, never
    modified); ``ratio``: ``trade_balance="ratio"`` (the bordered solve)."""

    template: Any
    recipe: Recipe
    run: FamilyRun
    after_check: MPArrays
    ratio: bool
    # the closure the shock period was last staged with (set by shock_system)
    _staged: Any = field(default=None, init=False, repr=False)

    def _model(self, shock: Mapping[str, Any], start: MPArrays | None) -> Any:
        """A period model on a copy of ``start`` (else the check's arrays) with
        ``shock`` written in the shock period (levels from the check's, as the run's
        own shock)."""
        mp = (start if start is not None else self.after_check).copy()
        _write_levels(mp, shock)
        return GTAPFamilyPeriodModel(self.run, mp)

    def _recipe(self, warm: bool) -> Recipe:
        """The run's recipe; ``warm``: its shock period keeps the start arrays' own
        shock levels instead of seeding the free cells from the check (the Fisher
        aggregates refreshed as a seed refreshes them)."""
        if not warm:
            return self.recipe
        plan_kind = self._period().kind
        plan = self._plan()
        before = tuple(
            _refresh_aggregates if getattr(s, "seeds_from", None) else s
            for s in plan.before
        )
        plans = {
            **self.recipe.plans,
            plan_kind: dataclasses.replace(plan, before=before),
        }
        return Recipe(self.recipe.periods, plans)

    def solve_shock(
        self, shock: Any, *, shock_levels: bool = False, start: MPArrays | None = None
    ) -> tuple[MPArrays, dict[str, Any]]:
        """The shock period at ``shock``, as the run solved its own. ``start``: arrays
        of a previous shock solve of this session; the free cells start from its shock
        values (warm start) instead of the check's. Returns ``(arrays, stats)``,
        ``stats`` with ``code`` (1 when solved), ``residual``, ``nit`` and the solve
        records (``solves``)."""
        if shock_levels:
            raise NotImplementedError("shock_levels: the core writes % shocks only")
        model = self._model(shock, start)
        n0 = len(self.run.solves)
        res = fly_period(
            model,
            self._recipe(warm=start is not None),
            SHOCK,
            template=self.template,
            shock=shock,
        )
        solves = [dict(r) for r in self.run.solves[n0:]]
        code = _code(res["code"])
        stats = {
            "code": code,
            "residual": res["residual"],
            "nit": solves[-1].get("nit") if solves else None,
            "solves": solves,
        }
        return model.mp, stats

    def _period(self) -> Any:
        return next(p for p in self.recipe.periods if p.name == SHOCK)

    def _context(self, model: Any, shock: Mapping[str, Any]) -> StepContext:
        period = self._period()
        return StepContext(
            model=model,
            template=self.template,
            recipe=self.recipe,
            period=period,
            prior=self.recipe.prior(period),
            shock=dict(shock),
        )

    def _plan(self) -> KindPlan:
        return self.recipe.plan(self._period())

    def shock_system(
        self,
        shock: Any,
        *,
        start: MPArrays | None = None,
        savf_targets: Mapping[str, float] | None = None,
        shock_levels: bool = False,
    ) -> tuple[Any, Squared, MPArrays]:
        """The shock period at ``shock`` prepared as :meth:`solve_shock` prepares one
        solve (the shock plan's steps before the solve, square, system build), not
        solved: ``(ss, sq, mp_sq)``, ``ss.x0`` the start point. ``savf_targets``
        (``{region: savf}``) holds the held regions' savf there
        (``trade_balance="ratio"``)."""
        if shock_levels:
            raise NotImplementedError("shock_levels: the core writes % shocks only")
        model = self._model(shock, start)
        recipe = self._recipe(warm=start is not None)
        ctx = self._context(model, shock)
        ctx = dataclasses.replace(ctx, recipe=recipe)
        for step in recipe.plan(self._period()).before:
            step(ctx)
        if savf_targets:
            sv = model.mp.vars["savf"]
            for r, value in savf_targets.items():
                sv.value[sv.pos((r, SHOCK))] = value
        sq, mp_sq, ss = _period_system(self.run, model.mp, SHOCK)
        # the closure the period was staged with: finish_shock's steps read it
        self._staged = model.staged
        return ss, sq, mp_sq

    def finish_shock(self, mp_sq: MPArrays, sq: Squared, x: np.ndarray) -> MPArrays:
        """The shock period at the point ``x`` of ``sq`` (from :meth:`shock_system`)
        as after a converged solve: the solution written back (as the core solver
        writes it), then the shock plan's steps after the solve, with the closure the
        period was staged with. ``mp_sq`` is modified and returned."""
        from equilibria.family.post_solve import writeback

        writeback(mp_sq, sq, x)
        model = GTAPFamilyPeriodModel(self.run, mp_sq)
        st = self._staged
        if st is None:
            raise RuntimeError("finish_shock needs a shock_system call first")
        model.stage(st.closure, st.params, gtap_mode=st.gtap_mode, period=SHOCK)
        ctx = self._context(model, {})
        ctx.status = "optimal"
        for step in self._plan().after:
            step(ctx)
        return model.mp


def _write_levels(mp: MPArrays, shock: Mapping[str, Mapping[tuple, float]]) -> None:
    """Each shock cell's level in the shock period, from its check level (``level``).
    The run's own shock validated these cells when it wrote them; here the shock
    period's rows are not live yet (the period is staged after), so the cells are
    written without ``apply_shock``'s live-row check, as the path's Newton steps
    write theirs."""
    from equilibria.family.shock import level

    for name, cells in shock.items():
        var = mp.vars[name]
        for cell, pct in cells.items():
            var.value[var.pos((*tuple(cell), SHOCK))] = level(
                mp, name, tuple(cell), pct
            )


def _refresh_aggregates(ctx: StepContext) -> None:
    """A warm start's seed: the cells keep the start arrays' levels; the Fisher
    aggregates are refreshed at them."""
    cast("GTAPFamilyPeriodModel", ctx.model).refresh_fisher_aggregates()


def _code(status: str) -> int:
    """The core status as the old int code: ``"optimal"`` is 1, ``"code <n>"`` is
    ``n``, any other status (``"not solved"``) 2. An ``"error: ..."`` status (the
    solver raised; the core keeps the message) raises again, as the old driver let
    the exception through: its last word must never be read as a code."""
    status = str(status)
    if status == "optimal":
        return 1
    if status.startswith("error:"):
        raise RuntimeError(f"the solve raised: {status[len('error:') :].strip()}")
    m = re.fullmatch(r"code (-?\d+)", status)
    return int(m.group(1)) if m else 2
