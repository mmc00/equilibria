"""``GTAPPeriodModel`` over ``MPArrays`` (ADR-0011): GTAP on the family backend.

The cell operations are ``FamilyPeriodModel``'s. The GTAP operations call the array
functions the old family route runs (``post_solve``, ``family.shock``,
``cross_period``), with the run context ``prepare_run`` made (``self.run``), so both
routes run the same code until the old one is deleted. No Pyomo model is built.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from equilibria.family.mp_arrays import MPArrays
from equilibria.family.periods import PERIODS, FamilyPeriodModel, positions
from equilibria.templates.gtap.period_model import Reference, Staged


class GTAPFamilyPeriodModel(FamilyPeriodModel):
    """``GTAPPeriodModel`` over ``self.mp``; ``self.run`` is the ``FamilyRun``."""

    def __init__(
        self, run: Any, mp: MPArrays, periods: tuple[str, ...] = PERIODS
    ) -> None:
        super().__init__(mp, periods)
        self.run = run
        self._staged: Staged | None = None
        self.last_solve: dict[str, Any] | None = None
        self._solved_period: str | None = None

    # ── exogeneity ─────────────────────────────────────────────────────────

    def exogenous_cells(self, name: str, period: str) -> frozenset[tuple]:
        if name not in self.mp.exogenous or name not in self.mp.vars:
            return frozenset()
        cells = positions(self.mp.vars[name], period)
        libres = self.mp.endogenous_instrument_cells.get(name)
        if not libres:
            return frozenset(cells)
        from equilibria.templates.gtap.instruments import is_free_cell

        return frozenset(k for k in cells if not is_free_cell((*k, period), libres))

    def exogenous(self) -> frozenset[tuple[str, tuple]]:
        return frozenset(
            (name, k)
            for name in self.mp.exogenous
            for k in self.exogenous_cells(name, "shock")
        )

    # ── what the build left on the arrays ──────────────────────────────────

    @property
    def residual_region(self) -> str:
        return self.run.residual_region

    @property
    def base_calibrated(self) -> bool:
        return bool(self.mp.base_calibrated)

    @property
    def settled_seed(self) -> Mapping[str, Mapping[Any, Any]] | None:
        return self.mp.settled_seed

    # ── the solve's inputs ─────────────────────────────────────────────────

    @property
    def staged(self) -> Staged | None:
        return self._staged

    def stage(self, closure: Any, params: Any, *, gtap_mode: bool, period: str) -> None:
        self.mp.gtap_mode = bool(gtap_mode)
        self._staged = Staged(closure, params, bool(gtap_mode), period)

    def record_solve(self, result: Mapping[str, Any]) -> None:
        self.last_solve = dict(result)
        self._solved_period = self._staged.period if self._staged else None

    def residual(self, period: str) -> float:
        if self.last_solve is None or self._solved_period != period:
            return math.nan
        r = self.last_solve.get("residual")
        return math.inf if r is None else float(r)

    def sp_reference(
        self, closure: Any, params: Any, period: str, *, fresh: bool
    ) -> Reference:
        """``run.ref``, the reference ``prepare_run`` read once: the old family route
        uses it for every period (``fresh`` changes nothing here)."""
        fixed: dict[str, dict[tuple, float]] = {}
        bounds: dict[str, dict[tuple, tuple[float | None, float | None]]] = {}
        for name, sp in self.run.ref.vars.items():
            if sp.fixed.any():
                fixed[name] = {
                    sp.keys[i]: float(sp.value[i]) for i in np.flatnonzero(sp.fixed)
                }
            lo, hi = np.isfinite(sp.lb), np.isfinite(sp.ub)
            if (lo | hi).any():
                bounds[name] = {
                    sp.keys[i]: (
                        float(sp.lb[i]) if lo[i] else None,
                        float(sp.ub[i]) if hi[i] else None,
                    )
                    for i in np.flatnonzero(lo | hi)
                }
        return Reference(fixed, bounds)

    # ── GTAP operations that are not cell edits ────────────────────────────

    def recalibrate_and_ava(self, period: str, prior: str, params: Any) -> int:
        raise NotImplementedError("family backend: altertax is not supported")

    def refresh_fisher_aggregates(self) -> None:
        from equilibria.family.cross_period import refresh_aggregates

        refresh_aggregates(self.run.cross, self.mp)

    def solve_welfare_tail(self, period: str) -> int:
        from equilibria.family.post_solve import solve_welfare_tail

        n, _ = solve_welfare_tail(self.mp, self.run.backend, self.run.state, period)
        return n

    def recompute_ifsub_report_vars(self, period: str, params: Any) -> int:
        from equilibria.family.post_solve import recompute_ifsub_report_vars

        return recompute_ifsub_report_vars(self.mp, params, self.run.state, period)

    def write_shock(self, shock: Mapping[str, Mapping[tuple, float]]) -> None:
        from equilibria.family.shock import apply_shock

        apply_shock(self.mp, self.run.backend, shock)

    def write_default_tariff(self, *, gtap_mode: bool) -> None:
        from equilibria.family.shock import apply_default_tariff

        apply_default_tariff(self.mp, self.run.backend)

    def shock_labels(self) -> list[str]:
        from equilibria.family.post_solve import shocked_cells

        return [
            f"{name}{tuple(k[:-1])}"
            for name in sorted(self.mp.exogenous)
            for k in shocked_cells(self.mp, name)
        ]

    def check_endowment_shock_entered(self) -> None:
        from equilibria.family.post_solve import check_endowment_shock_entered

        check_endowment_shock_entered(self.mp, self.run.backend, self.run.state)

    def debug_before_solve(self, period: str) -> None:
        return None
