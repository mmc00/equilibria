"""GTAPPeriodModel over the Pyomo multiperiod model ``build_block_model`` builds.

The cell operations are ``PyomoPeriodModel``'s. The GTAP operations call the
helpers of the multiperiod driver they replace, so both routes run the same code
until the old one is deleted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from equilibria.backends.pyomo_periods import PyomoPeriodModel
from equilibria.templates.gtap.instruments import exogenous_test
from equilibria.templates.gtap.period_model import Reference, Staged


def _drv() -> Any:
    from equilibria.templates.gtap import gtap_multiperiod_driver

    return gtap_multiperiod_driver


def _as_reference(snap: Any, period: str) -> Reference:
    d = _drv()
    fixed = {
        name: {d._mp_index(ix, k, period)[:-1]: v for k, v in cells}
        for name, (ix, cells) in snap.fixed.items()
    }
    bounds = {
        name: {d._mp_index(ix, k, period)[:-1]: (lb, ub) for k, lb, ub in cells}
        for name, (ix, cells) in snap.bounds.items()
    }
    return Reference(fixed, bounds)


class GTAPPyomoPeriodModel(PyomoPeriodModel):
    """``GTAPPeriodModel`` over a GTAP multiperiod Pyomo model (``self.pyomo``)."""

    def __init__(self, pyomo: Any) -> None:
        super().__init__(pyomo)
        self._staged: Staged | None = None
        self._preparer: Any = None
        self.last_solve: dict[str, Any] | None = None

    # ── exogeneity ─────────────────────────────────────────────────────────

    def exogenous_cells(self, name: str, period: str) -> frozenset[tuple]:
        if name not in getattr(self.pyomo, "_exogenous_instruments", ()):
            return frozenset()
        test = exogenous_test(self.pyomo, name)
        return frozenset(
            k for k, c in self._var(name, period).items() if test(c.index())
        )

    def exogenous(self) -> frozenset[tuple[str, tuple]]:
        return frozenset(
            (name, k)
            for name in getattr(self.pyomo, "_exogenous_instruments", ())
            for k in self.exogenous_cells(name, "shock")
        )

    # ── what the build left on the model ───────────────────────────────────

    @property
    def residual_region(self) -> str:
        return self.pyomo._residual_region

    @property
    def base_calibrated(self) -> bool:
        return bool(getattr(self.pyomo, "_base_calibrated", False))

    @property
    def settled_seed(self) -> Mapping[str, Mapping[Any, Any]] | None:
        return getattr(self.pyomo, "_settled_seed", None)

    # ── the solve's inputs ─────────────────────────────────────────────────

    @property
    def staged(self) -> Staged | None:
        return self._staged

    def stage(self, closure: Any, params: Any, *, gtap_mode: bool) -> None:
        m = self.pyomo
        if getattr(m, "_residual_region", None) is None:
            m._residual_region = list(params.sets.r)[-1]
        m._gtap_mode = bool(gtap_mode)
        self._staged = Staged(closure, params, bool(gtap_mode))

    def reference(
        self, closure: Any, params: Any, period: str, *, fresh: bool
    ) -> Reference:
        prep = self._reference_cache()
        if fresh:
            snap = prep._build_reference_model(self.pyomo, params, closure)
        else:
            snap = prep._reference_model(self.pyomo, params, closure, period)
        return _as_reference(snap, period)

    def _reference_cache(self) -> Any:
        """The reference models of this run, cached as ``PeriodPreparer`` caches
        them (and shared with other runs on the same ``params``)."""
        if self._preparer is None:
            from equilibria.templates.gtap.period_prep import PeriodPreparer

            self._preparer = PeriodPreparer(
                mode="gtap",
                base_closure=None,
                alt_closure=None,
                residual_region=self.residual_region,
            )
        return self._preparer

    # ── GTAP operations that are not cell edits ────────────────────────────

    def recalibrate_and_ava(self, period: str, prior: str, params: Any) -> int:
        return _drv()._recalibrate_and_ava(self.pyomo, params, period, prior)

    def refresh_fisher_aggregates(self) -> None:
        from equilibria.templates.gtap.gtap_model_multiperiod import (
            GTAPMultiPeriodModel,
        )

        GTAPMultiPeriodModel.refresh_fisher_aggregates(self.pyomo)

    def solve_welfare_tail(self, period: str) -> int:
        return _drv()._solve_welfare_tail(self.pyomo, period)

    def recompute_ifsub_report_vars(self, period: str, params: Any) -> int:
        return _drv()._recompute_ifsub_report_vars(self.pyomo, params, period)

    def write_shock(self, shock: Mapping[str, Mapping[tuple, float]]) -> None:
        from equilibria.templates.gtap.shocks import apply_shock

        apply_shock(self.pyomo, shock)

    def write_default_tariff(self, *, gtap_mode: bool) -> None:
        from equilibria.templates.gtap.shocks import apply_default_tariff

        apply_default_tariff(self.pyomo, gtap_mode=gtap_mode)

    def shock_labels(self) -> list[str]:
        from equilibria.templates.gtap.shocks import shock_of

        return shock_of(self.pyomo)

    def check_endowment_shock_entered(self) -> None:
        from equilibria.templates.gtap.shocks import check_endowment_shock_entered

        check_endowment_shock_entered(self.pyomo)

    def debug_before_solve(self, period: str) -> None:
        from equilibria.templates.gtap.period_debug import debug_before_solve

        debug_before_solve(self.pyomo, period)
