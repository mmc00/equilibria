"""The GTAP template for ``Exercise`` (ADR-0011).

``GTAP(params, closure)`` carries its data and builds the period model for its
backend::

    Exercise(template=GTAP(params, closure), recipe=gtap_static(),
             shock={"imptx": {("USA", "Mnfcs", "EU_28"): 5.0}}).run()

Not a ``ModelTemplate``: that base builds through the core reflection, and GTAP
builds its multiperiod model from its blocks (``build_block_model``), with the
GTAP-only cross-period rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from equilibria.core.recipe import Recipe

BACKENDS = ("pyomo", "family")
_PERIODS = (("base", "check", "shock"), ("base", "check"))


class GTAP:
    def __init__(
        self,
        params: Any,
        closure: Any,
        *,
        residual_region: str | None = None,
        base_calibrated: bool = False,
        ref_gdx: Any = None,
        backend: str = "pyomo",
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.params = params
        self.closure = closure
        self.residual_region = residual_region
        self.base_calibrated = base_calibrated
        self.ref_gdx = ref_gdx
        self.backend = backend

    def build(self, periods: Sequence[str], overwrite: Sequence[Any] = ()) -> Any:
        """A fresh multiperiod model (``base``, ``check``, ``shock``; a recipe
        may use the first two only)."""
        if tuple(periods) not in _PERIODS:
            raise ValueError(
                f"GTAP builds the periods base, check, shock; got {tuple(periods)}"
            )
        if overwrite:
            raise NotImplementedError(
                "pass GTAP @overwrite hooks through their registry (@overwrite) "
                "for now; Exercise(overwrite=...) for GTAP comes with the switch "
                "(PR 3 of gtap-on-core)"
            )
        if self.backend == "family":
            raise NotImplementedError(
                "the family backend joins the GTAP template in part B of PR 1"
            )
        from equilibria.templates.gtap.gtap_block_model import build_block_model
        from equilibria.templates.gtap.pyomo_periods import GTAPPyomoPeriodModel

        rr = self.residual_region or list(self.params.sets.r)[-1]
        m, _ = build_block_model(
            self.params,
            self.params.sets,
            self.closure,
            rr,
            base_calibrated=self.base_calibrated,
            ref_gdx=self.ref_gdx,
        )
        return GTAPPyomoPeriodModel(m)

    def run_scope(self) -> Any:
        """The GC setup of the whole run, as ``solve_multiperiod`` held it: the
        built model frozen once over base, check and shock. A solve inside it
        leaves it alone (``_gc_family_scope`` is reentrant); freezing per solve
        instead walked the whole model twice in every gap between solves
        (measured on 15x10: +2.3 s of 23.5, nearly all of it GC)."""
        from equilibria.solver.gc_tuning import _gc_family_scope

        return _gc_family_scope()

    def default_recipe(self) -> Recipe:
        from equilibria.templates.gtap.recipe import gtap_static

        return gtap_static(base_calibrated=self.base_calibrated)
