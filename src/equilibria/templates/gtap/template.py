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

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

from equilibria.core.recipe import Recipe

BACKENDS = ("pyomo", "family")
# the multiperiod model: the blocks composer, or one of its two variants (padding
# trade routes fixed; every row log-wrapped), all solved the same way
BUILDERS = ("blocks", "sparse", "loglevels")
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
        seed_gdx: Any = None,
        backend: str = "pyomo",
        builder: str = "blocks",
        cache_root: Any = None,
        jax_cache: bool = True,
        newton_tol: tuple[float, float] | None = None,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if builder not in BUILDERS:
            raise ValueError(f"builder must be one of {BUILDERS}, got {builder!r}")
        if backend == "family" and builder != "blocks":
            raise NotImplementedError(
                f"family backend: builder={builder!r} is not supported (blocks only)"
            )
        self.params = params
        self.closure = closure
        self.residual_region = residual_region
        self.base_calibrated = base_calibrated
        self.ref_gdx = ref_gdx
        # a GAMS solution every period starts from (``seed_all_periods``), as the
        # GAMS parity gates seed it; pyomo only
        self.seed_gdx = seed_gdx
        self.backend = backend
        self.builder = builder
        # the sparse builder's report of what it fixed (``fix_padding_routes``),
        # set by ``build``; None for the other builders
        self.fix_stats: dict[str, Any] | None = None
        # family only: where the recorded families live, the JAX compile cache,
        # and (reltol, ftol) of every Newton solve (None: the environment's)
        self.cache_root = cache_root
        self.jax_cache = jax_cache
        self.newton_tol = newton_tol
        self._family_run: Any = None

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
            if self.seed_gdx is not None:
                raise NotImplementedError("family backend: seed_gdx is not supported")
            return self._build_family()
        from equilibria.templates.gtap.pyomo_periods import GTAPPyomoPeriodModel

        rr = self.residual_region or list(self.params.sets.r)[-1]
        args = (self.params, self.params.sets, self.closure, rr)
        kw = {"base_calibrated": self.base_calibrated, "ref_gdx": self.ref_gdx}
        if self.builder == "sparse":
            from equilibria.templates.gtap_sparse.multiperiod import (
                build_sparse_model_mp,
            )

            m, mp, self.fix_stats = build_sparse_model_mp(*args, **kw)
        elif self.builder == "loglevels":
            from equilibria.templates.gtap_loglevels.multiperiod import (
                build_loglevels_model_mp,
            )

            m, mp = build_loglevels_model_mp(*args, **kw)
        else:
            from equilibria.templates.gtap.gtap_block_model import build_block_model

            m, mp = build_block_model(*args, **kw)
        if self.seed_gdx is not None:
            mp.seed_all_periods(m, self.seed_gdx)
        return GTAPPyomoPeriodModel(m)

    def _build_family(self) -> Any:
        """``solve_gtap_family``'s setup: refuse what the route cannot run, load
        or record the families (``prepare_run``), then with ``base_calibrated`` the
        settled seed (``calibrate_base_cached``). No Pyomo model is built."""
        import dataclasses

        from equilibria.blocks.gtap.overwrite import overwrite
        from equilibria.family import run_setup as fd
        from equilibria.family.capflex import require_ref_gdx
        from equilibria.family.gtap_periods import GTAPFamilyPeriodModel
        from equilibria.solver.gc_tuning import (
            _gc_family_scope,
            _gc_freeze_from_env,
            _gc_freeze_now,
        )

        if overwrite.active():
            raise NotImplementedError(
                "family backend: @overwrite hooks are not supported"
            )
        savf = str(getattr(self.closure, "savf_flag", "capFix"))
        # the closure and environment settings; the recipe's are refused by its
        # first step (check_backend), which sees them
        fd.check_supported(
            if_sub=bool(getattr(self.closure, "if_sub", False)),
            savf_flag=savf,
            mode="gtap",
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            solve_check=True,
            shock_steps=None,
            shock_method="nonlinear",
        )
        if savf == "capFlex":
            require_ref_gdx(self.ref_gdx)
        rr = self.residual_region or list(self.params.sets.r)[-1]
        # as solve_gtap_family: the load and the settle under one GC setup; the
        # loaded records frozen as soon as they are in
        with _gc_family_scope():
            run, mp = fd.prepare_run(
                self.params,
                self.closure,
                rr,
                cache_root=self.cache_root,
                jax_cache=self.jax_cache,
                freeze=_gc_freeze_from_env(),
                ref_gdx=self.ref_gdx,
                newton_tol=self.newton_tol,
            )
            _gc_freeze_now(_gc_freeze_from_env())
            if self.base_calibrated:
                settled = self._settle_family(run, mp)
                mp = dataclasses.replace(mp, settled_seed=settled, base_calibrated=True)
        self._family_run = run
        return GTAPFamilyPeriodModel(run, mp)

    def _settle_family(self, run: Any, mp: Any) -> dict[str, dict]:
        """``FactorBlock.calibrate_base`` on the family model, through the core
        recipe: base and check of ``gtap_static(settle_only=True)`` on a copy of
        the arrays, then the check slice without the instruments
        (``settled_check_point``). The opt-in settle cache wraps it as
        ``calibrate_base_cached`` does: only a converged settle is saved."""
        from equilibria.blocks.gtap import seed_cache
        from equilibria.core.driver import fly
        from equilibria.family import run_setup as fd
        from equilibria.family.gtap_periods import GTAPFamilyPeriodModel
        from equilibria.templates.gtap.recipe import gtap_static

        key = fd._settle_key(run)
        if key is not None:
            cached = seed_cache.load(key)
            if cached is not None:
                return cached
        # every period of the arrays: freeze_inactive fixes the shock cells too
        model = GTAPFamilyPeriodModel(run, mp.copy())
        results = fly(model, gtap_static(settle_only=True), template=self)
        settled = fd.settled_check_point(model.mp)
        if key is not None and results["check"]["code"] == "optimal":
            seed_cache.save(key, settled)
        return settled

    @contextlib.contextmanager
    def run_scope(self) -> Iterator[None]:
        """The GC setup of the whole run, as ``solve_multiperiod`` held it: the
        built model frozen once over base, check and shock. A solve inside it
        leaves it alone (``_gc_family_scope`` is reentrant); freezing per solve
        instead walked the whole model twice in every gap between solves
        (measured on 15x10: +2.3 s of 23.5, nearly all of it GC).

        On family the freeze skips the full collection (the build already froze
        the records: a collection would walk them, 7.4 s on 20x41) and the run's
        MUMPS analysis is dropped at the end, as ``solve_gtap_family`` drops it."""
        from equilibria.solver.gc_tuning import _gc_family_scope

        family = self.backend == "family"
        try:
            with _gc_family_scope(collect=not family):
                yield
        finally:
            if family and self._family_run is not None:
                self._family_run.mumps.clear()

    def default_recipe(self) -> Recipe:
        from equilibria.templates.gtap.recipe import gtap_static

        return gtap_static(base_calibrated=self.base_calibrated)
