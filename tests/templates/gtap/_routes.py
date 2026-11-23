"""One GTAP case through the core, for the parity gates.

``solve(route, ...)`` takes the options the parity gates pass (the names and
defaults of the old multiperiod driver they were written against) and runs
``Exercise(GTAP(...), gtap_static(...), shock=...)`` on the backend.

It gives a ``Solved``: the solved model (the Pyomo model, or the family
``MPArrays``) and the codes as ints (``{period: int}``, the skipped base as 1).
Each parity gate is parametrized over ``ROUTES``; the core is the only route
since PR 4 of gtap-on-core deleted the old drivers.

``@overwrite`` hooks are the global registry (the core build reads it); the
caller registers and clears them.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

ROUTES = ("core",)
BACKENDS = ("pyomo", "family")
_BACKEND_ENV = "EQUILIBRIA_GTAP_BACKEND"


@dataclass(frozen=True)
class Solved:
    model: Any
    codes: dict[str, int]
    route: str
    backend: str
    residuals: dict[str, Any] = field(default_factory=dict)


def core_codes(
    res: Mapping[str, Mapping[str, Any]], *, skip_base_solve: bool
) -> dict[str, int]:
    """The core driver's statuses as ints: ``"optimal"`` -> 1, ``"code N"`` -> N; a
    period not solved is left out, but for the skipped base, reported as 1."""
    from equilibria.core.driver import NOT_SOLVED

    out: dict[str, int] = {}
    for period, r in res.items():
        code = r["code"]
        if code == "optimal":
            out[period] = 1
        elif isinstance(code, str) and code.startswith("code "):
            out[period] = int(code.removeprefix("code "))
        elif code == NOT_SOLVED:
            if period == "base" and skip_base_solve:
                out[period] = 1
        else:
            raise ValueError(f"{period}: unknown core status {code!r}")
    return out


@contextlib.contextmanager
def _backend_env(value: str | None) -> Iterator[None]:
    """``EQUILIBRIA_GTAP_BACKEND`` set to ``value`` (unset if None) for the run,
    then back to what it was."""
    before = os.environ.get(_BACKEND_ENV)
    if value is None:
        os.environ.pop(_BACKEND_ENV, None)
    else:
        os.environ[_BACKEND_ENV] = value
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(_BACKEND_ENV, None)
        else:
            os.environ[_BACKEND_ENV] = before


def solve(
    route: str,
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    backend: str = "pyomo",
    shock: Mapping[str, Mapping[tuple, float]] | None = None,
    base_calibrated: bool = False,
    ref_gdx: Any = None,
    seed_gdx: Any = None,
    mode: str = "gtap",
    skip_base_solve: bool = False,
    mute_welfare: bool = True,
    seed_from_prior: bool = False,
    solve_check: bool = False,
    settle_only: bool = False,
    shock_steps: tuple[float, ...] | None = None,
    shock_method: str = "nonlinear",
    trade_balance: str | None = None,
    edit_model: Callable[[Any], None] | None = None,
    **template_kw: Any,
) -> Solved:
    """Run one case through ``route`` on ``backend``. ``seed_gdx``: every period
    starts from that GAMS solution (``seed_all_periods``), as the GAMS gates seed
    it. ``edit_model`` (core, pyomo): called once with the built Pyomo model before
    the recipe flies (``Exercise.run`` in its two parts), e.g. to fix an instrument
    by hand (``fix_instrument_shock``). ``template_kw`` goes to the family template
    (``cache_root``, ``jax_cache``...)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if edit_model is not None and backend != "pyomo":
        raise ValueError("edit_model runs on the pyomo backend only")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "family":
        if seed_gdx is not None:
            raise ValueError("seed_gdx runs on the pyomo backend only")
        return _family(
            route,
            params,
            closure,
            residual_region,
            shock=shock,
            ref_gdx=ref_gdx,
            trade_balance=trade_balance,
            mode=mode,
            **template_kw,
        )
    if template_kw:
        raise TypeError(f"pyomo takes no template options: {sorted(template_kw)}")
    if trade_balance is not None:
        raise ValueError("trade_balance runs on the family backend only")
    opts: dict[str, Any] = {
        "mute_welfare": mute_welfare,
        "seed_from_prior": seed_from_prior,
        "solve_check": solve_check,
        "settle_only": settle_only,
        "shock_steps": shock_steps,
        "shock_method": shock_method,
    }
    with _backend_env(None):
        from equilibria.core.exercise import Exercise
        from equilibria.templates.gtap.recipe import gtap_static
        from equilibria.templates.gtap.template import GTAP

        gtap = GTAP(
            params,
            closure,
            residual_region=residual_region,
            base_calibrated=base_calibrated,
            ref_gdx=ref_gdx,
            seed_gdx=seed_gdx,
        )
        recipe = gtap_static(
            mode,
            base_calibrated=base_calibrated,
            solve_base=not skip_base_solve,
            **opts,
        )
        if edit_model is None:
            ex = Exercise(template=gtap, recipe=recipe, shock=shock)
            res = ex.run()
            model = cast("Any", ex.model)
        else:
            from equilibria.core.driver import fly

            model = cast("Any", gtap.build(recipe.names))
            edit_model(model.pyomo)
            with gtap.run_scope():
                res = fly(model, recipe, template=gtap, shock=shock)
        return Solved(
            model=model.pyomo,
            codes=core_codes(res, skip_base_solve=skip_base_solve),
            route=route,
            backend=backend,
            residuals={k: r.get("residual") for k, r in res.items()},
        )


def _family(
    route: str,
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    shock: Mapping[str, Mapping[tuple, float]] | None,
    ref_gdx: Any,
    trade_balance: str | None,
    mode: str = "gtap",
    **template_kw: Any,
) -> Solved:
    """The family route: base not solved (calibrated), check solved. ``mode``
    reaches the recipe, which refuses anything but ``"gtap"`` on family."""
    with _backend_env(None):
        from equilibria.core.exercise import Exercise
        from equilibria.templates.gtap.recipe import gtap_static
        from equilibria.templates.gtap.template import GTAP

        ex = Exercise(
            GTAP(
                params,
                closure,
                residual_region=residual_region,
                base_calibrated=True,
                ref_gdx=ref_gdx,
                backend="family",
                **template_kw,
            ),
            gtap_static(
                mode,
                base_calibrated=True,
                solve_check=True,
                trade_balance=trade_balance,
            ),
            shock=shock,
        )
        res = ex.run()
        model = cast("Any", ex.model)
        return Solved(
            model=model.mp,
            codes=core_codes(res, skip_base_solve=True),
            route=route,
            backend="family",
            residuals={k: r.get("residual") for k, r in res.items()},
        )
