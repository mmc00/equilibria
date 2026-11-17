"""One GTAP case through the old driver or through the core, for the parity gates.

``solve(route, ...)`` takes the old driver's options (``solve_multiperiod``'s
names and defaults) and runs them either way:

- ``"old"``: ``build_block_model`` + ``apply_shock`` + ``solve_multiperiod`` on
  Pyomo; ``solve_gtap_family`` on family.
- ``"core"``: ``Exercise(GTAP(...), gtap_static(...), shock=...)`` on the backend.

Both give a ``Solved``: the solved model (the Pyomo model, or the family
``MPArrays``) and the codes in the old driver's form (``{period: int}``, the
skipped base as 1). Each parity gate is parametrized over ``ROUTES``; PR 3 of the
gtap-on-core design drops ``"old"`` and nothing else changes in the gates.

``@overwrite`` hooks are the global registry on both routes (the core build reads
it too); the caller registers and clears them, as before.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

ROUTES = ("old", "core")
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
    """The core driver's statuses in the old driver's form: ``"optimal"`` -> 1,
    ``"code N"`` -> N; a period not solved is left out, but for the skipped base,
    which the old driver reports as 1."""
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
    mode: str = "gtap",
    skip_base_solve: bool = False,
    mute_welfare: bool = True,
    seed_from_prior: bool = False,
    solve_check: bool = False,
    settle_only: bool = False,
    shock_steps: tuple[float, ...] | None = None,
    shock_method: str = "nonlinear",
    trade_balance: str | None = None,
    **template_kw: Any,
) -> Solved:
    """Run one case through ``route`` on ``backend``. ``template_kw`` goes to the
    family run on both routes (``cache_root``, ``jax_cache``...)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "family":
        return _family(
            route,
            params,
            closure,
            residual_region,
            shock=shock,
            ref_gdx=ref_gdx,
            trade_balance=trade_balance,
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
        if route == "old":
            from equilibria.templates.gtap.gtap_block_model import build_block_model
            from equilibria.templates.gtap.gtap_multiperiod_driver import (
                solve_multiperiod,
            )
            from equilibria.templates.gtap.shocks import apply_shock

            m, _ = build_block_model(
                params,
                params.sets,
                closure,
                residual_region,
                base_calibrated=base_calibrated,
                ref_gdx=ref_gdx,
            )
            if shock:
                apply_shock(m, shock)
            res = solve_multiperiod(
                m,
                params,
                closure,
                ref_gdx=ref_gdx,
                skip_base_solve=skip_base_solve,
                mode=mode,
                **opts,
            )
            return Solved(
                model=m,
                codes={k: int(r["code"]) for k, r in res.items()},
                route=route,
                backend=backend,
                residuals={k: r.get("residual") for k, r in res.items()},
            )
        from equilibria.core.exercise import Exercise
        from equilibria.templates.gtap.recipe import gtap_static
        from equilibria.templates.gtap.template import GTAP

        ex = Exercise(
            template=GTAP(
                params,
                closure,
                residual_region=residual_region,
                base_calibrated=base_calibrated,
                ref_gdx=ref_gdx,
            ),
            recipe=gtap_static(
                mode,
                base_calibrated=base_calibrated,
                solve_base=not skip_base_solve,
                **opts,
            ),
            shock=shock,
        )
        res = ex.run()
        model = cast("Any", ex.model)
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
    **template_kw: Any,
) -> Solved:
    """The family route runs what ``solve_gtap_family`` runs: base not solved
    (calibrated), check solved, mode ``gtap``."""
    with _backend_env(None):
        if route == "old":
            from equilibria.family.driver import solve_gtap_family

            out = solve_gtap_family(
                params,
                closure,
                residual_region,
                ref_gdx=ref_gdx,
                shock=shock,
                nominal_trade_balance=trade_balance or False,
                **template_kw,
            )
            return Solved(
                model=out.mp,
                codes={k: int(r["code"]) for k, r in out.results.items()},
                route=route,
                backend="family",
                residuals={k: r.get("residual") for k, r in out.results.items()},
            )
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
                base_calibrated=True, solve_check=True, trade_balance=trade_balance
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
