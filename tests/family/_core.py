"""One GTAP run on the family backend through the core driver, for the tests that
ran it through the old family driver until PR 4 of gtap-on-core.

``family_exercise`` runs what the old ``family`` entry point ran: the base not
solved (calibrated), the check and the shock solved, ``trade_balance`` as given.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def family_exercise(
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    shock: Any = None,
    ref_gdx: Any = None,
    trade_balance: str | None = None,
    **template_kw: Any,
) -> tuple[Any, dict]:
    """``(the family period model, the run's results)``; results by period, each
    with ``code`` ("optimal" when solved) and ``residual``."""
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
    return ex.model, res


def no_solve(recipe: Any, stop: str = "shock") -> Any:
    """``recipe`` with no solve: period ``stop`` ends after its steps (nothing after
    them), the periods before it only freeze themselves after theirs."""
    import dataclasses

    from equilibria.templates.gtap import steps as s

    kind = next(p.kind for p in recipe.periods if p.name == stop)
    plans = {
        k: dataclasses.replace(
            plan, solver=None, after=() if k == kind else (s.freeze_period,)
        )
        for k, plan in recipe.plans.items()
    }
    return dataclasses.replace(recipe, plans=plans)


def family_template(
    params: Any, closure: Any, residual_region: str, root: Any
) -> tuple[Any, Any]:
    """``(template, arrays)``: a family ``GTAP`` (not calibrated: no settle) built
    once, whose run the core steps use, and the arrays its build gave; its records
    under ``root``."""
    from equilibria.templates.gtap.template import GTAP

    t = GTAP(
        params,
        closure,
        residual_region=residual_region,
        backend="family",
        cache_root=root,
        jax_cache=False,
    )
    with newton_env():
        model = t.build(("base", "check", "shock"))
    return t, model.mp


def core_prep(built: tuple[Any, Any], mp: Any, period: str) -> Any:
    """``mp`` (a copy) with ``period`` prepared by the core: the period's steps of
    ``gtap_static(solve_check=True)``, without its solve, on the run of ``built``
    (``family_template``). From a photo A, the state the photo's run handed the
    squaring; a photo carries no instrument registry, so the build's is used."""
    import dataclasses

    from equilibria.core.driver import fly_period
    from equilibria.family.gtap_periods import GTAPFamilyPeriodModel
    from equilibria.templates.gtap.recipe import gtap_static

    template, like = built
    mp = dataclasses.replace(
        mp.copy(),
        exogenous=like.exogenous,
        endogenous_instrument_cells=like.endogenous_instrument_cells,
        fctts=like.fctts,
        residual_region=like.residual_region,
    )
    model = GTAPFamilyPeriodModel(template._family_run, mp)
    with newton_env():
        fly_period(
            model,
            no_solve(gtap_static(solve_check=True), stop=period),
            period,
            template=template,
        )
    return model.mp


@contextmanager
def newton_env() -> Iterator[None]:
    """The family backend's solver settings (Newton-TR, MUMPS) in the environment."""
    import pytest

    from tests.family._prep import _newton_mumps

    with pytest.MonkeyPatch.context() as env:
        for k, v in _newton_mumps().items():
            env.setenv(k, v)
        yield
