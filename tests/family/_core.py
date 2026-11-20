"""One GTAP run on the family backend through the core driver, for the tests that
ran it through the old family driver until PR 4 of gtap-on-core.

``family_exercise`` runs what the old ``family`` entry point ran: the base not
solved (calibrated), the check and the shock solved, ``trade_balance`` as given.
"""

from __future__ import annotations

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
