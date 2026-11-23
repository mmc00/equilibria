"""One core recipe step on a bare GTAP multiperiod Pyomo model, for the tests that
called the old driver's helper doing the same (``freeze_inactive``, ``seed_from``,
``copy_from_base``, ``gams_bounds``...)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def run_step(
    step: Any, m: Any, period: str, *, staged: tuple[Any, Any, bool] | None = None
) -> None:
    """``step`` on the Pyomo model ``m`` with ``period`` the active one; ``staged``
    (``(closure, params, gtap_mode)``): the period staged first, for the steps that
    read it."""
    from equilibria.templates.gtap.pyomo_periods import GTAPPyomoPeriodModel

    model = GTAPPyomoPeriodModel(m)
    if staged is not None:
        closure, params, gtap_mode = staged
        model.stage(closure, params, gtap_mode=gtap_mode, period=period)
    step(SimpleNamespace(model=model, period=SimpleNamespace(name=period)))
