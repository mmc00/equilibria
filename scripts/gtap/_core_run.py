"""One GTAP comparative-statics run on the core driver, for the scripts.

``solve_on_core`` builds ``Exercise(GTAP(...), gtap_static(...), shock=...)``,
runs it and gives back the solved Pyomo model and the codes in the old driver's
form (``{period: int}``: ``"optimal"`` is 1, ``"code N"`` is N, the skipped base
is 1), so a script's output reads as it did on the old Pyomo driver.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def old_codes(res: Mapping[str, Mapping[str, Any]], *, solve_base: bool) -> dict[str, int]:
    """The core statuses in the old driver's form; a period the core did not
    solve is left out, but for the skipped base, which the old driver reports as 1."""
    from equilibria.core.driver import NOT_SOLVED

    out: dict[str, int] = {}
    for period, r in res.items():
        code = r["code"]
        if code == "optimal":
            out[period] = 1
        elif isinstance(code, str) and code.startswith("code "):
            out[period] = int(code.removeprefix("code "))
        elif code == NOT_SOLVED:
            if period == "base" and not solve_base:
                out[period] = 1
        else:
            raise ValueError(f"{period}: unknown core status {code!r}")
    return out


def solve_on_core(
    params: Any,
    closure: Any,
    residual_region: str,
    *,
    mode: str = "gtap",
    ref_gdx: Any = None,
    seed_gdx: Any = None,
    base_calibrated: bool = False,
    solve_base: bool = False,
    solve_check: bool = False,
    mute_welfare: bool = True,
    seed_from_prior: bool = False,
    shock: Mapping[str, Mapping[tuple, float]] | None = None,
) -> tuple[Any, dict[str, int], dict[str, Any]]:
    """Solve base -> check -> shock on the core; returns (Pyomo model, old-form
    codes, the core's own result per period)."""
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    ex = Exercise(
        GTAP(
            params,
            closure,
            residual_region=residual_region,
            base_calibrated=base_calibrated,
            ref_gdx=ref_gdx,
            seed_gdx=seed_gdx,
        ),
        gtap_static(
            mode,
            base_calibrated=base_calibrated,
            solve_base=solve_base,
            solve_check=solve_check,
            mute_welfare=mute_welfare,
            seed_from_prior=seed_from_prior,
        ),
        shock=shock,
    )
    res = ex.run()
    model: Any = ex.model
    return model.pyomo, old_codes(res, solve_base=solve_base), res
