"""GTAP's static recipe: base -> check -> shock on the core driver (ADR-0011).

Each period's steps are those of the old Pyomo period recipe (``period_prep``),
in the same order, followed by what ``solve_multiperiod`` did around each solve.
``mode`` picks the gtap or the altertax variant of the check and the shock; the
options of ``solve_multiperiod`` are arguments here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from equilibria.core.recipe import KindPlan, Period, Recipe, Solver, solve_once
from equilibria.templates.gtap import steps as s
from equilibria.templates.gtap.strategies import in_steps, johansen


def _validate(mode: str, shock_method: str, shock_steps: Sequence[float] | None):
    if shock_method not in ("nonlinear", "johansen"):
        raise ValueError(
            f"shock_method must be 'nonlinear' or 'johansen', got {shock_method!r}"
        )
    if shock_method == "johansen" and shock_steps:
        raise ValueError(
            "shock_method='johansen' is one linear step from check: it takes no "
            "shock_steps"
        )
    if mode not in ("altertax", "gtap"):
        raise ValueError(f"mode must be 'altertax' or 'gtap', got {mode!r}")


def _by_backend(model: Any) -> str:
    """The default solver: PATH for the Pyomo model, Newton-TR for the family one."""
    if getattr(model, "backend", "pyomo") == "family":
        from equilibria.solver import newton_gtap

        return newton_gtap.newton_tr()(model)
    from equilibria.solver.path_gtap import path

    return path()(model)


def gtap_static(
    mode: str = "gtap",
    *,
    base_calibrated: bool = False,
    solve_base: bool = False,
    solve_check: bool = False,
    settle_only: bool = False,
    mute_welfare: bool = True,
    seed_from_prior: bool = False,
    shock_steps: Sequence[float] | None = None,
    shock_method: str = "nonlinear",
    solver: Solver | None = None,
) -> Recipe:
    """The recipe for a GTAP comparative-statics exercise.

    ``base_calibrated`` must match the model's build (F3.5): the check is then a
    copy of the settled base, unless ``solve_check``. ``solve_base`` solves the
    base (default: the calibrated point is the base, as GAMS). ``settle_only``
    stops after the check. ``shock_steps`` solves the shock in steps;
    ``shock_method="johansen"`` takes one linear step from check instead.
    ``solver`` defaults to the model's: PATH on Pyomo, Newton-TR on family.
    """
    _validate(mode, shock_method, shock_steps)
    if solver is None:
        solver = _by_backend
    gtap = mode == "gtap"
    closure = "base" if gtap else "altertax"
    copied_check = base_calibrated and not solve_check
    debug = (s.debug_before_solve,)
    after = (s.solve_welfare_tail, s.freeze_period)
    mute = s.mute_welfare(gtap_mode=gtap, enabled=mute_welfare)

    base = KindPlan(
        before=(
            s.write_exercise_shock,
            s.check_model_flags(base_calibrated),
            s.stage_closure("base", mode),
            s.seed_from_settled,
            s.freeze_inactive,
            s.replicate_reference(fresh=False),
            s.refresh_after_settled_seed,
        )
        + (debug if solve_base else ()),
        after=after,
        solver=solver if solve_base else None,
    )

    if copied_check:
        check = KindPlan(before=(s.copy_from_base,), after=after)
    else:
        if gtap:
            middle = (
                s.seed_from("base", only_if=seed_from_prior),
                s.unfix_regy,
                s.collapse_pft,
                s.replicate_reference(fresh=False),
                s.gams_bounds,
            )
        else:
            middle = (
                s.recalibrate_shares,
                s.seed_from("base", only_if=seed_from_prior),
                s.unfix_regy,
                s.deactivate_redundant_xft,
                s.replicate_reference(fresh=False),
            )
        check = KindPlan(
            before=(s.stage_closure(closure, mode), s.freeze_inactive)
            + middle
            + (mute, s.fnm_pf)
            + debug,
            after=after,
            solver=solver,
        )

    plans = {"calibrate": base, "replicate": check}
    periods = [Period("base", "calibrate"), Period("check", "replicate")]
    if not settle_only:
        prior = "base" if copied_check else "check"
        if gtap:
            middle = (
                s.seed_from(prior),
                s.unfix_regy,
                s.fix_shock_numeraire,
                s.collapse_pft,
                s.replicate_reference(fresh=True),
                s.gams_bounds,
            )
        else:
            middle = (
                s.recalibrate_shares,
                s.seed_from(prior),
                s.unfix_regy,
                s.fix_shock_numeraire,
                s.deactivate_redundant_xft,
                s.replicate_reference(fresh=True),
            )
        if shock_method == "johansen":
            strategy = johansen(after=s.after_shock_point)
            shock_after = (s.check_endowment, s.freeze_period)
        else:
            strategy = in_steps(shock_steps) if shock_steps else solve_once
            shock_after = (
                s.recompute_ifsub_report_vars,
                s.solve_welfare_tail,
                s.check_endowment,
                s.freeze_period,
            )
        plans["shock"] = KindPlan(
            before=(s.stage_closure(closure, mode), s.freeze_inactive)
            + middle
            + (mute, s.fnm_pf, s.write_default_tariff(gtap_mode=gtap))
            + debug,
            after=shock_after,
            solver=solver,
            strategy=strategy,
        )
        periods.append(Period("shock", "shock"))
    return Recipe(tuple(periods), plans)
