"""GTAP's period steps (ADR-0011), written once over ``GTAPPeriodModel``.

Each step mirrors one step of the old Pyomo period recipe (``period_prep``) or one
line of ``solve_multiperiod``, with the same cells and the same writes, so the core
driver leaves the model in the state the old driver left it (gated by the
period-preparation snapshot). They never touch a backend: the same steps fly the
Pyomo model and the family arrays.

A level is any float the model holds, NaN included (an undefined welfare is NaN
and travels as the old driver let it); the GTAP model has no cell without a level
after the build.

Steps that take options are factories: ``seed_from("check")`` returns the step.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping
from typing import Any, cast

from equilibria.core.recipe import Step, StepContext
from equilibria.templates.gtap.period_model import GTAPPeriodModel
from equilibria.templates.gtap.period_rules import (
    F35_DERIVED_DEMAND,
    GAMS_BOUNDED_VARS,
    RECALIBRATE_FROM,
    SHOCK_PNUM,
    WELFARE_LEAVES,
    period_closures,
)

_log = logging.getLogger(__name__)


def _gm(ctx: StepContext) -> GTAPPeriodModel:
    return cast("GTAPPeriodModel", ctx.model)


def _staged(m: GTAPPeriodModel) -> Any:
    staged = m.staged
    if staged is None:
        raise RuntimeError("no closure staged: stage_closure must run first")
    return staged


# ── the shock and the solve's inputs ────────────────────────────────────────


def write_exercise_shock(ctx: StepContext) -> None:
    """Write the exercise's shock (``{instrument: {cell: %}}``) into the shock
    period, before any period is prepared: the shock lives in the model from the
    start (ADR-0003), as when it was written right after the build."""
    if ctx.shock:
        _gm(ctx).write_shock(ctx.shock)


def check_model_flags(base_calibrated: bool) -> Step:
    """Refuse a model built with another ``base_calibrated`` than the recipe's:
    the recipe's copied check and seed periods depend on it."""

    def step(ctx: StepContext) -> None:
        built = _gm(ctx).base_calibrated
        if built != base_calibrated:
            raise ValueError(
                f"the recipe was made with base_calibrated={base_calibrated} but the "
                f"model was built with base_calibrated={built}"
            )

    return step


def check_backend(options: Mapping[str, Any]) -> Step:
    """Refuse, before any work, what the model's backend cannot run. ``options``:
    the recipe's (``gtap_static`` arguments, ``copied_check`` for a check copied
    from the base). The family backend runs what ``solve_gtap_family`` runs; the
    nominal trade balance (``trade_balance``) is family-only, capFix only."""

    def step(ctx: StepContext) -> None:
        backend = getattr(ctx.model, "backend", "pyomo")
        tb = options.get("trade_balance")
        if tb is not None:
            if backend != "family":
                raise NotImplementedError(
                    f"trade_balance={tb!r} runs on the family backend only"
                )
            savf = getattr(ctx.template.closure, "savf_flag", "capFix")
            if savf != "capFix":
                raise NotImplementedError(
                    f"trade_balance={tb!r} replaces the capFix eq_savf row; "
                    f"savf_flag={savf!r} has none to replace"
                )
        if backend != "family":
            return
        refused = (
            ("mode", options.get("mode") != "gtap", f"mode={options.get('mode')!r}"),
            ("solve_base", options.get("solve_base"), "solve_base=True"),
            ("seed_from_prior", options.get("seed_from_prior"), "seed_from_prior=True"),
            ("mute_welfare", not options.get("mute_welfare"), "mute_welfare=False"),
            (
                "copied_check",
                options.get("copied_check"),
                "base_calibrated=True with solve_check=False (the check copied "
                "from the base)",
            ),
            ("shock_steps", options.get("shock_steps"), "shock_steps"),
            (
                "shock_method",
                options.get("shock_method") != "nonlinear",
                f"shock_method={options.get('shock_method')!r}",
            ),
        )
        for _, bad, what in refused:
            if bad:
                raise NotImplementedError(f"family backend: {what} is not supported")

    return step


def stage_closure(which: str, mode: str) -> Step:
    """Stage the closure (``"base"`` or ``"altertax"``, from ``period_closures``
    of the template's closure) and the parameters the period is solved with:
    the template's in gtap mode, its altertax elasticities in altertax mode
    (computed once per run, at the first period)."""
    if which not in ("base", "altertax"):
        raise ValueError(f"closure must be 'base' or 'altertax', got {which!r}")
    gtap_mode = mode == "gtap"

    def step(ctx: StepContext) -> None:
        m = _gm(ctx)
        base, alt = period_closures(ctx.template.closure)
        prior = m.staged
        if prior is not None and prior.gtap_mode == gtap_mode:
            params = prior.params
        elif gtap_mode:
            params = ctx.template.params
        else:
            from equilibria.templates.gtap.altertax import apply_altertax_elasticities

            params = apply_altertax_elasticities(ctx.template.params, in_place=False)
        m.stage(
            base if which == "base" else alt,
            params,
            gtap_mode=gtap_mode,
            period=ctx.period.name,
        )

    return step


# ── preparing a period ──────────────────────────────────────────────────────


def freeze_inactive(ctx: StepContext) -> None:
    """Only this period is free: its cells unfixed and its rows on; every other
    period's cells fixed at their level and its rows off. Instruments are left
    as they are (``freeze_inactive_periods``)."""
    m, period = _gm(ctx), ctx.period.name
    for name in m.var_names():
        for t in m.periods:
            exo = m.exogenous_cells(name, t)
            fixed = m.fixed_cells(name, t)
            if t == period:
                m.unfix_cells(name, t, [k for k in fixed if k not in exo])
                continue
            m.fix_cells(
                name,
                t,
                {
                    k: None
                    for k in m.var_cells(name, t)
                    if k not in exo and k not in fixed
                },
            )
    for name in m.row_names():
        for t in m.periods:
            m.set_rows_active(name, t, m.row_cells(name, t), t == period)


def seed_from_settled(ctx: StepContext) -> None:
    """F3.5: the base starts at the settled point (``settled_seed``), but for the
    derived demands (``F35_DERIVED_DEMAND``) and the exogenous instruments. A
    no-op without ``base_calibrated``."""
    m, period = _gm(ctx), ctx.period.name
    seed = m.settled_seed
    if not (m.base_calibrated and seed):
        return
    for name, cells in seed.items():
        if name in F35_DERIVED_DEMAND or not m.has_var(name):
            continue
        here = set(m.var_cells(name, period))
        exo = m.exogenous_cells(name, period)
        levels = {}
        for body, val in cells.items():
            key = tuple(body) if isinstance(body, tuple) else (body,)
            if key in here and key not in exo and isinstance(val, numbers.Real):
                levels[key] = float(val)
        m.set_values(name, period, levels)


def refresh_after_settled_seed(ctx: StepContext) -> None:
    """F3.5: after the settled seed, recompute the Fisher aggregates from the
    sums they define. A no-op without the seed."""
    m = _gm(ctx)
    if m.base_calibrated and m.settled_seed:
        m.refresh_fisher_aggregates()


def recalibrate_shares(ctx: StepContext) -> None:
    """altertax: rebuild and/ava of this period from the prior one's levels."""
    m, period = _gm(ctx), ctx.period.name
    m.recalibrate_and_ava(period, RECALIBRATE_FROM[period], _staged(m).params)


def seed_from(prior: str, *, only_if: bool = True) -> Step:
    """Warm start: this period's free cells take ``prior``'s levels (instruments
    keep theirs), then the Fisher aggregates are refreshed. ``only_if=False``
    makes it a no-op (the recipe's ``seed_from_prior`` off)."""

    def step(ctx: StepContext) -> None:
        if not only_if:
            return
        m, period = _gm(ctx), ctx.period.name
        for name in m.var_names():
            exo = m.exogenous_cells(name, period)
            fixed = m.fixed_cells(name, period)
            source = m.values(name, prior)
            m.set_values(
                name,
                period,
                {
                    k: source[k]
                    for k in m.var_cells(name, period)
                    if k not in exo and k not in fixed and k in source
                },
            )
        m.refresh_fisher_aggregates()

    return step


def unfix_regy(ctx: StepContext) -> None:
    """regY is endogenous in compStat (GAMS regYeq)."""
    m, period = _gm(ctx), ctx.period.name
    if not m.has_var("regy"):
        return
    fixed = m.fixed_cells("regy", period)
    m.unfix_cells(
        "regy", period, [(r,) for r in _staged(m).params.sets.r if (r,) in fixed]
    )


def fix_shock_numeraire(ctx: StepContext) -> None:
    """Fix this period's pnum at the shock anchor (GAMS fixes the numeraire of
    the period it solves)."""
    m, period = _gm(ctx), ctx.period.name
    if not m.has_var("pnum") or () not in m.var_cells("pnum", period):
        return
    if () not in m.fixed_cells("pnum", period):
        m.fix_cells("pnum", period, {(): SHOCK_PNUM})


def deactivate_redundant_xft(ctx: StepContext) -> None:
    """altertax: switch off eq_xft[r,f] where eq_xfteq[r,f] is on (both would
    pin the same xft)."""
    m, period = _gm(ctx), ctx.period.name
    if not (m.has_row("eq_xft") and m.has_row("eq_xfteq")):
        return
    live = m.active_rows("eq_xft", period)
    off = [k for k in m.active_rows("eq_xfteq", period) if k in live]
    m.set_rows_active("eq_xft", period, off, False)
    if off:
        _log.info(
            "%s period: deactivated eq_xft for %d (r,f) pairs "
            "(eq_xfteq active → eq_xft redundant, multi-period index fix)",
            period,
            len(off),
        )


def collapse_pft(ctx: StepContext) -> None:
    """gtap: square the factor-price block (``_collapse_pft_pfteq``). A real
    factor (eq_xfteq on) drops its eq_pfteq, a GAMS free row. A dangling one
    fixes xft at its level and pft at 1, holdfixes every xf with a flow and
    drops its eq_pfeq."""
    m, period = _gm(ctx), ctx.period.name
    if not m.has_var("pft"):
        return
    real = m.active_rows("eq_xfteq", period) if m.has_row("eq_xfteq") else frozenset()
    pfteq_on = (
        m.active_rows("eq_pfteq", period) if m.has_row("eq_pfteq") else frozenset()
    )
    pfeq_on = m.active_rows("eq_pfeq", period) if m.has_row("eq_pfeq") else frozenset()
    xft = m.has_var("xft")
    xft_cells = set(m.var_cells("xft", period)) if xft else set()
    xft_fixed = m.fixed_cells("xft", period) if xft else frozenset()
    xft_lvl = m.values("xft", period) if xft else {}
    pft_fixed = m.fixed_cells("pft", period)
    xf = m.has_var("xf")
    xf_lvl = m.values("xf", period) if xf else {}
    xf_fixed = m.fixed_cells("xf", period) if xf else frozenset()
    by_rf: dict[tuple, list[tuple]] = {}
    for k in xf_lvl:
        by_rf.setdefault(k[:2], []).append(k)

    pfteq_off, pfeq_off = [], []
    fix_xft: dict[tuple, float | None] = {}
    fix_pft: dict[tuple, float | None] = {}
    fix_xf: dict[tuple, float | None] = {}
    for rf in m.var_cells("pft", period):
        if rf in real:
            if rf in pfteq_on:
                pfteq_off.append(rf)
            continue
        if rf in xft_cells and rf not in xft_fixed:
            fix_xft[rf] = xft_lvl[rf]
        if rf not in pft_fixed:
            fix_pft[rf] = 1.0
        for k in by_rf.get(rf, ()):
            if abs(xf_lvl[k]) < 1e-12:
                continue
            if k not in xf_fixed:
                fix_xf[k] = xf_lvl[k]
            if k in pfeq_on:
                pfeq_off.append(k)
    if pfteq_off:
        m.set_rows_active("eq_pfteq", period, pfteq_off, False)
    if fix_xft:
        m.fix_cells("xft", period, fix_xft)
    m.fix_cells("pft", period, fix_pft)
    if fix_xf:
        m.fix_cells("xf", period, fix_xf)
    if pfeq_off:
        m.set_rows_active("eq_pfeq", period, pfeq_off, False)
    n = len(pfteq_off) + len(fix_xft) + len(fix_pft) + len(fix_xf) + len(pfeq_off)
    if n:
        _log.info(
            "%s period: collapsed pft/eq_pfteq for %d (r,f) pairs "
            "(gtap-mode factor-price squaring)",
            period,
            n,
        )


def replicate_reference(*, fresh: bool) -> Step:
    """Fix what the single-period model with the staged closure fixes (cells
    already fixed keep their level) and copy its bounds. ``fresh=True`` builds
    the reference with this period's parameters instead of reusing the base's."""

    def step(ctx: StepContext) -> None:
        m, period = _gm(ctx), ctx.period.name
        staged = _staged(m)
        ref = m.sp_reference(staged.closure, staged.params, period, fresh=fresh)
        for name, cells in ref.fixed.items():
            if not m.has_var(name):
                continue
            here = set(m.var_cells(name, period))
            fixed = m.fixed_cells(name, period)
            m.fix_cells(
                name,
                period,
                {k: v for k, v in cells.items() if k in here and k not in fixed},
            )
        for name, cells in ref.bounds.items():
            if not m.has_var(name):
                continue
            here = set(m.var_cells(name, period))
            m.set_bounds(name, period, {k: b for k, b in cells.items() if k in here})

    return step


def gams_bounds(ctx: StepContext) -> None:
    """Free (real domain, no bounds) every cell of this period GAMS does not
    bound; instruments untouched."""
    m, period = _gm(ctx), ctx.period.name
    for name in m.var_names():
        if name in GAMS_BOUNDED_VARS:
            continue
        exo = m.exogenous_cells(name, period)
        m.free_cells(
            name, period, [k for k in m.var_cells(name, period) if k not in exo]
        )


def mute_welfare(*, gtap_mode: bool, enabled: bool = True) -> Step:
    """Fix the inert welfare leaves at their level and switch off their rows
    (square-preserving). In gtap mode ``walras`` stays free with its row on."""

    def step(ctx: StepContext) -> None:
        if not enabled:
            return
        m, period = _gm(ctx), ctx.period.name
        regions = list(_staged(m).params.sets.r)
        n = 0
        for eqn, vn in WELFARE_LEAVES:
            if not (m.has_row(eqn) and m.has_var(vn)):
                continue
            keep_row = gtap_mode and vn == "walras"
            cells = set(m.var_cells(vn, period))
            for r in regions:
                for cand in ((r,), ()):
                    if cand not in cells:
                        continue
                    if keep_row:
                        m.unfix_cells(vn, period, [cand])
                        break
                    if cand not in m.fixed_cells(vn, period):
                        m.fix_cells(vn, period, {cand: None})
                    if cand in m.active_rows(eqn, period):
                        m.set_rows_active(eqn, period, [cand], False)
                        n += 1
                    break
        if n:
            _log.info(
                "%s period: muted %d welfare-leaf rows (cv/ev/walras/u/ug/us)",
                period,
                n,
            )

    return step


def fnm_pf(ctx: StepContext) -> None:
    """Holdfix pf of sector-specific factors with etaff=0 and no factor flow
    (GAMS ``pf.fx(r,fp,a,tsim-1)``)."""
    m, period = _gm(ctx), ctx.period.name
    p = _staged(m).params
    fset = getattr(p, "sets", None)
    if fset is None or not m.has_var("pf"):
        return
    mf = {str(f) for f in getattr(fset, "mf", [])}
    sf = {str(f) for f in getattr(fset, "sf", [])}
    etaff = getattr(getattr(p, "elasticities", None), "etaff", {}) or {}
    bm = getattr(p, "benchmark", None)

    def has_flow(r: str, f: str, a: str) -> bool:
        if bm is None:
            return True
        v = bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0))
        return abs(float(v or 0.0)) > 1e-12

    fixed = m.fixed_cells("pf", period)
    hold = {}
    for key in m.var_cells("pf", period):
        if len(key) != 3:
            continue
        r, f, a = key
        if f in mf or f in sf:
            continue
        if float(etaff.get(key, 0.0) or 0.0) != 0.0 or has_flow(r, f, a):
            continue
        if key not in fixed:
            hold[key] = None
    m.fix_cells("pf", period, hold)
    if hold:
        _log.info("%s period: holdfixed %d fnm pf cells (etaff=0)", period, len(hold))


def copy_from_base(ctx: StepContext) -> None:
    """F3.5 without a check solve: the check is the settled base, copied cell by
    cell (instruments keep theirs)."""
    m, period = _gm(ctx), ctx.period.name
    for name in m.var_names():
        exo = m.exogenous_cells(name, period)
        base = m.values(name, "base")
        m.set_values(
            name,
            period,
            {
                k: base[k]
                for k in m.var_cells(name, period)
                if k not in exo and k in base
            },
        )


def hold_nominal_savf(ctx: StepContext) -> None:
    """GEMPACK ``del_tbal = 0``: in this period, ``savf`` of every region but the
    residual one stays at its seeded level (the check's) and its ``eq_savf`` row
    (``savf = pigbl * savf_bar``, a real balance) is off; one free variable and one
    row leave together. Under ``trade_balance="ratio"`` the solver then moves the
    held levels with world income."""
    m, period = _gm(ctx), ctx.period.name
    rows = set(m.row_cells("eq_savf", period))
    held = [
        (str(r),) for r in ctx.template.params.sets.r if str(r) != m.residual_region
    ]
    m.fix_cells("savf", period, dict.fromkeys(held))
    m.set_rows_active("eq_savf", period, [c for c in held if c in rows], False)


def write_default_tariff(*, gtap_mode: bool) -> Step:
    """With no shock in the model, write the +10 % default import tariff."""

    def step(ctx: StepContext) -> None:
        m = _gm(ctx)
        labels = m.shock_labels()
        if not labels:
            m.write_default_tariff(gtap_mode=gtap_mode)
            labels = m.shock_labels()
        _log.info("shock period: %d ShockBlock cells: %s", len(labels), labels[:5])

    return step


def debug_before_solve(ctx: StepContext) -> None:
    _gm(ctx).debug_before_solve(ctx.period.name)


# ── after the solve ─────────────────────────────────────────────────────────


def solve_welfare_tail(ctx: StepContext) -> None:
    """Solve each muted welfare row for its own variable, so the reported
    ev/cv/u/ug/us are the period's and not the seed."""
    _gm(ctx).solve_welfare_tail(ctx.period.name)


def freeze_period(ctx: StepContext) -> None:
    """Fix every cell of this period at its level (GAMS holdfixed)."""
    m, period = _gm(ctx), ctx.period.name
    for name in m.var_names():
        m.fix_cells(name, period, m.values(name, period))


def recompute_ifsub_report_vars(ctx: StepContext) -> None:
    """ifSUB: the report variables are recomputed from the solved levels (as
    GAMS postsim). A no-op without ifSUB."""
    m, period = _gm(ctx), ctx.period.name
    staged = _staged(m)
    if not getattr(staged.closure, "if_sub", False):
        return
    n = m.recompute_ifsub_report_vars(period, staged.params)
    if n:
        _log.info(
            "%s period: recomputed %d ifSUB report-var cells "
            "(pfa/pfy/pp/pwmg/pefob/pmcif/pm)",
            period,
            n,
        )


def check_endowment(ctx: StepContext) -> None:
    """After a converged solve: an endowment shock must reach the solution."""
    if ctx.status == "optimal":
        _gm(ctx).check_endowment_shock_entered()


def after_shock_point(model: Any) -> None:
    """The recomputations after each shock solve, for a strategy that solves
    the shock more than once (Johansen): ifSUB report vars and welfare tail."""
    m = cast("GTAPPeriodModel", model)
    staged = _staged(m)
    if getattr(staged.closure, "if_sub", False):
        m.recompute_ifsub_report_vars("shock", staged.params)
    m.solve_welfare_tail("shock")


__all__ = [
    "after_shock_point",
    "check_endowment",
    "check_model_flags",
    "collapse_pft",
    "copy_from_base",
    "deactivate_redundant_xft",
    "debug_before_solve",
    "fix_shock_numeraire",
    "fnm_pf",
    "freeze_inactive",
    "freeze_period",
    "gams_bounds",
    "mute_welfare",
    "recalibrate_shares",
    "recompute_ifsub_report_vars",
    "refresh_after_settled_seed",
    "replicate_reference",
    "seed_from",
    "seed_from_settled",
    "solve_welfare_tail",
    "stage_closure",
    "unfix_regy",
    "write_default_tariff",
    "write_exercise_shock",
]
