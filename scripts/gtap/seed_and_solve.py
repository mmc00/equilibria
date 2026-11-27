"""Cascade tool 11 — SEED-AND-SOLVE: the equilibrium-selection vs differing-equation
discriminator. THE tool that should run FIRST when the cell-by-cell gate misses target.

WHAT IT DOES: seed Python at the EXACT GAMS reference point, run the real PATH solve, and
report whether the point STAYS (residual ~0 after solve, vars hold) or GOES (solve drifts/
fails). The binary answer settles the root question that cost this project two false closures:

  - STAYS  → the GAMS point IS a fixed point of equilibria → the gap is EQUILIBRIUM SELECTION
             (two valid solutions, the solve picks the other). No equation differs. Stop
             chasing "the culprit equation" — it does not exist.
  - GOES   → the GAMS point is NOT a fixed point → an equation DIFFERS from GAMS. The drift
             direction + the residual TAIL name the candidate equation to chase next.

CRITICAL — read the residual TAIL, not the median. With ~1100 equations the median is ~1e-13
(looks like a clean fixed point) while a handful of >1e-4 equations carry the entire signal.
Reading the median instead of the tail is exactly what produced the two false "fixed-point to
1e-10, equilibrium selection" closures in this project. This tool ranks and reports the worst
N equation residuals AT the seed; that tail is the lead.

SETUP DISCIPLINES (each one previously manufactured a false "differing equation" lead):
  1. CASCADE the derived-var seed. warmstart_from_gams does NOT seed Python-only derived
     aggregates (xd/xmt/xc/xg/xi/xiagg/xigbl/pigbl/kapEnd) consistently with their components,
     so eq_xd_agg/eq_xc/eq_xaa/eq_xigbl show spurious residuals 1–2 that CASCADE to ~0 once
     the derived vars are set from their own identities. A genuinely-differing equation does
     NOT fall on consistent seed; an artifact does. This tool cascades the derived seed first.
  2. BUILD THE CASE ON THE CORE DRIVER (``_case``): both modes run the full
     multi-period model, the shock tariff written into the ShockBlock. altertax is
     built as the MCP gate builds it: the altertax closure, its elasticities and the
     FBEP sign of the fbep=header reference; with the base closure the shock's factor
     rows do not build and the solve never runs (MAR-216). gtap keeps its old build
     (default FBEP sign), so against the fbep=header reference it still shows an
     eq_ytax ft/fs seed residual of ~0.07 (MAR-219).
  3. SEED THE CDE HOUSEHOLD HELPERS (uh, zcons, xcshr): GAMS keeps them on a
     household index Python drops, and an unseeded uh=1.0 leaves a ~1e-3 residual
     on eq_xc/eq_uh at the shocked GAMS point.

A solve that never ran (code None) is a tool error (error_kind="solve_not_run"),
never "an equation differs".

DECISIVE PRECEDENT: gtap7_3x3 altertax shock. Seeded the exact GAMS point and solved → it
DRIFTED (code=2, pft[EU_28,NatRes] 1.004→1.05). The residual tail's only real non-ROW/non-leaf
entry was eq_xi (0.006–0.010). Traced to an esubi key-shape bug (sigmai=0 Leontief vs GAMS 1.01
CES; commit 0e2db11). The seed-and-solve test named the differing equation in ONE run — after
eight other leads. See project_gtap7_3x3_gap_is_factor_bias.

Usage:
    uv run python scripts/gtap/seed_and_solve.py --dataset gtap7_3x3 \\
        --gdx data/gtap7/gtap7_3x3/gams_mcp_altertax_ifsub0_fbep-header/out.gdx \\
        --period shock --top 20   # --mode altertax (default) or gtap

JSON contract (shared _parity_json schema): status=clean (STAYS = equilibrium selection,
nothing to fix) | dirty (GOES = a differing equation, see violations = residual tail) |
error (solve crash). exit 0/1/2.
"""

from __future__ import annotations
from equilibria.sources import dataset_dir
import argparse
import statistics
import sys
from pathlib import Path
from equilibria._local_refs import path_capi_src

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "tests" / "templates" / "gtap"))
_PATH_CAPI = path_capi_src()
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

from _parity_json import make_violation, make_detection, run_tool  # noqa: E402

# Residual above which an equation counts as "not satisfied at the GAMS point".
RESID_TOL = 1e-4
# The FBEP sign of the references the cascade reads (fbep=header, #191), the basis
# the MCP gate builds both modes with.
ALTERTAX_BASIS = "gempack"
# Families that are KNOWN-benign even when they carry residual at the GAMS point:
# rgdpmp/pgdpmp are report leaves (no feedback); ROW-region rows ride a corrupt reference
# (project_gtap7_3x3_ref_corrupt_ROW). The tool flags these separately so the real lead
# (non-leaf, non-ROW) is not buried.
#
# WELFARE LEAVES: eq_cv/eq_ev/eq_walras/eq_u/eq_ug/eq_us are inert report rows that the
# driver mutes (steps.mute_welfare) — no economic feedback into the equilibrium, so a
# residual there at the GAMS point is benign and must not bury the real lead. (eq_uh is
# NOT here: uh feeds eq_zcons = real CDE demand, so it has genuine blast radius.)
_LEAF_EQS = {
    "eq_rgdpmp",
    "eq_pgdpmp",
    "eq_cv",
    "eq_ev",
    "eq_walras",
    "eq_u",
    "eq_ug",
    "eq_us",
}
# Rows where Python follows GTAP v7 / GEMPACK, not GAMS, by decision: the GAMS point
# cannot satisfy them exactly, so a residual there is reported but is not a lead.
# eq_chisave: GTAP v7's E_psave (full adjustment, Tornqvist weights), not GAMS's
#   chiSaveeq, which divides by psave and so applies the square root (6cf5909e).
# eq_pwmg: Python's amgm shares sum to 1 (VTWR over its own total); GAMS's amgm is
#   VTWR / (VCIF - VFOB), whose sum carries the data's rounding (10x7 1.5e-4, 15x10
#   6.5e-4). pwmg is one of the 14 variables the MCP gate does not compare (RF).
_NOT_GAMS_EQS = {"eq_chisave", "eq_pwmg"}


def _cascade_derived_seed_mp(m, V, setv, period: str) -> int:
    """Period-aware DISCIPLINE-1 cascade for the MULTI-PERIOD gtap model. Sets the
    Python-only derived aggregates (xd/xmt/xc/xg/xi/xiagg/kapEnd/xigbl/pigbl) for the
    given period slice from their own identities, so the residual TAIL reflects the
    EQUATION, not an un-seeded init. Without this the MP gtap tail is dominated by
    spurious eq_xc/eq_xda/eq_xaa/eq_xigbl residuals of ~1-2 (seed-incomplete artifact,
    NOT a differing equation) — exactly the trap DISCIPLINE 1 in the module docstring
    warns about. Mirrors _cascade_derived_seed but every var carries the (...,period)
    index. Returns cells set."""
    t = period
    n = 0
    # The MP model has no `xscale` component (baked as a literal in each equation body).
    # The builder stashes the (r,a) floats on m._xscale_floats; non-activity agents
    # (hhd/gov/inv/tmg) have xscale=1. Using this recovers the SAME scaling the
    # eq_xd_agg/eq_xmt_agg bodies use — without it `xd`/`xmt` stay at init (phantom
    # eq_xd_agg residual ~1.45).
    _xsf = getattr(m, "_xscale_floats", {}) or {}

    def _xs(r, aa):
        return _xsf.get((r, aa), 1.0)

    for r in m.r:
        for i in m.i:
            try:
                n += setv(
                    m.xd,
                    (r, i, t),
                    sum(V(m.xda[r, i, aa, t]) / _xs(r, aa) for aa in m.aa),
                )
            except Exception:
                pass
            try:
                n += setv(
                    m.xmt,
                    (r, i, t),
                    sum(V(m.xma[r, i, aa, t]) / _xs(r, aa) for aa in m.aa),
                )
            except Exception:
                pass
            # DERIVE xc/xg/xi from the QUANTITY identity xc=xaa[hhd], xg=xaa[gov],
            # xi=xaa[inv] (eq_xaa_hhd/gov/inv) — NOT the price formula g_share·yg/pa.
            # xaa is seeded directly from GAMS, so the identity reproduces the GAMS
            # point; the price formula diverges from it (yg/pa not perfectly consistent
            # with the seeded xaa at the GAMS point) → xg 3.52 vs xaa[gov] 3.02 →
            # phantom ±0.50 residual split across eq_xg/eq_xaa_gov. Mirrors
            # complete_derived_seed. (xi already used the identity.)
            try:
                n += setv(m.xc, (r, i, t), V(m.xaa[r, i, "hhd", t]))
            except Exception:
                pass
            try:
                n += setv(m.xg, (r, i, t), V(m.xaa[r, i, "gov", t]))
            except Exception:
                pass
            try:
                n += setv(m.xi, (r, i, t), V(m.xaa[r, i, "inv", t]))
            except Exception:
                pass
    for r in m.r:
        try:
            n += setv(m.xiagg, (r, t), V(m.yi[r, t]) / max(V(m.pi[r, t]), 1e-9))
        except Exception:
            pass
        try:
            n += setv(
                m.kapEnd,
                (r, t),
                (1 - V(m.depr[r])) * V(m.kstock[r, t]) + V(m.xiagg[r, t]),
            )
        except Exception:
            pass
    try:
        n += setv(
            m.xigbl,
            (t,),
            sum(V(m.xiagg[r, t]) - V(m.depr[r]) * V(m.kstock[r, t]) for r in m.r),
        )
    except Exception:
        pass
    try:
        num = sum(
            V(m.pi[r, t]) * (V(m.xiagg[r, t]) - V(m.depr[r]) * V(m.kstock[r, t]))
            for r in m.r
        )
        n += setv(m.pigbl, (t,), num / max(V(m.xigbl[t]), 1e-9))
    except Exception:
        pass
    return n


# GAMS keeps the CDE household helpers on (r, [i,] h, t) with h = hhd; Python drops
# h. seed_all_periods misses them, so uh stays 1.0 and eq_xc/eq_uh carry a seed-only
# residual of ~1e-3 at the shocked GAMS point.
_HOUSEHOLD_VARS = ("uh", "zcons", "xcshr")


def _seed_household_vars(m, gdx_path: Path) -> int:
    """Seed the CDE household helpers from GAMS, dropping its household index."""
    from _diff_core import gams_levels  # type: ignore

    def _strip(x):
        return (
            x[2:]
            if isinstance(x, str) and len(x) > 2 and x[1] == "_" and x[0] in "acfr"
            else x
        )

    n = 0
    for vn in _HOUSEHOLD_VARS:
        comp = getattr(m, vn, None)
        if comp is None:
            continue
        for k, val in gams_levels(gdx_path, vn).items():
            key = tuple(_strip(x) for x in k if x != "hhd")
            if key in comp:
                comp[key].set_value(val)
                n += 1
    return n


def muted_welfare_rows(mode: str) -> frozenset[str]:
    """The welfare-leaf rows the driver switches off before every solve
    (steps.mute_welfare): cv/ev/u/ug/us, and walras outside the gtap mode. They are
    not in the system the solver is handed, and u/us/cv/ev move with psave, which
    follows GTAP v7, not GAMS (see _NOT_GAMS_EQS), so a layer that reads residuals
    at the GAMS point skips them."""
    from equilibria.templates.gtap.period_rules import WELFARE_LEAVES

    return frozenset(
        eqn for eqn, vn in WELFARE_LEAVES if not (mode == "gtap" and vn == "walras")
    )


def _case(dataset: str, if_sub: bool, mode: str):
    """(params, closure, residual region) of one case, as the MCP gate builds it
    (``test_gtap7_mcp_parity._solve_and_measure``), in both modes with the data's
    FBEP sign (the fbep=header reference has it, #191). gtap: the base closure.
    altertax: the altertax elasticities and the altertax closure; with the base
    closure the shock's factor rows do not build and the solve never runs."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    p.set_va_subsidy_basis(ALTERTAX_BASIS)
    if mode == "gtap":
        gc = GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            fix_endowments=False,
            fix_taxes=False,
            fix_technology=False,
            if_sub=if_sub,
            numeraire="pnum",
            va_subsidy_basis=ALTERTAX_BASIS,
        )
        return p, gc, rr
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities

    pa = apply_altertax_elasticities(p, in_place=False)
    gc = GTAPClosureConfig(
        name="altertax",
        closure_type="MCP",
        capital_mobility="mobile",
        fix_endowments=False,
        fix_taxes=True,
        fix_technology=True,
        if_sub=if_sub,
        numeraire="pnum",
        va_subsidy_basis=ALTERTAX_BASIS,
    )
    return pa, gc, rr


def period_levels(m, period: str) -> dict[tuple[str, tuple], float]:
    """``{(var, cell without the period): level}`` over ``period``'s cells."""
    import pyomo.environ as pyo

    out = {}
    for v in m.component_objects(pyo.Var, active=True):
        for idx in v:
            if isinstance(idx, tuple):
                if not idx or idx[-1] != period:
                    continue
                cell = idx[:-1]
            elif idx == period:
                cell = ()
            else:
                continue
            if v[idx].value is not None:
                out[(v.name, cell)] = float(v[idx].value)
    return out


def seed_levels(
    m0, periods=("check", "shock")
) -> dict[str, dict[tuple[str, tuple], float]]:
    """``{period: {(var, cell): level}}`` at the GAMS point of ``m0``, a build_seeded
    model, plus each period's derived aggregates: the levels drift_test compares
    against. ``m0`` is left as it came, so the residual tail read on it afterwards
    sees only its own period's cascade."""
    from pyomo.environ import value as V

    saved = []

    def _setv(comp, key, val):
        try:
            old = comp[key].value
            comp[key].set_value(val)
        except Exception:
            return 0
        saved.append((comp[key], old))
        return 1

    out = {}
    for period in periods:
        _cascade_derived_seed_mp(m0, V, _setv, period)
        out[period] = period_levels(m0, period)
    for cell, old in reversed(saved):
        cell.set_value(old, skip_validation=True)
    return out


def _put_seed(model, period: str, seed) -> None:
    """``period``'s free cells take their ``seed`` level; fixed and exogenous cells
    keep theirs."""
    for name in model.var_names():
        held = model.fixed_cells(name, period) | model.exogenous_cells(name, period)
        model.set_values(
            name,
            period,
            {
                k: seed[(name, k)]
                for k in model.var_cells(name, period)
                if k not in held and (name, k) in seed
            },
        )


def residual_tail(m, period: str) -> list[tuple[float, str, tuple]]:
    """``[(residual, row family, cell without the period)]`` over ``period``'s active
    rows, largest first. The rows recalibrate_shares builds are scalars named
    ``_eq_<nd|va>_recal_<r>_<a>_<t>``: they count for the period their name ends with."""
    from pyomo.environ import Constraint
    from pyomo.environ import value as V

    out = []
    for c in m.component_objects(Constraint, active=True):
        for idx in c:
            cd = c[idx]
            if not cd.active:
                continue
            if isinstance(idx, tuple):
                if not idx or idx[-1] != period:
                    continue
                body_idx = idx[:-1] if len(idx) > 1 else ()
            elif idx is None and "_recal_" in c.name:
                if not c.name.endswith(f"_{period}"):
                    continue
                body_idx = ()
            else:
                if idx != period:
                    continue
                body_idx = ()
            try:
                b = V(cd.body)
                lo = V(cd.lower) if cd.lower is not None else None
                up = V(cd.upper) if cd.upper is not None else None
                rr_ = 0.0
                if lo is not None:
                    rr_ = max(rr_, abs(b - lo))
                if up is not None:
                    rr_ = max(rr_, abs(b - up))
                out.append((rr_, c.name, body_idx))
            except Exception:
                pass
    out.sort(reverse=True)
    return out


def start_at_seed(seeds: dict[str, dict[tuple[str, tuple], float]], tails=None):
    """The default solver, but the solved period's free cells (not fixed, not
    exogenous) first take their ``seeds`` level. The shock's plan seeds them from
    Python's solved check (``seed_from(prior)``), so without this the shock would
    start there and not at the GAMS point (MAR-252).

    With a ``tails`` dict, right after the solve (before the driver freezes the
    period) it records ``tails[period] = residual_tail`` at the seed on the system
    the solver was handed: the rows recalibrate_shares rebuilt are in, the rows
    PATH switched off and the cells it fixed are as PATH left them. The solved
    levels are put back afterwards."""
    from equilibria.templates.gtap import recipe

    def solver(model):
        rows = model.row_names()
        active = [
            t for t in model.periods if any(model.active_rows(r, t) for r in rows)
        ]
        period = active[0] if len(active) == 1 and active[0] in seeds else None
        if period is not None:
            _put_seed(model, period, seeds[period])
        status = recipe._by_backend(model)
        if period is not None and tails is not None:
            solved = {n: model.values(n, period) for n in model.var_names()}
            _put_seed(model, period, seeds[period])
            tails[period] = residual_tail(model.pyomo, period)
            for name, levels in solved.items():
                model.set_values(name, period, levels)
        return status

    return solver


def _solve_gtap(
    dataset: str,
    gdx_path: Path,
    if_sub: bool,
    mode: str = "gtap",
    seeded=None,
    tails=None,
):
    """The full multi-period model seeded from ``gdx_path`` in every period, solved
    base (skipped) -> check -> shock on the core driver, each solve starting at the
    GAMS point (``start_at_seed``) read from ``seeded``, a build_seeded model of the
    case (built here when None); ``tails`` as in ``start_at_seed``. Returns
    (model, ``{period: {"code": old-form int, "residual": float}}``)."""
    from _core_run import solve_on_core

    if seeded is None:
        _p, _gc, seeded = build_seeded(dataset, gdx_path, "shock", mode, if_sub)
    p, gc, rr = _case(dataset, if_sub, mode)
    m, codes, core = solve_on_core(
        p,
        gc,
        rr,
        mode=mode,
        ref_gdx=gdx_path,
        seed_gdx=gdx_path,
        solver=start_at_seed(seed_levels(seeded), tails),
    )
    res = {k: {"code": codes[k], "residual": core[k]["residual"]} for k in codes}
    return m, res


def build_seeded(
    dataset: str, gdx_path: Path | None, period: str, mode: str, if_sub: bool
):
    """(params, closure, model): the full multi-period Pyomo model of the case as the
    MCP gate builds it (``_case``), not solved, seeded from the GAMS GDX in every
    period (``gdx_path=None``: left at Python's own calibration; the seed's
    ``{period: (exported, placed)}`` cells are ``m._seed_counts``); for the shock, with
    the default tariff the driver writes, so the rows read are those of the actual
    shock system."""
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS

    p, gc, rr = _case(dataset, if_sub, mode)
    mp = GTAPBlockMultiPeriodModel(p.sets, p, gc, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    if gdx_path is not None:
        m._seed_counts = mp.seed_all_periods(m, gdx_path)
        _seed_household_vars(m, gdx_path)
    m._gtap_mode = mode == "gtap"
    if period == "shock":
        from equilibria.templates.gtap.shocks import apply_default_tariff

        apply_default_tariff(m, gtap_mode=mode == "gtap")
    return p, gc, m


def run_seed_and_solve_gtap(
    dataset: str,
    gdx_path: Path,
    period: str,
    top: int,
    if_sub: bool,
    mode: str = "gtap",
):
    """Seed-and-solve on the core driver, for both modes (gtap real-CES, altertax CD).

    The shock is MULTI-PERIOD: the tariff is the 'shock' cell of imptx in the
    ShockBlock (shocks.apply_default_tariff). So we build the FULL multi-period
    model as the MCP gate does (``_case``), seed it from the GAMS GDX in every
    period, solve on the core driver (gtap_static) from that point, and read the
    residual TAIL of the requested period at the seed on the system the solver was
    handed (``start_at_seed``), not on the post-solve state.
    """
    from pyomo.environ import value as V

    def _build_seeded():
        return build_seeded(dataset, gdx_path, period, mode, if_sub)

    # 1) One hand build holds the SEED: the sentinel before the solve and the start
    #    levels of every solve. The SOLVE runs on the core driver.
    _p2, _gc2, m2 = _build_seeded()
    m = m2

    def _sentinel():
        # sentinel: the first finite price var FOR THE REQUESTED PERIOD.
        for nm in ("pft", "pf", "px", "pa"):
            comp = getattr(m, nm, None)
            if comp is None:
                continue
            for idx in comp:
                if not (isinstance(idx, tuple) and idx and idx[-1] == period):
                    continue
                try:
                    val = float(V(comp[idx]))
                    if val > 1e-6:
                        return f"{nm}{tuple(str(x) for x in idx)}", val
                except Exception:
                    pass
        return "(none)", None

    sent_name, sent_before = _sentinel()

    # DISCIPLINE 1: the seed carries the Python-only derived aggregates of the period
    # (xd/xmt/xc/xg/xi/...), else eq_xc/eq_xda/eq_xaa/eq_xigbl carry spurious ~1-2
    # seed-incomplete residuals that masquerade as differing equations; seed_levels
    # (in _solve_gtap) reads them the same way.
    def _setv(comp, key, val):
        try:
            comp[key].set_value(val)
            return 1
        except Exception:
            return 0

    n_cascade = _cascade_derived_seed_mp(m2, V, _setv, period)

    # 2) The solve, from the GAMS point; the residual TAIL at the seed is read inside
    #    the solver, on the system PATH was handed (start_at_seed), not on m2.
    tails: dict = {}
    m, res = _solve_gtap(dataset, gdx_path, if_sub, mode, seeded=m2, tails=tails)
    code = res.get(period, {}).get("code")
    resid = float(res.get(period, {}).get("residual") or 0.0)
    _, sent_after = _sentinel()
    eq_resid = tails.get(period, [])  # (resid, family, idx-without-period)
    median = statistics.median([r_ for r_, _, _ in eq_resid]) if eq_resid else 0.0

    return {
        "code": code,
        "resid": resid,
        "median": median,
        "eq_resid": eq_resid,
        "n_seed": None,
        "n_cascade": n_cascade,
        "sent_name": sent_name,
        "sent_before": sent_before,
        "sent_after": sent_after,
    }


def _classify_eq(family: str, idx) -> str:
    """not_gams | benign | real — rows that follow GEMPACK on purpose are not_gams;
    leaf (rgdpmp/pgdpmp) and ROW-region rows are benign-known."""
    if family in _NOT_GAMS_EQS:
        return "not_gams"
    if family in _LEAF_EQS:
        return "benign"
    idx_str = " ".join(
        str(x) for x in (idx if isinstance(idx, (tuple, list)) else (idx,))
    )
    if "ROW" in idx_str:
        return "benign"
    return "real"


def _work(
    dataset: str,
    gdx_path: Path,
    period: str,
    top: int,
    mode: str = "altertax",
    if_sub: bool = False,
) -> dict:
    out = run_seed_and_solve_gtap(dataset, gdx_path, period, top, if_sub, mode)
    code, resid = out["code"], out["resid"]
    if code is None:
        # The core reported the period as not solved (a solve that raises is an
        # "error" status, which old_codes turns into an exception): the GAMS point was
        # not tested. Never read it as "an equation differs".
        return {
            "status": "error",
            "headline": (
                f"the solve did not run at the GAMS {period} point "
                f"(code=None, resid={resid:.1e}); see the solver log on stderr"
            ),
            "violations": [],
            "period": period,
            "meta": {
                "error_kind": "solve_not_run",
                "mode": mode,
                "if_sub": int(if_sub),
            },
        }
    sb, sa = out["sent_before"], out["sent_after"]
    drift = (abs(sa - sb) / abs(sb)) if (sb and sa and abs(sb) > 1e-9) else None

    # Tail above tol, split benign vs real.
    tail = [(r_, fam, idx) for r_, fam, idx in out["eq_resid"] if r_ > RESID_TOL]
    real = [(r_, fam, idx) for r_, fam, idx in tail if _classify_eq(fam, idx) == "real"]
    benign = [
        (r_, fam, idx) for r_, fam, idx in tail if _classify_eq(fam, idx) == "benign"
    ]
    not_gams = [
        (r_, fam, idx) for r_, fam, idx in tail if _classify_eq(fam, idx) == "not_gams"
    ]

    # STAYS iff solve converged AND no REAL equation carries residual at the seed.
    stays = code == 1 and resid < 1e-6 and not real

    violations = []
    for r_, fam, idx in (real + benign + not_gams)[:top]:
        v = make_violation(fam, idx, "resid_at_gams", r_)
        v["class"] = _classify_eq(fam, idx)
        violations.append(v)

    if stays:
        headline = (
            f"STAYS: GAMS point is a fixed point of equilibria "
            f"(solve code={code}, resid={resid:.1e}, no real eq residual) "
            f"→ EQUILIBRIUM SELECTION, no differing equation to chase"
        )
        detection = make_detection(
            what="GAMS point is a stable fixed point",
            evidence=f"seeded GAMS {period}, solved code={code} resid={resid:.1e}, "
            f"residual tail has 0 real (non-leaf/non-ROW) equations",
            confidence="firm",
        )
        status = "clean"
    else:
        worst = real[0] if real else (tail[0] if tail else None)
        worst_str = (
            (
                f"{worst[1]}{tuple(str(x) for x in (worst[2] if isinstance(worst[2], (tuple, list)) else (worst[2],)))}"
                f"={worst[0]:.3g}"
            )
            if worst
            else "(solve failed)"
        )
        drift_str = (
            f", sentinel {out['sent_name']} {sb:.4f}→{sa:.4f} ({100 * drift:+.2f}%)"
            if drift
            else ""
        )
        headline = (
            f"GOES: GAMS point is NOT a fixed point (solve code={code}, resid={resid:.1e}"
            f"{drift_str}) → an equation DIFFERS. Worst REAL residual: {worst_str}. "
            f"Read the TAIL (below), not the median ({out['median']:.1e}). "
            f"Next: CONVERT-diff that equation vs GAMS."
        )
        detection = make_detection(
            what=f"an equation differs from GAMS (GAMS point not a fixed point); top real residual {worst_str}",
            evidence=f"seeded GAMS {period}, solved code={code} resid={resid:.1e}; "
            f"{len(real)} real + {len(benign)} benign(leaf/ROW) + {len(not_gams)} not_gams eqs with resid>{RESID_TOL:g}",
            confidence="firm",
        )
        status = "dirty"

    return {
        "status": status,
        "headline": headline,
        "violations": violations,
        "period": period,
        "meta": {
            "mode": mode,
            "if_sub": int(if_sub),
            "solve_code": code,
            "solve_resid": resid,
            "residual_median": out["median"],
            "n_seeded": out["n_seed"],
            "n_cascaded": out["n_cascade"],
            "n_real_resid": len(real),
            "n_benign_resid": len(benign),
            "n_not_gams_resid": len(not_gams),
            "sentinel": out["sent_name"],
            "sentinel_before": sb,
            "sentinel_after": sa,
            "sentinel_drift_pct": (100 * drift) if drift else None,
            "detection": detection,
            "note": (
                "median masks the signal — the tail (worst eqs) is the lead; "
                "benign = rgdpmp/pgdpmp report leaves + ROW corrupt-ref rows; "
                "not_gams = rows that follow GEMPACK on purpose (_NOT_GAMS_EQS)"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cascade tool 11: seed-and-solve (selection vs differing-eq)."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gdx", required=True, type=Path)
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument(
        "--top", type=int, default=20, help="how many residual-tail eqs to report"
    )
    ap.add_argument(
        "--mode",
        default="altertax",
        choices=["altertax", "gtap"],
        help="altertax CD (default) or pure-gtap real-CES",
    )
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1], help="ifSUB mode")
    args = ap.parse_args()
    return run_tool(
        "seed_and_solve",
        args.dataset,
        lambda: _work(
            args.dataset,
            args.gdx,
            args.period,
            args.top,
            mode=args.mode,
            if_sub=bool(args.ifsub),
        ),
        period_hint=args.period,
    )


if __name__ == "__main__":
    raise SystemExit(main())
