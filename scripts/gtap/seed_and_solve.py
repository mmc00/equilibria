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

TWO SETUP DISCIPLINES (each one previously manufactured a false "differing equation" lead):
  1. CASCADE the derived-var seed. warmstart_from_gams does NOT seed Python-only derived
     aggregates (xd/xmt/xc/xg/xi/xiagg/xigbl/pigbl/kapEnd) consistently with their components,
     so eq_xd_agg/eq_xc/eq_xaa/eq_xigbl show spurious residuals 1–2 that CASCADE to ~0 once
     the derived vars are set from their own identities. A genuinely-differing equation does
     NOT fall on consistent seed; an artifact does. This tool cascades the derived seed first.
  2. BUILD THE SHOCK MODEL WITH THE TAX SHOCK APPLIED BEFORE BUILD. For the shock period,
     imptx*1.10 must be applied to the params BEFORE build_model — NOT by seeding shock values
     onto a base-built model (that leaves model.imptx at base while pm is shock → false 0.014
     residual on eq_pmeq). diff_altertax's shock build does this; this tool mirrors it.

DECISIVE PRECEDENT: gtap7_3x3 altertax shock. Seeded the exact GAMS point and solved → it
DRIFTED (code=2, pft[EU_28,NatRes] 1.004→1.05). The residual tail's only real non-ROW/non-leaf
entry was eq_xi (0.006–0.010). Traced to an esubi key-shape bug (sigmai=0 Leontief vs GAMS 1.01
CES; commit 0e2db11). The seed-and-solve test named the differing equation in ONE run — after
eight other leads. See project_gtap7_3x3_gap_is_factor_bias.

Usage:
    uv run python scripts/gtap/seed_and_solve.py --dataset gtap7_3x3 \\
        --gdx tests/fixtures/gtap7_altertax/gtap7_3x3/out_altertax_ifsub0.gdx \\
        --period shock --top 20

JSON contract (shared _parity_json schema): status=clean (STAYS = equilibrium selection,
nothing to fix) | dirty (GOES = a differing equation, see violations = residual tail) |
error (solve crash). exit 0/1/2.
"""
from __future__ import annotations
import argparse
import copy
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "tests" / "templates" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

from _diff_core import gams_levels, list_populated_vars  # type: ignore  # noqa: E402
import diff_altertax as DA  # reuse the exact altertax build recipe + shock build  # noqa: E402
from _parity_json import make_violation, make_detection, run_tool  # noqa: E402

# Residual above which an equation counts as "not satisfied at the GAMS point".
RESID_TOL = 1e-4
# Families that are KNOWN-benign even when they carry residual at the GAMS point:
# rgdpmp/pgdpmp are report leaves (no feedback); ROW-region rows ride a corrupt reference
# (project_gtap7_3x3_ref_corrupt_ROW). The tool flags these separately so the real lead
# (non-leaf, non-ROW) is not buried.
_LEAF_EQS = {"eq_rgdpmp", "eq_pgdpmp"}


def _build_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    mod = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cascade_derived_seed(m, V, setv) -> int:
    """DISCIPLINE 1: set the Python-only derived aggregates from their own identities so
    their residual reflects the equation, not an un-seeded init. Returns cells set."""
    n = 0
    for r in m.r:
        for i in m.i:
            try:
                n += setv(m.xd, (r, i), sum(V(m.xda[r, i, aa]) / V(m.xscale[r, aa])
                          for aa in m.aa if (r, aa) in m.xscale))
            except Exception:
                pass
            try:
                n += setv(m.xmt, (r, i), sum(V(m.xma[r, i, aa]) / V(m.xscale[r, aa])
                          for aa in m.aa if (r, aa) in m.xscale))
            except Exception:
                pass
            try:
                n += setv(m.xc, (r, i), V(m.xcshr[r, i]) * V(m.yc[r]) / max(V(m.pa[r, i, "hhd"]), 1e-9))
            except Exception:
                pass
            try:
                n += setv(m.xg, (r, i), V(m.g_share[r, i]) * V(m.yg[r]) / max(V(m.pa[r, i, "gov"]), 1e-9))
            except Exception:
                pass
            try:
                n += setv(m.xi, (r, i), V(m.xaa[r, i, "inv"]))
            except Exception:
                pass
    for r in m.r:
        try:
            n += setv(m.xiagg, (r,), V(m.yi[r]) / max(V(m.pi[r]), 1e-9))
        except Exception:
            pass
        try:
            n += setv(m.kapEnd, (r,), (1 - V(m.depr[r])) * V(m.kstock[r]) + V(m.xiagg[r]))
        except Exception:
            pass
    try:
        n += setv(m.xigbl, None, sum(V(m.xiagg[r]) - V(m.depr[r]) * V(m.kstock[r]) for r in m.r))
    except Exception:
        pass
    try:
        num = sum(V(m.pi[r]) * (V(m.xiagg[r]) - V(m.depr[r]) * V(m.kstock[r])) for r in m.r)
        n += setv(m.pigbl, None, num / max(V(m.xigbl), 1e-9))
    except Exception:
        pass
    return n


def run_seed_and_solve(dataset: str, gdx_path: Path, period: str, top: int):
    import pyomo.environ as pyo
    from pyomo.environ import value as V, Constraint
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    run_gtap = _build_run_gtap()
    data_dir = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(basedata_path=data_dir / "basedata.har", sets_path=data_dir / "sets.har",
                    default_path=data_dir / "default.prm", baserate_path=data_dir / "baserate.har")
    res = list(p.sets.r)[-1]
    base_clo = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False, if_sub=False, numeraire="pnum")
    alt_clo = GTAPClosureConfig(name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True, if_sub=False, numeraire="pnum")

    # DISCIPLINE 2: build the shock model with the tax shock applied BEFORE build.
    p_shock = p
    if period == "shock":
        p_shock = copy.deepcopy(p)
        for k in list(p_shock.taxes.imptx.keys()):
            p_shock.taxes.imptx[k] = float(p_shock.taxes.imptx[k] or 0.0) * 1.10
    _mb, p_alt, m = DA.build_altertax_models(p_shock, res, base_clo, alt_clo)

    # Seed the exact GAMS point for the requested period (complete warm-start).
    n_seed = DA.warmstart_from_gams(m, gdx_path, period)

    def setv(comp, key, val):
        try:
            comp[key].set_value(val)
            return 1
        except Exception:
            return 0

    n_cascade = _cascade_derived_seed(m, V, setv)

    # Record a sentinel var to report stay/drift unambiguously (first finite price).
    def _sentinel():
        for nm in ("pft", "pf", "px", "pa"):
            comp = getattr(m, nm, None)
            if comp is None:
                continue
            for idx in comp:
                try:
                    val = float(V(comp[idx]))
                    if val > 1e-6:
                        return f"{nm}{tuple(str(x) for x in (idx if isinstance(idx, tuple) else (idx,)))}", val
                except Exception:
                    pass
        return "(none)", None
    sent_name, sent_before = _sentinel()

    # SOLVE from the seed.
    r = run_gtap._run_path_capi_nonlinear_full(
        m, p_alt, enforce_post_checks=False, strict_path_capi=False,
        closure_config=alt_clo, equation_scaling=True,
        solution_hint=GTAPVariableSnapshot.from_python_model(m))
    code = r.get("termination_code")
    resid = float(r.get("residual") or 0.0)
    _, sent_after = _sentinel()

    # Residual TAIL at the (cascaded) seed — re-seed the GAMS point + cascade AGAIN on a
    # fresh build so the residual reflects the SEED, not the post-solve state.
    _mb2, p_alt2, m2 = DA.build_altertax_models(p_shock, res, base_clo, alt_clo)
    DA.warmstart_from_gams(m2, gdx_path, period)
    _cascade_derived_seed(m2, V, setv)
    eq_resid = []  # (resid, family, idx)
    for c in m2.component_objects(Constraint, active=True):
        for idx in c:
            cd = c[idx]
            if not cd.active:
                continue
            try:
                b = V(cd.body)
                lo = V(cd.lower) if cd.lower is not None else None
                up = V(cd.upper) if cd.upper is not None else None
                rr = 0.0
                if lo is not None:
                    rr = max(rr, abs(b - lo))
                if up is not None:
                    rr = max(rr, abs(b - up))
                eq_resid.append((rr, c.name, idx))
            except Exception:
                pass
    eq_resid.sort(reverse=True)
    median = statistics.median([r_ for r_, _, _ in eq_resid]) if eq_resid else 0.0

    return {
        "code": code, "resid": resid, "median": median, "eq_resid": eq_resid,
        "n_seed": n_seed, "n_cascade": n_cascade,
        "sent_name": sent_name, "sent_before": sent_before, "sent_after": sent_after,
    }


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
                n += setv(m.xd, (r, i, t), sum(V(m.xda[r, i, aa, t]) / _xs(r, aa)
                          for aa in m.aa))
            except Exception:
                pass
            try:
                n += setv(m.xmt, (r, i, t), sum(V(m.xma[r, i, aa, t]) / _xs(r, aa)
                          for aa in m.aa))
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
            n += setv(m.kapEnd, (r, t), (1 - V(m.depr[r])) * V(m.kstock[r, t]) + V(m.xiagg[r, t]))
        except Exception:
            pass
    try:
        n += setv(m.xigbl, (t,), sum(V(m.xiagg[r, t]) - V(m.depr[r]) * V(m.kstock[r, t]) for r in m.r))
    except Exception:
        pass
    try:
        num = sum(V(m.pi[r, t]) * (V(m.xiagg[r, t]) - V(m.depr[r]) * V(m.kstock[r, t])) for r in m.r)
        n += setv(m.pigbl, (t,), num / max(V(m.xigbl[t]), 1e-9))
    except Exception:
        pass
    return n


def run_seed_and_solve_gtap(dataset: str, gdx_path: Path, period: str, top: int,
                            if_sub: bool):
    """PURE-GTAP (real-CES, non-altertax) variant.

    The pure-gtap shock is intrinsically MULTI-PERIOD: the tariff wedge enters via
    solve_multiperiod(mode="gtap") (the _rebuild_eq_pmeq_shock / NatRes anchor /
    ytax[mt] rebuild links), which the single-period altertax build does NOT call.
    So we build the FULL multi-period gtap model (mirroring
    test_gtap_multiperiod_parity._solve_and_match / measure_gtap_pure_tols.py),
    seed it from the GAMS gtap GDX (the fixture IS the seed across base/check/shock),
    solve via solve_multiperiod(mode="gtap"), and compute the residual TAIL for the
    requested period on a FRESH re-seeded build (so the residual reflects the SEED,
    not the post-solve state — same discipline as the altertax path).
    """
    from pyomo.environ import value as V, Constraint
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel, PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    data_dir = ROOT / "datasets" / dataset

    def _load_params():
        p = GTAPParameters()
        p.load_from_har(basedata_path=data_dir / "basedata.har", sets_path=data_dir / "sets.har",
                        default_path=data_dir / "default.prm", baserate_path=data_dir / "baserate.har")
        return p

    def _gc(p, if_sub_):
        return GTAPClosureConfig(
            name="base", closure_type="MCP", capital_mobility="sluggish",
            fix_endowments=False, fix_taxes=False, fix_technology=False,
            if_sub=if_sub_, numeraire="pnum")

    def _build_seeded():
        p = _load_params()
        rr = list(p.sets.r)[-1]
        gc = _gc(p, if_sub)
        mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
        m = mp.build_sets()
        mp.build_vars(m)
        for per in PERIODS:
            mp.build_equations_intra(m, per)
        mp.build_equations_fisher(m)
        m._residual_region = rr
        mp.seed_all_periods(m, gdx_path)
        m._gtap_mode = True
        # CRITICAL for the shock period: the pure-gtap tariff shock is NOT a param
        # multiply before build — the DRIVER injects it into the shock-slice equations
        # at solve time (_rebuild_eq_pmeq_shock / _rebuild_import_demand_shock_ifsub /
        # _rebuild_eq_ytax_mt_shock).  A fresh built model has those eqs in BASE form,
        # so evaluating the residual at shocked seed values fabricates phantom residuals
        # on everything tariff-dependent (eq_pmeq/eq_ytax[mt]/eq_xweq).  Replicate the
        # driver's shock injection here so the residual reflects the ACTUAL shock system.
        if period == "shock":
            import copy as _copy
            from equilibria.templates.gtap import gtap_multiperiod_driver as _drv
            p_shock = _copy.deepcopy(p)
            _drv._apply_imptx_shock(p_shock, 0.10, gtap_mode=True)
            _n_pmeq = _drv._rebuild_eq_pmeq_shock(m, p_shock)
            if not _n_pmeq:  # ifSUB=1 leg: eq_pmeq inactive, inject into eq_xweq/eq_pmteq
                _drv._rebuild_import_demand_shock_ifsub(m, p, p_shock)
            _drv._rebuild_eq_ytax_mt_shock(m, p_shock)
        return p, gc, m

    # 1) Build + seed + SOLVE the full multi-period gtap model.
    p, gc, m = _build_seeded()

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

    res = solve_multiperiod(
        m, p, gc, ref_gdx=gdx_path,
        skip_base_solve=True, mute_welfare=True,
        seed_from_prior=False, holdfix_cd=False, mode="gtap")
    code = res.get(period, {}).get("code")
    resid = float(res.get(period, {}).get("residual") or 0.0)
    _, sent_after = _sentinel()

    # 2) Residual TAIL at the SEED — fresh re-seeded build, filter constraints to
    #    the requested period (MP constraints are indexed (...,period)). DISCIPLINE 1:
    #    cascade the Python-only derived aggregates for this period BEFORE reading the
    #    residual, else eq_xc/eq_xda/eq_xaa/eq_xigbl carry spurious ~1-2 seed-incomplete
    #    residuals that masquerade as differing equations.
    _p2, _gc2, m2 = _build_seeded()

    def _setv(comp, key, val):
        try:
            comp[key].set_value(val)
            return 1
        except Exception:
            return 0

    n_cascade = _cascade_derived_seed_mp(m2, V, _setv, period)
    eq_resid = []  # (resid, family, idx-without-period)
    for c in m2.component_objects(Constraint, active=True):
        for idx in c:
            cd = c[idx]
            if not cd.active:
                continue
            # period filter: keep only this period's slice.
            if isinstance(idx, tuple):
                if not idx or idx[-1] != period:
                    continue
                body_idx = idx[:-1] if len(idx) > 1 else ()
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
                eq_resid.append((rr_, c.name, body_idx))
            except Exception:
                pass
    eq_resid.sort(reverse=True)
    median = statistics.median([r_ for r_, _, _ in eq_resid]) if eq_resid else 0.0

    return {
        "code": code, "resid": resid, "median": median, "eq_resid": eq_resid,
        "n_seed": None, "n_cascade": n_cascade,
        "sent_name": sent_name, "sent_before": sent_before, "sent_after": sent_after,
    }


def _classify_eq(family: str, idx) -> str:
    """benign | real — leaf (rgdpmp/pgdpmp) and ROW-region rows are benign-known."""
    if family in _LEAF_EQS:
        return "benign"
    idx_str = " ".join(str(x) for x in (idx if isinstance(idx, (tuple, list)) else (idx,)))
    if "ROW" in idx_str:
        return "benign"
    return "real"


def _work(dataset: str, gdx_path: Path, period: str, top: int,
          mode: str = "altertax", if_sub: bool = False) -> dict:
    if mode == "gtap":
        out = run_seed_and_solve_gtap(dataset, gdx_path, period, top, if_sub)
    else:
        out = run_seed_and_solve(dataset, gdx_path, period, top)
    code, resid = out["code"], out["resid"]
    sb, sa = out["sent_before"], out["sent_after"]
    drift = (abs(sa - sb) / abs(sb)) if (sb and sa and abs(sb) > 1e-9) else None

    # Tail above tol, split benign vs real.
    tail = [(r_, fam, idx) for r_, fam, idx in out["eq_resid"] if r_ > RESID_TOL]
    real = [(r_, fam, idx) for r_, fam, idx in tail if _classify_eq(fam, idx) == "real"]
    benign = [(r_, fam, idx) for r_, fam, idx in tail if _classify_eq(fam, idx) == "benign"]

    # STAYS iff solve converged AND no REAL equation carries residual at the seed.
    stays = (code == 1 and resid < 1e-6 and not real)

    violations = []
    for r_, fam, idx in (real + benign)[:top]:
        v = make_violation(fam, idx, "resid_at_gams", r_)
        v["class"] = _classify_eq(fam, idx)
        violations.append(v)

    if stays:
        headline = (f"STAYS: GAMS point is a fixed point of equilibria "
                    f"(solve code={code}, resid={resid:.1e}, no real eq residual) "
                    f"→ EQUILIBRIUM SELECTION, no differing equation to chase")
        detection = make_detection(
            what="GAMS point is a stable fixed point",
            evidence=f"seeded GAMS {period}, solved code={code} resid={resid:.1e}, "
                     f"residual tail has 0 real (non-leaf/non-ROW) equations",
            confidence="firm")
        status = "clean"
    else:
        worst = real[0] if real else (tail[0] if tail else None)
        worst_str = (f"{worst[1]}{tuple(str(x) for x in (worst[2] if isinstance(worst[2], (tuple, list)) else (worst[2],)))}"
                     f"={worst[0]:.3g}") if worst else "(solve failed)"
        drift_str = f", sentinel {out['sent_name']} {sb:.4f}→{sa:.4f} ({100*drift:+.2f}%)" if drift else ""
        headline = (f"GOES: GAMS point is NOT a fixed point (solve code={code}, resid={resid:.1e}"
                    f"{drift_str}) → an equation DIFFERS. Worst REAL residual: {worst_str}. "
                    f"Read the TAIL (below), not the median ({out['median']:.1e}). "
                    f"Next: CONVERT-diff that equation vs GAMS.")
        detection = make_detection(
            what=f"an equation differs from GAMS (GAMS point not a fixed point); top real residual {worst_str}",
            evidence=f"seeded GAMS {period}, solved code={code} resid={resid:.1e}; "
                     f"{len(real)} real + {len(benign)} benign(leaf/ROW) eqs with resid>{RESID_TOL:g}",
            confidence="firm")
        status = "dirty"

    return {
        "status": status,
        "headline": headline,
        "violations": violations,
        "period": period,
        "meta": {
            "mode": mode, "if_sub": int(if_sub),
            "solve_code": code, "solve_resid": resid, "residual_median": out["median"],
            "n_seeded": out["n_seed"], "n_cascaded": out["n_cascade"],
            "n_real_resid": len(real), "n_benign_resid": len(benign),
            "sentinel": out["sent_name"], "sentinel_before": sb, "sentinel_after": sa,
            "sentinel_drift_pct": (100 * drift) if drift else None,
            "detection": detection,
            "note": ("median masks the signal — the tail (worst eqs) is the lead; "
                     "benign = rgdpmp/pgdpmp report leaves + ROW corrupt-ref rows"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cascade tool 11: seed-and-solve (selection vs differing-eq).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gdx", required=True, type=Path)
    ap.add_argument("--period", default="shock", choices=["base", "check", "shock"])
    ap.add_argument("--top", type=int, default=20, help="how many residual-tail eqs to report")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap build only)")
    args = ap.parse_args()
    return run_tool("seed_and_solve", args.dataset,
                    lambda: _work(args.dataset, args.gdx, args.period, args.top,
                                  mode=args.mode, if_sub=bool(args.ifsub)),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
