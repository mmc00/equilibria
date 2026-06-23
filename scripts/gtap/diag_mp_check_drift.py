"""PURE DIAGNOSTIC (no fix): instrument the multi-period CHECK solve.

The 3x3 multi-period gate fails: base code=1, but check/shock code=2 (res~3e-4),
shock match 38.51% with a CONCENTRATED Services blowup (xs/x/xds[Svces] ~18x).
The check fails first and warm-starts the shock, so the check is the root.

This script (measure-before-fix) builds the SAME model as diag_mp_3x3, seeds from
GAMS, solves base -> check, then at the CHECK solution reports:

  [A] Per-equation residual ranking on the CHECK period (which family carries 3e-4).
  [B] Per-variable drift of the CHECK vars from their GAMS seed, flagging with the
      Tool-7 marker (drifts while paired eq resid ~ 0 = free/tautological DOF).

It does NOT change the model or driver. Output only.

Usage:
    uv run python scripts/gtap/diag_mp_check_drift.py [--top 30]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

DATASET_DIR = ROOT / "datasets" / "gtap7_3x3"
REF = Path("/Users/marmol/proyectos2/equilibria_refs/gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx")


def _load_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    mod = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--period", default="check", choices=["check", "shock"])
    ap.add_argument("--mute-welfare", action="store_true",
                    help="DECISIVE PROBE: fix the inert welfare leaf vars "
                         "{cv,ev,walras,u,ug,us} at their seed + deactivate their "
                         "rows (keeps square) BEFORE the check solve, to separate "
                         "the welfare-poison problem from the real-side blowup. "
                         "Leaves uh/eq_uh intact (real blast radius via eq_zcons).")
    args = ap.parse_args()
    ACTIVE = args.period

    import pyomo.environ as pyo
    from pyomo.environ import value as V, Constraint, Var
    from pyomo.core.expr import identify_variables
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel, PERIODS
    from equilibria.templates.gtap.gtap_multiperiod_driver import (
        freeze_inactive_periods, freeze_period, _seed_period_from_prior,
        _replicate_sp_fixing, _replicate_sp_bounds,
    )
    from equilibria.templates.gtap import GTAPModelEquations

    run_gtap = _load_run_gtap()

    print("=== Build params + altertax CD ===")
    p_raw = GTAPParameters()
    p_raw.load_from_har(
        basedata_path=DATASET_DIR / "basedata.har",
        sets_path=DATASET_DIR / "sets.har",
        default_path=DATASET_DIR / "default.prm",
        baserate_path=DATASET_DIR / "baserate.har",
    )
    rr = list(p_raw.sets.r)[-1]
    p_alt = apply_altertax_elasticities(p_raw, in_place=False)

    print("=== Build multi-period model ===")
    alt_closure = GTAPClosureConfig(
        name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True,
        if_sub=False, numeraire="pnum",
    )
    base_closure = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )
    mp = GTAPMultiPeriodModel(p_alt.sets, p_alt, alt_closure, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for period in PERIODS:
        mp.build_equations_intra(m, period)
    mp.build_equations_fisher(m)
    m._residual_region = rr

    print("=== Seed all periods from GAMS ===")
    mp.seed_all_periods(m, REF)

    # Snapshot the GAMS-seeded values for the ACTIVE period BEFORE solving
    # (this is the GAMS reference point for the drift test).
    seed_vals = {}
    for v in m.component_objects(Var, active=True):
        for idx in v:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != ACTIVE:
                continue
            vd = v[idx]
            if vd.value is not None:
                seed_vals[(v.name, idx)] = float(vd.value)

    # ---- Replicate the driver up to (and including) the ACTIVE solve ----
    print(f"=== Solve base -> ... -> {ACTIVE} (driver replication) ===")

    def _deact_xft(period):
        eq_xft = getattr(m, "eq_xft", None)
        eq_xfteq = getattr(m, "eq_xfteq", None)
        if eq_xft is None or eq_xfteq is None:
            return
        for r in m.r:
            for f in m.f:
                try:
                    cd = eq_xfteq[(r, f, period)]
                except KeyError:
                    continue
                if not cd.active:
                    continue
                try:
                    xc = eq_xft[(r, f, period)]
                except KeyError:
                    continue
                if xc.active:
                    xc.deactivate()

    # BASE
    freeze_inactive_periods(m, "base")
    _deact_xft("base")
    sp_base = GTAPModelEquations(p_alt.sets, p_alt, base_closure, residual_region=rr).build_model()
    _replicate_sp_fixing(m, sp_base, "base")
    _replicate_sp_bounds(m, sp_base, "base")
    del sp_base
    run_gtap._run_path_capi_nonlinear_full(
        m, p_alt, enforce_post_checks=False, strict_path_capi=False,
        closure_config=base_closure, equation_scaling=True, solution_hint=None)
    freeze_period(m, "base")
    for r in m.r:
        try:
            if not m.pabs[r, "base"].fixed:
                m.pabs[r, "base"].fix(1.0)
        except (KeyError, AttributeError):
            pass

    if ACTIVE == "check":
        freeze_inactive_periods(m, "check")
        _seed_period_from_prior(m, "base", "check")
        for r in p_alt.sets.r:
            try:
                if hasattr(m, "regy") and m.regy[r, "check"].fixed:
                    m.regy[r, "check"].unfix()
            except Exception:
                pass
        _deact_xft("check")
        sp_chk = GTAPModelEquations(p_alt.sets, p_alt, alt_closure, residual_region=rr).build_model()
        _replicate_sp_fixing(m, sp_chk, "check")
        _replicate_sp_bounds(m, sp_chk, "check")
        del sp_chk

        # DECISIVE PROBE: mute the inert welfare leaf rows.
        # Each leaf eq_X pairs with its own var X; fixing X at its seed and
        # deactivating eq_X removes one row + one free var → stays square.
        if args.mute_welfare:
            n_mute = 0
            for eqn, vn in [("eq_cv", "cv"), ("eq_ev", "ev"),
                            ("eq_walras", "walras"), ("eq_u", "u"),
                            ("eq_ug", "ug"), ("eq_us", "us")]:
                eqc = getattr(m, eqn, None)
                vc = getattr(m, vn, None)
                if eqc is None or vc is None:
                    continue
                for r in p_alt.sets.r:
                    # walras may be scalar-per-t or per-region; handle both.
                    for cand in [(r, "check"), ("check",)]:
                        try:
                            vd = vc[cand]
                        except (KeyError, TypeError):
                            continue
                        if not vd.fixed and vd.value is not None:
                            vd.fix(float(vd.value))
                        try:
                            cd = eqc[cand]
                            if cd.active:
                                cd.deactivate()
                                n_mute += 1
                        except (KeyError, TypeError):
                            pass
                        break
            print(f"  [mute-welfare] deactivated {n_mute} welfare leaf rows "
                  f"(cv/ev/walras/u/ug/us), fixed their vars at seed")

        r_chk = run_gtap._run_path_capi_nonlinear_full(
            m, p_alt, enforce_post_checks=False, strict_path_capi=False,
            closure_config=alt_closure, equation_scaling=True, solution_hint=None)
        code = int(r_chk.get("termination_code") or 0)
        res = float(r_chk.get("residual") or float("inf"))
        print(f"  CHECK solve: code={code}, residual={res:.3e}")
    else:
        raise SystemExit("only --period check supported in this diag")

    # ------------------------------------------------------------------
    # [A] Per-equation residual ranking on the ACTIVE period
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[A] Per-equation residual ranking ({ACTIVE} period, ACTIVE constraints)")
    print("=" * 70)
    eq_resid = []  # (family, idx, |resid|)
    fam_max = {}   # family -> (|resid|, idx)
    for con in m.component_objects(Constraint, active=True):
        for idx in con:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != ACTIVE:
                continue
            cd = con[idx]
            if not cd.active:
                continue
            try:
                body = V(cd.body)
            except Exception:
                continue
            # residual = body - lb (these are equality g(x)==0 written as body==lb)
            lb = cd.lower
            try:
                lbv = V(lb) if lb is not None else 0.0
            except Exception:
                lbv = 0.0
            r = abs(body - lbv)
            eq_resid.append((con.name, idx, r))
            if con.name not in fam_max or r > fam_max[con.name][0]:
                fam_max[con.name] = (r, idx)

    eq_resid.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top {args.top} single-cell residuals:")
    for name, idx, r in eq_resid[:args.top]:
        print(f"    {name}[{idx}]  resid={r:.3e}")

    print(f"\n  Top 20 families by max residual:")
    fam_sorted = sorted(fam_max.items(), key=lambda kv: kv[1][0], reverse=True)
    for name, (r, idx) in fam_sorted[:20]:
        print(f"    {name:24s} max_resid={r:.3e}  at {idx}")

    # ------------------------------------------------------------------
    # [B] Per-variable drift from GAMS seed (Tool-7 style)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[B] Per-variable DRIFT from GAMS seed ({ACTIVE} period, FREE vars)")
    print("=" * 70)

    # Build a quick paired-eq-residual lookup by variable id for the ⚑ flag:
    # for each active constraint, record max residual among the vars it touches.
    var_to_eqresid = {}
    for con in m.component_objects(Constraint, active=True):
        for idx in con:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != ACTIVE:
                continue
            cd = con[idx]
            if not cd.active:
                continue
            try:
                body = V(cd.body)
                lb = cd.lower
                lbv = V(lb) if lb is not None else 0.0
                r = abs(body - lbv)
            except Exception:
                continue
            try:
                for var in identify_variables(cd.body, include_fixed=False):
                    key = id(var)
                    if key not in var_to_eqresid or r < var_to_eqresid[key]:
                        # min residual among touching eqs: if ANY paired eq has
                        # near-zero residual the var is "satisfied" => free DOF
                        var_to_eqresid[key] = r
            except Exception:
                pass

    drifts = []  # (drift_abs, rel, vname, idx, seed, now, min_eqresid)
    for v in m.component_objects(Var, active=True):
        for idx in v:
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t != ACTIVE:
                continue
            vd = v[idx]
            if vd.fixed:
                continue
            now = vd.value
            seed = seed_vals.get((v.name, idx))
            if now is None or seed is None:
                continue
            d = abs(float(now) - seed)
            rel = d / abs(seed) if abs(seed) > 1e-9 else (0.0 if d < 1e-9 else float("inf"))
            min_eqr = var_to_eqresid.get(id(vd), None)
            drifts.append((d, rel, v.name, idx, seed, float(now), min_eqr))

    drifts.sort(key=lambda x: x[1], reverse=True)  # by relative drift
    print(f"\n  Top {args.top} drifters (by relative drift, FREE vars only):")
    print(f"    {'flag':4s} {'var':14s} {'rel':>9s} {'seed':>12s} {'now':>12s} {'min_eqresid':>12s}  idx")
    for d, rel, name, idx, seed, now, minr in drifts[:args.top]:
        flag = "FREE" if (minr is not None and minr < 1e-7) else ""
        minr_s = f"{minr:.2e}" if minr is not None else "n/a"
        print(f"    {flag:4s} {name:14s} {rel:9.2%} {seed:12.5g} {now:12.5g} {minr_s:>12s}  {idx}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
