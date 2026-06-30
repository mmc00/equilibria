"""Tool #7 of the parity-debug cascade: the DRIFT TEST (free-DOF / basin detector).

THE BLIND SPOT this covers: tools 3 (.nl coeff diff), 4 (calibration diff), 5 (equation-
form diff) and 6 (MCP-pairing diff) all compare Python AGAINST GAMS assuming the gap is a
DIFFERENCE between them — a different coefficient, calibration input, equation form, or
eq↔var pairing. They are structurally BLIND to a gap that is NOT a difference: a variable
that is a FREE / DEGENERATE degree of freedom IDENTICALLY in both models (e.g. under
forced-CD sigmav=sigmap=1, eq_pvaeq/eq_pndeq degenerate to the tautology exp(log(pva))==pva
in BOTH Python and GAMS). Such a variable has no restoring force, so PATH places it on a
different (but valid) equilibrium branch than GAMS — yet:
  - the .nl diff sees IDENTICAL coefficients (tautology == tautology) → 0 diffs, gate green;
  - the MCP-pairing diff sees IDENTICAL pairing (both pair pvaeq.pva) → "pairing matches";
  - the residual at the GAMS point is ~0 (a tautology is satisfied by ANY value);
so none of them flags it. This is exactly the gtap7_3x3 78.69%→97.68% gap: pva/pnd are
free under CD; GAMS pins them only via its multi-period holdfix(t-1) sequencing, a solver-
side mechanism that lives OUTSIDE the model equations and is invisible to static compares.

WHAT THIS TOOL DOES: seed Python at the GAMS reference point, run the real PATH solve, and
report which variables DRIFT most from the seeded GAMS value. A large drift on a converged
(code=1) solve = a free/degenerate DOF (or a basin the warm-start can't hold). It also
flags equations whose residual at the GAMS point is ~0 BUT whose paired variable is in the
top drifters — the signature of a tautological/vacuous equation (the smoking gun).

USAGE:
    uv run python scripts/gtap/drift_test.py --dataset gtap7_3x3 \\
        --gdx /Users/.../out_altertax_ifsub0.gdx --period shock --top 25

INTERPRETATION:
  - Top drifters are PRICES/quantities of one nest (pva/pnd/va/nd) → free CD subsystem;
    the fix is to HOLD them (holdfix: fix + deactivate the tautological paired eq), not to
    chase an equation/coefficient bug that doesn't exist.
  - Top drifters spread across capital/tax (chif/savf/ytaxshr) with the price nest stable →
    a closure or seeding hole (check the seeder completeness first — see warmstart_from_gams).
  - Solve is code!=1 → not a drift problem; fix convergence first (tool 0 check-warmstart).
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import gams_levels, list_populated_vars  # type: ignore
import diff_altertax as DA  # reuse the exact altertax build recipe + complete seeder
from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract

# var → its (likely) paired eq, for the tautology / free-DOF (⚑) flag.
_PAIR = {"pva": "eq_pvaeq", "pnd": "eq_pndeq", "px": "eq_po", "pf": "eq_pfeq",
         "pfact": "eq_pfact", "pwfact": "eq_pwfact"}
_PRICE_NEST = {"pva", "pnd", "va", "nd", "px"}


def _build_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    mod = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_drift(dataset: str, gdx_path: Path, period: str, top: int, tol: float):
    import pyomo.environ as pyo
    from pyomo.environ import value as V, Constraint
    from pyomo.core.expr import identify_variables
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    run_gtap = _build_run_gtap()
    data_dir = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(basedata_path=data_dir / "basedata.har", sets_path=data_dir / "sets.har",
                    default_path=data_dir / "default.prm", baserate_path=data_dir / "baserate.har")
    p = apply_altertax_elasticities(p, in_place=False)
    res = list(p.sets.r)[-1]
    base_clo = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False, if_sub=False, numeraire="pnum")
    alt_clo = GTAPClosureConfig(name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True, if_sub=False, numeraire="pnum")
    m_b = GTAPModelEquations(p.sets, p, base_clo, residual_region=res).build_model()

    p_per = p
    if period == "shock":
        p_per = copy.deepcopy(p)
        for k in list(p_per.taxes.imptx.keys()):
            p_per.taxes.imptx[k] = float(p_per.taxes.imptx[k] or 0.0) * 1.10
    m = GTAPModelEquations(p_per.sets, p_per, alt_clo, residual_region=res, t0_snapshot=m_b).build_model()
    for _r in p_per.sets.r:
        if hasattr(m, "regy") and m.regy[_r].fixed:
            m.regy[_r].unfix()
        if hasattr(m, "phip"):
            m.phip[_r].set_value(1.0)

    # SEED at the GAMS point. Use the INLINE map (xd/xm + camelCase, NO xa/p/pp/piGbl):
    # the complete seeder over-seeds GAMS derived values that conflict with Python's init
    # and lands a code=2 basin, whereas the inline subset reproduces the code=1 baseline
    # (78.69%) — the basin we want to measure drift FROM. (Counter-intuitive but verified:
    # see project_gtap7_3x3_sigma_test — more seeding ≠ better warm start here.)
    _imap = {"ytaxInd": "ytax_ind", "factY": "facty", "phiP": "phip",
             "regY": "regy", "xd": "xda", "xm": "xma"}
    for _vn in list_populated_vars(gdx_path):
        _gv = gams_levels(gdx_path, _vn)
        _pn = _imap.get(_vn, _vn)
        _pv = getattr(m, _pn, None)
        if _pv is None:
            _pv = getattr(m, _vn, None)
        if _pv is None:
            _pv = getattr(m, _vn.lower(), None)
        if _pv is None:
            continue
        for _gk, _g in _gv.items():
            if not (isinstance(_gk, tuple) and _gk[-1] == period):
                continue
            _pk = tuple(DA._strip_set_prefix(k) for k in _gk[:-1])
            try:
                _v = _pv[_pk] if len(_pk) > 1 else _pv[_pk[0]]
                if not _v.fixed:
                    _v.set_value(float(_g))
            except Exception:
                pass
    seeded = {}
    for v in m.component_objects(pyo.Var, active=True):
        for idx in v:
            if v[idx].value is not None:
                seeded[(v.name, idx)] = float(v[idx].value)

    # Equation residuals AT the seed (to flag tautologies: resid≈0 but paired var drifts).
    eq_resid = {}
    for c in m.component_objects(Constraint, active=True):
        mx = 0.0
        for idx in c:
            try:
                body = V(c[idx].body); lo = V(c[idx].lower) if c[idx].lower is not None else 0.0
                mx = max(mx, abs(body - lo))
            except Exception:
                pass
        eq_resid[c.name] = mx

    # SOLVE from the seed.
    r = run_gtap._run_path_capi_nonlinear_full(
        m, p_per, enforce_post_checks=False, strict_path_capi=False,
        closure_config=alt_clo, equation_scaling=True,
        solution_hint=GTAPVariableSnapshot.from_python_model(m))
    code = r.get("termination_code"); resid = float(r.get("residual") or 0.0)

    # DRIFT: |solved − seeded| / max(|seeded|, floor). The floor avoids spurious huge
    # rel-drifts on near-zero vars (e.g. walras≈0 → 2e7%) that aren't economically real;
    # vars whose seeded magnitude is below the floor are skipped from the ranking.
    floor = 1e-3
    drifts = []
    for (vname, idx), sv in seeded.items():
        if abs(sv) < floor:
            continue
        v = getattr(m, vname, None)
        if v is None:
            continue
        try:
            now = float(V(v[idx]))
            drifts.append((abs(now - sv) / abs(sv), vname, idx, sv, now))
        except Exception:
            pass
    drifts.sort(reverse=True)
    return code, resid, drifts, eq_resid


def _is_free_dof(vname: str, rel: float, eq_resid: dict, tol: float) -> tuple[bool, str, float]:
    """Return (is_free_dof, paired_eq, paired_eq_resid) for the ⚑ flag.

    A var is a FREE/TAUTOLOGICAL DOF when it drifts >= tol while its paired
    equation's residual at the seed is ~0 (the eq does not restrain its own var).
    """
    eqn = _PAIR.get(vname)
    if not eqn:
        return False, "", 0.0
    er = eq_resid.get(eqn, 1.0)
    return (er < 1e-6 and rel >= tol), eqn, er


# ---------------------------------------------------------------------------
# Human-readable formatter — KEPT for debug only; writes to STDERR so it can
# never contaminate the JSON on stdout. Not called in the normal path.
# ---------------------------------------------------------------------------
def _debug_print(args, code, resid, drifts, eq_resid):
    print(f"=== DRIFT TEST: {args.dataset} period={args.period} "
          f"ref={args.gdx.name} ===", file=sys.stderr)
    print(f"Solve: code={code} resid={resid:.3e}", file=sys.stderr)
    if code != 1:
        print("  ⚠️  Solve did NOT converge (code!=1) — CONVERGENCE problem, "
              "not drift.", file=sys.stderr)
    n_drift = sum(1 for d in drifts if d[0] >= args.tol)
    print(f"\nVariables drifting >= {args.tol:g}: {n_drift}/{len(drifts)}",
          file=sys.stderr)
    for rel, vname, idx, sv, nv in drifts[:args.top]:
        free, eqn, _er = _is_free_dof(vname, rel, eq_resid, args.tol)
        flag = (f"  ⚑ {eqn} resid≈0 but var drifts → FREE/TAUTOLOGICAL DOF"
                if free else "")
        print(f"{rel*100:>7.2f}%  {vname:<12} {str(idx):<28} "
              f"{sv:>12.5f} {nv:>12.5f}{flag}", file=sys.stderr)


def _work(args) -> dict:
    code, resid, drifts, eq_resid = run_drift(
        args.dataset, args.gdx, args.period, args.top, args.tol)
    _debug_print(args, code, resid, drifts, eq_resid)

    # By its own docstring: code != 1 is a CONVERGENCE problem, not a drift one;
    # the drift ranking is unreliable, so report it as an error for the cascade.
    if code != 1:
        return dict(
            status="error", period=args.period,
            headline=(f"solve did NOT converge (code={code}, resid={resid:.3e}) "
                      f"— convergence problem, not drift; run tool 0 "
                      f"(check-warmstart) first"),
            violations=[],
            meta={"error_kind": "no_convergence",
                  "termination_code": code, "residual": resid,
                  "gdx_ref": str(args.gdx), "tol": args.tol})

    # Converged: rank drifters; a drifter whose paired eq resid≈0 is a free DOF.
    viols = []
    for rel, vname, idx, sv, nv in drifts:
        if rel < args.tol:
            continue
        free, eqn, er = _is_free_dof(vname, rel, eq_resid, args.tol)
        v = make_violation(vname, idx, "drift_rel", rel)
        v["free_dof"] = bool(free)
        v["paired_eq"] = eqn or None
        v["paired_eq_resid"] = er
        v["seeded"] = sv
        v["solved"] = nv
        viols.append(v)

    # Tally leading drifters and detect the price-nest free-DOF signature.
    top_vars: dict[str, int] = {}
    for rel, vname, idx, sv, nv in drifts[: max(args.top, 10)]:
        if rel >= args.tol:
            top_vars[vname] = top_vars.get(vname, 0) + 1
    lead = sorted(top_vars.items(), key=lambda kv: -kv[1])
    price_nest_leads = any(v in _PRICE_NEST for v, _ in lead[:4])
    n_free = sum(1 for v in viols if v["free_dof"])

    status = "dirty" if viols else "clean"
    if not viols:
        headline = (f"seed-at-GAMS {args.period}: solve converged (code=1), "
                    f"no variable drifts >= {args.tol:g} — the GAMS branch is "
                    f"reachable; this layer does not explain the gap")
    else:
        worst = viols[0]
        nest_note = ""
        if price_nest_leads:
            nest_note = (" — a PRICE/QUANTITY NEST leads; likely a FREE DOF, "
                         "fix = HOLD (fix var + deactivate tautological paired eq)")
        headline = (
            f"seed-at-GAMS {args.period}: {len(viols)} var(s) drift >= "
            f"{args.tol:g}, {n_free} flagged FREE-DOF (⚑); worst "
            f"{worst['entity']}{worst['index']}={worst['value']*100:.2f}%"
            f"{nest_note}")

    return dict(
        status=status, period=args.period, headline=headline, violations=viols,
        meta={"termination_code": code, "residual": resid,
              "n_drifters": len(viols), "n_free_dof": n_free,
              "price_nest_leads": price_nest_leads,
              "leading_vars": [{"var": v, "cells": n} for v, n in lead[:8]],
              "tol": args.tol, "gdx_ref": str(args.gdx)})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, required=True, help="GAMS reference GDX to seed from")
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1e-3, help="rel-drift threshold to count as drifting")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap only)")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED, honestly: drift_test seeds + solves the SINGLE-period
    # altertax model. The pure-gtap shock is multi-period (the tariff wedge enters
    # via solve_multiperiod's in-place rebuilds), so a single-period gtap solve would
    # drift FROM the wrong (un-shocked) equations. seed_and_solve --mode gtap already
    # runs the faithful multi-period gtap solve and reports the sentinel drift +
    # residual tail — use it for the pure-gtap drift/free-DOF question.
    if args.mode == "gtap":
        def _unsupported() -> dict:
            return dict(
                status="error", period=args.period,
                headline=("drift_test does not support --mode gtap: it seeds+solves the "
                          "single-period altertax model; the pure-gtap shock is wired in "
                          "solve_multiperiod (multi-period, in-place rebuilds). Use "
                          "seed_and_solve --mode gtap (it reports the gtap sentinel drift)."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("drift_test", args.dataset, _unsupported,
                        period_hint=args.period)

    return run_tool("drift_test", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
