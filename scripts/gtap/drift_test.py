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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, required=True, help="GAMS reference GDX to seed from")
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1e-3, help="rel-drift threshold to count as drifting")
    args = ap.parse_args()

    print(f"=== DRIFT TEST: {args.dataset} period={args.period} ref={args.gdx.name} ===")
    code, resid, drifts, eq_resid = run_drift(args.dataset, args.gdx, args.period, args.top, args.tol)
    print(f"Solve: code={code} resid={resid:.3e}")
    if code != 1:
        print("  ⚠️  Solve did NOT converge (code!=1) — this is a CONVERGENCE problem, not a")
        print("     drift problem. Run tool 0 (triage.py --check-warmstart) first. Drift below")
        print("     is still informative but read it as 'where the non-converged point sits'.")

    n_drift = sum(1 for d in drifts if d[0] >= args.tol)
    print(f"\nVariables drifting >= {args.tol:g} from the GAMS seed: {n_drift}/{len(drifts)}")
    print(f"\n{'rel%':>8}  {'variable':<12} {'cell':<28} {'seeded':>12} {'solved':>12}")
    print("-" * 80)
    # map var → its (likely) paired eq for the tautology flag
    pair = {"pva": "eq_pvaeq", "pnd": "eq_pndeq", "px": "eq_po", "pf": "eq_pfeq",
            "pfact": "eq_pfact", "pwfact": "eq_pwfact"}
    for rel, vname, idx, sv, nv in drifts[:args.top]:
        flag = ""
        eqn = pair.get(vname)
        if eqn and eq_resid.get(eqn, 1.0) < 1e-6 and rel >= args.tol:
            flag = f"  ⚑ {eqn} resid≈0 but var drifts → FREE/TAUTOLOGICAL DOF"
        print(f"{rel*100:>7.2f}%  {vname:<12} {str(idx):<28} {sv:>12.5f} {nv:>12.5f}{flag}")

    # Summary verdict
    top_vars = {}
    for rel, vname, idx, sv, nv in drifts[: max(args.top, 10)]:
        if rel >= args.tol:
            top_vars[vname] = top_vars.get(vname, 0) + 1
    if top_vars:
        lead = sorted(top_vars.items(), key=lambda kv: -kv[1])
        print(f"\nTop drifting variables (count of cells): "
              f"{', '.join(f'{v}×{n}' for v, n in lead[:8])}")
        price_nest = {"pva", "pnd", "va", "nd", "px"}
        if any(v in price_nest for v, _ in lead[:4]):
            print("  → A PRICE/QUANTITY NEST leads the drift. Likely a FREE DOF (CD-degenerate "
                  "eqs). Fix = HOLD it (fix var + deactivate the tautological paired eq), not an "
                  "equation/coefficient bug. See --holdfix-pva in diff_altertax.py.")


if __name__ == "__main__":
    main()
