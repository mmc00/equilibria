"""Tool #9 of the parity-debug cascade: TAUTOLOGY / UNANCHORED-VARIABLE detector.

THE BLIND SPOT this covers (the one that capped gtap7_3x3 at 78.69% for many sessions):
the gap was a MISSING equation, not a DIFFERENT one. Under forced-CD (sigmav=sigmap=1)
eq_pvaeq degenerated to the tautology `exp(log(pva))==pva` (i.e. `1==1`) — identical in
Python AND GAMS — so it placed NO constraint on its paired variable pva, leaving pva free
to slide. Every static-compare tool is structurally blind to this:
  - tool 3 (.nl): identical coefficients (tautology==tautology) → 0 diff, gate green;
  - tool 6 (MCP-pairing): identical pairing (both pair pvaeq.pva) → "matches";
  - residual at the GAMS point: ~0 (a tautology is satisfied by ANY value);
they all "pass" because there is no DIFFERENCE to find — the equation is equally vacuous
in both. Comparison tools detect differences; they cannot detect an ABSENCE.

WHAT THIS TOOL DOES (numerical, not symbolic): for each equation↔variable MCP pair, it
PERTURBS the variable by a small δ and measures whether the equation's residual responds
(∂residual/∂var ≈ |Δresid|/δ). If the sensitivity is ~0, the equation does NOT constrain
its own paired variable → that variable is UNANCHORED (a tautology / free DOF). It then
cross-checks with the drift test's symptom: an unanchored var is exactly the one that
slides after a solve. This finds the CAUSE (vacuous equation) that tools 3/4/6 miss and
tool 7 only sees the symptom of.

USAGE:
    uv run python scripts/gtap/diff_tautology.py --dataset gtap7_3x3 \\
        --gdx /path/to/out_altertax_ifsub0.gdx --period shock

INTERPRETATION:
  - A flagged pair (eq_X ↔ x, sensitivity≈0) = eq_X is a tautology for x → x is a free DOF.
    The fix is to give x a real determinant (the missing economic identity), NOT to tweak a
    coefficient. Precedent: eq_pvaeq under CD → fixed with the VA-value identity
    pva·va=Σ(pfa·xf). See plan_gtap7_3x3_shock_close.md.
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
from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract
import diff_altertax as DA

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"


def _build_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    mod = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = mod
    spec.loader.exec_module(mod)
    return mod


# GAMS model-statement MCP pairs (eq → its complementary var). Subset that matters for
# the CD altertax block; extend as needed. Mirrors `model gtap /eq.var/`.
_PAIRS = [
    ("eq_pvaeq", "pva"), ("eq_pxeq", "px"), ("eq_po", "px"), ("eq_pndeq", "pnd"),
    ("eq_pfeq", "pf"), ("eq_pfteq", "pft"), ("eq_xfteq", "xft"), ("eq_va", "va"),
    ("eq_pdeq", "pd"), ("eq_peq", "p_rai"), ("eq_pmteq", "pmt"), ("eq_pwfact", "pwfact"),
    ("eq_pfact", "pfact"), ("eq_pabs", "pabs"), ("eq_yc", "yc"), ("eq_regy", "regy"),
]


def run_tautology_scan(dataset: str, gdx_path: Path, period: str, delta: float, tol: float):
    import pyomo.environ as pyo
    from pyomo.environ import value as V, Constraint
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

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
    DA.warmstart_from_gams(m, gdx_path, period)  # seed at the GAMS point

    def _resid(con):
        body = V(con.body); lo = V(con.lower) if con.lower is not None else 0.0
        return body - lo

    rows = []
    for eqn, vn in _PAIRS:
        eq = getattr(m, eqn, None)
        v = getattr(m, vn, None)
        if eq is None or v is None:
            continue
        # sample a few cells of the paired var; measure ∂resid_eq/∂var
        worst_sens = None
        worst_cell = None
        n = 0
        for idx in v:
            if v[idx].fixed or v[idx].value is None:
                continue
            # find the eq cell with the SAME index (or a compatible prefix)
            con = None
            try:
                con = eq[idx] if eq.is_indexed() else eq
            except Exception:
                # eq indexed differently (e.g. eq has (r,a), var has (r,f,a)); skip mismatch
                continue
            if con is None or not con.active:
                continue
            try:
                r0 = _resid(con)
                v0 = float(v[idx].value)
                v[idx].set_value(v0 * (1.0 + delta))
                r1 = _resid(con)
                v[idx].set_value(v0)  # restore
                sens = abs(r1 - r0) / (abs(v0) * delta + 1e-12)
                n += 1
                if worst_sens is None or sens < worst_sens:  # we want the LEAST sensitive
                    worst_sens = sens
                    worst_cell = idx
            except Exception:
                v[idx].set_value(v0)
        if n > 0:
            rows.append((worst_sens, eqn, vn, worst_cell, n))
    return rows


def _default_gdx(dataset: str) -> Path:
    return Path(f"{DEFAULT_REFS}/{dataset}_altertax_cd/out_altertax_ifsub0.gdx")


def _work(args) -> dict:
    gdx = args.gdx or _default_gdx(args.dataset)
    if not gdx.exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found: {gdx}",
                    violations=[],
                    meta={"error_kind": "gdx_not_found", "gdx": str(gdx)})

    rows = run_tautology_scan(args.dataset, gdx, args.period, args.delta, args.tol)
    rows.sort()  # lowest sensitivity first = most tautological

    # human-readable table → stderr (never contaminates the JSON stdout line)
    print(f"=== TAUTOLOGY / unanchored-var scan: {args.dataset} period={args.period} ===",
          file=sys.stderr)
    print(f"{'∂resid/∂var':>12}  {'equation':<14} {'var':<8} {'cell':<22} flag",
          file=sys.stderr)
    for sens, eqn, vn, cell, n in rows:
        flag = "  ⚑ TAUTOLOGY → UNANCHORED" if sens < args.tol else ""
        print(f"{sens:>12.3e}  {eqn:<14} {vn:<8} {str(cell):<22}{flag}", file=sys.stderr)

    flagged = [r for r in rows if r[0] < args.tol]
    violations = []
    for sens, eqn, vn, cell, n in flagged:
        v = make_violation(vn, list(cell) if isinstance(cell, tuple) else [cell],
                           "dresid_dvar", float(sens))
        v["kind"] = "unanchored_dof"
        v["equation"] = eqn
        v["cells_sampled"] = n
        v["note"] = (f"{eqn} does NOT constrain {vn} (∂resid/∂var≈0) → {vn} is a FREE DOF "
                     f"PATH places arbitrarily (root-selection cause)")
        violations.append(v)

    status = "dirty" if flagged else "clean"
    if flagged:
        worst = flagged[0]
        headline = (f"unanchored-DOF scan ({args.period}): {len(flagged)} eq↔var pair(s) "
                    f"with ∂resid/∂var<{args.tol:g} ⚑ — vacuous paired row = free DOF "
                    f"(root-selection cause); worst {worst[1]}↔{worst[2]}")
    else:
        headline = (f"unanchored-DOF scan ({args.period}): every paired equation constrains "
                    f"its variable — no free DOF; this layer does not explain the gap")
    return dict(status=status, period=args.period, headline=headline,
                violations=violations,
                meta={"gdx": str(gdx), "n_pairs_checked": len(rows),
                      "n_unanchored": len(flagged), "delta": args.delta, "tol": args.tol})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, default=None,
                    help="reference GDX (default: DEFAULT_REFS/<dataset>_altertax_cd/out_altertax_ifsub0.gdx)")
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--delta", type=float, default=1e-4, help="relative perturbation of the var")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="sensitivity below this = the eq does NOT constrain its var (tautology)")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB mode (pure-gtap only)")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED, honestly: diff_tautology perturbs vars in the SINGLE-period
    # altertax model and measures ∂resid/∂var. The pure-gtap shock equations only exist
    # after solve_multiperiod's in-place rebuilds (multi-period); perturbing the single-
    # period gtap build would probe the wrong (un-shocked) equations. seed_and_solve
    # --mode gtap is the faithful pure-gtap path (its residual tail already surfaces an
    # unanchored/tautological eq as a high resid-at-GAMS with a clean post-solve resid).
    if args.mode == "gtap":
        def _unsupported() -> dict:
            return dict(
                status="error", period=args.period,
                headline=("diff_tautology does not support --mode gtap: it perturbs the "
                          "single-period altertax model; the pure-gtap shock equations are "
                          "built by solve_multiperiod (multi-period, in-place rebuilds). "
                          "Use seed_and_solve --mode gtap."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("diff_tautology", args.dataset, _unsupported,
                        period_hint=args.period)

    return run_tool("diff_tautology", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
