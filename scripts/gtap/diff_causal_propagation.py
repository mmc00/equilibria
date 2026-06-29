"""Tool #10 of the parity-debug cascade: CAUSAL-PROPAGATION (symptom-vs-cause detector).

THE QUESTION it answers WITHOUT running the fix: when a gap has an upstream candidate
cause (bucket A: pft/pf/pfy/pva root-selection) and a downstream-suspect bucket (bucket
O: demand quantities xa/xd/xm), is O a SYMPTOM of A or a SEPARATE cause? That decides
the reachable ceiling (e.g. gtap7_3x3: ~99% if O is downstream of pft, ~86% if independent)
— and today no tool answers it without spending the free-row experiment.

HOW (numerical, local, no full solve): seed at the GAMS point, perturb each bucket-A
variable by δ, and measure how much the bucket-O EQUATION residuals respond
(∂resid(eq_O)/∂var_A ≈ |Δresid|/(|var_A|·δ)). High sensitivity = O is reachable from A
(A perturbs O's defining equations) → O is a SYMPTOM. ~0 = A does not enter O's equations
locally → O looks INDEPENDENT.

THE DECLARED ASYMMETRY (so the tool never claims more than it measures — the #1 lesson):
  - sensitivity HIGH  → "O is a SYMPTOM of A"      : confidence FIRM. A local derivative
    that is already large only grows under the LARGE (branch-jump) move of a real fix.
  - sensitivity ~0    → "O looks INDEPENDENT"      : confidence TENTATIVE. A local
    derivative CANNOT see a non-linear response to a big move of A (a branch jump is not
    a small push). Absence of local sensitivity does NOT prove absence of propagation.
The two verdicts are emitted with DIFFERENT confidence — firm toward symptom, tentative
toward independent — never as if the tool measured the same thing in both directions.

KNOWN LIMIT (measured on gtap7_3x3, do NOT misread the output): this tool sees only
DIRECT coupling — it measures ∂resid(eq_O)/∂var_A, which is non-zero ONLY if var_A
appears LITERALLY in eq_O. CONTROL: perturbing pva moves eq_xfeq (pva is in it) by ~0.58,
so direct coupling IS detected. BUT pft/pva are NOT in the demand equations xa/xd/xm —
the real propagation pft→pva→xf→va→pa→xa is MULTI-LINK, and a one-step derivative gives
EXACTLY 0.0 for it. So a 0.0 here means "A is not in O's equation", NOT "O is independent
of A". The tentative-independent verdict is therefore the EXPECTED output for any O
separated from A by a chain — this tool CANNOT confirm "symptom via a long chain"; it can
only confirm "symptom via direct coupling" (firm) or "no direct coupling" (tentative).
To answer symptom-via-chain (the gtap7_3x3 bucket-O / 86%-vs-99% question) you need
TRANSITIVE reachability on the eq↔var graph (is there a path pft→…→xa?), a different tool,
OR run the fix. This tool is the direct-coupling layer, honest about that boundary.

Usage:
  uv run python scripts/gtap/diff_causal_propagation.py --dataset gtap7_3x3 --period shock
"""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import gams_levels  # noqa: E402
from _parity_json import (  # noqa: E402
    make_violation, run_tool, make_detection, make_prescription, with_diagnosis,
)
import diff_altertax as DA  # noqa: E402

DEFAULT_REFS = "/Users/marmol/proyectos2/equilibria_refs"

# upstream candidate cause (bucket A): the root-selection cluster
A_VARS = ["pft", "pf", "pfy", "pva"]
# downstream-suspect equations (bucket O): demand quantities
O_EQS = ["eq_xaa_activity", "eq_xaa_hhd", "eq_xaa_gov", "eq_xaa_inv",
         "eq_xda", "eq_xma", "eq_xds"]

# above this, the A→O coupling is strong → O is a symptom (FIRM)
SYMPTOM_THRESHOLD = 1e-3


def _default_gdx(dataset: str) -> Path:
    return Path(f"{DEFAULT_REFS}/{dataset}_altertax_cd/out_altertax_ifsub0.gdx")


def run_propagation_scan(dataset: str, gdx_path: Path, period: str, delta: float):
    from pyomo.environ import value as V
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
    DA.warmstart_from_gams(m, gdx_path, period)

    def _resid(con):
        body = V(con.body)
        lo = V(con.lower) if con.lower is not None else 0.0
        return body - lo

    # baseline residuals of all bucket-O equation cells
    o_cells = []
    for eqn in O_EQS:
        eq = getattr(m, eqn, None)
        if eq is None:
            continue
        for idx in eq:
            c = eq[idx]
            if c.active:
                o_cells.append((eqn, idx, c))
    base_resid = {(eqn, idx): _resid(c) for eqn, idx, c in o_cells}

    # for each A var cell, perturb and measure max |Δresid_O| / (|var|·δ)
    results = []  # (max_sens, a_var, a_cell, n_O_touched, worst_O)
    for vn in A_VARS:
        v = getattr(m, vn, None)
        if v is None:
            continue
        for a_idx in v:
            vd = v[a_idx]
            if vd.fixed or vd.value is None:
                continue
            v0 = float(vd.value)
            if abs(v0) < 1e-12:
                continue
            vd.set_value(v0 * (1.0 + delta))
            max_sens = 0.0
            worst_O = None
            n_touched = 0
            for eqn, idx, c in o_cells:
                r1 = _resid(c)
                dr = abs(r1 - base_resid[(eqn, idx)])
                sens = dr / (abs(v0) * delta + 1e-12)
                if sens > SYMPTOM_THRESHOLD:
                    n_touched += 1
                if sens > max_sens:
                    max_sens = sens
                    worst_O = (eqn, idx)
            vd.set_value(v0)  # restore
            results.append((max_sens, vn, a_idx, n_touched, worst_O))
    results.sort(reverse=True)
    return results, len(o_cells)


def _work(args) -> dict:
    gdx = args.gdx or _default_gdx(args.dataset)
    if not gdx.exists():
        return dict(status="error", period=args.period,
                    headline=f"reference GDX not found: {gdx}",
                    violations=[], meta={"error_kind": "gdx_not_found", "gdx": str(gdx)})

    results, n_o = run_propagation_scan(args.dataset, gdx, args.period, args.delta)
    max_overall = results[0][0] if results else 0.0
    is_symptom = max_overall > SYMPTOM_THRESHOLD

    # human-readable → stderr
    print(f"=== CAUSAL-PROPAGATION: does bucket O (demand qty) react to bucket A "
          f"(pft/pf/pfy/pva)? {args.dataset} {args.period} ===", file=sys.stderr)
    print(f"{'∂residO/∂A':>12}  {'A var':<8} {'cell':<22} {'O eqs touched':>14}", file=sys.stderr)
    for sens, vn, a_idx, n, worstO in results[:12]:
        print(f"{sens:>12.3e}  {vn:<8} {str(a_idx):<22} {n:>14}", file=sys.stderr)

    # ONE violation carrying the verdict with the DECLARED ASYMMETRY of confidence.
    if is_symptom:
        conf = "firm"
        verdict = "symptom"
        what = (f"perturbing bucket-A vars (pft/pf/pfy/pva) moves bucket-O demand-quantity "
                f"equation residuals (max ∂residO/∂A={max_overall:.2e} > {SYMPTOM_THRESHOLD:g}) "
                f"→ O is REACHABLE from A → O is a SYMPTOM of the A root-selection")
        note = ("FIRM toward symptom: a large local derivative only grows under the large "
                "branch-jump move of a real fix. Ceiling: if the A fix works, O closes too.")
    else:
        conf = "tentative"
        verdict = "independent"
        what = (f"perturbing bucket-A vars barely moves bucket-O residuals "
                f"(max ∂residO/∂A={max_overall:.2e} ≤ {SYMPTOM_THRESHOLD:g}) → O looks INDEPENDENT of A")
        note = ("TENTATIVE toward independent: a LOCAL derivative cannot see a non-linear "
                "response to a LARGE move of A (a branch jump is not a small push). Absence "
                "of local sensitivity does NOT prove O is independent — only the fix settles it.")

    v = make_violation("bucket_O_vs_A", [], "dresidO_dA", float(max_overall))
    v["verdict"] = verdict
    with_diagnosis(
        v,
        detection=make_detection(what=what, evidence=f"finite-diff δ={args.delta} at GAMS seed, "
                                 f"{n_o} O-eq cells × {len(A_VARS)} A vars", confidence=conf),
        prescription=make_prescription(
            how=("if symptom: the pft free-row fix should close O too; "
                 "if independent: O needs its own lead — " + note),
            validated_by=None,  # this tool measures local coupling, not the fix's effect
        ),
    )
    # status: this layer is informational about the gap structure; dirty=symptom found
    # (actionable: one fix closes both), clean=no local coupling (but tentative).
    status = "dirty" if is_symptom else "clean"
    headline = (f"causal-propagation ({args.period}): bucket O is "
                f"{'a SYMPTOM of A (FIRM, max ∂residO/∂A=' if is_symptom else 'INDEPENDENT of A (TENTATIVE, max ∂residO/∂A='}"
                f"{max_overall:.2e}) — {note.split(':')[0]}")
    return dict(status=status, period=args.period, headline=headline, violations=[v],
                meta={"gdx": str(gdx), "max_sensitivity": max_overall,
                      "symptom_threshold": SYMPTOM_THRESHOLD, "verdict": verdict,
                      "confidence": conf, "n_O_cells": n_o, "delta": args.delta})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, default=None)
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--delta", type=float, default=1e-4)
    args = ap.parse_args()
    return run_tool("diff_causal_propagation", args.dataset, lambda: _work(args),
                    period_hint=args.period)


if __name__ == "__main__":
    raise SystemExit(main())
