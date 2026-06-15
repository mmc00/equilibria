"""GAMS-faithful homotopy driver for the altertax +10% import-tariff shock.

WHY THIS EXISTS
---------------
Under the forced-CD altertax run (sigmav=sigmap=1), pvaeq/pxeq degenerate to
tautologies, so the activity output price `px` (and `pva` downstream) is an
UNANCHORED degree of freedom. A single-jump solve lets PATH slide px/pva ~1.7%/4.7%
off the GAMS branch, capping the match at ~78.7%. Fixing px at GAMS values proves
it: match jumps to 98.65% (pva auto-lands at the GAMS value). See project memory
`project_gtap7_3x3_pva_is_sole_dof`.

GAMS avoids the slide with a two-phase procedure (model_altertax_ifsub*.gms:3955-3994):
  Phase 1 RAMP: raise imptx base→×1.1 in card(hstep)=30 steps, solving each (keeps
                px near 1.0 along the path so PATH stays on the right branch).
  Phase 2 CLEAN-UP: at ×1.1, loop re-seeding the lagging CD definition vars before
                each solve to a fixed point — the key one is the pva inversion
                pva=(px/pnd^and)^(1/ava) (reproduces GAMS pva to 1e-8).

Python bakes imptx into the equations at build time, so the ramp REBUILDS the
altertax model each step. CRUCIALLY it reuses diff_altertax.build_altertax_models
(the betaCal + phiP=1 + regy-unfix recipe) + warmstart_from_gams — a NAKED altertax
build does NOT converge (even at mult=1.0). The ramp starts from a CONVERGED check
(mult=1.0) warm-started from the GAMS check period, exactly as GAMS enters the ramp
from its just-solved check.

Usage:
    uv run python scripts/gtap/homotopy_shock.py --dataset gtap7_3x3 \\
        --gdx /path/to/out_altertax_ifsub0.gdx --ramp-steps 30 --cleanup-steps 5
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import gams_levels, list_populated_vars  # type: ignore
import diff_altertax as DA  # reuse the exact altertax model-construction recipe


def _build_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location("run_gtap", str(ROOT / "scripts" / "gtap" / "run_gtap.py"))
    mod = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_homotopy_shock(dataset: str, gdx_path: Path, ramp_steps: int,
                       cleanup_steps: int, verbose: bool = True):
    """Two-phase GAMS-faithful shock. Returns (model, params, result)."""
    from pyomo.environ import value as V
    from equilibria.templates.gtap import GTAPParameters, GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

    run_gtap = _build_run_gtap()

    data_dir = ROOT / "datasets" / dataset
    p_b_raw = GTAPParameters()
    p_b_raw.load_from_har(basedata_path=data_dir / "basedata.har", sets_path=data_dir / "sets.har",
                          default_path=data_dir / "default.prm", baserate_path=data_dir / "baserate.har")
    res = list(p_b_raw.sets.r)[-1]
    base_clo = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False, if_sub=False, numeraire="pnum")
    alt_clo = GTAPClosureConfig(name="altertax", closure_type="MCP", capital_mobility="mobile",
        fix_endowments=False, fix_taxes=True, fix_technology=True, if_sub=False, numeraire="pnum")

    def solve(m, p_s, hint):
        return run_gtap._run_path_capi_nonlinear_full(
            m, p_s, enforce_post_checks=False, strict_path_capi=False,
            closure_config=alt_clo, equation_scaling=True, solution_hint=hint)

    # Build a check-style altertax model at a given imptx multiplier, reusing the
    # diff_altertax recipe. mult=1.0 → the check (no shock). The GAMS warm-start
    # PERIOD is 'check' for mult==1.0, else 'shock' (closest available reference).
    def build_at(mult: float):
        p_raw = copy.deepcopy(p_b_raw)
        if mult != 1.0:
            for k in list(p_raw.taxes.imptx.keys()):
                p_raw.taxes.imptx[k] = float(p_raw.taxes.imptx[k] or 0.0) * mult
        _m_b, p_alt, m_chk = DA.build_altertax_models(p_raw, res, base_clo, alt_clo)
        return m_chk, p_alt

    # ---- PHASE 0: converged check anchor (mult=1.0, GAMS check warm-start) ----
    m, p_s = build_at(1.0)
    DA.warmstart_from_gams(m, gdx_path, "check")
    r = solve(m, p_s, GTAPVariableSnapshot.from_python_model(m))
    if verbose:
        print(f"  [check mult=1.0] code={r.get('termination_code')} resid={r.get('residual'):.2e} "
              f"pva[USA,Mnfcs]={V(m.pva['USA','Mnfcs']):.5f}")
    prev_snap = GTAPVariableSnapshot.from_python_model(m)

    # ---- PHASE 1: ramp imptx 1.0 → ×1.1, warm-start from previous step ----
    for step in range(1, ramp_steps + 1):
        mult = 1.0 + 0.1 * (step / ramp_steps)
        m, p_s = build_at(mult)
        r = solve(m, p_s, prev_snap)
        prev_snap = GTAPVariableSnapshot.from_python_model(m)
        if verbose and (step in (1, max(1, ramp_steps // 2), ramp_steps)
                        or r.get("termination_code") != 1):
            print(f"  [ramp {step}/{ramp_steps}] mult={mult:.4f} code={r.get('termination_code')} "
                  f"resid={r.get('residual'):.2e} pva[USA,Mnfcs]={V(m.pva['USA','Mnfcs']):.5f}")

    # ---- PHASE 2: clean-up re-seed pnd → pva inversion, re-solve to fixed point ----
    for cs in range(1, cleanup_steps + 1):
        for _r in p_s.sets.r:
            for _a in p_s.sets.a:
                try:
                    if hasattr(m, "pnd"):
                        pnd = 1.0
                        for _i in p_s.sets.i:
                            io = float(V(m.io[_r, _i, _a])) if hasattr(m, "io") else 0.0
                            if io <= 0.0:
                                continue
                            lam = float(V(m.lambdaio[_r, _i, _a])) if hasattr(m, "lambdaio") else 1.0
                            pnd *= (float(V(m.pa[_r, _i, _a])) / max(lam, 1e-12)) ** io
                        if not m.pnd[_r, _a].fixed:
                            m.pnd[_r, _a].set_value(max(pnd, 1e-8))
                except Exception:
                    pass
                try:
                    av = float(V(m.ava_param[_r, _a])) if hasattr(m, "ava_param") else 0.0
                    if av <= 1e-6:
                        continue
                    an = float(V(m.and_param[_r, _a])) if hasattr(m, "and_param") else 0.0
                    px = float(V(m.px[_r, _a]))
                    pnd = float(V(m.pnd[_r, _a])) if hasattr(m, "pnd") else 1.0
                    pva = (px / max(pnd ** an, 1e-12)) ** (1.0 / av)
                    if hasattr(m, "pva") and not m.pva[_r, _a].fixed:
                        m.pva[_r, _a].set_value(max(pva, 1e-8))
                except Exception:
                    pass
        r = solve(m, p_s, GTAPVariableSnapshot.from_python_model(m))
        if verbose:
            print(f"  [cleanup {cs}/{cleanup_steps}] code={r.get('termination_code')} "
                  f"resid={r.get('residual'):.2e} pva[USA,Mnfcs]={V(m.pva['USA','Mnfcs']):.5f}")

    return m, p_s, r


def _score(model, gdx_path: Path, tol: float = 1e-3):
    from pyomo.environ import value as V
    match = tot = 0
    worst = []
    for vn in list_populated_vars(gdx_path):
        gv = gams_levels(gdx_path, vn)
        pn = DA.GAMS_TO_PY_NAME.get(vn, vn)
        pv = getattr(model, pn, None)
        if pv is None:
            pv = getattr(model, vn, None)
        if pv is None:
            pv = getattr(model, vn.lower(), None)
        if pv is None:
            continue
        for gk, gval in gv.items():
            if not (isinstance(gk, tuple) and gk[-1] == "shock"):
                continue
            pk = tuple(DA._strip_set_prefix(k) for k in gk[:-1])
            try:
                now = float(V(pv[pk] if len(pk) > 1 else pv[pk[0]]))
                tot += 1
                rel = abs(now - float(gval)) / (abs(float(gval)) + 1e-9)
                if rel < tol:
                    match += 1
                else:
                    worst.append((rel, pn, pk, float(gval), now))
            except Exception:
                pass
    worst.sort(reverse=True)
    return match, tot, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--gdx", type=Path, required=True)
    ap.add_argument("--ramp-steps", type=int, default=30)
    ap.add_argument("--cleanup-steps", type=int, default=5)
    args = ap.parse_args()

    print(f"=== Homotopy shock: {args.dataset}  ramp={args.ramp_steps} cleanup={args.cleanup_steps} ===")
    m, p, r = run_homotopy_shock(args.dataset, args.gdx, args.ramp_steps, args.cleanup_steps)
    match, tot, worst = _score(m, args.gdx)
    print(f"\nFINAL: code={r.get('termination_code')} resid={r.get('residual'):.3e}")
    print(f"Match vs GAMS shock: {match}/{tot} = {100*match/max(tot,1):.2f}%")
    print("Top remaining diffs:")
    for rel, pn, pk, gv, nv in worst[:15]:
        print(f"  {rel*100:6.2f}%  {pn:<10} {str(pk):<28} g={gv:.5f} p={nv:.5f}")


if __name__ == "__main__":
    main()
