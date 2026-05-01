"""
GAMS parity validation for GTAP Standard 7 (9x10).

Applies the GAMS-equivalent uniform 10% import tariff shock to all bilateral
pairs (imptx_new = (1+imptx_old)*1.1 - 1, matching GAMS tm.fx = tm.l*1.1),
warm-starts the shocked model from the nonlinear baseline solution, and prints
a delta table comparing Python results vs GAMS reference values.

With --n-steps N, the shock is applied in N equal increments using continuation
(each step multiplies the tariff power by 1.1^(1/N)), which improves convergence
for large shocks.

Usage:
    .venv/bin/python scripts/gtap/validate_gams_parity.py [--shock-factor 0.1] [--n-steps 1]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from pyomo.core import value
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig, build_gtap_contract
from run_gtap import _run_path_capi_nonlinear_full, _build_gtap_contract_with_calibration

GDX = Path(__file__).parent.parent.parent / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"

# GAMS reference deltas (×1e6, from NEOS run 18737509, tariff_comp.gms ifSUB=0)
GAMS_REF = {
    "regy":  {"EastAsia": -20_101},
    "gdpmp": {"EastAsia": -21_378},
    "pabs":  {"EastAsia": -696},
}

REGIONS = ["Oceania", "EastAsia", "SEAsia", "SouthAsia", "NAmerica",
           "LatinAmer", "EU_28", "MENA", "SSA", "RestofWorld"]


def _print_top_residuals(model, top_n=20):
    """Evaluate all active constraints and print the ones with largest absolute residuals."""
    from pyomo.environ import Constraint, value as pyo_value
    rows = []
    for con in model.component_data_objects(Constraint, active=True, descend_into=True):
        try:
            body_val = float(pyo_value(con.body, exception=False))
            lb = con.lb
            ub = con.ub
            if lb is not None and ub is not None and abs(float(lb) - float(ub)) < 1e-14:
                res = body_val - float(lb)
            elif lb is not None and body_val < float(lb):
                res = body_val - float(lb)
            elif ub is not None and body_val > float(ub):
                res = body_val - float(ub)
            else:
                res = 0.0
            rows.append((abs(res), res, con.name))
        except Exception:
            pass
    rows.sort(reverse=True)
    print(f"\n  Top {top_n} constraint residuals at PATH stopping point:")
    print(f"  {'Constraint':55s}  {'Residual':>14s}")
    print("  " + "-" * 72)
    for abs_r, r, name in rows[:top_n]:
        print(f"  {name[:55]:55s}  {r:+14.6e}")


def get_dict(model, name):
    v = getattr(model, name, None)
    if v is None:
        return {}
    return {str(k): float(value(v[k])) for k in v}


def copy_solution(src_model, dst_model):
    from pyomo.environ import Var
    for comp in src_model.component_objects(Var, active=True):
        dst_comp = getattr(dst_model, comp.name, None)
        if dst_comp is None:
            continue
        for idx in comp:
            try:
                v = float(value(comp[idx]))
                lb = dst_comp[idx].lb
                ub = dst_comp[idx].ub
                if lb is not None and v < float(lb):
                    v = float(lb)
                if ub is not None and v > float(ub):
                    v = float(ub)
                dst_comp[idx].set_value(v)
            except Exception:
                pass


def print_delta_table(var_name, bl_dict, sh_dict, scale=1e6):
    ref = GAMS_REF.get(var_name, {})
    print(f"\n  {var_name} deltas (×1e6):")
    print(f"  {'Region':14s} {'Python':>10s}  {'GAMS':>10s}  {'Diff':>10s}  {'%err':>7s}")
    print("  " + "-" * 58)
    for r in REGIONS:
        delta = (sh_dict.get(r, 0) - bl_dict.get(r, 0)) * scale
        gams = ref.get(r)
        if gams is not None:
            diff = delta - gams
            pct = diff / abs(gams) * 100 if gams != 0 else float("inf")
            print(f"  {r:14s} {delta:+10.0f}  {gams:+10.0f}  {diff:+10.0f}  {pct:+6.1f}%")
        else:
            print(f"  {r:14s} {delta:+10.0f}  {'—':>10s}  {'—':>10s}  {'—':>7s}")


def main():
    parser = argparse.ArgumentParser(description="GAMS parity validation for GTAP 9x10")
    parser.add_argument("--shock-factor", type=float, default=0.10,
                        help="Tariff shock factor (default 0.10 = 10%%)")
    parser.add_argument("--n-steps", type=int, default=1,
                        help="Number of continuation steps (default 1 = single shot). "
                             "Use >1 for large shocks that fail to converge in one step. "
                             "Each step multiplies the tariff power by (1+factor)^(1/n_steps).")
    args = parser.parse_args()

    factor = args.shock_factor
    n_steps = max(1, args.n_steps)

    print(f"GDX: {GDX}")
    if not GDX.exists():
        print("ERROR: GDX file not found.")
        sys.exit(1)

    if n_steps > 1:
        print(f"Shock strategy: {n_steps}-step continuation")
        print(f"  Each step multiplies tariff power by (1+{factor})^(1/{n_steps}) = {(1+factor)**(1/n_steps):.6f}")
    else:
        print(f"Shock strategy: single shot (factor={factor:.0%})")

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")

    # ── BASELINE ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Solving baseline (nonlinear)")
    print("=" * 60)

    base_params = GTAPParameters()
    base_params.load_from_gdx(GDX)
    base_eq = GTAPModelEquations(base_params.sets, base_params, contract.closure)
    base_model = base_eq.build_model()

    bl_result = _run_path_capi_nonlinear_full(
        base_model, base_params,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=contract.closure,
        equation_scaling=True,
    )
    bl_res = bl_result.get("residual", float("nan"))
    bl_code = bl_result.get("termination_code", -1)
    print(f"  Baseline residual: {bl_res:.2e}  termination_code={bl_code}")
    if bl_code not in (1, 2):
        print("  WARNING: baseline did not converge cleanly")

    bl_regy  = get_dict(base_model, "regy")
    bl_gdpmp = get_dict(base_model, "gdpmp")
    bl_pabs  = get_dict(base_model, "pabs")
    print(f"  regy[EastAsia]  = {bl_regy.get('EastAsia', float('nan')):.6f}")
    print(f"  gdpmp[EastAsia] = {bl_gdpmp.get('EastAsia', float('nan')):.6f}")

    # ── SHOCKED (with optional continuation) ─────────────────────────────────
    print(f"\n{'='*60}")
    if n_steps > 1:
        print(f"STEP 2: Applying shock in {n_steps} continuation steps")
    else:
        print(f"STEP 2: Applying GAMS-equivalent shock (factor={factor:.0%})")
    print(f"  formula: imptx_new = (1 + imptx_old) * {1+factor} - 1")
    print("=" * 60)

    # Track the "current" solution model across continuation steps
    prev_model = base_model
    sh_result = None
    sh_res = float("nan")
    sh_code = -1

    # Per-step incremental multiplier for tariff power
    # After k steps: imptx = (1+imptx_original) * (1+factor)^(k/n_steps) - 1
    step_multiplier = (1.0 + factor) ** (1.0 / n_steps)

    # Load original GDX params once; we'll track current imptx values per step
    shock_params = GTAPParameters()
    shock_params.load_from_gdx(GDX)

    # Current tariff values (updated per step)
    current_imptx = {k: float(v) for k, v in shock_params.taxes.imptx.items()}

    for step in range(1, n_steps + 1):
        if n_steps > 1:
            print(f"\n  -- Continuation step {step}/{n_steps} "
                  f"(cumulative power multiplier = {step_multiplier**step:.6f}) --")

        # Load fresh params and apply cumulative shock from original values
        step_params = GTAPParameters()
        step_params.load_from_gdx(GDX)

        n_shocked = 0
        for key in list(step_params.taxes.imptx.keys()):
            old_val = float(step_params.taxes.imptx[key])
            # Cumulative: (1+original) * (1+factor)^(step/n_steps) - 1
            step_params.taxes.imptx[key] = (1.0 + old_val) * (step_multiplier ** step) - 1.0
            n_shocked += 1

        if step == 1:
            print(f"  Shocked {n_shocked} bilateral imptx pairs")

        step_eq = GTAPModelEquations(
            step_params.sets, step_params, contract.closure, is_counterfactual=True
        )
        step_model = step_eq.build_model()

        if step == 1:
            print(f"  Warm-starting step {step} from baseline solution...")
        else:
            print(f"  Warm-starting step {step} from step {step-1} solution...")
        copy_solution(prev_model, step_model)

        step_result = _run_path_capi_nonlinear_full(
            step_model, step_params,
            enforce_post_checks=False,
            strict_path_capi=False,
            closure_config=contract.closure,
            equation_scaling=True,
        )
        step_res = step_result.get("residual", float("nan"))
        step_code = step_result.get("termination_code", -1)
        print(f"  Step {step} residual: {step_res:.2e}  termination_code={step_code}")
        if step_code not in (1, 2):
            print(f"  WARNING: step {step} did not converge cleanly")

        prev_model = step_model
        sh_result = step_result
        sh_res = step_res
        sh_code = step_code

    shock_model = prev_model

    if sh_code not in (1, 2):
        print("\n  WARNING: shocked model did not converge cleanly — deltas are directional only")
        _print_top_residuals(shock_model, top_n=20)

    sh_regy  = get_dict(shock_model, "regy")
    sh_gdpmp = get_dict(shock_model, "gdpmp")
    sh_pabs  = get_dict(shock_model, "pabs")

    # ── DELTAS ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Delta comparison vs GAMS reference")
    print("=" * 60)

    print_delta_table("regy",  bl_regy,  sh_regy)
    print_delta_table("gdpmp", bl_gdpmp, sh_gdpmp)
    print_delta_table("pabs",  bl_pabs,  sh_pabs)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    py_ea_regy = (sh_regy.get("EastAsia", 0) - bl_regy.get("EastAsia", 0)) * 1e6
    gams_ea_regy = GAMS_REF["regy"]["EastAsia"]
    err_pct = (py_ea_regy - gams_ea_regy) / abs(gams_ea_regy) * 100

    if abs(err_pct) < 10:
        assessment = "PASS"
    elif abs(err_pct) < 25:
        assessment = "WARN"
    else:
        assessment = "FAIL"

    print(f"\n{'='*60}")
    print("GAMS Parity Validation Summary")
    print("=" * 60)
    print(f"Baseline residual:   {bl_res:.2e}  (code={bl_code})")
    print(f"Shocked residual:    {sh_res:.2e}  (code={sh_code})")
    if n_steps > 1:
        print(f"Continuation steps:  {n_steps}")
    print(f"\nKey benchmark: EastAsia regy delta")
    print(f"  Python:  {py_ea_regy:+.0f}")
    print(f"  GAMS:    {gams_ea_regy:+.0f}")
    print(f"  Error:   {err_pct:+.1f}%")
    print(f"\nAssessment: {assessment}")
    print("=" * 60)


if __name__ == "__main__":
    main()
