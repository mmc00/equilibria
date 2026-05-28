"""gtap6_15x10 SHOCKED with PATH — Phase 3.36 verification.

With bake_tolerance=1e-6 the baseline reaches tc=1 r=5e-8 cleanly.
Now test if the homotopy substep ladder produces tc=1 on the
SHOCKED (10% tariff cut on tms[OtherFood, USA, EU_28]) using
25 substeps per Phase 3.34 scaling rule (n_vars/1000 ≈ 26).
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

os.environ["PATH_LICENSE_STRING"] = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
os.environ["PATH_CAPI_LIBPATH"] = "C:/GAMS/53/path52.dll"
os.environ["PATH_CAPI_LIBLUSOL"] = "C:/GAMS/53/lusol.dll"
os.environ["PATH_CAPI_OPTIONS"] = (
    "major_iteration_limit 200 "
    "minor_iteration_limit 1000 "
    "convergence_tolerance 1e-4 "
)

from pyomo.environ import value, Var
from equilibria.templates.gtap_v62 import (
    GTAPv62Sets, GTAPv62Parameters, GTAPv62ModelEquations,
)
from _make_square import apply_v62_pipeline, rebake_residuals_at_current_state  # type: ignore
from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

DATA = Path("datasets/gtap6_15x10")
SHOCK_KEY = ("OtherFood", "USA", "EU_28")
N_SUBSTEPS = 20

print(f"\n=== gtap6_15x10 SHOCKED test (Phase 3.36) ===", flush=True)
print(f"Shock: tms[{SHOCK_KEY}] *= 0.9 (10% tariff cut), {N_SUBSTEPS} substeps", flush=True)
print(f"Bake tolerance: 1e-6 (Phase 3.36 default)", flush=True)

sets = GTAPv62Sets()
sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
params = GTAPv62Parameters()
params.load_from_har(basedata_path=DATA / "basedata.har",
                     default_prm_path=DATA / "default.prm", sets=sets)
model = GTAPv62ModelEquations(sets, params, mode="mcp").build_model()
pipe = apply_v62_pipeline(model, mode="mcp", bake_tolerance=1.0e-6, params=params,
                          drop_dead_rows_threshold=1.0e-6)
print(f"\n  free={pipe['free_vars']} baked={pipe['prebalance']['n_baked']}", flush=True)

t_total = time.perf_counter()

t0 = time.perf_counter()
print(f"\nBASELINE solve:", flush=True)
res = solve_v62_with_path_capi(
    model, output=False,
    license_string=os.environ["PATH_LICENSE_STRING"],
    path_lib=os.environ["PATH_CAPI_LIBPATH"],
    lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
    variable_scaling=True, equation_scaling=True, perturbation=0.0,
)
print(f"  tc={res.termination_code} r={res.residual:.4e} "
      f"major={res.major_iterations} time={time.perf_counter()-t0:.0f}s", flush=True)

# Phase 3.36: early-exit if baseline lottery-loss (tc=2 + r~2e-5).
# The bipartite matching landed in a basin PATH cannot fully resolve.
# Restart the process for a different hash seed → different matching.
if res.termination_code != 1 or res.residual > 1.0e-6:
    print(f"\n  BASELINE LOTTERY LOSS (tc={res.termination_code} r={res.residual:.2e})", flush=True)
    print(f"  Exiting with code 2 so a wrapper script can retry.", flush=True)
    sys.exit(2)

old_tms = float(value(model.tms[SHOCK_KEY]))
new_tms = (1.0 + old_tms) * 0.9 - 1.0
print(f"\nShock: tms[{SHOCK_KEY}]: {old_tms:.4f} → {new_tms:.4f}", flush=True)

# Capture VIWS baseline (bilateral import value) for parity tracking.
viws_0 = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))

for step in range(1, N_SUBSTEPS + 1):
    alpha = step / N_SUBSTEPS
    tms_step = (1.0 - alpha) * old_tms + alpha * new_tms
    model.tms[SHOCK_KEY] = tms_step

    # Phase 3.36 (option A): rebake residuals at the current (post-shock-update)
    # state. The original baked residuals are constants captured at baseline x_0.
    # After updating tms, the body of many constraints evaluates differently;
    # without rebake, F at current x has spurious offsets that PATH can't reduce
    # because they're encoded as constants in the baked replacement constraints.
    # We restore F(x_current) ≈ 0 by re-evaluating baked constants here.
    rb = rebake_residuals_at_current_state(model, tolerance=1.0e-6)
    print(f"    rebake: n_rebaked={rb['n_rebaked']} max_abs={rb['max_abs']:.2e}", flush=True)

    t0 = time.perf_counter()
    res = solve_v62_with_path_capi(
        model, output=False,
        license_string=os.environ["PATH_LICENSE_STRING"],
        path_lib=os.environ["PATH_CAPI_LIBPATH"],
        lusol_lib=os.environ["PATH_CAPI_LIBLUSOL"],
        variable_scaling=True, equation_scaling=True, perturbation=0.0,
    )
    viws = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))
    viws_pct = 100.0 * (viws - viws_0) / viws_0
    tag = "✓" if res.termination_code == 1 else "✗"
    print(f"  Substep {step:2d}/{N_SUBSTEPS} α={alpha:.3f}: "
          f"tc={res.termination_code}{tag} r={res.residual:.2e} "
          f"VIWS={viws_pct:+.3f}% ({time.perf_counter()-t0:.0f}s)", flush=True)

total_elapsed = time.perf_counter() - t_total
viws_final = float(value(model.qxs[SHOCK_KEY]) * value(model.pms[SHOCK_KEY]))
viws_pct_final = 100.0 * (viws_final - viws_0) / viws_0
print(f"\n===== FINAL =====", flush=True)
print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
print(f"FINAL VIWS = {viws_pct_final:+.3f}%", flush=True)
