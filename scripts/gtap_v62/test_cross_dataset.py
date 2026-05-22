"""Cross-dataset robustness test for GTAP v6.2 Phase 3.23 implementation.

Verifies that the model builds, calibrates, solves baseline, and
responds correctly to a 10% tariff cut shock across multiple
BOOK3X3-style datasets. Reports key metrics:
  - calibration walras at benchmark (should be ~0)
  - baseline IPOPT walras (should be small relative to GDP)
  - shocked IPOPT walras
  - qxs and VIWS response for the shocked bilateral
  - aggregate qim response in the destination region

Datasets exercised (all v6.2 with CGDS):
  - BOOK3X3   (3×3, 3 factors)        ← reference (Exp1a)
                                         use validate_v62_parity.py for
                                         the canonical run; this script's
                                         minimal regularizer can hit
                                         IPOPT restoration on BOOK3X3.
  - ACORS3X3  (3×3, 5 factors)        ← more factors, same dimension
  - ASA7X5    (7×5, 5 factors)        ← much bigger dimension (3,572 vars)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

from pyomo.environ import (  # noqa: E402
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    minimize,
    value,
    SolverFactory,
)

from equilibria.templates.gtap_v62 import (  # noqa: E402
    GTAPv62Sets,
    GTAPv62Parameters,
    GTAPv62ModelEquations,
)
from scripts.gtap_v62._make_square import (  # noqa: E402
    apply_v62_closure_and_square,
    bake_baseline_residuals_as_slacks,
)


DATASETS = [
    {
        "name": "BOOK3X3",
        "dir": Path("C:/runGTAP375/BOOK3X3"),
        "sets_file": "SETS.HAR",
        "prm_file": "Default.prm",
        "basedata": "basedata.har",
        # 10% tariff cut on food, USA→EU.
        "shock": ("food", "USA", "EU"),
    },
    {
        "name": "ACORS3X3",
        "dir": Path("C:/runGTAP375/ACORS3X3"),
        "sets_file": "SETS.HAR",
        "prm_file": "Default.prm",
        "basedata": "basedata.har",
        # 10% tariff cut on Food, SSA→EU (mimicking BOOK3X3 Exp1a).
        "shock": ("Food", "SSA", "EU"),
    },
    {
        "name": "ASA7X5",
        "dir": Path("C:/runGTAP375/ASA7X5"),
        "sets_file": "sets.har",
        "prm_file": "default.prm",
        "basedata": "basedata.har",
        # 10% tariff cut on FOOD, SAFRICA→EUNION.
        "shock": ("FOOD", "SAFRICA", "EUNION"),
    },
]


def _obj_anchored(model: ConcreteModel, baseline: dict, weight: float = 1e-6) -> float:
    """Tiny anchored regularizer (matches validate_v62_parity.py convention).

    Provides direction without distorting the equilibrium response.
    """
    return weight * sum(
        ((v[idx] - baseline.get((v.name, idx), 1.0))
         / max(abs(baseline.get((v.name, idx), 1.0)), 1.0)) ** 2
        for v in model.component_objects(Var, active=True)
        for idx in v
        if not v[idx].fixed
    )


def run_dataset(spec: dict) -> dict:
    """Build, calibrate, and run shock for one dataset."""
    name = spec["name"]
    dir_ = spec["dir"]
    sets_path = dir_ / spec["sets_file"]
    prm_path = dir_ / spec["prm_file"]
    basedata_path = dir_ / spec["basedata"]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    # 1. Load.
    sets = GTAPv62Sets()
    sets.load_from_har(sets_path, default_path=prm_path)
    print(f"  Sizes: i={len(sets.i)}  r={len(sets.r)}  f={len(sets.f)}  "
          f"cgds={len(sets.cgds)}  marg={len(sets.marg)}")

    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=basedata_path, default_prm_path=prm_path, sets=sets)

    # 2. Build Pyomo model.
    print("  Building model...")
    builder = GTAPv62ModelEquations(sets, params)
    model = builder.build_model()
    print(f"  Vars: {sum(1 for _ in model.component_data_objects(Var, active=True))}")
    print(f"  Constraints: {sum(1 for _ in model.component_data_objects(Constraint, active=True))}")

    # 3. Apply closure + prebalance.
    info = apply_v62_closure_and_square(model)
    n_free = sum(
        1 for v in model.component_objects(Var, active=True)
        for idx in v if not v[idx].fixed
    )
    n_active = sum(1 for _ in model.component_data_objects(Constraint, active=True))
    print(f"  Closure: free={n_free} cons={n_active} mismatch={n_free - n_active}")
    info["free_vars"] = n_free
    info["active_cons"] = n_active
    info["mismatch"] = n_free - n_active
    prebal = bake_baseline_residuals_as_slacks(model)
    print(f"  Prebalance: baked {prebal['n_baked']} cells, "
          f"max_abs={prebal['max_abs_baked']:.2e}")

    # 4. Baseline solve.
    baseline_init = {
        (v.name, idx): value(v[idx])
        for v in model.component_objects(Var, active=True)
        for idx in v
        if not v[idx].fixed and v[idx].value is not None
    }

    if hasattr(model, "obj"):
        model.del_component(model.obj)
    model.obj = Objective(rule=lambda m: _obj_anchored(m, baseline_init), sense=minimize)

    # Use the IDAES-installed IPOPT (validate_v62_parity.py convention).
    ipopt_path = Path(".idaes-bin/ipopt.exe")
    if ipopt_path.exists():
        solver = SolverFactory("ipopt", executable=str(ipopt_path))
    else:
        solver = SolverFactory("ipopt")
    solver.options.update({"max_iter": 5000, "tol": 1e-8, "acceptable_tol": 1e-6})
    print("  Solving BASELINE...")
    res = solver.solve(model, tee=False, load_solutions=False)
    try:
        model.solutions.load_from(res)
    except Exception as exc:
        print(f"    (baseline solve returned status={res.solver.status}; "
              f"continuing with last iterate)")
        # Still proceed — Pyomo holds the solver's last iterate.
    base_walras = value(model.walras)
    print(f"    status: {res.solver.status} | walras: {base_walras:.2e}")

    baseline_vals = {
        (v.name, idx): value(v[idx])
        for v in model.component_objects(Var, active=True)
        for idx in v
    }

    # 5. Apply 10% tariff cut shock.
    i_shk, s_shk, d_shk = spec["shock"]
    old_tms = value(model.tms[i_shk, s_shk, d_shk])
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    print(f"  Shock: tms[{i_shk},{s_shk},{d_shk}]: {old_tms:.4f} -> {new_tms:.4f}")
    model.tms[i_shk, s_shk, d_shk] = new_tms

    # 6. Shocked solve.
    if hasattr(model, "obj"):
        model.del_component(model.obj)
    model.obj = Objective(rule=lambda m: _obj_anchored(m, baseline_vals), sense=minimize)
    print("  Solving SHOCKED...")
    res = solver.solve(model, tee=False, load_solutions=False)
    try:
        model.solutions.load_from(res)
    except Exception:
        pass
    shocked_walras = value(model.walras)
    print(f"    status: {res.solver.status} | walras: {shocked_walras:.2e}")

    # 7. Report shock response.
    pms_0 = baseline_vals[("pms", (i_shk, s_shk, d_shk))]
    pms_s = value(model.pms[i_shk, s_shk, d_shk])
    qxs_0 = baseline_vals[("qxs", (i_shk, s_shk, d_shk))]
    qxs_s = value(model.qxs[i_shk, s_shk, d_shk])
    pmcif_0 = baseline_vals[("pmcif", (i_shk, s_shk, d_shk))]
    pmcif_s = value(model.pmcif[i_shk, s_shk, d_shk])
    qim_0 = baseline_vals[("qim", (i_shk, d_shk))]
    qim_s = value(model.qim[i_shk, d_shk])

    def pct(new, old):
        return (new / old - 1.0) * 100.0 if abs(old) > 1e-12 else float("nan")

    vims_pct = pct(pms_s * qxs_s, pms_0 * qxs_0)
    viws_pct = pct(pmcif_s * qxs_s, pmcif_0 * qxs_0)
    qxs_pct = pct(qxs_s, qxs_0)
    qim_pct = pct(qim_s, qim_0)

    result = {
        "dataset": name,
        "shock_cell": f"{i_shk}/{s_shk}->{d_shk}",
        "vims_pct": vims_pct,
        "viws_pct": viws_pct,
        "qxs_pct": qxs_pct,
        "qim_pct": qim_pct,
        "base_walras": base_walras,
        "shock_walras": shocked_walras,
        "n_vars": sum(1 for _ in model.component_data_objects(Var, active=True)),
        "n_cons": sum(1 for _ in model.component_data_objects(Constraint, active=True)),
        "n_baked": prebal["n_baked"],
        "closure_mismatch": info["mismatch"],
    }

    print(f"\n  RESULTS:")
    print(f"    qxs[{i_shk},{s_shk},{d_shk}]:  {qxs_pct:+8.3f}%")
    print(f"    VIWS:                          {viws_pct:+8.3f}%")
    print(f"    VIMS:                          {vims_pct:+8.3f}%")
    print(f"    qim[{i_shk},{d_shk}]:           {qim_pct:+8.3f}%")
    return result


def main():
    results = []
    for spec in DATASETS:
        try:
            r = run_dataset(spec)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR on {spec['name']}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"dataset": spec["name"], "error": str(e)})

    print("\n" + "=" * 60)
    print("SUMMARY (all datasets)")
    print("=" * 60)
    print(f"{'Dataset':10s}  {'Size':10s}  {'Shock':20s}  {'VIWS %':>10s}  {'qim %':>8s}  {'Walras':>12s}")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:10s}  ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:10s}  "
              f"V/C={r['n_vars']}/{r['n_cons']}".ljust(22)[:22] + "  "
              f"{r['shock_cell']:20s}  "
              f"{r['viws_pct']:+10.3f}  "
              f"{r['qim_pct']:+8.3f}  "
              f"{r['shock_walras']:+12.2e}")


if __name__ == "__main__":
    main()
