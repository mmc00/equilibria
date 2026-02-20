#!/usr/bin/env python3
"""
Check parity between Python initialized variables and GAMS benchmark levels.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GAMS parity for initialized solver variables.")
    parser.add_argument(
        "--init-mode",
        choices=["gams", "excel"],
        default="excel",
        help="Initialization mode for solver state.",
    )
    parser.add_argument(
        "--sam-file",
        default="/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/SAM-V2_0.gdx",
        help="Path to SAM GDX file.",
    )
    parser.add_argument(
        "--gams-results-gdx",
        default="src/equilibria/templates/reference/pep2/scripts/Results.gdx",
        help="Path to Results.gdx for gams levels.",
    )
    parser.add_argument(
        "--gams-results-slice",
        choices=["base", "sim1"],
        default="base",
        help="Slice from Results.gdx for gams levels.",
    )
    args = parser.parse_args()

    sam_file = Path(args.sam_file)
    state = PEPModelCalibrator(sam_file=sam_file).calibrate()
    solver = PEPModelSolver(
        calibrated_state=state,
        init_mode=args.init_mode,
        gams_results_gdx=args.gams_results_gdx,
        gams_results_slice=args.gams_results_slice,
        sam_file=sam_file,
    )
    vars_ = solver._create_initial_guess()
    residuals = solver.equations.calculate_all_residuals(vars_)

    checks: list[tuple[str, float, float, float]] = []

    def add(name: str, py_val: float, gams_val: float) -> None:
        checks.append((name, py_val, gams_val, py_val - gams_val))

    for i in solver.sets.get("I", []):
        add(f"PD[{i}] vs PDO", vars_.PD.get(i, 0), state.trade.get("PDO", {}).get(i, 0))
        add(f"PM[{i}] vs PMO", vars_.PM.get(i, 0), state.trade.get("PMO", {}).get(i, 0))
        add(f"PC[{i}] vs PCO", vars_.PC.get(i, 0), state.trade.get("PCO", {}).get(i, 0))
        add(f"PL[{i}] vs PLO", vars_.PL.get(i, 0), state.trade.get("PLO", {}).get(i, 0))
        add(f"PWM[{i}] vs PWMO", vars_.PWM.get(i, 0), state.trade.get("PWMO", {}).get(i, 0))

    for j in solver.sets.get("J", []):
        add(f"PP[{j}] vs PPO", vars_.PP.get(j, 0), state.production.get("PPO", {}).get(j, 0))
        add(f"PVA[{j}] vs PVAO", vars_.PVA.get(j, 0), state.production.get("PVAO", {}).get(j, 0))
        add(f"PCI[{j}] vs PCIO", vars_.PCI.get(j, 0), state.production.get("PCIO", {}).get(j, 0))

    add("TIWT vs TIWTO", vars_.TIWT, state.income.get("TIWTO", 0))
    add("TIKT vs TIKTO", vars_.TIKT, state.income.get("TIKTO", 0))
    add("TPRODN vs TPRODNO", vars_.TPRODN, state.income.get("TPRODNO", 0))
    add("YG vs YGO", vars_.YG, state.income.get("YGO", 0))
    add("GDP_IB vs GDP_IBO", vars_.GDP_IB, state.gdp.get("GDP_IBO", 0))
    add("SROW vs -CABO", vars_.SROW, -state.income.get("CABO", 0))

    checks.sort(key=lambda x: abs(x[3]), reverse=True)
    mismatches = [c for c in checks if abs(c[3]) > 1e-9]

    print("=" * 72)
    print(f"GAMS PARITY CHECK (init_mode={args.init_mode})")
    print("=" * 72)
    print(f"Variable mismatches > 1e-9: {len(mismatches)} / {len(checks)}")
    print()
    print("Top variable differences:")
    for name, py_val, gams_val, diff in checks[:15]:
        print(f"  {name:<30} py={py_val:>12.6f} gams={gams_val:>12.6f} diff={diff:>12.6f}")

    rvals = list(residuals.values())
    print()
    print(f"Residual RMS: {_rms(rvals):.6e}")
    print(f"Residual MAX: {max(abs(v) for v in rvals):.6e}")
    print("Top residuals:")
    for eq, val in sorted(residuals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]:
        print(f"  {eq:<30} {val:>12.6e}")


if __name__ == "__main__":
    main()
