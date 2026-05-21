"""GTAP v6.2 parity validation: Python implementation vs GEMPACK reference.

Compares the equilibria Python v6.2 model against ``gtap.exe`` (the
official GEMPACK v6.2 executable shipped with RunGTAP) on the BOOK3X3
dataset. Two comparison modes:

1. **Baseline structural validation** (always available):
   - Verifies that the Python variable initialization matches the SAM
     benchmark values exactly. This validates the calibration cascade
     without requiring a solver.

2. **Shock parity** (requires solver: IPOPT / PATH / path-capi):
   - Solves the Python baseline + shocked counterfactual.
   - Compares against ``Exp1a-upd.har`` (the post-shock SAM levels
     produced by ``gtap.exe`` running Experiment 1a — a 10% tariff
     cut on US food exports to the EU).

Output: a JSON report at ``runs/gtap_v62_parity/BOOK3X3_<scenario>.json``
with cell-by-cell diffs and aggregate pass/fail counters.

Usage (Phase 3 baseline structural check)::

    python scripts/gtap_v62/validate_v62_parity.py baseline \\
        --workdir runs/gtap_v62_parity/BOOK3X3_baseline

When a solver becomes available::

    python scripts/gtap_v62/validate_v62_parity.py shock \\
        --experiment Exp1a --solver ipopt \\
        --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Make scripts/ importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from equilibria.babel.har import read_har
from equilibria.templates.gtap_v62 import (
    GTAPv62ModelEquations,
    GTAPv62Parameters,
    GTAPv62Sets,
)

logger = logging.getLogger(__name__)


BOOK3X3_DIR = Path("C:/runGTAP375/BOOK3X3")


# ----------------------------------------------------------------------
# Comparison record types
# ----------------------------------------------------------------------


@dataclass
class CellDiff:
    """Per-cell comparison between two sources."""

    key: Tuple[str, ...]
    var_name: str
    python_value: float
    gempack_value: float

    @property
    def abs_diff(self) -> float:
        return abs(self.python_value - self.gempack_value)

    @property
    def rel_diff(self) -> float:
        denom = max(abs(self.gempack_value), 1e-12)
        return self.abs_diff / denom

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": list(self.key),
            "var": self.var_name,
            "py": self.python_value,
            "gp": self.gempack_value,
            "abs": self.abs_diff,
            "rel": self.rel_diff,
        }


@dataclass
class ParityReport:
    """Aggregated parity statistics for one variable family."""

    var_name: str
    n_cells: int = 0
    n_diverging: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    max_abs_key: Optional[Tuple[str, ...]] = None
    max_rel_key: Optional[Tuple[str, ...]] = None
    cells: List[CellDiff] = field(default_factory=list)

    def add(self, diff: CellDiff, abs_tol: float, rel_tol: float) -> None:
        self.n_cells += 1
        self.cells.append(diff)
        if diff.abs_diff > abs_tol and diff.rel_diff > rel_tol:
            self.n_diverging += 1
        if diff.abs_diff > self.max_abs:
            self.max_abs = diff.abs_diff
            self.max_abs_key = diff.key
        if diff.rel_diff > self.max_rel:
            self.max_rel = diff.rel_diff
            self.max_rel_key = diff.key

    def summary_dict(self, include_cells: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "var": self.var_name,
            "n_cells": self.n_cells,
            "n_diverging": self.n_diverging,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "max_abs_key": list(self.max_abs_key) if self.max_abs_key else None,
            "max_rel_key": list(self.max_rel_key) if self.max_rel_key else None,
        }
        if include_cells:
            d["cells"] = [c.as_dict() for c in self.cells]
        return d


# ----------------------------------------------------------------------
# Helpers — building the model and loading reference data
# ----------------------------------------------------------------------


def build_book3x3_model(
    dataset_dir: Path = BOOK3X3_DIR,
) -> Tuple[GTAPv62Sets, GTAPv62Parameters, Any]:
    """Build a fresh BOOK3X3 model. Returns (sets, params, pyomo_model)."""
    sets = GTAPv62Sets()
    sets.load_from_har(
        dataset_dir / "SETS.HAR",
        default_path=dataset_dir / "Default.prm",
    )
    params = GTAPv62Parameters()
    params.load_from_har(
        basedata_path=dataset_dir / "basedata.har",
        default_prm_path=dataset_dir / "Default.prm",
        sets=sets,
    )
    model = GTAPv62ModelEquations(sets, params).build_model()
    return sets, params, model


def har_to_dense_dict(
    har_path: Path,
    header: str,
) -> Dict[Tuple[str, ...], float]:
    """Read a HAR header and return ``{(label, ...): value}``."""
    data = read_har(har_path, select_headers=[header])
    if header not in data:
        return {}
    arr_obj = data[header]
    arr = np.asarray(arr_obj.array)
    labels_per_dim = [
        [str(lbl).strip() for lbl in dim]
        for dim in (arr_obj.set_elements or [])
    ]
    result: Dict[Tuple[str, ...], float] = {}
    if arr.ndim == 0:
        result[()] = float(arr.item())
        return result
    for idx, value in np.ndenumerate(arr):
        key = tuple(
            labels_per_dim[d][idx[d]] if d < len(labels_per_dim) else str(idx[d])
            for d in range(arr.ndim)
        )
        result[key] = float(value)
    return result


# ----------------------------------------------------------------------
# Baseline comparisons (Python init vs SAM benchmark)
# ----------------------------------------------------------------------


# Each entry maps a Pyomo variable name to the SAM HAR header (and an
# optional key permutation function when index orders differ).
_BASELINE_VAR_TO_HAR: List[Tuple[str, str]] = [
    # Direct value-flow headers — Pyomo var init = HAR value
    ("qfe", "VFM"),       # Note: var indices are (f, j, r); HAR is (f, j, r)
    ("qfd", "VDFM"),      # (i, j, r)
    ("qfm", "VIFM"),
    ("qpd", "VDPM"),      # (i, r)
    ("qpm", "VIPM"),
    ("qgd", "VDGM"),
    ("qgm", "VIGM"),
    ("qst", "VST"),       # (m, r)
]


def collect_baseline_diffs(
    model: Any,
    basedata_path: Path,
    abs_tol: float = 1.0,
    rel_tol: float = 1e-4,
) -> Dict[str, ParityReport]:
    """Compare Python variable INIT values against the SAM benchmark.

    For each (variable, HAR header) pair the function reads both sources
    and reports the cell-by-cell agreement. The variable init values are
    determined by ``GTAPv62ModelEquations`` calibration; if they don't
    match the SAM, the calibration code has drifted.
    """
    from pyomo.environ import value

    reports: Dict[str, ParityReport] = {}
    for var_name, header in _BASELINE_VAR_TO_HAR:
        if not hasattr(model, var_name):
            logger.warning("Model has no variable %r — skipping", var_name)
            continue
        var = getattr(model, var_name)
        har_values = har_to_dense_dict(basedata_path, header)
        report = ParityReport(var_name=var_name)
        for idx in var:
            key = idx if isinstance(idx, tuple) else (idx,)
            py_val = float(value(var[idx])) if var[idx].value is not None else 0.0
            gp_val = har_values.get(key, 0.0)
            diff = CellDiff(
                key=key, var_name=var_name,
                python_value=py_val, gempack_value=gp_val,
            )
            report.add(diff, abs_tol=abs_tol, rel_tol=rel_tol)
        reports[var_name] = report
    return reports


# ----------------------------------------------------------------------
# Equation residual reporting
# ----------------------------------------------------------------------


@dataclass
class EquationResidualReport:
    """Benchmark residual analysis per equation family."""

    eq_name: str
    n_cells: int
    max_abs: float
    sum_abs: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "eq": self.eq_name,
            "n_cells": self.n_cells,
            "max_abs": self.max_abs,
            "sum_abs": self.sum_abs,
        }


def collect_equation_residuals(model: Any) -> List[EquationResidualReport]:
    """Evaluate every Pyomo Constraint at its current state and report stats."""
    from pyomo.environ import Constraint, value as pyo_value

    out: List[EquationResidualReport] = []
    for c in model.component_objects(Constraint):
        max_abs = 0.0
        sum_abs = 0.0
        n = 0
        for idx in c:
            cobj = c[idx]
            body = pyo_value(cobj.body)
            if cobj.upper is not None:
                upper = pyo_value(cobj.upper)
                residual = body - upper
            elif cobj.lower is not None:
                lower = pyo_value(cobj.lower)
                residual = body - lower
            else:
                residual = body
            r_abs = abs(residual)
            max_abs = max(max_abs, r_abs)
            sum_abs += r_abs
            n += 1
        out.append(EquationResidualReport(
            eq_name=c.name, n_cells=n, max_abs=max_abs, sum_abs=sum_abs,
        ))
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def baseline_command(args: argparse.Namespace) -> int:
    """Run baseline structural validation (no solver required)."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BOOK3X3 model from {args.dataset_dir}...")
    sets, params, model = build_book3x3_model(Path(args.dataset_dir))
    print(f"  sets:    {sets.n_regions}r × {sets.n_commodities}i × {sets.n_factors}f")

    print("\nComparing variable init values to SAM benchmark (basedata.har):")
    reports = collect_baseline_diffs(
        model,
        Path(args.dataset_dir) / "basedata.har",
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )
    n_total_cells = 0
    n_total_diverging = 0
    print(f"\n{'variable':<10s} {'cells':>6s} {'diverging':>10s} {'max_abs':>14s} {'max_rel':>12s}")
    for var_name, report in reports.items():
        n_total_cells += report.n_cells
        n_total_diverging += report.n_diverging
        print(
            f"  {var_name:<8s} {report.n_cells:>6d} {report.n_diverging:>10d}"
            f" {report.max_abs:>14.4e} {report.max_rel:>12.4e}"
        )
    print(f"\nTOTAL: {n_total_diverging}/{n_total_cells} cells diverge above tolerance.")

    print("\nEquation residuals at benchmark:")
    eq_reports = collect_equation_residuals(model)
    eq_reports.sort(key=lambda r: r.max_abs, reverse=True)
    print(f"  {'equation':<18s} {'cells':>6s} {'max_abs':>14s} {'sum_abs':>14s}")
    for er in eq_reports[:15]:  # top 15 worst
        print(f"  {er.eq_name:<18s} {er.n_cells:>6d} {er.max_abs:>14.4e} {er.sum_abs:>14.4e}")

    # Save full report
    report_path = workdir / "baseline_parity_report.json"
    payload = {
        "dataset": args.dataset_dir,
        "n_total_cells": n_total_cells,
        "n_total_diverging": n_total_diverging,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "variables": {
            name: rep.summary_dict(include_cells=args.full)
            for name, rep in reports.items()
        },
        "equations": [er.as_dict() for er in eq_reports],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFull report: {report_path}")

    return 0 if n_total_diverging == 0 else 1


def shock_command(args: argparse.Namespace) -> int:
    """Solve baseline + shock and compare against GEMPACK reference.

    Phase 3 status: requires a working solver (IPOPT / PATH / path-capi).
    On Windows hosts without solvers installed, this command exits with
    status 2 and prints instructions.
    """
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Probe solver availability. The IDAES-installed IPOPT binary
    # lives at .idaes-bin/ipopt.exe on Windows; SolverFactory's
    # default probe won't find it without the explicit path.
    import pyomo.environ
    from pyomo.opt import SolverFactory
    use_path_capi = args.solver == "path-capi"
    available = False
    solver_obj = None
    if use_path_capi:
        # Defer availability check to runtime; the PATH dll + license
        # are validated on first call.
        available = True
    elif args.solver == "ipopt":
        idaes_ipopt = Path(".idaes-bin/ipopt.exe")
        if idaes_ipopt.exists():
            available = True
    if not available and not use_path_capi:
        try:
            solver_obj = SolverFactory(args.solver)
            available = solver_obj.available(False)
        except Exception:
            available = False

    if not available:
        print(
            f"ERROR: solver {args.solver!r} not available on this host.\n"
            f"  Shock parity requires a working solver (IPOPT, PATH, or\n"
            f"  the custom path-capi backend).  On Windows, install one\n"
            f"  via conda-forge:\n"
            f"      conda install -c conda-forge ipopt\n"
            f"  or use the GAMS PATH C-API integration documented in\n"
            f"  scripts/gtap/run_gtap.py.",
            file=sys.stderr,
        )
        return 2

    # --- Wire the full shock pipeline ----------------------------------
    from pyomo.environ import value, Objective, minimize, Var

    # The auto-square helper applies the canonical v6.2 closure +
    # identity equations + auto-fixes any dangling variable cells.
    sys.path.insert(0, str(Path(__file__).parent))
    from _make_square import apply_v62_closure_and_square

    sets, params, model = build_book3x3_model(Path(args.dataset_dir))
    closure_info = apply_v62_closure_and_square(model)
    print(f"Closure: free={closure_info['free_vars']} cons={closure_info['active_cons']} "
          f"mismatch={closure_info['mismatch']}")

    # Phase 3.7 reconciliation: KEEP the SAM-implicit ``to``.
    # The benchmark residuals after closure are dominated by:
    #   eq_qtm  ~6.6e4 (intra-region VTWR — s==d entries)
    #   eq_market ~2.3e4 (~1% SAM imbalance)
    #   eq_qo   ~6e-2  (output tax wedge — kept implicit via ``to``)
    # All other equations are at machine epsilon. Setting ``to=0``
    # (the previous Phase 3.6 strategy) inflates eq_market by 20×
    # because the SAM calibrates implicit ``to`` to ABSORB the vom/vop
    # wedge. Keeping ``to`` as-calibrated leaves the benchmark much
    # closer to feasibility and PATH converges to a usable equilibrium.

    if closure_info["mismatch"] != 0:
        print(
            f"WARNING: model has {closure_info['mismatch']} extra degenerate "
            f"degrees of freedom. Using benchmark-anchored objective for IPOPT."
        )

    # Capture init values for the regularizer objective
    init_values = {
        (v.name, idx): v[idx].value
        for v in model.component_objects(Var, active=True)
        for idx in v
        if not v[idx].fixed and v[idx].value is not None
    }

    # Tiny-weight regularizer (1e-6) — provides direction without
    # distorting the equilibrium response.
    def _obj(model, anchor: Dict, weight: float = 1e-6) -> Any:
        return weight * sum(
            ((v[idx] - anchor.get((v.name, idx), 1.0))
             / max(abs(anchor.get((v.name, idx), 1.0)), 1.0)) ** 2
            for v in model.component_objects(Var, active=True)
            for idx in v
            if not v[idx].fixed
        )

    # Build solver with the configured backend
    if use_path_capi:
        from _path_capi_solver import solve_v62_with_path_capi  # type: ignore

        license_string = os.environ.get("PATH_LICENSE_STRING") or args.path_license
        path_lib = os.environ.get("PATH_CAPI_LIBPATH") or args.path_lib
        lusol_lib = os.environ.get("PATH_CAPI_LIBLUSOL") or args.lusol_lib

        def _solve_path_capi(label: str) -> Any:
            print(f"\nSolving {label} with PATH C-API (licensed)...")
            res = solve_v62_with_path_capi(
                model,
                output=bool(os.environ.get("PATH_CAPI_VERBOSE")),
                license_string=license_string,
                path_lib=path_lib,
                lusol_lib=lusol_lib,
            )
            print(
                f"  term_code={res.termination_code} residual={res.residual:.2e} "
                f"walras={value(model.walras):.2e} "
                f"major={res.major_iterations} minor={res.minor_iterations}"
            )
            return res
    else:
        if args.solver == "ipopt":
            # IDAES-installed ipopt at .idaes-bin/ipopt.exe (Windows)
            ipopt_path = Path(".idaes-bin/ipopt.exe")
            if ipopt_path.exists():
                solver_obj = SolverFactory("ipopt", executable=str(ipopt_path))
            else:
                solver_obj = SolverFactory("ipopt")
            solver_obj.options.update({
                "max_iter": 5000,
                "tol": 1e-6,
                "print_level": 0,
                "nlp_scaling_method": "gradient-based",
            })
        else:
            solver_obj = SolverFactory(args.solver)

    # === BASELINE SOLVE ===
    if use_path_capi:
        _solve_path_capi("BASELINE")
    else:
        model.obj = Objective(rule=lambda mm: _obj(mm, init_values), sense=minimize)
        print("\nSolving BASELINE...")
        res = solver_obj.solve(model, tee=False)
        print(f"  status: {res.solver.status}, walras: {value(model.walras):.2e}")

    # Diagnostic: equation residuals at the solved state (top 10 by max_abs).
    print("\nBaseline equation residuals (top 10):")
    eq_reports = collect_equation_residuals(model)
    eq_reports.sort(key=lambda r: r.max_abs, reverse=True)
    for er in eq_reports[:10]:
        print(f"  {er.eq_name:<22s} max_abs={er.max_abs:>12.4e} sum_abs={er.sum_abs:>12.4e}")

    baseline = {
        (v.name, idx): value(v[idx])
        for v in model.component_objects(Var, active=True)
        for idx in v
    }

    # === SHOCK ===
    # 10% tariff cut on the bilateral US food → EU import duty.
    # GEMPACK Exp1a: ``Shock tms("food","usa","eu") = -10`` — that's
    # a -10% change in the POWER of the tariff (= (1+t)*0.9 - 1).
    old_tms = value(model.tms["food", "USA", "EU"])
    new_tms = (1.0 + old_tms) * 0.9 - 1.0
    model.tms["food", "USA", "EU"] = new_tms
    print(f"\nShock: tms[food,USA,EU]: {old_tms:.4f} -> {new_tms:.4f}")

    if use_path_capi:
        _solve_path_capi("SHOCKED")
    else:
        model.del_component(model.obj)
        model.del_component("obj_index")
        model.obj = Objective(rule=lambda mm: _obj(mm, baseline), sense=minimize)
        print("Solving SHOCKED...")
        res = solver_obj.solve(model, tee=False)
        print(f"  status: {res.solver.status}, walras: {value(model.walras):.2e}")
    shocked = {
        (v.name, idx): value(v[idx])
        for v in model.component_objects(Var, active=True)
        for idx in v
    }

    # === COMPARE vs GEMPACK Exp1a-upd.har ===
    upd_har_path = (
        Path("runs/gtap_v62_oracle") / f"BOOK3X3_{args.experiment}"
        / f"{args.experiment}-upd.har"
    )
    if not upd_har_path.exists():
        print(f"\nGEMPACK reference not found: {upd_har_path}")
        print(f"  Run: python scripts/gtap_v62/run_gempack_oracle.py "
              f"{args.experiment} --dataset-dir {args.dataset_dir} "
              f"--workdir runs/gtap_v62_oracle/BOOK3X3_{args.experiment}")
        return 2

    base_har_path = Path(args.dataset_dir) / "basedata.har"
    cmp_results = compare_shock_vs_gempack(
        baseline=baseline,
        shocked=shocked,
        base_har_path=base_har_path,
        upd_har_path=upd_har_path,
    )

    print(f"\n{'Cell':<28s} {'GEMPACK %':>10s} {'Python %':>10s} {'pp diff':>10s}")
    print("-" * 65)
    for row in cmp_results:
        print(f"  {row['label']:<26s} {row['gp_pct']:>+10.3f} {row['py_pct']:>+10.3f} "
              f"{row['py_pct'] - row['gp_pct']:>+10.3f}")

    report_path = workdir / f"shock_parity_{args.experiment}.json"
    payload = {
        "experiment": args.experiment,
        "shock_applied": {
            "var": "tms",
            "key": ["food", "USA", "EU"],
            "old": old_tms,
            "new": new_tms,
        },
        "closure": closure_info,
        "comparisons": cmp_results,
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nShock parity report: {report_path}")
    return 0


def compare_shock_vs_gempack(
    baseline: Dict,
    shocked: Dict,
    base_har_path: Path,
    upd_har_path: Path,
) -> List[Dict[str, Any]]:
    """Compute Python vs GEMPACK percent changes for canonical cells.

    The Python values are constructed as price-quantity products to
    match GEMPACK's value-aggregate (V*) headers in the SAM.
    """
    base_har = read_har(base_har_path)
    upd_har = read_har(upd_har_path)

    def pct(new: float, old: float) -> float:
        return (new / old - 1.0) * 100.0 if abs(old) > 1e-12 else 0.0

    def gp_change(header: str, idx_tuple: Tuple[int, ...]) -> float:
        base_arr = np.asarray(base_har[header].array)
        upd_arr = np.asarray(upd_har[header].array)
        return pct(upd_arr[idx_tuple], base_arr[idx_tuple])

    rows: List[Dict[str, Any]] = []

    # VIMS food USA -> EU (bilateral imports at market prices)
    py_pct = pct(
        shocked[("pms", ("food", "USA", "EU"))] * shocked[("qxs", ("food", "USA", "EU"))],
        baseline[("pms", ("food", "USA", "EU"))] * baseline[("qxs", ("food", "USA", "EU"))],
    )
    rows.append({"label": "VIMS food USA->EU",
                 "gp_pct": gp_change("VIMS", (0, 0, 1)), "py_pct": py_pct})

    # VIWS food USA -> EU (bilateral imports at world prices)
    py_pct = pct(
        shocked[("pmcif", ("food", "USA", "EU"))] * shocked[("qxs", ("food", "USA", "EU"))],
        baseline[("pmcif", ("food", "USA", "EU"))] * baseline[("qxs", ("food", "USA", "EU"))],
    )
    rows.append({"label": "VIWS food USA->EU",
                 "gp_pct": gp_change("VIWS", (0, 0, 1)), "py_pct": py_pct})

    # VXMD food USA -> EU (bilateral exports at market prices)
    py_pct = pct(
        shocked[("pe", ("food", "USA", "EU"))] * shocked[("qxs", ("food", "USA", "EU"))],
        baseline[("pe", ("food", "USA", "EU"))] * baseline[("qxs", ("food", "USA", "EU"))],
    )
    rows.append({"label": "VXMD food USA->EU",
                 "gp_pct": gp_change("VXMD", (0, 0, 1)), "py_pct": py_pct})

    # VDPM food EU (household domestic food in EU)
    py_pct = pct(
        shocked[("pds", ("food", "EU"))] * shocked[("qpd", ("food", "EU"))],
        baseline[("pds", ("food", "EU"))] * baseline[("qpd", ("food", "EU"))],
    )
    rows.append({"label": "VDPM food EU",
                 "gp_pct": gp_change("VDPM", (0, 1)), "py_pct": py_pct})

    # VIPM food EU (household imported food in EU)
    py_pct = pct(
        shocked[("pim", ("food", "EU"))] * shocked[("qpm", ("food", "EU"))],
        baseline[("pim", ("food", "EU"))] * baseline[("qpm", ("food", "EU"))],
    )
    rows.append({"label": "VIPM food EU",
                 "gp_pct": gp_change("VIPM", (0, 1)), "py_pct": py_pct})

    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GTAP v6.2 parity validator (Python vs GEMPACK)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser(
        "baseline",
        help="Structural validation: variable init vs SAM benchmark.",
    )
    base.add_argument(
        "--dataset-dir", default=str(BOOK3X3_DIR),
        help="Dataset directory (default: BOOK3X3).",
    )
    base.add_argument(
        "--workdir", required=True,
        help="Where to write the JSON parity report.",
    )
    base.add_argument(
        "--abs-tol", type=float, default=1.0,
        help="Absolute tolerance for diverging cells (default: 1.0).",
    )
    base.add_argument(
        "--rel-tol", type=float, default=1e-4,
        help="Relative tolerance for diverging cells (default: 1e-4).",
    )
    base.add_argument(
        "--full", action="store_true",
        help="Include per-cell diffs in the JSON report (large file).",
    )
    base.set_defaults(func=baseline_command)

    shock = sub.add_parser(
        "shock",
        help="Shock parity: solve Python + diff against GEMPACK Exp1a.",
    )
    shock.add_argument(
        "--experiment", default="Exp1a",
        help="GEMPACK experiment name (default: Exp1a).",
    )
    shock.add_argument(
        "--dataset-dir", default=str(BOOK3X3_DIR),
        help="Dataset directory (default: BOOK3X3).",
    )
    shock.add_argument(
        "--workdir", required=True,
        help="Where to write the JSON parity report.",
    )
    shock.add_argument(
        "--solver", default="ipopt",
        help="Solver backend: 'ipopt' (NLP), 'path-capi' (MCP), or any "
             "Pyomo SolverFactory name (default: ipopt).",
    )
    shock.add_argument(
        "--path-license", default=None,
        help="PATH license string (env: PATH_LICENSE_STRING). "
             "Required for n>300 (full BOOK3X3 has 582 free vars).",
    )
    shock.add_argument(
        "--path-lib", default="C:/GAMS/53/path52.dll",
        help="PATH shared library path (env: PATH_CAPI_LIBPATH).",
    )
    shock.add_argument(
        "--lusol-lib", default="C:/GAMS/53/lusol.dll",
        help="LUSOL shared library path (env: PATH_CAPI_LIBLUSOL).",
    )
    shock.set_defaults(func=shock_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
