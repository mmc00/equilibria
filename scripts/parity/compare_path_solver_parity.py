#!/usr/bin/env python
"""Compare a small MCP solved with PATHAMPL (Pyomo PATH) vs path-capi-python."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")


def _solve_with_pathampl(
    *,
    output: bool,
) -> dict[str, Any]:
    from pyomo.environ import ConcreteModel, Set, SolverFactory, Var, value
    from pyomo.mpec import Complementarity, complements

    plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]
    capacity = {"seattle": 350.0, "san-diego": 600.0}
    demand = {"new-york": 325.0, "chicago": 300.0, "topeka": 275.0}
    distance = {
        ("seattle", "new-york"): 2.5,
        ("seattle", "chicago"): 1.7,
        ("seattle", "topeka"): 1.8,
        ("san-diego", "new-york"): 2.5,
        ("san-diego", "chicago"): 1.8,
        ("san-diego", "topeka"): 1.4,
    }
    cost = {(i, j): 90.0 * distance[(i, j)] / 1000.0 for i, j in product(plants, markets)}

    model = ConcreteModel()
    model.P = Set(initialize=plants)
    model.M = Set(initialize=markets)

    model.w = Var(model.P, bounds=(0.0, None), initialize=0.1)
    model.p = Var(model.M, bounds=(0.0, None), initialize=0.2)
    model.x = Var(model.P, model.M, bounds=(0.0, None), initialize=100.0)

    model.arc = Complementarity(
        model.P,
        model.M,
        rule=lambda m, i, j: complements(0 <= m.x[i, j], m.w[i] + cost[(i, j)] - m.p[j] >= 0),
    )
    model.cap = Complementarity(
        model.P,
        rule=lambda m, i: complements(0 <= m.w[i], capacity[i] - sum(m.x[i, j] for j in m.M) >= 0),
    )
    model.dem = Complementarity(
        model.M,
        rule=lambda m, j: complements(0 <= m.p[j], sum(m.x[i, j] for i in m.P) - demand[j] >= 0),
    )

    solver = SolverFactory("path")
    if not solver.available(exception_flag=False):
        raise RuntimeError("Pyomo solver 'path' is not available in this environment.")
    result = solver.solve(model, tee=bool(output))

    return {
        "status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition),
        "plants": plants,
        "markets": markets,
        "capacity": capacity,
        "demand": demand,
        "w": {i: float(value(model.w[i])) for i in plants},
        "p": {j: float(value(model.p[j])) for j in markets},
        "x": {f"{i}|{j}": float(value(model.x[i, j])) for i, j in product(plants, markets)},
    }


def _solve_with_path_capi(
    *,
    path_capi_src: Path | None,
    path_capi_libpath: Path | None,
    path_capi_lusol: Path | None,
    output: bool,
) -> dict[str, Any]:
    if path_capi_src is not None:
        src = str(path_capi_src.resolve())
        if src not in sys.path:
            sys.path.insert(0, src)

    from path_capi_python import PATHLoader, solve_linear_mcp  # type: ignore

    plants = ["seattle", "san-diego"]
    markets = ["new-york", "chicago", "topeka"]
    capacity = {"seattle": 350.0, "san-diego": 600.0}
    demand = {"new-york": 325.0, "chicago": 300.0, "topeka": 275.0}
    distance = {
        ("seattle", "new-york"): 2.5,
        ("seattle", "chicago"): 1.7,
        ("seattle", "topeka"): 1.8,
        ("san-diego", "new-york"): 2.5,
        ("san-diego", "chicago"): 1.8,
        ("san-diego", "topeka"): 1.4,
    }
    cost = {(i, j): 90.0 * distance[(i, j)] / 1000.0 for i, j in product(plants, markets)}

    var_index: dict[tuple[str, ...], int] = {}
    variables: list[tuple[str, ...]] = []
    for i in plants:
        var_index[("w", i)] = len(variables)
        variables.append(("w", i))
    for j in markets:
        var_index[("p", j)] = len(variables)
        variables.append(("p", j))
    for i, j in product(plants, markets):
        var_index[("x", i, j)] = len(variables)
        variables.append(("x", i, j))

    n = len(variables)
    mtx = [[0.0 for _ in range(n)] for _ in range(n)]
    q = [0.0 for _ in range(n)]

    for i in plants:
        row = var_index[("w", i)]
        q[row] = capacity[i]
        for j in markets:
            mtx[row][var_index[("x", i, j)]] = -1.0

    for j in markets:
        row = var_index[("p", j)]
        q[row] = -demand[j]
        for i in plants:
            mtx[row][var_index[("x", i, j)]] = 1.0

    for i, j in product(plants, markets):
        row = var_index[("x", i, j)]
        q[row] = cost[(i, j)]
        mtx[row][var_index[("w", i)]] = 1.0
        mtx[row][var_index[("p", j)]] = -1.0

    if path_capi_libpath is not None:
        loader = PATHLoader(path_lib=path_capi_libpath, lusol_lib=path_capi_lusol)
    else:
        loader = PATHLoader.from_environment()
    runtime = loader.load()
    solve_result = solve_linear_mcp(
        runtime,
        mtx,
        q,
        [0.0] * n,
        [1.0e20] * n,
        [0.0] * n,
        output=bool(output),
    )

    x = {}
    w = {}
    p = {}
    for i in plants:
        w[i] = float(solve_result.x[var_index[("w", i)]])
    for j in markets:
        p[j] = float(solve_result.x[var_index[("p", j)]])
    for i, j in product(plants, markets):
        x[f"{i}|{j}"] = float(solve_result.x[var_index[("x", i, j)]])

    return {
        "termination_code": int(solve_result.termination_code),
        "residual": float(solve_result.residual),
        "major_iterations": int(solve_result.major_iterations),
        "minor_iterations": int(solve_result.minor_iterations),
        "plants": plants,
        "markets": markets,
        "capacity": capacity,
        "demand": demand,
        "w": w,
        "p": p,
        "x": x,
    }


def run_transmcp_parity(
    *,
    path_capi_src: Path | None,
    path_capi_libpath: Path | None,
    path_capi_lusol: Path | None,
    output: bool,
    price_tol: float,
    aggregate_flow_tol: float,
    residual_tol: float,
    arc_flow_tol: float,
) -> dict[str, Any]:
    pathampl = _solve_with_pathampl(output=output)
    path_capi = _solve_with_path_capi(
        path_capi_src=path_capi_src,
        path_capi_libpath=path_capi_libpath,
        path_capi_lusol=path_capi_lusol,
        output=output,
    )

    plants = pathampl["plants"]
    markets = pathampl["markets"]

    price_diffs = {}
    max_abs_price_diff = 0.0
    for i in plants:
        diff = abs(float(pathampl["w"][i]) - float(path_capi["w"][i]))
        price_diffs[f"w|{i}"] = diff
        max_abs_price_diff = max(max_abs_price_diff, diff)
    for j in markets:
        diff = abs(float(pathampl["p"][j]) - float(path_capi["p"][j]))
        price_diffs[f"p|{j}"] = diff
        max_abs_price_diff = max(max_abs_price_diff, diff)

    arc_diffs = {}
    max_abs_arc_diff = 0.0
    for i, j in product(plants, markets):
        key = f"{i}|{j}"
        diff = abs(float(pathampl["x"][key]) - float(path_capi["x"][key]))
        arc_diffs[key] = diff
        max_abs_arc_diff = max(max_abs_arc_diff, diff)

    market_sum_diffs = {}
    max_abs_market_sum_diff = 0.0
    for j in markets:
        key = f"demand_sum|{j}"
        path_sum = sum(float(pathampl["x"][f"{i}|{j}"]) for i in plants)
        capi_sum = sum(float(path_capi["x"][f"{i}|{j}"]) for i in plants)
        diff = abs(path_sum - capi_sum)
        market_sum_diffs[key] = diff
        max_abs_market_sum_diff = max(max_abs_market_sum_diff, diff)

    plant_sum_diffs = {}
    max_abs_plant_sum_diff = 0.0
    for i in plants:
        key = f"supply_sum|{i}"
        path_sum = sum(float(pathampl["x"][f"{i}|{j}"]) for j in markets)
        capi_sum = sum(float(path_capi["x"][f"{i}|{j}"]) for j in markets)
        diff = abs(path_sum - capi_sum)
        plant_sum_diffs[key] = diff
        max_abs_plant_sum_diff = max(max_abs_plant_sum_diff, diff)

    path_ok = pathampl["status"].lower() == "ok" and pathampl["termination_condition"].lower() in {
        "optimal",
        "locallyoptimal",
    }
    path_capi_ok = int(path_capi["termination_code"]) == 1 and float(path_capi["residual"]) <= float(residual_tol)

    parity_checks = {
        "pathampl_solve_ok": bool(path_ok),
        "path_capi_solve_ok": bool(path_capi_ok),
        "price_match": max_abs_price_diff <= float(price_tol),
        "aggregate_flow_match": max(max_abs_market_sum_diff, max_abs_plant_sum_diff) <= float(aggregate_flow_tol),
        "arc_flow_match": max_abs_arc_diff <= float(arc_flow_tol),
    }
    passed = all(parity_checks.values())

    return {
        "passed": bool(passed),
        "tolerances": {
            "price_tol": float(price_tol),
            "aggregate_flow_tol": float(aggregate_flow_tol),
            "arc_flow_tol": float(arc_flow_tol),
            "residual_tol": float(residual_tol),
        },
        "checks": parity_checks,
        "metrics": {
            "max_abs_price_diff": max_abs_price_diff,
            "max_abs_market_sum_diff": max_abs_market_sum_diff,
            "max_abs_plant_sum_diff": max_abs_plant_sum_diff,
            "max_abs_arc_diff": max_abs_arc_diff,
        },
        "diffs": {
            "prices": price_diffs,
            "market_sums": market_sum_diffs,
            "plant_sums": plant_sum_diffs,
            "arc_flows": arc_diffs,
        },
        "pathampl": pathampl,
        "path_capi": path_capi,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-capi-src",
        type=Path,
        default=DEFAULT_PATH_CAPI_SRC if DEFAULT_PATH_CAPI_SRC.exists() else None,
        help="Path to path-capi-python/src (for local import).",
    )
    parser.add_argument(
        "--path-capi-libpath",
        type=Path,
        default=None,
        help="Optional explicit path to libpath.dylib (otherwise PATH_CAPI_LIBPATH env is used).",
    )
    parser.add_argument(
        "--path-capi-lusol",
        type=Path,
        default=None,
        help="Optional explicit path to liblusol.dylib.",
    )
    parser.add_argument(
        "--price-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for price variables (w, p).",
    )
    parser.add_argument(
        "--aggregate-flow-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for market and plant aggregate flow sums.",
    )
    parser.add_argument(
        "--arc-flow-tol",
        type=float,
        default=2e-3,
        help="Absolute tolerance for individual bilateral flow values.",
    )
    parser.add_argument(
        "--residual-tol",
        type=float,
        default=1e-6,
        help="Residual tolerance for path-capi solve status.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Return non-zero exit code if parity checks fail.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Enable solver output streams.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    report = run_transmcp_parity(
        path_capi_src=args.path_capi_src,
        path_capi_libpath=args.path_capi_libpath,
        path_capi_lusol=args.path_capi_lusol,
        output=bool(args.output),
        price_tol=float(args.price_tol),
        aggregate_flow_tol=float(args.aggregate_flow_tol),
        residual_tol=float(args.residual_tol),
        arc_flow_tol=float(args.arc_flow_tol),
    )

    print("PATHAMPL status:", report["pathampl"]["status"], report["pathampl"]["termination_condition"])
    print(
        "PATH CAPI status:",
        f"term_code={report['path_capi']['termination_code']}",
        f"residual={report['path_capi']['residual']:.3e}",
    )
    print(
        "Max diffs:",
        f"price={report['metrics']['max_abs_price_diff']:.3e}",
        f"market_sum={report['metrics']['max_abs_market_sum_diff']:.3e}",
        f"plant_sum={report['metrics']['max_abs_plant_sum_diff']:.3e}",
        f"arc={report['metrics']['max_abs_arc_diff']:.3e}",
    )
    print("Parity passed:", report["passed"])

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.gate and not bool(report["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
