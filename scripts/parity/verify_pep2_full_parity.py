#!/usr/bin/env python3
"""
Full parity audit between pep2 GAMS outputs and Python equilibria state.

Compares:
- Variables: BASE slice from pep2/scripts/Results.gdx (val* symbols)
- Parameters: pep2/scripts/Parameters.gdx
against Python calibration + solver initialization (excel mode).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver


@dataclass
class DiffRow:
    symbol: str
    key: tuple[str, ...]
    gams: float
    python: float
    diff: float


_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_LAB_RE = re.compile(r"'([^']*)'")


def gdxdump_records(gdxdump_bin: str, gdx_path: Path, symbol: str) -> list[tuple[tuple[str, ...], float]]:
    out = subprocess.check_output(
        [gdxdump_bin, str(gdx_path), f"symb={symbol}"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    records: list[tuple[tuple[str, ...], float]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.startswith(("/", "Parameter ", "Set ", "*")):
            continue
        num_match = _NUM_RE.search(line)
        if not num_match:
            continue
        labels = tuple(x.lower() for x in _LAB_RE.findall(line))
        value = float(num_match.group(1))
        records.append((labels, value))
    return records


def get_var_value(
    vars_obj: Any,
    symbol: str,
    idx: tuple[str, ...],
    params: dict[str, Any],
) -> float | None:
    # Variable aliases represented as parameters/derived values in Python.
    if symbol == "valPWX" and len(idx) == 1:
        return params.get("PWX", {}).get(idx[0], 1.0)
    if symbol == "valPT" and len(idx) == 1:
        return params.get("PT", {}).get(idx[0], 1.0)
    if symbol == "valttdh1" and len(idx) == 1:
        return params.get("ttdh1", {}).get(idx[0])
    if symbol == "valttic" and len(idx) == 1:
        return params.get("ttic", {}).get(idx[0])
    if symbol == "valtr1" and len(idx) == 1:
        return params.get("tr1", {}).get(idx[0])
    if symbol == "valttim" and len(idx) == 1:
        return params.get("ttim", {}).get(idx[0])
    if symbol == "valttiw" and len(idx) == 2:
        return params.get("ttiw", {}).get((idx[0], idx[1]))
    if symbol == "valKS" and len(idx) == 1:
        return params.get("KS", {}).get(idx[0])
    if symbol == "valLS" and len(idx) == 1:
        return params.get("LS", {}).get(idx[0])
    if symbol == "valRK" and len(idx) == 1:
        return 1.0
    if symbol == "valsh1" and len(idx) == 1:
        return params.get("sh1", {}).get(idx[0])
    if symbol == "valttip" and len(idx) == 1:
        return params.get("ttip", {}).get(idx[0])
    if symbol == "valttdf1" and len(idx) == 1:
        return params.get("ttdf1", {}).get(idx[0])
    if symbol == "valttik" and len(idx) == 2:
        return params.get("ttik", {}).get((idx[0], idx[1]))
    if symbol == "valttix" and len(idx) == 1:
        return params.get("ttix", {}).get(idx[0])
    if symbol == "valGFCF_REAL" and len(idx) == 0:
        pixinv = vars_obj.PIXINV if abs(vars_obj.PIXINV) > 1e-12 else 1.0
        return vars_obj.GFCF / pixinv

    # valXYZ -> field XYZ, except vale -> e
    if symbol == "vale":
        field = "e"
    else:
        field = symbol[3:]

    if not hasattr(vars_obj, field):
        return None
    obj = getattr(vars_obj, field)

    if isinstance(obj, dict):
        if len(idx) == 0:
            return None
        if len(idx) == 1:
            return obj.get(idx[0])
        return obj.get(tuple(idx))

    if len(idx) != 0:
        return None
    try:
        return float(obj)
    except Exception:
        return None


def get_param_value(params: dict[str, Any], symbol: str, idx: tuple[str, ...]) -> float | None:
    candidates = [symbol, symbol.lower(), symbol.upper()]
    obj = None
    for c in candidates:
        if c in params:
            obj = params[c]
            break
    if obj is None:
        if symbol == "kmob" and len(idx) == 0:
            return 1.0
        return None

    if isinstance(obj, dict):
        if len(idx) == 0:
            return None
        if len(idx) == 1:
            v = obj.get(idx[0])
            return v
        v = obj.get(tuple(idx))
        return v

    if len(idx) != 0:
        return None
    try:
        return float(obj)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify full pep2 parity vs Python.")
    parser.add_argument(
        "--pep2-root",
        type=Path,
        default=Path("/Users/marmol/proyectos/equilibria/src/equilibria/templates/reference/pep2"),
    )
    parser.add_argument(
        "--gdxdump",
        default="/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    )
    parser.add_argument(
        "--presolve-gdx",
        type=Path,
        default=None,
        help="Optional pre-solve GDX baseline for selected variables (EXD, PE_FOB, PP, XST, MRGN, GDP_MP_REAL).",
    )
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.pep2_root.resolve()
    results_gdx = root / "scripts" / "Results.gdx"
    params_gdx = root / "scripts" / "Parameters.gdx"
    presolve_gdx = args.presolve_gdx.resolve() if args.presolve_gdx else None
    sam_gdx = root / "data" / "SAM-V2_0.gdx"
    val_xlsx = root / "data" / "VAL_PAR.xlsx"

    state = PEPModelCalibrator(sam_file=sam_gdx, val_par_file=val_xlsx).calibrate()
    solver = PEPModelSolver(calibrated_state=state, init_mode="excel")
    vars_ = solver._create_initial_guess()
    py_params = solver.params

    results_meta = read_gdx(results_gdx)
    params_meta = read_gdx(params_gdx)

    var_symbols = [s["name"] for s in results_meta["symbols"] if s["name"].startswith("val")]
    par_symbols = [s["name"] for s in params_meta["symbols"]]

    var_diffs: list[DiffRow] = []
    var_missing = 0
    var_compared = 0
    presolve_symbol_map = {
        "valEXD": "EXD",
        "valPE_FOB": "PE_FOB",
        "valPP": "PP",
        "valXST": "XST",
        "valMRGN": "MRGN",
        "valGDP_MP_REAL": "GDP_MP_REAL",
    }

    for sym in var_symbols:
        source_gdx = results_gdx
        source_symbol = sym
        if presolve_gdx and sym in presolve_symbol_map:
            source_gdx = presolve_gdx
            source_symbol = presolve_symbol_map[sym]

        for labels, gams_val in gdxdump_records(args.gdxdump, source_gdx, source_symbol):
            # Keep BASE slice for scenario-style symbols
            idx = labels
            if source_gdx == results_gdx and labels and labels[-1] in {"base", "sim1", "var"}:
                if labels[-1] != "base":
                    continue
                idx = labels[:-1]
            py_val = get_var_value(vars_, sym, idx, py_params)
            if py_val is None:
                var_missing += 1
                continue
            var_compared += 1
            diff = py_val - gams_val
            if abs(diff) > args.tol:
                var_diffs.append(DiffRow(sym, idx, gams_val, py_val, diff))

    par_diffs: list[DiffRow] = []
    par_missing = 0
    par_compared = 0
    for sym in par_symbols:
        for labels, gams_val in gdxdump_records(args.gdxdump, params_gdx, sym):
            idx = labels
            py_val = get_param_value(py_params, sym, idx)
            if py_val is None:
                par_missing += 1
                continue
            par_compared += 1
            diff = py_val - gams_val
            if abs(diff) > args.tol:
                par_diffs.append(DiffRow(sym, idx, gams_val, py_val, diff))

    var_diffs.sort(key=lambda r: abs(r.diff), reverse=True)
    par_diffs.sort(key=lambda r: abs(r.diff), reverse=True)

    print("PEP2 Full Parity Audit")
    print(f"  root: {root}")
    print(f"  tolerance: {args.tol:g}")
    print("-" * 84)
    print(f"Variables: compared={var_compared}, mismatches={len(var_diffs)}, missing={var_missing}")
    print(f"Parameters: compared={par_compared}, mismatches={len(par_diffs)}, missing={par_missing}")

    if var_diffs:
        print("\nTop variable mismatches:")
        for r in var_diffs[:20]:
            print(
                f"  {r.symbol}{r.key}: gams={r.gams:.12g} py={r.python:.12g} diff={r.diff:.12g}"
            )
    if par_diffs:
        print("\nTop parameter mismatches:")
        for r in par_diffs[:20]:
            print(
                f"  {r.symbol}{r.key}: gams={r.gams:.12g} py={r.python:.12g} diff={r.diff:.12g}"
            )

    if args.save_json:
        payload = {
            "root": str(root),
            "tolerance": args.tol,
            "variables": {
                "compared": var_compared,
                "mismatches": len(var_diffs),
                "missing": var_missing,
                "top_mismatches": [
                    {
                        "symbol": r.symbol,
                        "key": r.key,
                        "gams": r.gams,
                        "python": r.python,
                        "diff": r.diff,
                    }
                    for r in var_diffs[:200]
                ],
            },
            "parameters": {
                "compared": par_compared,
                "mismatches": len(par_diffs),
                "missing": par_missing,
                "top_mismatches": [
                    {
                        "symbol": r.symbol,
                        "key": r.key,
                        "gams": r.gams,
                        "python": r.python,
                        "diff": r.diff,
                    }
                    for r in par_diffs[:200]
                ],
            },
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved report: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
