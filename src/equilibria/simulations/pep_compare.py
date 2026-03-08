"""PEP comparison helpers used by the new simulations API."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx
from equilibria.templates.pep_model_equations import PEPModelVariables

DEFAULT_GDXDUMP_BIN = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_LAB_RE = re.compile(r"'([^']*)'")
_SCENARIOS = {"base", "sim1", "var"}


def get_solution_value(
    vars_obj: PEPModelVariables,
    symbol: str,
    idx: tuple[str, ...],
    params: dict[str, Any],
) -> float | None:
    """Map one `val*` symbol from Results.gdx to Python solution values."""
    if symbol == "valPWX" and len(idx) == 1:
        return params.get("PWX", {}).get(idx[0], 1.0)
    if symbol == "valPT" and len(idx) == 1:
        return vars_obj.PT.get(idx[0], params.get("PT", {}).get(idx[0], 1.0))
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
        return vars_obj.RK.get(idx[0], 1.0)
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

    field = "e" if symbol == "vale" else symbol[3:]
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


def key_indicators(vars_obj: PEPModelVariables) -> dict[str, float]:
    """Return standardized high-level indicators for PEP results."""
    total_exports = float(sum(vars_obj.EXD.values()))
    total_imports = float(sum(vars_obj.IM.values()))
    return {
        "GDP_BP": float(vars_obj.GDP_BP),
        "GDP_MP": float(vars_obj.GDP_MP),
        "GDP_IB": float(vars_obj.GDP_IB),
        "GDP_FD": float(vars_obj.GDP_FD),
        "IT": float(vars_obj.IT),
        "CAB": float(vars_obj.CAB),
        "TIXT": float(vars_obj.TIXT),
        "TPRODN": float(vars_obj.TPRODN),
        "TPRCTS": float(vars_obj.TPRCTS),
        "total_exports": total_exports,
        "total_imports": total_imports,
        "trade_balance": total_exports - total_imports,
        "PIXCON": float(vars_obj.PIXCON),
        "e": float(vars_obj.e),
    }


def compare_with_gams(
    *,
    solution_vars: PEPModelVariables,
    solution_params: dict[str, Any],
    gams_results_gdx: Path | str,
    gams_slice: str,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    gdxdump_bin: str = DEFAULT_GDXDUMP_BIN,
) -> dict[str, Any]:
    """Compare one PEP solution against one GAMS results slice."""
    results_path = Path(gams_results_gdx)
    gdxdump = _resolve_gdxdump_binary(gdxdump_bin)
    symbols = [
        s["name"]
        for s in read_gdx(results_path).get("symbols", [])
        if s["name"].startswith("val")
    ]

    compared = 0
    missing = 0
    mismatches: list[dict[str, Any]] = []

    for symbol in symbols:
        for idx, gams_val in _iter_slice_records(gdxdump, results_path, symbol, gams_slice):
            py_val = get_solution_value(solution_vars, symbol, idx, solution_params)
            if py_val is None:
                missing += 1
                continue
            compared += 1
            abs_diff = abs(float(py_val) - float(gams_val))
            rel_diff = abs_diff / max(abs(float(gams_val)), abs(float(py_val)), 1.0)
            if abs_diff > abs_tol and rel_diff > rel_tol:
                mismatches.append(
                    {
                        "symbol": symbol,
                        "key": list(idx),
                        "gams": float(gams_val),
                        "python": float(py_val),
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                    }
                )

    mismatches.sort(key=lambda x: x["abs_diff"], reverse=True)
    max_abs = max((m["abs_diff"] for m in mismatches), default=0.0)
    max_rel = max((m["rel_diff"] for m in mismatches), default=0.0)
    rms = (
        math.sqrt(sum(m["abs_diff"] ** 2 for m in mismatches) / len(mismatches))
        if mismatches
        else 0.0
    )

    return {
        "gams_slice": gams_slice.lower(),
        "compared": compared,
        "missing": missing,
        "mismatches": len(mismatches),
        "passed": len(mismatches) == 0,
        "compare_abs_tol": float(abs_tol),
        "compare_rel_tol": float(rel_tol),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "rms_abs_diff": rms,
        "top_mismatches": mismatches[:30],
    }


def _resolve_gdxdump_binary(raw: str) -> Path:
    token = str(raw).strip()
    if token:
        candidate = Path(token)
        if candidate.exists():
            return candidate
        resolved = shutil.which(token)
        if resolved:
            return Path(resolved)

    fallback = Path(DEFAULT_GDXDUMP_BIN)
    if fallback.exists():
        return fallback

    resolved = shutil.which("gdxdump")
    if resolved:
        return Path(resolved)

    raise FileNotFoundError(
        "gdxdump binary not found. Set --gdxdump-bin to your GAMS gdxdump path."
    )


def _gdxdump_records(
    gdxdump_bin: Path,
    gdx_file: Path,
    symbol: str,
) -> list[tuple[tuple[str, ...], float]]:
    out = subprocess.check_output(
        [str(gdxdump_bin), str(gdx_file), f"symb={symbol}"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows: list[tuple[tuple[str, ...], float]] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.startswith(("/", "Parameter ", "Set ", "*")):
            continue
        nums = _NUM_RE.findall(line)
        if not nums:
            continue
        labels = tuple(x.lower() for x in _LAB_RE.findall(line))
        value = float(nums[-1])
        rows.append((labels, value))
    return rows


def _iter_slice_records(
    gdxdump_bin: Path,
    gams_results_gdx: Path,
    symbol: str,
    gams_slice: str,
) -> list[tuple[tuple[str, ...], float]]:
    wanted = gams_slice.lower()
    out: list[tuple[tuple[str, ...], float]] = []
    for labels, value in _gdxdump_records(gdxdump_bin, gams_results_gdx, symbol):
        if labels and labels[-1] in _SCENARIOS:
            if labels[-1] != wanted:
                continue
            out.append((labels[:-1], value))
            continue
        if wanted == "base":
            out.append((labels, value))
    return out
