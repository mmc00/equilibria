"""
Systemic parity pipeline utilities for PEP solver diagnostics.

This module provides:
- Equation-contract grouping by block
- Residual metrics by block
- Fail-fast gate evaluation
- JSON-serializable report helpers
"""

from __future__ import annotations

import copy
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values


@dataclass(frozen=True)
class EquationContract:
    """Defines which equations belong to a parity control block."""

    block: str
    eq_prefixes: tuple[str, ...]
    max_abs_gate: float
    rms_gate: float


@dataclass
class ResidualSummary:
    count: int
    rms: float
    max_abs: float
    top_abs: list[tuple[str, float]]


@dataclass
class BlockGateResult:
    block: str
    passed: bool
    max_abs: float
    rms: float
    max_abs_gate: float
    rms_gate: float
    top_abs: list[tuple[str, float]]


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def summarize_residuals(residuals: dict[str, float], top_n: int = 15) -> ResidualSummary:
    values = list(residuals.values())
    max_abs = max((abs(v) for v in values), default=0.0)
    top = sorted(residuals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return ResidualSummary(
        count=len(values),
        rms=_rms(values),
        max_abs=max_abs,
        top_abs=top,
    )


def default_equation_contracts() -> list[EquationContract]:
    """
    Default block contracts for systemic convergence debugging.

    Gates are intentionally strict for benchmark parity workflows.
    """
    return [
        EquationContract(
            block="production_tax_consistency",
            eq_prefixes=("EQ29", "EQ39", "EQ40"),
            max_abs_gate=1e-6,
            rms_gate=1e-7,
        ),
        EquationContract(
            block="trade_price_index_consistency",
            eq_prefixes=("EQ79", "EQ84"),
            max_abs_gate=1e-6,
            rms_gate=1e-7,
        ),
        EquationContract(
            block="trade_market_clearing",
            # Include EQ63 so Armington quantity consistency cannot be hidden by
            # passing EQ64/EQ88 alone.
            eq_prefixes=("EQ63", "EQ64", "EQ88"),
            max_abs_gate=1e-6,
            rms_gate=1e-7,
        ),
        EquationContract(
            block="macro_closure",
            eq_prefixes=("EQ44", "EQ45", "EQ46", "EQ87", "EQ93", "WALRAS"),
            max_abs_gate=1e-6,
            rms_gate=1e-7,
        ),
    ]


def _filter_block_residuals(
    residuals: dict[str, float],
    eq_prefixes: tuple[str, ...],
) -> dict[str, float]:
    return {
        name: value
        for name, value in residuals.items()
        if any(name.startswith(prefix) for prefix in eq_prefixes)
    }


def evaluate_block_gates(
    residuals: dict[str, float],
    contracts: list[EquationContract] | None = None,
    fail_fast: bool = True,
) -> dict[str, Any]:
    contracts = contracts or default_equation_contracts()
    block_results: list[BlockGateResult] = []
    first_failed_block: str | None = None

    for contract in contracts:
        subset = _filter_block_residuals(residuals, contract.eq_prefixes)
        values = list(subset.values())
        block_max = max((abs(v) for v in values), default=0.0)
        block_rms = _rms(values)
        passed = (block_max <= contract.max_abs_gate) and (block_rms <= contract.rms_gate)

        block_results.append(
            BlockGateResult(
                block=contract.block,
                passed=passed,
                max_abs=block_max,
                rms=block_rms,
                max_abs_gate=contract.max_abs_gate,
                rms_gate=contract.rms_gate,
                top_abs=sorted(subset.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8],
            )
        )

        if fail_fast and not passed:
            first_failed_block = contract.block
            break

    if not fail_fast:
        for r in block_results:
            if not r.passed:
                first_failed_block = r.block
                break

    overall_passed = (first_failed_block is None)

    return {
        "overall_passed": overall_passed,
        "fail_fast": fail_fast,
        "first_failed_block": first_failed_block,
        "blocks": [asdict(r) for r in block_results],
    }


_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
_LAB_RE = re.compile(r"'([^']*)'")


def _gdxdump_records(gdxdump_bin: str, gdx_path: Path, symbol: str) -> list[tuple[tuple[str, ...], float]]:
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
        nums = _NUM_RE.findall(line)
        if not nums:
            continue
        labels = tuple(x.lower() for x in _LAB_RE.findall(line))
        value = float(nums[-1])
        records.append((labels, value))
    return records


def _slice_map(
    records: list[tuple[tuple[str, ...], float]],
    gams_slice: str,
) -> dict[tuple[str, ...], float]:
    out: dict[tuple[str, ...], float] = {}
    s = gams_slice.lower()
    for labels, value in records:
        if not labels:
            continue
        if labels[-1] == s:
            out[labels[:-1]] = value
    return out


def _read_symbol_records(
    gdxdump_bin: str,
    gdx_path: Path,
    symbol: str,
    gams_slice: str,
) -> dict[tuple[str, ...], float]:
    """Read one val* symbol, preferring gdxdump and falling back to read_gdx."""
    gdxdump_path = shutil.which(gdxdump_bin) if Path(gdxdump_bin).name == gdxdump_bin else gdxdump_bin
    records: list[tuple[tuple[str, ...], float]] = []

    if gdxdump_bin:
        try:
            bin_for_call = str(gdxdump_path) if (gdxdump_path and Path(gdxdump_path).exists()) else str(gdxdump_bin)
            records = _gdxdump_records(bin_for_call, gdx_path, symbol)
        except Exception:
            records = []

    if not records:
        gdx = read_gdx(gdx_path)
        try:
            values = read_parameter_values(gdx, symbol)
        except Exception:
            return {}
        for raw_key, raw_val in values.items():
            if isinstance(raw_key, tuple):
                labels = tuple(str(k).lower() for k in raw_key)
            elif raw_key == ():
                labels = ()
            else:
                labels = (str(raw_key).lower(),)
            records.append((labels, float(raw_val)))

    return _slice_map(records, gams_slice)


def evaluate_eq29_eq39_against_gams(
    vars_obj: Any,
    results_gdx: Path | str,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    gams_slice: str = "base",
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Compare Python EQ29/EQ39/EQ40 equation residuals against residuals computed from GAMS levels.

    EQ29: TIPT = sum_j TIP(j)
    EQ39_j: TIP(j) = ttip(j) * PP(j) * XST(j)
    EQ40_i: TIC(i) = [ttic(i)/(1+ttic(i))] * [PD(i)*DD(i) + PM(i)*IM(i)]
    """
    gdx_path = Path(results_gdx)

    tip_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valTIP", gams_slice)
    tipt_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valTIPT", gams_slice)
    pp_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPP", gams_slice)
    xst_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valXST", gams_slice)
    ttip_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valttip", gams_slice)
    tic_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valTIC", gams_slice)
    ttic_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valttic", gams_slice)
    pd_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPD", gams_slice)
    dd_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valDD", gams_slice)
    pm_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPM", gams_slice)
    im_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valIM", gams_slice)

    py_eq29 = float(vars_obj.TIPT - sum(vars_obj.TIP.values()))
    gams_tipt = float(tipt_rec.get((), 0.0))
    gams_eq29 = float(gams_tipt - sum(tip_rec.values()))
    eq29_delta = py_eq29 - gams_eq29

    j_keys = sorted(
        {
            k[0]
            for k in list(tip_rec.keys()) + list(pp_rec.keys()) + list(xst_rec.keys()) + list(ttip_rec.keys())
            if len(k) == 1
        }
    )

    eq39_rows: list[dict[str, Any]] = []
    for j in j_keys:
        py_tip = float(vars_obj.TIP.get(j, 0.0))
        py_pp = float(vars_obj.PP.get(j, 0.0))
        py_xst = float(vars_obj.XST.get(j, 0.0))
        # ttip in vars may be missing depending on mode; use GAMS ttip anchor for parity.
        py_ttip = float(ttip_rec.get((j,), 0.0))
        py_res = py_tip - py_ttip * py_pp * py_xst

        g_tip = float(tip_rec.get((j,), 0.0))
        g_pp = float(pp_rec.get((j,), 0.0))
        g_xst = float(xst_rec.get((j,), 0.0))
        g_ttip = float(ttip_rec.get((j,), 0.0))
        g_res = g_tip - g_ttip * g_pp * g_xst

        delta = py_res - g_res
        eq39_rows.append(
            {
                "j": j,
                "python_residual": py_res,
                "gams_residual": g_res,
                "delta": delta,
            }
        )

    max_abs_eq39_delta = max((abs(r["delta"]) for r in eq39_rows), default=0.0)

    i_keys = sorted(
        {
            k[0]
            for k in list(tic_rec.keys()) + list(ttic_rec.keys()) + list(pd_rec.keys()) + list(dd_rec.keys()) + list(pm_rec.keys()) + list(im_rec.keys())
            if len(k) == 1
        }
    )
    eq40_rows: list[dict[str, Any]] = []
    for i in i_keys:
        py_tic = float(vars_obj.TIC.get(i, 0.0))
        py_ttic = float(ttic_rec.get((i,), 0.0))
        py_pd = float(vars_obj.PD.get(i, 0.0))
        py_dd = float(vars_obj.DD.get(i, 0.0))
        py_pm = float(vars_obj.PM.get(i, 0.0))
        py_im = float(vars_obj.IM.get(i, 0.0))
        py_denom = 1.0 + py_ttic
        py_expected = 0.0 if abs(py_denom) < 1e-12 else (py_ttic / py_denom) * (py_pd * py_dd + py_pm * py_im)
        py_res = py_tic - py_expected

        g_tic = float(tic_rec.get((i,), 0.0))
        g_ttic = float(ttic_rec.get((i,), 0.0))
        g_pd = float(pd_rec.get((i,), 0.0))
        g_dd = float(dd_rec.get((i,), 0.0))
        g_pm = float(pm_rec.get((i,), 0.0))
        g_im = float(im_rec.get((i,), 0.0))
        g_denom = 1.0 + g_ttic
        g_expected = 0.0 if abs(g_denom) < 1e-12 else (g_ttic / g_denom) * (g_pd * g_dd + g_pm * g_im)
        g_res = g_tic - g_expected

        delta = py_res - g_res
        eq40_rows.append(
            {
                "i": i,
                "python_residual": py_res,
                "gams_residual": g_res,
                "delta": delta,
            }
        )

    max_abs_eq40_delta = max((abs(r["delta"]) for r in eq40_rows), default=0.0)
    max_abs_overall_delta = max(abs(eq29_delta), max_abs_eq39_delta, max_abs_eq40_delta)
    passed = (abs(eq29_delta) <= tol) and (max_abs_eq39_delta <= tol) and (max_abs_eq40_delta <= tol)

    eq39_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    eq40_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)

    return {
        "passed": passed,
        "tol": tol,
        "eq29": {
            "python_residual": py_eq29,
            "gams_residual": gams_eq29,
            "delta": eq29_delta,
        },
        "eq39": {
            "max_abs_delta": max_abs_eq39_delta,
            "rows": eq39_rows,
        },
        "eq40": {
            "max_abs_delta": max_abs_eq40_delta,
            "rows": eq40_rows,
        },
        "max_abs_delta_overall": max_abs_overall_delta,
    }


def evaluate_eq79_eq84_against_gams(
    vars_obj: Any,
    results_gdx: Path | str,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    gams_slice: str = "base",
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Compare Python EQ79/EQ84 residuals against residuals computed from GAMS levels.

    EQ79_i: PC(i)*Q(i) = PM(i)*IM(i) + PD(i)*DD(i)
    EQ84_i: Q(i) = sum_h C(i,h) + CG(i) + INV(i) + VSTK(i) + DIT(i) + MRGN(i)
    """
    gdx_path = Path(results_gdx)

    pc_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPC", gams_slice)
    q_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valQ", gams_slice)
    pm_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPM", gams_slice)
    im_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valIM", gams_slice)
    pd_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valPD", gams_slice)
    dd_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valDD", gams_slice)
    c_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valC", gams_slice)
    cg_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valCG", gams_slice)
    inv_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valINV", gams_slice)
    vstk_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valVSTK", gams_slice)
    dit_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valDIT", gams_slice)
    mrgn_rec = _read_symbol_records(gdxdump_bin, gdx_path, "valMRGN", gams_slice)

    i_keys = sorted(
        {
            k[0]
            for k in list(q_rec.keys())
            + list(pc_rec.keys())
            + list(pm_rec.keys())
            + list(im_rec.keys())
            + list(pd_rec.keys())
            + list(dd_rec.keys())
            if len(k) == 1
        }
    )

    eq79_rows: list[dict[str, Any]] = []
    for i in i_keys:
        py_res = vars_obj.PC.get(i, 0.0) * vars_obj.Q.get(i, 0.0) - (
            vars_obj.PM.get(i, 0.0) * vars_obj.IM.get(i, 0.0)
            + vars_obj.PD.get(i, 0.0) * vars_obj.DD.get(i, 0.0)
        )
        g_res = pc_rec.get((i,), 0.0) * q_rec.get((i,), 0.0) - (
            pm_rec.get((i,), 0.0) * im_rec.get((i,), 0.0)
            + pd_rec.get((i,), 0.0) * dd_rec.get((i,), 0.0)
        )
        eq79_rows.append(
            {
                "i": i,
                "python_residual": float(py_res),
                "gams_residual": float(g_res),
                "delta": float(py_res - g_res),
            }
        )

    eq79_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    max_abs_eq79_delta = max((abs(r["delta"]) for r in eq79_rows), default=0.0)

    i84_keys = sorted(
        {
            k[0]
            for k in list(q_rec)
            + [k for k in cg_rec if len(k) == 1]
            + [k for k in inv_rec if len(k) == 1]
            + [k for k in vstk_rec if len(k) == 1]
            + [k for k in dit_rec if len(k) == 1]
            + [k for k in mrgn_rec if len(k) == 1]
            + [k for k in c_rec if len(k) == 2]
        }
    )
    eq84_rows: list[dict[str, Any]] = []
    for i in i84_keys:
        py_cons = sum(v for (ci, _h), v in vars_obj.C.items() if ci == i)
        py_expected = (
            py_cons
            + vars_obj.CG.get(i, 0.0)
            + vars_obj.INV.get(i, 0.0)
            + vars_obj.VSTK.get(i, 0.0)
            + vars_obj.DIT.get(i, 0.0)
            + vars_obj.MRGN.get(i, 0.0)
        )
        py_res = vars_obj.Q.get(i, 0.0) - py_expected

        g_cons = sum(v for (ci, _h), v in c_rec.items() if ci == i)
        g_expected = (
            g_cons
            + cg_rec.get((i,), 0.0)
            + inv_rec.get((i,), 0.0)
            + vstk_rec.get((i,), 0.0)
            + dit_rec.get((i,), 0.0)
            + mrgn_rec.get((i,), 0.0)
        )
        g_res = q_rec.get((i,), 0.0) - g_expected
        eq84_rows.append(
            {
                "i": i,
                "python_residual": float(py_res),
                "gams_residual": float(g_res),
                "delta": float(py_res - g_res),
            }
        )

    eq84_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    max_abs_eq84_delta = max((abs(r["delta"]) for r in eq84_rows), default=0.0)

    max_abs_overall_delta = max(max_abs_eq79_delta, max_abs_eq84_delta)
    passed = (max_abs_eq79_delta <= tol) and (max_abs_eq84_delta <= tol)

    return {
        "passed": passed,
        "tol": tol,
        "eq79": {
            "max_abs_delta": max_abs_eq79_delta,
            "rows": eq79_rows,
        },
        "eq84": {
            "max_abs_delta": max_abs_eq84_delta,
            "rows": eq84_rows,
        },
        "max_abs_delta_overall": max_abs_overall_delta,
    }


def evaluate_levels_against_gams(
    vars_obj: Any,
    results_gdx: Path | str,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    gams_slice: str = "base",
    tol: float = 1e-9,
) -> dict[str, Any]:
    """
    Compare Python initialized levels against GAMS val* symbols (slice BASE/SIM1).

    Only compares entries where Python has the corresponding field/key.
    """
    gdx_path = Path(results_gdx)
    gdx = read_gdx(gdx_path)
    symbols = [
        str(sym.get("name", ""))
        for sym in gdx.get("symbols", [])
        if str(sym.get("name", "")).startswith("val")
    ]

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for sym in sorted(set(symbols)):
        field = "e" if sym == "vale" else sym[3:]
        if not hasattr(vars_obj, field):
            continue
        target = getattr(vars_obj, field)
        records = _read_symbol_records(gdxdump_bin, gdx_path, sym, gams_slice)

        for labels, gval in records.items():
            if isinstance(target, dict):
                if len(labels) == 1:
                    key: Any = labels[0]
                else:
                    key = (labels[0], labels[1])
                if key not in target:
                    missing_rows.append(
                        {
                            "symbol": sym,
                            "field": field,
                            "labels": labels,
                            "gams_value": float(gval),
                        }
                    )
                    continue
                pval = float(target.get(key, 0.0))
            else:
                if len(labels) != 0:
                    continue
                pval = float(target)

            delta = pval - float(gval)
            rows.append(
                {
                    "symbol": sym,
                    "field": field,
                    "labels": labels,
                    "python_value": pval,
                    "gams_value": float(gval),
                    "delta": float(delta),
                }
            )

    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    max_abs_delta = max((abs(r["delta"]) for r in rows), default=0.0)
    rms_delta = _rms([float(r["delta"]) for r in rows])
    passed = max_abs_delta <= tol

    return {
        "passed": passed,
        "tol": tol,
        "count_compared": len(rows),
        "count_missing_in_python": len(missing_rows),
        "max_abs_delta": max_abs_delta,
        "rms_delta": rms_delta,
        "top_deltas": rows[:50],
        "missing_in_python": missing_rows[:50],
    }


def _build_vars_from_gams_levels(
    vars_obj: Any,
    results_gdx: Path | str,
    gdxdump_bin: str,
    gams_slice: str,
) -> Any:
    """
    Build a vars object with levels overlaid from GAMS val* symbols.

    Keeps the same object shape as vars_obj and only overwrites fields present
    in both Python vars and GAMS val* symbols.
    """
    gdx_path = Path(results_gdx)
    gdx = read_gdx(gdx_path)
    symbols = [
        str(sym.get("name", ""))
        for sym in gdx.get("symbols", [])
        if str(sym.get("name", "")).startswith("val")
    ]

    out = copy.deepcopy(vars_obj)
    for sym in sorted(set(symbols)):
        field = "e" if sym == "vale" else sym[3:]
        if not hasattr(out, field):
            continue
        target = getattr(out, field)
        records = _read_symbol_records(gdxdump_bin, gdx_path, sym, gams_slice)
        for labels, gval in records.items():
            if isinstance(target, dict):
                if len(labels) == 1:
                    key: Any = labels[0]
                elif len(labels) >= 2:
                    key = (labels[0], labels[1])
                else:
                    continue
                target[key] = float(gval)
            else:
                if len(labels) == 0:
                    setattr(out, field, float(gval))
    return out


def evaluate_residual_parity_against_gams(
    vars_obj: Any,
    equations: Any,
    results_gdx: Path | str,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    gams_slice: str = "base",
    contracts: list[EquationContract] | None = None,
    max_abs_tol: float = 1e-6,
    rms_tol: float = 1e-7,
) -> dict[str, Any]:
    """
    Evaluate init gates by residual-delta parity against GAMS levels.

    Instead of requiring small absolute residuals, this checks whether Python
    reproduces the same residual pattern as the selected GAMS baseline.
    """
    contracts = contracts or default_equation_contracts()
    gams_vars = _build_vars_from_gams_levels(
        vars_obj=vars_obj,
        results_gdx=results_gdx,
        gdxdump_bin=gdxdump_bin,
        gams_slice=gams_slice,
    )

    py_residuals = equations.calculate_all_residuals(vars_obj)
    gams_residuals = equations.calculate_all_residuals(gams_vars)

    names = sorted(set(py_residuals) | set(gams_residuals))
    delta_residuals = {
        name: float(py_residuals.get(name, 0.0) - gams_residuals.get(name, 0.0))
        for name in names
    }

    blocks: list[dict[str, Any]] = []
    first_failed_block: str | None = None
    for contract in contracts:
        subset = _filter_block_residuals(delta_residuals, contract.eq_prefixes)
        vals = list(subset.values())
        block_max = max((abs(v) for v in vals), default=0.0)
        block_rms = _rms(vals)
        passed = (block_max <= max_abs_tol) and (block_rms <= rms_tol)
        blocks.append(
            {
                "block": contract.block,
                "passed": passed,
                "max_abs_delta": block_max,
                "rms_delta": block_rms,
                "max_abs_tol": max_abs_tol,
                "rms_tol": rms_tol,
                "top_abs_delta": sorted(subset.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8],
            }
        )
        if first_failed_block is None and not passed:
            first_failed_block = contract.block

    overall_max_abs_delta = max((abs(v) for v in delta_residuals.values()), default=0.0)
    overall_rms_delta = _rms(list(delta_residuals.values()))

    return {
        "passed": first_failed_block is None,
        "first_failed_block": first_failed_block,
        "max_abs_delta": overall_max_abs_delta,
        "rms_delta": overall_rms_delta,
        "max_abs_tol": max_abs_tol,
        "rms_tol": rms_tol,
        "top_abs_delta": sorted(delta_residuals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20],
        "blocks": blocks,
    }


def evaluate_results_baseline_compatibility(
    state: Any,
    results_gdx: Path | str,
    gdxdump_bin: str = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump",
    gams_slice: str = "base",
    rel_tol: float = 1e-4,
) -> dict[str, Any]:
    """
    Quick guard to detect stale/mismatched Results.gdx baseline.

    Uses GDP_BP parity vs calibrated GDP_BPO as anchor signal.

    For scenario slices (SIM1, ...), anchor parity is checked against BASE and
    we also require the requested slice to be present in Results.gdx.
    """
    gdx_path = Path(results_gdx)
    requested_slice = str(gams_slice).lower()
    anchor_slice = "base"

    anchor_recs = _read_symbol_records(gdxdump_bin, gdx_path, "valGDP_BP", anchor_slice)
    gams_gdp_bp = float(anchor_recs.get((), 0.0))
    py_gdp_bp = float(state.gdp.get("GDP_BPO", 0.0))
    abs_delta = abs(py_gdp_bp - gams_gdp_bp)
    denom = max(abs(py_gdp_bp), 1.0)
    rel_delta = abs_delta / denom

    anchor_passed = rel_delta <= rel_tol
    requested_recs = _read_symbol_records(gdxdump_bin, gdx_path, "valGDP_BP", requested_slice)
    requested_slice_has_records = bool(requested_recs)
    requested_slice_gdp_bp = float(requested_recs.get((), 0.0)) if requested_recs else 0.0

    if requested_slice == "base":
        passed = anchor_passed
    else:
        passed = anchor_passed and requested_slice_has_records

    return {
        "passed": passed,
        "rel_tol": rel_tol,
        "anchor_slice": anchor_slice,
        "requested_slice": requested_slice,
        "python_gdp_bp": py_gdp_bp,
        "gams_gdp_bp": gams_gdp_bp,
        "requested_slice_has_records": requested_slice_has_records,
        "requested_slice_gdp_bp": requested_slice_gdp_bp,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }


def classify_pipeline_outcome(
    *,
    sam_qa_report: Any | None,
    init_gates: dict[str, Any] | None,
    solve_report: dict[str, Any] | None,
    method: str,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Classify pipeline result for CRI diagnostics.

    Classification contract:
    - data_contract: failures caused by QA/initialization/parity contract violations.
    - solver_dynamics: failures only after passing init contracts.
    - pass: all requested gates passed.
    - unknown: runtime failure without clear attribution.
    """
    qa_passed: bool | None = None
    if sam_qa_report is not None:
        if isinstance(sam_qa_report, dict):
            qa_passed = bool(sam_qa_report.get("passed", False))
        else:
            qa_passed = bool(getattr(sam_qa_report, "passed", False))

    if qa_passed is False:
        return {
            "kind": "data_contract",
            "reason": "sam_qa_failed",
            "first_failed_block": None,
        }

    err = (error or "").lower()
    if err:
        if any(
            token in err
            for token in (
                "strict_gams",
                "baseline",
                "manifest",
                "initial",
                "sam qa",
                "compatibility",
            )
        ):
            return {
                "kind": "data_contract",
                "reason": "initialization_error",
                "first_failed_block": None,
            }
        if any(token in err for token in ("ipopt", "solve", "converg", "iteration", "residual")):
            return {
                "kind": "solver_dynamics",
                "reason": "solve_runtime_error",
                "first_failed_block": None,
            }
        return {
            "kind": "unknown",
            "reason": "runtime_error",
            "first_failed_block": None,
        }

    if init_gates is not None and not bool(init_gates.get("overall_passed", False)):
        return {
            "kind": "data_contract",
            "reason": "init_gate_failed",
            "first_failed_block": init_gates.get("first_failed_block"),
        }

    if method != "none":
        if solve_report is None:
            return {
                "kind": "solver_dynamics",
                "reason": "solve_missing_report",
                "first_failed_block": None,
            }
        if not bool(solve_report.get("converged", False)):
            return {
                "kind": "solver_dynamics",
                "reason": "solve_not_converged",
                "first_failed_block": None,
            }
        solve_gates = solve_report.get("gates") or {}
        if not bool(solve_gates.get("overall_passed", False)):
            return {
                "kind": "solver_dynamics",
                "reason": "solve_gate_failed",
                "first_failed_block": solve_gates.get("first_failed_block"),
            }
        return {
            "kind": "pass",
            "reason": "solve_and_gates_passed",
            "first_failed_block": None,
        }

    return {
        "kind": "pass",
        "reason": "init_gates_passed",
        "first_failed_block": None,
    }
