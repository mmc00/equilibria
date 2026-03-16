"""CRI SAM transformation aligned with the cge_babel balance-preserving pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from equilibria.sam_tools import SAMFormat
from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.parsers import export_sam, parse_sam
from equilibria.sam_tools.sam_transforms import (
    clear_agent_self_transfers_on_sam,
    collapse_margin_account_on_sam,
    collapse_tx_account_into_ti_on_sam,
    create_x_block_on_sam,
)

FixMode = Literal["auto", "on", "off"]

DEFAULT_COMMODITY_TO_SECTOR: dict[str, str] = {
    "agr": "agr",
    "othind": "ind",
    "ser": "ser",
    "food": "ind",
    "adm": "adm",
}


def _norm_text(value: Any) -> str:
    return str(value).strip()


def _norm_lower(value: Any) -> str:
    return _norm_text(value).lower()


def _ensure_key(df: pd.DataFrame, key: tuple[str, str]) -> pd.DataFrame:
    out = df
    if key not in out.index:
        new_index = out.index.append(pd.MultiIndex.from_tuples([key]))
        out = out.reindex(index=new_index, fill_value=0.0)
    if key not in out.columns:
        new_columns = out.columns.append(pd.MultiIndex.from_tuples([key]))
        out = out.reindex(columns=new_columns, fill_value=0.0)
    return out


def _sam_balance_stats(sam: Sam) -> dict[str, float]:
    df = sam.to_dataframe().astype(float)
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)
    return {
        "total": float(df.to_numpy().sum()),
        "max_row_col_abs_diff": float(np.max(np.abs(row_sums - col_sums))) if not df.empty else 0.0,
    }


def _convert_exports_to_x_fixed_on_sam(sam: Sam, *, tol: float = 1e-12) -> dict[str, float]:
    df = sam.to_dataframe()
    ag_row_col = ("AG", "row") if ("AG", "row") in df.columns else None
    if ag_row_col is None:
        return {"converted_commodities": 0, "total_export_value": 0.0}

    commodities = sorted({elem for cat, elem in df.index if _norm_lower(cat) == "i"})
    activities = sorted({elem for cat, elem in df.index if _norm_lower(cat) == "j"})
    converted = 0
    total_export = 0.0

    for commodity in commodities:
        i_key = ("I", commodity)
        x_key = ("X", commodity)
        df = _ensure_key(df, x_key)

        i_to_row = float(df.loc[i_key, ag_row_col]) if i_key in df.index else 0.0
        x_to_row = float(df.loc[x_key, ag_row_col]) if x_key in df.index else 0.0
        if abs(i_to_row) <= tol and abs(x_to_row) <= tol:
            continue

        export_total = x_to_row if abs(x_to_row) > tol else i_to_row
        df.loc[x_key, ag_row_col] = export_total
        if i_key in df.index and abs(i_to_row) > tol:
            df.loc[i_key, ag_row_col] = 0.0

        current_exo = sum(float(df.loc[("J", j), x_key]) for j in activities if ("J", j) in df.index)
        if abs(current_exo - export_total) <= tol * max(1.0, abs(export_total), abs(current_exo)):
            converted += 1
            total_export += export_total
            continue

        prod_by_j: dict[str, float] = {}
        denom = 0.0
        for j in activities:
            j_key = ("J", j)
            if j_key not in df.index or i_key not in df.columns:
                continue
            prod_val = float(df.loc[j_key, i_key])
            if prod_val > tol:
                prod_by_j[j] = prod_val
                denom += prod_val

        if denom <= tol:
            continue

        if export_total > denom + tol:
            export_total = denom
            df.loc[x_key, ag_row_col] = export_total

        for j in activities:
            j_key = ("J", j)
            if j_key in df.index and x_key in df.columns:
                df.loc[j_key, x_key] = 0.0

        for j, prod_val in prod_by_j.items():
            j_key = ("J", j)
            alloc = export_total * (prod_val / denom)
            df.loc[j_key, x_key] = alloc
            df.loc[j_key, i_key] = float(df.loc[j_key, i_key]) - alloc

        converted += 1
        total_export += export_total

    sam.replace_dataframe(df)
    return {"converted_commodities": converted, "total_export_value": total_export}


def _align_ti_to_gvt_j_preserve_balance_on_sam(sam: Sam) -> dict[str, float]:
    df = sam.to_dataframe()
    ti_row = ("AG", "ti")
    gvt_row = ("AG", "gvt")
    ti_col = ("AG", "ti")
    if ti_row not in df.index or gvt_row not in df.index:
        return {"moved_total": 0.0, "columns": 0, "ti_offset_applied": 0.0}

    moved = 0.0
    cols = 0
    for col_key in list(df.columns):
        if _norm_lower(col_key[0]) != "j":
            continue
        value = float(df.loc[ti_row, col_key])
        if abs(value) <= 1e-14:
            continue
        df.loc[ti_row, col_key] = 0.0
        df.loc[gvt_row, col_key] += value
        moved += value
        cols += 1

    if abs(moved) > 1e-14 and ti_col in df.columns:
        df.loc[gvt_row, ti_col] = float(df.loc[gvt_row, ti_col]) - moved

    sam.replace_dataframe(df)
    return {"moved_total": moved, "columns": cols, "ti_offset_applied": -moved}


def _rebucket_inputs_to_activity_columns_preserve_balance_on_sam(
    sam: Sam,
    *,
    source_categories: list[str],
    commodity_to_sector: dict[str, str],
    margin_commodities: list[str] | None = None,
) -> dict[str, Any]:
    df = sam.to_dataframe()
    source_cats = {_norm_lower(item) for item in source_categories}
    margin_rows = {_norm_lower(item) for item in (margin_commodities or [])}

    moved_total = 0.0
    cells_moved = 0
    moved_by_bridge: dict[str, float] = {}
    missing_targets: list[str] = []

    for col_key in list(df.columns):
        if _norm_lower(col_key[0]) != "i":
            continue
        commodity = _norm_lower(col_key[1])
        target_sector = _norm_lower(commodity_to_sector.get(commodity, "ind"))
        target_key = ("J", target_sector)
        df = _ensure_key(df, target_key)
        bridge_label = f"{target_key[0]}.{target_key[1]}->{col_key[0]}.{col_key[1]}"

        if target_key not in df.index or target_key not in df.columns:
            missing_targets.append(bridge_label)
            continue

        for row_key in list(df.index):
            row_cat = _norm_lower(row_key[0])
            if row_cat not in source_cats:
                continue
            if row_cat == "i" and _norm_lower(row_key[1]) in margin_rows:
                continue

            value = float(df.loc[row_key, col_key])
            if abs(value) <= 1e-14:
                continue

            df.loc[row_key, col_key] = 0.0
            df.loc[row_key, target_key] = float(df.loc[row_key, target_key]) + value
            df.loc[target_key, col_key] = float(df.loc[target_key, col_key]) + value

            moved_total += value
            cells_moved += 1
            moved_by_bridge[bridge_label] = moved_by_bridge.get(bridge_label, 0.0) + value

    sam.replace_dataframe(df)
    return {
        "source_categories": sorted(source_cats),
        "moved_total": moved_total,
        "bridge_added_total": moved_total,
        "cells_moved": cells_moved,
        "moved_by_bridge": moved_by_bridge,
        "missing_targets": sorted(set(missing_targets)),
    }


def _move_row_factor_inflows_to_owning_agents_preserve_balance_on_sam(
    sam: Sam,
    op: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = op or {}
    factor_categories = {
        _norm_lower(item)
        for item in cfg.get("factor_categories", ["K", "L"])
    }
    source_agent = _norm_lower(cfg.get("source_agent", "row"))
    source_col = ("AG", source_agent)
    excluded_agent_elems = {
        _norm_lower(item)
        for item in cfg.get("excluded_agent_elems", ["row", "td", "ti", "tm", "tx"])
    }

    df = sam.to_dataframe()
    if source_col not in df.columns:
        return {
            "source_agent": source_agent,
            "moved_total": 0.0,
            "moved_by_factor": {},
            "unresolved_factors": [],
            "offset_on_factor_columns": 0.0,
        }

    moved_total = 0.0
    moved_by_factor: dict[str, float] = {}
    unresolved_factors: list[str] = []
    factor_rows = [key for key in df.index if _norm_lower(key[0]) in factor_categories]
    for factor_key in factor_rows:
        value = float(df.loc[factor_key, source_col])
        if abs(value) <= 1e-14:
            continue
        if factor_key not in df.columns:
            unresolved_factors.append(f"{factor_key[0]}.{factor_key[1]}")
            continue

        weights: dict[tuple[str, str], float] = {}
        for row_key in df.index:
            if _norm_lower(row_key[0]) != "ag":
                continue
            if _norm_lower(row_key[1]) in excluded_agent_elems:
                continue
            weight = float(df.loc[row_key, factor_key])
            if weight > 1e-14:
                weights[row_key] = weight

        denom = float(sum(weights.values()))
        if denom <= 1e-14 or value > denom + 1e-9:
            unresolved_factors.append(f"{factor_key[0]}.{factor_key[1]}")
            continue

        df.loc[factor_key, source_col] = 0.0
        for row_key, weight in weights.items():
            alloc = value * (weight / denom)
            df.loc[row_key, source_col] = float(df.loc[row_key, source_col]) + alloc
            df.loc[row_key, factor_key] = float(df.loc[row_key, factor_key]) - alloc

        moved_total += value
        moved_by_factor[f"{factor_key[0]}.{factor_key[1]}"] = value

    sam.replace_dataframe(df)
    return {
        "source_agent": source_agent,
        "moved_total": moved_total,
        "moved_by_factor": moved_by_factor,
        "unresolved_factors": unresolved_factors,
        "offset_on_factor_columns": moved_total,
    }


def transform_sam_to_pep_compatible(
    input_sam: Path | str,
    output_sam: Path | str,
    *,
    report_json: Path | str | None = None,
    target_mode: str = "geomean",
    margin_commodity: str = "ser",
    epsilon: float = 1e-9,
    tol: float = 1e-8,
    max_iter: int = 20000,
) -> dict[str, Any]:
    input_path = Path(input_sam)
    output_path = Path(output_sam)
    report_path = Path(report_json) if report_json else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input SAM not found: {input_path}")

    sam = parse_sam(input_path, SAMFormat.EXCEL, {"symbol": "SAM"})
    steps: list[dict[str, Any]] = []

    def record(step: str, details: dict[str, Any] | None = None) -> None:
        steps.append(
            {
                "step": step,
                "details": details or {},
                "balance": _sam_balance_stats(sam),
            }
        )

    record("input_loaded", {"input_sam": str(input_path)})
    record("create_x_block", create_x_block_on_sam(sam))
    record("convert_exports", _convert_exports_to_x_fixed_on_sam(sam))
    record("align_ti", _align_ti_to_gvt_j_preserve_balance_on_sam(sam))
    record("clear_self_transfers", clear_agent_self_transfers_on_sam(sam, {"agents": ["gvt", "row"]}))
    record(
        "move_k",
        _rebucket_inputs_to_activity_columns_preserve_balance_on_sam(
            sam,
            source_categories=["K"],
            commodity_to_sector=DEFAULT_COMMODITY_TO_SECTOR,
        ),
    )
    record(
        "move_l",
        _rebucket_inputs_to_activity_columns_preserve_balance_on_sam(
            sam,
            source_categories=["L"],
            commodity_to_sector=DEFAULT_COMMODITY_TO_SECTOR,
        ),
    )
    record(
        "collapse_marg",
        collapse_margin_account_on_sam(
            sam,
            {"margin_commodity": margin_commodity, "strict": True},
        ),
    )
    record("collapse_tx", collapse_tx_account_into_ti_on_sam(sam))
    record(
        "move_row_factor_inflows",
        _move_row_factor_inflows_to_owning_agents_preserve_balance_on_sam(sam),
    )
    record(
        "move_nonmargin_i_to_ji",
        _rebucket_inputs_to_activity_columns_preserve_balance_on_sam(
            sam,
            source_categories=["I"],
            commodity_to_sector=DEFAULT_COMMODITY_TO_SECTOR,
            margin_commodities=[margin_commodity],
        ),
    )

    export_sam(sam, output_path, output_format="excel", output_symbol="SAM")

    report: dict[str, Any] = {
        "input_sam": str(input_path),
        "output_sam": str(output_path),
        "pipeline": "cgebabel_balance_preserving_from_pep_layout",
        "settings": {
            "target_mode": target_mode,
            "margin_commodity": margin_commodity,
            "epsilon": epsilon,
            "tol": tol,
            "max_iter": max_iter,
        },
        "steps": steps,
        "after": {
            "balance": _sam_balance_stats(sam),
        },
    }

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def should_apply_cri_pep_fix(sam_file: Path | str, mode: FixMode = "auto") -> bool:
    mode_norm = mode.strip().lower()
    if mode_norm == "off":
        return False

    sam_path = Path(sam_file)
    if sam_path.suffix.lower() not in {".xlsx", ".xls"}:
        return False

    if mode_norm == "on":
        return True
    if mode_norm != "auto":
        raise ValueError(f"Unknown fix mode: {mode}")

    name = sam_path.name.lower()
    if "pep-compatible" in name or "fixed" in name:
        return False
    return "sam-cri" in name


__all__ = [
    "DEFAULT_COMMODITY_TO_SECTOR",
    "should_apply_cri_pep_fix",
    "transform_sam_to_pep_compatible",
]
