"""Utility transforms over a ``Sam`` matrix without extra state."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.selectors import (
    indices_for_selector,
    norm_text,
    norm_text_lower,
)

DEFAULT_COMMODITY_TO_SECTOR: dict[str, str] = {
    "agr": "agr",
    "othind": "ind",
    "ser": "ser",
    "food": "ind",
    "adm": "adm",
}
SPECIAL_LABELS = {"S-HH", "S-FIRM", "S-GVT", "S-ROW", "INV", "VSTK"}


def _add_value(matrix: np.ndarray, key_index: dict[tuple[str, str], int], row_key: tuple[str, str], col_key: tuple[str, str], value: float) -> None:
    if abs(value) <= 1e-14:
        return
    if row_key not in key_index or col_key not in key_index:
        return
    matrix[key_index[row_key], key_index[col_key]] += float(value)


def _label_to_key(label: str) -> tuple[str, str] | None:
    label_up = norm_text(label).upper()
    if label_up in SPECIAL_LABELS:
        return None
    if label_up.startswith("A-"):
        return ("J", label_up[2:].lower())
    if label_up.startswith("C-"):
        return ("I", label_up[2:].lower())
    if label_up in {"USK", "SK"}:
        return ("L", label_up.lower())
    if label_up in {"CAP", "LAND"}:
        return ("K", label_up.lower())
    if label_up in {"HRP", "HRR", "HUP", "HUR", "FIRM", "GVT", "ROW", "TD", "TI", "TM", "TX"}:
        return ("AG", label_up.lower())
    if label_up == "MARG":
        return ("MARG", "MARG")
    return None


def _build_pep_key_order(labels: list[str]) -> list[tuple[str, str]]:
    j_keys, i_keys, l_keys, k_keys, ag_keys = [], [], [], [], []
    has_marg = False
    for label in labels:
        key = _label_to_key(label)
        if not key:
            continue
        cat = key[0]
        if cat == "J" and key not in j_keys:
            j_keys.append(key)
        elif cat == "I" and key not in i_keys:
            i_keys.append(key)
        elif cat == "L" and key not in l_keys:
            l_keys.append(key)
        elif cat == "K" and key not in k_keys:
            k_keys.append(key)
        elif cat == "AG" and key not in ag_keys:
            ag_keys.append(key)
        elif cat == "MARG":
            has_marg = True
    ag_order = ["ti", "tm", "tx", "td", "hrp", "hrr", "hup", "hur", "firm", "gvt", "row"]
    ag_sorted = [("AG", elem) for elem in ag_order if ("AG", elem) in ag_keys]
    keys = j_keys + i_keys + l_keys + k_keys + ag_sorted
    if has_marg:
        keys.append(("MARG", "MARG"))
    keys.extend([("OTH", "INV"), ("OTH", "VSTK")])
    return keys


def _locate_key(keys: list[tuple[str, str]], target: tuple[str, str]) -> tuple[str, str] | None:
    target_norm = (norm_text_lower(target[0]), norm_text_lower(target[1]))
    for key in keys:
        key_norm = (norm_text_lower(key[0]), norm_text_lower(key[1]))
        if key_norm == target_norm:
            return key
    return None


def _ensure_key(sam: Sam, key: tuple[str, str]) -> bool:
    if key in sam.row_keys:
        return False
    df = sam.to_dataframe()
    index = sam.row_keys
    new_index = pd.MultiIndex.from_tuples(index + [key])
    new_df = pd.DataFrame(0.0, index=new_index, columns=new_index)
    new_df.loc[df.index, df.columns] = df
    sam.replace_dataframe(new_df)
    return True


def ensure_key(sam: Sam, key: tuple[str, str]) -> bool:
    return _ensure_key(sam, key)


def create_x_block_on_sam(sam: Sam) -> dict[str, int]:
    commodities = [elem for cat, elem in sam.row_keys if norm_text_lower(cat) == "i"]
    added = sum(1 for commodity in commodities if ensure_key(sam, ("X", commodity)))
    return {"commodities": len(commodities), "added_x_accounts": added}


def convert_exports_to_x_on_sam(sam: Sam) -> dict[str, float]:
    df = sam.to_dataframe()
    if ("AG", "row") not in df.columns:
        return {"converted_commodities": 0, "total_export_value": 0.0}
    j_rows = [key for key in sam.row_keys if norm_text_lower(key[0]) == "j"]
    converted = 0
    total_export = 0.0
    for commodity in [elem for cat, elem in sam.row_keys if norm_text_lower(cat) == "i"]:
        i_key = ("I", commodity)
        x_key = ("X", commodity)
        if i_key not in df.index or x_key not in df.index:
            continue
        export_value = float(df.loc[i_key, ("AG", "row")])
        if abs(export_value) <= 1e-14:
            continue
        df.loc[i_key, ("AG", "row")] = 0.0
        df.loc[x_key, ("AG", "row")] += export_value
        supply = [
            (j_key, float(df.loc[j_key, i_key]))
            for j_key in j_rows
            if float(df.loc[j_key, i_key]) > 0
        ]
        total_supply = sum(v for _, v in supply)
        if total_supply > 1e-14:
            for j_key, value in supply:
                moved = export_value * (value / total_supply)
                df.loc[j_key, i_key] -= moved
                df.loc[j_key, x_key] += moved
        converted += 1
        total_export += export_value
    sam.replace_dataframe(df)
    return {"converted_commodities": converted, "total_export_value": total_export}


def move_margin_to_i_margin_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = op or {}
    margin_commodity = norm_text(cfg.get("margin_commodity", "ser"))
    margin_row = ("MARG", "MARG")
    margin_target = ("I", norm_text(margin_commodity))
    df = sam.to_dataframe()
    if margin_row not in df.index or margin_target not in df.index:
        return {"margin_commodity": margin_target[1], "moved_total": 0.0, "columns": 0}

    commodity_cols = [(idx, key) for idx, key in enumerate(sam.col_keys) if norm_text_lower(key[0]) == "i"]
    moved_total = 0.0
    for c_idx, col_key in commodity_cols:
        value = float(df.loc[margin_row, col_key])
        if abs(value) <= 1e-14:
            continue
        df.loc[margin_row, col_key] = 0.0
        df.loc[margin_target, col_key] += value
        moved_total += value

    sam.replace_dataframe(df)
    return {
        "margin_commodity": margin_target[1],
        "moved_total": moved_total,
        "columns": len(commodity_cols),
    }


def move_tx_to_ti_on_i_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = op or {}
    row_tx = ("AG", "tx")
    row_ti = ("AG", "ti")
    df = sam.to_dataframe()
    if row_tx not in df.index or row_ti not in df.index:
        return {"moved_total": 0.0, "columns": 0}

    commodity_cols = [(idx, key) for idx, key in enumerate(sam.col_keys) if norm_text_lower(key[0]) == "i"]
    moved_total = 0.0
    cols_moved = 0
    for c_idx, col_key in commodity_cols:
        value = float(df.loc[row_tx, col_key])
        if abs(value) <= 1e-14:
            continue
        df.loc[row_tx, col_key] = 0.0
        df.loc[row_ti, col_key] += value
        moved_total += value
        cols_moved += 1

    sam.replace_dataframe(df)
    return {
        "moved_total": moved_total,
        "columns": cols_moved,
    }


def collapse_margin_account_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = op or {}
    margin_key = ("MARG", "MARG")
    margin_commodity = norm_text(cfg.get("margin_commodity", "ser"))
    margin_target = ("I", margin_commodity)
    strict = bool(cfg.get("strict", False))

    df = sam.to_dataframe()
    if margin_key not in df.index or margin_target not in df.index:
        return {
            "margin_commodity": margin_target[1],
            "moved_total": 0.0,
            "cleared_column_total": 0.0,
            "active_columns": 0,
            "unexpected_inflows": [],
        }

    moved_total = 0.0
    active_columns = 0
    for col_key in list(df.columns):
        if col_key == margin_key:
            continue
        value = float(df.loc[margin_key, col_key])
        if abs(value) <= 1e-14:
            continue
        df.loc[margin_target, col_key] += value
        df.loc[margin_key, col_key] = 0.0
        moved_total += value
        active_columns += 1

    cleared_column_total = 0.0
    unexpected_inflows: list[dict[str, Any]] = []
    if margin_key in df.columns:
        for row_key in list(df.index):
            value = float(df.loc[row_key, margin_key])
            if abs(value) <= 1e-14:
                continue
            if row_key == margin_target:
                df.loc[row_key, margin_key] = 0.0
                cleared_column_total += value
                continue
            unexpected_inflows.append(
                {
                    "row": f"{row_key[0]}.{row_key[1]}",
                    "value": value,
                }
            )
        if strict and unexpected_inflows:
            rows = ", ".join(item["row"] for item in unexpected_inflows)
            raise ValueError(f"unexpected inflows on MARG column: {rows}")

    sam.replace_dataframe(df)
    return {
        "margin_commodity": margin_target[1],
        "moved_total": moved_total,
        "cleared_column_total": cleared_column_total,
        "active_columns": active_columns,
        "unexpected_inflows": unexpected_inflows,
    }


def collapse_tx_account_into_ti_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = op or {}
    tx_key = ("AG", "tx")
    ti_key = ("AG", "ti")

    df = sam.to_dataframe()
    if tx_key not in df.index or ti_key not in df.index or tx_key not in df.columns or ti_key not in df.columns:
        return {
            "row_moved_total": 0.0,
            "column_moved_total": 0.0,
            "self_overlap": 0.0,
        }

    row_values = df.loc[tx_key, :].copy()
    col_values = df.loc[:, tx_key].copy()
    self_overlap = float(df.loc[tx_key, tx_key])

    df.loc[ti_key, :] = df.loc[ti_key, :] + row_values
    df.loc[:, ti_key] = df.loc[:, ti_key] + col_values
    if abs(self_overlap) > 1e-14:
        df.loc[ti_key, ti_key] = float(df.loc[ti_key, ti_key]) - self_overlap

    df.loc[tx_key, :] = 0.0
    df.loc[:, tx_key] = 0.0
    sam.replace_dataframe(df)
    return {
        "row_moved_total": float(row_values.sum()),
        "column_moved_total": float(col_values.sum()),
        "self_overlap": self_overlap,
    }


def _move_factor_to_ji_on_sam(
    sam: Sam,
    *,
    factor_category: str,
    op: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import ensure_key

    cfg = op or {}
    raw_mapping = cfg.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    commodity_to_sector = {
        norm_text_lower(k): norm_text(v)
        for k, v in raw_mapping.items()
    }
    default_sector = norm_text(cfg.get("default_sector", "ind"))
    strict_targets = bool(cfg.get("strict_targets", True))

    factor_keys = [key for key in sam.row_keys if norm_text_lower(key[0]) == norm_text_lower(factor_category)]
    moved_by_commodity: dict[str, float] = {}
    missing_targets: list[str] = []

    for col_key in [key for key in sam.col_keys if norm_text_lower(key[0]) == "i"]:
        commodity = norm_text(col_key[1])
        sector = commodity_to_sector.get(norm_text_lower(commodity), default_sector)
        target_key = ("J", sector)
        if target_key not in sam.row_keys:
            added = ensure_key(sam, target_key)
            if not added and strict_targets:
                raise ValueError(f"target_row[{commodity}] missing for {target_key}")
            if not added:
                missing_targets.append(f"J.{sector}")
                continue
        df = sam.to_dataframe()
        moved_col = 0.0
        for row_key in factor_keys:
            value = float(df.loc[row_key, col_key])
            if abs(value) <= 1e-14:
                continue
            df.loc[row_key, col_key] = 0.0
            df.loc[target_key, col_key] += value
            moved_col += value
        if abs(moved_col) > 1e-14:
            moved_by_commodity[commodity] = moved_col
        sam.replace_dataframe(df)

    df = sam.to_dataframe()
    moved_total = float(sum(moved_by_commodity.values()))
    return {
        "source_category": factor_category.upper(),
        "sources": len(factor_keys),
        "commodities": len([key for key in df.columns if norm_text_lower(key[0]) == "i"]),
        "default_sector": default_sector,
        "moved_total": moved_total,
        "moved_by_commodity": moved_by_commodity,
        "missing_targets": missing_targets,
    }


def move_k_to_ji_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    return _move_factor_to_ji_on_sam(sam, factor_category="K", op=op)


def move_l_to_ji_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    return _move_factor_to_ji_on_sam(sam, factor_category="L", op=op)


def move_nonmargin_i_to_ji_on_sam(sam: Sam, op: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = op or {}
    raw_mapping = cfg.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    commodity_to_sector = {
        norm_text_lower(k): norm_text(v)
        for k, v in raw_mapping.items()
    }
    default_sector = norm_text(cfg.get("default_sector", "ind"))
    margin_commodities = cfg.get("margin_commodities")
    if margin_commodities is None:
        margin_commodities = [cfg.get("margin_commodity", "ser")]
    margin_rows = {norm_text_lower(item) for item in margin_commodities}
    strict_targets = bool(cfg.get("strict_targets", True))

    df = sam.to_dataframe()
    moved_total = 0.0
    moved_by_source: dict[str, float] = {}
    cells_moved = 0
    missing_targets: list[str] = []

    commodity_rows = [key for key in sam.row_keys if norm_text_lower(key[0]) == "i"]
    commodity_cols = [key for key in sam.col_keys if norm_text_lower(key[0]) == "i"]
    for row_key in commodity_rows:
        commodity = norm_text(row_key[1])
        if norm_text_lower(commodity) in margin_rows:
            continue
        sector = commodity_to_sector.get(norm_text_lower(commodity), default_sector)
        target_key = ("J", sector)
        if target_key not in df.index:
            if strict_targets:
                raise ValueError(f"target_row[{commodity}] missing for {target_key}")
            missing_targets.append(f"J.{sector}")
            continue
        moved_source_total = 0.0
        for col_key in commodity_cols:
            value = float(df.loc[row_key, col_key])
            if abs(value) <= 1e-14:
                continue
            df.loc[target_key, col_key] += value
            df.loc[row_key, col_key] = 0.0
            moved_total += value
            moved_source_total += value
            cells_moved += 1
        if abs(moved_source_total) > 1e-14:
            moved_by_source[commodity] = moved_source_total

    sam.replace_dataframe(df)
    return {
        "margin_commodities": sorted(margin_rows),
        "moved_total": moved_total,
        "cells_moved": cells_moved,
        "moved_by_source": moved_by_source,
        "missing_targets": missing_targets,
    }


def move_row_factor_inflows_to_owning_agents_on_sam(
    sam: Sam,
    op: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = op or {}
    factor_categories = {
        norm_text_lower(item)
        for item in cfg.get("factor_categories", ["K", "L"])
    }
    source_agent = norm_text(cfg.get("source_agent", "row"))
    source_col = ("AG", source_agent)
    excluded_agent_elems = {
        norm_text_lower(item)
        for item in cfg.get("excluded_agent_elems", ["row", "td", "ti", "tm", "tx"])
    }

    df = sam.to_dataframe()
    if source_col not in df.columns:
        return {
            "source_agent": source_agent,
            "moved_total": 0.0,
            "moved_by_factor": {},
            "unresolved_factors": [],
        }

    moved_total = 0.0
    moved_by_factor: dict[str, float] = {}
    unresolved_factors: list[str] = []
    factor_rows = [key for key in sam.row_keys if norm_text_lower(key[0]) in factor_categories]
    for factor_key in factor_rows:
        value = float(df.loc[factor_key, source_col])
        if abs(value) <= 1e-14:
            continue
        if factor_key not in df.columns:
            unresolved_factors.append(f"{factor_key[0]}.{factor_key[1]}")
            continue
        weights: dict[tuple[str, str], float] = {}
        for row_key in sam.row_keys:
            if norm_text_lower(row_key[0]) != "ag":
                continue
            if norm_text_lower(row_key[1]) in excluded_agent_elems:
                continue
            weight = float(df.loc[row_key, factor_key])
            if weight > 1e-14:
                weights[row_key] = weight
        if not weights:
            unresolved_factors.append(f"{factor_key[0]}.{factor_key[1]}")
            continue
        denom = float(sum(weights.values()))
        df.loc[factor_key, source_col] = 0.0
        for row_key, weight in weights.items():
            df.loc[row_key, source_col] += value * (weight / denom)
        moved_total += value
        moved_by_factor[f"{factor_key[0]}.{factor_key[1]}"] = value

    sam.replace_dataframe(df)
    return {
        "source_agent": source_agent,
        "moved_total": moved_total,
        "moved_by_factor": moved_by_factor,
        "unresolved_factors": unresolved_factors,
    }


def clear_agent_self_transfers_on_sam(
    sam: Sam,
    op: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = op or {}
    category = norm_text(cfg.get("category", "AG")).upper()
    raw_agents = cfg.get("agents")
    if raw_agents is None:
        raw_agents = [cfg.get("agent", "gvt")]

    df = sam.to_dataframe()
    removed_total = 0.0
    removed_by_agent: dict[str, float] = {}

    for agent in raw_agents:
        key = (category, norm_text(agent).lower())
        if key not in df.index or key not in df.columns:
            continue
        value = float(df.loc[key, key])
        if abs(value) <= 1e-14:
            continue
        df.loc[key, key] = 0.0
        removed_total += value
        removed_by_agent[f"{key[0]}.{key[1]}"] = value

    sam.replace_dataframe(df)
    return {
        "category": category,
        "agents": [norm_text(agent).lower() for agent in raw_agents],
        "removed_total": removed_total,
        "removed_by_agent": removed_by_agent,
    }


def align_ti_to_gvt_j_on_sam(sam: Sam) -> dict[str, float]:
    df = sam.to_dataframe()
    ti_row = ("AG", "ti")
    gvt_row = ("AG", "gvt")
    if ti_row not in df.index or gvt_row not in df.index:
        return {"moved_total": 0.0, "columns": 0}
    moved = 0.0
    cols = 0
    for col_idx, col_key in enumerate(sam.col_keys):
        if norm_text_lower(col_key[0]) != "j":
            continue
        value = float(df.iat[df.index.get_loc(ti_row), col_idx])
        if abs(value) <= 1e-14:
            continue
        df.iat[df.index.get_loc(ti_row), col_idx] = 0.0
        df.iat[df.index.get_loc(gvt_row), col_idx] += value
        moved += value
        cols += 1
    sam.replace_dataframe(df)
    return {"moved_total": moved, "columns": cols}


def normalize_pep_accounts_on_sam(sam: Sam) -> dict[str, int]:
    df = sam.to_dataframe()
    raw_index = list(df.index)
    raw_columns = list(df.columns)
    labels = [norm_text(label[1]) for label in df.index]
    pep_keys = _build_pep_key_order(labels)
    key_index = {key: idx for idx, key in enumerate(pep_keys)}
    pep_matrix = np.zeros((len(pep_keys), len(pep_keys)), dtype=float)
    for r_label in df.index:
        r_key = _label_to_key(norm_text(r_label[1]))
        if r_key is None:
            continue
        for c_label in df.columns:
            c_key = _label_to_key(norm_text(c_label[1]))
            if c_key is None:
                continue
            _add_value(pep_matrix, key_index, r_key, c_key, float(df.loc[r_label, c_label]))
    savings_to_agent = {
        "S-HH": ["HRP", "HRR", "HUP", "HUR"],
        "S-FIRM": ["FIRM"],
        "S-GVT": ["GVT"],
        "S-ROW": ["ROW"],
    }
    for savings_row, agents in savings_to_agent.items():
        row_key = _locate_key(raw_index, ("RAW", savings_row))
        if row_key is None:
            continue
        for agent in agents:
            raw_col_key = _locate_key(raw_columns, ("RAW", agent))
            if raw_col_key is None:
                continue
            value = float(df.loc[row_key, raw_col_key])
            _add_value(pep_matrix, key_index, ("OTH", "INV"), ("AG", agent.lower()), value)
    if ("OTH", "VSTK") in df.index:
        vstk_total = float(df.loc[("OTH", "VSTK"), :].sum())
        _add_value(pep_matrix, key_index, ("OTH", "VSTK"), ("OTH", "INV"), vstk_total)
    for key in pep_keys:
        if key[0] != "I":
            continue
        commodity = key[1]
        c_label = f"C-{commodity.upper()}"
        raw_key = _locate_key(raw_index, ("RAW", c_label))
        if raw_key is not None:
            for raw_target, pep_target in [
                (("RAW", "INV"), ("OTH", "INV")),
                (("RAW", "VSTK"), ("OTH", "VSTK")),
            ]:
                raw_target_key = _locate_key(raw_columns, raw_target)
                if raw_target_key is None:
                    continue
                _add_value(
                    pep_matrix,
                    key_index,
                    ("I", commodity),
                    pep_target,
                    float(df.loc[raw_key, raw_target_key]),
                )
    multi_index = pd.MultiIndex.from_tuples(pep_keys)
    sam.replace_dataframe(pd.DataFrame(pep_matrix, index=multi_index, columns=multi_index))
    return {
        "raw_labels": len(labels),
        "pep_accounts": len(pep_keys),
        "commodities": len([k for k in pep_keys if k[0] == "I"]),
    }
