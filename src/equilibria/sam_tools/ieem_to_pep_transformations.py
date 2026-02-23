"""IEEM -> PEP structural transformations for SAM matrices.

These operations move flows from unsupported IEEM cells into PEP-compatible
channels before rebalancing.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.selectors import (
    index_for_key,
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


class SAMLikeState(Protocol):
    """Protocol expected by IEEM->PEP transforms."""

    matrix: np.ndarray
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]


_RAS_BALANCER = RASBalancer()


def _raw_state_to_dataframe(state: SAMLikeState) -> pd.DataFrame:
    if not state.row_keys or not state.col_keys:
        raise ValueError("State has no support keys")
    if any(norm_text_lower(cat) != "raw" for cat, _ in state.row_keys):
        raise ValueError("State rows must be category 'RAW' for this operation")
    if any(norm_text_lower(cat) != "raw" for cat, _ in state.col_keys):
        raise ValueError("State cols must be category 'RAW' for this operation")

    labels_r = [norm_text(elem) for _cat, elem in state.row_keys]
    labels_c = [norm_text(elem) for _cat, elem in state.col_keys]
    return pd.DataFrame(state.matrix.copy(), index=labels_r, columns=labels_c, dtype=float)


def _replace_state_from_raw_dataframe(state: SAMLikeState, df: pd.DataFrame) -> None:
    labels = [norm_text(label) for label in df.index]
    state.matrix = df.to_numpy(dtype=float)
    state.row_keys = [("RAW", label) for label in labels]
    state.col_keys = [("RAW", label) for label in labels]


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
    j_keys: list[tuple[str, str]] = []
    i_keys: list[tuple[str, str]] = []
    l_keys: list[tuple[str, str]] = []
    k_keys: list[tuple[str, str]] = []
    ag_keys: list[tuple[str, str]] = []
    has_marg = False

    for label in labels:
        key = _label_to_key(label)
        if key is None:
            continue
        cat, _ = key
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

    keys: list[tuple[str, str]] = []
    keys.extend(j_keys)
    keys.extend(i_keys)
    keys.extend(l_keys)
    keys.extend(k_keys)
    keys.extend(ag_sorted)
    if has_marg:
        keys.append(("MARG", "MARG"))
    keys.extend([("OTH", "INV"), ("OTH", "VSTK")])
    return keys


def _ensure_key(state: SAMLikeState, key: tuple[str, str]) -> bool:
    if key in state.row_keys:
        return False
    old_n = state.matrix.shape[0]
    new_matrix = np.zeros((old_n + 1, old_n + 1), dtype=float)
    new_matrix[:old_n, :old_n] = state.matrix
    state.matrix = new_matrix
    state.row_keys = state.row_keys + [key]
    state.col_keys = state.col_keys + [key]
    return True


def _add_value(
    matrix: np.ndarray,
    key_index: dict[tuple[str, str], int],
    row_key: tuple[str, str],
    col_key: tuple[str, str],
    value: float,
) -> None:
    if abs(value) <= 1e-14:
        return
    if row_key not in key_index or col_key not in key_index:
        return
    matrix[key_index[row_key], key_index[col_key]] += float(value)


def balance_state_ras(state: SAMLikeState, op: dict[str, Any]) -> dict[str, Any]:
    """RAS rebalance on RAW matrices before IEEM->PEP normalization."""

    tol = float(op.get("tol", op.get("tolerance", 1e-9)))
    max_iter = int(op.get("max_iter", 200))
    ras_type = str(op.get("ras_type", "arithmetic"))
    df = _raw_state_to_dataframe(state)
    result = _RAS_BALANCER.balance_dataframe(
        df,
        ras_type=ras_type,
        tolerance=tol,
        max_iterations=max_iter,
    )
    _replace_state_from_raw_dataframe(state, result.matrix)
    return {
        "tol": tol,
        "max_iter": max_iter,
        "ras_type": result.ras_type,
        "iterations": result.iterations,
        "converged": result.converged,
        "max_diff_before": result.max_diff_before,
        "max_diff_after": result.max_diff_after,
    }


def normalize_state_to_pep_accounts(
    state: SAMLikeState,
    _op: dict[str, Any],
) -> dict[str, Any]:
    """Map aggregated RAW labels into canonical PEP accounts."""

    df = _raw_state_to_dataframe(state)
    labels = [norm_text(label) for label in df.index]
    pep_keys = _build_pep_key_order(labels)
    key_index = {k: i for i, k in enumerate(pep_keys)}
    pep_matrix = np.zeros((len(pep_keys), len(pep_keys)), dtype=float)

    for r_label in df.index:
        r_key = _label_to_key(norm_text(r_label))
        if r_key is None:
            continue
        for c_label in df.columns:
            c_key = _label_to_key(norm_text(c_label))
            if c_key is None:
                continue
            _add_value(pep_matrix, key_index, r_key, c_key, float(df.loc[r_label, c_label]))

    savings_to_agent: dict[str, list[str]] = {
        "S-HH": ["HRP", "HRR", "HUP", "HUR"],
        "S-FIRM": ["FIRM"],
        "S-GVT": ["GVT"],
        "S-ROW": ["ROW"],
    }
    for savings_row, agents in savings_to_agent.items():
        if savings_row not in df.index:
            continue
        for agent in agents:
            if agent not in df.columns:
                continue
            value = float(df.loc[savings_row, agent])
            _add_value(pep_matrix, key_index, ("OTH", "INV"), ("AG", agent.lower()), value)

    if "VSTK" in df.index:
        vstk_total = float(df.loc["VSTK", :].sum())
        _add_value(pep_matrix, key_index, ("OTH", "VSTK"), ("OTH", "INV"), vstk_total)

    for key in pep_keys:
        if key[0] != "I":
            continue
        commodity = key[1]
        c_label = f"C-{commodity.upper()}"
        if c_label in df.index and "INV" in df.columns:
            _add_value(
                pep_matrix,
                key_index,
                ("I", commodity),
                ("OTH", "INV"),
                float(df.loc[c_label, "INV"]),
            )
        if c_label in df.index and "VSTK" in df.columns:
            _add_value(
                pep_matrix,
                key_index,
                ("I", commodity),
                ("OTH", "VSTK"),
                float(df.loc[c_label, "VSTK"]),
            )

    state.matrix = pep_matrix
    state.row_keys = pep_keys
    state.col_keys = pep_keys.copy()
    return {
        "raw_labels": len(labels),
        "pep_accounts": len(pep_keys),
        "commodities": len([k for k in pep_keys if k[0] == "I"]),
    }


def create_x_block(state: SAMLikeState, _op: dict[str, Any]) -> dict[str, Any]:
    """Create one X.* export row/column per commodity I.*."""

    commodities = [elem for cat, elem in state.row_keys if norm_text_lower(cat) == "i"]
    added = 0
    for commodity in commodities:
        if _ensure_key(state, ("X", commodity)):
            added += 1
    return {"commodities": len(commodities), "added_x_accounts": added}


def convert_exports_to_x(state: SAMLikeState, _op: dict[str, Any]) -> dict[str, Any]:
    """Route exports through X.* and split activity supply into J->I/J->X."""

    key_index = {k: i for i, k in enumerate(state.row_keys)}
    if ("AG", "row") not in key_index:
        return {"converted_commodities": 0, "total_export_value": 0.0}

    ag_row_col = key_index[("AG", "row")]
    commodities = [elem for cat, elem in state.row_keys if norm_text_lower(cat) == "i"]
    j_rows = [k for k in state.row_keys if norm_text_lower(k[0]) == "j"]

    converted = 0
    total_export = 0.0
    for commodity in commodities:
        i_key = ("I", commodity)
        x_key = ("X", commodity)
        if i_key not in key_index or x_key not in key_index:
            continue
        i_row = key_index[i_key]
        x_row = key_index[x_key]
        export_value = float(state.matrix[i_row, ag_row_col])
        if abs(export_value) <= 1e-14:
            continue

        state.matrix[i_row, ag_row_col] = 0.0
        state.matrix[x_row, ag_row_col] += export_value

        i_col = key_index[i_key]
        x_col = key_index[x_key]
        j_supply: list[tuple[int, float]] = []
        for j_key in j_rows:
            j_idx = key_index[j_key]
            value = float(state.matrix[j_idx, i_col])
            if value > 0:
                j_supply.append((j_idx, value))
        total_supply = float(sum(v for _, v in j_supply))
        if total_supply > 1e-14:
            for j_idx, value in j_supply:
                moved = export_value * (value / total_supply)
                state.matrix[j_idx, i_col] -= moved
                state.matrix[j_idx, x_col] += moved

        converted += 1
        total_export += export_value

    return {"converted_commodities": converted, "total_export_value": total_export}


def align_ti_to_gvt_j(state: SAMLikeState, _op: dict[str, Any]) -> dict[str, Any]:
    """Move AG.ti -> J.* entries into AG.gvt -> J.*."""

    key_index = {k: i for i, k in enumerate(state.row_keys)}
    if ("AG", "ti") not in key_index or ("AG", "gvt") not in key_index:
        return {"moved_total": 0.0, "columns": 0}

    ti_row = key_index[("AG", "ti")]
    gvt_row = key_index[("AG", "gvt")]
    moved = 0.0
    cols = 0
    for key in state.col_keys:
        if norm_text_lower(key[0]) != "j":
            continue
        col = key_index[key]
        value = float(state.matrix[ti_row, col])
        if abs(value) <= 1e-14:
            continue
        state.matrix[ti_row, col] = 0.0
        state.matrix[gvt_row, col] += value
        moved += value
        cols += 1
    return {"moved_total": moved, "columns": cols}


def _commodity_columns(state: SAMLikeState) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    for idx, (cat, elem) in enumerate(state.col_keys):
        if norm_text_lower(cat) == "i":
            columns.append((idx, norm_text(elem)))
    return columns


def _row_indices_by_category(state: SAMLikeState, category: str) -> list[int]:
    cat_norm = norm_text_lower(category)
    return [
        idx
        for idx, (cat, _elem) in enumerate(state.row_keys)
        if norm_text_lower(cat) == cat_norm
    ]


def apply_move_k_to_ji(state: SAMLikeState, op: dict[str, Any]) -> dict[str, Any]:
    """Move capital-factor inflows on commodity columns to sector supply rows.

    Economic intuition:
    - PEP expects commodity supply to come from activities (`J`), not directly
      from primary factors (`K`).
    - This operation takes any `K.* -> I.i` flow and reallocates it to
      `J.map(i) -> I.i`.
    """

    raw_mapping = op.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    if not isinstance(raw_mapping, dict):
        raise ValueError("commodity_to_sector must be a mapping")
    commodity_to_sector = {
        norm_text_lower(k): norm_text(v) for k, v in raw_mapping.items()
    }
    default_sector = norm_text(op.get("default_sector", "ind"))
    strict_targets = bool(op.get("strict_targets", True))

    k_rows = _row_indices_by_category(state, "K")
    commodity_cols = _commodity_columns(state)
    moved_by_commodity: dict[str, float] = {}
    missing_targets: list[str] = []
    target_cache: dict[tuple[str, str], int] = {}

    for c_idx, commodity in commodity_cols:
        sector = commodity_to_sector.get(norm_text_lower(commodity), default_sector)
        target_key = ("J", sector)
        if target_key not in target_cache:
            try:
                target_cache[target_key] = index_for_key(
                    state.row_keys, target_key, f"target_row[{commodity}]"
                )
            except ValueError:
                if strict_targets:
                    raise
                missing_targets.append(f"J.{sector}")
                continue
        target_row = target_cache[target_key]

        moved_col = 0.0
        for r_idx in k_rows:
            value = float(state.matrix[r_idx, c_idx])
            if abs(value) <= 1e-14:
                continue
            state.matrix[r_idx, c_idx] = 0.0
            state.matrix[target_row, c_idx] += value
            moved_col += value
        if abs(moved_col) > 1e-14:
            moved_by_commodity[commodity] = moved_col

    moved_total = float(sum(moved_by_commodity.values()))
    return {
        "source_category": "K",
        "sources": len(k_rows),
        "commodities": len(commodity_cols),
        "default_sector": default_sector,
        "moved_total": moved_total,
        "moved_by_commodity": moved_by_commodity,
        "missing_targets": missing_targets,
    }


def apply_move_l_to_ji(state: SAMLikeState, op: dict[str, Any]) -> dict[str, Any]:
    """Move labor-factor inflows on commodity columns to sector supply rows.

    Economic intuition:
    - Igual que con capital, PEP no interpreta `L.* -> I.i` como oferta válida
      de commodities.
    - Se reasignan a `J.map(i) -> I.i` para que la oferta venga de actividades.
    """

    raw_mapping = op.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    if not isinstance(raw_mapping, dict):
        raise ValueError("commodity_to_sector must be a mapping")
    commodity_to_sector = {
        norm_text_lower(k): norm_text(v) for k, v in raw_mapping.items()
    }
    default_sector = norm_text(op.get("default_sector", "ind"))
    strict_targets = bool(op.get("strict_targets", True))

    l_rows = _row_indices_by_category(state, "L")
    commodity_cols = _commodity_columns(state)
    moved_by_commodity: dict[str, float] = {}
    missing_targets: list[str] = []
    target_cache: dict[tuple[str, str], int] = {}

    for c_idx, commodity in commodity_cols:
        sector = commodity_to_sector.get(norm_text_lower(commodity), default_sector)
        target_key = ("J", sector)
        if target_key not in target_cache:
            try:
                target_cache[target_key] = index_for_key(
                    state.row_keys, target_key, f"target_row[{commodity}]"
                )
            except ValueError:
                if strict_targets:
                    raise
                missing_targets.append(f"J.{sector}")
                continue
        target_row = target_cache[target_key]

        moved_col = 0.0
        for r_idx in l_rows:
            value = float(state.matrix[r_idx, c_idx])
            if abs(value) <= 1e-14:
                continue
            state.matrix[r_idx, c_idx] = 0.0
            state.matrix[target_row, c_idx] += value
            moved_col += value
        if abs(moved_col) > 1e-14:
            moved_by_commodity[commodity] = moved_col

    moved_total = float(sum(moved_by_commodity.values()))
    return {
        "source_category": "L",
        "sources": len(l_rows),
        "commodities": len(commodity_cols),
        "default_sector": default_sector,
        "moved_total": moved_total,
        "moved_by_commodity": moved_by_commodity,
        "missing_targets": missing_targets,
    }


def apply_move_margin_to_i_margin(
    state: SAMLikeState, op: dict[str, Any]
) -> dict[str, Any]:
    """Move margin-row inflows into a chosen margin commodity row.

    Economic intuition:
    - `MARG.MARG -> I.i` is not directly consumed by PEP commodity supply logic.
    - Reallocate to `I.margin_commodity -> I.i` so margins become commodity flows.
    """

    source_row = index_for_key(state.row_keys, ("MARG", "MARG"), "source_row")
    margin_commodity = norm_text(op.get("margin_commodity", "ser"))
    target_row = index_for_key(state.row_keys, ("I", margin_commodity), "target_row")
    cols = indices_for_selector(state.col_keys, op.get("col", "I.*"), "col")

    moved_total = 0.0
    moved_by_column: dict[str, float] = {}
    for c_idx in cols:
        value = float(state.matrix[source_row, c_idx])
        if abs(value) <= 1e-14:
            continue
        state.matrix[source_row, c_idx] = 0.0
        state.matrix[target_row, c_idx] += value
        moved_total += value
        moved_by_column[norm_text(state.col_keys[c_idx][1])] = value

    return {
        "source_row": "MARG.MARG",
        "target_row": f"I.{margin_commodity}",
        "columns": len(cols),
        "moved_total": moved_total,
        "moved_by_column": moved_by_column,
    }


def apply_move_tx_to_ti_on_i(state: SAMLikeState, op: dict[str, Any]) -> dict[str, Any]:
    """Move `AG.tx` commodity-column flows into `AG.ti`.

    Economic intuition:
    - For commodity columns, PEP tax intake is expected in the indirect-tax
      account (`AG.ti`) rather than in the export-tax bucket (`AG.tx`).
    - This makes tax routing consistent before calibration.
    """

    source_row = index_for_key(state.row_keys, ("AG", "tx"), "source_row")
    target_row = index_for_key(state.row_keys, ("AG", "ti"), "target_row")
    cols = indices_for_selector(state.col_keys, op.get("col", "I.*"), "col")

    moved_total = 0.0
    moved_by_column: dict[str, float] = {}
    for c_idx in cols:
        value = float(state.matrix[source_row, c_idx])
        if abs(value) <= 1e-14:
            continue
        state.matrix[source_row, c_idx] = 0.0
        state.matrix[target_row, c_idx] += value
        moved_total += value
        moved_by_column[norm_text(state.col_keys[c_idx][1])] = value

    return {
        "source_row": "AG.tx",
        "target_row": "AG.ti",
        "columns": len(cols),
        "moved_total": moved_total,
        "moved_by_column": moved_by_column,
    }


def apply_pep_structural_moves(state: SAMLikeState, op: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible composite operation over the 4 disaggregated moves."""

    common_mapping = op.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    margin_commodity = norm_text(op.get("margin_commodity", "ser"))
    default_sector = norm_text(op.get("default_sector", "ind"))
    strict_targets = bool(op.get("strict_targets", True))

    k_step = apply_move_k_to_ji(
        state,
        {
            "commodity_to_sector": common_mapping,
            "default_sector": default_sector,
            "strict_targets": strict_targets,
        },
    )
    l_step = apply_move_l_to_ji(
        state,
        {
            "commodity_to_sector": common_mapping,
            "default_sector": default_sector,
            "strict_targets": strict_targets,
        },
    )
    margin_step = apply_move_margin_to_i_margin(
        state,
        {
            "margin_commodity": margin_commodity,
            "col": op.get("col", "I.*"),
        },
    )
    tx_step = apply_move_tx_to_ti_on_i(
        state,
        {
            "col": op.get("col", "I.*"),
        },
    )

    moved_total_abs = float(
        abs(float(k_step["moved_total"]))
        + abs(float(l_step["moved_total"]))
        + abs(float(margin_step["moved_total"]))
        + abs(float(tx_step["moved_total"]))
    )
    return {
        "composite": True,
        "steps": {
            "move_k_to_ji": k_step,
            "move_l_to_ji": l_step,
            "move_margin_to_i_margin": margin_step,
            "move_tx_to_ti_on_i": tx_step,
        },
        "moved_total_abs": moved_total_abs,
    }
