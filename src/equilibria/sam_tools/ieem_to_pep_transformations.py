"""IEEM -> PEP structural transformations for SAM matrices.

These operations move flows from unsupported IEEM cells into PEP-compatible
channels before rebalancing.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

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


class SAMLikeState(Protocol):
    """Protocol expected by IEEM->PEP transforms."""

    matrix: np.ndarray
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]


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
