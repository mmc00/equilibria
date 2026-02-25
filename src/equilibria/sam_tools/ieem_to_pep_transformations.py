"""IEEM -> PEP structural transformations for SAM matrices.

These operations move flows from unsupported IEEM cells into PEP-compatible
channels before rebalancing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.models import SamTransform
from equilibria.sam_tools.sam_transforms import (
    align_ti_to_gvt_j_on_sam,
    convert_exports_to_x_on_sam,
    create_x_block_on_sam,
    move_k_to_ji_on_sam,
    move_l_to_ji_on_sam,
    move_margin_to_i_margin_on_sam,
    move_tx_to_ti_on_i_on_sam,
    normalize_pep_accounts_on_sam,
)
from equilibria.sam_tools.selectors import index_for_key, norm_text, norm_text_lower

DEFAULT_COMMODITY_TO_SECTOR: dict[str, str] = {
    "agr": "agr",
    "othind": "ind",
    "ser": "ser",
    "food": "ind",
    "adm": "adm",
}

SPECIAL_LABELS = {"S-HH", "S-FIRM", "S-GVT", "S-ROW", "INV", "VSTK"}


_RAS_BALANCER = RASBalancer()


def _raw_state_to_dataframe(state: SamTransform) -> pd.DataFrame:
    """Convert a RAW-key workflow state into a pandas dataframe."""
    if not state.row_keys or not state.col_keys:
        raise ValueError("State has no support keys")
    if any(norm_text_lower(cat) != "raw" for cat, _ in state.row_keys):
        raise ValueError("State rows must be category 'RAW' for this operation")
    if any(norm_text_lower(cat) != "raw" for cat, _ in state.col_keys):
        raise ValueError("State cols must be category 'RAW' for this operation")

    labels_r = [norm_text(elem) for _cat, elem in state.row_keys]
    labels_c = [norm_text(elem) for _cat, elem in state.col_keys]
    return pd.DataFrame(state.matrix.copy(), index=labels_r, columns=labels_c, dtype=float)


def _replace_state_from_raw_dataframe(state: SamTransform, df: pd.DataFrame) -> None:
    """Overwrite a state object from one RAW dataframe."""
    labels = [norm_text(label) for label in df.index]
    keys = [("RAW", label) for label in labels]
    multi_index = pd.MultiIndex.from_tuples(keys)
    df = df.copy()
    df.index = multi_index
    df.columns = multi_index
    state.sam.replace_dataframe(df)


def _label_to_key(label: str) -> tuple[str, str] | None:
    """Map one aggregated RAW label into a canonical PEP account key."""
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
    """Build deterministic PEP key ordering from observed RAW labels."""
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


def _add_value(
    matrix: np.ndarray,
    key_index: dict[tuple[str, str], int],
    row_key: tuple[str, str],
    col_key: tuple[str, str],
    value: float,
) -> None:
    """Add one value into a keyed matrix if source/target keys exist."""
    if abs(value) <= 1e-14:
        return
    if row_key not in key_index or col_key not in key_index:
        return
    matrix[key_index[row_key], key_index[col_key]] += float(value)


def balance_state_ras(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
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
    from equilibria.sam_tools.sam_transforms import normalize_pep_accounts_on_sam

    return normalize_pep_accounts_on_sam(state.sam)


def create_x_block(state: SamTransform, _op: dict[str, Any]) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import create_x_block_on_sam

    return create_x_block_on_sam(state.sam)


def convert_exports_to_x(state: SamTransform, _op: dict[str, Any]) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import convert_exports_to_x_on_sam

    return convert_exports_to_x_on_sam(state.sam)


def align_ti_to_gvt_j(state: SamTransform, _op: dict[str, Any]) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import align_ti_to_gvt_j_on_sam

    return align_ti_to_gvt_j_on_sam(state.sam)


def _commodity_columns(state: SamTransform) -> list[tuple[int, str]]:
    """Return all commodity columns as ``(index, commodity)`` pairs."""
    columns: list[tuple[int, str]] = []
    for idx, (cat, elem) in enumerate(state.col_keys):
        if norm_text_lower(cat) == "i":
            columns.append((idx, norm_text(elem)))
    return columns


def _row_indices_by_category(state: SamTransform, category: str) -> list[int]:
    """Return row indices that belong to one account category."""
    cat_norm = norm_text_lower(category)
    return [
        idx
        for idx, (cat, _elem) in enumerate(state.row_keys)
        if norm_text_lower(cat) == cat_norm
    ]


def apply_move_k_to_ji(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    return move_k_to_ji_on_sam(state.sam, op)


def apply_move_l_to_ji(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    return move_l_to_ji_on_sam(state.sam, op)


def apply_move_margin_to_i_margin(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import move_margin_to_i_margin_on_sam

    return move_margin_to_i_margin_on_sam(state.sam, op)


def apply_move_tx_to_ti_on_i(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    from equilibria.sam_tools.sam_transforms import move_tx_to_ti_on_i_on_sam

    return move_tx_to_ti_on_i_on_sam(state.sam, op)


def apply_pep_structural_moves(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
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
