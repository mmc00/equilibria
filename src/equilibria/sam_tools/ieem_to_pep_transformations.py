"""IEEM -> PEP structural transformations over ``Sam`` tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.models import Sam
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

DEFAULT_COMMODITY_TO_SECTOR: dict[str, str] = {
    "agr": "agr",
    "othind": "ind",
    "ser": "ser",
    "food": "ind",
    "adm": "adm",
}

_RAS_BALANCER = RASBalancer()


def _raw_sam_to_dataframe(sam: Sam) -> pd.DataFrame:
    if any(str(cat).strip().lower() != "raw" for cat, _ in sam.row_keys):
        raise ValueError("SAM rows must be category 'RAW' for this operation")
    if any(str(cat).strip().lower() != "raw" for cat, _ in sam.col_keys):
        raise ValueError("SAM cols must be category 'RAW' for this operation")
    return sam.to_dataframe()


def balance_sam_ras(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """RAS rebalance on RAW SAM matrices before IEEM->PEP normalization."""

    tol = float(op.get("tol", op.get("tolerance", 1e-9)))
    max_iter = int(op.get("max_iter", 200))
    ras_type = str(op.get("ras_type", "arithmetic"))
    df = _raw_sam_to_dataframe(sam)
    result = _RAS_BALANCER.balance_dataframe(
        df,
        ras_type=ras_type,
        tolerance=tol,
        max_iterations=max_iter,
    )
    sam.replace_dataframe(result.matrix)
    return {
        "tol": tol,
        "max_iter": max_iter,
        "ras_type": result.ras_type,
        "iterations": result.iterations,
        "converged": result.converged,
        "max_diff_before": result.max_diff_before,
        "max_diff_after": result.max_diff_after,
    }


def normalize_pep_accounts(sam: Sam, _op: dict[str, Any]) -> dict[str, Any]:
    return normalize_pep_accounts_on_sam(sam)


def create_x_block(sam: Sam, _op: dict[str, Any]) -> dict[str, Any]:
    return create_x_block_on_sam(sam)


def convert_exports_to_x(sam: Sam, _op: dict[str, Any]) -> dict[str, Any]:
    return convert_exports_to_x_on_sam(sam)


def align_ti_to_gvt_j(sam: Sam, _op: dict[str, Any]) -> dict[str, Any]:
    return align_ti_to_gvt_j_on_sam(sam)


def apply_move_k_to_ji(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    return move_k_to_ji_on_sam(sam, op)


def apply_move_l_to_ji(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    return move_l_to_ji_on_sam(sam, op)


def apply_move_margin_to_i_margin(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    return move_margin_to_i_margin_on_sam(sam, op)


def apply_move_tx_to_ti_on_i(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    return move_tx_to_ti_on_i_on_sam(sam, op)


def apply_pep_structural_moves(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible composite operation over the four disaggregated moves."""

    common_mapping = op.get("commodity_to_sector") or DEFAULT_COMMODITY_TO_SECTOR
    margin_commodity = str(op.get("margin_commodity", "ser")).strip()
    default_sector = str(op.get("default_sector", "ind")).strip()
    strict_targets = bool(op.get("strict_targets", True))

    k_step = apply_move_k_to_ji(
        sam,
        {
            "commodity_to_sector": common_mapping,
            "default_sector": default_sector,
            "strict_targets": strict_targets,
        },
    )
    l_step = apply_move_l_to_ji(
        sam,
        {
            "commodity_to_sector": common_mapping,
            "default_sector": default_sector,
            "strict_targets": strict_targets,
        },
    )
    margin_step = apply_move_margin_to_i_margin(
        sam,
        {
            "margin_commodity": margin_commodity,
            "col": op.get("col", "I.*"),
        },
    )
    tx_step = apply_move_tx_to_ti_on_i(
        sam,
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

