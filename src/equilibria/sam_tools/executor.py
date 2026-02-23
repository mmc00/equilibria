"""Execution engine for YAML SAM workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.sam_tools.config_loader import load_workflow_config
from equilibria.sam_tools.enums import (
    IPFPSupportMode,
    IPFPTargetMode,
    WorkflowOperation,
)
from equilibria.sam_tools.ieem_raw_excel import aggregate_state_with_mapping
from equilibria.sam_tools.ieem_to_pep_transformations import (
    align_ti_to_gvt_j,
    apply_move_k_to_ji,
    apply_move_l_to_ji,
    apply_move_margin_to_i_margin,
    apply_move_tx_to_ti_on_i,
    apply_pep_structural_moves,
    balance_state_ras,
    convert_exports_to_x,
    create_x_block,
    normalize_state_to_pep_accounts,
)
from equilibria.sam_tools.models import SamTransform
from equilibria.sam_tools.selectors import (
    index_for_key,
    indices_for_selector,
)
from equilibria.sam_tools.state_store import load_state, write_state
from equilibria.templates.pep_sam_compat import (
    balance_stats,
    build_support_mask,
    compute_targets,
    enforce_export_value_balance,
    ipfp_rebalance,
)


def _apply_scale_all(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Scale all SAM cells by one multiplicative factor."""
    factor = float(op.get("factor", 1.0))
    state.matrix *= factor
    return {"factor": factor, "cells": int(state.matrix.size)}


def _apply_scale_slice(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Scale one selected matrix slice by a factor."""
    factor = float(op.get("factor", 1.0))
    rows = indices_for_selector(state.row_keys, op.get("row"), "row")
    cols = indices_for_selector(state.col_keys, op.get("col"), "col")

    view = state.matrix[np.ix_(rows, cols)]
    before = float(view.sum())
    state.matrix[np.ix_(rows, cols)] = view * factor
    after = float(state.matrix[np.ix_(rows, cols)].sum())

    return {
        "factor": factor,
        "rows": len(rows),
        "cols": len(cols),
        "sum_before": before,
        "sum_after": after,
    }


def _apply_shift_row_slice(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Move a share of values from one row to another over selected columns."""
    source_row = index_for_key(state.row_keys, op.get("source_row"), "source_row")
    target_row = index_for_key(state.row_keys, op.get("target_row"), "target_row")
    cols = indices_for_selector(state.col_keys, op.get("col"), "col")
    share = float(op.get("share", 1.0))

    moved_total = 0.0
    for c in cols:
        moved = float(state.matrix[source_row, c]) * share
        state.matrix[source_row, c] -= moved
        state.matrix[target_row, c] += moved
        moved_total += moved

    return {"share": share, "cols": len(cols), "moved_total": moved_total}


def _apply_move_cell(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Transfer value from one specific cell to another cell."""
    source = op.get("source") or {}
    target = op.get("target") or {}
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("move_cell requires source/target mappings")

    r_from = index_for_key(state.row_keys, source.get("row"), "source.row")
    c_from = index_for_key(state.col_keys, source.get("col"), "source.col")
    r_to = index_for_key(state.row_keys, target.get("row"), "target.row")
    c_to = index_for_key(state.col_keys, target.get("col"), "target.col")

    current = float(state.matrix[r_from, c_from])
    if "amount" in op:
        amount = float(op["amount"])
    else:
        amount = current * float(op.get("share", 1.0))

    state.matrix[r_from, c_from] -= amount
    state.matrix[r_to, c_to] += amount

    return {
        "amount": amount,
        "source_value_before": current,
        "source_value_after": float(state.matrix[r_from, c_from]),
    }


def _apply_rebalance_ipfp(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Run IPFP rebalancing over a masked support with configurable targets."""
    target_mode = IPFPTargetMode.from_alias(str(op.get("target_mode", "geomean")))
    support_mode = IPFPSupportMode.from_alias(str(op.get("support", "pep_compat")))

    if support_mode == IPFPSupportMode.PEP_COMPAT:
        support = build_support_mask(state.row_keys, state.col_keys)
    elif support_mode == IPFPSupportMode.FULL:
        support = np.ones_like(state.matrix, dtype=bool)
    else:
        raise ValueError(f"Unsupported support mode: {support_mode}")

    epsilon = float(op.get("epsilon", 1e-9))
    tol = float(op.get("tol", 1e-8))
    max_iter = int(op.get("max_iter", 20000))

    seeded = np.where(support, np.maximum(state.matrix, epsilon), 0.0)
    row_targets, col_targets = compute_targets(
        seeded,
        mode=target_mode.value,
        original=state.matrix,
    )

    balanced, iterations, residual = ipfp_rebalance(
        seeded,
        support,
        row_targets,
        col_targets,
        eps=epsilon,
        tol=tol,
        max_iter=max_iter,
    )
    state.matrix = balanced

    return {
        "target_mode": target_mode.value,
        "support_mode": support_mode.value,
        "epsilon": epsilon,
        "tol": tol,
        "max_iter": max_iter,
        "iterations": iterations,
        "residual": residual,
    }


def _apply_enforce_export_balance(
    state: SamTransform,
    op: dict[str, Any],
) -> dict[str, Any]:
    """Enforce export value identity after structural and balancing steps."""
    tol = float(op.get("tol", 1e-12))
    adjustments = enforce_export_value_balance(
        state.matrix,
        state.row_keys,
        state.col_keys,
        tol=tol,
    )
    return {
        "tol": tol,
        "adjustments": adjustments,
        "total_adjustment_abs": float(sum(abs(float(v)) for v in adjustments.values())),
    }


def _resolve_operation_name(op: dict[str, Any]) -> WorkflowOperation:
    """Parse and validate the operation enum from a transform mapping."""
    raw_name = op.get("op")
    if raw_name is None:
        raise ValueError("Each transform entry must include 'op'")
    try:
        return WorkflowOperation(str(raw_name).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported transform op: {raw_name}") from exc


def _apply_operation(state: SamTransform, op: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one operation mapping to its concrete transformation handler."""
    op_name = _resolve_operation_name(op)

    if op_name == WorkflowOperation.SCALE_ALL:
        return _apply_scale_all(state, op)
    if op_name == WorkflowOperation.SCALE_SLICE:
        return _apply_scale_slice(state, op)
    if op_name == WorkflowOperation.AGGREGATE_MAPPING:
        return aggregate_state_with_mapping(state, op)
    if op_name == WorkflowOperation.BALANCE_RAS:
        return balance_state_ras(state, op)
    if op_name == WorkflowOperation.NORMALIZE_PEP_ACCOUNTS:
        return normalize_state_to_pep_accounts(state, op)
    if op_name == WorkflowOperation.CREATE_X_BLOCK:
        return create_x_block(state, op)
    if op_name == WorkflowOperation.CONVERT_EXPORTS_TO_X:
        return convert_exports_to_x(state, op)
    if op_name == WorkflowOperation.ALIGN_TI_TO_GVT_J:
        return align_ti_to_gvt_j(state, op)
    if op_name == WorkflowOperation.SHIFT_ROW_SLICE:
        return _apply_shift_row_slice(state, op)
    if op_name == WorkflowOperation.MOVE_CELL:
        return _apply_move_cell(state, op)
    if op_name == WorkflowOperation.MOVE_K_TO_JI:
        return apply_move_k_to_ji(state, op)
    if op_name == WorkflowOperation.MOVE_L_TO_JI:
        return apply_move_l_to_ji(state, op)
    if op_name == WorkflowOperation.MOVE_MARGIN_TO_I_MARGIN:
        return apply_move_margin_to_i_margin(state, op)
    if op_name == WorkflowOperation.MOVE_TX_TO_TI_ON_I:
        return apply_move_tx_to_ti_on_i(state, op)
    if op_name == WorkflowOperation.PEP_STRUCTURAL_MOVES:
        return apply_pep_structural_moves(state, op)
    if op_name == WorkflowOperation.REBALANCE_IPFP:
        return _apply_rebalance_ipfp(state, op)
    if op_name == WorkflowOperation.ENFORCE_EXPORT_BALANCE:
        return _apply_enforce_export_balance(state, op)

    raise ValueError(f"Unsupported transform op: {op_name.value}")


def _resolve_operation_paths(op: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve any *_path string fields against workflow file location."""
    resolved: dict[str, Any] = {}
    for key, value in op.items():
        if key.endswith("_path") and isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            resolved[key] = str(path)
        else:
            resolved[key] = value
    return resolved


def run_sam_transform_workflow(
    config_file: Path | str,
    *,
    output_override: Path | str | None = None,
    report_override: Path | str | None = None,
) -> dict[str, Any]:
    """Run a YAML-defined SAM transformation workflow."""

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow YAML not found: {config_path}")

    cfg = load_workflow_config(config_path)
    if output_override is not None:
        cfg.output_path = Path(output_override).resolve()
    if report_override is not None:
        cfg.report_path = Path(report_override).resolve()

    state = load_state(cfg.input_path, cfg.input_format, options=cfg.input_options)

    steps: list[dict[str, Any]] = []
    for idx, op in enumerate(cfg.transforms, start=1):
        if not isinstance(op, dict):
            raise ValueError(f"Transform at position {idx} must be a mapping")

        op_resolved = _resolve_operation_paths(op, config_path.parent)
        op_name = _resolve_operation_name(op_resolved).value
        before = balance_stats(state.matrix)
        details = _apply_operation(state, op_resolved)
        after = balance_stats(state.matrix)
        steps.append(
            {
                "step": idx,
                "op": op_name,
                "before": before,
                "after": after,
                "details": details,
            }
        )

    write_info = write_state(state, cfg.output_path, cfg.output_format, cfg.output_symbol)

    report: dict[str, Any] = {
        "workflow": {
            "name": cfg.name,
            "country": cfg.country,
            "config_file": str(config_path),
        },
        "input": {
            "path": str(cfg.input_path),
            "format": cfg.input_format,
            "shape": [int(state.matrix.shape[0]), int(state.matrix.shape[1])],
        },
        "output": {
            "path": str(cfg.output_path),
            **write_info,
        },
        "summary": {
            "steps": len(steps),
            "final_balance": balance_stats(state.matrix),
        },
        "steps": steps,
    }

    if cfg.report_path:
        cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
