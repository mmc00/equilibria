"""Execution engine for YAML SAM workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.sam_tools.config_loader import load_workflow_config
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
from equilibria.sam_tools.io import load_state, write_state
from equilibria.sam_tools.models import SAMTransformState
from equilibria.sam_tools.selectors import (
    index_for_key,
    indices_for_selector,
    norm_text_lower,
)
from equilibria.templates.pep_sam_compat import (
    balance_stats,
    build_support_mask,
    compute_targets,
    enforce_export_value_balance,
    ipfp_rebalance,
)


def _apply_scale_all(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    factor = float(op.get("factor", 1.0))
    state.matrix *= factor
    return {"factor": factor, "cells": int(state.matrix.size)}


def _apply_scale_slice(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
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


def _apply_shift_row_slice(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
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


def _apply_move_cell(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
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


def _apply_rebalance_ipfp(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    target_mode = norm_text_lower(op.get("target_mode", "geomean"))
    support_mode = norm_text_lower(op.get("support", "pep_compat"))

    if support_mode == "pep_compat":
        support = build_support_mask(state.row_keys, state.col_keys)
    elif support_mode == "full":
        support = np.ones_like(state.matrix, dtype=bool)
    else:
        raise ValueError(f"Unsupported support mode: {support_mode}")

    epsilon = float(op.get("epsilon", 1e-9))
    tol = float(op.get("tol", 1e-8))
    max_iter = int(op.get("max_iter", 20000))

    seeded = np.where(support, np.maximum(state.matrix, epsilon), 0.0)
    row_targets, col_targets = compute_targets(
        seeded,
        mode=target_mode,
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
        "target_mode": target_mode,
        "support_mode": support_mode,
        "epsilon": epsilon,
        "tol": tol,
        "max_iter": max_iter,
        "iterations": iterations,
        "residual": residual,
    }


def _apply_enforce_export_balance(
    state: SAMTransformState,
    op: dict[str, Any],
) -> dict[str, Any]:
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


def _apply_operation(state: SAMTransformState, op: dict[str, Any]) -> dict[str, Any]:
    op_name = norm_text_lower(op.get("op"))
    if not op_name:
        raise ValueError("Each transform entry must include 'op'")

    if op_name == "scale_all":
        return _apply_scale_all(state, op)
    if op_name == "scale_slice":
        return _apply_scale_slice(state, op)
    if op_name == "aggregate_mapping":
        return aggregate_state_with_mapping(state, op)
    if op_name == "balance_ras":
        return balance_state_ras(state, op)
    if op_name == "normalize_pep_accounts":
        return normalize_state_to_pep_accounts(state, op)
    if op_name == "create_x_block":
        return create_x_block(state, op)
    if op_name == "convert_exports_to_x":
        return convert_exports_to_x(state, op)
    if op_name == "align_ti_to_gvt_j":
        return align_ti_to_gvt_j(state, op)
    if op_name == "shift_row_slice":
        return _apply_shift_row_slice(state, op)
    if op_name == "move_cell":
        return _apply_move_cell(state, op)
    if op_name == "move_k_to_ji":
        return apply_move_k_to_ji(state, op)
    if op_name == "move_l_to_ji":
        return apply_move_l_to_ji(state, op)
    if op_name == "move_margin_to_i_margin":
        return apply_move_margin_to_i_margin(state, op)
    if op_name == "move_tx_to_ti_on_i":
        return apply_move_tx_to_ti_on_i(state, op)
    if op_name == "pep_structural_moves":
        return apply_pep_structural_moves(state, op)
    if op_name == "rebalance_ipfp":
        return _apply_rebalance_ipfp(state, op)
    if op_name == "enforce_export_balance":
        return _apply_enforce_export_balance(state, op)

    raise ValueError(f"Unsupported transform op: {op_name}")


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
        op_name = norm_text_lower(op_resolved.get("op"))
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
