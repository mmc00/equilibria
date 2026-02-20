"""
Utilities to transform CRI SAM Excel files into a PEP-compatible structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from equilibria.babel.sam_loader import SAM4DLoader

TargetMode = Literal["geomean", "average", "original"]
FixMode = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class SAMGrid:
    raw_df: pd.DataFrame
    data_start_row: int
    data_start_col: int
    row_keys: list[tuple[str, str]]
    col_keys: list[tuple[str, str]]
    matrix: np.ndarray


def _normalize_key(cat: str, elem: str) -> tuple[str, str]:
    return (str(cat).strip(), str(elem).strip())


def _build_index_map(keys: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    return {k: i for i, k in enumerate(keys)}


def load_sam_grid(filepath: Path) -> SAMGrid:
    loader = SAM4DLoader(rdim=2, cdim=2, sparse=False, unique_elements=False)
    raw_df = loader._read_excel(filepath, "SAM")
    data_start_row, data_start_col = loader._detect_boundaries(raw_df)

    row_data = loader._extract_row_indices(raw_df, data_start_row, data_start_col)
    col_data = loader._extract_col_indices(raw_df, data_start_row, data_start_col)
    matrix = loader._extract_data(
        raw_df,
        data_start_row,
        data_start_col,
        len(row_data["tuples"]),
        len(col_data["tuples"]),
    ).astype(float)

    row_keys = [_normalize_key(r[0], r[1]) for r in row_data["tuples"]]
    col_keys = [_normalize_key(c[0], c[1]) for c in col_data["tuples"]]
    return SAMGrid(
        raw_df=raw_df,
        data_start_row=data_start_row,
        data_start_col=data_start_col,
        row_keys=row_keys,
        col_keys=col_keys,
        matrix=matrix,
    )


def apply_pep_structural_moves(
    matrix: np.ndarray,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
    commodity_to_sector: dict[str, str],
    margin_commodity: str = "ser",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Reallocate unsupported inflows on commodity columns to PEP-compatible rows.

    Rules:
    - K.* -> I.i  => J.map(i) -> I.i
    - L.* -> I.i  => J.map(i) -> I.i
    - MARG.MARG -> I.i => I.margin_commodity -> I.i
    - AG.tx -> I.i => AG.ti -> I.i
    """
    out = matrix.copy().astype(float)
    ri = _build_index_map(row_keys)
    ci = _build_index_map(col_keys)

    moves: dict[str, float] = {}

    def add_move(rule: str, value: float) -> None:
        moves[rule] = moves.get(rule, 0.0) + float(value)

    def move_cell(
        source_row: tuple[str, str],
        column: tuple[str, str],
        target_row: tuple[str, str],
        rule: str,
    ) -> None:
        if source_row not in ri or target_row not in ri or column not in ci:
            return
        r_from = ri[source_row]
        c = ci[column]
        r_to = ri[target_row]
        value = out[r_from, c]
        if abs(value) <= 1e-14:
            return
        out[r_from, c] = 0.0
        out[r_to, c] += value
        add_move(rule, value)

    commodity_cols = [k for k in col_keys if k[0] == "I"]
    factor_k_rows = [k for k in row_keys if k[0] == "K"]
    factor_l_rows = [k for k in row_keys if k[0] == "L"]

    for col in commodity_cols:
        commodity = col[1]
        mapped_sector = commodity_to_sector.get(commodity, "ind")
        target_j = ("J", mapped_sector)

        for row in factor_k_rows:
            move_cell(row, col, target_j, "K_to_JI")
        for row in factor_l_rows:
            move_cell(row, col, target_j, "L_to_JI")

        move_cell(("MARG", "MARG"), col, ("I", margin_commodity), "MARG_to_I_margin")
        move_cell(("AG", "tx"), col, ("AG", "ti"), "TX_to_TI_on_I")

    return out, moves


def build_support_mask(
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
) -> np.ndarray:
    """
    Allowed support for rebalancing.

    Disallow only known unsupported inflows on commodity columns.
    """
    support = np.ones((len(row_keys), len(col_keys)), dtype=bool)
    for r, (row_cat, row_elem) in enumerate(row_keys):
        for c, (col_cat, _col_elem) in enumerate(col_keys):
            if col_cat != "I":
                continue
            if row_cat in {"K", "L", "MARG"}:
                support[r, c] = False
            if row_cat == "AG" and row_elem == "tx":
                support[r, c] = False
    return support


def compute_targets(
    seed: np.ndarray,
    mode: TargetMode,
    original: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "original":
        if original is None:
            raise ValueError("original matrix required for target mode 'original'")
        return original.sum(axis=1), original.sum(axis=0)

    row0 = seed.sum(axis=1)
    col0 = seed.sum(axis=0)

    if mode == "average":
        target = 0.5 * (row0 + col0)
    elif mode == "geomean":
        target = np.zeros_like(row0)
        for i in range(len(row0)):
            if row0[i] <= eps and col0[i] <= eps:
                target[i] = 0.0
            else:
                target[i] = np.sqrt(max(row0[i], eps) * max(col0[i], eps))
    else:
        raise ValueError(f"Unknown target mode: {mode}")

    target_sum = target.sum()
    row_sum = row0.sum()
    if target_sum > eps and row_sum > eps:
        target *= row_sum / target_sum
    return target, target.copy()


def ipfp_rebalance(
    seed: np.ndarray,
    support: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    eps: float = 1e-9,
    tol: float = 1e-8,
    max_iter: int = 20000,
) -> tuple[np.ndarray, int, float]:
    if seed.shape != support.shape:
        raise ValueError("seed/support shape mismatch")
    if seed.shape[0] != len(row_targets) or seed.shape[1] != len(col_targets):
        raise ValueError("target sizes do not match matrix shape")

    for i in range(seed.shape[0]):
        if row_targets[i] > 0 and not np.any(support[i, :]):
            raise ValueError(f"Row {i} has positive target but zero support")
    for j in range(seed.shape[1]):
        if col_targets[j] > 0 and not np.any(support[:, j]):
            raise ValueError(f"Column {j} has positive target but zero support")

    x = np.where(support, np.maximum(seed, eps), 0.0)
    err = float("inf")
    iterations = 0

    for it in range(max_iter):
        row_sums = x.sum(axis=1)
        row_factors = np.ones_like(row_sums)
        nz = row_sums > 0
        row_factors[nz] = row_targets[nz] / row_sums[nz]
        x *= row_factors[:, None]
        x[~support] = 0.0

        col_sums = x.sum(axis=0)
        col_factors = np.ones_like(col_sums)
        nz = col_sums > 0
        col_factors[nz] = col_targets[nz] / col_sums[nz]
        x *= col_factors[None, :]
        x[~support] = 0.0

        err = max(
            float(np.max(np.abs(x.sum(axis=1) - row_targets))),
            float(np.max(np.abs(x.sum(axis=0) - col_targets))),
        )
        iterations = it + 1
        if err <= tol:
            break

    return x, iterations, err


def write_grid_to_excel(grid: SAMGrid, matrix: np.ndarray, output: Path) -> None:
    out_df = grid.raw_df.copy()
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            out_df.iat[grid.data_start_row + i, grid.data_start_col + j] = float(matrix[i, j])

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="SAM", index=False, header=False)


def balance_stats(matrix: np.ndarray) -> dict[str, float]:
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    return {
        "total": float(matrix.sum()),
        "max_row_col_abs_diff": float(np.max(np.abs(row_sums - col_sums))),
    }


def pep_column_compatibility(
    matrix: np.ndarray,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"commodities": {}, "totals": {}}
    ignored_total = 0.0

    for c, (col_cat, col_elem) in enumerate(col_keys):
        if col_cat != "I":
            continue
        col_total = float(matrix[:, c].sum())
        pep_supply = 0.0
        ignored = 0.0
        for r, (row_cat, row_elem) in enumerate(row_keys):
            value = float(matrix[r, c])
            if row_cat == "J":
                pep_supply += value
            if row_cat == "AG" and row_elem == "row":
                pep_supply += value
            if row_cat in {"K", "L", "MARG"}:
                ignored += value
            if row_cat == "AG" and row_elem == "tx":
                ignored += value

        ignored_total += ignored
        result["commodities"][col_elem] = {
            "column_total": col_total,
            "pep_supply_channels": pep_supply,
            "ignored_inflows": ignored,
            "supply_gap_vs_total": col_total - pep_supply,
        }

    result["totals"]["ignored_inflows"] = ignored_total
    return result


def enforce_export_value_balance(
    matrix: np.ndarray,
    row_keys: list[tuple[str, str]],
    col_keys: list[tuple[str, str]],
    tol: float = 1e-12,
) -> dict[str, float]:
    """
    Enforce EXP001 identity exactly with symmetric two-cell adjustments.
    """
    out = matrix
    ri = _build_index_map(row_keys)
    ci = _build_index_map(col_keys)
    adjustments: dict[str, float] = {}

    if ("AG", "row") not in ci or ("AG", "row") not in ri:
        return adjustments

    col_ag_row = ci[("AG", "row")]
    row_ag_row = ri[("AG", "row")]

    for col_key in col_keys:
        if col_key[0] != "X":
            continue
        commodity = col_key[1]
        row_x = ("X", commodity)
        col_x = ("X", commodity)
        if row_x not in ri or col_x not in ci:
            continue

        lhs = float(out[ri[row_x], col_ag_row])
        rhs = 0.0
        x_col = ci[col_x]
        for r, (row_cat, row_elem) in enumerate(row_keys):
            value = float(out[r, x_col])
            if row_cat == "J":
                rhs += value
            elif row_cat == "I":
                rhs += value
            elif row_cat == "AG" and row_elem == "gvt":
                rhs += value

        delta = rhs - lhs
        if abs(delta) <= tol:
            continue

        out[ri[row_x], col_ag_row] += delta
        out[row_ag_row, x_col] += delta
        adjustments[commodity] = float(delta)

    return adjustments


def transform_sam_to_pep_compatible(
    input_sam: Path | str,
    output_sam: Path | str,
    *,
    report_json: Path | str | None = None,
    target_mode: TargetMode = "geomean",
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

    grid = load_sam_grid(input_path)
    before_balance = balance_stats(grid.matrix)
    before_compat = pep_column_compatibility(grid.matrix, grid.row_keys, grid.col_keys)

    commodity_to_sector = {
        "agr": "agr",
        "othind": "ind",
        "ser": "ser",
        "food": "ind",
        "adm": "adm",
    }
    moved_matrix, moves = apply_pep_structural_moves(
        grid.matrix,
        grid.row_keys,
        grid.col_keys,
        commodity_to_sector=commodity_to_sector,
        margin_commodity=margin_commodity,
    )

    support = build_support_mask(grid.row_keys, grid.col_keys)
    seeded = np.where(support, np.maximum(moved_matrix, epsilon), 0.0)
    row_targets, col_targets = compute_targets(
        seeded,
        mode=target_mode,
        original=grid.matrix,
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
    exp_adjustments = enforce_export_value_balance(balanced, grid.row_keys, grid.col_keys)

    write_grid_to_excel(grid, balanced, output_path)

    after_balance = balance_stats(balanced)
    after_compat = pep_column_compatibility(balanced, grid.row_keys, grid.col_keys)

    report: dict[str, Any] = {
        "input_sam": str(input_path),
        "output_sam": str(output_path),
        "settings": {
            "target_mode": target_mode,
            "margin_commodity": margin_commodity,
            "epsilon": epsilon,
            "tol": tol,
            "max_iter": max_iter,
        },
        "moves": moves,
        "export_balance_adjustments": exp_adjustments,
        "ipfp": {
            "iterations": iterations,
            "residual": residual,
        },
        "before": {
            "balance": before_balance,
            "pep_compatibility": before_compat,
        },
        "after": {
            "balance": after_balance,
            "pep_compatibility": after_compat,
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
    if "pep-compatible" in name:
        return False
    return "sam-cri" in name
