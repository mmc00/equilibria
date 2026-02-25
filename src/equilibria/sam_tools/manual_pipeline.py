"""Manual IEEM->PEP pipeline built on top of the core ``Sam`` class."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from equilibria.sam_tools.balancing import RASBalancer
from equilibria.sam_tools.ieem_raw_excel import IEEMRawSAM
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


BUILD_STEPS: Sequence[str] = (
    "normalize_accounts",
    "create_x_block",
    "convert_exports",
    "move_margins",
    "move_tx",
    "move_k",
    "move_l",
    "align_ti",
)


class ManualPipelineSummary:
    """Summary object returned by :func:`run_manual_pipeline`."""

    def __init__(self, sam: Sam, steps: list[dict[str, float]]):
        self.sam = sam
        self.steps = steps

    @property
    def matrix(self) -> list[list[float]]:
        return self.sam.matrix.tolist()

    @property
    def total_flow(self) -> float:
        return float(self.sam.matrix.sum())

    @property
    def balance_stats(self) -> dict[str, float]:
        df = self.sam.to_dataframe()
        row_imbalance = float((df.sum(axis=1) - df.sum(axis=0)).abs().max())
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "max_row_col_diff": row_imbalance,
            "total_flow": self.total_flow,
        }


def load_raw_ieem_sam(path: Path, *, sheet_name: str = "MCS2016") -> Sam:
    """Load raw IEEM Excel data as a :class:`Sam` object without metadata."""

    ieem = IEEMRawSAM.from_ieem_excel(path, sheet_name=sheet_name)
    return ieem


def _record_step(name: str, detail: dict[str, float]) -> dict[str, float]:
    detail["step"] = name
    return detail


def run_manual_pipeline(sam: Sam, *, ras_tol: float = 1e-9, ras_max_iter: int = 500) -> ManualPipelineSummary:
    """Run the manual sequence of IEEM -> PEP transformations.

    Args:
        sam: Initial ``Sam`` object sourced from a RAW IEEM file.
        ras_tol: Tolerance passed to :class:`RASBalancer` for the final rebalance.
        ras_max_iter: Iteration limit for the final rebalance.
    """

    steps: list[dict[str, float]] = []

    steps.append(_record_step("normalize_accounts", normalize_pep_accounts_on_sam(sam)))
    steps.append(_record_step("create_x_block", create_x_block_on_sam(sam)))
    steps.append(_record_step("convert_exports", convert_exports_to_x_on_sam(sam)))
    steps.append(_record_step("move_margins", move_margin_to_i_margin_on_sam(sam)))
    steps.append(_record_step("move_tx", move_tx_to_ti_on_i_on_sam(sam)))
    steps.append(_record_step("move_k", move_k_to_ji_on_sam(sam)))
    steps.append(_record_step("move_l", move_l_to_ji_on_sam(sam)))
    steps.append(_record_step("align_ti", align_ti_to_gvt_j_on_sam(sam)))

    balancer = RASBalancer()
    balance_result = balancer.balance_dataframe(
        sam.to_dataframe(), tolerance=ras_tol, max_iterations=ras_max_iter
    )
    steps.append(_record_step("balance_ras", {
        "max_diff_before": balance_result.max_diff_before,
        "max_diff_after": balance_result.max_diff_after,
        "iterations": float(balance_result.iterations),
    }))
    sam.replace_dataframe(balance_result.matrix)

    return ManualPipelineSummary(sam, steps)


def run_from_excel(path: Path, *, sheet_name: str = "MCS2016") -> ManualPipelineSummary:
    """Load a RAW IEEM Excel file and execute the manual pipeline."""

    sam = load_raw_ieem_sam(path, sheet_name=sheet_name)
    summary = run_manual_pipeline(sam)
    return summary
