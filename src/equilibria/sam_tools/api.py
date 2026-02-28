"""High-level SAM API focused on ``Sam`` as the user-facing object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from equilibria.sam_tools.ieem_raw_excel import IEEMRawSAM
from equilibria.sam_tools.ieem_to_pep_transformations import balance_sam_ras
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
from equilibria.sam_tools.parsers import export_sam

DEFAULT_IEEM_SHEET = "MCS2016"


class IEEMToPEPResult(NamedTuple):
    """Result object returned by :func:`run_ieem_to_pep`."""

    sam: Sam
    steps: list[dict[str, Any]]
    output_path: Path | None
    report_path: Path | None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _sam_balance_stats(sam: Sam) -> dict[str, float]:
    matrix = sam.matrix
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    if matrix.size == 0:
        max_diff = 0.0
    else:
        max_diff = float(np.max(np.abs(row_sums - col_sums)))
    return {
        "total": float(matrix.sum()),
        "max_row_col_abs_diff": max_diff,
    }


def _identity_commodity_to_sector(sam: Sam) -> dict[str, str]:
    return {
        elem: elem
        for cat, elem in sam.row_keys
        if str(cat).strip().lower() == "i"
    }


def run_ieem_to_pep(
    input_path: Path | str,
    mapping_path: Path | str,
    *,
    sheet_name: str = DEFAULT_IEEM_SHEET,
    commodity_to_sector: Mapping[str, str] | None = None,
    margin_commodity: str = "ser",
    ras_type: str = "arithmetic",
    ras_tol: float = 1e-9,
    ras_max_iter: int = 200,
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> IEEMToPEPResult:
    """Run a linear IEEM->PEP conversion and return the transformed ``Sam``."""

    input_file = Path(input_path).resolve()
    mapping_file = Path(mapping_path).resolve()
    sam = IEEMRawSAM.from_ieem_excel(input_file, sheet_name=sheet_name)
    steps: list[dict[str, Any]] = []

    def record(step: str, details: dict[str, Any] | None = None) -> None:
        steps.append(
            {
                "step": step,
                "details": details or {},
                "balance": _sam_balance_stats(sam),
            }
        )

    shape_before = list(sam.matrix.shape)
    sam.aggregate(mapping_file)
    record(
        "aggregate_mapping",
        {
            "mapping_path": str(mapping_file),
            "shape_before": shape_before,
            "shape_after": list(sam.matrix.shape),
        },
    )
    record(
        "balance_ras",
        balance_sam_ras(
            sam,
            {
                "ras_type": ras_type,
                "tol": ras_tol,
                "max_iter": ras_max_iter,
            },
        ),
    )
    record("normalize_pep_accounts", normalize_pep_accounts_on_sam(sam))
    record("create_x_block", create_x_block_on_sam(sam))
    record("convert_exports", convert_exports_to_x_on_sam(sam))
    record("align_ti", align_ti_to_gvt_j_on_sam(sam))

    mapping = (
        {str(k).strip(): str(v).strip() for k, v in commodity_to_sector.items()}
        if commodity_to_sector is not None
        else _identity_commodity_to_sector(sam)
    )
    record("move_k", move_k_to_ji_on_sam(sam, {"commodity_to_sector": mapping}))
    record("move_l", move_l_to_ji_on_sam(sam, {"commodity_to_sector": mapping}))
    record("move_margins", move_margin_to_i_margin_on_sam(sam, {"margin_commodity": margin_commodity, "col": "I.*"}))
    record("move_tx", move_tx_to_ti_on_i_on_sam(sam, {"col": "I.*"}))

    resolved_output: Path | None = Path(output_path).resolve() if output_path is not None else None
    if resolved_output is not None:
        export_sam(sam, resolved_output, output_format="excel", output_symbol="SAM")

    resolved_report: Path | None = Path(report_path).resolve() if report_path is not None else None
    if resolved_report is not None:
        resolved_report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_path": str(input_file),
            "mapping_path": str(mapping_file),
            "sheet_name": sheet_name,
            "output_path": str(resolved_output) if resolved_output is not None else None,
            "steps": _to_jsonable(steps),
        }
        resolved_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return IEEMToPEPResult(
        sam=sam,
        steps=steps,
        output_path=resolved_output,
        report_path=resolved_report,
    )


__all__ = [
    "IEEMToPEPResult",
    "run_ieem_to_pep",
]
