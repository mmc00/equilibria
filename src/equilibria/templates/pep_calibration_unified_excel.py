"""
PEP unified calibration using SAM directly from Excel.

This variant mirrors GAMS-style Excel ingestion and avoids GDX reads for SAM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from equilibria.templates.data.pep import load_pep_sam
from equilibria.templates.pep_calibration_unified import (
    CalibrationReport,
    PEPModelCalibrator,
    PEPModelState,
)
from equilibria.templates.pep_dynamic_sets import derive_dynamic_sets_from_sam
from equilibria.templates.pep_val_par_loader import load_val_par

logger = logging.getLogger(__name__)


def _build_sam_data_from_excel(filepath: Path | str) -> dict[str, Any]:
    """Convert SAM Excel file into internal dict consumed by calibrators."""
    sam_path = Path(filepath)
    sam4d = load_pep_sam(sam_path, rdim=2, cdim=2, sparse=True, unique_elements=False)

    sam_matrix: dict[tuple[str, str, str, str], float] = {}
    records: list[dict[str, Any]] = []

    for keys, value in sam4d.to_gdx_records():
        if len(keys) != 4:
            continue
        key = tuple(str(k).upper() for k in keys)
        val = float(value)
        sam_matrix[key] = val
        records.append({"indices": list(key), "value": val})

    return {
        "filepath": str(sam_path),
        "source": "excel",
        "sam_matrix": sam_matrix,
        "symbols": [{"name": "SAM", "records": records}],
        "elements": [],
    }


class PEPModelCalibratorExcel(PEPModelCalibrator):
    """Unified calibrator that loads SAM from Excel instead of GDX."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        sets: dict[str, list[str]] | None = None,
        dynamic_sets: bool = False,
    ):
        self.sam_file = Path(sam_file)
        self.val_par_file = Path(val_par_file) if val_par_file else None
        self.val_par_data = load_val_par(self.val_par_file)
        self.state = PEPModelState()
        self.report = CalibrationReport()

        logger.info(f"Loading SAM (Excel) from {self.sam_file}")
        self.sam_data = _build_sam_data_from_excel(self.sam_file)
        sam_records = len(self.sam_data.get("sam_matrix", {}))
        logger.info(f"✓ Loaded SAM from Excel with {sam_records} records")
        self._input_sets = sets
        self._use_dynamic_sets = dynamic_sets
        self._resolved_sets = (
            sets if sets is not None
            else (derive_dynamic_sets_from_sam(self.sam_data) if dynamic_sets else None)
        )
