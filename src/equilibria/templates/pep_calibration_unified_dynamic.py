"""
PEP unified calibration with dynamic set derivation from SAM.
"""

from __future__ import annotations

from pathlib import Path

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel


class PEPModelCalibratorDynamic(PEPModelCalibrator):
    """GDX-based unified calibrator with dynamic sets enabled by default."""

    def __init__(self, sam_file: Path | str, val_par_file: Path | str | None = None):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)


class PEPModelCalibratorExcelDynamic(PEPModelCalibratorExcel):
    """Excel-based unified calibrator with dynamic sets enabled by default."""

    def __init__(self, sam_file: Path | str, val_par_file: Path | str | None = None):
        super().__init__(sam_file=sam_file, val_par_file=val_par_file, dynamic_sets=True)
