"""PEP-CRI calibration — extends PEP standard with cross-border labor (L→ROW).

This template is for SAMs derived from ICIO/IEEM databases (e.g. Costa Rica)
that record compensation paid to non-resident employees (cross-border workers).

Extended variables / parameters vs. standard PEP-1-1:
  - YROWO:       += SUM[l, SAM('AG','ROW','L',l)]   (labor paid to ROW)
  - lambda_WL:   includes ('row', l) entries

The corresponding GAMS counterpart is PEP-1-1_v2_1_cri.gms.

Hierarchy:
    IncomeCalibrator          (standard GAMS PEP-1-1)
        └── PEPCRIIncomeCalibrator   (adds L→ROW to YROWO and lambda_WL)

    PEPModelCalibrator        (standard GDX-based)
        └── PEPCRIModelCalibrator
    PEPModelCalibratorExcel   (standard Excel-based)
        └── PEPCRIModelCalibratorExcel
    PEPModelCalibratorDynamicSAM
        └── PEPCRIModelCalibratorDynamicSAM
    PEPModelCalibratorExcelDynamicSAM
        └── PEPCRIModelCalibratorExcelDynamicSAM
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.templates.pep_calibration_income import IncomeCalibrator
from equilibria.templates.pep_calibration_unified import (
    PEPModelCalibrator,
    PEPModelState,
)
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_calibration_unified_dynamic import (
    PEPModelCalibratorDynamicSAM,
    PEPModelCalibratorExcelDynamicSAM,
)

import logging

logger = logging.getLogger(__name__)


class PEPCRIIncomeCalibrator(IncomeCalibrator):
    """Income calibrator for CRI/ICIO SAMs.

    Adds cross-border labor compensation (L→ROW) to YROWO and lambda_WL,
    matching the GAMS formulas in PEP-1-1_v2_1_cri.gms.
    """

    def _calibrate_rest_of_world(self) -> None:
        """Calibrate YROWO including cross-border labor (L→ROW).

        CRI formula (extends standard GAMS):
            YROWO = SUM[i, IMO(i)]
                  + SUM[k, lambda_RK('row',k)]
                  + SUM[ag, TRO('row',ag)]
                  + SUM[l, SAM('AG','ROW','L',l)]   ← CRI extension
        """
        super()._calibrate_rest_of_world()

        L = self.sets['L']
        labor_to_row = sum(
            self._get_sam_value('SAM', 'AG', 'ROW', 'L', l.upper()) for l in L
        )
        self.result.YROWO += labor_to_row

    def _calibrate_shares(self) -> None:
        """Calibrate shares including lambda_WL('row', l) for cross-border workers.

        CRI formula (extends standard GAMS):
            lambda_WL('row', l) = SAM('AG','ROW','L',l) / SUM[j, LDO(l,j)]
        """
        super()._calibrate_shares()

        J = self.sets['J']
        L = self.sets['L']

        # Add lambda_WL for 'row' (cross-border labor share)
        for l in L:
            l_upper = l.upper()
            lambda_wl_raw = self._get_sam_value('SAM', 'AG', 'ROW', 'L', l_upper)
            ldo_sum = sum(
                self._get_sam_value('SAM', 'L', l_upper, 'J', j.upper())
                for j in J
            )
            if ldo_sum != 0:
                self.result.lambda_WL[("row", l)] = lambda_wl_raw / ldo_sum


def _make_cri_income_calibration_mixin():
    """Return a mixin that overrides _run_income_calibration to use PEPCRIIncomeCalibrator."""

    class _CRIMixin:
        def _run_income_calibration(self) -> None:
            """Phase 1-2 with CRI cross-border labor extension."""
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 1-2: Income and Shares Calibration [CRI extension]")
            logger.info("=" * 70)

            calibrator = PEPCRIIncomeCalibrator(
                self.sam_data,
                val_par_data=self.val_par_data,
                sets=self._resolved_sets,
            )
            result = calibrator.calibrate()

            # Store in state — identical to base PEPModelCalibrator._run_income_calibration
            self.state.sets = calibrator.sets
            self.state.income = {
                "YHKO": result.YHKO,
                "YHLO": result.YHLO,
                "YHTRO": result.YHTRO,
                "YHO": result.YHO,
                "YDHO": result.YDHO,
                "CTHO": result.CTHO,
                "TDHO": result.TDHO,
                "SHO": result.SHO,
                "YFKO": result.YFKO,
                "YFTRO": result.YFTRO,
                "YFO": result.YFO,
                "YDFO": result.YDFO,
                "TDFO": result.TDFO,
                "SFO": result.SFO,
                "YGKO": result.YGKO,
                "TDHTO": result.TDHTO,
                "TDFTO": result.TDFTO,
                "TICTO": result.TICTO,
                "TIMTO": result.TIMTO,
                "TIXTO": result.TIXTO,
                "TIWTO": result.TIWTO,
                "TIKTO": result.TIKTO,
                "TIPTO": result.TIPTO,
                "TPRODNO": result.TPRODNO,
                "TPRCTSO": result.TPRCTSO,
                "YGTRO": result.YGTRO,
                "YGO": result.YGO,
                "YROWO": result.YROWO,
                "CABO": result.CABO,
                "SROWO": result.SROWO,
                "ITO": result.ITO,
                "SGO": result.SGO,
                "TRO": result.TRO,
                "lambda_RK": result.lambda_RK,
                "lambda_WL": result.lambda_WL,
                "lambda_TR_households": result.lambda_TR_households,
                "lambda_TR_firms": result.lambda_TR_firms,
                "sh1O": result.sh1O,
                "tr1O": result.tr1O,
                "sh0O": self.val_par_data.get("sh0O", {}),
                "tr0O": self.val_par_data.get("tr0O", {}),
                "ttdf0O": self.val_par_data.get("ttdf0O", {}),
                "ttdh0O": self.val_par_data.get("ttdh0O", {}),
            }

            self.report.phases_completed.append("Income and Shares — CRI (Phase 1-2)")
            logger.info("✓ CRI income calibration complete (YROWO includes L→ROW)")

    return _CRIMixin


_CRIMixin = _make_cri_income_calibration_mixin()


class PEPCRIModelCalibrator(_CRIMixin, PEPModelCalibrator):
    """GDX-based calibrator for CRI/ICIO SAMs (standard dynamic_sets=False)."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        sets: dict[str, Any] | None = None,
        dynamic_sets: bool = False,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            sets=sets,
            dynamic_sets=dynamic_sets,
        )


class PEPCRIModelCalibratorExcel(_CRIMixin, PEPModelCalibratorExcel):
    """Excel-based calibrator for CRI/ICIO SAMs."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        sets: dict[str, Any] | None = None,
        dynamic_sets: bool = False,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            sets=sets,
            dynamic_sets=dynamic_sets,
        )


class PEPCRIModelCalibratorDynamicSAM(_CRIMixin, PEPModelCalibratorDynamicSAM):
    """GDX dynamic-SAM calibrator for CRI/ICIO SAMs."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        accounts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            accounts=accounts,
        )


class PEPCRIModelCalibratorExcelDynamicSAM(_CRIMixin, PEPModelCalibratorExcelDynamicSAM):
    """Excel dynamic-SAM calibrator for CRI/ICIO SAMs."""

    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        accounts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            sam_file=sam_file,
            val_par_file=val_par_file,
            accounts=accounts,
        )
