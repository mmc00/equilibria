"""
PEP Unified Model Calibration Runner

This module provides a unified interface to run all calibration phases
and produce a complete calibrated model state.

Usage:
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    
    calibrator = PEPModelCalibrator(sam_file_path)
    result = calibrator.calibrate()
    
    # Access all calibrated variables
    print(f"GDP: {result.gdp['GDP_BPO']}")
    print(f"Imports: {result.trade['IMO']}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values
from equilibria.templates.pep_calibration_income import IncomeCalibrator
from equilibria.templates.pep_calibration_production import ProductionCalibrator
from equilibria.templates.pep_calibration_trade import TradeCalibrator
from equilibria.templates.pep_calibration_final import FinalCalibrator
from equilibria.templates.pep_dynamic_sets import derive_dynamic_sets_from_sam
from equilibria.templates.pep_val_par_loader import load_val_par

logger = logging.getLogger(__name__)


def _build_uppercase_sam_matrix(sam_data: dict[str, Any]) -> dict[tuple[str, str, str, str], float]:
    """Materialize SAM into an uppercase-keyed 4D matrix.

    GDX files produced from Excel can preserve mixed/lower-case element labels
    (for example ``gvt``/``row``/``hrp``), while the calibration code queries
    SAM entries using uppercase labels to match the original GAMS notation.
    Normalizing once at load time keeps the calibration path case-insensitive
    and avoids repeated raw GDX decoding inside each calibration module.
    """
    existing = sam_data.get("sam_matrix")
    if isinstance(existing, dict) and existing:
        out: dict[tuple[str, str, str, str], float] = {}
        for key, value in existing.items():
            norm_key = tuple(str(part).upper() for part in key)
            out[norm_key] = out.get(norm_key, 0.0) + float(value)
        return out

    records: list[tuple[tuple[str, ...], float]] = []

    for symbol in sam_data.get("symbols", []):
        if symbol.get("name") != "SAM":
            continue
        raw_records = symbol.get("records", [])
        if not isinstance(raw_records, list):
            continue
        for record in raw_records:
            indices = tuple(record.get("indices", []))
            if len(indices) != 4:
                continue
            records.append((indices, float(record.get("value", 0.0))))

    if not records:
        try:
            raw_values = read_parameter_values(sam_data, "SAM")
        except Exception:
            raw_values = {}
        records = [(tuple(key), float(value)) for key, value in raw_values.items()]

    out: dict[tuple[str, str, str, str], float] = {}
    for key, value in records:
        if len(key) != 4:
            continue
        norm_key = tuple(str(part).upper() for part in key)
        out[norm_key] = out.get(norm_key, 0.0) + float(value)
    return out


@dataclass
class PEPModelState:
    """Complete calibrated state of the PEP model."""
    
    # Model sets
    sets: dict[str, list[str]] = field(default_factory=dict)
    
    # Income block (Phase 1-2)
    income: dict[str, Any] = field(default_factory=dict)
    
    # Production block (Phase 3)
    production: dict[str, Any] = field(default_factory=dict)
    
    # Trade block (Phase 4)
    trade: dict[str, Any] = field(default_factory=dict)
    
    # Final integration (Phase 5)
    consumption: dict[str, Any] = field(default_factory=dict)
    gdp: dict[str, float] = field(default_factory=dict)
    real_variables: dict[str, Any] = field(default_factory=dict)
    les_parameters: dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "sets": self.sets,
            "income": self.income,
            "production": self.production,
            "trade": self.trade,
            "consumption": self.consumption,
            "gdp": self.gdp,
            "real_variables": self.real_variables,
            "les_parameters": self.les_parameters,
            "validation": self.validation,
        }
    
    def save_json(self, filepath: Path) -> None:
        """Save state to JSON file."""
        # Convert to dict and handle non-serializable types
        data = self.to_dict()
        
        def convert_value(v):
            if isinstance(v, dict):
                return {str(k): convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            elif isinstance(v, float):
                return round(v, 6)
            else:
                return v
        
        data = convert_value(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Model state saved to {filepath}")


class CalibrationReport(BaseModel):
    """Report generated from calibration run."""
    
    status: str = "pending"
    phases_completed: list[str] = []
    phases_failed: list[str] = []
    
    # Summary statistics
    total_households: int = 0
    total_sectors: int = 0
    total_commodities: int = 0
    
    # Key indicators
    gdp_bpo: float = 0.0
    gdp_mpo: float = 0.0
    total_exports: float = 0.0
    total_imports: float = 0.0
    trade_balance: float = 0.0
    total_consumption: float = 0.0
    total_investment: float = 0.0
    
    # Validation
    validation_passed: bool = False
    validation_warnings: list[str] = []
    validation_errors: list[str] = []
    
    def print_summary(self) -> None:
        """Print a formatted summary report."""
        print("\n" + "=" * 70)
        print("PEP MODEL CALIBRATION REPORT")
        print("=" * 70)
        
        print(f"\nStatus: {self.status}")
        
        print("\nCompleted Phases:")
        for phase in self.phases_completed:
            print(f"  ✓ {phase}")
        
        if self.phases_failed:
            print("\nFailed Phases:")
            for phase in self.phases_failed:
                print(f"  ✗ {phase}")
        
        print(f"\nModel Dimensions:")
        print(f"  Households:   {self.total_households}")
        print(f"  Sectors:      {self.total_sectors}")
        print(f"  Commodities:  {self.total_commodities}")
        
        print(f"\nKey Indicators:")
        print(f"  GDP (Basic Prices):     {self.gdp_bpo:15,.2f}")
        print(f"  GDP (Market Prices):    {self.gdp_mpo:15,.2f}")
        print(f"  Total Exports:          {self.total_exports:15,.2f}")
        print(f"  Total Imports:          {self.total_imports:15,.2f}")
        print(f"  Trade Balance:          {self.trade_balance:15,.2f}")
        print(f"  Total Consumption:      {self.total_consumption:15,.2f}")
        print(f"  Total Investment:       {self.total_investment:15,.2f}")
        
        print(f"\nValidation:")
        if self.validation_passed:
            print("  ✓ All validation checks passed")
        else:
            print("  ✗ Validation issues detected")
        
        if self.validation_warnings:
            print("\n  Warnings:")
            for warning in self.validation_warnings[:5]:  # Show first 5
                print(f"    - {warning}")
            if len(self.validation_warnings) > 5:
                print(f"    ... and {len(self.validation_warnings) - 5} more")
        
        if self.validation_errors:
            print("\n  Errors:")
            for error in self.validation_errors[:5]:  # Show first 5
                print(f"    - {error}")
            if len(self.validation_errors) > 5:
                print(f"    ... and {len(self.validation_errors) - 5} more")
        
        print("\n" + "=" * 70)


class PEPModelCalibrator:
    """Unified calibrator that runs all PEP calibration phases."""
    
    def __init__(
        self,
        sam_file: Path | str,
        val_par_file: Path | str | None = None,
        sets: dict[str, list[str]] | None = None,
        dynamic_sets: bool = False,
    ):
        """Initialize the unified calibrator.
        
        Args:
            sam_file: Path to SAM GDX file
            val_par_file: Optional path to VAL_PAR Excel file
        """
        self.sam_file = Path(sam_file)
        self.val_par_file = Path(val_par_file) if val_par_file else None
        self.val_par_data = load_val_par(self.val_par_file)
        self.state = PEPModelState()
        self.report = CalibrationReport()
        
        # Load SAM data
        logger.info(f"Loading SAM from {self.sam_file}")
        self.sam_data = read_gdx(self.sam_file)
        self.sam_data["sam_matrix"] = _build_uppercase_sam_matrix(self.sam_data)
        logger.info(f"✓ Loaded SAM with {len(self.sam_data.get('symbols', []))} symbols")
        self._input_sets = sets
        self._use_dynamic_sets = dynamic_sets
        self._resolved_sets = (
            sets if sets is not None
            else (derive_dynamic_sets_from_sam(self.sam_data) if dynamic_sets else None)
        )
    
    def calibrate(self) -> PEPModelState:
        """Run complete calibration of all phases.
        
        Returns:
            PEPModelState with all calibrated variables
        """
        logger.info("=" * 70)
        logger.info("STARTING UNIFIED PEP MODEL CALIBRATION")
        logger.info("=" * 70)
        
        try:
            # Phase 1-2: Income and Shares
            self._run_income_calibration()
            
            # Phase 3: Production
            self._run_production_calibration()
            
            # Phase 4: Trade
            self._run_trade_calibration()
            
            # Phase 5: Final Integration
            self._run_final_calibration()

            # Post-calibration savings closure: enforce equation-consistency for
            # SGO and SROWO so that EQ43 (government budget identity) and EQ45
            # (rest-of-world balance identity) hold exactly at the calibrated
            # point. This is the standard CGE practice when the input SAM is
            # not perfectly balanced (e.g. converted CRI SAM): savings absorb
            # the residual imbalance while the rest of the SAM is preserved.
            self._apply_savings_residual_closure()

            # Generate report
            self._generate_report()
            
            self.report.status = "completed"
            logger.info("✓ Unified calibration completed successfully")
            
        except Exception as e:
            self.report.status = "failed"
            logger.error(f"✗ Calibration failed: {e}")
            raise
        
        return self.state

    # Relative threshold above which the savings closure is skipped.
    # When |Δ / old| exceeds this, the delta is structural (not a rounding
    # artifact) and overriding SROWO/SGO would break EQ53.  Leave the
    # solver to close EQ43/EQ45 endogenously instead.
    _SAVINGS_CLOSURE_REL_TOL: float = 1e-4  # 0.01 %

    def _apply_savings_residual_closure(self) -> None:
        """Recalibrate SGO and SROWO as residuals of EQ43 / EQ45.

        The PEP CGE model treats government savings (SG) and rest-of-world
        savings (SROW) as endogenous residuals defined by:

            EQ43:  SG   = YG   - SUM(agng, TR(agng,'gvt'))   - G
            EQ45:  SROW = YROW - SUM(i,    PE_FOB(i)*EXD(i)) - SUM(agd, TR(agd,'row'))

        When the underlying SAM is perfectly balanced, reading SGO and SROWO
        directly from the SAM is equivalent to evaluating these formulas.
        However, converted SAMs (e.g. CRI from IEEM/ICIO) can carry small
        accounting inconsistencies that leave EQ43 and EQ45 non-zero at the
        calibrated point. We restore identity-consistency by overriding SGO
        and SROWO with the equation-implied values.

        This is the standard "savings closure" practice in CGE: the residual
        imbalance is absorbed in government / external savings rather than
        being silently distributed across the model.

        Guard: if the implied correction is larger than _SAVINGS_CLOSURE_REL_TOL
        (relative to the SAM-direct value), the discrepancy is structural rather
        than a rounding artifact.  In that case the SAM-direct values already
        satisfy EQ53 (GFCF identity) perfectly, and forcing the EQ45-implied
        SROWO would introduce a large EQ53 residual that prevents convergence.
        The function is skipped so the solver can close EQ43/EQ45 endogenously.
        """
        income = self.state.income
        consumption = self.state.consumption
        trade = self.state.trade
        sets = self.state.sets

        AGNG = sets.get("AGNG", [])
        AGD = sets.get("AGD", [])
        I = sets.get("I", [])

        TRO = income.get("TRO", {})

        # EQ43: SG = YG - sum(TR(agng,'gvt')) - G
        ygo = float(income.get("YGO", 0.0))
        go = float(consumption.get("GO", 0.0))
        tr_to_gvt = sum(float(TRO.get((agng, "gvt"), 0.0)) for agng in AGNG)
        sgo_new = ygo - tr_to_gvt - go

        # EQ45: SROW = YROW - sum(PE_FOB*EXD) - sum(TR(agd,'row'))
        yrowo = float(income.get("YROWO", 0.0))
        pe_fobo = trade.get("PE_FOBO", {})
        exdo = trade.get("EXDO", {})
        exports = sum(
            float(pe_fobo.get(i, 0.0)) * float(exdo.get(i, 0.0)) for i in I
        )
        tr_from_row = sum(float(TRO.get((agd, "row"), 0.0)) for agd in AGD)
        srowo_new = yrowo - exports - tr_from_row

        sgo_old = float(income.get("SGO", 0.0))
        srowo_old = float(income.get("SROWO", 0.0))

        sg_delta = sgo_new - sgo_old
        srow_delta = srowo_new - srowo_old

        # Guard: large corrections indicate a structural SAM mapping difference,
        # not a rounding error.  Applying the override would break EQ53, which
        # is harder for PATH to close than leaving EQ43/EQ45 as solver residuals.
        sg_rel = abs(sg_delta) / max(abs(sgo_old), 1.0)
        srow_rel = abs(srow_delta) / max(abs(srowo_old), 1.0)
        tol = self._SAVINGS_CLOSURE_REL_TOL
        if sg_rel > tol or srow_rel > tol:
            logger.info(
                "Savings closure SKIPPED (structural delta too large): "
                "ΔSGO=%+.2f (%.4f%%), ΔSROWO=%+.2f (%.4f%%). "
                "SAM-direct values preserved; EQ43/EQ45 resolved by solver.",
                sg_delta, sg_rel * 100,
                srow_delta, srow_rel * 100,
            )
            return

        # Update savings (only for small / rounding-level corrections)
        income["SGO"] = sgo_new
        income["SROWO"] = srowo_new
        # CABO = -SROWO (current account balance identity, EQ46)
        income["CABO"] = -srowo_new
        # Keep ITO = SH + SF + SG + SROW consistent so the macro identity holds
        sho_sum = sum(float(v) for v in income.get("SHO", {}).values())
        sfo_sum = sum(float(v) for v in income.get("SFO", {}).values())
        ito_new = sho_sum + sfo_sum + sgo_new + srowo_new
        ito_old = float(income.get("ITO", 0.0))
        income["ITO"] = ito_new

        # Keep GFCFO consistent with EQ53 (GFCF = IT - sum(PC*VSTK))
        # so the calibrated point satisfies the investment identity exactly.
        # PCO = 1.0 at base; VSTKO comes from consumption.
        pco = trade.get("PCO", {})
        vstko = consumption.get("VSTKO", {})
        vstk_value = sum(
            float(pco.get(i, 1.0)) * float(vstko.get(i, 0.0)) for i in I
        )
        gfcfo_old = float(consumption.get("GFCFO", 0.0))
        gfcfo_new = ito_new - vstk_value
        consumption["GFCFO"] = gfcfo_new

        logger.info(
            "Savings closure applied: ΔSGO=%+.2f (%.2f→%.2f), "
            "ΔSROWO=%+.2f (%.2f→%.2f), ΔITO=%+.2f, ΔGFCFO=%+.2f",
            sg_delta, sgo_old, sgo_new,
            srow_delta, srowo_old, srowo_new,
            ito_new - ito_old,
            gfcfo_new - gfcfo_old,
        )

    def _run_income_calibration(self) -> None:
        """Run Phase 1-2: Income and Shares calibration."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1-2: Income and Shares Calibration")
        logger.info("=" * 70)
        
        calibrator = IncomeCalibrator(
            self.sam_data,
            val_par_data=self.val_par_data,
            sets=self._resolved_sets,
        )
        result = calibrator.calibrate()
        
        # Store in state
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
        
        self.report.phases_completed.append("Income and Shares (Phase 1-2)")
        logger.info("✓ Income calibration complete")
    
    def _run_production_calibration(self) -> None:
        """Run Phase 3: Production calibration."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: Production Block Calibration")
        logger.info("=" * 70)
        
        # Get income result from state
        from equilibria.templates.pep_calibration_income import IncomeCalibrationResult
        income_result = IncomeCalibrationResult(**self.state.income)
        
        calibrator = ProductionCalibrator(
            self.sam_data,
            income_result,
            val_par_data=self.val_par_data,
            sets=self.state.sets,
        )
        result = calibrator.calibrate()
        
        # Store in state
        self.state.production = {
            "KDO": result.KDO,
            "LDO": result.LDO,
            "KDCO": result.KDCO,
            "LDCO": result.LDCO,
            "KSO": result.KSO,
            "LSO": result.LSO,
            "ttiwO": result.ttiwO,
            "WTIO": result.WTIO,
            "ttikO": result.ttikO,
            "RTIO": result.RTIO,
            "WCO": result.WCO,
            "RCO": result.RCO,
            "VAO": result.VAO,
            "PVAO": result.PVAO,
            "ttipO": result.ttipO,
            "PPO": result.PPO,
            "DIO": result.DIO,
            "CIO": result.CIO,
            "PCIO": result.PCIO,
            "DITO": result.DITO,
            "XSTO": result.XSTO,
            "PTO": result.PTO,
            "io": result.io,
            "v": result.v,
            "aij": result.aij,
            "GDP_BPO": result.GDP_BPO,
            "rho_KD": result.rho_KD,
            "beta_KD": result.beta_KD,
            "B_KD": result.B_KD,
            "sigma_KD": result.sigma_KD,
            "rho_LD": result.rho_LD,
            "beta_LD": result.beta_LD,
            "B_LD": result.B_LD,
            "sigma_LD": result.sigma_LD,
            "rho_VA": result.rho_VA,
            "sigma_VA": result.sigma_VA,
            "beta_VA": result.beta_VA,
            "B_VA": result.B_VA,
        }
        
        self.report.phases_completed.append("Production (Phase 3)")
        logger.info("✓ Production calibration complete")
    
    def _run_trade_calibration(self) -> None:
        """Run Phase 4: Trade calibration."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: Trade Block Calibration")
        logger.info("=" * 70)
        
        # Get previous results
        from equilibria.templates.pep_calibration_income import IncomeCalibrationResult
        from equilibria.templates.pep_calibration_production import ProductionCalibrationResult
        
        income_result = IncomeCalibrationResult(**self.state.income)
        production_result = ProductionCalibrationResult(**self.state.production)
        
        calibrator = TradeCalibrator(
            self.sam_data,
            production_result,
            val_par_data=self.val_par_data,
            sets=self.state.sets,
        )
        result = calibrator.calibrate()
        
        # Store in state
        self.state.trade = {
            "IMO": result.IMO,
            "DDO": result.DDO,
            "QO": result.QO,
            "EXDO": result.EXDO,
            "EXO": result.EXO,
            "DSO": result.DSO,
            "XSO": result.XSO,
            "PCO": result.PCO,
            "PLO": result.PLO,
            "PMO": result.PMO,
            "PDO": result.PDO,
            "PEO": result.PEO,
            "PE_FOBO": result.PE_FOBO,
            "PWXO": result.PWXO,
            "PWMO": result.PWMO,
            "PO": result.PO,
            "tticO": result.tticO,
            "ttimO": result.ttimO,
            "ttixO": result.ttixO,
            "tmrg": result.tmrg,
            "tmrg_X": result.tmrg_X,
            "TICO": result.TICO,
            "TIMO": result.TIMO,
            "TIXO": result.TIXO,
            "eO": result.eO,
            "rho_XT": result.rho_XT,
            "beta_XT": result.beta_XT,
            "B_XT": result.B_XT,
            "rho_X": result.rho_X,
            "beta_X": result.beta_X,
            "B_X": result.B_X,
            "rho_M": result.rho_M,
            "beta_M": result.beta_M,
            "B_M": result.B_M,
            "MRGNO": result.MRGNO,
        }
        
        self.report.phases_completed.append("Trade (Phase 4)")
        logger.info("✓ Trade calibration complete")
    
    def _run_final_calibration(self) -> None:
        """Run Phase 5: Final integration."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: Final Integration")
        logger.info("=" * 70)
        
        # Get all previous results
        from equilibria.templates.pep_calibration_income import IncomeCalibrationResult
        from equilibria.templates.pep_calibration_production import ProductionCalibrationResult
        from equilibria.templates.pep_calibration_trade import TradeCalibrationResult
        
        income_result = IncomeCalibrationResult(**self.state.income)
        production_result = ProductionCalibrationResult(**self.state.production)
        trade_result = TradeCalibrationResult(**self.state.trade)
        
        calibrator = FinalCalibrator(
            self.sam_data,
            income_result,
            production_result,
            trade_result,
            val_par_data=self.val_par_data,
            sets=self.state.sets,
        )
        result = calibrator.calibrate()
        
        # Store in state
        self.state.consumption = {
            "CO": result.CO,
            "CGO": result.CGO,
            "GO": result.GO,
            "INVO": result.INVO,
            "VSTKO": result.VSTKO,
            "GFCFO": result.GFCFO,
        }
        
        self.state.gdp = {
            "GDP_BPO": result.GDP_BPO,
            "GDP_MPO": result.GDP_MPO,
            "GDP_IBO": result.GDP_IBO,
            "GDP_FDO": result.GDP_FDO,
        }
        
        self.state.real_variables = {
            "CTH_REALO": result.CTH_REALO,
            "G_REALO": result.G_REALO,
            "GDP_BP_REALO": result.GDP_BP_REALO,
            "GDP_MP_REALO": result.GDP_MP_REALO,
            "GFCF_REALO": result.GFCF_REALO,
            "PIXCONO": result.PIXCONO,
            "PIXGDPO": result.PIXGDPO,
            "PIXGVTO": result.PIXGVTO,
            "PIXINVO": result.PIXINVO,
        }
        
        self.state.les_parameters = {
            "sigma_Y": result.sigma_Y,
            "gamma_LES": result.gamma_LES,
            "CMINO": result.CMINO,
            "frisch": result.frisch,
        }
        
        self.state.validation = {
            "passed": result.validation_passed,
            "errors": result.validation_errors,
        }
        
        self.report.phases_completed.append("Final Integration (Phase 5)")
        logger.info("✓ Final calibration complete")
    
    def _generate_report(self) -> None:
        """Generate calibration report."""
        # Model dimensions
        self.report.total_households = len(self.state.sets.get("H", []))
        self.report.total_sectors = len(self.state.sets.get("J", []))
        self.report.total_commodities = len(self.state.sets.get("I", []))
        
        # Key indicators
        self.report.gdp_bpo = self.state.gdp.get("GDP_BPO", 0.0)
        self.report.gdp_mpo = self.state.gdp.get("GDP_MPO", 0.0)
        
        trade_data = self.state.trade
        self.report.total_exports = sum(trade_data.get("EXDO", {}).values())
        self.report.total_imports = sum(trade_data.get("IMO", {}).values())
        self.report.trade_balance = self.report.total_exports - self.report.total_imports
        
        consumption_data = self.state.consumption
        co_data = consumption_data.get("CO", {})
        self.report.total_consumption = sum(co_data.values())
        self.report.total_investment = sum(consumption_data.get("INVO", {}).values())
        
        # Validation
        validation_data = self.state.validation
        self.report.validation_passed = validation_data.get("passed", False)
        self.report.validation_errors = validation_data.get("errors", [])
        
        # Add warnings for GDP differences
        gdp_diff = abs(self.report.gdp_bpo - self.state.gdp.get("GDP_FDO", 0.0))
        if gdp_diff > 0.01 * self.report.gdp_bpo:
            self.report.validation_warnings.append(
                f"GDP_BPO differs from GDP_FDO by {gdp_diff:,.2f}"
            )
    
    def save_report(self, filepath: Path | str) -> None:
        """Save calibration report to file."""
        filepath = Path(filepath)
        report_dict = self.report.model_dump()
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def print_report(self) -> None:
        """Print calibration report to console."""
        self.report.print_summary()
