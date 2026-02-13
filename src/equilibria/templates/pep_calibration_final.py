"""
PEP Final Calibration and Integration (GAMS Sections 4.6.4, 4.7, 4.8)

This module implements:
- Consumption block calibration (LES parameters)
- GDP calculations (GDP_MPO, GDP_IBO, GDP_FDO)
- Real variable calibration
- Final model integration and validation

GAMS references:
- Section 4.6.4: LES parameters (lines 668-676)
- Section 4.7: GDP calculations (lines 678-685)
- Section 4.8: Real variables (lines 687-692)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinalCalibrationResult(BaseModel):
    """Container for all final calibration results."""

    # Consumption block (lines 330, 525-529)
    CO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Consumption of commodity i by household h (CO(i,h))",
    )
    CGO: dict[str, float] = Field(
        default_factory=dict,
        description="Public final consumption of commodity i (CGO(i))",
    )
    GO: float = Field(default=0.0, description="Total government expenditure (GO)")

    # Investment and stock (lines 338-339, 530)
    INVO: dict[str, float] = Field(
        default_factory=dict,
        description="Final demand for investment (INVO(i))",
    )
    VSTKO: dict[str, float] = Field(
        default_factory=dict, description="Inventory change (VSTKO(i))"
    )
    GFCFO: float = Field(
        default=0.0, description="Gross fixed capital formation (GFCFO)"
    )

    # LES parameters (lines 673-676)
    sigma_Y: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Income elasticity of demand (sigma_Y(i,h))",
    )
    gamma_LES: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="LES marginal budget share (gamma_LES(i,h))",
    )
    CMINO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Minimum consumption (CMINO(i,h))",
    )
    frisch: dict[str, float] = Field(
        default_factory=dict, description="Frisch parameter (frisch(h))"
    )

    # GDP measures (lines 678-685)
    GDP_BPO: float = Field(default=0.0, description="GDP at basic prices (GDP_BPO)")
    GDP_MPO: float = Field(
        default=0.0, description="GDP at market prices (GDP_MPO)"
    )
    GDP_IBO: float = Field(
        default=0.0, description="GDP by income approach (GDP_IBO)"
    )
    GDP_FDO: float = Field(
        default=0.0, description="GDP by expenditure approach (GDP_FDO)"
    )

    # Real variables (lines 687-692)
    CTH_REALO: dict[str, float] = Field(
        default_factory=dict,
        description="Real consumption budget (CTH_REALO(h))",
    )
    G_REALO: float = Field(
        default=0.0, description="Real government expenditure (G_REALO)"
    )
    GDP_BP_REALO: float = Field(
        default=0.0, description="Real GDP at basic prices (GDP_BP_REALO)"
    )
    GDP_MP_REALO: float = Field(
        default=0.0, description="Real GDP at market prices (GDP_MP_REALO)"
    )
    GFCF_REALO: float = Field(
        default=0.0, description="Real gross fixed capital formation (GFCF_REALO)"
    )

    # Price indices (lines 567, 571, 575, 579)
    PIXCONO: float = Field(default=1.0, description="Consumer price index (PIXCONO)")
    PIXGDPO: float = Field(default=1.0, description="GDP deflator (PIXGDPO)")
    PIXGVTO: float = Field(default=1.0, description="Government price index (PIXGVTO)")
    PIXINVO: float = Field(
        default=1.0, description="Investment price index (PIXINVO)"
    )

    # Validation results
    validation_passed: bool = Field(
        default=False, description="Whether all validation checks passed"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors"
    )


class FinalCalibrator:
    """Final model calibration integrating all blocks."""

    def __init__(
        self,
        sam_data: dict[str, Any],
        income_result: Any,
        production_result: Any,
        trade_result: Any,
        val_par_data: dict[str, Any] | None = None,
        sets: dict[str, list[str]] | None = None,
    ):
        """Initialize the final calibrator.

        Args:
            sam_data: Dictionary with SAM data
            income_result: IncomeCalibrationResult
            production_result: ProductionCalibrationResult
            trade_result: TradeCalibrationResult
            val_par_data: Optional VAL_PAR parameters
            sets: Dictionary with set definitions
        """
        self.sam = sam_data
        self.income = income_result
        self.production = production_result
        self.trade = trade_result
        self.sets = sets or self._detect_sets()
        self.result = FinalCalibrationResult()

        # Load VAL_PAR parameters for LES
        if val_par_data is None:
            val_par_path = Path(
                "/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/VAL_PAR.xlsx"
            )
            self.val_par = self._read_val_par_excel(val_par_path)
        else:
            self.val_par = val_par_data

    def _detect_sets(self) -> dict[str, list[str]]:
        """Auto-detect sets from SAM data."""
        return {
            "H": ["hrp", "hup", "hrr", "hur"],
            "F": ["firm"],
            "K": ["cap", "land"],
            "L": ["usk", "sk"],
            "J": ["agr", "ind", "ser", "adm"],
            "I": ["agr", "food", "othind", "ser", "adm"],
            "AG": ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"],
            "AGNG": ["hrp", "hup", "hrr", "hur", "firm", "row"],
        }

    def _read_val_par_excel(self, filepath: Path) -> dict[str, Any]:
        """Read VAL_PAR.xlsx for LES parameters."""
        import pandas as pd

        logger.info(f"Reading VAL_PAR for LES parameters from: {filepath}")

        try:
            df = pd.read_excel(filepath, sheet_name="PAR", header=None)
        except Exception as e:
            logger.warning(f"Could not read VAL_PAR Excel: {e}. Using defaults.")
            return self._get_default_val_par()

        params = {"sigma_Y": {}, "frisch": {}}

        # Frisch parameters - typically in PARAG sheet or specific cells
        households = ["HRP", "HUP", "HRR", "HUR"]

        # Try to read frisch from PARAG sheet
        try:
            df_ag = pd.read_excel(filepath, sheet_name="PARAG", header=None)
            # Frisch is typically in a specific location - use defaults for now
            # TODO: Properly parse PARAG sheet structure
            for h in households:
                params["frisch"][h.lower()] = -2.0
        except Exception:
            for h in households:
                params["frisch"][h.lower()] = -2.0

        # Income elasticities (sigma_Y) from PAR sheet
        # Typically in rows 20-24 (0-indexed: 19-23) for each household
        commodities = ["AGR", "FOOD", "OTHIND", "SER"]
        for i, h in enumerate(households):
            for j, comm in enumerate(commodities):
                row_idx = 20 + j
                try:
                    val = df.iloc[row_idx, 6 + i]  # Columns 6-9 for households
                    if pd.notna(val):
                        params["sigma_Y"][(comm.lower(), h.lower())] = float(val)
                    else:
                        params["sigma_Y"][(comm.lower(), h.lower())] = 1.0
                except (IndexError, ValueError):
                    params["sigma_Y"][(comm.lower(), h.lower())] = 1.0

        logger.info("Successfully read VAL_PAR LES parameters")
        return params

    def _get_default_val_par(self) -> dict[str, Any]:
        """Return default VAL_PAR parameters for LES."""
        households = ["hrp", "hup", "hrr", "hur"]
        commodities = ["agr", "food", "othind", "ser"]

        return {
            "sigma_Y": {(c, h): 1.0 for c in commodities for h in households},
            "frisch": {h: -2.0 for h in households},
        }

    def _get_sam_value(self, *indices) -> float:
        """Get value from SAM matrix."""
        try:
            indices_upper = tuple(str(idx).upper() for idx in indices)

            sam_matrix = self.sam.get("sam_matrix")
            if isinstance(sam_matrix, dict):
                if len(indices_upper) == 4:
                    return float(sam_matrix.get(indices_upper, 0.0))
                total = 0.0
                for key, value in sam_matrix.items():
                    if all(i >= len(indices_upper) or key[i] == indices_upper[i] for i in range(len(indices_upper))):
                        total += float(value)
                return total

            from equilibria.babel.gdx.decoder import decode_parameter_delta
            from equilibria.babel.gdx.reader import read_data_sections

            sam_path = Path(self.sam.get("filepath", ""))
            raw_data = sam_path.read_bytes()
            data_sections = read_data_sections(raw_data)
            _, section = data_sections[0]

            elements = self.sam.get("elements", [])
            values = decode_parameter_delta(section, elements, 4)

            if len(indices_upper) == 4:
                return values.get(indices_upper, 0.0)

            total = 0.0
            for key, value in values.items():
                match = True
                for i, idx in enumerate(indices_upper):
                    if i < len(key) and key[i] != idx:
                        match = False
                        break
                if match:
                    total += value

            return total
        except Exception as e:
            logger.warning(f"Error reading SAM: {e}")
            return 0.0

    def calibrate(self) -> FinalCalibrationResult:
        """Run final calibration."""
        logger.info("Starting final model calibration (Phase 5)")

        logger.info("Step 1: Reading consumption data from SAM...")
        self._read_consumption_data()

        logger.info("Step 2: Calibrating LES parameters...")
        self._calibrate_les_parameters()

        logger.info("Step 3: Calibrating GDP measures...")
        self._calibrate_gdp_measures()

        logger.info("Step 4: Calibrating real variables...")
        self._calibrate_real_variables()

        logger.info("Step 5: Running validation checks...")
        self._validate_calibration()

        logger.info("Final calibration complete!")
        return self.result

    def _read_consumption_data(self) -> None:
        """Read consumption data from SAM (lines 330-331, 338-339)."""
        I = self.sets["I"]
        H = self.sets["H"]

        # CO(i,h) = SAM('I',i,'Agents',h)
        # GAMS 4.4 converts nominal SAM flows to quantities: CO(i,h)=CO(i,h)/PCO(i)
        for i in I:
            i_upper = i.upper()
            pco = self.trade.PCO.get(i, 1.0)
            for h in H:
                h_upper = h.upper()
                co = self._get_sam_value("I", i_upper, "AG", h_upper)
                if pco != 0:
                    co = co / pco
                if co != 0:
                    self.result.CO[(i, h)] = co

        # CGO(i) = SAM('I',i,'Agents','gvt')
        # GAMS 4.4 converts: CGO(i)=CGO(i)/PCO(i)
        for i in I:
            i_upper = i.upper()
            pco = self.trade.PCO.get(i, 1.0)
            cgo = self._get_sam_value("I", i_upper, "AG", "GVT")
            if pco != 0:
                cgo = cgo / pco
            if cgo != 0:
                self.result.CGO[i] = cgo

        # INVO(i) = SAM('I',i,'OTH','INV')
        # GAMS 4.4 converts: INVO(i)=INVO(i)/PCO(i)
        for i in I:
            i_upper = i.upper()
            pco = self.trade.PCO.get(i, 1.0)
            invo = self._get_sam_value("I", i_upper, "OTH", "INV")
            if pco != 0:
                invo = invo / pco
            if invo != 0:
                self.result.INVO[i] = invo

        # VSTKO(i) = SAM('I',i,'OTH','VSTK')
        # GAMS 4.4 converts: VSTKO(i)=VSTKO(i)/PCO(i)
        for i in I:
            i_upper = i.upper()
            pco = self.trade.PCO.get(i, 1.0)
            vstko = self._get_sam_value("I", i_upper, "OTH", "VSTK")
            if pco != 0:
                vstko = vstko / pco
            if vstko != 0:
                self.result.VSTKO[i] = vstko

        # Calculate GO (total government expenditure) from CGO
        # GO = SUM[i,PCO(i)*CGO(i)]
        self.result.GO = sum(
            self.trade.PCO.get(i, 1.0) * self.result.CGO.get(i, 0) for i in I
        )

        # Calculate GFCFO (line 530)
        # GFCFO = ITO - SUM[i,PCO(i)*VSTKO(i)]
        ito = self.income.ITO
        vstko_sum = sum(
            self.trade.PCO.get(i, 1.0) * self.result.VSTKO.get(i, 0) for i in I
        )
        self.result.GFCFO = ito - vstko_sum

        logger.info(
            f"Read {len(self.result.CO)} consumption entries, "
            f"{len(self.result.CGO)} government entries"
        )

    def _calibrate_les_parameters(self) -> None:
        """Calibrate LES parameters (lines 673-676)."""
        I = self.sets["I"]
        H = self.sets["H"]

        # Get parameters from VAL_PAR
        sigma_y_raw = self.val_par.get("sigma_Y", {})
        frisch = self.val_par.get("frisch", {h: -2.0 for h in H})

        # Assign frisch parameters
        for h in H:
            self.result.frisch[h] = frisch.get(h, -2.0)

        # sigma_Y(i,h) - first normalize to ensure sum = 1
        for h in H:
            ctho = self.income.CTHO.get(h, 0)

            # Calculate denominator: SUM[ij,sigma_Y(ij,h)*PCO(ij)*CO(ij,h)]
            denom = sum(
                sigma_y_raw.get((ij, h), 1.0)
                * self.trade.PCO.get(ij, 1.0)
                * self.result.CO.get((ij, h), 0)
                for ij in I
            )

            for i in I:
                # sigma_Y(i,h) = sigma_Y(i,h)*CTHO(h)/denom
                if denom != 0 and ctho != 0:
                    self.result.sigma_Y[(i, h)] = (
                        sigma_y_raw.get((i, h), 1.0) * ctho / denom
                    )
                else:
                    self.result.sigma_Y[(i, h)] = sigma_y_raw.get((i, h), 1.0)

        # gamma_LES(i,h) = PCO(i)*CO(i,h)*sigma_Y(i,h)/CTHO(h)
        for h in H:
            ctho = self.income.CTHO.get(h, 0)
            for i in I:
                co = self.result.CO.get((i, h), 0)
                pco = self.trade.PCO.get(i, 1.0)
                sigma_y = self.result.sigma_Y.get((i, h), 0)

                if ctho != 0:
                    self.result.gamma_LES[(i, h)] = pco * co * sigma_y / ctho

        # CMINO(i,h) = CO(i,h)+gamma_LES(i,h)*[CTHO(h)/(PCO(i)*frisch(h))]
        for h in H:
            ctho = self.income.CTHO.get(h, 0)
            frisch_h = self.result.frisch.get(h, -2.0)

            for i in I:
                co = self.result.CO.get((i, h), 0)
                pco = self.trade.PCO.get(i, 1.0)
                gamma = self.result.gamma_LES.get((i, h), 0)

                if pco != 0 and frisch_h != 0:
                    self.result.CMINO[(i, h)] = co + gamma * (
                        ctho / (pco * frisch_h)
                    )

        logger.info(f"Calibrated LES parameters for {len(self.result.gamma_LES)} entries")

    def _calibrate_gdp_measures(self) -> None:
        """Calibrate all GDP measures (lines 680-685)."""
        I = self.sets["I"]
        H = self.sets["H"]
        J = self.sets["J"]
        K = self.sets["K"]
        L = self.sets["L"]

        # GDP_BPO - already calculated in production calibration
        self.result.GDP_BPO = self.production.GDP_BPO

        # GDP_MPO = GDP_BPO + TPRCTSO (line 681)
        self.result.GDP_MPO = self.result.GDP_BPO + self.income.TPRCTSO

        # GDP_IBO (line 682-683)
        # GDP_IBO = SUM[(l,j),WO(l)*LDO(l,j)] + SUM[(k,j),RO(k,j)*KDO(k,j)] + TPRODNO + TPRCTSO
        labor_income = sum(
            1.0 * self.production.LDO.get((l, j), 0)  # WO(l) = 1.0 (numeraire)
            for l in L
            for j in J
        )
        capital_income = sum(
            1.0 * self.production.KDO.get((k, j), 0)  # RO(k,j) = 1.0 (numeraire)
            for k in K
            for j in J
        )
        self.result.GDP_IBO = (
            labor_income + capital_income + self.income.TPRODNO + self.income.TPRCTSO
        )

        # GDP_FDO (line 684-685)
        # GDP_FDO = SUM[i,PCO(i)*(SUM[h,CO(i,h)]+CGO(i)+INVO(i)+VSTKO(i))]
        #           + SUM[i,PE_FOBO(i)*EXDO(i)] - SUM[i,PWMO(i)*eO*IMO(i)]
        domestic_demand = sum(
            self.trade.PCO.get(i, 1.0)
            * (
                sum(self.result.CO.get((i, h), 0) for h in H)
                + self.result.CGO.get(i, 0)
                + self.result.INVO.get(i, 0)
                + self.result.VSTKO.get(i, 0)
            )
            for i in I
        )
        exports = sum(
            self.trade.PE_FOBO.get(i, 0) * self.trade.EXDO.get(i, 0) for i in I
        )
        imports = sum(
            self.trade.PWMO.get(i, 1.0)
            * self.trade.eO
            * self.trade.IMO.get(i, 0)
            for i in I
        )
        self.result.GDP_FDO = domestic_demand + exports - imports

        logger.info(
            f"GDP_BPO={self.result.GDP_BPO:,.2f}, "
            f"GDP_MPO={self.result.GDP_MPO:,.2f}, "
            f"GDP_IBO={self.result.GDP_IBO:,.2f}, "
            f"GDP_FDO={self.result.GDP_FDO:,.2f}"
        )

    def _calibrate_real_variables(self) -> None:
        """Calibrate real variables (lines 687-692)."""
        # Price indices are all 1.0 in base year
        self.result.PIXCONO = 1.0
        self.result.PIXGDPO = 1.0
        self.result.PIXGVTO = 1.0
        self.result.PIXINVO = 1.0

        # CTH_REALO(h) = CTHO(h)/PIXCONO
        for h in self.sets["H"]:
            ctho = self.income.CTHO.get(h, 0)
            self.result.CTH_REALO[h] = ctho / self.result.PIXCONO

        # G_REALO = GO/PIXGVTO
        self.result.G_REALO = self.result.GO / self.result.PIXGVTO

        # GDP_BP_REALO = GDP_BPO/PIXGDPO
        self.result.GDP_BP_REALO = self.result.GDP_BPO / self.result.PIXGDPO

        # GDP_MP_REALO = GDP_MPO/PIXCONO
        self.result.GDP_MP_REALO = self.result.GDP_MPO / self.result.PIXCONO

        # GFCF_REALO = GFCFO/PIXINVO
        self.result.GFCF_REALO = self.result.GFCFO / self.result.PIXINVO

        logger.info("Real variables calibrated (all indices = 1.0 in base year)")

    def _validate_calibration(self) -> None:
        """Run validation checks on calibration."""
        errors = []

        # Check 1: GDP consistency (all three measures should be close)
        gdp_diff_ib = abs(self.result.GDP_BPO - self.result.GDP_IBO)
        gdp_diff_fd = abs(self.result.GDP_BPO - self.result.GDP_FDO)
        tolerance = 0.01 * self.result.GDP_BPO  # 1% tolerance

        if gdp_diff_ib > tolerance:
            errors.append(
                f"GDP_BPO ({self.result.GDP_BPO:,.2f}) differs from "
                f"GDP_IBO ({self.result.GDP_IBO:,.2f}) by {gdp_diff_ib:,.2f}"
            )

        if gdp_diff_fd > tolerance:
            errors.append(
                f"GDP_BPO ({self.result.GDP_BPO:,.2f}) differs from "
                f"GDP_FDO ({self.result.GDP_FDO:,.2f}) by {gdp_diff_fd:,.2f}"
            )

        # Check 2: LES parameter constraints
        for h in self.sets["H"]:
            gamma_sum = sum(
                self.result.gamma_LES.get((i, h), 0) for i in self.sets["I"]
            )
            # Gamma should sum to approximately 1
            if abs(gamma_sum - 1.0) > 0.01 and gamma_sum > 0:
                errors.append(
                    f"Household {h}: gamma_LES sum = {gamma_sum:.4f} (should be ~1.0)"
                )

        # Check 3: Government budget
        # GO should equal CGO + transfers + subsidies (simplified check)

        self.result.validation_errors = errors
        self.result.validation_passed = len(errors) == 0

        if errors:
            for error in errors:
                logger.warning(f"Validation error: {error}")
        else:
            logger.info("All validation checks passed!")
