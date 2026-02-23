"""
PEP Production Block Calibration (GAMS Section 4.4-4.6)

This module implements the calibration of production variables following
the exact order and formulas from GAMS PEP-1-1_v2_1_modular.gms
lines 538-650+.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProductionCalibrationResult(BaseModel):
    """Container for all calibrated production variables."""

    # Factor demands (lines 341-342, 545-553)
    KDO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Capital demand by type k and sector j (KDO(k,j))",
    )
    LDO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Labor demand by type l and sector j (LDO(l,j))",
    )
    KDCO: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate capital demand by sector j (KDCO(j))",
    )
    LDCO: dict[str, float] = Field(
        default_factory=dict, description="Aggregate labor demand by sector j (LDCO(j))"
    )
    KSO: dict[str, float] = Field(
        default_factory=dict, description="Capital supply by type k (KSO(k))"
    )
    LSO: dict[str, float] = Field(
        default_factory=dict, description="Labor supply by type l (LSO(l))"
    )

    # Factor prices and taxes (lines 538-544)
    ttiwO: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Labor tax rate (TIWO(l,j)/LDO(l,j))"
    )
    WTIO: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Wage including tax (WO(l)*(1+ttiwO(l,j)))"
    )
    ttikO: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Capital tax rate (TIKO(k,j)/KDO(k,j))"
    )
    RTIO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Rental rate including tax (RO(k,j)*(1+ttikO(k,j)))",
    )
    WCO: dict[str, float] = Field(
        default_factory=dict, description="Composite wage by sector j (WCO(j))"
    )
    RCO: dict[str, float] = Field(
        default_factory=dict, description="Composite rental rate by sector j (RCO(j))"
    )

    # Value added (lines 554-555)
    VAO: dict[str, float] = Field(
        default_factory=dict,
        description="Value added by sector j (VAO(j) = LDCO(j) + KDCO(j))",
    )
    PVAO: dict[str, float] = Field(
        default_factory=dict, description="Price of value added by sector j (PVAO(j))"
    )

    # Production tax (lines 557-559)
    ttipO: dict[str, float] = Field(
        default_factory=dict, description="Production tax rate by sector j (ttipO(j))"
    )
    PPO: dict[str, float] = Field(
        default_factory=dict, description="Producer price by sector j (PPO(j))"
    )

    # Intermediate consumption (lines 520-524)
    DIO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Intermediate consumption of commodity i by sector j (DIO(i,j))",
    )
    CIO: dict[str, float] = Field(
        default_factory=dict,
        description="Total intermediate consumption by sector j (CIO(j))",
    )
    PCIO: dict[str, float] = Field(
        default_factory=dict,
        description="Intermediate consumption price index by sector j (PCIO(j))",
    )
    DITO: dict[str, float] = Field(
        default_factory=dict,
        description="Total intermediate demand for commodity i (DITO(i))",
    )

    # Output and prices
    XSTO: dict[str, float] = Field(
        default_factory=dict, description="Total aggregate output by sector j (XSTO(j))"
    )
    PTO: dict[str, float] = Field(
        default_factory=dict, description="Average output price by sector j (PTO(j))"
    )

    # Technical coefficients (lines 600-601)
    io: dict[str, float] = Field(
        default_factory=dict,
        description="Intermediate input coefficient by sector j (io(j) = CIO(j)/XSTO(j))",
    )
    v: dict[str, float] = Field(
        default_factory=dict,
        description="Value added coefficient by sector j (v(j) = VAO(j)/XSTO(j))",
    )
    aij: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Intermediate input share (aij(i,j) = DIO(i,j)/CIO(j))",
    )

    # GDP
    GDP_BPO: float = Field(default=0.0, description="GDP at basic prices (GDP_BPO)")

    # CES/CET parameters (lines 638-650+)
    rho_KD: dict[str, float] = Field(
        default_factory=dict,
        description="CES elasticity parameter for capital (rho_KD(j) = (1-sigma_KD(j))/sigma_KD(j))",
    )
    beta_KD: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="CES share parameter for capital (beta_KD(k,j))",
    )
    B_KD: dict[str, float] = Field(
        default_factory=dict, description="CES scale parameter for capital (B_KD(j))"
    )

    rho_LD: dict[str, float] = Field(
        default_factory=dict,
        description="CES elasticity parameter for labor (rho_LD(j) = (1-sigma_LD(j))/sigma_LD(j))",
    )
    beta_LD: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="CES share parameter for labor (beta_LD(l,j))"
    )
    B_LD: dict[str, float] = Field(
        default_factory=dict, description="CES scale parameter for labor (B_LD(j))"
    )

    # Value added CES parameters (lines 656-666)
    rho_VA: dict[str, float] = Field(
        default_factory=dict,
        description="CES elasticity parameter for value added (rho_VA(j))",
    )
    beta_VA: dict[str, float] = Field(
        default_factory=dict,
        description="CES share parameter for value added (beta_VA(j))",
    )
    B_VA: dict[str, float] = Field(
        default_factory=dict,
        description="CES scale parameter for value added (B_VA(j))",
    )


def read_val_par_excel(filepath: Path) -> dict[str, Any]:
    """Read VAL_PAR.xlsx and extract parameters.

    Args:
        filepath: Path to VAL_PAR.xlsx

    Returns:
        Dictionary with all parameters organized by type
    """
    logger.info(f"Reading VAL_PAR from: {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name="PAR", header=None)
    except Exception as e:
        logger.warning(f"Could not read VAL_PAR Excel: {e}. Using defaults.")
        return _get_default_val_par()

    params = {
        "sigma_KD": {},
        "sigma_LD": {},
        "sigma_VA": {},
        "sigma_XT": {},
        "sigma_M": {},
        "sigma_XD": {},
    }

    # Extract sectors (j) - rows 5-8 (0-indexed: 4-7)
    sectors = ["AGR", "IND", "SER", "ADM"]
    for i, sector in enumerate(sectors):
        row_idx = 5 + i
        try:
            params["sigma_KD"][sector] = (
                float(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else 0.8
            )
            params["sigma_LD"][sector] = (
                float(df.iloc[row_idx, 2]) if pd.notna(df.iloc[row_idx, 2]) else 0.8
            )
            params["sigma_VA"][sector] = (
                float(df.iloc[row_idx, 3]) if pd.notna(df.iloc[row_idx, 3]) else 1.5
            )
            params["sigma_XT"][sector] = (
                float(df.iloc[row_idx, 4]) if pd.notna(df.iloc[row_idx, 4]) else 2.0
            )
        except (IndexError, ValueError):
            params["sigma_KD"][sector] = 0.8
            params["sigma_LD"][sector] = 0.8
            params["sigma_VA"][sector] = 1.5
            params["sigma_XT"][sector] = 2.0

    # Extract commodities (i) - rows 12-16 (0-indexed: 11-15)
    commodities = ["AGR", "FOOD", "OTHIND", "SER", "ADM"]
    for i, comm in enumerate(commodities):
        row_idx = 12 + i
        try:
            params["sigma_M"][comm] = (
                float(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else 2.0
            )
            params["sigma_XD"][comm] = (
                float(df.iloc[row_idx, 2]) if pd.notna(df.iloc[row_idx, 2]) else 2.0
            )
        except (IndexError, ValueError):
            params["sigma_M"][comm] = 2.0
            params["sigma_XD"][comm] = 2.0

    logger.info("Successfully read VAL_PAR parameters")
    return params


def _get_default_val_par() -> dict[str, Any]:
    """Return default VAL_PAR parameters."""
    sectors = ["AGR", "IND", "SER", "ADM"]
    commodities = ["AGR", "FOOD", "OTHIND", "SER", "ADM"]

    return {
        "sigma_KD": {s: 0.8 for s in sectors},
        "sigma_LD": {s: 0.8 for s in sectors},
        "sigma_VA": {s: 1.5 for s in sectors},
        "sigma_XT": {s: 2.0 for s in sectors},
        "sigma_M": {c: 2.0 for c in commodities},
        "sigma_XD": {c: 2.0 for c in commodities},
    }


class ProductionCalibrator:
    """Calibrates production block variables following GAMS Section 4.4-4.6."""

    def __init__(
        self,
        sam_data: dict[str, Any],
        income_result: Any,
        val_par_data: dict[str, Any] | None = None,
        sets: dict[str, list[str]] | None = None,
    ):
        """Initialize the production calibrator.

        Args:
            sam_data: Dictionary with SAM data from read_gdx()
            income_result: IncomeCalibrationResult with calibrated income variables
            val_par_data: Optional dictionary with VAL_PAR parameters
            sets: Dictionary with set definitions
        """
        self.sam = sam_data
        self.income = income_result
        self.sets = sets or self._detect_sets()
        self.result = ProductionCalibrationResult()

        # Load VAL_PAR parameters
        if val_par_data is None:
            val_par_path = (
                Path(__file__).resolve().parent
                / "reference"
                / "pep2"
                / "data"
                / "VAL_PAR.xlsx"
            )
            self.val_par = read_val_par_excel(val_par_path)
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
            "I": ["agr", "othind", "ser", "food", "adm"],
            "AG": ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"],
            "AGNG": ["hrp", "hup", "hrr", "hur", "firm", "row"],
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

    def _compute_pco_from_sam(self) -> dict[str, float]:
        """
        Reconstruct benchmark PCO(i) from SAM, matching GAMS calibration order.

        PCO(i) = [DDO(i) + IMO(i) + SUM[ij, tmrg(ij,i)] + TICO(i) + TIMO(i)]
                 / [DDO(i) + IMO(i)]
        """
        I = self.sets["I"]
        J = self.sets["J"]
        pco: dict[str, float] = {}

        for i in I:
            i_upper = i.upper()
            ddo = sum(
                self._get_sam_value("J", j.upper(), "I", i_upper)
                for j in J
            )
            imo = self._get_sam_value("AG", "ROW", "I", i_upper)
            tico = self._get_sam_value("AG", "TI", "I", i_upper)
            timo = self._get_sam_value("AG", "TM", "I", i_upper)
            margin_sum = sum(
                self._get_sam_value("I", ij.upper(), "I", i_upper)
                for ij in I
            )

            denominator = ddo + imo
            if abs(denominator) > 1e-12:
                pco[i] = (ddo + imo + margin_sum + tico + timo) / denominator
            else:
                pco[i] = 1.0

        return pco

    def _compute_xsto_from_sam(self) -> dict[str, float]:
        """
        Reconstruct benchmark XSTO(j) from SAM trade flows:
        XSTO(j) = SUM[i, XSO(j,i)] with XSO(j,i)=DSO(j,i)+EXO(j,i).
        """
        I = self.sets["I"]
        J = self.sets["J"]
        xsto: dict[str, float] = {}

        for j in J:
            j_upper = j.upper()
            total = 0.0
            for i in I:
                i_upper = i.upper()
                dso = self._get_sam_value("J", j_upper, "I", i_upper)
                exo = self._get_sam_value("J", j_upper, "X", i_upper)
                total += dso + exo
            xsto[j] = total

        return xsto

    def calibrate(self) -> ProductionCalibrationResult:
        """Run full production calibration."""
        logger.info("Starting production block calibration (Phase 3)")

        logger.info("Step 1: Calibrating factor demands and prices...")
        self._calibrate_factor_demands()

        logger.info("Step 2: Calibrating value added...")
        self._calibrate_value_added()

        logger.info("Step 3: Calibrating intermediate consumption...")
        self._calibrate_intermediate_consumption()

        logger.info("Step 4: Calibrating production taxes...")
        self._calibrate_production_taxes()

        logger.info("Step 5: Calibrating output and technical coefficients...")
        self._calibrate_output()

        logger.info("Step 6: Calibrating CES parameters...")
        self._calibrate_ces_parameters()

        logger.info("Production block calibration complete!")
        return self.result

    def _calibrate_factor_demands(self) -> None:
        """Calibrate factor demands and prices (lines 538-553)."""
        K = self.sets["K"]
        L = self.sets["L"]
        J = self.sets["J"]

        logger.debug("Reading KDO and LDO from SAM...")

        # Read raw factor demands from SAM
        # KDO(k,j) = SAM('K',k,'J',j)
        for k in K:
            k_upper = k.upper()
            for j in J:
                j_upper = j.upper()
                kdo_raw = self._get_sam_value("K", k_upper, "J", j_upper)
                self.result.KDO[(k, j)] = kdo_raw

        # LDO(l,j) = SAM('L',l,'J',j)
        for l in L:
            l_upper = l.upper()
            for j in J:
                j_upper = j.upper()
                ldo_raw = self._get_sam_value("L", l_upper, "J", j_upper)
                self.result.LDO[(l, j)] = ldo_raw

        # Base prices (numeraire = 1.0)
        WO = {l: 1.0 for l in L}
        RO = {(k, j): 1.0 for k in K for j in J}

        logger.debug("Calculating labor taxes and wages...")

        # Calculate labor taxes and wages
        # TIWO(l,j) = SAM('AG', l, 'J', j)
        for l in L:
            l_upper = l.upper()
            for j in J:
                j_upper = j.upper()
                tiwo = self._get_sam_value("AG", l_upper, "J", j_upper)
                ldo = self.result.LDO.get((l, j), 0)

                # ttiwO(l,j) = TIWO(l,j)/LDO(l,j)
                if ldo != 0:
                    self.result.ttiwO[(l, j)] = tiwo / ldo
                else:
                    self.result.ttiwO[(l, j)] = 0.0

                # WTIO(l,j) = WO(l)*(1+ttiwO(l,j))
                self.result.WTIO[(l, j)] = WO[l] * (1 + self.result.ttiwO[(l, j)])

        # Calculate capital taxes and rental rates
        # TIKO(k,j) = SAM('AG', k, 'J', j)
        for k in K:
            k_upper = k.upper()
            for j in J:
                j_upper = j.upper()
                tiko = self._get_sam_value("AG", k_upper, "J", j_upper)
                kdo = self.result.KDO.get((k, j), 0)

                # ttikO(k,j) = TIKO(k,j)/KDO(k,j)
                if kdo != 0:
                    self.result.ttikO[(k, j)] = tiko / kdo
                else:
                    self.result.ttikO[(k, j)] = 0.0

                # RTIO(k,j) = RO(k,j)*(1+ttikO(k,j))
                self.result.RTIO[(k, j)] = RO[(k, j)] * (1 + self.result.ttikO[(k, j)])

        # Normalize LDO by base wage
        for l in L:
            for j in J:
                ldo_raw = self.result.LDO.get((l, j), 0)
                if WO[l] != 0:
                    self.result.LDO[(l, j)] = ldo_raw / WO[l]

        # Calculate aggregates
        for j in J:
            # LDCO(j) = aggregate labor demand
            self.result.LDCO[j] = sum(self.result.LDO.get((l, j), 0) for l in L)

            # KDCO(j) = aggregate capital demand
            self.result.KDCO[j] = sum(self.result.KDO.get((k, j), 0) for k in K)

            # WCO(j) = composite wage
            if self.result.LDCO[j] != 0:
                self.result.WCO[j] = (
                    sum(
                        self.result.WTIO.get((l, j), 0) * self.result.LDO.get((l, j), 0)
                        for l in L
                    )
                    / self.result.LDCO[j]
                )

            # RCO(j) = composite rental rate
            if self.result.KDCO[j] != 0:
                self.result.RCO[j] = (
                    sum(
                        self.result.RTIO.get((k, j), 0) * self.result.KDO.get((k, j), 0)
                        for k in K
                    )
                    / self.result.KDCO[j]
                )

        # Calculate factor supplies
        for l in L:
            self.result.LSO[l] = sum(self.result.LDO.get((l, j), 0) for j in J)

        for k in K:
            self.result.KSO[k] = sum(self.result.KDO.get((k, j), 0) for j in J)

    def _calibrate_value_added(self) -> None:
        """Calibrate value added (lines 554-555).

        GAMS formulas:
            VAO(j) = LDCO(j) + KDCO(j)
            PVAO(j) = [WCO(j)*LDCO(j) + RCO(j)*KDCO(j)] / VAO(j)
        """
        J = self.sets["J"]

        for j in J:
            ldco = self.result.LDCO.get(j, 0)
            kdco = self.result.KDCO.get(j, 0)

            # VAO(j) = value added
            vao = ldco + kdco
            self.result.VAO[j] = vao

            # PVAO(j) = price of value added
            if vao != 0:
                wco = self.result.WCO.get(j, 0)
                rco = self.result.RCO.get(j, 0)
                pvao = (wco * ldco + rco * kdco) / vao
                self.result.PVAO[j] = pvao

    def _calibrate_production_taxes(self) -> None:
        """Calibrate production taxes (lines 557-559).

        GAMS formulas:
            ttipO(j) = TIPO(j) / {PVAO(j)*VAO(j) + Sum[i, PCO(i)*DIO(i,j)]}
            PPO(j) = PTO(j) / (1 + ttipO(j))

        Note: TIPO(j) = SAM('AG', 'GVT', 'J', j) - production tax
        """
        J = self.sets["J"]
        I = self.sets["I"]

        # Match GAMS denominator with benchmark PCO calibrated from SAM.
        PCO = self._compute_pco_from_sam()

        for j in J:
            j_upper = j.upper()

            # TIPO(j) = production tax from SAM
            tipo = self._get_sam_value("AG", "GVT", "J", j_upper)

            # Calculate denominator: PVAO*VAO + Sum[i, PCO*DIO]
            pvao = self.result.PVAO.get(j, 0)
            vao = self.result.VAO.get(j, 0)
            intermediate = sum(PCO[i] * self.result.DIO.get((i, j), 0) for i in I)
            denominator = pvao * vao + intermediate

            # ttipO(j) = production tax rate
            if denominator != 0:
                self.result.ttipO[j] = tipo / denominator

            # PTO(j) = output price (initialize to 1.0 as numeraire)
            pto = 1.0
            self.result.PTO[j] = pto

            # PPO(j) = producer price
            ttip = self.result.ttipO.get(j, 0)
            if (1 + ttip) != 0:
                self.result.PPO[j] = pto / (1 + ttip)

    def _calibrate_intermediate_consumption(self) -> None:
        """Calibrate intermediate consumption (lines 520-524).

        GAMS formulas:
            DIO(i,j) = SAM('I',i,'J',j)
            DIO(i,j) = DIO(i,j) / PCO(i)
            CIO(j) = Sum[i, DIO(i,j)]
            DITO(i) = Sum[j, DIO(i,j)]
            PCIO(j) = Sum[i, PCO(i)*DIO(i,j)] / CIO(j)
        """
        I = self.sets["I"]
        J = self.sets["J"]

        # DIO is calibrated in quantity units using benchmark PCO(i).
        PCO = self._compute_pco_from_sam()

        # Read intermediate consumption from SAM
        for i in I:
            i_upper = i.upper()
            for j in J:
                j_upper = j.upper()
                dio_raw = self._get_sam_value("I", i_upper, "J", j_upper)

                # DIO(i,j) = DIO(i,j) / PCO(i)
                if PCO[i] != 0:
                    self.result.DIO[(i, j)] = dio_raw / PCO[i]

        # Calculate total intermediate consumption by sector
        for j in J:
            self.result.CIO[j] = sum(self.result.DIO.get((i, j), 0) for i in I)

            # PCIO(j) = Sum[i, PCO(i)*DIO(i,j)] / CIO(j)
            if self.result.CIO[j] != 0:
                self.result.PCIO[j] = (
                    sum(PCO[i] * self.result.DIO.get((i, j), 0) for i in I)
                    / self.result.CIO[j]
                )

        # Calculate total intermediate demand by commodity
        for i in I:
            self.result.DITO[i] = sum(self.result.DIO.get((i, j), 0) for j in J)

    def _calibrate_output(self) -> None:
        """Calibrate output and technical coefficients (lines 600-601).

        GAMS formulas:
            XSTO(j) = SUM[i, XSO(j,i)]
            io(j) = CIO(j) / XSTO(j)
            v(j) = VAO(j) / XSTO(j)
            aij(i,j) = DIO(i,j) / CIO(j)
            GDP_BPO = Sum[j, PVAO(j)*VAO(j)] + TIPTO
        """
        I = self.sets["I"]
        J = self.sets["J"]
        xsto_from_sam = self._compute_xsto_from_sam()

        # XSTO(j) = total output
        for j in J:
            xsto = xsto_from_sam.get(j, 0.0)
            self.result.XSTO[j] = xsto

            # Technical coefficients
            if xsto != 0:
                self.result.io[j] = self.result.CIO.get(j, 0) / xsto
                self.result.v[j] = self.result.VAO.get(j, 0) / xsto

            # Input shares aij(i,j)
            if self.result.CIO[j] != 0:
                for i in I:
                    self.result.aij[(i, j)] = (
                        self.result.DIO.get((i, j), 0) / self.result.CIO[j]
                    )

        # GDP at basic prices
        self.result.GDP_BPO = (
            sum(self.result.PVAO.get(j, 0) * self.result.VAO.get(j, 0) for j in J)
            + self.income.TIPTO
        )

    def _calibrate_ces_parameters(self) -> None:
        """Calibrate CES parameters (lines 638-650+).

        GAMS formulas:
            rho_KD(j) = (1-sigma_KD(j))/sigma_KD(j)
            beta_KD(k,j) = RTIO(k,j)*KDO(k,j)**(rho_KD(j)+1) / Sum[kj,RTIO(kj,j)*KDO(kj,j)**(rho_KD(j)+1)]
            B_KD(j) = KDCO(j) / Sum[k,beta_KD(k,j)*KDO(k,j)**(-rho_KD(j))]**(-1/rho_KD(j))

            rho_LD(j) = (1-sigma_LD(j))/sigma_LD(j)
            beta_LD(l,j) = WTIO(l,j)*LDO(l,j)**(rho_LD(j)+1) / Sum[lj,WTIO(lj,j)*LDO(lj,j)**(rho_LD(j)+1)]
            B_LD(j) = LDCO(j) / Sum[l,beta_LD(l,j)*LDO(l,j)**(-rho_LD(j))]**(-1/rho_LD(j))
        """
        K = self.sets["K"]
        L = self.sets["L"]
        J = self.sets["J"]

        # Read elasticity parameters from VAL_PAR
        sigma_KD = self.val_par.get("sigma_KD", {j: 0.8 for j in J})
        sigma_LD = self.val_par.get("sigma_LD", {j: 0.8 for j in J})

        # Calculate rho parameters
        for j in J:
            j_upper = j.upper()

            # rho_KD(j) = (1-sigma_KD(j))/sigma_KD(j)
            sigma_kd = sigma_KD.get(j_upper, 0.8)
            if sigma_kd != 0:
                self.result.rho_KD[j] = (1 - sigma_kd) / sigma_kd

            # rho_LD(j) = (1-sigma_LD(j))/sigma_LD(j)
            sigma_ld = sigma_LD.get(j_upper, 0.8)
            if sigma_ld != 0:
                self.result.rho_LD[j] = (1 - sigma_ld) / sigma_ld

        # Calculate beta_KD(k,j) - CES share parameters for capital
        for j in J:
            kdco = self.result.KDCO.get(j, 0)
            if kdco != 0:
                rho = self.result.rho_KD.get(j, 0)

                # Calculate denominator: Sum[kj,RTIO(kj,j)*KDO(kj,j)**(rho+1)]
                denominator = sum(
                    self.result.RTIO.get((kj, j), 0)
                    * (self.result.KDO.get((kj, j), 0) ** (rho + 1))
                    for kj in K
                )

                for k in K:
                    kdo = self.result.KDO.get((k, j), 0)
                    rtio = self.result.RTIO.get((k, j), 0)

                    # beta_KD(k,j) = RTIO(k,j)*KDO(k,j)**(rho+1) / denominator
                    if denominator != 0:
                        numerator = rtio * (kdo ** (rho + 1))
                        self.result.beta_KD[(k, j)] = numerator / denominator

                # B_KD(j) = KDCO(j) / Sum[k,beta_KD(k,j)*KDO(k,j)**(-rho)]**(-1/rho)
                if rho != 0:
                    sum_term = sum(
                        self.result.beta_KD.get((k, j), 0)
                        * (self.result.KDO.get((k, j), 0) ** (-rho))
                        for k in K
                        if self.result.KDO.get((k, j), 0) != 0
                    )
                    if sum_term != 0:
                        self.result.B_KD[j] = kdco / (sum_term ** (-1 / rho))

        # Calculate beta_LD(l,j) - CES share parameters for labor
        for j in J:
            ldco = self.result.LDCO.get(j, 0)
            if ldco != 0:
                rho = self.result.rho_LD.get(j, 0)

                # Calculate denominator: Sum[lj,WTIO(lj,j)*LDO(lj,j)**(rho+1)]
                denominator = sum(
                    self.result.WTIO.get((lj, j), 0)
                    * (self.result.LDO.get((lj, j), 0) ** (rho + 1))
                    for lj in L
                )

                for l in L:
                    ldo = self.result.LDO.get((l, j), 0)
                    wtio = self.result.WTIO.get((l, j), 0)

                    # beta_LD(l,j) = WTIO(l,j)*LDO(l,j)**(rho+1) / denominator
                    if denominator != 0:
                        numerator = wtio * (ldo ** (rho + 1))
                        self.result.beta_LD[(l, j)] = numerator / denominator

                # B_LD(j) = LDCO(j) / Sum[l,beta_LD(l,j)*LDO(l,j)**(-rho)]**(-1/rho)
                if rho != 0:
                    sum_term = sum(
                        self.result.beta_LD.get((l, j), 0)
                        * (self.result.LDO.get((l, j), 0) ** (-rho))
                        for l in L
                        if self.result.LDO.get((l, j), 0) != 0
                    )
                    if sum_term != 0:
                        self.result.B_LD[j] = ldco / (sum_term ** (-1 / rho))

        # Calculate VA (value added) CES parameters (lines 656-666)
        # rho_VA(j), beta_VA(j), B_VA(j)
        sigma_VA = self.val_par.get("sigma_VA", {j: 1.5 for j in J})

        for j in J:
            ldc = self.result.LDCO.get(j, 0)
            kdc = self.result.KDCO.get(j, 0)
            va = self.result.VAO.get(j, 0)

            # rho_VA(j) depends on whether both LDC and KDC exist
            if ldc != 0 and kdc != 0:
                # Both labor and capital exist
                sigma_va = sigma_VA.get(j.upper(), 1.5)
                if sigma_va != 0:
                    self.result.rho_VA[j] = (1 - sigma_va) / sigma_va
            else:
                # Only one factor exists
                self.result.rho_VA[j] = -1.0

            rho = self.result.rho_VA.get(j, -1.0)

            # beta_VA(j) - only calculate if both factors exist
            if ldc != 0 and kdc != 0 and rho != -1.0:
                wco = self.result.WCO.get(j, 1.0)
                rco = self.result.RCO.get(j, 1.0)

                # beta_VA(j) = WCO(j)*LDCO(j)**(rho+1) / [WCO(j)*LDCO(j)**(rho+1) + RCO(j)*KDCO(j)**(rho+1)]
                numerator = wco * (ldc ** (rho + 1))
                denominator = numerator + rco * (kdc ** (rho + 1))

                if denominator != 0:
                    self.result.beta_VA[j] = numerator / denominator

                # B_VA(j) = VAO(j) / [beta_VA(j)*LDCO(j)**(-rho) + (1-beta_VA(j))*KDCO(j)**(-rho)]**(-1/rho)
                beta = self.result.beta_VA.get(j, 0.5)
                sum_term = beta * (ldc ** (-rho)) + (1 - beta) * (kdc ** (-rho))

                if sum_term != 0 and rho != 0:
                    self.result.B_VA[j] = va / (sum_term ** (-1 / rho))
