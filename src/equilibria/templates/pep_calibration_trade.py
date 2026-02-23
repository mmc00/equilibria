"""
PEP Trade Block Calibration (GAMS Section 4.4, 4.6.2, 4.6.3)

This module implements the calibration of trade variables following
the exact order and formulas from GAMS PEP-1-1_v2_1_modular.gms
lines 481-521 and 598-635.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TradeCalibrationResult(BaseModel):
    """Container for all calibrated trade variables."""

    # Import demand (lines 335, 488-490, 499-503, 519)
    IMO: dict[str, float] = Field(
        default_factory=dict, description="Import quantity by commodity i (IMO(i))"
    )
    DDO: dict[str, float] = Field(
        default_factory=dict,
        description="Domestic demand for local product i (DDO(i))",
    )
    QO: dict[str, float] = Field(
        default_factory=dict,
        description="Composite commodity demand (QO(i) = [PMO(i)*IMO(i)+PDO(i)*DDO(i)]/PCO(i))",
    )

    # Export supply (lines 335-337, 504-511)
    EXDO: dict[str, float] = Field(
        default_factory=dict,
        description="World demand for exports of product i (EXDO(i))",
    )
    EXO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Export by sector j of commodity i (EXO(j,i))",
    )

    # Domestic supply (lines 332-333, 513-514)
    DSO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Domestic supply of commodity i by sector j (DSO(j,i))",
    )
    XSO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Total output supply by sector j of commodity i (XSO(j,i) = DSO(j,i)+EXO(j,i))",
    )

    # Prices (lines 497-498, 501-502, 509-510, 515)
    PCO: dict[str, float] = Field(
        default_factory=dict,
        description="Purchaser price of composite commodity i (PCO(i))",
    )
    PLO: dict[str, float] = Field(
        default_factory=dict,
        description="Price of local product i excluding taxes (PLO(i))",
    )
    PMO: dict[str, float] = Field(
        default_factory=dict,
        description="Price of imported product i (PMO(i))",
    )
    PDO: dict[str, float] = Field(
        default_factory=dict,
        description="Price of local product i sold domestically (PDO(i))",
    )
    PEO: dict[str, float] = Field(
        default_factory=dict,
        description="Price received for exported commodity i (PEO(i))",
    )
    PE_FOBO: dict[str, float] = Field(
        default_factory=dict,
        description="FOB price of exported commodity i (PE_FOBO(i))",
    )
    PWXO: dict[str, float] = Field(
        default_factory=dict,
        description="World price of exported product i (PWXO(i))",
    )
    PWMO: dict[str, float] = Field(
        default_factory=dict,
        description="World price of imported product i (PWMO(i))",
    )
    PO: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Basic price of sector j's production of commodity i (PO(j,i))",
    )

    # Tax rates (lines 493-495, 499, 507-508)
    tticO: dict[str, float] = Field(
        default_factory=dict,
        description="Commodity tax rate on composite good i (tticO(i))",
    )
    ttimO: dict[str, float] = Field(
        default_factory=dict,
        description="Import tax rate (ttimO(i) = TIMO(i)/[eO*PWMO(i)*IMO(i)])",
    )
    ttixO: dict[str, float] = Field(
        default_factory=dict,
        description="Export tax rate (ttixO(i) = TIXO(i)/[EXDO(i)-TIXO(i)])",
    )

    # Trade and transport margins (lines 483-486, 491, 505-506)
    tmrg: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Trade and transport margins (tmrg(i,ij))",
    )
    tmrg_X: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Export margins (tmrg_X(i,ij))",
    )

    # Taxes from SAM
    TICO: dict[str, float] = Field(
        default_factory=dict, description="Indirect commodity taxes (TICO(i))"
    )
    TIMO: dict[str, float] = Field(
        default_factory=dict, description="Import taxes (TIMO(i))"
    )
    TIXO: dict[str, float] = Field(
        default_factory=dict, description="Export taxes (TIXO(i))"
    )

    # Exchange rate
    eO: float = Field(default=1.0, description="Exchange rate (eO)")

    # CET parameters for trade (lines 598-621)
    rho_XT: dict[str, float] = Field(
        default_factory=dict,
        description="CET elasticity parameter between commodities (rho_XT(j) = (1+sigma_XT(j))/sigma_XT(j))",
    )
    beta_XT: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="CET share parameter between commodities (beta_XT(j,i))",
    )
    B_XT: dict[str, float] = Field(
        default_factory=dict,
        description="CET scale parameter between commodities (B_XT(j))",
    )

    rho_X: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="CET elasticity parameter between exports and local (rho_X(j,i))",
    )
    beta_X: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="CET share parameter between exports and local (beta_X(j,i))",
    )
    B_X: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="CET scale parameter between exports and local (B_X(j,i))",
    )

    # CES parameters for imports (lines 625-635)
    rho_M: dict[str, float] = Field(
        default_factory=dict,
        description="CES elasticity parameter for composite good (rho_M(i) = (1-sigma_M(i))/sigma_M(i))",
    )
    beta_M: dict[str, float] = Field(
        default_factory=dict,
        description="CES share parameter for composite good (beta_M(i))",
    )
    B_M: dict[str, float] = Field(
        default_factory=dict,
        description="CES scale parameter for composite good (B_M(i))",
    )

    # Trade demand for margins
    MRGNO: dict[str, float] = Field(
        default_factory=dict,
        description="Demand for commodity i as trade or transport margin (MRGNO(i))",
    )


class TradeCalibrator:
    """Calibrates trade block variables following GAMS Section 4.4 and 4.6."""

    def __init__(
        self,
        sam_data: dict[str, Any],
        production_result: Any,
        val_par_data: dict[str, Any] | None = None,
        sets: dict[str, list[str]] | None = None,
    ):
        """Initialize the trade calibrator.

        Args:
            sam_data: Dictionary with SAM data from read_gdx()
            production_result: ProductionCalibrationResult with production variables
            val_par_data: Optional dictionary with VAL_PAR parameters
            sets: Dictionary with set definitions
        """
        self.sam = sam_data
        self.production = production_result
        self.sets = sets or self._detect_sets()
        self.result = TradeCalibrationResult()

        # Load VAL_PAR parameters
        if val_par_data is None:
            val_par_path = (
                Path(__file__).resolve().parent
                / "reference"
                / "pep2"
                / "data"
                / "VAL_PAR.xlsx"
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
            "I": ["agr", "food", "othind", "ser", "adm"],  # Added 'adm'
            "AG": ["hrp", "hup", "hrr", "hur", "firm", "gvt", "row"],
            "AGNG": ["hrp", "hup", "hrr", "hur", "firm", "row"],
        }

    def _read_val_par_excel(self, filepath: Path) -> dict[str, Any]:
        """Read VAL_PAR.xlsx and extract trade-related parameters."""
        import pandas as pd

        logger.info(f"Reading VAL_PAR for trade parameters from: {filepath}")

        try:
            df = pd.read_excel(filepath, sheet_name="PAR", header=None)
        except Exception as e:
            logger.warning(f"Could not read VAL_PAR Excel: {e}. Using defaults.")
            return self._get_default_val_par()

        params = {
            "sigma_XT": {},
            "sigma_X": {},
            "sigma_M": {},
            "sigma_XD": {},
        }

        # Extract sectors (j) for sigma_XT - rows 5-8 (0-indexed: 4-7)
        sectors = ["AGR", "IND", "SER", "ADM"]
        for i, sector in enumerate(sectors):
            row_idx = 5 + i
            try:
                params["sigma_XT"][sector] = (
                    float(df.iloc[row_idx, 4]) if pd.notna(df.iloc[row_idx, 4]) else 2.0
                )
            except (IndexError, ValueError):
                params["sigma_XT"][sector] = 2.0

        # Extract commodities (i) for sigma_M and sigma_XD - rows 12-16 (0-indexed: 11-15)
        commodities = ["AGR", "FOOD", "OTHIND", "SER", "ADM"]
        for i, comm in enumerate(commodities):
            row_idx = 12 + i
            try:
                params["sigma_M"][comm] = (
                    float(df.iloc[row_idx, 1]) if pd.notna(df.iloc[row_idx, 1]) else 2.0
                )
                params["sigma_XD"][comm] = (
                    float(df.iloc[row_idx, 3]) if pd.notna(df.iloc[row_idx, 3]) else 2.0
                )
            except (IndexError, ValueError):
                params["sigma_M"][comm] = 2.0
                params["sigma_XD"][comm] = 2.0

        # sigma_X(j,i) from PARJI sheet
        try:
            df_ji = pd.read_excel(filepath, sheet_name="PARJI", header=None)
            for i, sector in enumerate(sectors):
                for j, comm in enumerate(commodities):
                    try:
                        val = df_ji.iloc[i + 1, j + 1]
                        if pd.notna(val):
                            params["sigma_X"][(sector.lower(), comm.lower())] = float(val)
                        else:
                            params["sigma_X"][(sector.lower(), comm.lower())] = 2.0
                    except (IndexError, ValueError):
                        params["sigma_X"][(sector.lower(), comm.lower())] = 2.0
        except Exception as e:
            logger.warning(f"Could not read PARJI sheet: {e}")
            for sector in sectors:
                for comm in commodities:
                    params["sigma_X"][(sector.lower(), comm.lower())] = 2.0

        logger.info("Successfully read VAL_PAR trade parameters")
        return params

    def _get_default_val_par(self) -> dict[str, Any]:
        """Return default VAL_PAR parameters for trade."""
        sectors = ["AGR", "IND", "SER", "ADM"]
        commodities = ["AGR", "FOOD", "OTHIND", "SER", "ADM"]

        return {
            "sigma_XT": {s: 2.0 for s in sectors},
            "sigma_X": {(j.lower(), i.lower()): 2.0 for j in sectors for i in commodities},
            "sigma_M": {c: 2.0 for c in commodities},
            "sigma_XD": {c: 2.0 for c in commodities},
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

    def calibrate(self) -> TradeCalibrationResult:
        """Run full trade calibration."""
        logger.info("Starting trade block calibration (Phase 4)")

        logger.info("Step 1: Reading trade data from SAM...")
        self._read_sam_trade_data()

        logger.info("Step 2: Calibrating domestic and import demand...")
        self._calibrate_import_demand()

        logger.info("Step 3: Calibrating export supply...")
        self._calibrate_export_supply()

        logger.info("Step 4: Calibrating prices and taxes...")
        self._calibrate_prices_and_taxes()

        logger.info("Step 5: Calibrating trade margins...")
        self._calibrate_margins()

        logger.info("Step 6: Finalizing commodity prices and taxes...")
        self._finalize_commodity_prices_and_taxes()

        logger.info("Step 7: Calculating XSO (total output by sector-commodity)...")
        # XSO must be calculated BEFORE CET parameters
        for j in self.sets["J"]:
            for i in self.sets["I"]:
                dso = self.result.DSO.get((j, i), 0)
                exo = self.result.EXO.get((j, i), 0)
                self.result.XSO[(j, i)] = dso + exo

        logger.info("Step 8: Calculating composite demand and output...")
        self._calculate_composite_demand()

        logger.info("Step 9: Calibrating CET parameters for exports...")
        self._calibrate_cet_parameters()

        logger.info("Step 10: Calibrating CES parameters for imports...")
        self._calibrate_ces_parameters()

        logger.info("Trade block calibration complete!")
        return self.result

    def _read_sam_trade_data(self) -> None:
        """Read raw trade data from SAM."""
        I = self.sets["I"]
        J = self.sets["J"]

        # IMO(i) = SAM('Agents','row','I',i) - Import quantity
        for i in I:
            i_upper = i.upper()
            imo = self._get_sam_value("AG", "ROW", "I", i_upper)
            self.result.IMO[i] = imo

        # EXDO(i) = SAM('X',i,'Agents','row') - Export demand (note: uses 'X' category not 'I')
        for i in I:
            i_upper = i.upper()
            exdo = self._get_sam_value("X", i_upper, "AG", "ROW")
            self.result.EXDO[i] = exdo

        # DSO(j,i) = SAM('J',j,'I',i) - Domestic supply
        for j in J:
            j_upper = j.upper()
            for i in I:
                i_upper = i.upper()
                dso = self._get_sam_value("J", j_upper, "I", i_upper)
                self.result.DSO[(j, i)] = dso

        # EXO(j,i) = SAM('J',j,'X',i) - Exports by sector and commodity
        for j in J:
            j_upper = j.upper()
            for i in I:
                i_upper = i.upper()
                exo = self._get_sam_value("J", j_upper, "X", i_upper)
                self.result.EXO[(j, i)] = exo

        # TICO(i) = SAM('Agents','ti','I',i) - Commodity taxes
        for i in I:
            i_upper = i.upper()
            tico = self._get_sam_value("AG", "TI", "I", i_upper)
            self.result.TICO[i] = tico

        # TIMO(i) = SAM('Agents','tm','I',i) - Import taxes
        for i in I:
            i_upper = i.upper()
            timo = self._get_sam_value("AG", "TM", "I", i_upper)
            self.result.TIMO[i] = timo

        # TIXO(i) = SAM('Agents','gvt','X',i) - Export taxes
        for i in I:
            i_upper = i.upper()
            tixo = self._get_sam_value("AG", "GVT", "X", i_upper)
            self.result.TIXO[i] = tixo

        # tmrg(i,ij) = SAM('I',i,'I',ij) - Trade margins
        for i in I:
            i_upper = i.upper()
            for ij in I:
                ij_upper = ij.upper()
                tmrg = self._get_sam_value("I", i_upper, "I", ij_upper)
                if tmrg != 0:
                    self.result.tmrg[(i, ij)] = tmrg

        # tmrg_X(i,ij) = SAM('I',i,'X',ij) - Export margins
        for i in I:
            i_upper = i.upper()
            for ij in I:
                ij_upper = ij.upper()
                tmrg_x = self._get_sam_value("I", i_upper, "X", ij_upper)
                if tmrg_x != 0:
                    self.result.tmrg_X[(i, ij)] = tmrg_x

        logger.info(f"Read {len(self.result.IMO)} imports, {len(self.result.EXDO)} exports")

    def _calibrate_import_demand(self) -> None:
        """Calibrate import and domestic demand."""
        I = self.sets["I"]

        # DDO(i) = SUM[j,DSO(j,i)] - Domestic demand
        for i in I:
            ddo = sum(self.result.DSO.get((j, i), 0) for j in self.sets["J"])
            self.result.DDO[i] = ddo

        # Base prices (numeraire = 1.0)
        self.result.PLO = {i: 1.0 for i in I}
        self.result.PWMO = {i: 1.0 for i in I}

    def _calibrate_export_supply(self) -> None:
        """Calibrate export supply."""
        I = self.sets["I"]

        # Base export price
        self.result.PEO = {i: 1.0 for i in I}

    def _calibrate_prices_and_taxes(self) -> None:
        """Calibrate preliminary prices and taxes (PCO and ttimO)."""
        I = self.sets["I"]
        eO = self.result.eO

        # PCO(i) calculation (line 483)
        for i in I:
            ddo = self.result.DDO.get(i, 0)
            imo = self.result.IMO.get(i, 0)
            tico = self.result.TICO.get(i, 0)
            timo = self.result.TIMO.get(i, 0)

            # Sum of margins
            margin_sum = sum(self.result.tmrg.get((ij, i), 0) for ij in I)

            numerator = ddo + imo + margin_sum + tico + timo
            denominator = ddo + imo if (ddo + imo) != 0 else 1

            self.result.PCO[i] = numerator / denominator

        # ttimO(i) = TIMO(i)/[eO*PWMO(i)*IMO(i)] (line 499)
        for i in I:
            timo = self.result.TIMO.get(i, 0)
            pwm = self.result.PWMO.get(i, 1.0)
            imo = self.result.IMO.get(i, 0)

            if imo != 0 and eO * pwm != 0:
                self.result.ttimO[i] = timo / (eO * pwm * imo)

    def _finalize_commodity_prices_and_taxes(self) -> None:
        """Calibrate tticO, PDO, and PMO after margin normalization."""
        I = self.sets["I"]
        eO = self.result.eO

        # PMO(i) = {(1+ttimO(i))*eO*PWMO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)]}*(1+tticO(i)) (line 501-502)
        # PDO(i) = {PLO(i)+SUM[ij,PCO(ij)*tmrg(ij,i)]}*(1+tticO(i)) (line 497-498)
        # tticO(i) calculation (line 493-495)
        for i in I:
            plo = self.result.PLO.get(i, 1.0)
            ddo = self.result.DDO.get(i, 0)
            imo = self.result.IMO.get(i, 0)
            tico = self.result.TICO.get(i, 0)
            timo = self.result.TIMO.get(i, 0)

            # Margin term is SUM[ij,PCO(ij)*tmrg(ij,i)] in GAMS
            margin_sum = sum(
                self.result.PCO.get(ij, 1.0) * self.result.tmrg.get((ij, i), 0.0)
                for ij in I
            )

            # tticO(i)
            denominator = (
                (plo + margin_sum) * ddo
                + (eO * self.result.PWMO.get(i, 1.0) + margin_sum) * imo
                + timo
            )
            if denominator != 0:
                self.result.tticO[i] = tico / denominator

            # PDO(i)
            self.result.PDO[i] = (plo + margin_sum) * (1 + self.result.tticO.get(i, 0))

            # PMO(i)
            pwm = self.result.PWMO.get(i, 1.0)
            ttim = self.result.ttimO.get(i, 0)
            self.result.PMO[i] = ((1 + ttim) * eO * pwm + margin_sum) * (1 + self.result.tticO.get(i, 0))

    def _calibrate_margins(self) -> None:
        """Calibrate trade margins."""
        I = self.sets["I"]

        # Normalize margins (lines 485-486, 491)
        for i in I:
            for ij in I:
                tmrg = self.result.tmrg.get((i, ij), 0)
                pc = self.result.PCO.get(i, 1.0)
                if pc != 0:
                    self.result.tmrg[(i, ij)] = tmrg / pc

                # tmrg(i,ij) = tmrg(i,ij)/{DDO(ij)+IMO(ij)} (line 491)
                ddo_ij = self.result.DDO.get(ij, 0)
                imo_ij = self.result.IMO.get(ij, 0)
                denom = ddo_ij + imo_ij
                if denom != 0:
                    self.result.tmrg[(i, ij)] = self.result.tmrg.get((i, ij), 0) / denom

        # Export margins
        for i in I:
            for ij in I:
                pc = self.result.PCO.get(i, 1.0)
                if pc != 0:
                    self.result.tmrg_X[(i, ij)] = self.result.tmrg_X.get((i, ij), 0) / pc

                exdo = self.result.EXDO.get(ij, 0)
                if exdo != 0:
                    # tmrg_X(ij,i)$EXDO(i) = tmrg_X(ij,i)/SUM[j,EXO(j,i)] (line 505-506)
                    sum_exo = sum(self.result.EXO.get((j, ij), 0) for j in self.sets["J"])
                    if sum_exo != 0:
                        self.result.tmrg_X[(i, ij)] = self.result.tmrg_X.get((i, ij), 0) / sum_exo

    def _calibrate_cet_parameters(self) -> None:
        """Calibrate CET parameters for trade (Section 4.6.2)."""
        I = self.sets["I"]
        J = self.sets["J"]

        sigma_XT = self.val_par.get("sigma_XT", {j: 2.0 for j in J})
        sigma_X = self.val_par.get("sigma_X", {})

        # rho_XT(j) = (1+sigma_XT(j))/sigma_XT(j) (line 600)
        for j in J:
            j_upper = j.upper()
            sigma = sigma_XT.get(j_upper, 2.0)
            if sigma != 0:
                self.result.rho_XT[j] = (1 + sigma) / sigma

        # beta_XT(j,i) and B_XT(j) (lines 601-606)
        for j in J:
            rho = self.result.rho_XT.get(j, 0)

            # Calculate denominator for beta_XT
            denom = sum(
                self.result.PO.get((j, i), 0)
                * (self.result.XSO.get((j, i), 0) ** (1 - rho))
                for i in I
                if self.result.XSO.get((j, i), 0) != 0
            )

            for i in I:
                xso = self.result.XSO.get((j, i), 0)
                po = self.result.PO.get((j, i), 0)

                if xso != 0 and denom != 0:
                    self.result.beta_XT[(j, i)] = (
                        po * (xso ** (1 - rho)) / denom
                    )

            # B_XT(j)
            xsto = sum(
                self.result.XSO.get((j, i), 0.0)
                for i in I
            )
            if xsto != 0 and rho != 0:
                sum_term = sum(
                    self.result.beta_XT.get((j, i), 0) * (self.result.XSO.get((j, i), 0) ** rho)
                    for i in I
                    if self.result.XSO.get((j, i), 0) != 0
                )
                if sum_term != 0:
                    self.result.B_XT[j] = xsto / (sum_term ** (1 / rho))

        # rho_X(j,i), beta_X(j,i), B_X(j,i) (lines 609-621)
        for j in J:
            for i in I:
                exo = self.result.EXO.get((j, i), 0)
                dso = self.result.DSO.get((j, i), 0)

                if exo != 0 and dso != 0:
                    sigma = sigma_X.get((j.upper(), i.upper()), 2.0)
                    if sigma != 0:
                        self.result.rho_X[(j, i)] = (1 + sigma) / sigma
                else:
                    self.result.rho_X[(j, i)] = 1.0

                rho = self.result.rho_X.get((j, i), 1.0)

                # beta_X(j,i)
                xso = self.result.XSO.get((j, i), 0)
                if xso != 0:
                    peo = self.result.PEO.get(i, 0)
                    plo = self.result.PLO.get(i, 0)

                    denom = peo * (exo ** (1 - rho)) + plo * (dso ** (1 - rho))
                    if denom != 0:
                        self.result.beta_X[(j, i)] = peo * (exo ** (1 - rho)) / denom

                # B_X(j,i)
                if xso != 0 and rho != 0:
                    beta = self.result.beta_X.get((j, i), 0.5)
                    term = beta * (exo ** rho) + (1 - beta) * (dso ** rho)
                    if term != 0:
                        self.result.B_X[(j, i)] = xso / (term ** (1 / rho))

    def _calibrate_ces_parameters(self) -> None:
        """Calibrate CES parameters for imports (Section 4.6.3.1)."""
        I = self.sets["I"]

        sigma_M = self.val_par.get("sigma_M", {i: 2.0 for i in I})

        # rho_M(i) = (1-sigma_M(i))/sigma_M(i) (line 625)
        for i in I:
            i_upper = i.upper()
            sigma = sigma_M.get(i_upper, 2.0)
            if sigma != 0:
                imo = self.result.IMO.get(i, 0)
                ddo = self.result.DDO.get(i, 0)

                if imo != 0 and ddo != 0:
                    self.result.rho_M[i] = (1 - sigma) / sigma
                else:
                    self.result.rho_M[i] = -1.0

        # beta_M(i) and B_M(i) (lines 629-635)
        for i in I:
            rho = self.result.rho_M.get(i, -1.0)
            qo = self.result.QO.get(i, 0)

            if qo != 0:
                pmo = self.result.PMO.get(i, 0)
                pdo = self.result.PDO.get(i, 0)
                imo = self.result.IMO.get(i, 0)
                ddo = self.result.DDO.get(i, 0)

                # beta_M(i)
                denom = pmo * (imo ** (rho + 1)) + pdo * (ddo ** (rho + 1))
                if denom != 0:
                    self.result.beta_M[i] = pmo * (imo ** (rho + 1)) / denom

                # B_M(i)
                if rho != 0:
                    beta = self.result.beta_M.get(i, 0.5)
                    term = beta * (imo ** (-rho)) + (1 - beta) * (ddo ** (-rho))
                    if term != 0:
                        self.result.B_M[i] = qo / (term ** (-1 / rho))

    def _calculate_composite_demand(self) -> None:
        """Calculate composite demand and margins."""
        I = self.sets["I"]

        # QO(i) = [PMO(i)*IMO(i)+PDO(i)*DDO(i)]/PCO(i) (line 519)
        for i in I:
            pmo = self.result.PMO.get(i, 0)
            imo = self.result.IMO.get(i, 0)
            pdo = self.result.PDO.get(i, 0)
            ddo = self.result.DDO.get(i, 0)
            pc = self.result.PCO.get(i, 1.0)

            if pc != 0:
                self.result.QO[i] = (pmo * imo + pdo * ddo) / pc

        # XSO(j,i) = DSO(j,i)+EXO(j,i) (line 514)
        for j in self.sets["J"]:
            for i in I:
                dso = self.result.DSO.get((j, i), 0)
                exo = self.result.EXO.get((j, i), 0)
                self.result.XSO[(j, i)] = dso + exo

        # PO(j,i) = [PLO(i)*DSO(j,i)+PEO(i)*EXO(j,i)]/XSO(j,i) (line 515)
        for j in self.sets["J"]:
            for i in I:
                xso = self.result.XSO.get((j, i), 0)
                if xso != 0:
                    plo = self.result.PLO.get(i, 0)
                    dso = self.result.DSO.get((j, i), 0)
                    peo = self.result.PEO.get(i, 0)
                    exo = self.result.EXO.get((j, i), 0)
                    self.result.PO[(j, i)] = (plo * dso + peo * exo) / xso

        # ttixO(i) and PE_FOBO(i), PWXO(i) (lines 507-511)
        for i in I:
            exdo = self.result.EXDO.get(i, 0)
            if exdo != 0:
                tixo = self.result.TIXO.get(i, 0)
                self.result.ttixO[i] = tixo / (exdo - tixo) if (exdo - tixo) != 0 else 0

                # PE_FOBO(i) = (1+ttixO(i))*(PEO(i)+SUM[ij,PCO(ij)*tmrg_X(ij,i)])
                margin_sum = sum(
                    self.result.PCO.get(ij, 1.0) * self.result.tmrg_X.get((ij, i), 0)
                    for ij in I
                )
                peo = self.result.PEO.get(i, 0)
                self.result.PE_FOBO[i] = (1 + self.result.ttixO.get(i, 0)) * (peo + margin_sum)

                # PWXO(i) = PE_FOBO(i)/eO
                self.result.PWXO[i] = self.result.PE_FOBO[i] / self.result.eO

                # EXDO(i) = EXDO(i)/(PWXO(i)*eO)
                denom = self.result.PWXO[i] * self.result.eO
                if abs(denom) > 1e-12:
                    self.result.EXDO[i] = self.result.EXDO.get(i, 0.0) / denom

        # MRGNO(i) = SUM[ij,tmrg(i,ij)*DDO(ij)] + SUM[ij,tmrg(i,ij)*IMO(ij)] + SUM[(j,ij),tmrg_X(i,ij)*EXO(j,ij)]
        for i in I:
            mrgn_d = sum(self.result.tmrg.get((i, ij), 0) * self.result.DDO.get(ij, 0) for ij in I)
            mrgn_m = sum(self.result.tmrg.get((i, ij), 0) * self.result.IMO.get(ij, 0) for ij in I)
            mrgn_x = sum(self.result.tmrg_X.get((i, ij), 0) * self.result.EXO.get((j, ij), 0) for j in self.sets["J"] for ij in I)
            self.result.MRGNO[i] = mrgn_d + mrgn_m + mrgn_x

        logger.info(f"Calibrated composite demand for {len(self.result.QO)} commodities")
