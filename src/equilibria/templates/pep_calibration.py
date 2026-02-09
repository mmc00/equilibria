"""PEP calibration system for equilibria CGE framework.

This module provides calibration functionality for PEP models,
including loading elasticities from VAL_PAR files and computing
model parameters from SAM data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from equilibria.babel import SAM


class ElasticitySet(BaseModel):
    """Container for PEP elasticities.

    Attributes:
        sigma_kd: Substitution between capital and labor (by sector)
        sigma_ld: Substitution between labor types (by sector)
        sigma_va: Substitution in value-added (by sector)
        sigma_xt: CET transformation elasticity (by sector)
        sigma_m: Armington import substitution (by commodity)
        sigma_xd: Domestic-export substitution (by commodity)
        sigma_ij: Intermediate substitution (by sector, commodity)
        frisch: Frisch parameter (by household)
        les_elasticities: LES income elasticities (by commodity, household)
    """

    sigma_kd: dict[str, float] = Field(default_factory=dict)
    sigma_ld: dict[str, float] = Field(default_factory=dict)
    sigma_va: dict[str, float] = Field(default_factory=dict)
    sigma_xt: dict[str, float] = Field(default_factory=dict)
    sigma_m: dict[str, float] = Field(default_factory=dict)
    sigma_xd: dict[str, float] = Field(default_factory=dict)
    sigma_ij: dict[str, dict[str, float]] = Field(default_factory=dict)
    frisch: dict[str, float] = Field(default_factory=dict)
    les_elasticities: dict[str, dict[str, float]] = Field(default_factory=dict)

    def get_sigma_va(self, sector: str, default: float = 1.0) -> float:
        """Get value-added substitution elasticity for a sector."""
        return self.sigma_va.get(sector, default)

    def get_sigma_m(self, commodity: str, default: float = 2.0) -> float:
        """Get import substitution elasticity for a commodity."""
        return self.sigma_m.get(commodity, default)

    def get_frisch(self, household: str, default: float = -1.5) -> float:
        """Get Frisch parameter for a household."""
        return self.frisch.get(household, default)


class TaxRates(BaseModel):
    """Container for PEP tax rates.

    These are calibrated from SAM data.

    Attributes:
        ti: Indirect tax rates (by sector)
        tm: Import tariff rates (by commodity)
        tx: Export tax rates (by commodity)
        td: Direct tax rates (by household)
    """

    ti: dict[str, float] = Field(default_factory=dict)
    tm: dict[str, float] = Field(default_factory=dict)
    tx: dict[str, float] = Field(default_factory=dict)
    td: dict[str, float] = Field(default_factory=dict)


class PEPCalibrator:
    """Calibrator for PEP CGE models.

    Loads elasticities from VAL_PAR files and computes all model
    parameters from SAM data.

    Example:
        >>> calibrator = PEPCalibrator(sam, param_file="VAL_PAR.xlsx")
        >>> params = calibrator.calibrate()
        >>> model.set_parameters(params)
    """

    def __init__(
        self,
        sam: SAM | None,
        param_file: Path | str | None = None,
        sectors: list[str] | None = None,
        commodities: list[str] | None = None,
        labor_types: list[str] | None = None,
        capital_types: list[str] | None = None,
        households: list[str] | None = None,
    ):
        """Initialize the calibrator.

        Args:
            sam: Social Accounting Matrix
            param_file: Path to VAL_PAR.xlsx (uses defaults if None)
            sectors: List of sector names (auto-detected if None)
            commodities: List of commodity names (auto-detected if None)
            labor_types: List of labor types (auto-detected if None)
            capital_types: List of capital types (auto-detected if None)
            households: List of household names (auto-detected if None)
        """
        self.sam = sam
        self.param_file = Path(param_file) if param_file else None

        # Auto-detect sets if not provided
        self.sectors = sectors or self._detect_sectors()
        self.commodities = commodities or self._detect_commodities()
        self.labor_types = labor_types or self._detect_labor_types()
        self.capital_types = capital_types or self._detect_capital_types()
        self.households = households or self._detect_households()

        self.elasticities: ElasticitySet | None = None
        self.tax_rates: TaxRates | None = None

    def _detect_sectors(self) -> list[str]:
        """Detect sectors from SAM account names."""
        if self.sam is None:
            return ["agr", "othind", "food", "ser", "adm"]
        # Look for typical sector names in SAM
        accounts = set(self.sam.data.index)
        # Filter out non-sector accounts
        non_sectors = {
            "usk",
            "sk",
            "cap",
            "land",
            "hrp",
            "hup",
            "hrr",
            "hur",
            "gvt",
            "row",
            "ti",
            "tm",
            "tx",
            "td",
            "inv",
            "marg",
        }
        sectors = [
            a for a in accounts if a not in non_sectors and not a.startswith("ROW_")
        ]
        return sectors or ["agr", "othind", "food", "ser", "adm"]

    def _detect_commodities(self) -> list[str]:
        """Detect commodities from SAM (same as sectors in PEP)."""
        return self._detect_sectors()

    def _detect_labor_types(self) -> list[str]:
        """Detect labor types from SAM account names."""
        if self.sam is None:
            return ["usk", "sk"]
        accounts = set(self.sam.data.index)
        labor = [a for a in accounts if a in ["usk", "sk"]]
        return labor or ["usk", "sk"]

    def _detect_capital_types(self) -> list[str]:
        """Detect capital types from SAM account names."""
        if self.sam is None:
            return ["cap", "land"]
        accounts = set(self.sam.data.index)
        capital = [a for a in accounts if a in ["cap", "land"]]
        return capital or ["cap", "land"]

    def _detect_households(self) -> list[str]:
        """Detect households from SAM account names."""
        if self.sam is None:
            return ["hrp", "hup", "hrr", "hur"]
        accounts = set(self.sam.data.index)
        households = [a for a in accounts if a in ["hrp", "hup", "hrr", "hur"]]
        return households or ["hrp", "hup", "hrr", "hur"]

    def load_elasticities(self) -> ElasticitySet:
        """Load elasticities from VAL_PAR file.

        Returns:
            ElasticitySet with all elasticities loaded
        """
        if self.param_file and self.param_file.exists():
            df = pd.read_excel(self.param_file, header=None)
        else:
            # Load default
            from equilibria.templates.data.pep import get_default_pep_data_dir

            default_path = get_default_pep_data_dir() / "VAL_PAR.xlsx"
            df = pd.read_excel(default_path, header=None)

        elasticities = ElasticitySet()

        # Parse the VAL_PAR structure
        # Parameters indexed in J (sectors)
        j_params_start: int | None = None
        for idx, row in df.iterrows():
            if "Parameters indexed in j" in str(row.iloc[0]):
                j_params_start = int(idx)
                break

        if j_params_start is not None:
            # Header row with parameter names
            header_row = df.iloc[int(j_params_start) + 1]
            param_names = [
                str(v).strip() if pd.notna(v) else "" for v in header_row.iloc[1:]
            ]

            # Data rows
            for idx in range(int(j_params_start) + 2, len(df)):
                row = df.iloc[idx]
                sector = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                if not sector or sector in ["nan", "NaN"]:
                    continue

                # Map sector names to lowercase
                sector_map = {"AGR": "agr", "IND": "othind", "SER": "ser", "ADM": "adm"}
                sector_key = sector_map.get(sector, sector.lower())

                for j, param_name in enumerate(param_names):
                    if j + 1 >= len(row):
                        continue
                    value = row.iloc[j + 1]
                    if pd.notna(value):
                        try:
                            val = float(value)
                            if param_name == "sigma_KD":
                                elasticities.sigma_kd[sector_key] = val
                            elif param_name == "sigma_LD":
                                elasticities.sigma_ld[sector_key] = val
                            elif param_name == "sigma_VA":
                                elasticities.sigma_va[sector_key] = val
                            elif param_name == "sigma_XT":
                                elasticities.sigma_xt[sector_key] = val
                        except (ValueError, TypeError):
                            pass

        # Parameters indexed in I (commodities)
        i_params_start: int | None = None
        for idx, row in df.iterrows():
            if "Parameters indexed in i" in str(row.iloc[0]):
                i_params_start = int(idx)  # type: ignore[arg-type]
                break

        if i_params_start is not None:
            header_row = df.iloc[i_params_start + 1]
            param_names = [
                str(v).strip() if pd.notna(v) else "" for v in header_row.iloc[1:]
            ]

            for idx in range(i_params_start + 2, len(df)):
                row = df.iloc[idx]
                comm = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                if not comm or comm in ["nan", "NaN"]:
                    continue

                comm_map = {
                    "AGR": "agr",
                    "FOOD": "food",
                    "OTHIND": "othind",
                    "SER": "ser",
                    "ADM": "adm",
                }
                comm_key = comm_map.get(comm, comm.lower())

                for j, param_name in enumerate(param_names):
                    if j + 1 >= len(row):
                        continue
                    value = row.iloc[j + 1]
                    if pd.notna(value):
                        try:
                            val = float(value)
                            if param_name == "sigma_M":
                                elasticities.sigma_m[comm_key] = val
                            elif param_name == "sigma_XD":
                                elasticities.sigma_xd[comm_key] = val
                        except (ValueError, TypeError):
                            pass

        # Parameters indexed in AG (agents/households) - Frisch and LES
        ag_params_start: int | None = None
        for idx, row in df.iterrows():
            if "Parameters indexed in ag" in str(row.iloc[0]):
                ag_params_start = int(idx)  # type: ignore[arg-type]
                break

        if ag_params_start is not None:
            header_row = df.iloc[int(ag_params_start) + 1]
            hh_names = [
                str(v).strip() if pd.notna(v) else "" for v in header_row.iloc[1:]
            ]
            hh_map = {"HRP": "hrp", "HUP": "hup", "HRR": "hrr", "HUR": "hur"}

            for idx in range(int(ag_params_start) + 2, len(df)):
                row = df.iloc[idx]
                param_name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

                if param_name == "frisch":
                    for j, hh in enumerate(hh_names):
                        if j + 1 >= len(row):
                            continue
                        value = row.iloc[j + 1]
                        if pd.notna(value):
                            try:
                                val = float(value)
                                hh_key = hh_map.get(hh, hh.lower())
                                elasticities.frisch[hh_key] = val
                            except (ValueError, TypeError):
                                pass
                elif param_name in ["AGR", "FOOD", "OTHIND", "SER", "ADM"]:
                    comm_key = param_name.lower()
                    for j, hh in enumerate(hh_names):
                        if j + 1 >= len(row):
                            continue
                        value = row.iloc[j + 1]
                        if pd.notna(value):
                            try:
                                val = float(value)
                                hh_key = hh_map.get(hh, hh.lower())
                                if comm_key not in elasticities.les_elasticities:
                                    elasticities.les_elasticities[comm_key] = {}
                                elasticities.les_elasticities[comm_key][hh_key] = val
                            except (ValueError, TypeError):
                                pass

        self.elasticities = elasticities
        return elasticities

    def calibrate_taxes(self) -> TaxRates:
        """Calibrate tax rates from SAM data.

        Returns:
            TaxRates with all tax rates computed
        """
        taxes = TaxRates()
        if self.sam is None:
            return taxes
        sam = self.sam.data

        # Indirect taxes (ti) - from SAM ti row
        if "ti" in sam.index:
            for sector in self.sectors:
                if sector in sam.columns:
                    tax_revenue = sam.loc["ti", sector]
                    # Base is sector output (sum of column)
                    base = sam[sector].sum() if sector in sam.columns else 0
                    if base > 0:
                        taxes.ti[sector] = tax_revenue / base
                    else:
                        taxes.ti[sector] = 0.0

        # Import tariffs (tm) - from SAM tm row
        if "tm" in sam.index:
            for comm in self.commodities:
                if comm in sam.columns:
                    tariff_revenue = sam.loc["tm", comm]
                    # Base is imports (from ROW)
                    imports = (
                        sam.loc["row", comm]
                        if "row" in sam.index and comm in sam.columns
                        else 0
                    )
                    if imports > 0:
                        taxes.tm[comm] = tariff_revenue / imports
                    else:
                        taxes.tm[comm] = 0.0

        # Export taxes (tx) - from SAM tx row
        if "tx" in sam.index:
            for comm in self.commodities:
                if comm in sam.index:
                    export_tax = sam.loc["tx", comm]
                    # Base is exports (to ROW)
                    exports = (
                        sam.loc[comm, "row"]
                        if "row" in sam.columns and comm in sam.index
                        else 0
                    )
                    if exports > 0:
                        taxes.tx[comm] = export_tax / exports
                    else:
                        taxes.tx[comm] = 0.0

        # Direct taxes (td) - from SAM td row
        if "td" in sam.index:
            for hh in self.households:
                if hh in sam.columns:
                    direct_tax = sam.loc["td", hh]
                    # Base is household income
                    hh_income = sam[hh].sum() if hh in sam.columns else 0
                    if hh_income > 0:
                        taxes.td[hh] = direct_tax / hh_income
                    else:
                        taxes.td[hh] = 0.0

        self.tax_rates = taxes
        return taxes

    def calibrate(self) -> dict[str, Any]:
        """Run full calibration.

        Returns:
            Dictionary with all calibrated parameters
        """
        # Load elasticities
        if self.elasticities is None:
            self.load_elasticities()

        # Calibrate taxes
        if self.tax_rates is None:
            self.calibrate_taxes()

        return {
            "elasticities": self.elasticities,
            "tax_rates": self.tax_rates,
            "sectors": self.sectors,
            "commodities": self.commodities,
            "labor_types": self.labor_types,
            "capital_types": self.capital_types,
            "households": self.households,
        }


def load_pep_elasticities(param_file: Path | str | None = None) -> ElasticitySet:
    """Load PEP elasticities from file.

    Args:
        param_file: Path to VAL_PAR.xlsx (uses default if None)

    Returns:
        ElasticitySet with all elasticities
    """
    calibrator = PEPCalibrator(sam=None, param_file=param_file)
    return calibrator.load_elasticities()
