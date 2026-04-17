"""Standardized MIP (Input-Output Matrix) loading utilities.

This module provides data structures and functions for loading and validating
Input-Output matrices from various Excel formats.

Example:
    >>> from equilibria.sam_tools.mip_loader import load_mip_excel, MIPConfig
    >>> config = MIPConfig(
    ...     z_sheet="consumo intermedio",
    ...     va_sheet="valor agregado",
    ...     fd_sheet="DF nal",
    ...     imp_sheet="importaciones",
    ... )
    >>> mip = load_mip_excel("bolivia_mip.xlsx", config)
    >>> errors = validate_mip_balances(mip)
    >>> print(f"PIB error: {errors['pib_error']:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class MIPData:
    """Data structure for a separated MIP (domestic/imported blocks).

    This class holds the core matrices and vectors of an Input-Output table
    with separation between domestic and imported flows.

    Attributes:
        Z_d: Domestic intermediate consumption matrix (n_products x n_sectors)
        Z_m: Imported intermediate consumption matrix (n_products x n_sectors)
        F_d: Domestic final demand matrix (n_products x n_fd_components)
        F_m: Imported final demand matrix (n_products x n_fd_components)
        VA: Value added matrix (n_va_components x n_sectors)
        X: Total production by sector (n_sectors,)
        M: Total imports by product (n_products,), optional
        sector_names: Names of sectors/industries
        product_names: Names of products (often same as sectors in symmetric MIP)
        fd_components: Names of final demand components (e.g., ["C_hh", "C_gov", ...])
        va_components: Names of value added components (e.g., ["L", "K", "taxes"])

    System of Equations:
        1. Product balance (domestic): X = Z_d @ 1 + F_d @ 1
        2. Import balance: M = Z_m @ 1 + F_m @ 1
        3. Industry balance: X = Z_d.T @ 1 + Z_m.T @ 1 + VA.T @ 1
        4. Aggregate identity: sum(F_d) = sum(VA) + sum(Z_m)
    """

    Z_d: pd.DataFrame
    Z_m: pd.DataFrame
    F_d: pd.DataFrame
    F_m: pd.DataFrame
    VA: pd.DataFrame
    X: pd.Series
    M: pd.Series | None = None
    sector_names: list[str] = field(default_factory=list)
    product_names: list[str] = field(default_factory=list)
    fd_components: list[str] = field(default_factory=list)
    va_components: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive names from DataFrames if not provided."""
        if not self.sector_names and hasattr(self.Z_d, "columns"):
            self.sector_names = list(self.Z_d.columns)
        if not self.product_names and hasattr(self.Z_d, "index"):
            self.product_names = list(self.Z_d.index)
        if not self.fd_components and hasattr(self.F_d, "columns"):
            self.fd_components = list(self.F_d.columns)
        if not self.va_components and hasattr(self.VA, "index"):
            self.va_components = list(self.VA.index)

    @property
    def n_products(self) -> int:
        """Number of products."""
        return len(self.Z_d)

    @property
    def n_sectors(self) -> int:
        """Number of sectors."""
        return len(self.Z_d.columns)

    @property
    def n_fd_components(self) -> int:
        """Number of final demand components."""
        return len(self.F_d.columns)

    @property
    def n_va_components(self) -> int:
        """Number of value added components."""
        return len(self.VA)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert all DataFrames to numpy arrays.

        Returns:
            Dictionary with keys: Z_d, Z_m, F_d, F_m, VA, X, M (if present)
        """
        result = {
            "Z_d": self.Z_d.values.astype(float),
            "Z_m": self.Z_m.values.astype(float),
            "F_d": self.F_d.values.astype(float),
            "F_m": self.F_m.values.astype(float),
            "VA": self.VA.values.astype(float),
            "X": self.X.values.astype(float),
        }
        if self.M is not None:
            result["M"] = self.M.values.astype(float)
        return result

    def total_output(self) -> float:
        """Total gross output (sum of X)."""
        return float(self.X.sum())

    def total_va(self) -> float:
        """Total value added (VAB at basic prices)."""
        return float(self.VA.values.sum())

    def total_imports(self) -> float:
        """Total imports (Z_m + F_m)."""
        return float(self.Z_m.values.sum() + self.F_m.values.sum())

    def total_final_demand_domestic(self) -> float:
        """Total domestic final demand."""
        return float(self.F_d.values.sum())

    def total_final_demand(self) -> float:
        """Total final demand (domestic + imported)."""
        return float(self.F_d.values.sum() + self.F_m.values.sum())


@dataclass
class MIPConfig:
    """Configuration for loading MIP from Excel.

    This class specifies the sheet names and structure of an Excel file
    containing an Input-Output matrix.

    Attributes:
        z_sheet: Sheet name for intermediate consumption matrix
        va_sheet: Sheet name for value added
        fd_sheet: Sheet name for final demand (domestic)
        imp_sheet: Sheet name for imports (optional, for separated MIP)
        fd_imp_sheet: Sheet name for imported final demand (optional)
        header_row: Row index for column headers (0-indexed)
        index_col: Column index for row labels (0-indexed)
        n_products: Number of products (auto-detect if None)
        n_sectors: Number of sectors (auto-detect if None)
        n_fd_cols: Number of final demand columns (auto-detect if None)
        n_va_rows: Number of value added rows (auto-detect if None)
        skip_rows: Rows to skip at top of each sheet
        skip_cols: Columns to skip at left of each sheet
        combined_sheet: If True, all data is in one sheet (legacy format)
        combined_sheet_name: Name of combined sheet (if combined_sheet=True)
    """

    z_sheet: str = "consumo intermedio"
    va_sheet: str = "valor agregado"
    fd_sheet: str = "DF nal"
    imp_sheet: str | None = "importaciones"
    fd_imp_sheet: str | None = "DF imp"
    header_row: int = 0
    index_col: int = 0
    n_products: int | None = None
    n_sectors: int | None = None
    n_fd_cols: int | None = None
    n_va_rows: int | None = None
    skip_rows: int = 0
    skip_cols: int = 0
    combined_sheet: bool = False
    combined_sheet_name: str = "mip"

    @classmethod
    def bolivia_format(cls) -> "MIPConfig":
        """Configuration for Bolivia MIP format.

        The Bolivia MIP uses separate sheets for each component.
        """
        return cls(
            z_sheet="consumo intermedio",
            va_sheet="valor agregado",
            fd_sheet="DF nal",
            imp_sheet="importaciones",
            fd_imp_sheet="DF imp",
            n_products=70,
            n_sectors=70,
            n_fd_cols=5,
            n_va_rows=3,
        )

    @classmethod
    def combined_format(cls, sheet_name: str = "mip") -> "MIPConfig":
        """Configuration for combined single-sheet MIP format.

        This format has all data in one sheet with structure:
        [Z | F]
        [IMP_Z | IMP_F]
        [VA | 0]
        """
        return cls(
            combined_sheet=True,
            combined_sheet_name=sheet_name,
        )


def load_mip_excel(filepath: Path | str, config: MIPConfig | None = None) -> MIPData:
    """Load MIP from Excel file with flexible configuration.

    This function can handle both separated MIP formats (multiple sheets)
    and combined formats (single sheet with all blocks).

    Args:
        filepath: Path to Excel file
        config: Configuration for loading (auto-detect if None)

    Returns:
        MIPData with all matrices loaded

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required sheets are missing

    Example:
        >>> # Load with auto-detection
        >>> mip = load_mip_excel("mip_bolivia.xlsx")

        >>> # Load with specific config
        >>> config = MIPConfig.bolivia_format()
        >>> mip = load_mip_excel("mip_bolivia.xlsx", config)

        >>> # Load combined format
        >>> config = MIPConfig.combined_format("MIP2023")
        >>> mip = load_mip_excel("mip_single_sheet.xlsx", config)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MIP file not found: {filepath}")

    if config is None:
        config = _auto_detect_config(filepath)

    if config.combined_sheet:
        return _load_combined_mip(filepath, config)
    else:
        return _load_separated_mip(filepath, config)


def _auto_detect_config(filepath: Path) -> MIPConfig:
    """Auto-detect MIP format from Excel file."""
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names

    # Check for Bolivia-style separated format
    if "consumo intermedio" in sheets or "DF nal" in sheets:
        return MIPConfig.bolivia_format()

    # Check for common combined sheet names
    for name in ["mip", "MIP", "IO", "matriz"]:
        if name in sheets:
            return MIPConfig.combined_format(name)

    # Default to first sheet as combined
    return MIPConfig.combined_format(sheets[0])


def _load_separated_mip(filepath: Path, config: MIPConfig) -> MIPData:
    """Load MIP from multiple sheets."""
    xl = pd.ExcelFile(filepath)

    # Load intermediate consumption (Z)
    Z_d = pd.read_excel(
        xl,
        sheet_name=config.z_sheet,
        header=config.header_row,
        index_col=config.index_col,
    )

    # Remove any totals row/column
    if "X" in Z_d.index:
        Z_d = Z_d.drop("X")
    if "X" in Z_d.columns:
        Z_d = Z_d.drop(columns="X")

    n_products = config.n_products or len(Z_d)
    n_sectors = config.n_sectors or len(Z_d.columns)

    Z_d = Z_d.iloc[:n_products, :n_sectors].fillna(0)

    # Load value added
    VA = pd.read_excel(
        xl,
        sheet_name=config.va_sheet,
        header=config.header_row,
        index_col=config.index_col,
    )
    n_va = config.n_va_rows or len(VA)
    VA = VA.iloc[:n_va, :n_sectors].fillna(0)

    # Load final demand (domestic)
    F_d = pd.read_excel(
        xl,
        sheet_name=config.fd_sheet,
        header=config.header_row,
        index_col=config.index_col,
    )
    n_fd = config.n_fd_cols or len(F_d.columns)
    F_d = F_d.iloc[:n_products, :n_fd].fillna(0)

    # Load imports if available
    if config.imp_sheet and config.imp_sheet in xl.sheet_names:
        IMP = pd.read_excel(
            xl,
            sheet_name=config.imp_sheet,
            header=config.header_row,
            index_col=config.index_col,
        )
        # Imports sheet typically has Z_m and F_m combined
        Z_m = IMP.iloc[:n_products, :n_sectors].fillna(0)
        if IMP.shape[1] > n_sectors:
            F_m = IMP.iloc[:n_products, n_sectors : n_sectors + n_fd].fillna(0)
        else:
            F_m = pd.DataFrame(
                np.zeros((n_products, n_fd)),
                index=Z_d.index,
                columns=F_d.columns,
            )
    else:
        Z_m = pd.DataFrame(
            np.zeros((n_products, n_sectors)),
            index=Z_d.index,
            columns=Z_d.columns,
        )
        F_m = pd.DataFrame(
            np.zeros((n_products, n_fd)),
            index=Z_d.index,
            columns=F_d.columns,
        )

    # Load imported final demand if separate sheet
    if config.fd_imp_sheet and config.fd_imp_sheet in xl.sheet_names:
        F_m = pd.read_excel(
            xl,
            sheet_name=config.fd_imp_sheet,
            header=config.header_row,
            index_col=config.index_col,
        )
        F_m = F_m.iloc[:n_products, :n_fd].fillna(0)

    # Calculate production totals
    X = Z_d.sum(axis=0) + VA.sum(axis=0)
    X.name = "X"

    # Calculate import totals
    M = Z_m.sum(axis=1) + F_m.sum(axis=1)
    M.name = "M"

    return MIPData(
        Z_d=Z_d,
        Z_m=Z_m,
        F_d=F_d,
        F_m=F_m,
        VA=VA,
        X=X,
        M=M,
    )


def _load_combined_mip(filepath: Path, config: MIPConfig) -> MIPData:
    """Load MIP from a single combined sheet."""
    df = pd.read_excel(
        filepath,
        sheet_name=config.combined_sheet_name,
        header=config.header_row,
        index_col=config.index_col,
    )

    # Remove totals if present
    if "X" in df.index:
        df = df.drop("X")
    if "X" in df.columns:
        df = df.drop(columns="X")

    df = df.fillna(0)

    # Auto-detect dimensions
    n_rows, n_cols = df.shape

    # Assume symmetric MIP with structure:
    # [Z (n x n) | F (n x k)]
    # [IMP_Z (n x n) | IMP_F (n x k)]  -- optional
    # [VA (v x n) | 0 (v x k)]

    # Detect by looking for VA row labels
    va_keywords = ["remun", "excedente", "surplus", "compensation", "impuesto", "tax"]

    va_start = None
    for i, idx in enumerate(df.index):
        idx_lower = str(idx).lower()
        if any(kw in idx_lower for kw in va_keywords):
            va_start = i
            break

    if va_start is None:
        # Assume VA is at the bottom, with 3 rows
        va_start = n_rows - 3

    # Detect import rows (between products and VA)
    n_products = config.n_products
    if n_products is None:
        # Assume products are labeled with numbers or "ind-" prefix
        n_products = 0
        for i, idx in enumerate(df.index):
            idx_str = str(idx).lower()
            if "ind-" in idx_str or idx_str.isdigit() or idx_str.startswith("prod"):
                n_products = i + 1
            else:
                break
        if n_products == 0:
            n_products = va_start // 2  # Assume half are products, half imports

    n_sectors = config.n_sectors or n_products

    # Detect FD columns
    fd_start = n_sectors
    n_fd = config.n_fd_cols or (n_cols - n_sectors)

    # Check for import rows
    has_imports = va_start > n_products

    # Extract blocks
    Z_d = df.iloc[:n_products, :n_sectors].copy()

    if has_imports:
        Z_m = df.iloc[n_products:va_start, :n_sectors].copy()
        Z_m.index = Z_d.index  # Align indices
    else:
        Z_m = pd.DataFrame(
            np.zeros((n_products, n_sectors)),
            index=Z_d.index,
            columns=Z_d.columns,
        )

    F_d = df.iloc[:n_products, fd_start : fd_start + n_fd].copy()

    if has_imports:
        F_m = df.iloc[n_products:va_start, fd_start : fd_start + n_fd].copy()
        F_m.index = F_d.index
    else:
        F_m = pd.DataFrame(
            np.zeros((n_products, n_fd)),
            index=F_d.index,
            columns=F_d.columns,
        )

    VA = df.iloc[va_start:, :n_sectors].copy()

    # Calculate production totals
    X = Z_d.sum(axis=0) + VA.sum(axis=0)
    X.name = "X"

    # Calculate import totals
    M = Z_m.sum(axis=1) + F_m.sum(axis=1)
    M.name = "M"

    return MIPData(
        Z_d=Z_d,
        Z_m=Z_m,
        F_d=F_d,
        F_m=F_m,
        VA=VA,
        X=X,
        M=M,
    )


def validate_mip_balances(mip: MIPData) -> dict[str, float]:
    """Validate MIP balance constraints and return errors.

    Checks the three fundamental MIP constraints:
    1. Product balance (domestic): X = Z_d @ 1 + F_d @ 1
    2. Industry balance: X = Z_d.T @ 1 + Z_m.T @ 1 + VA.T @ 1
    3. PIB identity: sum(VA) = sum(F_d) - sum(F_m)

    Note: Supply-Demand balance (per product) is NOT checked because
    it is mathematically incompatible with constraints 1-3 when F_m > 0.
    See bolivia_mip_technical_report.md for the proof.

    Args:
        mip: MIPData to validate

    Returns:
        Dictionary with error metrics:
        - product_error_max: Maximum product balance error
        - product_error_mean: Mean product balance error
        - industry_error_max: Maximum industry balance error
        - industry_error_mean: Mean industry balance error
        - pib_error: Absolute PIB identity error
        - pib_error_pct: PIB error as percentage of VA
        - supply_demand_error_max: Max S-D error (informational only)
        - is_balanced: True if all errors below tolerance

    Example:
        >>> mip = load_mip_excel("mip.xlsx")
        >>> errors = validate_mip_balances(mip)
        >>> if errors["is_balanced"]:
        ...     print("MIP is balanced")
        >>> else:
        ...     print(f"PIB error: {errors['pib_error_pct']:.2%}")
    """
    Z_d = mip.Z_d.values.astype(float)
    Z_m = mip.Z_m.values.astype(float)
    F_d = mip.F_d.values.astype(float)
    F_m = mip.F_m.values.astype(float)
    VA = mip.VA.values.astype(float)
    X = mip.X.values.astype(float)

    # 1. Product balance: X = Z_d @ 1 + F_d @ 1
    product_supply = X
    product_demand = Z_d.sum(axis=1) + F_d.sum(axis=1)
    product_error = np.abs(product_supply - product_demand)
    product_error_max = float(product_error.max())
    product_error_mean = float(product_error.mean())

    # 2. Industry balance: X = Z_d.T @ 1 + Z_m.T @ 1 + VA
    industry_output = X
    industry_input = Z_d.sum(axis=0) + Z_m.sum(axis=0) + VA.sum(axis=0)
    industry_error = np.abs(industry_output - industry_input)
    industry_error_max = float(industry_error.max())
    industry_error_mean = float(industry_error.mean())

    # 3. PIB identity: sum(VA) = sum(F_d) - sum(F_m)
    pib_production = float(VA.sum())
    pib_expenditure = float(F_d.sum() - F_m.sum())
    pib_error = abs(pib_production - pib_expenditure)
    pib_error_pct = pib_error / pib_production if pib_production > 0 else 0.0

    # 4. Supply-Demand balance (informational only - NOT a requirement)
    # Supply = X + M, Demand = Z_d.sum(1) + Z_m.sum(1) + F_d.sum(1) + F_m.sum(1)
    supply = X + Z_m.sum(axis=1) + F_m.sum(axis=1)
    demand = Z_d.sum(axis=1) + Z_m.sum(axis=1) + F_d.sum(axis=1) + F_m.sum(axis=1)
    sd_error = np.abs(supply - demand)
    sd_error_max = float(sd_error.max())

    # Determine if balanced (tolerances)
    tol_absolute = 1.0  # 1 USD tolerance
    tol_pib_pct = 0.01  # 1% PIB error

    is_balanced = (
        product_error_max < tol_absolute
        and industry_error_max < tol_absolute
        and pib_error_pct < tol_pib_pct
    )

    return {
        "product_error_max": product_error_max,
        "product_error_mean": product_error_mean,
        "industry_error_max": industry_error_max,
        "industry_error_mean": industry_error_mean,
        "pib_error": pib_error,
        "pib_error_pct": pib_error_pct,
        "supply_demand_error_max": sd_error_max,
        "is_balanced": is_balanced,
        "total_va": pib_production,
        "total_final_demand": float(F_d.sum()),
        "total_imports": float(Z_m.sum() + F_m.sum()),
    }


def mip_summary(mip: MIPData) -> str:
    """Generate a human-readable summary of MIP data.

    Args:
        mip: MIPData to summarize

    Returns:
        Formatted string with MIP statistics
    """
    errors = validate_mip_balances(mip)

    lines = [
        "=" * 60,
        "MIP Summary",
        "=" * 60,
        "",
        "Dimensions:",
        f"  Products:       {mip.n_products}",
        f"  Sectors:        {mip.n_sectors}",
        f"  FD Components:  {mip.n_fd_components} ({', '.join(mip.fd_components[:5])}...)"
        if len(mip.fd_components) > 5
        else f"  FD Components:  {mip.n_fd_components} ({', '.join(mip.fd_components)})",
        f"  VA Components:  {mip.n_va_components} ({', '.join(mip.va_components)})",
        "",
        "Totals:",
        f"  Gross Output (X):     {mip.total_output():,.2f}",
        f"  Value Added (VA):     {mip.total_va():,.2f}",
        f"  Final Demand (F_d):   {mip.total_final_demand_domestic():,.2f}",
        f"  Total Imports (M):    {mip.total_imports():,.2f}",
        f"  Z_d (domestic CI):    {float(mip.Z_d.values.sum()):,.2f}",
        f"  Z_m (imported CI):    {float(mip.Z_m.values.sum()):,.2f}",
        "",
        "Balance Errors:",
        f"  Product balance (max):   {errors['product_error_max']:.2f}",
        f"  Industry balance (max):  {errors['industry_error_max']:.2f}",
        f"  PIB error:               {errors['pib_error']:.2f} ({errors['pib_error_pct']:.2%})",
        f"  S-D error (info only):   {errors['supply_demand_error_max']:.2f}",
        "",
        f"Status: {'BALANCED' if errors['is_balanced'] else 'NOT BALANCED'}",
        "=" * 60,
    ]

    return "\n".join(lines)
