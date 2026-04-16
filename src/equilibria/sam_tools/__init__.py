"""SAM transformation pipeline utilities.

This module provides tools for:
- Parsing and exporting SAM (Social Accounting Matrix) data
- Converting MIP (Input-Output Matrix) to SAM
- Balancing matrices using RAS, GRAS, and other methods
- Loading and validating MIP data from Excel

Key functions:
- run_mip_to_sam: Convert MIP to SAM with flexible balancing
- run_ieem_to_pep: Convert IEEM SAM to PEP format
- load_mip_excel: Load MIP from Excel with auto-detection
- validate_mip_balances: Check MIP balance constraints
"""

from equilibria.sam_tools.api import (
    BalancingMethod,
    IEEMToPEPResult,
    MIPToSAMResult,
    run_ieem_to_pep,
    run_mip_to_sam,
)
from equilibria.sam_tools.balancing import (
    MIPBalanceResult,
    RASBalancer,
    RASBalanceResult,
    balance_mip_entropy,
    balance_mip_gras,
    balance_mip_ras,
    balance_mip_sut_ras,
    gras_balance,
    ras_balance,
)
from equilibria.sam_tools.enums import (
    IPFPSupportMode,
    IPFPTargetMode,
    RASMode,
    SAMFormat,
    WorkflowOperation,
)
from equilibria.sam_tools.mip_loader import (
    MIPConfig,
    MIPData,
    load_mip_excel,
    mip_summary,
    validate_mip_balances,
)
from equilibria.sam_tools.models import Sam
from equilibria.sam_tools.parsers import export_sam, parse_sam

__all__ = [
    # Core SAM class
    "Sam",
    # Parsing and export
    "parse_sam",
    "export_sam",
    # API functions
    "run_ieem_to_pep",
    "IEEMToPEPResult",
    "run_mip_to_sam",
    "MIPToSAMResult",
    "BalancingMethod",
    # MIP loading
    "MIPData",
    "MIPConfig",
    "load_mip_excel",
    "validate_mip_balances",
    "mip_summary",
    # Balancing
    "RASBalancer",
    "RASBalanceResult",
    "MIPBalanceResult",
    "ras_balance",
    "gras_balance",
    "balance_mip_ras",
    "balance_mip_gras",
    "balance_mip_sut_ras",
    "balance_mip_entropy",
    # Enums
    "SAMFormat",
    "WorkflowOperation",
    "RASMode",
    "IPFPTargetMode",
    "IPFPSupportMode",
]
