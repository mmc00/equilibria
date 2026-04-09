"""GTAP Contract Models (Standard GTAP 7)

This module defines canonical contract models for GTAP following Standard GTAP 7 implementation.
Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/comp.gms

Closures determine which variables are fixed (exogenous) and which are endogenous.
GTAP Standard 7 supports multiple closure types:
- Standard GTAP closure (default)
- Trade policy closure
- Single region closure
- Full model closure
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, field_validator

from equilibria.contracts import (
    ModelBoundsConfig,
    ModelClosureConfig,
    ModelContract,
    ModelEquationConfig,
    deep_merge_model_dicts,
)


def _full_gtap_equation_ids() -> Tuple[str, ...]:
    """Return all GTAP equation IDs.
    
    Based on GTAP Standard 7 model.gms equation definitions.
    """
    # Production block
    production = (
        "e_ptNest", "e_pdtNest", "e_xtNest", "e_lambdaio",
        "e_nd", "e_va", "e_lambdaf", "e_xf", "e_xfTNest",
        "e_prodShift", "e_gva", "e_p", "e_paVint", "e_pavgVintD",
        "e_pavgVint", "e_pfOldCap", "e_pfOldCapVNest", "e_rrat", "e_rratVNest",
        "e_soldCap",
    )
    
    # Trade block
    trade = (
        "e_pe", "e_xe", "e_pet", "e_pdet", "e_pmcif", "e_pefob",
        "e_xtmg", "e_xatmg", "e_ptmg", "e_pd",
    )
    
    # Factor block
    factors = (
        "e_xft", "e_xftNewCap", "e_xfNest", "e_pf", "e_pfNat", "e_xfNat",
        "e_pfa", "e_pfNest", "e_pftNat", "e_xftReg", "e_pft", "e_kstock",
    )
    
    # Demand block
    demand = (
        "e_xc", "e_xg", "e_xi", "e_pa",
    )
    
    # Income block
    income = (
        "e_facty", "e_regy", "e_yc", "e_ycF", "e_pcons",
        "e_pg", "e_xg_agg", "e_pi", "e_yi", "e_valDep",
        "e_xi_agg", "e_valSavf", "e_savf", "e_bop",
        "e_valFobCif", "e_valMarg", "e_netInv", "e_gblValNetInv",
        "e_gblValNetInv1", "e_rorg", "e_capAcct", "e_psaveHelp",
        "e_psave", "e_xigbl", "e_pigbl",
    )
    
    # Price indices
    prices = (
        "e_pabs", "e_pftFnm", "e_xftFnm", "e_vft", "e_pfact",
        "e_pprod", "e_pwfact", "e_pnum", "e_walras",
    )
    
    # Government
    government = (
        "e_ytax", "e_ytaxTot", "e_yg",
    )
    
    # GDP
    gdp = (
        "e_gdpmp", "e_qgdp", "e_rgdpmp", "e_rgdpmpPc", "e_pgdpmp",
    )
    
    # Market clearing
    market_clearing = (
        "mkt_pa", "mkt_pf", "mkt_ps", "mkt_ptmg", "mkt_pnum",
    )
    
    # Zero profit conditions
    zero_profit = (
        "prf_y", "prf_c", "prf_g", "prf_m", "prf_yt", "prf_ft",
    )
    
    return (
        production + trade + factors + demand + income + 
        prices + government + gdp + market_clearing + zero_profit
    )


def _closure_template_data(name: str) -> Dict[str, Any]:
    """Get closure template data by name.
    
    Args:
        name: Closure name (e.g., "gtap_standard", "trade_policy")
        
    Returns:
        Dictionary with closure configuration
    """
    closure_name = str(name).strip().lower()
    
    if not closure_name:
        raise ValueError("Closure name must be non-empty.")
    
    # Base closure (GTAP Standard)
    base = {
        "name": closure_name,
        "numeraire": "pnum",
        "numeraire_mode": "fixed_benchmark",
        "closure_type": "CNS",  # CNS = Constrained Nonlinear System
        "capital_mobility": "mobile",
        "savf_flag": "capFix",
        "apply_flag_fixing": True,
        "close_mcp_gap": True,
        "fix_taxes": True,
        "fix_technology": True,
        "fix_endowments": True,
        "fix_world_prices": False,
        "fixed": (
            # Taxes (fixed at benchmark values)
            "prdtx", "fctts", "fcttx", "exptx", "imptx",
            "dintx", "mintx", "itxshftGen", "kappashft",
            "kappaf", "kappafG", "etax", "mtax",
            # Technology parameters
            "axp", "lambdaN", "lambdaio", "lambdaf",
            "lambdaxm", "lambdam", "lambdamg", "lambdaDN",
            # Endowments
            "xft", "pop",
            # Trade margin cost shares
            "tmarg",
        ),
        "endogenous": (
            # Savings and investment
            "psave", "yi", "xi", "xigbl", "pigbl", "rorg",
            "chiInv", "netInv", "gblValNetInv",
        ),
        "label": None,
    }
    
    if closure_name == "gtap_standard":
        base["label"] = "Standard GTAP closure"
        return base
    
    elif closure_name == "gtap_full":
        base["label"] = "GTAP Standard 7 full closure with all equations"
        base["closure_type"] = "CNS"
        return base
    
    elif closure_name == "trade_policy":
        base["label"] = "Trade policy closure - allows tax changes"
        base["fix_taxes"] = False
        # Allow import and export taxes to vary
        base["fixed"] = tuple(f for f in base["fixed"] if f not in ["imptx", "exptx", "mtax", "etax"])
        return base
    
    elif closure_name == "single_region":
        base["label"] = "Single region closure - fixed world prices"
        base["fix_world_prices"] = True
        base["fixed"] = base["fixed"] + ("pmcif", "pefob", "ptmg")
        return base
    
    elif closure_name == "mcp":
        base["label"] = "MCP (Mixed Complementarity Problem) closure"
        base["closure_type"] = "MCP"
        return base
    
    raise ValueError(f"Unsupported GTAP closure name: {name!r}")


class GTAPClosureConfig(ModelClosureConfig):
    """Economic closure choices for GTAP model.
    
    This defines which variables remain fixed (exogenous) and which 
    are determined endogenously by the model.
    
    Attributes:
        name: Closure configuration name
        numeraire: Price variable used as numeraire (default: "pnum")
        numeraire_mode: How to handle numeraire ("fixed_benchmark")
        closure_type: Type of closure ("CNS" or "MCP")
        capital_mobility: Factor mobility assumption ("mobile" or "sluggish")
        fix_taxes: Whether to fix tax rates at benchmark values
        fix_technology: Whether to fix technology parameters
        fix_endowments: Whether to fix factor endowments
        fix_world_prices: Whether to fix world prices (single region mode)
        fixed: Tuple of variable names to fix
        endogenous: Tuple of variable names that are endogenous
    
    Example:
        >>> closure = GTAPClosureConfig(name="gtap_standard")
        >>> print(closure.numeraire)
        'pnum'
    """
    
    name: str = "gtap_standard"
    numeraire: str = "pnum"
    numeraire_mode: Literal["fixed_benchmark"] = "fixed_benchmark"
    closure_type: Literal["CNS", "MCP"] = "CNS"
    capital_mobility: Literal["mobile", "sluggish"] = "mobile"
    savf_flag: Literal["capFix", "capSFix", "capShrFix", "capFlex"] = "capFix"
    apply_flag_fixing: bool = True
    close_mcp_gap: bool = False
    
    # Closure flags
    fix_taxes: bool = True
    fix_technology: bool = True
    fix_endowments: bool = True
    fix_world_prices: bool = False
    
    # Fixed and endogenous variables
    fixed: Tuple[str, ...] = Field(
        default_factory=lambda: (
            # Taxes
            "prdtx", "fctts", "fcttx", "exptx", "imptx",
            "dintx", "mintx", "itxshftGen", "kappashft",
            "kappaf", "kappafG", "etax", "mtax",
            # Technology
            "axp", "lambdaN", "lambdaio", "lambdaf",
            "lambdaxm", "lambdam", "lambdamg", "lambdaDN",
            # Endowments
            "xft", "pop",
            # Trade margins
            "tmarg",
        )
    )
    
    endogenous: Tuple[str, ...] = Field(
        default_factory=lambda: (
            # Savings and investment
            "psave", "yi", "xi", "xigbl", "pigbl", "rorg",
            "chiInv", "netInv", "gblValNetInv",
        )
    )
    
    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        """Normalize closure name."""
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Closure name must be non-empty.")
        return text
    
    @field_validator("closure_type", mode="before")
    @classmethod
    def _normalize_closure_type(cls, value: Any) -> str:
        """Normalize closure type."""
        text = str(value).strip().upper()
        if text not in ("CNS", "MCP"):
            raise ValueError(f"Closure type must be 'CNS' or 'MCP', got {value!r}")
        return text


class GTAPEquationConfig(ModelEquationConfig):
    """Activated equation system for GTAP.
    
    This determines which equations are included in the model.
    
    Attributes:
        name: Equation configuration name
        include: Tuple of equation IDs to include
        activation_masks: How to handle equation activation
    """
    
    name: str = "full_gtap"
    include: Tuple[str, ...] = Field(default_factory=_full_gtap_equation_ids)
    activation_masks: Literal["gtap_standard", "all_active"] = "gtap_standard"
    
    @field_validator("activation_masks", mode="before")
    @classmethod
    def _normalize_activation_masks(cls, value: Any) -> str:
        """Normalize activation masks."""
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Activation masks must be non-empty.")
        return text


class GTAPBoundsConfig(ModelBoundsConfig):
    """Domain/bounds policy for GTAP variables.
    
    This defines bounds for model variables to ensure 
    economic meaningfulness and numerical stability.
    
    Attributes:
        name: Bounds configuration name
        positive: How to handle positive variables
        fixed_from_closure: Whether to apply bounds from closure
        free: Variables that should be unrestricted
        lower_bound: Default lower bound for positive variables
        upper_bound: Default upper bound (if any)
    """
    
    name: str = "economic"
    positive: Literal["lower_only", "both_bounds"] = "lower_only"
    fixed_from_closure: bool = True
    free: Tuple[str, ...] = Field(
        default_factory=lambda: (
            "savf", "valFobCif", "walras", "v_obje",
            "bopSlack", "ytax", "ytaxTot",
        )
    )
    lower_bound: float = 1e-6
    upper_bound: Optional[float] = None


class GTAPContract(ModelContract):
    """Resolved contract for GTAP CGE model.
    
    This combines closure, equation, and bounds configurations
    into a complete model contract.
    
    Attributes:
        name: Contract name
        closure: Closure configuration
        equations: Equation configuration
        bounds: Bounds configuration
    
    Example:
        >>> contract = GTAPContract()
        >>> print(contract.closure.name)
        'gtap_standard'
    """
    
    name: str = "gtap_standard7_9x10"
    closure: GTAPClosureConfig = Field(default_factory=GTAPClosureConfig)
    equations: GTAPEquationConfig = Field(default_factory=GTAPEquationConfig)
    bounds: GTAPBoundsConfig = Field(default_factory=GTAPBoundsConfig)


def default_gtap_contract() -> GTAPContract:
    """Return the canonical GTAP contract.
    
    Returns:
        GTAPContract with standard GTAP closure
    """
    return GTAPContract(
        closure=GTAPClosureConfig.model_validate(_closure_template_data("gtap_standard"))
    )


def build_gtap_closure_config(
    value: str | Mapping[str, Any] | GTAPClosureConfig | None = None,
) -> GTAPClosureConfig:
    """Resolve a GTAP closure configuration.
    
    Args:
        value: Can be:
            - None: Use default closure
            - str: Closure name (e.g., "gtap_standard")
            - Mapping: Dict with closure overrides
            - GTAPClosureConfig: Use as-is
            
    Returns:
        Resolved GTAPClosureConfig
    """
    if value is None:
        return GTAPClosureConfig.model_validate(_closure_template_data("gtap_standard"))
    
    if isinstance(value, GTAPClosureConfig):
        return value
    
    if isinstance(value, str):
        return GTAPClosureConfig.model_validate(_closure_template_data(value))
    
    if isinstance(value, Mapping):
        closure_name = value.get("name", value.get("preset", "gtap_standard"))
        base = _closure_template_data(str(closure_name))
        merged = deep_merge_model_dicts(base, value)
        if "preset" in merged and "name" in merged:
            merged.pop("preset", None)
        return GTAPClosureConfig.model_validate(merged)
    
    raise TypeError(
        "GTAP closure value must be None, a closure name string, a mapping, or GTAPClosureConfig."
    )


def build_gtap_contract(
    value: str | Mapping[str, Any] | GTAPContract | None = None
) -> GTAPContract:
    """Resolve a GTAP contract.
    
    Args:
        value: Can be:
            - None: Use default contract
            - str: Contract name (currently only "gtap_standard7_9x10")
            - Mapping: Dict with contract overrides
            - GTAPContract: Use as-is
            
    Returns:
        Resolved GTAPContract
    """
    if value is None:
        return default_gtap_contract()
    
    if isinstance(value, GTAPContract):
        return value
    
    if isinstance(value, str):
        contract_name = value.strip()
        if contract_name == "gtap_standard7_9x10":
            return default_gtap_contract()
        raise ValueError(f"Unsupported GTAP contract name: {value!r}")
    
    if isinstance(value, Mapping):
        base = default_gtap_contract().model_dump(mode="python")
        updates = dict(value)
        
        # Handle closure updates
        closure_value = updates.get("closure")
        if isinstance(closure_value, Mapping):
            updates["closure"] = build_gtap_closure_config(closure_value).model_dump(mode="python")
        elif isinstance(closure_value, (str, GTAPClosureConfig)):
            updates["closure"] = build_gtap_closure_config(closure_value).model_dump(mode="python")
        
        merged = deep_merge_model_dicts(base, updates)
        return GTAPContract.model_validate(merged)
    
    raise TypeError(
        "GTAP contract value must be None, a contract name string, a mapping, or GTAPContract."
    )
