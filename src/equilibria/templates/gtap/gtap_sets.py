"""GTAP Sets and Declarations (CGEBox version)

This module defines all GTAP model sets following the CGEBox implementation.
Reference: /Users/marmol/proyectos2/cge_babel/cgebox/gams/model/model.gms

Key Sets:
- r: Regions
- i: Commodities/goods
- a: Activities/sectors (alias of i in standard GTAP)
- f: Factors of production
- mf: Mobile factors (subset of f)
- sf: Sector-specific factors (subset of f)
- m: Transport modes (for international trade margins)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from equilibria.babel.gdx.reader import read_gdx


@dataclass
class GTAPSets:
    """GTAP model sets following CGEBox structure.
    
    Attributes:
        r: Regions in the model (e.g., ["EUR", "USA", "CHN", "BRA", "IND"])
        i: Commodities/goods (e.g., ["agr", "food", "mfg", "srv", "ene"])
        a: Activities/sectors (typically alias of i, but allows multi-product)
        f: Factors of production (e.g., ["lnd", "skl", "unsk", "cap", "nrs"])
        mf: Mobile factors (subset of f that can move across sectors)
        sf: Sector-specific factors (subset of f that are fixed to sectors)
        m: Transport modes for international trade (e.g., ["air", "sea", "road", "rail"])
        h: Households (for myGTAP extension, optional)
    
    Example:
        >>> sets = GTAPSets()
        >>> sets.load_from_gdx(Path("asa7x5.gdx"))
        >>> print(f"Regions: {sets.r}")
        >>> print(f"Commodities: {sets.i}")
    """
    
    # Core GTAP sets
    r: List[str] = field(default_factory=list)   # Regions
    i: List[str] = field(default_factory=list)   # Commodities
    a: List[str] = field(default_factory=list)   # Activities (alias of i)
    f: List[str] = field(default_factory=list)   # Factors
    
    # Factor subsets
    mf: List[str] = field(default_factory=list)  # Mobile factors
    sf: List[str] = field(default_factory=list)  # Sector-specific factors
    
    # Trade and transport
    m: List[str] = field(default_factory=list)   # Transport modes
    
    # Optional extensions
    h: Optional[List[str]] = None  # Households (for myGTAP)
    
    # Aliases
    s: Optional[List[str]] = None  # Alias of r (for bilateral trade)
    
    # Metadata
    aggregation_name: str = ""
    base_year: int = 2014
    
    def load_from_gdx(self, gdx_path: Path) -> None:
        """Load sets from GTAP GDX file.
        
        Args:
            gdx_path: Path to GDX file (e.g., asa7x5.gdx)
            
        Raises:
            FileNotFoundError: If GDX file doesn't exist
            ValueError: If required sets are not found
        """
        if not gdx_path.exists():
            raise FileNotFoundError(f"GDX file not found: {gdx_path}")
        
        # Read GDX file
        gdx_data = read_gdx(gdx_path)
        symbols = {s["name"]: s for s in gdx_data.get("symbols", [])}
        
        # Load core sets
        self.r = self._load_set(symbols, "r", required=True)
        self.i = self._load_set(symbols, "i", required=True)
        self.a = self._load_set(symbols, "a", required=False) or self.i.copy()
        self.f = self._load_set(symbols, "f", required=True)
        
        # Load factor subsets
        self.mf = self._load_set(symbols, "mf", required=False) or []
        self.sf = self._load_set(symbols, "sf", required=False) or []
        
        # Load transport modes
        self.m = self._load_set(symbols, "m", required=False) or ["air", "sea"]
        
        # If mf/sf not defined, determine from etrae parameter
        if not self.mf and not self.sf:
            self._determine_factor_mobility(symbols)
        
        # Set aliases
        self.s = self.r.copy()
        
        # Store metadata
        self.aggregation_name = gdx_path.stem
        
    def _load_set(self, symbols: Dict, name: str, required: bool = True) -> Optional[List[str]]:
        """Load a set from GDX symbols.
        
        Args:
            symbols: Dictionary of GDX symbols
            name: Set name to load
            required: Whether the set is required
            
        Returns:
            List of set elements or None if not found and not required
        """
        if name not in symbols:
            if required:
                raise ValueError(f"Required set '{name}' not found in GDX file")
            return None
        
        symbol = symbols[name]
        if symbol.get("type") == "set":
            # Get elements from symbol data
            elements = symbol.get("elements", [])
            if not elements and "data" in symbol:
                # Extract from data records
                elements = [str(e) for e in symbol["data"].keys()]
            return sorted(elements) if elements else []
        
        return None
    
    def _determine_factor_mobility(self, symbols: Dict) -> None:
        """Determine factor mobility from etrae parameter.
        
        In GTAP, factor mobility is determined by the elasticity of 
        transformation (etrae). Infinite etrae means mobile factor.
        """
        if "etrae" in symbols:
            # Load etrae values
            etrae_data = symbols["etrae"].get("data", {})
            for factor, value in etrae_data.items():
                factor_name = str(factor)
                # Infinite elasticity = mobile factor
                if value == float('inf') or value > 1e10:
                    self.mf.append(factor_name)
                else:
                    self.sf.append(factor_name)
        else:
            # Default: all factors mobile except land and natural resources
            mobile_defaults = ["skl", "unsk", "cap", "lab"]
            sluggish_defaults = ["lnd", "nrs"]
            
            for f in self.f:
                if any(m in f.lower() for m in mobile_defaults):
                    self.mf.append(f)
                else:
                    self.sf.append(f)
    
    @property
    def n_regions(self) -> int:
        """Number of regions."""
        return len(self.r)
    
    @property
    def n_commodities(self) -> int:
        """Number of commodities."""
        return len(self.i)
    
    @property
    def n_activities(self) -> int:
        """Number of activities/sectors."""
        return len(self.a)
    
    @property
    def n_factors(self) -> int:
        """Number of factors."""
        return len(self.f)
    
    @property
    def n_mobile_factors(self) -> int:
        """Number of mobile factors."""
        return len(self.mf)
    
    @property
    def n_specific_factors(self) -> int:
        """Number of sector-specific factors."""
        return len(self.sf)
    
    def get_region_index(self, region: str) -> int:
        """Get index of a region."""
        return self.r.index(region)
    
    def get_commodity_index(self, commodity: str) -> int:
        """Get index of a commodity."""
        return self.i.index(commodity)
    
    def get_activity_index(self, activity: str) -> int:
        """Get index of an activity."""
        return self.a.index(activity)
    
    def get_factor_index(self, factor: str) -> int:
        """Get index of a factor."""
        return self.f.index(factor)
    
    def is_mobile_factor(self, factor: str) -> bool:
        """Check if a factor is mobile."""
        return factor in self.mf
    
    def is_specific_factor(self, factor: str) -> bool:
        """Check if a factor is sector-specific."""
        return factor in self.sf
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate sets consistency.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check non-empty sets
        if not self.r:
            errors.append("Regions set (r) is empty")
        if not self.i:
            errors.append("Commodities set (i) is empty")
        if not self.f:
            errors.append("Factors set (f) is empty")
        
        # Check factor subsets cover all factors
        if self.f:
            all_factors = set(self.mf) | set(self.sf)
            if all_factors != set(self.f):
                missing = set(self.f) - all_factors
                extra = all_factors - set(self.f)
                if missing:
                    errors.append(f"Factors not in mf or sf: {missing}")
                if extra:
                    errors.append(f"Extra factors in mf/sf: {extra}")
        
        # Check activity-commodity relationship
        if self.a and self.i:
            if set(self.a) != set(self.i):
                errors.append("Activities (a) and commodities (i) should match in standard GTAP")
        
        return len(errors) == 0, errors
    
    def get_info(self) -> Dict:
        """Get summary information about sets."""
        is_valid, errors = self.validate()
        return {
            "aggregation": self.aggregation_name,
            "base_year": self.base_year,
            "n_regions": self.n_regions,
            "n_commodities": self.n_commodities,
            "n_activities": self.n_activities,
            "n_factors": self.n_factors,
            "n_mobile_factors": self.n_mobile_factors,
            "n_specific_factors": self.n_specific_factors,
            "regions": self.r,
            "commodities": self.i,
            "activities": self.a,
            "factors": self.f,
            "mobile_factors": self.mf,
            "specific_factors": self.sf,
            "valid": is_valid,
            "errors": errors,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GTAPSets({self.aggregation_name}: "
            f"{self.n_regions} regions × "
            f"{self.n_commodities} commodities × "
            f"{self.n_factors} factors)"
        )
