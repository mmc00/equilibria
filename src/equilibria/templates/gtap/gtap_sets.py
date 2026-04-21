"""GTAP Sets and Declarations (Standard GTAP 7)

This module defines all GTAP model sets following the GTAP Standard 7 implementation.
Reference: /Users/marmol/proyectos2/cge_babel/standard_gtap_7/model.gms

Key Sets:
- r: Regions
- i: Commodities/goods
- a: Activities/sectors (alias of i in standard GTAP)
- f: Factors of production
- mf: Mobile factors (subset of f)
- sf: Sector-specific factors (subset of f)
- m: Alias of i used by the GTAP trade-margin mode block
- marg: Active margin commodities from data (subset of i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from equilibria.babel.gdx.reader import read_gdx, read_parameter_values, read_set_elements
from equilibria.babel.gdx.gdxdump import read_parameter_with_gdxdump, read_set_with_gdxdump


@dataclass
class GTAPSets:
    """GTAP model sets following GTAP Standard 7 structure.
    
    Attributes:
        r: Regions in the model (e.g., ["EUR", "USA", "CHN", "BRA", "IND"])
        i: Commodities/goods (e.g., ["agr", "food", "mfg", "srv", "ene"])
        a: Activities/sectors (typically alias of i, but allows multi-product)
        f: Factors of production (e.g., ["lnd", "skl", "unsk", "cap", "nrs"])
        mf: Mobile factors (subset of f that can move across sectors)
        sf: Sector-specific factors (subset of f that are fixed to sectors)
        m: Alias of i for trade/transport modes in model.gms (`alias(m,i)`)
        marg: Active margin commodities from data (subset from MARG set)
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
    m: List[str] = field(default_factory=list)      # Alias of i (GAMS: alias(m,i))
    marg: List[str] = field(default_factory=list)   # Active margin commodities from data
    
    # Optional extensions
    h: Optional[List[str]] = None  # Households (for myGTAP)
    
    # Aliases
    s: Optional[List[str]] = None  # Alias of r (for bilateral trade)

    # Output structure metadata
    i_to_a: Dict[str, str] = field(default_factory=dict)
    a_to_i: Dict[str, str] = field(default_factory=dict)
    output_pairs: List[Tuple[str, str]] = field(default_factory=list)
    activity_commodities: Dict[str, List[str]] = field(default_factory=dict)
    commodity_activities: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    aggregation_name: str = ""
    base_year: int = 2014
    source_gdx: Optional[Path] = None
    
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
        self.source_gdx = gdx_path

        # Load core sets
        self.r = self._load_first_available_set(gdx_data, symbols, ("r", "reg"), gdx_path, required=True)
        self.i = self._load_first_available_set(gdx_data, symbols, ("i", "comm"), gdx_path, required=True)
        self.a = self._load_first_available_set(gdx_data, symbols, ("a", "acts"), gdx_path, required=False) or self.i.copy()
        self.f = self._load_first_available_set(gdx_data, symbols, ("f", "fp", "endw"), gdx_path, required=True)
        
        # Load factor subsets
        self.mf = self._load_first_available_set(gdx_data, symbols, ("mf", "fm", "endwm"), gdx_path, required=False) or []
        self.sf = self._load_first_available_set(gdx_data, symbols, ("sf", "fnm", "endws"), gdx_path, required=False) or []
        
        # Raw GTAP data provides active margin commodities via marg(comm),
        # but the standard model declares alias(m,i), i.e. full commodity set.
        self.marg = self._load_first_available_set(gdx_data, symbols, ("marg",), gdx_path, required=False) or []
        self.m = self.i.copy()
        
        # If mf/sf not defined, determine from etrae parameter
        if not self.mf and not self.sf:
            self._determine_factor_mobility(gdx_data)

        self._set_activity_mappings(gdx_data)
        
        # Set aliases
        self.s = self.r.copy()
        
        # Store metadata
        self.aggregation_name = gdx_path.stem
        
    def _load_first_available_set(
        self,
        gdx_data: Dict[str, Any],
        symbols: Dict[str, Dict[str, Any]],
        names: Sequence[str],
        gdx_path: Path,
        required: bool = True,
    ) -> Optional[List[str]]:
        """Load the first available set among a family of aliases."""
        tried = []
        for name in names:
            for alias in (name, name.upper()):
                if alias in tried:
                    continue
                tried.append(alias)
                fallback = read_set_with_gdxdump(gdx_path, alias)
                if fallback:
                    return fallback
                elements = self._load_set(gdx_data, symbols, alias)
                if elements:
                    return elements

        if required:
            raise ValueError(f"Required set aliases {names} not found in GDX file")
        return None

    def _load_set(
        self,
        gdx_data: Dict[str, Any],
        symbols: Dict[str, Dict[str, Any]],
        name: str,
    ) -> Optional[List[str]]:
        """Load a set from GDX symbols, supporting both decoded and mocked payloads."""
        if name not in symbols:
            return None

        symbol = symbols[name]
        symbol_type = symbol.get("type")
        if symbol_type not in (0, "set"):
            return None

        elements = symbol.get("elements", [])
        if elements:
            return [str(element) for element in elements]

        raw_data = symbol.get("data", {})
        if raw_data:
            if isinstance(raw_data, dict):
                return [str(element) for element in raw_data.keys()]
            return [str(element) for element in raw_data]

        try:
            records = read_set_elements(gdx_data, name)
        except (ValueError, FileNotFoundError):
            return None

        if not records:
            return []

        labels: List[str] = []
        for record in records:
            if len(record) == 1:
                labels.append(str(record[0]))
            else:
                labels.append(str(record))
        return labels


    def _determine_factor_mobility(self, gdx_data: Dict[str, Any]) -> None:
        """Determine factor mobility from etrae parameter.
        
        In GTAP, factor mobility is determined by the elasticity of 
        transformation (etrae). Infinite etrae means mobile factor.
        """
        try:
            etrae_data = read_parameter_values(gdx_data, "etrae")
        except (ValueError, FileNotFoundError):
            etrae_data = {}

        if etrae_data:
            factor_is_mobile: Dict[str, bool] = {}
            for factor_key, value in etrae_data.items():
                if isinstance(factor_key, tuple):
                    # ETRAE is often indexed by (factor,region) in GTAP.
                    # Some exports can swap order, so select the first token
                    # that is an actual factor label.
                    factor_name = ""
                    for token in factor_key:
                        token_str = str(token)
                        if token_str in self.f:
                            factor_name = token_str
                            break
                    if not factor_name:
                        factor_name = str(factor_key[0])
                else:
                    factor_name = str(factor_key)
                if factor_name not in self.f:
                    continue
                is_mobile = bool(value == float('inf') or value > 1e10)
                if factor_name not in factor_is_mobile:
                    factor_is_mobile[factor_name] = is_mobile
                else:
                    # If any region marks a factor as mobile, keep it mobile.
                    factor_is_mobile[factor_name] = factor_is_mobile[factor_name] or is_mobile

            for factor_name in self.f:
                is_mobile = factor_is_mobile.get(factor_name)
                if is_mobile is True:
                    self.mf.append(factor_name)
                elif is_mobile is False:
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

    def _infer_activity_commodity_pair(self, key: Tuple[str, ...] | str) -> Optional[Tuple[str, str]]:
        """Infer an (activity, commodity) pair from a parameter key."""
        labels = [str(part) for part in (key if isinstance(key, tuple) else (key,))]

        if len(labels) >= 2:
            if labels[0] in self.i and labels[1] in self.a:
                return labels[1], labels[0]
            if labels[0] in self.a and labels[1] in self.i:
                return labels[0], labels[1]

        activity_hits = [label for label in labels if label in self.a]
        commodity_hits = [label for label in labels if label in self.i]
        if len(set(activity_hits)) == 1 and len(set(commodity_hits)) == 1:
            return activity_hits[0], commodity_hits[0]

        return None

    def _extract_output_pairs(self, gdx_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract non-zero output pairs from make-style symbols when available."""
        for symbol_name in ("makb", "maks", "x"):
            try:
                values = read_parameter_values(gdx_data, symbol_name)
            except (ValueError, FileNotFoundError):
                if self.source_gdx is None:
                    continue
                values = read_parameter_with_gdxdump(self.source_gdx, symbol_name)
                if not values:
                    continue

            pairs: List[Tuple[str, str]] = []
            for key, value in values.items():
                if abs(value) <= 1e-10:
                    continue

                pair = self._infer_activity_commodity_pair(key)
                if pair is None or pair in pairs:
                    continue
                pairs.append(pair)

            if pairs:
                return pairs

        return []

    def _set_activity_mappings(self, gdx_data: Dict[str, Any]) -> None:
        """Populate activity/commodity mappings from make structure."""
        self.i_to_a = {}
        self.a_to_i = {}
        self.output_pairs = []
        self.activity_commodities = {activity: [] for activity in self.a}
        self.commodity_activities = {commodity: [] for commodity in self.i}

        pairs = self._extract_output_pairs(gdx_data)
        if not pairs and self.is_diagonal:
            pairs = list(zip(self.a, self.i))

        self.output_pairs = pairs

        for activity, commodity in pairs:
            self.activity_commodities.setdefault(activity, [])
            self.commodity_activities.setdefault(commodity, [])

            if commodity not in self.activity_commodities[activity]:
                self.activity_commodities[activity].append(commodity)
            if activity not in self.commodity_activities[commodity]:
                self.commodity_activities[commodity].append(activity)

        if not pairs:
            return

        if (
            all(len(outputs) == 1 for outputs in self.activity_commodities.values())
            and all(len(activities) == 1 for activities in self.commodity_activities.values())
        ):
            self.a_to_i = {
                activity: outputs[0]
                for activity, outputs in self.activity_commodities.items()
                if outputs
            }
            self.i_to_a = {
                commodity: activities[0]
                for commodity, activities in self.commodity_activities.items()
                if activities
            }

    @property
    def is_diagonal(self) -> bool:
        """Whether activities and commodities share the same labels."""
        return bool(self.a) and len(self.a) == len(self.i) and set(self.a) == set(self.i)

    @property
    def has_multi_output_activities(self) -> bool:
        """Whether any activity supplies more than one commodity."""
        return any(len(outputs) > 1 for outputs in self.activity_commodities.values())

    @property
    def has_multi_source_commodities(self) -> bool:
        """Whether any commodity is supplied by more than one activity."""
        return any(len(activities) > 1 for activities in self.commodity_activities.values())

    @property
    def is_bijective_output_structure(self) -> bool:
        """Whether activities and commodities can be matched one-to-one."""
        return (
            bool(self.a_to_i)
            and bool(self.i_to_a)
            and len(self.a_to_i) == len(self.a)
            and len(self.i_to_a) == len(self.i)
            and not self.has_multi_output_activities
            and not self.has_multi_source_commodities
        )

    @property
    def structure(self) -> str:
        """High-level output structure tag."""
        if not self.a and not self.i:
            return "unloaded"
        if self.has_multi_output_activities or self.has_multi_source_commodities:
            return "multi_output"
        return "diagonal" if self.is_diagonal else "non_diagonal"
    
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
        if self.output_pairs:
            unknown_activities = {activity for activity, _ in self.output_pairs if activity not in set(self.a)}
            unknown_commodities = {commodity for _, commodity in self.output_pairs if commodity not in set(self.i)}
            if unknown_activities:
                errors.append(f"Output pairs reference unknown activities: {sorted(unknown_activities)}")
            if unknown_commodities:
                errors.append(f"Output pairs reference unknown commodities: {sorted(unknown_commodities)}")
        elif self.a and self.i and not self.is_diagonal:
            errors.append(
                "Non-diagonal GTAP structure requires make/output pairs from makb, maks, or x(a,i)"
            )
        
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
            "trade_modes_m": self.m,
            "active_margin_commodities": self.marg,
            "structure": self.structure,
            "output_pairs": self.output_pairs,
            "is_bijective_output_structure": self.is_bijective_output_structure,
            "has_multi_output_activities": self.has_multi_output_activities,
            "valid": is_valid,
            "errors": errors,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GTAPSets({self.aggregation_name}: "
            f"{self.n_regions} regions × "
            f"{self.n_commodities} commodities × "
            f"{self.n_factors} factors, structure={self.structure}, "
            f"output_pairs={len(self.output_pairs)})"
        )
