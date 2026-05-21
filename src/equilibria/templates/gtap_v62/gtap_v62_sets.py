"""GTAP v6.2 Sets — Hertel/Itakura/McDougall (2003) structure.

Differences from `templates.gtap.gtap_sets.GTAPSets` (which targets v7):

- **No ACTS set.** v6.2 uses a single index ``i ∈ TRAD_COMM`` for both
  commodities and sectors. The make matrix is implicitly diagonal.
- **SLUG binary flag** for factor mobility (read from GTAPPARM header
  "SLUG"), not the v7 ``ENDOWFLAG(e,t)`` matrix.
- **No ENDWF / ENDWMS** sets. v6.2 has only mobile (SLUG=0) and sluggish
  (SLUG=1) factors. Sector-specific (fixed) factors don't exist as a
  separate category.
- **HAR header convention is v6.2-style** (``H1``=REG, ``H2``=TRAD_COMM,
  ``H6``=ENDW_COMM, ``H9``=CGDS_COMM, ``MARG``=MARG_COMM).
- ``a`` is exposed as a property returning ``self.i`` so callers that
  still use the v7 naming convention keep working.

Reference: ``runs/gtap_v62_vs_v7/notation_crosswalk.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# v6.2 SETS.HAR header conventions. The classic GEMPACK names use
# cryptic 2-character codes (H1, H2, ...) while newer NUS333/9x10
# datasets use uppercase set names. Try both.
_REG_HEADERS = ("H1", "REG")
_TRAD_COMM_HEADERS = ("H2", "TRAD_COMM", "COMM")
_ENDW_COMM_HEADERS = ("H6", "ENDW_COMM", "ENDW")
_CGDS_COMM_HEADERS = ("H9", "CGDS_COMM", "CGDS")
_MARG_HEADERS = ("MARG", "MARG_COMM")


@dataclass
class GTAPv62Sets:
    """GTAP v6.2 model sets.

    Single-index structure (no ACT/COMM split). The bijective output
    relation is always implicit: each sector ``i ∈ TRAD_COMM`` produces
    exactly the commodity ``i``.

    Attributes:
        r: Regions (REG / H1)
        i: Traded commodities (TRAD_COMM / H2) — also the sectors
        cgds: Capital goods commodities (CGDS_COMM / H9) — usually 1 element
        f: Endowment commodities (ENDW_COMM / H6) — primary factors
        marg: Margin commodities (MARG_COMM / MARG) — subset of i
        mf: Mobile factors (SLUG=0 from GTAPPARM)
        sf: Sluggish factors (SLUG=1 from GTAPPARM)
        m: Alias of i used by the trade-margin block
        s: Alias of r used for bilateral trade

    Derived sets (Table 1 §1):
        prod_comm: TRAD_COMM ∪ CGDS_COMM
        demd_comm: ENDW_COMM ∪ TRAD_COMM
        nsav_comm: DEMD_COMM ∪ CGDS_COMM (everything except savings)

    Example:
        >>> sets = GTAPv62Sets()
        >>> sets.load_from_har(Path("C:/runGTAP375/BOOK3X3/SETS.HAR"),
        ...                    default_path=Path("C:/runGTAP375/BOOK3X3/Default.prm"))
        >>> print(sets.r)            # ['USA', 'EU', 'ROW']
        >>> print(sets.i)            # ['food', 'mnfcs', 'svces']
        >>> print(sets.f)            # ['Land', 'Labor', 'Capital']
        >>> print(sets.sf, sets.mf)  # ['Land'], ['Labor', 'Capital']
    """

    # Core sets
    r: List[str] = field(default_factory=list)
    i: List[str] = field(default_factory=list)
    cgds: List[str] = field(default_factory=list)
    f: List[str] = field(default_factory=list)
    marg: List[str] = field(default_factory=list)

    # Factor mobility partition (derived from SLUG)
    mf: List[str] = field(default_factory=list)
    sf: List[str] = field(default_factory=list)

    # Aliases
    m: List[str] = field(default_factory=list)
    s: List[str] = field(default_factory=list)

    # Metadata
    aggregation_name: str = ""
    source_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_from_har(
        self,
        sets_path: Path,
        default_path: Optional[Path] = None,
    ) -> None:
        """Load v6.2 sets from a GEMPACK SETS.HAR file.

        Args:
            sets_path: Path to SETS.HAR (e.g. ``BOOK3X3/SETS.HAR``).
            default_path: Optional path to GTAPPARM (e.g. ``Default.prm``).
                When provided, the SLUG binary header is used to split
                ``f`` into ``mf`` (SLUG=0) and ``sf`` (SLUG=1).

        Raises:
            FileNotFoundError: If sets_path doesn't exist.
            ValueError: If any required set header is missing.
        """
        from equilibria.babel.har import read_har

        if not sets_path.exists():
            raise FileNotFoundError(f"SETS.HAR not found: {sets_path}")

        data = read_har(sets_path)
        self.source_path = sets_path
        self.aggregation_name = sets_path.parent.name or sets_path.stem

        self.r = self._first_set(data, _REG_HEADERS, required=True, name="REG")
        self.i = self._first_set(data, _TRAD_COMM_HEADERS, required=True, name="TRAD_COMM")
        self.cgds = self._first_set(data, _CGDS_COMM_HEADERS, required=False, name="CGDS_COMM") or []
        self.f = self._first_set(data, _ENDW_COMM_HEADERS, required=True, name="ENDW_COMM")
        self.marg = self._first_set(data, _MARG_HEADERS, required=False, name="MARG_COMM") or []

        # Aliases follow the primary sets
        self.m = list(self.i)
        self.s = list(self.r)

        # Factor mobility split via SLUG
        if default_path is not None and default_path.exists():
            self._load_slug(default_path)

        if not self.mf and not self.sf:
            # Fallback: classic GTAP convention is Land=sluggish (sector-
            # specific land), Labor/Capital=mobile. Used when no GTAPPARM
            # is supplied.
            sluggish_hints = ("land", "natres", "natural", "lnd", "nrs")
            for fac in self.f:
                if any(hint in fac.lower() for hint in sluggish_hints):
                    self.sf.append(fac)
                else:
                    self.mf.append(fac)

    def _first_set(
        self,
        data: Dict[str, Any],
        candidate_headers: Sequence[str],
        *,
        required: bool,
        name: str,
    ) -> List[str]:
        """Return the elements of the first available header from the candidates."""
        for header in candidate_headers:
            if header in data:
                arr = data[header]
                inner = arr.array if hasattr(arr, "array") else arr
                return [str(elem).strip() for elem in inner]
        if required:
            raise ValueError(
                f"Required v6.2 set {name!r} not found. "
                f"Tried headers: {list(candidate_headers)}"
            )
        return []

    def _load_slug(self, default_path: Path) -> None:
        """Populate ``mf``/``sf`` from the binary SLUG header in GTAPPARM.

        The standard v6.2 convention (from ``gtap.tab``) is:
            ENDWS_COMM = {i ∈ ENDW_COMM : SLUG(i) > 0}

        However, some legacy datasets (notably ``BOOK3X3/Default.prm``)
        encode SLUG with values like ``[3, 1, 1]`` where every entry is
        ``> 0`` even though only Land is meant to be sluggish. To stay
        robust we apply a heuristic:

        1. If any SLUG value is exactly ``0`` → use the standard
           convention ``SLUG > 0 = sluggish``.
        2. Else if SLUG values are mixed (e.g. ``[3, 1, 1]``) → assume
           the largest values mark the sluggish factors and treat
           ``SLUG > min(SLUG)`` as the threshold.
        3. Else (all SLUG identical and > 0, e.g. ``[1, 1, 1]``) → no
           information, leave ``mf``/``sf`` empty so the caller's
           hint-based fallback kicks in.
        """
        from equilibria.babel.har import read_har

        try:
            data = read_har(default_path, select_headers=["SLUG"])
        except Exception:
            return
        if "SLUG" not in data:
            return

        slug = data["SLUG"]
        slug_arr = slug.array if hasattr(slug, "array") else slug

        flat = list(slug_arr.flatten()) if hasattr(slug_arr, "flatten") else list(slug_arr)
        values = [float(v) for v in flat[: len(self.f)]]
        if not values:
            return

        zero_present = any(abs(v) < 1e-9 for v in values)
        all_equal = len(set(values)) == 1

        if zero_present:
            threshold = 0.0  # standard convention
        elif not all_equal:
            threshold = min(values)
        else:
            # Ambiguous — let the caller fall back to hints.
            return

        for idx, factor in enumerate(self.f):
            if idx >= len(values):
                self.mf.append(factor)
                continue
            if values[idx] > threshold:
                self.sf.append(factor)
            else:
                self.mf.append(factor)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def a(self) -> List[str]:
        """Activity set — alias of ``i`` in v6.2 (no ACT/COMM split).

        Exposed so that v7-style callers (``model.a``, ``sets.a``) keep
        working without modification.
        """
        return self.i

    @property
    def prod_comm(self) -> List[str]:
        """PROD_COMM = TRAD_COMM ∪ CGDS_COMM."""
        return list(self.i) + list(self.cgds)

    @property
    def demd_comm(self) -> List[str]:
        """DEMD_COMM = ENDW_COMM ∪ TRAD_COMM."""
        return list(self.f) + list(self.i)

    @property
    def nsav_comm(self) -> List[str]:
        """NSAV_COMM = DEMD_COMM ∪ CGDS_COMM."""
        return list(self.f) + list(self.i) + list(self.cgds)

    @property
    def is_diagonal(self) -> bool:
        """Always True in v6.2 — make matrix is implicitly diagonal."""
        return True

    @property
    def n_regions(self) -> int:
        return len(self.r)

    @property
    def n_commodities(self) -> int:
        return len(self.i)

    @property
    def n_factors(self) -> int:
        return len(self.f)

    @property
    def n_mobile_factors(self) -> int:
        return len(self.mf)

    @property
    def n_sluggish_factors(self) -> int:
        return len(self.sf)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Tuple[bool, List[str]]:
        """Check internal consistency of the sets."""
        errors: List[str] = []

        if not self.r:
            errors.append("REG set is empty")
        if not self.i:
            errors.append("TRAD_COMM set is empty")
        if not self.f:
            errors.append("ENDW_COMM set is empty")

        # Factor partition must cover f exactly
        if self.f:
            partition = set(self.mf) | set(self.sf)
            f_set = set(self.f)
            if partition != f_set:
                missing = f_set - partition
                extra = partition - f_set
                if missing:
                    errors.append(f"Factors not in mf or sf: {sorted(missing)}")
                if extra:
                    errors.append(f"Extra labels in mf/sf not in f: {sorted(extra)}")
            overlap = set(self.mf) & set(self.sf)
            if overlap:
                errors.append(f"Factors in both mf and sf: {sorted(overlap)}")

        # MARG must be subset of i
        if self.marg:
            extra = set(self.marg) - set(self.i)
            if extra:
                errors.append(f"MARG_COMM not in TRAD_COMM: {sorted(extra)}")

        return len(errors) == 0, errors

    def get_info(self) -> Dict[str, Any]:
        """Summary dict for logging / debugging."""
        is_valid, errors = self.validate()
        return {
            "aggregation": self.aggregation_name,
            "source": str(self.source_path) if self.source_path else None,
            "n_regions": self.n_regions,
            "n_commodities": self.n_commodities,
            "n_factors": self.n_factors,
            "n_mobile_factors": self.n_mobile_factors,
            "n_sluggish_factors": self.n_sluggish_factors,
            "regions": self.r,
            "commodities": self.i,
            "cgds": self.cgds,
            "factors": self.f,
            "mobile_factors": self.mf,
            "sluggish_factors": self.sf,
            "margins": self.marg,
            "version": "6.2",
            "valid": is_valid,
            "errors": errors,
        }

    def __repr__(self) -> str:
        return (
            f"GTAPv62Sets({self.aggregation_name}: "
            f"{self.n_regions}r × {self.n_commodities}i × {self.n_factors}f, "
            f"mf={self.n_mobile_factors}, sf={self.n_sluggish_factors})"
        )
