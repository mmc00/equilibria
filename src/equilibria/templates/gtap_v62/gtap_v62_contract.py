"""GTAP v6.2 Contract — closure and equation configurations.

Mirrors the v7 contract API (``templates.gtap.gtap_contract``) but
configured for the v6.2 structure:

- Numeraire defaults to ``pgdpwld`` (world GDP price index) — the
  classic v6.2 choice. v7 introduced the ``pnum`` numeraire alongside
  the Tornqvist ``pmuv``.
- Fixed-variable list **omits** v7-only objects: ``ifSUB``-related
  dual prices (``pdp``, ``pmp``, ``pfa``, ``pfy``), the activity-level
  factor income tax ``tinc``, the make-matrix transformation
  parameters (``ETRAQ``, ``ESUBQ``), and v7's intermediate bundle
  shifter ``lambdaN``.
- Endogenous list excludes the v7 dynamic investment accounting
  (``gblValNetInv``, ``chiInv``) when running the canonical v6.2
  comparative-static closure. RORDELTA is honored via ``RDLT``.

Reference: ``gtap_v62.tab`` (Hertel/Itakura/McDougall 2003), ``Default.prm``,
and the BOOK3X3/NUS333 closure files (``standard.cls``, ``altertax.cls``).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import Field, field_validator

from equilibria.contracts import (
    ModelBoundsConfig,
    ModelClosureConfig,
    ModelContract,
    ModelEquationConfig,
)


# ----------------------------------------------------------------------
# Equation IDs registered by the v6.2 model.
#
# These mirror the v6.2 GEMPACK equation set with the GAMS-style
# ``e_*`` naming convention used in equilibria. Phase 2b/2c will
# actually wire them into Pyomo Constraint objects; for now this
# list is the contract: it documents the intended equation set
# and is what closure validation checks against.
# ----------------------------------------------------------------------

_V62_PRODUCTION = (
    # Top production nest (CES between VA and aggregate intermediates,
    # default σ_top = 0 → Leontief in v6.2)
    "e_qo",         # qo = activity output identity
    "e_ps",         # ps = unit cost of production (zero-profit)
    "e_qf",         # qf(i,j,r) firm intermediate demand (Leontief)
    "e_pf",         # pf(i,j,r) firm intermediate composite price
    "e_qva",        # qva(j,r) value-added demand
    "e_pva",        # pva(j,r) VA composite price
    "e_qfe",        # qfe(f,j,r) factor demand within VA (CES, σ=esubva)
    "e_pfe",        # pfe(f,j,r) factor price within VA
)

_V62_FINAL_DEMAND = (
    # Private household — CDE (uses INCP and SUBP)
    "e_qpd",        # qpd(i,r) household domestic demand
    "e_qpm",        # qpm(i,r) household import demand
    "e_qp",         # qp(i,r) Armington composite
    "e_pp",         # pp(i,r) Armington composite price
    "e_pq",         # pq(i,r) CDE price aggregator
    "e_up",         # up(r)   private utility (CDE)
    "e_yp",         # yp(r)   household income
    # Government — Cobb-Douglas (§7 v6.2 has no ESUBG)
    "e_qgd",        # qgd(i,r) gov domestic demand
    "e_qgm",        # qgm(i,r) gov import demand
    "e_qg",         # qg(i,r)  gov Armington composite
    "e_pg",         # pg(i,r)  Armington composite gov price
    "e_pgov",       # pgov(r)  gov price index (CD)
    "e_ug",         # ug(r)    gov utility
    "e_yg",         # yg(r)    gov income
    # Investment — § v6.2: investment is a sector cgds (no agent split)
    "e_qcgds",      # qcgds(r) investment good output
    "e_pcgds",      # pcgds(r) investment good price
    "e_qfd_cgds",   # qfd(i, cgds, r) — inv intermediates via sector cgds
    "e_qfm_cgds",   # qfm(i, cgds, r)
)

_V62_TRADE = (
    # Top Armington (§4 single-region σ)
    "e_qfd_arm",    # qfd from top Armington composite (CES σ=esubd)
    "e_qfm_arm",    # qfm from top Armington
    "e_qfa",        # qfa(i,j,r) Armington composite firm demand
    "e_pfa",        # pfa(i,j,r) Armington composite firm price
    # Bottom Armington (§9 cross-source CES σ=esubm)
    "e_qxs",        # qxs(i,s,r) bilateral imports
    "e_pms",        # pms(i,s,r) bilateral market price (importer side)
    "e_pmcif",      # pmcif(i,s,r) CIF price
    "e_pe",         # pe(i,r,s) FOB price (exporter side)
    "e_pim",        # pim(i,r) composite import price
    # Export / output balance (v6.2 has no CET; pure allocation balance)
    "e_qds",        # qds(i,r) domestic absorption (output minus exports minus stocks)
)

_V62_MARGINS = (
    # Cobb-Douglas margins (§10 — v6.2 has no ESUBS)
    "e_qst",        # qst(m,r) margin sale supply
    "e_pst",        # pst(m,r) margin sale price
    "e_qtm",        # qtm(m) world demand for margin service m
    "e_ptmg",       # ptmg(m) world margin price (CD aggregator)
    "e_pwmg",       # pwmg(i,r,s) bilateral margin cost
    "e_qtmfsd",     # qtmfsd(m,i,r,s) per-shipment margin demand
)

_V62_FACTOR_MARKETS = (
    # Sluggish factors with CET (§12 — at commodity level, not activity)
    "e_qoes",       # qoes(i,r) sluggish factor allocation across sectors
    "e_pmes",       # pmes(i,r) sluggish factor wage by sector
    "e_pm_endw",    # pm(i,r) for ENDW_COMM — factor regional price
    # Mobile factor wages and supply
    "e_qe",         # qe(f,r) total factor supply
    "e_pe_endw",    # pe(f,r) factor wage (mobile factors)
)

_V62_INCOME_AND_CLOSURE = (
    # Regional income and savings
    "e_y",          # y(r) regional income (sum of factor earnings + taxes - subsidies)
    "e_ysav",       # ysav(r) savings (= save header at benchmark)
    "e_psave",      # psave(r) savings price (depends on chiInv)
    "e_rorg",       # rorg = global rate of return (when RORDELTA active)
    # Capital accumulation
    "e_kb",         # kb(r) beginning-of-period capital
    "e_ke",         # ke(r) end-of-period capital
    "e_walras",     # global Walras check
    # Numeraire
    "e_pgdpwld",    # pgdpwld = 1 (numeraire identity)
    # Tax revenue aggregator (replaces v7 ytax(r,gy,t) stream split)
    "e_taxrev",     # tax revenue per region (single aggregate in v6.2)
    # GDP identities
    "e_gdpmp",      # gdpmp(r) nominal GDP at market prices
    "e_rgdpmp",     # rgdpmp(r) real GDP
    "e_pgdpmp",     # pgdpmp(r) GDP deflator
)


def _full_gtap_v62_equation_ids() -> Tuple[str, ...]:
    """Return every v6.2 equation ID registered by the contract."""
    return (
        _V62_PRODUCTION
        + _V62_FINAL_DEMAND
        + _V62_TRADE
        + _V62_MARGINS
        + _V62_FACTOR_MARKETS
        + _V62_INCOME_AND_CLOSURE
    )


# ----------------------------------------------------------------------
# Closure config
# ----------------------------------------------------------------------


class GTAPv62ClosureConfig(ModelClosureConfig):
    """Economic closure for GTAP v6.2 model.

    v6.2 standard closure (from ``BOOK3X3/standard.cls``):

    - **Exogenous**: tax rates, technology shifters, factor endowments,
      population, trade margin shares.
    - **Endogenous**: prices, quantities, savings, investment, factor
      allocations.
    - **Numeraire**: ``pgdpwld = 1`` (world GDP price index).
    - **Capital**: globally equal expected rates of return
      (``RORDELTA = 1`` in v6.2 ``RDLT`` header).
    - **Margins**: Cobb-Douglas demand for trade and transport.

    The ``altertax`` and ``trade_policy`` variants follow the same
    pattern as the v7 contract: see :func:`build_gtap_v62_contract`.
    """

    name: str = "gtap_v62_standard"
    numeraire: str = "pgdpwld"
    numeraire_mode: Literal["fixed_benchmark"] = "fixed_benchmark"
    closure_type: Literal["CNS", "MCP"] = "CNS"
    capital_mobility: Literal["mobile", "sluggish"] = "mobile"
    rordelta: bool = True
    if_sub: bool = False  # v6.2 has no ifSUB macro
    calibration_source: str = "python"
    calibration_dump: Optional[str] = None
    apply_flag_fixing: bool = True
    close_mcp_gap: bool = False

    # Closure flags
    fix_taxes: bool = True
    fix_technology: bool = True
    fix_endowments: bool = True
    fix_world_prices: bool = False

    fixed: Tuple[str, ...] = Field(
        default_factory=lambda: (
            # Taxes (v6.2 tax names)
            "to",      # output tax
            "tp",      # private consumption tax (general)
            "tg",      # government consumption tax (general)
            "tf",      # factor tax (general)
            "tm",      # import tariff (powers)
            "tms",     # import tariff (bilateral, GEMPACK style)
            "tx",      # export tax (general)
            "txs",     # export tax (bilateral)
            "tpd", "tpi",   # private domestic/import-specific
            "tgd", "tgi",   # gov domestic/import-specific
            "tfd", "tfi",   # firm domestic/import-specific
            # Technology / productivity shifters (v6.2 names from gtap.tab)
            "aosec", "aoreg", "aoall",
            "avasec", "avareg",
            "afcom", "afsec", "afreg", "afall",
            "afecom", "afesec", "afereg", "afeall",
            "ams", "atm", "atf", "ats", "atd",
            # Population and structural slacks
            "pop",
            "psaveslack", "pfactwld",
            "profitslack", "incomeslack", "endwslack",
            "cgdslack", "tradslack",
            # Hicks-neutral demand shifters
            "au", "dppriv", "dpgov", "dpsave",
            # Trade margin shares (calibrated, fixed)
            "amgm",
        )
    )

    endogenous: Tuple[str, ...] = Field(
        default_factory=lambda: (
            # Savings, investment, factor allocation, prices
            "psave", "qcgds", "rorg",
            "yi", "yg", "y", "yp",
            # Standard endogenous variables
            "qo", "ps", "qfe", "pfe", "qf", "pf", "qva", "pva",
            "qp", "qg", "qpd", "qpm", "qgd", "qgm",
            "qfd", "qfm", "qxs", "pms", "pmcif", "pe",
            "qst", "qtm", "ptmg",
        )
    )

    # World price index numeraire — v6.2 typical alternatives are
    # pgdpwld (world GDP price), pfactwld (world factor price), or
    # pmuv (Tornqvist manufactures unit value). Default is pgdpwld.
    pgdpwld_basket_regions: Tuple[str, ...] = Field(default_factory=tuple)
    pgdpwld_basket_commodities: Tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Closure name must be non-empty.")
        return text

    @field_validator("closure_type", mode="before")
    @classmethod
    def _normalize_closure_type(cls, value: Any) -> str:
        text = str(value).strip().upper()
        if text not in ("CNS", "MCP"):
            raise ValueError(f"closure_type must be 'CNS' or 'MCP'; got {value!r}")
        return text

    @field_validator("calibration_source", mode="before")
    @classmethod
    def _normalize_calibration_source(cls, value: Any) -> str:
        text = str(value or "python").strip().lower()
        if text in {"python", "gempack"}:
            return text
        raise ValueError(
            "calibration_source must be 'python' or 'gempack' for v6.2"
        )


# ----------------------------------------------------------------------
# Equation config
# ----------------------------------------------------------------------


class GTAPv62EquationConfig(ModelEquationConfig):
    """v6.2 equation system selection."""

    name: str = "full_gtap_v62"
    include: Tuple[str, ...] = Field(default_factory=_full_gtap_v62_equation_ids)
    activation_masks: Literal["gtap_v62_standard", "all_active"] = "gtap_v62_standard"

    @field_validator("activation_masks", mode="before")
    @classmethod
    def _normalize_activation_masks(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("activation_masks must be non-empty.")
        return text


# ----------------------------------------------------------------------
# Contract wrapper and factory
# ----------------------------------------------------------------------


class GTAPv62BoundsConfig(ModelBoundsConfig):
    """Domain/bounds policy for GTAP v6.2 variables.

    Notable differences from v7's :class:`GTAPBoundsConfig`:

    - Drops v7-only ``valFobCif``, ``ytax``, ``ytaxTot``, ``bopSlack``.
    - Keeps ``walras``, ``savf`` as the canonical free variables.
    - Lower bound defaults to ``1e-6`` (same as v7) for positive
      quantity variables.
    """

    name: str = "economic_v62"
    positive: Literal["lower_only", "both_bounds"] = "lower_only"
    fixed_from_closure: bool = True
    free: Tuple[str, ...] = Field(
        default_factory=lambda: (
            "savf", "walras", "v_obje",
            # v6.2 also has explicit Walras slack — kept free for diagnostics
            "tradslack", "endwslack", "cgdslack",
        )
    )
    lower_bound: float = 1e-6
    upper_bound: Optional[float] = None


class GTAPv62Contract(ModelContract):
    """Complete v6.2 contract: closure + equations + bounds."""

    closure: GTAPv62ClosureConfig = Field(default_factory=GTAPv62ClosureConfig)
    equations: GTAPv62EquationConfig = Field(default_factory=GTAPv62EquationConfig)
    bounds: GTAPv62BoundsConfig = Field(default_factory=GTAPv62BoundsConfig)


def _closure_for(name: str) -> GTAPv62ClosureConfig:
    """Build a closure config for one of the canonical v6.2 names."""
    canonical = name.strip().lower()
    base_default = GTAPv62ClosureConfig()  # snapshot the default fixed/endogenous lists

    if canonical in ("gtap_v62_standard", "gtap_standard", "standard"):
        return GTAPv62ClosureConfig(name="gtap_v62_standard")

    if canonical in ("altertax", "alt_tax", "altertax_v62"):
        # Altertax: all elasticities set to unity (CD) so a tax change
        # only re-balances the SAM, leaving quantities unchanged.
        # Tax rates become endogenous; the post-solve SAM is the new
        # baseline.
        return GTAPv62ClosureConfig(
            name="altertax",
            fix_taxes=False,
            fixed=base_default.fixed,
            endogenous=base_default.endogenous,
        )

    if canonical in ("trade_policy", "tariff", "trade"):
        # Allow tariffs and export taxes to vary (drop them from fixed).
        relaxed_fixed = tuple(
            f for f in base_default.fixed
            if f not in {"tm", "tms", "tx", "txs"}
        )
        return GTAPv62ClosureConfig(
            name="trade_policy",
            fix_taxes=False,
            fixed=relaxed_fixed,
            endogenous=base_default.endogenous,
        )

    raise ValueError(f"Unknown v6.2 closure name: {name!r}")


def build_gtap_v62_contract(closure_name: str = "gtap_v62_standard") -> GTAPv62Contract:
    """Build a complete GTAP v6.2 contract.

    Args:
        closure_name: One of ``"gtap_v62_standard"``, ``"altertax"``,
            ``"trade_policy"``.

    Returns:
        A :class:`GTAPv62Contract` with the closure, equations, and
        default bounds wired together.
    """
    closure = _closure_for(closure_name)
    equations = GTAPv62EquationConfig()
    return GTAPv62Contract(
        name=f"gtap_v62:{closure_name}",
        closure=closure,
        equations=equations,
    )


def default_gtap_v62_contract() -> GTAPv62Contract:
    """Return the standard v6.2 contract."""
    return build_gtap_v62_contract("gtap_v62_standard")


__all__ = [
    "GTAPv62Contract",
    "GTAPv62ClosureConfig",
    "GTAPv62EquationConfig",
    "GTAPv62BoundsConfig",
    "build_gtap_v62_contract",
    "default_gtap_v62_contract",
]
