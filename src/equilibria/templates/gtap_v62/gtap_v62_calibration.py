"""GTAP v6.2 calibration — derive tax rates and aggregate values from SAM.

In the v6.2 GEMPACK convention, tax rates are *implicit* in the basedata
SAM: agents pay a price that includes the tax (V*A headers), while
producers receive the market price (V*M headers). The ad-valorem rate is

    rate = V*A / V*M - 1

This module computes those derived rates, plus a few aggregate value
flows (VOM = value of output at market prices, VOA = value of output at
agent prices, EVOM = factor income at market prices) that the model
equations need.

The derivations are pure functions of the loaded ``GTAPv62Parameters``
benchmark; they don't touch Pyomo at all. The result is a flat
:class:`DerivedV62Calibration` dataclass that the model builder
consumes during ``_add_parameters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from equilibria.templates.gtap_v62.gtap_v62_parameters import GTAPv62Parameters
from equilibria.templates.gtap_v62.gtap_v62_sets import GTAPv62Sets


@dataclass
class DerivedV62Calibration:
    """Tax rates, aggregate values, and calibrated shares from the SAM."""

    # Tax rates on intermediate inputs (firm side)
    tfd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (i, j, r)
    tfi: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (i, j, r)
    # Tax rates on household final demand
    tpd: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    tpi: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    # Tax rates on government final demand
    tgd: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    tgi: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    # Factor tax rates
    tf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)   # (f, j, r)
    # Trade tax rates
    tms: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (i, s, r) — tariff
    txs: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # (i, s, r) — export tax
    # Output tax — derived per-region per-sector
    to: Dict[Tuple[str, str], float] = field(default_factory=dict)        # (i, r)

    # Aggregate value flows
    vom: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (j, r) market price output (output-side)
    voa: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (j, r) agent/producer price output
    vop: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (j, r) production cost base (= VA + intermediates)
    evom: Dict[Tuple[str, str], float] = field(default_factory=dict)      # (f, r) factor income market
    # Value-added aggregate per sector
    va_total: Dict[Tuple[str, str], float] = field(default_factory=dict)  # (j, r) total VA at market prices
    # Domestic supply absorbed locally (sum over agents of domestic demand)
    vds: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    # Total imports at market and world prices
    vim: Dict[Tuple[str, str], float] = field(default_factory=dict)       # (i, r)
    vimw: Dict[Tuple[str, str], float] = field(default_factory=dict)      # (i, r)

    # ---- Production-block calibrated shares (Phase 2b) -----------------
    # These describe the CES nest structure of each sector.

    # Top nest (CES between VA and aggregate intermediate composite).
    # share_va(j,r) + sum_i share_int(i,j,r) = 1 by construction.
    share_va: Dict[Tuple[str, str], float] = field(default_factory=dict)
    share_int: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    # VA nest (CES across factors). sum_f share_fac(f,j,r) = 1 by construction.
    share_fac: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    # Top Armington nest within each commodity input (CES between
    # domestic and imported).
    #
    # ``share_dom`` and ``share_imp`` are the **value shares** at
    # benchmark (VDFM and VIFM each over their sum). They sum to 1 by
    # construction.
    #
    # ``alpha_dom`` and ``alpha_imp`` are the calibrated **distribution
    # parameters** that go into the CES first-order conditions. They
    # absorb the benchmark price ratio so that
    # ``qfd = alpha_dom * qf * (pf_int / pfd)^σ`` is identically true at
    # the benchmark with non-unit agent prices (because of tax wedges).
    # They are computed from value shares via
    #     alpha_d = share_dom * (pfd_0 / pf_int_0)^σ
    # and similarly for alpha_m.
    #
    # ``pf_int_0`` is the benchmark CES composite Armington price for
    # firm intermediate input ``i`` in sector ``j`` of region ``r``.
    share_dom: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    share_imp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    alpha_dom: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    alpha_imp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pf_int_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)


def derive_calibration(
    sets: GTAPv62Sets,
    params: GTAPv62Parameters,
) -> DerivedV62Calibration:
    """Compute derived tax rates and aggregate value flows.

    Tax rates use the canonical GEMPACK convention:

        rate = V*A / V*M - 1     (so V*A = (1 + rate) * V*M)

    When ``V*M`` is zero (no flow), the rate defaults to zero.

    Args:
        sets: GTAP v6.2 sets (regions, commodities, factors, ...).
        params: Loaded benchmark + elasticities.

    Returns:
        :class:`DerivedV62Calibration` with all derived tables filled in.
    """
    b = params.benchmark
    out = DerivedV62Calibration()

    # ------------------------------------------------------------------
    # Tax rates from agent/market price differentials
    # ------------------------------------------------------------------

    out.tfd = _derived_rate_3d(b.vdfa, b.vdfm)
    out.tfi = _derived_rate_3d(b.vifa, b.vifm)
    out.tpd = _derived_rate_2d(b.vdpa, b.vdpm)
    out.tpi = _derived_rate_2d(b.vipa, b.vipm)
    out.tgd = _derived_rate_2d(b.vdga, b.vdgm)
    out.tgi = _derived_rate_2d(b.viga, b.vigm)

    # Factor tax: EVFA / VFM - 1
    out.tf = _derived_rate_3d(b.evfa, b.vfm)

    # Trade rates:
    #   tms(i,s,r) = VIMS(i,s,r) / VIWS(i,s,r) - 1   (importer tariff)
    #   txs(i,s,r) = VXMD(i,s,r) / VXWD(i,s,r) - 1   (exporter tax/subsidy)
    out.tms = _derived_rate_3d(b.vims, b.viws)
    out.txs = _derived_rate_3d(b.vxmd, b.vxwd)

    # ------------------------------------------------------------------
    # Aggregate value flows: VOM, VOA, EVOM, VDS, VIM
    # ------------------------------------------------------------------

    # VOM(i,r) at MARKET prices = total use of commodity i in region r,
    # split into domestic absorption + exports.
    for r in sets.r:
        for i in sets.i:
            # Intermediate use by all sectors (domestic + imported go to
            # different agents but VOM is on the *output* side, so we only
            # count domestic absorption here).
            uses_dom = sum(
                b.vdfm.get((i, j, r), 0.0) for j in sets.prod_comm
            ) + b.vdpm.get((i, r), 0.0) + b.vdgm.get((i, r), 0.0)

            exports = sum(b.vxmd.get((i, r, rp), 0.0) for rp in sets.r if rp != r)
            margin_sales = b.vst.get((i, r), 0.0)

            out.vom[(i, r)] = uses_dom + exports + margin_sales

        # Capital goods: CGDS output = sum of intermediate inputs (VDFM/VIFM into cgds sector)
        for c in sets.cgds:
            cgds_use_dom = sum(b.vdfm.get((i, c, r), 0.0) for i in sets.i)
            cgds_use_imp = sum(b.vifm.get((i, c, r), 0.0) for i in sets.i)
            out.vom[(c, r)] = cgds_use_dom + cgds_use_imp

        # Factor outputs from EVOA header (factors are produced by households)
        for f in sets.f:
            out.evom[(f, r)] = b.evoa.get((f, r), 0.0)

    # VOA(i,r) at PRODUCER prices = VOM * (1 + to). For now, set VOA = VOM
    # and let to default to zero. The output tax can be calibrated from
    # a separate header (the v6.2 SAM stores it implicitly in the
    # difference between EVFA/VFM at the column total level).
    for key, val in out.vom.items():
        out.voa[key] = val
        out.to[key] = 0.0

    # Domestic absorption (used in market clearing identity)
    for r in sets.r:
        for i in sets.i:
            absorption = (
                sum(b.vdfm.get((i, j, r), 0.0) for j in sets.prod_comm)
                + b.vdpm.get((i, r), 0.0)
                + b.vdgm.get((i, r), 0.0)
            )
            out.vds[(i, r)] = absorption

    # Total imports (market and world price)
    for r in sets.r:
        for i in sets.i:
            out.vim[(i, r)] = sum(
                b.vims.get((i, s, r), 0.0) for s in sets.r if s != r
            )
            out.vimw[(i, r)] = sum(
                b.viws.get((i, s, r), 0.0) for s in sets.r if s != r
            )

    # ------------------------------------------------------------------
    # Production-block calibrated shares
    # ------------------------------------------------------------------

    # VA total per sector (at market prices, used as the calibration
    # value for the value-added aggregate variable).
    for r in sets.r:
        for j in sets.prod_comm:
            va = sum(b.vfm.get((f, j, r), 0.0) for f in sets.f)
            out.va_total[(j, r)] = va

    # Top nest shares: VA share and intermediate-by-commodity share.
    # Denominator (``vop``) is the production cost base — total
    # purchases at MARKET prices. This is the price-taking unit cost
    # before output tax. The output-side aggregate (``vom``) may
    # differ slightly: ``to = vom / vop - 1`` is the implicit output
    # tax/subsidy. ``qo`` is initialized at ``vop`` so the production
    # equations balance exactly at benchmark.
    for r in sets.r:
        for j in sets.prod_comm:
            va_val = out.va_total.get((j, r), 0.0)
            int_total = 0.0
            int_vals: Dict[str, float] = {}
            for i in sets.i:
                vfa = b.vdfm.get((i, j, r), 0.0) + b.vifm.get((i, j, r), 0.0)
                int_vals[i] = vfa
                int_total += vfa
            base = va_val + int_total
            if base <= 0.0:
                continue
            out.vop[(j, r)] = base
            out.share_va[(j, r)] = va_val / base
            for i in sets.i:
                out.share_int[(i, j, r)] = int_vals[i] / base

    # Recompute implicit output tax: to = vom / vop - 1.
    # When vop is missing (sector with no production cost), fall back to 0.
    for key, vop_val in out.vop.items():
        if vop_val > 0.0:
            vom_val = out.vom.get(key, vop_val)
            out.to[key] = vom_val / vop_val - 1.0

    # VA nest factor shares.
    for r in sets.r:
        for j in sets.prod_comm:
            va_val = out.va_total.get((j, r), 0.0)
            if va_val <= 0.0:
                continue
            for f in sets.f:
                vf = b.vfm.get((f, j, r), 0.0)
                out.share_fac[(f, j, r)] = vf / va_val

    # Top Armington shares per (commodity, sector, region):
    # share of domestic vs imported in firm intermediate use.
    # Also computes the **distribution parameters** alpha_dom /
    # alpha_imp that absorb the benchmark agent-price ratios so the CES
    # first-order conditions hold identically.
    for r in sets.r:
        for j in sets.prod_comm:
            for i in sets.i:
                vd = b.vdfm.get((i, j, r), 0.0)
                vm = b.vifm.get((i, j, r), 0.0)
                total = vd + vm
                if total <= 0.0:
                    continue
                sd = vd / total
                sm = vm / total
                out.share_dom[(i, j, r)] = sd
                out.share_imp[(i, j, r)] = sm

                # Benchmark agent prices.
                pfd0 = (1.0 + out.to.get((i, r), 0.0)) * (
                    1.0 + out.tfd.get((i, j, r), 0.0)
                )
                pfm0 = 1.0 + out.tfi.get((i, j, r), 0.0)

                # Benchmark composite price (cost-weighted average over qf):
                #     pf_int_0 * qf_0 = pfd_0 * vd + pfm_0 * vm
                pf_int0 = (pfd0 * vd + pfm0 * vm) / total
                out.pf_int_0[(i, j, r)] = pf_int0

                # Distribution parameters for CES.
                # σ_d defaults to ESBD(i); the CD-perturbation (σ → 1+1e-3)
                # is applied at equation-construction time inside the
                # model builder, so we use σ_d directly here.
                sigma_d = float(params.elasticities.esubd.get(i, 1.0))
                if abs(sigma_d - 1.0) < 1e-8:
                    sigma_d = 1.0 + 1e-3
                if pf_int0 > 0.0:
                    out.alpha_dom[(i, j, r)] = sd * (pfd0 / pf_int0) ** sigma_d
                    out.alpha_imp[(i, j, r)] = sm * (pfm0 / pf_int0) ** sigma_d
                else:
                    out.alpha_dom[(i, j, r)] = sd
                    out.alpha_imp[(i, j, r)] = sm

    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _derived_rate_2d(
    agent: Dict[Tuple[str, str], float],
    market: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """Compute (agent/market - 1) for every key present in either dict.

    Where the market value is zero or missing, the rate defaults to 0.
    """
    out: Dict[Tuple[str, str], float] = {}
    keys = set(agent.keys()) | set(market.keys())
    for key in keys:
        m = market.get(key, 0.0)
        a = agent.get(key, 0.0)
        if abs(m) < 1e-12:
            out[key] = 0.0
        else:
            out[key] = a / m - 1.0
    return out


def _derived_rate_3d(
    agent: Dict[Tuple[str, str, str], float],
    market: Dict[Tuple[str, str, str], float],
) -> Dict[Tuple[str, str, str], float]:
    """3-d variant of :func:`_derived_rate_2d`."""
    out: Dict[Tuple[str, str, str], float] = {}
    keys = set(agent.keys()) | set(market.keys())
    for key in keys:
        m = market.get(key, 0.0)
        a = agent.get(key, 0.0)
        if abs(m) < 1e-12:
            out[key] = 0.0
        else:
            out[key] = a / m - 1.0
    return out
