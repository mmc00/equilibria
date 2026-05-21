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
    # domestic and imported). See firm-side notes; same scheme applies
    # to households (``alpha_dom_hhd``) and government
    # (``alpha_dom_gov``).
    share_dom: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    share_imp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    alpha_dom: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    alpha_imp: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pf_int_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    # ---- Household Armington (CES) + CD allocation across goods ---------
    #
    # Phase 2c.1 uses a Cobb-Douglas demand allocation across goods (a
    # simplification of v6.2's CDE; full CDE port deferred to Phase 2d).
    # The Armington between domestic and imported uses ESBD just like
    # firm intermediates.
    alpha_dom_hhd: Dict[Tuple[str, str], float] = field(default_factory=dict)
    alpha_imp_hhd: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pp_0: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # CD budget share of household total expenditure on good i in r
    share_hhd_cd: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # Benchmark household total consumption expenditure (= sum_i VDPA+VIPA)
    yp_0: Dict[str, float] = field(default_factory=dict)

    # ---- Government Armington (CES) + CD allocation across goods --------
    alpha_dom_gov: Dict[Tuple[str, str], float] = field(default_factory=dict)
    alpha_imp_gov: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pg_0: Dict[Tuple[str, str], float] = field(default_factory=dict)
    share_gov_cd: Dict[Tuple[str, str], float] = field(default_factory=dict)
    yg_0: Dict[str, float] = field(default_factory=dict)

    # ---- Trade block (Phase 2c.2) ---------------------------------------
    # Benchmark prices along the export → import chain. At benchmark
    # with ps_0 = 1 (basic supply):
    #     pe_0(i,s,d)    = 1 + txs(i,s,d)         (FOB)
    #     pwmg_0(i,s,d)  = sum_m VTWR/VXWD        (per-unit transport)
    #     pmcif_0(i,s,d) = pe_0 + pwmg_0          (CIF)
    #     pms_0(i,s,d)   = pmcif_0 * (1 + tms)    (market price at d)
    pe_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pwmg_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pmcif_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pms_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    qxs_0: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    # CES import composite (bottom Armington across sources):
    # pim_0 = cost-weighted avg of pms_0 across sources
    # alpha_xs(i,s,d) = (qxs_0_s / qim_0) * (pms_0_s / pim_0)^σ_m
    pim_0: Dict[Tuple[str, str], float] = field(default_factory=dict)
    qim_0: Dict[Tuple[str, str], float] = field(default_factory=dict)
    alpha_xs: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    # ---- Margins block (Phase 2c.2) -------------------------------------
    # Margin commodity benchmark sales and demand
    qst_0: Dict[Tuple[str, str], float] = field(default_factory=dict)   # margin sales (m, r)
    qtm_0: Dict[str, float] = field(default_factory=dict)               # world margin demand (m)
    ptmg_0: Dict[str, float] = field(default_factory=dict)              # world margin price (m)
    # CD share for ptmg = prod_r pst^share_st(m,r)
    share_st: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # Per-shipment margin cost share: amgm(m,i,s,d) = VTWR(m,i,s,d) / sum_m VTWR(...)
    amgm: Dict[Tuple[str, str, str, str], float] = field(default_factory=dict)

    # ---- Market clearing aggregates (Phase 2c.2) ------------------------
    # Domestic absorption sum: VDS(i,r) = VDPM + VDGM + sum_j VDFM (already
    # in ``vds``). Margin commodity total: sum across regions of vst(m, r).


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

    # Composite import benchmark price pim_0(i, r): aggregate ratio of
    # imports at agent (pms) prices to basic-price quantities. Needed
    # *early* so the household/government/firm Armington calibrations
    # can compute their imported-side agent prices consistently:
    #     pfm_0 = pim_0 * (1 + tfi)
    #     ppm_0 = pim_0 * (1 + tpi)
    #     pgm_0 = pim_0 * (1 + tgi)
    # When there is no import flow into (i,r), pim_0 defaults to 1.
    for r in sets.r:
        for i in sets.i:
            sum_vims = sum(b.vims.get((i, s, r), 0.0) for s in sets.r if s != r)
            sum_vxwd = sum(b.vxwd.get((i, s, r), 0.0) for s in sets.r if s != r)
            if sum_vxwd > 0.0:
                out.pim_0[(i, r)] = sum_vims / sum_vxwd
            else:
                out.pim_0[(i, r)] = 1.0

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

    # ------------------------------------------------------------------
    # Household and government Armington calibration (Phase 2c.1)
    # ------------------------------------------------------------------

    def _armington_calibrate(
        vdm: float,
        vim: float,
        vda: float,
        via: float,
        sigma: float,
        pds_0: float = 1.0,
        pim_0: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Return ``(alpha_d, alpha_m, p_int_0)`` for an Armington nest.

        Calibrated so that the CES first-order conditions hold
        identically at benchmark with agent prices

            ``pdom_0 = pds_0 * (vda / vdm)``
            ``pimp_0 = pim_0 * (via / vim)``

        That is, the agent price equals the basic supply price times
        ``(1 + agent_tax_rate)`` where ``(1 + rate)`` is read from the
        SAM ratio of agent-price value to market-price value.

        ``pds_0`` and ``pim_0`` are the basic prices seen by the
        downstream agent (default 1.0 for the trivial case). In v6.2
        with the standard convention, the basic supply price for
        domestic commodities is ``pds = ps * (1 + to)`` so
        ``pds_0 = 1 + to`` when ``ps_0 = 1``.
        """
        total_basic = vdm + vim
        if total_basic <= 0.0:
            return 0.0, 0.0, 1.0
        # Agent-price wedges (just the agent-specific tax rate)
        wedge_d = vda / vdm if vdm > 0.0 else 1.0
        wedge_m = via / vim if vim > 0.0 else 1.0
        pdom_0 = pds_0 * wedge_d
        pimp_0 = pim_0 * wedge_m
        total_agent = pdom_0 * vdm + pimp_0 * vim  # not vda+via because we include pds_0/pim_0 wedge
        # cost-weighted benchmark composite (basic-qty divisor)
        p_int0 = total_agent / total_basic
        share_d = vdm / total_basic
        share_m = vim / total_basic
        sigma_eff = sigma if abs(sigma - 1.0) > 1e-8 else 1.0 + 1e-3
        if p_int0 > 0.0:
            ad = share_d * (pdom_0 / p_int0) ** sigma_eff
            am = share_m * (pimp_0 / p_int0) ** sigma_eff
        else:
            ad, am = share_d, share_m
        return ad, am, p_int0

    for r in sets.r:
        yp_total = 0.0
        yg_total = 0.0
        for i in sets.i:
            sigma_d = float(params.elasticities.esubd.get(i, 1.0))
            # Basic prices for downstream agents:
            # - pds_0 = 1 + to (output tax wedge on domestic supply)
            # - pim_0 = sum_s VIMS / sum_s VXWD (composite import; precomputed
            #   above before the Armington calibrations).
            pds0_i = 1.0 + out.to.get((i, r), 0.0)
            pim0_i = out.pim_0.get((i, r), 1.0)

            # Household
            vd_h, vm_h = b.vdpm.get((i, r), 0.0), b.vipm.get((i, r), 0.0)
            va_h, vai_h = b.vdpa.get((i, r), 0.0), b.vipa.get((i, r), 0.0)
            ad_h, am_h, pp0 = _armington_calibrate(vd_h, vm_h, va_h, vai_h, sigma_d, pds0_i, pim0_i)
            out.alpha_dom_hhd[(i, r)] = ad_h
            out.alpha_imp_hhd[(i, r)] = am_h
            out.pp_0[(i, r)] = pp0
            # Household expenditure at agent prices includes the output-tax
            # wedge carried by pds_0 (since pds = ps*(1+to) and VDPM is at
            # pds level). Total = pp_0 * (VDPM + VIPM).
            yp_total += pp0 * (vd_h + vm_h)

            # Government
            vd_g, vm_g = b.vdgm.get((i, r), 0.0), b.vigm.get((i, r), 0.0)
            va_g, vai_g = b.vdga.get((i, r), 0.0), b.viga.get((i, r), 0.0)
            ad_g, am_g, pg0 = _armington_calibrate(vd_g, vm_g, va_g, vai_g, sigma_d, pds0_i, pim0_i)
            out.alpha_dom_gov[(i, r)] = ad_g
            out.alpha_imp_gov[(i, r)] = am_g
            out.pg_0[(i, r)] = pg0
            yg_total += pg0 * (vd_g + vm_g)

        out.yp_0[r] = yp_total
        out.yg_0[r] = yg_total

        # CD budget shares: at-benchmark expenditure share on good i
        for i in sets.i:
            spend_h = out.pp_0.get((i, r), 1.0) * (
                b.vdpm.get((i, r), 0.0) + b.vipm.get((i, r), 0.0)
            )
            spend_g = out.pg_0.get((i, r), 1.0) * (
                b.vdgm.get((i, r), 0.0) + b.vigm.get((i, r), 0.0)
            )
            if yp_total > 0.0:
                out.share_hhd_cd[(i, r)] = spend_h / yp_total
            if yg_total > 0.0:
                out.share_gov_cd[(i, r)] = spend_g / yg_total

    # ------------------------------------------------------------------
    # Trade block calibration (Phase 2c.2)
    # ------------------------------------------------------------------

    # Per-shipment transport cost share amgm(m,i,s,d).
    for m_lbl in sets.marg:
        for i in sets.i:
            for s in sets.r:
                for d in sets.r:
                    if s == d:
                        continue
                    total_vtwr = sum(
                        b.vtwr.get((m_other, i, s, d), 0.0)
                        for m_other in sets.marg
                    )
                    if total_vtwr <= 0.0:
                        continue
                    vtwr_m = b.vtwr.get((m_lbl, i, s, d), 0.0)
                    out.amgm[(m_lbl, i, s, d)] = vtwr_m / total_vtwr

    # Bilateral trade benchmark prices and quantities.
    for i in sets.i:
        sigma_m = float(params.elasticities.esubm.get(i, 1.0))
        sigma_m_eff = sigma_m if abs(sigma_m - 1.0) > 1e-8 else 1.0 + 1e-3
        for d in sets.r:
            sources_with_flow = []
            for s in sets.r:
                if s == d:
                    continue
                vxwd = b.vxwd.get((i, s, d), 0.0)
                vxmd = b.vxmd.get((i, s, d), 0.0)
                vims = b.vims.get((i, s, d), 0.0)
                viws = b.viws.get((i, s, d), 0.0)
                if vxwd <= 0.0:
                    continue
                txs = out.txs.get((i, s, d), 0.0)
                tms = out.tms.get((i, s, d), 0.0)
                vtwr_total = sum(
                    b.vtwr.get((m_lbl, i, s, d), 0.0) for m_lbl in sets.marg
                )

                # qxs_0 = VXWD (basic-price quantity at ps_0 = 1)
                qxs0 = vxwd
                pe0 = 1.0 + txs
                # per-unit transport cost at benchmark
                pwmg0 = vtwr_total / qxs0 if qxs0 > 0.0 else 0.0
                pmcif0 = pe0 + pwmg0
                pms0 = pmcif0 * (1.0 + tms)

                out.qxs_0[(i, s, d)] = qxs0
                out.pe_0[(i, s, d)] = pe0
                out.pwmg_0[(i, s, d)] = pwmg0
                out.pmcif_0[(i, s, d)] = pmcif0
                out.pms_0[(i, s, d)] = pms0

                sources_with_flow.append((s, qxs0, pms0))

            if not sources_with_flow:
                continue

            # CES bottom Armington composite
            qim0 = sum(q for _, q, _ in sources_with_flow)
            # Cost identity: pim * qim = sum_s pms * qxs
            pim0_cost = sum(p * q for _, q, p in sources_with_flow) / qim0
            out.qim_0[(i, d)] = qim0
            out.pim_0[(i, d)] = pim0_cost

            # Distribution parameters: α_x_s = (qxs_s/qim) * (pms_s/pim)^σ
            for s, qxs0, pms0 in sources_with_flow:
                share_s = qxs0 / qim0
                if pim0_cost > 0.0 and pms0 > 0.0:
                    out.alpha_xs[(i, s, d)] = share_s * (pms0 / pim0_cost) ** sigma_m_eff
                else:
                    out.alpha_xs[(i, s, d)] = share_s

    # ------------------------------------------------------------------
    # Margins block calibration (Phase 2c.2)
    # ------------------------------------------------------------------

    for m_lbl in sets.marg:
        # qst_0(m,r) = VST(m,r): margin commodity supply
        for r in sets.r:
            out.qst_0[(m_lbl, r)] = b.vst.get((m_lbl, r), 0.0)
        # qtm_0(m) = total margin services demanded worldwide
        out.qtm_0[m_lbl] = sum(b.vst.get((m_lbl, r), 0.0) for r in sets.r)
        # ptmg_0(m) = 1 at benchmark (basic price normalization)
        out.ptmg_0[m_lbl] = 1.0
        # share_st(m,r) = VST(m,r) / total margin sales (CD aggregator)
        total_vst = out.qtm_0[m_lbl]
        if total_vst > 0.0:
            for r in sets.r:
                out.share_st[(m_lbl, r)] = b.vst.get((m_lbl, r), 0.0) / total_vst

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

                # Benchmark agent prices (firm-side):
                #     pfd_0 = (1 + to) * (1 + tfd)
                #     pfm_0 = pim_0 * (1 + tfi)
                # where pim_0 is the composite import benchmark price
                # (precomputed above; defaults to 1 if no import flow).
                pim0_ir = out.pim_0.get((i, r), 1.0)
                pfd0 = (1.0 + out.to.get((i, r), 0.0)) * (
                    1.0 + out.tfd.get((i, j, r), 0.0)
                )
                pfm0 = pim0_ir * (1.0 + out.tfi.get((i, j, r), 0.0))

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
