"""Literal port of comp_altertax.gms:14469-14730 variable initialization.

Each function is a pure transformer: takes ``params`` (post-elasticity-override)
and returns a ``dict[index_tuple, float]`` of values for one or more variables.

Order of operations matches GAMS exactly:
  1. compute_factor_prices_and_volumes  -> lines 14469-14487  (pft, kappaf, pfy, pf, xf, xft)
  2. compute_factor_tax_wedges          -> lines 14489-14493  (fctts, fcttx, pfa)
  3. compute_xp                          -> lines 14495-14496  (xp)
  4. compute_nd_and_va                   -> lines 14512-14521  (nd, va)
  5. compute_private_demand              -> lines 14563-14576  (yc, xcshr, pcons, uh, ev, cv)
  6. compute_public_demand               -> lines 14584-14588  (yg, xg, ug)
  7. compute_ytax_streams                -> lines 14730-14731  (ytax[r,'ft'], ytax[r,'fs'])

These are CONSUMED by ``altertax_snapshot.build_altertax_warm_start_snapshot``
which assembles them into a ``GTAPVariableSnapshot`` for the solver's warm-start.
"""

from __future__ import annotations

from typing import Dict, Tuple

from equilibria.templates.gtap.gtap_parameters import GTAPParameters


def compute_altertax_initial_values(params: GTAPParameters) -> Dict[str, Dict[Tuple, float]]:
    """Run the full comp_altertax.gms post-override calibration sequence.

    Returns a dict ``{var_name: {index_tuple: value}}`` covering every variable
    listed in the module docstring's order of operations.
    """
    out: Dict[str, Dict[Tuple, float]] = {}
    out.update(compute_factor_prices_and_volumes(params))
    out.update(compute_factor_tax_wedges(params, out))
    out.update(compute_xp(params, out))
    out.update(compute_nd_and_va(params, out))
    out.update(compute_private_demand(params, out))
    out.update(compute_public_demand(params, out))
    out.update(compute_ytax_streams(params, out))
    return out


def compute_factor_prices_and_volumes(params: GTAPParameters) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14469-14487 (if(1,...) branch).

    Sets:
      pft(r,f)         = 1.0  [numeraire-relative; baseline value preserved]
      kappaf(r,f,a)    = (EVFB - EVOS) / EVFB
      pfy(r,f,a)       = pft(r,f)
      pf(r,f,a)        = pfy / (1 - kappaf)
      xf(r,f,a)        = EVFB / pf
      xft(r,f)         = sum_a(pfy*xf) / pft
    """
    bench = params.benchmark
    sets = params.sets

    pft: Dict[Tuple[str, str], float] = {}
    kappaf: Dict[Tuple[str, str, str], float] = {}
    pfy: Dict[Tuple[str, str, str], float] = {}
    pf: Dict[Tuple[str, str, str], float] = {}
    xf: Dict[Tuple[str, str, str], float] = {}
    xft: Dict[Tuple[str, str], float] = {}

    # pft initialized to 1.0 (numeraire-relative); GAMS does not re-init pft
    # in this block -- it stays at 1.0 from baseline calibration.
    for r in sets.r:
        for f in sets.f:
            pft[(r, f)] = 1.0

    for r in sets.r:
        for f in sets.f:
            for a in sets.a:
                key = (r, f, a)
                evfb = float(bench.evfb.get(key, 0.0) or 0.0)
                if evfb <= 0:
                    continue
                evos = float(bench.evos.get(key, 0.0) or 0.0)
                kappaf_val = (evfb - evos) / evfb
                kappaf[key] = kappaf_val
                pft_val = pft.get((r, f), 1.0)
                pfy[key] = pft_val
                denom = 1.0 - kappaf_val
                if abs(denom) < 1e-12:
                    continue
                pf_val = pft_val / denom
                pf[key] = pf_val
                xf[key] = evfb / pf_val

    # xft(r,f) = sum_a(pfy*xf) / pft
    for r in sets.r:
        for f in sets.f:
            num = 0.0
            for a in sets.a:
                key = (r, f, a)
                num += pfy.get(key, 0.0) * xf.get(key, 0.0)
            pft_val = pft.get((r, f), 1.0)
            if pft_val > 0:
                xft[(r, f)] = num / pft_val

    return {"pft": pft, "kappaf": kappaf, "pfy": pfy, "pf": pf, "xf": xf, "xft": xft}


def compute_factor_tax_wedges(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14489-14493. Reads pf, xf from ``prior``.

    Side effect: ALSO mutates ``params.taxes.fctts_rate`` and
    ``params.taxes.fcttx_rate`` so model-build-time eq_pfaeq/eq_pfyeq
    (which read taxes.*) see the same wedge.
    """
    bench = params.benchmark
    pf = prior["pf"]
    xf = prior["xf"]

    fctts: Dict[Tuple[str, str, str], float] = {}
    fcttx: Dict[Tuple[str, str, str], float] = {}
    pfa: Dict[Tuple[str, str, str], float] = {}

    for key, pf_val in pf.items():
        xf_val = xf.get(key, 0.0)
        if xf_val <= 0 or pf_val <= 0:
            continue
        denom = pf_val * xf_val  # GAMS uses recomputed pf*xf, NOT EVFB
        fbep = float(bench.fbep.get(key, 0.0) or 0.0)
        ftrv = float(bench.ftrv.get(key, 0.0) or 0.0)
        fctts_val = -fbep / denom
        fcttx_val = ftrv / denom
        if abs(fctts_val) > 1e-12:
            fctts[key] = fctts_val
        if abs(fcttx_val) > 1e-12:
            fcttx[key] = fcttx_val
        pfa[key] = pf_val * (1.0 + fctts_val + fcttx_val)

    # Mutate taxes so model build picks up the literal-port wedges.
    params.taxes.fctts_rate = fctts
    params.taxes.fcttx_rate = fcttx

    return {"fctts": fctts, "fcttx": fcttx, "pfa": pfa}


def compute_xp(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14495-14496.

    xp uses baseline-derived pdp/xd/pmp/xm/px which we leave at 1.0/benchmark
    defaults. Most useful here is the FACTOR side, which uses the new pfa/xf.
    """
    sets = params.sets
    pfa = prior.get("pfa", {})
    xf = prior.get("xf", {})

    xp: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            factor_part = sum(
                pfa.get((r, f, a), 0.0) * xf.get((r, f, a), 0.0)
                for f in sets.f
            )
            if factor_part > 0:
                # px=1, intermediate side left to model default init
                xp[(r, a)] = factor_part
    return {"xp": xp}


def compute_nd_and_va(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14512-14521.

    Only va is computed here (depends on new pfa, xf).
    nd is left to the model's default initializer (pa, xa unchanged).
    """
    sets = params.sets
    pfa = prior.get("pfa", {})
    xf = prior.get("xf", {})

    va: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            num = sum(
                pfa.get((r, f, a), 0.0) * xf.get((r, f, a), 0.0)
                for f in sets.f
            )
            if num > 0:
                # pva default = 1.0
                va[(r, a)] = num
    return {"va": va}


def compute_nd_levels(params: GTAPParameters) -> Dict[Tuple[str, str], float]:
    """nd(r,a) = sum_i (VDFP(r,i,a) + VMFP(r,i,a)).

    Intermediate purchaser-price aggregate. Identical to baseline since the
    altertax bundle does not perturb consumer/intermediate flows — kept as a
    first-class helper so the share recalibration block has a clean source.
    """
    bench = params.benchmark
    sets = params.sets

    nd: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            total = 0.0
            for i in sets.i:
                total += float(bench.vdfp.get((r, i, a), 0.0) or 0.0)
                total += float(bench.vmfp.get((r, i, a), 0.0) or 0.0)
            nd[(r, a)] = total
    return nd


def compute_xp_levels(
    nd: Dict[Tuple[str, str], float],
    va: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """xp(r,a) = nd(r,a) + va(r,a). Activity output at producer prices."""
    xp: Dict[Tuple[str, str], float] = {}
    keys = set(nd.keys()) | set(va.keys())
    for key in keys:
        xp[key] = nd.get(key, 0.0) + va.get(key, 0.0)
    return xp


def compute_share_recalibration(
    params: GTAPParameters,
) -> Dict[str, Dict[Tuple, float]]:
    """Port of cal.gms:15052-15058 share recalibration (numeraire form).

    At the recalibration point all price aggregators equal 1.0, so the
    CES-derived share formulas collapse to value-share ratios:

      and(r,a)    = nd / xp
      ava(r,a)    = va / xp
      io(r,i,a)   = (vdfp+vmfp) / nd                    (sigmand collapsed)
      af(r,fp,a)  = (pfa*xf) / va                       (pfa carries wedge)

    Invariants this routine preserves at the calibration point:
      - and(r,a) + ava(r,a) = 1
      - sum_i io(r,i,a)    = 1
      - sum_f af(r,f,a)    = 1
    """
    sets = params.sets
    bench = params.benchmark

    vals = compute_altertax_initial_values(params)
    pfa = vals["pfa"]
    xf = vals["xf"]
    va = vals["va"]

    nd = compute_nd_levels(params)
    xp = compute_xp_levels(nd, va)

    and_param: Dict[Tuple[str, str], float] = {}
    ava_param: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            xp_val = xp.get((r, a), 0.0)
            if xp_val <= 0:
                continue
            and_param[(r, a)] = nd.get((r, a), 0.0) / xp_val
            ava_param[(r, a)] = va.get((r, a), 0.0) / xp_val

    io_param: Dict[Tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            nd_val = nd.get((r, a), 0.0)
            if nd_val <= 0:
                continue
            for i in sets.i:
                v = float(bench.vdfp.get((r, i, a), 0.0) or 0.0) + float(
                    bench.vmfp.get((r, i, a), 0.0) or 0.0
                )
                if v > 0:
                    io_param[(r, i, a)] = v / nd_val

    af_param: Dict[Tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            va_val = va.get((r, a), 0.0)
            if va_val <= 0:
                continue
            for f in sets.f:
                key = (r, f, a)
                v = pfa.get(key, 0.0) * xf.get(key, 0.0)
                if v > 0:
                    af_param[key] = v / va_val

    return {
        "and_param": and_param,
        "ava_param": ava_param,
        "io_param": io_param,
        "af_param": af_param,
    }


def compute_private_demand(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14563-14576.

    yc(r)      = sum_i (VDPB + VMPB)         [private consumption value]
    xcshr(r,i) = (vdpb+vmpb) / yc            [budget share, frozen Param]
    pcons(r)   = 1.0                         [composite consumer price index]
    uh(r)      = 1.0                         [per-capita utility, numeraire]
    ev(r)      = yc                          [equivalent variation level]
    cv(r)      = yc                          [compensating variation level]

    9x10 sets do not expose an agent-h dimension; benchmark flows are keyed by
    (r,i) and the household is the implicit single agent ``"hhd"``.
    """
    bench = params.benchmark
    sets = params.sets

    yc: Dict[Tuple[str], float] = {}
    xcshr: Dict[Tuple[str, str], float] = {}
    pcons: Dict[Tuple[str], float] = {}
    uh: Dict[Tuple[str, str], float] = {}
    ev: Dict[Tuple[str, str], float] = {}
    cv: Dict[Tuple[str, str], float] = {}

    for r in sets.r:
        total = 0.0
        per_i: Dict[str, float] = {}
        for i in sets.i:
            v = float(bench.vdpb.get((r, i), 0.0) or 0.0) + float(
                bench.vmpb.get((r, i), 0.0) or 0.0
            )
            per_i[i] = v
            total += v
        if total <= 0:
            continue
        yc[(r,)] = total
        pcons[(r,)] = 1.0
        uh[(r, "hhd")] = 1.0
        ev[(r, "hhd")] = total
        cv[(r, "hhd")] = total
        for i, v in per_i.items():
            if v > 0:
                xcshr[(r, i)] = v / total

    return {
        "yc": yc,
        "xcshr": xcshr,
        "pcons": pcons,
        "uh": uh,
        "ev": ev,
        "cv": cv,
    }


def compute_public_demand(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14584-14588.

    yg(r)    = sum_i (VDGB + VMGB)   [public consumption value]
    xg(r,i)  = vdgb+vmgb              [agent-i public quantity at p=1]
    ug(r)    = 1.0                    [public utility numeraire]
    """
    bench = params.benchmark
    sets = params.sets

    yg: Dict[Tuple[str], float] = {}
    xg: Dict[Tuple[str, str], float] = {}
    ug: Dict[Tuple[str, str], float] = {}

    for r in sets.r:
        total = 0.0
        for i in sets.i:
            v = float(bench.vdgb.get((r, i), 0.0) or 0.0) + float(
                bench.vmgb.get((r, i), 0.0) or 0.0
            )
            if v > 0:
                xg[(r, i)] = v
            total += v
        if total <= 0:
            continue
        yg[(r,)] = total
        ug[(r, "gov")] = 1.0

    return {"yg": yg, "xg": xg, "ug": ug}


def compute_ytax_streams(
    params: GTAPParameters, prior: Dict[str, Dict[Tuple, float]]
) -> Dict[str, Dict[Tuple, float]]:
    """Port of comp_altertax.gms:14730-14731 (ft, fs streams only).

    ytax(r,'ft') = sum_{fp,a} fcttx*pf*xf      (factor input tax revenue)
    ytax(r,'fs') = sum_{fp,a} fctts*pf*xf      (factor output tax — typically <0)

    Other gy streams (pt, fc, pc, gc, ic, dt, mt, et) are left to the model's
    standard initialiser since the altertax bundle only re-derives factor wedges.
    """
    sets = params.sets
    fctts = prior.get("fctts", {})
    fcttx = prior.get("fcttx", {})
    pf = prior.get("pf", {})
    xf = prior.get("xf", {})

    ytax: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        ft = 0.0
        fs = 0.0
        for f in sets.f:
            for a in sets.a:
                key = (r, f, a)
                pf_val = pf.get(key, 0.0)
                xf_val = xf.get(key, 0.0)
                ft += fcttx.get(key, 0.0) * pf_val * xf_val
                fs += fctts.get(key, 0.0) * pf_val * xf_val
        ytax[(r, "ft")] = ft
        ytax[(r, "fs")] = fs
    return {"ytax": ytax}
