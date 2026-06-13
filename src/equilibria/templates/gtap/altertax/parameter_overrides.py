"""Altertax elasticity overrides.

Mirror of ``cgebox/gtap/scen/Parameters/parameter_altertax.gms``:

| Param           | Original GTAP | Altertax |
|-----------------|---------------|----------|
| ``esubva``      | calibrated    | **1**    |
| ``esubinv``     | calibrated    | **1**    |
| ``esubd``       | calibrated    | **0.95** |
| ``esubm``       | calibrated    | **0.95** |
| ``etrae``       | calibrated    | **1**    |
| ``omegaf``      | calibrated    | **1**    |

Equilibria does not (yet) carry a separate ``esubinv`` container —
investment substitution sits under the same Leontief/CES bucket as
the production aggregator. We override the closest equivalents
(``sigmav``/``sigmap``) instead.

NOT overridden (kept at calibrated/getData.gms defaults):
- ``omegax`` stays ``inf`` (GAMS getData.gms line 377)
- ``omegaw`` stays ``inf`` (GAMS getData.gms line 378)
- ``omegas`` stays ``-etraq`` (GAMS getData.gms line 350)
- ``sigmas`` stays calibrated (not in comp_altertax.gms)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

from equilibria.templates.gtap.gtap_parameters import (
    GTAPElasticities,
    GTAPParameters,
)


@dataclass(frozen=True)
class AltertaxElasticityOverrides:
    """Altertax preset values. Tweak only if you know what you're doing."""

    esubva: float = 1.0
    esubd: float = 0.95
    esubm: float = 0.95
    etrae: float = 1.0
    omegaf: float = 1.0
    # Nested CES for production (downstream of esubva)
    sigmav: float = 1.0
    sigmap: float = 1.0
    sigmand: float = 1.0
    # Government / investment expenditure (GAMS: esubg=esubi=1 in altertax)
    esubg: float = 1.0
    esubi: float = 1.0


ALTERTAX_ELASTICITY_DEFAULTS = AltertaxElasticityOverrides()


def _override_dict(target: Dict, value: float) -> int:
    """Replace every value in ``target`` with ``value``. Returns count."""
    n = 0
    for key in list(target.keys()):
        target[key] = value
        n += 1
    return n


def apply_altertax_elasticities(
    params: GTAPParameters,
    overrides: AltertaxElasticityOverrides | None = None,
    *,
    in_place: bool = False,
) -> GTAPParameters:
    """Return a copy of ``params`` with altertax elasticity overrides.

    The defaults match cgebox ``parameter_altertax.gms``: CD value-added
    (esubva=1), Armington 0.95, mobile factors with CD CET (etrae=1).

    Args:
        params: source GTAPParameters (calibrated baseline).
        overrides: replacement values (defaults to ALTERTAX_ELASTICITY_DEFAULTS).
        in_place: mutate ``params`` directly. Default False (deep copy).

    Returns:
        GTAPParameters with overridden elasticities. Calibrated shares,
        SAM benchmarks, and tax rates are untouched — only elasticities.
    """
    o = overrides or ALTERTAX_ELASTICITY_DEFAULTS
    target = params if in_place else copy.deepcopy(params)
    e: GTAPElasticities = target.elasticities

    _override_dict(e.esubva, o.esubva)
    _override_dict(e.esubd, o.esubd)
    _override_dict(e.esubm, o.esubm)
    _override_dict(e.omegaf, o.omegaf)
    _override_dict(e.sigmav, o.sigmav)
    _override_dict(e.sigmap, o.sigmap)
    _override_dict(e.sigmand, o.sigmand)
    for key in list(e.etrae.keys()):
        e.etrae[key] = o.etrae
    # GAMS parameter_altertax.gms: esubg(r)=esubi(r)=1 for all regions.
    # esubg may already be 1 in the GDX; esubi is often absent (loaded as na/empty)
    # and must be explicitly set so sigmai=1→1.01 in eq_pi/eq_xi.
    for r in target.sets.r:
        e.esubg[(r,)] = o.esubg
        e.esubi[(r,)] = o.esubi

    # Recalibrate trade shares with the new esubm (sigmaw).
    # _calibrate_trade_shares uses esubm to compute amw weights; skipping this
    # leaves p_amw calibrated with the old (e.g. 5.07) exponent, causing
    # get_pmt_init to blow up to ~40,000 when esubm changes to 0.95.
    target.shares._calibrate_trade_shares(target.benchmark, target.elasticities, target.sets)
    target.shares.normalized.update_from_shares(target.shares)

    # Recompute af shares consistent with sigma=1 (CD).
    # GAMS calibrates af at base with pva=1 → af = pfa*xf/va, sum(af)=1.
    # The original GTAP af_param was calibrated with original sigmav (≠1) and
    # does NOT sum to 1. With sigma=1 the pvaeq degenerates to 1 = sum(af).
    _recalibrate_af_shares(target)

    # Use pva/pnd init = 1.0 (GAMS default: pva.l=1 for all periods).
    # _recalibrate_bench_prices computes prod(pfa^af) ≈ 1.4-1.7, which
    # diverges from GAMS equilibrium (pva=1) and causes PATH to fail.
    target.calibrated.pva_bench = {}
    target.calibrated.pnd_bench = {}

    # Mirror GAMS parameter_altertax.gms lines 1218-1223:
    #   fnm(fp) = no ;           (remove all sector-specific factors)
    #   fm(fp)  = yes ;          (make ALL factors mobile)
    #   omegaf(r,fp) = 1.0 ;    (CET=1 / CD distribution for all factors)
    # This ensures:
    #   1. xftflag=1 for all factors (GAMS xftFlag(r,fm)$xft.l>0)
    #   2. omegaf=1 (CD CET) — not inf — for all factors
    #   3. etaf=0 for all mobile factors (set via gtap_model_equations.py)
    all_factors = list(target.sets.f)
    target.sets.mf = all_factors[:]
    target.sets.sf = []
    for r in target.sets.r:
        for f in all_factors:
            target.elasticities.omegaf[(r, f)] = o.omegaf  # = 1.0

    return target


def _recalibrate_bench_prices(params: "GTAPParameters") -> None:  # noqa: F821
    """Recompute pva_bench / pnd_bench consistent with current elasticities.

    Mirrors GAMS cal.gms recalibration that runs after parameter_altertax.gms
    overrides sigmav/sigmand/sigmap. At benchmark all quantities and prices are
    at SAM values with pf=1/(1-kappaf), pfa=pf*(1+rtf), pa=1, etc.

    CD case (sigma=1): pva = prod_f(pfa/lambdaf)^af  (Cobb-Douglas)
    CES case:          pva^(1-sigma) = sum_f af*(pfa/lambdaf)^(1-sigma)
    """
    import math

    cal = params.calibrated
    e = params.elasticities
    sets = params.sets
    taxes = params.taxes

    def _kappa(r: str, f: str, a: str) -> float:
        kap = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
        if kap == 0.0:
            kap = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
        return kap

    def _pfa_bench(r: str, f: str, a: str) -> float:
        pf = 1.0 / max(1.0 - _kappa(r, f, a), 1e-12)
        rtf = float(taxes.rtf.get((r, f, a), 0.0) or 0.0)
        return pf * max(1.0 + rtf, 1e-12)

    def _pa_bench(r: str, i: str, a: str) -> float:
        # At benchmark pa = pdp*(1+dintx) ~ 1; import tax also ~1 for pmp.
        # The CES aggregate price at benchmark uses pa=1 for all agents.
        return 1.0

    # --- pva_bench: CD or CES over factors ---
    pva_bench: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            sigma_v = e.sigmav.get((r, a), 1.0)
            af_vals = {f: cal.af_param.get((r, f, a), 0.0) for f in sets.f}
            active = {f: v for f, v in af_vals.items() if v > 0.0}
            if not active:
                pva_bench[(r, a)] = 1.0
                continue
            if abs(1.0 - sigma_v) < 1e-8:  # CD
                log_pva = sum(av * math.log(max(_pfa_bench(r, f, a), 1e-12))
                              for f, av in active.items())
                pva_bench[(r, a)] = math.exp(log_pva)
            else:
                expo = 1.0 - sigma_v
                ces_sum = sum(av * (_pfa_bench(r, f, a) ** expo)
                              for f, av in active.items())
                pva_bench[(r, a)] = max(ces_sum, 1e-12) ** (1.0 / expo)

    # --- pnd_bench: CD or CES over intermediates ---
    pnd_bench: Dict[Tuple[str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            sigma_nd = e.sigmand.get((r, a), 1.0)
            io_vals = {i: cal.io_param.get((r, i, a), 0.0) for i in sets.i}
            active = {i: v for i, v in io_vals.items() if v > 0.0}
            if not active:
                pnd_bench[(r, a)] = 1.0
                continue
            if abs(1.0 - sigma_nd) < 1e-8:  # CD
                log_pnd = sum(iv * math.log(max(_pa_bench(r, i, a), 1e-12))
                              for i, iv in active.items())
                pnd_bench[(r, a)] = math.exp(log_pnd)
            else:
                expo = 1.0 - sigma_nd
                ces_sum = sum(iv * (_pa_bench(r, i, a) ** expo)
                              for i, iv in active.items())
                pnd_bench[(r, a)] = max(ces_sum, 1e-12) ** (1.0 / expo)

    cal.pva_bench = pva_bench
    cal.pnd_bench = pnd_bench


def _recalibrate_af_shares(params: "GTAPParameters") -> None:  # noqa: F821
    """Recompute af_param so sum_f(af[r,f,a]) = 1 for every (r,a).

    With sigmav=1 (CD) the pvaeq constraint degenerates to 1 = sum_f(af).
    The original af_param calibrated with original sigmav≠1 does not satisfy
    this. Mirrors GAMS comp_altertax.gms:14469-14487 + post-betaCal calibration
    where pva=1 at the benchmark point:

        af(r,f,a) = pfa(r,f,a) * xf(r,f,a) / va(r,a)
                  = pfa * xf / sum_f2(pfa * xf)

    Factor prices and volumes computed from benchmark SAM at pva=1:
        kappaf  = (EVFB - EVOS) / EVFB
        pf      = 1 / (1 - kappaf)
        xf      = EVFB / pf
        pfa     = pf * (1 + rtf)
        va      = sum_f(pfa * xf)  =  sum_f(EVFB * (1 + rtf))
    """
    cal = params.calibrated
    sets = params.sets
    taxes = params.taxes
    bench = params.benchmark

    def _kappa(r: str, f: str, a: str) -> float:
        kap = float(taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
        if kap == 0.0:
            kap = float(taxes.kappaf.get((r, f), 0.0) or 0.0)
        return kap

    def _pf(r: str, f: str, a: str) -> float:
        return 1.0 / max(1.0 - _kappa(r, f, a), 1e-12)

    def _pfa(r: str, f: str, a: str) -> float:
        # GAMS comp_altertax.gms:2867 — pfa = pf*(1 + fctts + fcttx), NOT rtf.
        # fcttx=ftrv/(pf·xf)=RTFD/EVFB, fctts=-fbep/(pf·xf)=-RTFI; both 0 when
        # RTFD=RTFM=0. rtf folds in kappaf (factor rent) and varies by factor,
        # distorting the relative af shares. Faithful to GAMS base calibration.
        rtfd = getattr(taxes, "rtfd", {}) or {}
        rtfi = getattr(taxes, "rtfi", {}) or {}
        wedge = (float(rtfd.get((r, f, a), 0.0) or 0.0)
                 - abs(float(rtfi.get((r, f, a), 0.0) or 0.0)))
        return _pf(r, f, a) * max(1.0 + wedge, 1e-12)

    af_param: Dict[Tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            pfa_xf: Dict[str, float] = {}
            for f in sets.f:
                key = (r, f, a)
                evfb = float(bench.evfb.get(key, bench.vfm.get(key, 0.0)) or 0.0)
                if evfb <= 0.0:
                    continue
                pf_val = _pf(r, f, a)
                xf_val = evfb / pf_val
                pfa_val = _pfa(r, f, a)
                pfa_xf[f] = pfa_val * xf_val
            va_val = sum(pfa_xf.values())
            if va_val <= 0.0:
                continue
            for f, pxa in pfa_xf.items():
                af_param[(r, f, a)] = pxa / va_val

    cal.af_param.update(af_param)
