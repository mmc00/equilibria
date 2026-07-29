"""GAMS ifSUB substitution macros (M_*) as shared block helpers.

Under GAMS ``$setGlobal ifSUB 1`` (gtap_model_equations.py macros 5497-5572,
GAMS ``$macro M_*`` 8013-8023) the margin/price report equations
(eq_pfaeq/eq_pfyeq/eq_pp_rai/eq_pmeq/eq_pmcifeq/eq_pefobeq/eq_pwmg/eq_xwmg/eq_xmgm)
are deactivated and the tariff/margin wedge is substituted INLINE into the REAL
equations, expanded down to the live vars (pe/ptmg/pf/p_rai) — so a shock on
``imptx``/``prdtx``/tax wedges propagates through the solved system.

Each macro takes ``(model, p, indices...)`` and returns a Pyomo expression: the
ifSUB inline form. The caller decides ifSUB vs the plain report var — the block
does ``m_pm(model, p, e, c, imp) if if_sub else model.pm[e, c, imp]`` (or wraps a
thin ``_m_*`` closure). Signatures use (exporter, commodity, importer) = (e, c,
imp) mirroring the monolith macro args.

Shared by ArmingtonBilateral, Factor, ProductionSupply and Income blocks so the
inline forms stay identical across every consumer (no per-block drift).
"""

from __future__ import annotations

from typing import Any


def _safe(p: Any, dotted: str, key: tuple, default: float = 0.0) -> float:
    """params.<a>.<b>.get(key, default) as a float, tolerant of missing paths."""
    obj = p
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return float(default)
    try:
        return float(obj.get(key, default) or default)
    except Exception:
        return float(default)


def mtax_value(p: Any, importer: str, commodity: str, exporter: str) -> float:
    # GAMS mtax(r,i) indexed by importer,commodity.
    return _safe(p, "taxes.mtax", (importer, commodity), 0.0)


def etax_value(p: Any, exporter: str, commodity: str, importer: str) -> float:
    # GAMS etax(r,i) indexed by exporter,commodity; 0 at benchmark (comp-stat).
    return _safe(p, "taxes.etax", (exporter, commodity), 0.0)


def chipm_value(p: Any, exporter: str, commodity: str, importer: str) -> float:
    return max(_safe(p, "taxes.chipm", (exporter, commodity, importer), 1.0), 1e-12)


def kappaf_value(p: Any, r: str, f: str, a: str) -> float:
    kappa = _safe(p, "taxes.kappaf_activity", (r, f, a), 0.0)
    if kappa == 0.0:
        kappa = _safe(p, "taxes.kappaf", (r, f), 0.0)
    return kappa


# --- factor / production side ------------------------------------------------
def m_pfa(model, p, r, f, a):
    """M_PFA = pf·(1 + fctts + fcttx). fctts/fcttx are mutable Params on model."""
    return model.pf[r, f, a] * (1.0 + model.fctts[r, f, a] + model.fcttx[r, f, a])


def m_pfy(model, p, r, f, a):
    """M_PFY = pf·(1 - kappaf)."""
    return model.pf[r, f, a] * (1.0 - kappaf_value(p, r, f, a))


def m_pp(model, p, r, a, i):
    """M_PP = p_rai·(1 + prdtx_rai). prdtx_rai is a Param on model."""
    return (1.0 + model.prdtx_rai[r, a, i]) * model.p_rai[r, a, i]


# --- trade / margin side -----------------------------------------------------
def m_pwmg(model, p, e, c, imp):
    """M_PWMG = Σ_m amgm·ptmg/lambdamg."""
    return sum(
        model.amgm[mm, e, c, imp]
        * model.ptmg[mm]
        / (model.lambdamg[mm, e, c, imp] + 1e-12)
        for mm in model.m
    )


def m_pefob(model, p, e, c, imp):
    """M_PEFOB = (1 + rtxs + etax)·pe."""
    export_tax = _safe(p, "taxes.rtxs", (e, c, imp), 0.0)
    return (1.0 + export_tax + etax_value(p, e, c, imp)) * model.pe[e, c, imp]


def m_pmcif(model, p, e, c, imp):
    """M_PMCIF = M_PEFOB + M_PWMG·tmarg."""
    return (
        m_pefob(model, p, e, c, imp)
        + m_pwmg(model, p, e, c, imp) * model.tmarg[e, c, imp]
    )


def m_pm(model, p, e, c, imp):
    """M_PM = (1 + imptx + mtax)·M_PMCIF/chipm (imptx live → shock propagates)."""
    mtax = mtax_value(p, imp, c, e)
    chipm = chipm_value(p, e, c, imp)
    return (
        (1.0 + model.imptx[e, c, imp] + mtax) * m_pmcif(model, p, e, c, imp)
    ) / chipm


def m_xwmg(model, p, e, c, imp):
    """M_XWMG = tmarg·xw."""
    return model.tmarg[e, c, imp] * model.xw[e, c, imp]


def m_xmgm(model, p, mode, e, c, imp):
    """M_XMGM = amgm·M_XWMG/lambdamg."""
    return (
        model.amgm[mode, e, c, imp]
        * m_xwmg(model, p, e, c, imp)
        / (model.lambdamg[mode, e, c, imp] + 1e-12)
    )
