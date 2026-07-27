"""Deterministic GTAP model-parameter derivations, transcribed from the monolith.

The GTAP monolith builds its Pyomo Params in ``GTAPModelEquations._add_parameters``
(``gtap_model_equations.py`` 1620-2789) by deriving them ONCE from the calibrated
``GTAPParameters`` + ``GTAPSets`` (plus the closure switch). This is
model-parameter SETUP, not runtime calibration (that is
``apply_production_scaling`` / ``_recalibrate_*``, which the composer/Task-5 owns).

The block units need the SAME Pyomo Params, keyed and valued identically, so their
``build_expression`` bodies reference ``model.<param>[...]`` exactly as the monolith
does and ``form_diff`` stays clean. Rather than duplicate the derivation in every
block, the recipes live here as pure functions of ``(params, sets)`` returning
``{index_tuple: float}`` dicts (the shape ``create_indexed_param`` consumes).

FIDELITY: each function is a verbatim transcription of the corresponding monolith
block — same benchmark headers, same guards, same fallbacks. Line references point
at the monolith source. Do NOT re-derive economics here; if a monolith recipe
changes, mirror it.

KNOWN CARRY (Blocker C, pre-adjudicated to Task 5): ``gd_share``/``ge_share``/
``gw_share`` are seeded here from ``params.shares`` but the monolith's
``apply_production_scaling`` recomputes them at runtime from post-scaling Var
levels. The composer must run that recompute after block registration. On
gtap7_3x3 all ``omegax=inf`` so gd/ge never enter an active (Leontief) body;
``gw_share`` does enter unit-4 bodies (see that unit's notes).
"""

from __future__ import annotations

import math
from typing import Any


def _f(x: Any) -> float:
    return float(x or 0.0)


def to_array(data: dict, elem_lists: list[list[str]], default: float = 0.0):
    """Fill a numpy array from a {index_tuple: value} dict.

    ``elem_lists`` is the element list per domain in order; the array shape is
    the product of their lengths, filled by index-tuple lookup (matching the
    bridge's itertools.product ordering). Missing cells get ``default`` — the
    same semantics as the monolith's create_indexed_param ``default=``.
    """
    import itertools

    import numpy as np

    shape = tuple(len(e) for e in elem_lists)
    arr = np.full(shape, default, dtype=float)
    pos = [{e: k for k, e in enumerate(el)} for el in elem_lists]
    for key, val in data.items():
        ktup = key if isinstance(key, tuple) else (key,)
        if len(ktup) != len(elem_lists):
            continue
        try:
            idx = tuple(pos[d][ktup[d]] for d in range(len(ktup)))
        except KeyError:
            continue
        arr[idx] = float(val)
    return arr


def _gams_round(value: float) -> int:
    """Monolith _gams_round (1638-1641)."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def xscale_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """xScale(r,aa) — monolith 1628-1702.

    Aggregate agents (hhd/gov/inv/tmg) → 1.0. Activity columns rescaled to
    xpScale*10**(-round(log10(xp.l))) with xp = nd + va at purchaser value
    (VA includes the (ftrv-fbep)/evfb subsidy/tax wedge).
    """
    from equilibria.templates.gtap.gtap_parameters import (
        GTAP_GOVERNMENT_AGENT,
        GTAP_HOUSEHOLD_AGENT,
        GTAP_INVESTMENT_AGENT,
        GTAP_MARGIN_AGENT,
    )

    bm = params.benchmark
    out: dict[tuple[str, str], float] = {}
    for r in sets.r:
        for aa in (
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
            GTAP_MARGIN_AGENT,
        ):
            out[(r, aa)] = 1.0

    for r in sets.r:
        for a in sets.a:
            nd_level = sum(
                _f(bm.vdfp.get((r, i, a), 0.0)) + _f(bm.vmfp.get((r, i, a), 0.0))
                for i in sets.i
            )
            if nd_level <= 0.0:
                nd_level = sum(
                    _f(bm.vdfm.get((r, i, a), 0.0)) + _f(bm.vifm.get((r, i, a), 0.0))
                    for i in sets.i
                )
            va_level = 0.0
            for f in sets.f:
                evfb_val = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if evfb_val <= 0.0:
                    continue
                fbep_val = _f(bm.fbep.get((r, f, a), 0.0))
                ftrv_val = _f(bm.ftrv.get((r, f, a), 0.0))
                va_level += evfb_val + (ftrv_val - fbep_val)
            xp_level = nd_level + va_level
            if xp_level <= 0.0:
                xp_level = _f(bm.vom.get((r, a), 0.0))
            if xp_level <= 0.0:
                xp_level = sum(_f(bm.makb.get((r, a, i), 0.0)) for i in sets.i)
            if xp_level > 0.0:
                exponent = _gams_round(math.log10(xp_level))
                out[(r, a)] = float(10.0 ** (-exponent))
            else:
                out[(r, a)] = 1.0
    return out


def xflag_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """xflag(r,a,i) = 1 if |makb|>1e-12 — monolith 1802-1809/1826."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            for i in sets.i:
                val = bm.makb.get((r, a, i), 0.0)
                out[(r, a, i)] = 1.0 if abs(_f(val)) > 1e-12 else 0.0
    return out


def _and_ava_nd_pio(params: Any, sets: Any):
    """adjusted_and_param/ava_param/nd_share/p_io — monolith 1865-1940."""
    bm = params.benchmark
    adjusted_and: dict[tuple[str, str], float] = {}
    adjusted_ava: dict[tuple[str, str], float] = {}
    adjusted_nd_share: dict[tuple[str, str], float] = {}
    adjusted_p_io: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            adjusted_total_intermediate = 0.0
            adjusted_values: dict[str, float] = {}
            for i in sets.i:
                domestic_val = _f(bm.vdfp.get((r, i, a), 0.0))
                import_val = _f(bm.vmfp.get((r, i, a), 0.0))
                adjusted_val = max(domestic_val, 0.0) + max(import_val, 0.0)
                adjusted_values[i] = adjusted_val
                adjusted_total_intermediate += adjusted_val

            nd_p = sum(
                _f(bm.vdfp.get((r, i, a), 0.0)) + _f(bm.vmfp.get((r, i, a), 0.0))
                for i in sets.i
            )
            if nd_p <= 0.0:
                nd_p = adjusted_total_intermediate
            va_p = 0.0
            for f in sets.f:
                evfb_val = _f(bm.evfb.get((r, f, a), bm.vfm.get((r, f, a), 0.0)))
                if evfb_val <= 0.0:
                    continue
                fbep_val = _f(bm.fbep.get((r, f, a), 0.0))
                ftrv_val = _f(bm.ftrv.get((r, f, a), 0.0))
                va_p += evfb_val + (ftrv_val - fbep_val)
            xp_model_equiv = nd_p + va_p
            adjusted_and[(r, a)] = (
                nd_p / xp_model_equiv if xp_model_equiv > 0.0 else 0.0
            )
            adjusted_ava[(r, a)] = (
                va_p / xp_model_equiv if xp_model_equiv > 0.0 else 0.0
            )
            adjusted_nd_share[(r, a)] = adjusted_and[(r, a)]
            if adjusted_total_intermediate > 0.0:
                for i, adjusted_val in adjusted_values.items():
                    adjusted_p_io[(r, i, a)] = (
                        adjusted_val / adjusted_total_intermediate
                    )
            else:
                for i in sets.i:
                    adjusted_p_io[(r, i, a)] = 0.0
    return adjusted_and, adjusted_ava, adjusted_nd_share, adjusted_p_io


def and_param_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[0]


def ava_param_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[1]


def nd_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    return _and_ava_nd_pio(params, sets)[2]


def p_io_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    return _and_ava_nd_pio(params, sets)[3]


def io_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """io_param — read straight from params.calibrated (monolith 1960-1966)."""
    return dict(params.calibrated.io_param)


def af_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """af_param — read straight from params.calibrated (monolith 1975-1981)."""
    return dict(params.calibrated.af_param)


def gx_param_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """gx_param — read straight from params.calibrated (monolith 1987-1993).

    CARRY: apply_production_scaling recomputes gx per period; the seed is used
    at build (matches the oracle which also seeds from calibrated before scaling
    leaves gx unchanged on the base period)."""
    return dict(params.calibrated.gx_param)


def p_ax_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """p_ax — read straight from params.shares (monolith 1997-1999)."""
    return dict(params.shares.p_ax)


def lambdaio_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """lambdaio — params.shifts if present else default 1.0 (monolith 1967-1974)."""
    return dict(params.shifts.lambdaio)


def gd_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """gd_share — seed from params.shares.p_gd (monolith 2000-2002). CARRY."""
    return dict(params.shares.p_gd)


def ge_share_data(params: Any, sets: Any) -> dict[tuple[str, str], float]:
    """ge_share — seed from params.shares.p_ge (monolith 2003-2005). CARRY."""
    return dict(params.shares.p_ge)


def prdtx_rai_data(params: Any, sets: Any) -> dict[tuple[str, str, str], float]:
    """prdtx_rai(r,a,i) = makb/maks - 1 (else rto) — monolith 2146-2165."""
    bm = params.benchmark
    out: dict[tuple[str, str, str], float] = {}
    for r in sets.r:
        for a in sets.a:
            outputs = sets.activity_commodities.get(a, [])
            if not outputs:
                outputs = list(sets.i)
            for i in outputs:
                makb_val = _f(bm.makb.get((r, a, i), 0.0))
                maks_val = _f(bm.maks.get((r, a, i), 0.0))
                if maks_val > 0 and makb_val > 0:
                    out[(r, a, i)] = makb_val / maks_val - 1.0
                else:
                    out[(r, a, i)] = _f(params.taxes.rto.get((r, a), 0.0))
    return out
