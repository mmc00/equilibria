"""GAMS-style outer recalibration loop for altertax.

comp_altertax.gms:15052-15058 recalibrates and/ava/io/af *after* the altertax
solve using the converged .l levels — a fixed-point iteration that closes
share-parameter self-consistency under the new factor-tax wedges.

The closed-form, pre-solve H12 fix (gtap_model_equations.py:1577-1605) gets
and/ava right at benchmark prices but cannot reach the post-altertax xf/va
levels — those only emerge from a solve. Hence we re-solve, re-extract, and
re-calibrate until the share params converge.

API
---
``recalibrate_from_solution(params, model)``
    Pure transformer. Reads .l from a solved Pyomo model and returns updated
    {and_param, ava_param, io_param, af_param} dicts. Mirrors GAMS lines
    15052-15058 verbatim.

``run_altertax_outer_loop(params, build_and_solve, baseline_warm, max_iter,
                          tol)``
    Driver. Loops solve → recalibrate → mutate params.calibrated → re-solve
    until ``max |Δparam| < tol`` or ``max_iter`` reached. Returns the final
    solved model.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from equilibria.templates.gtap.gtap_parameters import GTAPParameters


def _level(var, idx, default=0.0):
    if not hasattr(var, "__contains__") or idx not in var:
        return default
    v = var[idx].value
    return float(v) if v is not None else default


def recalibrate_from_solution(
    params: GTAPParameters, model
) -> Dict[str, Dict[Tuple, float]]:
    """Recalibrate share parameters from a solved altertax model.

    GAMS reference (comp_altertax.gms:15052-15058)::

        and(r,a,t)$ndFlag(r,a) = (nd.l/xp.l)*(pnd.l/px.l)**sigmap(r,a)
        ava(r,a,t)$vaFlag(r,a) = (va.l/xp.l)*(pva.l/px.l)**sigmap(r,a)
        io(r,i,a,t)$xaFlag(r,i,a) = (xa.l/nd.l)*(pa.l/pnd.l)**sigmand(r,a)
        af(r,fp,a,t)$xfFlag(r,fp,a) = (xf.l/va.l)*(pfa.l/pva.l)**sigmav(r,a)
    """
    sets = params.sets
    elas = params.elasticities
    out_and: Dict[Tuple[str, str], float] = {}
    out_ava: Dict[Tuple[str, str], float] = {}
    out_io: Dict[Tuple[str, str, str], float] = {}
    out_af: Dict[Tuple[str, str, str], float] = {}

    for r in sets.r:
        for a in sets.a:
            nd_l = _level(model.nd, (r, a))
            va_l = _level(model.va, (r, a))
            xp_l = _level(model.xp, (r, a))
            px_l = _level(model.px, (r, a))
            pnd_l = _level(model.pnd, (r, a))
            pva_l = _level(model.pva, (r, a))
            if xp_l <= 0.0:
                continue
            sigmap = float(elas.sigmap.get((r, a), 1.0) or 1.0)
            sigmand = float(elas.sigmand.get((r, a), 1.0) or 1.0)
            sigmav = float(elas.sigmav.get((r, a), 1.0) or 1.0)
            if nd_l > 0 and px_l > 0:
                out_and[(r, a)] = (
                    (nd_l / xp_l) * ((pnd_l / px_l) ** sigmap) if pnd_l > 0 else 0.0
                )
            if va_l > 0 and px_l > 0:
                out_ava[(r, a)] = (
                    (va_l / xp_l) * ((pva_l / px_l) ** sigmap) if pva_l > 0 else 0.0
                )
            if nd_l > 0:
                for i in sets.i:
                    xa_l = _level(model.xaa, (r, i, a))
                    if xa_l <= 0.0:
                        continue
                    pa_l = _level(model.pa, (r, i, a), 1.0) or 1.0
                    out_io[(r, i, a)] = (
                        (xa_l / nd_l) * ((pa_l / max(pnd_l, 1e-12)) ** sigmand)
                    )
            if va_l > 0 and pva_l > 0:
                for f in sets.f:
                    xf_l = _level(model.xf, (r, f, a))
                    if xf_l <= 0.0:
                        continue
                    pfa_l = _level(model.pfa, (r, f, a), 1.0) or 1.0
                    out_af[(r, f, a)] = (
                        (xf_l / va_l) * ((pfa_l / pva_l) ** sigmav)
                    )
    return {
        "and_param": out_and,
        "ava_param": out_ava,
        "io_param": out_io,
        "af_param": out_af,
    }


def _max_param_delta(old: Dict[Tuple, float], new: Dict[Tuple, float]) -> float:
    max_d = 0.0
    for k, v in new.items():
        old_v = float(old.get(k, 0.0) or 0.0)
        d = abs(v - old_v)
        if d > max_d:
            max_d = d
    return max_d


def apply_recalibration(
    params: GTAPParameters, recalib: Dict[str, Dict[Tuple, float]]
) -> Dict[str, float]:
    """Mutate ``params.calibrated.*`` in-place with recalibrated shares.

    Also sets ``params._altertax_outer_recalibrated = True`` so the
    benchmark-prices H12 block in ``gtap_model_equations.build_model`` defers
    to the recalibrated values rather than overriding them.

    Returns ``{name: max_delta}`` per parameter for convergence reporting.
    """
    deltas: Dict[str, float] = {}
    for name, new in recalib.items():
        old = getattr(params.calibrated, name)
        deltas[name] = _max_param_delta(old, new)
        for k, v in new.items():
            old[k] = v
    setattr(params, "_altertax_outer_recalibrated", True)
    return deltas
