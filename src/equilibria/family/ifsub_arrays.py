"""The ifSUB report-variable closure on whole arrays of the family state.

``_apply_ifsub_closure`` fixes the nine report variables at the value of their M_*
macro (``blocks/gtap/_ifsub_macros.py``). On a ``StateModel`` the cell-by-cell form goes
through ``_StateVar`` accessors twelve million times on 20x41; here each macro is one
numpy expression over the full grid.

The result is bitwise the one the scalar macros give: operands are combined in the same
order and with the same constants, ``sum`` over the margin modes is a sequential loop
that starts from zero (not ``np.sum``), and the tax parameters that live in the
``GTAPParameters`` dicts are densified through the SAME ``_safe`` helpers the macros call
(so ``float(v or default)`` maps a stored 0.0 or None to the default). No macro reads a
report variable (they read ``pf``, ``p_rai``, ``pe``, ``xw`` and parameters), so the
variables do not depend on one another and the order they are fixed in is irrelevant.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from equilibria.blocks.gtap import _ifsub_macros as mac
from equilibria.family.state import _StateParam, _StateVar


class NotAligned(Exception):
    """An operand is missing or its domains differ from the report variable's."""


def _ref(model: Any, name: str, labels: list[list[str]]) -> np.ndarray:
    """Array of ``model.<name>``, checked to be indexed by exactly ``labels``."""
    comp = model.__dict__["_components"].get(name)
    if isinstance(comp, _StateParam):
        arr = comp.data
    elif isinstance(comp, _StateVar):
        arr = comp._vals
    else:
        raise NotAligned(name)
    if comp._labels != labels:
        raise NotAligned(name)
    return arr


def _dense(fn: Callable[..., float], *axes: list[str]) -> np.ndarray:
    """``fn`` evaluated on every cell of the grid ``axes``, as a float array."""
    out = np.empty(tuple(len(a) for a in axes))
    for pos, key in zip(
        itertools.product(*(range(len(a)) for a in axes)),
        itertools.product(*axes),
        strict=True,
    ):
        out[pos] = fn(*key)
    return out


def _pwmg(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    modes = model.__dict__["sets"].get("m")
    if modes is None:
        raise NotAligned("m")
    amgm = _ref(model, "amgm", [modes, *lab])
    ptmg = _ref(model, "ptmg", [modes])
    lam = _ref(model, "lambdamg", [modes, *lab])
    acc = np.zeros(amgm.shape[1:])
    for i in range(len(modes)):
        acc = acc + amgm[i] * ptmg[i] / (lam[i] + 1e-12)
    return acc


def _pefob(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    e, c, imp = lab
    rtxs = _dense(lambda a, b, d: mac._safe(params, "taxes.rtxs", (a, b, d), 0.0), *lab)
    etax = _dense(lambda a, b: mac.etax_value(params, a, b, ""), e, c)
    return (1.0 + rtxs + etax[:, :, None]) * _ref(model, "pe", lab)


def _pmcif(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    return _pefob(model, params, lab) + _pwmg(model, params, lab) * _ref(
        model, "tmarg", lab
    )


def _pm(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    e, c, imp = lab
    mtax = _dense(lambda i, k: mac.mtax_value(params, i, k, ""), imp, c)  # (imp, c)
    chipm = _dense(lambda a, b, d: mac.chipm_value(params, a, b, d), *lab)
    return (
        (1.0 + _ref(model, "imptx", lab) + mtax.T[None, :, :])
        * _pmcif(model, params, lab)
    ) / chipm


def _xwmg(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    return _ref(model, "tmarg", lab) * _ref(model, "xw", lab)


def _xmgm(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    # ``lab`` is the full (mode, e, c, imp) grid; the margin base is the (e, c, imp) one.
    amgm = _ref(model, "amgm", lab)
    lam = _ref(model, "lambdamg", lab)
    return amgm * _xwmg(model, params, lab[1:])[None] / (lam + 1e-12)


def _pfa(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    return _ref(model, "pf", lab) * (
        1.0 + _ref(model, "fctts", lab) + _ref(model, "fcttx", lab)
    )


def _pfy(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    kappa = _dense(lambda r, f, a: mac.kappaf_value(params, r, f, a), *lab)
    return _ref(model, "pf", lab) * (1.0 - kappa)


def _pp(model: Any, params: Any, lab: list[list[str]]) -> np.ndarray:
    return (1.0 + _ref(model, "prdtx_rai", lab)) * _ref(model, "p_rai", lab)


_ARRAY_MACROS: dict[str, Callable[[Any, Any, list[list[str]]], np.ndarray]] = {
    "pfa": _pfa,
    "pfy": _pfy,
    "pp_rai": _pp,
    "pwmg": _pwmg,
    "pefob": _pefob,
    "pmcif": _pmcif,
    "pm": _pm,
    "xwmg": _xwmg,
    "xmgm": _xmgm,
}


def fix_report_var(
    state: Any,
    params: Any,
    var_name: str,
    scalar_macro: Callable[..., Any],
    arity: int,
    macro_fallos: list[str],
) -> bool:
    """Fix the free cells of one report variable from its macro, on arrays.

    Returns False, touching nothing, when an operand is not aligned with the variable
    (the caller then runs the cell-by-cell code). Cells whose array value is not finite
    are re-evaluated with the scalar macro, which raises exactly where the cell-by-cell
    code does; those failures go to ``macro_fallos`` (and the cell keeps its level, as
    there) for the caller to raise.
    """
    var = state.__dict__["_components"].get(var_name)
    if not isinstance(var, _StateVar) or len(var.domains) != arity:
        return False
    try:
        with np.errstate(all="ignore"):
            target = np.array(
                _ARRAY_MACROS[var_name](state, params, var._labels), dtype=float
            )
    except NotAligned:
        return False
    if target.shape != var.shape:
        return False
    free = ~var._fx
    for pos in np.argwhere(free & ~np.isfinite(target)):
        pos = tuple(int(i) for i in pos)
        key = tuple(lab[i] for lab, i in zip(var._labels, pos, strict=True))
        try:
            target[pos] = float(scalar_macro(state, params, *key))
        except Exception as exc:  # noqa: BLE001 — accumulated and re-raised by the caller
            macro_fallos.append(f"{var_name}{key}: {type(exc).__name__} {exc}")
            target[pos] = var._vals[pos]
    # Respect the bounds as the cell code does: clamp to lb, then to ub.
    target = np.where(target < var._lo, var._lo, target)
    target = np.where(target > var._hi, var._hi, target)
    var._vals[free] = target[free]
    var._fx[free] = True
    return True
