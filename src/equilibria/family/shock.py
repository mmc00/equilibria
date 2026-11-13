"""Shocks on the multiperiod arrays: ``check_shock_cell`` and ``apply_shock`` without Pyomo.

Ports ``templates/gtap/instruments.check_shock_cell`` (which cells of a ShockBlock
instrument can take a shock) and ``templates/gtap/shocks.apply_shock`` (write a
``{instrument: {cell: pct}}`` GEMPACK shock into the shock period, all or nothing) over
``MPArrays``. The Pyomo versions read the live ``ConcreteModel``; here:

* "exists" / "live" are the cell and the active flag of the ``MPArrays`` var / row;
* the one place Pyomo reads an expression, ``_read_inline`` (under ifSUB the tariff
  enters ``eq_xweq`` through a macro, not through ``eq_pmeq``), asks whether the cell is
  among the variable leaves the recording kept for that row, **before** the zero folding
  (``family/fscale.row_leaves``), which is what ``identify_variables`` lists;
* ``sys`` is the single-period recording (``FamilyBackend``: ``records``,
  ``var_domains``, ``sets``).

The percent semantics, the order of the checks, the all-or-nothing write and the errors
are those of the Pyomo functions; the shock goes to the ``shock`` period only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from equilibria.blocks.gtap.periods import CHECK, SHOCK
from equilibria.family.fscale import row_leaves
from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr
from equilibria.family.prep_recipe import _table
from equilibria.templates.gtap.instruments import (
    INLINE_EQS as PYOMO_INLINE_EQS,
)
from equilibria.templates.gtap.instruments import (
    INSTRUMENT_EQS,
    check_shock_value,
    is_free_cell,
    period_key,
)
from equilibria.templates.gtap.shocks import gempack_kind, shocked

# Family-local: rows that read an instrument inline under ifSUB=1. The Pyomo route's map
# (``PYOMO_INLINE_EQS``) has only imptx and rejects prdtx_rai under ifSUB=1; the array
# route also accepts it, because M_PP (``_m_pp`` in blocks/gtap/production_supply.py)
# carries ``prdtx_rai[r,a,i]`` into ``eq_peq[r,a,i]`` while ``eq_pp_rai`` is off.
# A documented divergence from Pyomo, needed by ALTERTAX (``to(AGR,AGR,USA)``).
INLINE_EQS: dict[str, tuple[str, ...]] = {
    **PYOMO_INLINE_EQS,
    "prdtx_rai": ("eq_peq",),
}


def _flat(sys: Any, name: str, cell: tuple) -> int:
    """Flat C-order index of ``cell`` in the recording's variable ``name`` (a leaf's
    second element)."""
    cache = sys.__dict__.setdefault("_shock_pos", {})
    if "sets" not in cache:
        cache["sets"] = {
            d: {e: i for i, e in enumerate(es)} for d, es in sys.sets.items()
        }
    doms = sys.var_domains[name]
    if not doms:
        return 0
    pos = [cache["sets"][d][e] for d, e in zip(doms, cell, strict=True)]
    shape = tuple(len(sys.sets[d]) for d in doms)
    return int(np.ravel_multi_index(pos, shape))


def _leaves(sys: Any, eq_name: str, cell: tuple) -> set[tuple[str, int]]:
    """Variable leaves of the recorded row ``eq_name[cell]`` (cached per family)."""
    cache = sys.__dict__.setdefault("_shock_leaves", {})
    if eq_name not in cache:
        cache[eq_name] = {
            tuple(idx): res
            for idx, res in sys.records.get(eq_name, ())  # a family may be absent
        }
    res = cache[eq_name].get(tuple(cell))
    if res is None:
        return set()
    key = ("leaves", eq_name, tuple(cell))
    if key not in cache:
        cache[key] = row_leaves(res)
    return cache[key]


def _live(mp: MPArrays, eq_name: str, cell: tuple) -> bool:
    """The row ``eq_name`` has a cell at ``cell`` in the shock period and it is active."""
    eq = mp.rows.get(eq_name)
    if eq is None:
        return False
    i = _table(eq, SHOCK).get(tuple(cell))
    return i is not None and bool(eq.active[i])


def _read_inline(mp: MPArrays, sys: Any, name: str, cell: tuple) -> bool:
    """A live row of ``INLINE_EQS[name]`` at ``cell`` reads ``name[cell]``."""
    leaf = (name, _flat(sys, name, cell))
    return any(
        _live(mp, eq_name, cell) and leaf in _leaves(sys, eq_name, cell)
        for eq_name in INLINE_EQS.get(name, ())
    )


def _labels(comp: VarArr | RowArr | None, region: str | None, *, live: bool) -> list:
    """Second index of the shock cells of ``region`` (live ones only if ``live``)."""
    if comp is None:
        return []
    act = comp.active if isinstance(comp, RowArr) else None
    return sorted(
        {
            k[1]
            for i, k in enumerate(comp.keys)
            if len(k) > 1
            and k[0] == region
            and k[-1] == SHOCK
            and (act is None or not live or bool(act[i]))
        }
    )


def check_instrument_cell(mp: MPArrays, sys: Any, name: str, cell: tuple) -> None:
    """``ValueError`` if ``name[*cell, 'shock']`` cannot take a shock: the cell does not
    exist, no live row of ``INSTRUMENT_EQS`` (or ``INLINE_EQS``) reads it, or it is
    ``aft`` with ``xft`` fixed."""
    var = mp.vars.get(name)
    idx = period_key(cell, SHOCK)
    region = cell[0] if cell else None
    i = None if var is None else _table(var, SHOCK).get(tuple(cell))
    if i is None:
        raise ValueError(
            f"{name}{idx} does not exist. Valid for {region}: "
            f"{_labels(var, region, live=False)}"
        )
    for eq_name in INSTRUMENT_EQS.get(name, ()):
        if not _live(mp, eq_name, cell):
            if _read_inline(mp, sys, name, cell):
                continue
            raise ValueError(
                f"{name}{idx}: {eq_name}{idx} does not exist or is inactive, the "
                "shock would not enter the model. Live for "
                f"{region}: {_labels(mp.rows.get(eq_name), region, live=True)}"
            )
    if name == "aft":
        xft = mp.var("xft")
        if xft.fixed[xft.pos((*cell, SHOCK))]:
            raise ValueError(
                f"aft{idx}: xft{idx} is fixed; the solver would deactivate eq_xfteq "
                "and the shock would be lost"
            )


def check_shock_cell(mp: MPArrays, sys: Any, name: str, cell: tuple) -> None:
    """``ValueError`` if ``name[*cell, 'shock']`` cannot take a shock: not a registered
    instrument, an endogenous (``@overwrite``) cell, or no live row reads it."""
    if name not in mp.exogenous:
        raise ValueError(
            f"{name!r} is not a registered instrument; registered: {sorted(mp.exogenous)}"
        )
    libres = mp.endogenous_instrument_cells.get(name)
    if libres and is_free_cell(period_key(cell, SHOCK), libres):
        raise ValueError(
            f"{name}{tuple(cell)} is endogenous (@overwrite): it cannot take a shock"
        )
    check_instrument_cell(mp, sys, name, tuple(cell))


def can_take_shock(mp: MPArrays, sys: Any, name: str, cell: tuple) -> bool:
    """``check_shock_cell`` as a predicate: only its ``ValueError`` says no."""
    try:
        check_shock_cell(mp, sys, name, cell)
    except ValueError:
        return False
    return True


def _check_value(mp: MPArrays, name: str, cell: tuple) -> float:
    v = mp.var(name)
    return float(v.value[v.pos((*cell, CHECK))])


def level(mp: MPArrays, name: str, cell: tuple, pct: float) -> float:
    """Level of ``name[*cell, 'shock']`` for the GEMPACK percent ``pct``."""
    g = 1.0 + float(pct) / 100.0
    if not g > 0.0:
        raise ValueError(f"{name}{cell}: {pct}% — a GEMPACK change must be > -100%")
    chk = _check_value(mp, name, cell)
    kind = gempack_kind(name)
    fs = 0.0
    if kind == "power_fct":
        if mp.fctts is None:
            raise ValueError(
                f"{name}{cell}: the model carries no fctts (build_block_model)"
            )
        fs = float(mp.fctts.get(cell, 0.0))
    return float(shocked(kind, chk, g, fs))


def apply_shock(
    mp: MPArrays,
    sys: Any,
    shock: Mapping[str, Mapping[tuple, float]],
    *,
    levels: bool = False,
) -> dict[str, dict[tuple, float]]:
    """Write ``shock`` (``{instrument: {cell: %}}``) into the shock period of ``mp``, in
    place, as ``shocks.apply_shock`` does on the model.

    All or nothing: every cell is validated and converted before any is fixed. Returns
    the levels fixed. ``ValueError`` as the Pyomo function: unregistered instrument,
    missing / endogenous / unread cell, percent <= -100, value outside the instrument's
    domain, or a value equal to the check one."""
    out: dict[str, dict[tuple, float]] = {}
    for name, cells in shock.items():
        for cell, x in cells.items():
            cell = tuple(cell)
            check_shock_cell(mp, sys, name, cell)  # before reading its check
            v = float(x) if levels else level(mp, name, cell, x)
            check_shock_value(name, cell, v, _check_value(mp, name, cell))
            out.setdefault(name, {})[cell] = v
    for name, cells in out.items():
        var = mp.var(name)
        for cell, v in cells.items():
            i = var.pos((*cell, SHOCK))
            var.value[i] = v
            var.fixed[i] = True
    return out


__all__ = ["apply_shock", "can_take_shock", "check_shock_cell", "level"]
