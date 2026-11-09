"""Percent changes of ALTERTAX on the family route: the content of GEMPACK's ``solution.sl4``.

``pct_changes(run)`` is ``100 * (shock / check - 1)`` of every GEMPACK variable that has an
equilibria counterpart (cells where the ``check`` value is 0 have no change, as in
:mod:`volume`). ``check`` reproduces the base and ``shock`` is the solved shock, so the
ratio is the cumulative % change of a GEMPACK multistep solution. Cells are keyed by the
GEMPACK element labels in the GEMPACK dimension order.

Names and shapes come from ``compare/maps/gtap7.yaml`` (the GEMPACK <-> equilibria
variable map in levels: ``a`` equilibria, ``b`` GEMPACK; ``gtap7_names.yaml`` only pairs
GAMS names). The equilibria side of each row (including the derived ones such as
``xds_net`` or ``xw_fob``) is built by ``scripts/gtap/compare_gempack_levels.equilibria_side``;
scales cancel in a ratio, so ``divide_by`` is ignored. The closure's fixed variables that
map one to one (``pop``, ``pfactwld``, ``to``) are added from :data:`FIXED`: they change by
0, except ``to``, which carries the shock. The other fixed GEMPACK variables are many to
one (``tfd``, ``tgd``... share ``dintx_tgt``) or have no equilibria cell: they are not
emitted (see :func:`pct_changes`).

A reader for a ``.sl4`` is :func:`equilibria.babel.har.sl4.read_sl4_changes`;
:func:`gempack_changes` turns it, or any ``{var: {cell: pct}}`` dict (e.g. one read from a
``sl4dump.har``), into the shape :func:`pct_changes` returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har.sl4 import Sl4Change, Sl4Changes
from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.compare import Mapping
from equilibria.compare.mapping import VariableMap
from equilibria.family.altertax.compare import load_compare_script
from equilibria.family.altertax.levels import family_levels
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.updated import _elements, _Period

Pct = dict[str, dict[tuple[str, ...], float]]

# GEMPACK set name of each dimension label of gtap7.yaml
_SET = {
    "REG": "REG",
    "SRC": "REG",
    "DST": "REG",
    "COMM": "COMM",
    "ACT": "ACTS",
    "ENDW": "ENDW",
    "MARG": "MARG",
}

# gtap7.yaml rows that are not GEMPACK variables, with the reason
NOT_VARIABLES = {
    "vdfb": "VDFB is a data coefficient (basedata.har), not a variable of the .sl4"
}

# GEMPACK variables that are the % change of a POWER (1 + rate) of an equilibria rate
POWER = ("to",)

# fixed GEMPACK variable -> (equilibria variable, axis permutation to GEMPACK order,
# GEMPACK set of each axis); equilibria cells are (..., period)
FIXED: dict[str, tuple[str, tuple[int, ...], tuple[str, ...]]] = {
    "pop": ("pop", (0,), ("REG",)),
    "pfactwld": ("pnum", (), ()),
    "to": ("prdtx_rai", (2, 1, 0), ("COMM", "ACTS", "REG")),  # prdtx_rai[r,a,i]
}


def _ratio(pre: float, post: float) -> float:
    return 0.0 if pre == 0.0 else 100.0 * (post / pre - 1.0)


def gtap7_rows() -> dict[str, VariableMap]:
    """The compared rows of ``gtap7.yaml`` by GEMPACK variable name."""
    return {
        v.b.var: v
        for v in Mapping.builtin("gtap7").variables
        if v.name not in NOT_VARIABLES
    }


def pct_dims(name: str) -> tuple[str, ...]:
    """GEMPACK set names of each dimension of ``name`` (as written by :func:`write_pct_har`)."""
    if name in FIXED:
        return FIXED[name][2]
    return tuple(_SET[d] for d in gtap7_rows()[name].b.dims)


def _changes(run: AltertaxRun, pre: str, post: str) -> tuple[Pct, dict[str, set]]:
    """``(% changes, cells whose pre value is 0)``: the second has no change to report."""
    cg = load_compare_script()
    levels = family_levels(run.result.mp, cg.EQ_VARS)
    from equilibria.blocks.gtap._derived_params import xscale_data

    side = cg.equilibria_side(levels, dict(xscale_data(run.params, run.params.sets)))
    out: Pct = {}
    zero: dict[str, set] = {}
    for name, row in gtap7_rows().items():
        if name in FIXED:  # pfactwld: FIXED overrides the gtap7.yaml row (pwfact)
            continue
        cells = side.levels[row.a.var]
        perm = [row.a.dims.index(d) for d in row.b.dims]
        out[name], zero[name] = {}, set()
        for period, idx in cells:
            if period != pre:
                continue
            key = tuple(idx[p] for p in perm)
            out[name][key] = _ratio(cells[(pre, idx)], cells[(post, idx)])
            if cells[(pre, idx)] == 0.0:
                zero[name].add(key)
    A, B = _Period(run, pre), _Period(run, post)
    for name, (var, perm, _) in FIXED.items():
        one = 1.0 if name in POWER else 0.0
        out[name] = {
            tuple(k[p] for p in perm): _ratio(one + x, one + B[var][k])
            for k, x in A[var].items()
        }
    return out, {n: c for n, c in zero.items() if c}


def pct_changes(run: AltertaxRun, pre: str = "check", post: str = "shock") -> Pct:
    """``{GEMPACK variable: {cell: 100 * (post / pre - 1)}}``; cells in GEMPACK order.

    Emitted: the 31 compared rows of ``gtap7.yaml`` plus the closure's fixed variables
    ``pop``, ``pfactwld`` and ``to`` (:data:`FIXED`). The other fixed GEMPACK variables
    (``tfd``, ``tgd``, ``tm``, ``tx``, ``atm``... many to one with equilibria's
    ``dintx_tgt``/``imptx``/``lambdamg``, or with no equilibria cell) are NOT emitted:
    they change by 0 in GEMPACK too, but no cell-by-cell answer exists for them here.
    A cell whose ``pre`` value is 0 reports 0 (see :func:`zero_base_cells`)."""
    return _changes(run, pre, post)[0]


def zero_base_cells(run: AltertaxRun, pre: str = "check") -> dict[str, set]:
    """Cells (by GEMPACK variable) whose ``pre`` flow is 0, where :func:`pct_changes`
    reports 0. A % change on a zero flow is undefined: GEMPACK still reports the value of
    its price-difference formula there (``E_qes2`` holds for every cell regardless of the
    weight), so these cells are not comparable and are excluded from the gate."""
    return _changes(run, pre, pre)[1]


def read_sl4dump(path: Path) -> Sl4Changes:
    """The % changes of a ``sl4dump.har`` (``sltoht`` HAR output of the cumulative
    solution) in the shape ``read_sl4_changes`` returns. Headers ``0001...`` hold one
    variable each: its name is the first word of ``long_name`` (``"qo # output of ... #"``)
    and every variable is read as a percent change (``kind == "p"``); a variable with a
    zero-sized set (``qesf``) has no cells. Level-only and creation headers are skipped.

    Limitation: the HAR does not say whether a variable is a percent change or an
    ordinary change (``Variable (change)`` of the tab, e.g. ``del_tbal``, ``del_tbalry``,
    ``EV``): all are labelled ``"p"``. The gates read only the percent variables of
    ``gtap7.yaml`` and ``pop``/``pfactwld``/``to``, so the label is harmless there; do not
    read the change variables of this result as percentages."""
    from equilibria.babel.har.reader import read_har
    from equilibria.babel.har.sl4_levels import is_leva_header

    out: dict[str, Sl4Change] = {}
    for hid, ha in read_har(str(path)).items():
        if not is_leva_header(hid):
            continue
        name = ha.long_name.split("#", 1)[0].strip()
        elems = [[str(e).strip() for e in es] for es in ha.set_elements]
        arr = np.asarray(ha.array, dtype=np.float64)
        shape = tuple(len(e) for e in elems)
        out[name] = Sl4Change(elems, "p", arr.reshape(shape) if arr.size else arr)
    return Sl4Changes(out, {})


def gempack_changes(changes: Sl4Changes | Pct) -> Pct:
    """GEMPACK's % changes in the shape of :func:`pct_changes`: from the result of
    ``read_sl4_changes`` (variables of kind ``p`` only) or from a ready
    ``{var: {cell: pct}}`` dict (returned as is)."""
    if not isinstance(changes, Sl4Changes):
        return changes
    out: Pct = {}
    for name, ch in changes.changes.items():
        if ch.kind != "p":
            continue
        out[name] = {
            tuple(ch.set_elements[d][i] for d, i in enumerate(ix)): float(ch.values[ix])
            for ix in np.ndindex(ch.values.shape)
        }
    return out


def pct_headers(run: AltertaxRun, pct: Pct | None = None) -> dict[str, HeaderArray]:
    """One header per GEMPACK variable (header id: the upper-cased name if it has at most 4
    characters, else its first two and a counter, see ``_header_ids``); ``coeff_name`` and
    ``long_name`` are the GEMPACK variable name."""
    return _headers(run, pct_changes(run) if pct is None else pct)


def write_pct_har(run: AltertaxRun, path: Path) -> None:
    """Write the % changes of ``run`` to ``path``, one header per GEMPACK variable."""
    write_har(path, pct_headers(run))


def _header_ids(names: list[str]) -> dict[str, str]:
    """Unique header ids of at most 4 characters: a short name is its own id in upper case, a longer
    one becomes its first two characters and a two-digit counter (``PF01``)."""
    ids: dict[str, str] = {n: n.upper() for n in names if len(n) <= 4}
    taken = set(ids.values())
    for n in names:
        if n in ids:
            continue
        for k in range(100):
            cand = f"{n[:2].upper()}{k:02d}"
            if cand not in taken:
                ids[n] = cand
                taken.add(cand)
                break
        else:  # pragma: no cover - 100 names with one prefix
            raise ValueError(f"no free header id for {n!r}")
    return ids


def _headers(run: AltertaxRun, pct: Pct) -> dict[str, HeaderArray]:
    els = _elements(run)
    names = sorted(pct)
    ids = _header_ids(names)
    out: dict[str, HeaderArray] = {}
    for name in names:
        sets = list(pct_dims(name))
        elems = [els[s] for s in sets]
        shape = tuple(len(e) for e in elems)
        arr = np.zeros(shape if shape else (1,))
        for cell, x in pct[name].items():
            arr[tuple(e.index(c) for e, c in zip(elems, cell, strict=True)) or (0,)] = x
        out[ids[name]] = HeaderArray(
            name=ids[name],
            coeff_name=name,
            long_name=name,
            array=arr,
            set_names=sets,
            set_elements=elems,
        )
    return out
