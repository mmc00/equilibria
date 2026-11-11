"""Incidence of one period read from the family backend's Jacobian pattern.

``prep_square.square`` pairs rows with columns from the variables each row mentions.
Pyomo supplies that incidence as strings; here it comes from the J pattern of the
backend's recorded rows, so the period prep needs no Pyomo model. The cross-period
rows (Fisher / Tornqvist indices and their ``mq_*`` aggregates) come from their own
recording (``family.cross_period``); ``proxies=True`` keeps the old unit placeholder
for scripts that time the system without them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp

from equilibria.family.cell_index import CellIndex, SquaredIds
from equilibria.family.layout import VarLayout
from equilibria.family.mp_arrays import MPArrays, Squared
from equilibria.family.recorder import Node

# Cross-period rows (the Fisher/Tornqvist block of ``build_equations_fisher``): the
# backend records them in their single-period form, which is not the multiperiod row,
# so they are left out and the caller supplies them. Component names as defined in
# gtap_model_multiperiod.py (``eq_pabs_base``, ``eq_pfact_base`` are the base-period
# twins). The ``eq_mq_*`` rows (the Fisher aggregates' own equations, e.g. 13 such
# proxy families on 20x41) are matched by prefix.
CROSS_PERIOD_FAMILIES = frozenset(
    {
        "eq_rgdpmp",
        "eq_pgdpmp",
        "eq_pabs",
        "eq_pabs_base",
        "eq_pfact",
        "eq_pfact_base",
        "eq_pwfact",
        "eq_chisave",
    }
)
_CROSS_PERIOD_PREFIX = "eq_mq_"


def is_cross_period(family: str) -> bool:
    return family in CROSS_PERIOD_FAMILIES or family.startswith(_CROSS_PERIOD_PREFIX)


def _all_free_layout(backend: Any, state: Any) -> VarLayout:
    """Every variable cell a column, Parameters fixed slots (the pattern is the same
    whichever cells a closure fixes later)."""
    values, free = {}, {}
    for v in backend.var_domains:
        values[v] = np.asarray(state.values[v], dtype=float)
        free[v] = np.ones(values[v].shape, dtype=bool)
    for p in backend.param_domains:
        values[p] = state.find_component(p).data
        free[p] = np.zeros(values[p].shape, dtype=bool)
    domains = {**backend.var_domains, **backend.param_domains}
    return VarLayout.from_state(state.sets, domains, values, free)


def _with_period(name: str, period: str) -> str:
    """``eq[a,b]`` → ``eq[a,b,t]``; ``eq`` → ``eq[t]``."""
    if not name.endswith("]"):
        return f"{name}[{period}]"
    return f"{name[:-1]},{period}]"


class PatternCSR(NamedTuple):
    """Structure of a system's J as integer CSR: row ``i`` has the layout columns
    ``indices[indptr[i]:indptr[i + 1]]``, ascending."""

    indptr: np.ndarray
    indices: np.ndarray
    row_names: list[str]
    col_names: list[str]
    family_off: dict[str, int]  # first row of each family
    family_len: dict[str, int]


def _pattern_csr(system: Any, layout: VarLayout) -> PatternCSR:
    """The stored pattern of ``system`` (``FamilySystem._pat_rows/_pat_cols``: unique
    (row, col) pairs sorted by row, then column) as CSR, without evaluating J."""
    n = system.n_rows
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(system._pat_rows, minlength=n), out=indptr[1:])
    off, size, start = {}, {}, 0
    for f in system.families:
        off[f.name], size[f.name] = start, f.n_rows
        start += f.n_rows
    return PatternCSR(
        indptr,
        np.asarray(system._pat_cols, dtype=np.int64),
        list(system.row_names),
        list(layout.col_names),
        off,
        size,
    )


def _pattern(backend: Any, layout: VarLayout, period: str | None) -> dict[str, list]:
    """Row name -> columns of its J pattern. ``period`` is appended to the single-period
    names; ``None`` keeps them (the cross-period names carry their period)."""
    pat = _pattern_csr(backend.system(layout), layout)
    cols, rows = pat.col_names, pat.row_names
    if period is not None:
        cols = [_with_period(n, period) for n in cols]
        rows = [_with_period(n, period) for n in rows]
    ind, ptr = pat.indices, pat.indptr
    return {
        name: [cols[j] for j in ind[ptr[i] : ptr[i + 1]].tolist()]
        for i, name in enumerate(rows)
    }


def incidence_csr(backend: Any, state: Any) -> PatternCSR:
    """The single-period J pattern with every variable cell free, once per backend
    (the pattern does not depend on the closure, the period or the levels)."""
    cached = getattr(backend, "_incidence_csr", None)
    if cached is None:
        layout = _all_free_layout(backend, state)
        cached = backend._incidence_csr = _pattern_csr(backend.system(layout), layout)
    return cached


def _cross_pattern(cross: Any) -> dict[str, list[str]]:
    """Columns of every cross-period row (all periods, all cells free), once per
    recording.

    Read off the records: with every cell free nothing is folded, so the J pattern of a
    row is the set of its variable leaves, in column order. Same rows (family order,
    then record order) and columns as evaluating J, without compiling or differentiating
    the rows."""
    cached = getattr(cross, "_pattern_cache", None)
    if cached is None:
        offsets, names = {}, []
        for v, doms in cross.var_domains.items():
            offsets[v] = len(names)
            for idx in itertools.product(*(cross.sets[d] for d in doms)):
                names.append(f"{v}[{','.join(idx)}]" if idx else v)
        cached = {}
        for fam, recs in cross.records.items():
            for idx, res in recs:
                lab = ",".join(str(i) for i in idx)
                cols = {
                    offsets[v] + flat
                    for node in (res.lhs, res.rhs)
                    if isinstance(node, Node)
                    for v, flat in node.var_leaves()
                }
                cached[f"{fam}[{lab}]" if lab != "" else fam] = [
                    names[c] for c in sorted(cols)
                ]
        cross._pattern_cache = cached
    return cached


def _single_period_incidence(backend: Any, state: Any, period: str) -> dict:
    """The single-period rows' incidence of ``period``, once per backend and period."""
    cache = backend.__dict__.setdefault("_incidence_by_period", {})
    hit = cache.get(period)
    if hit is None:
        pat = incidence_csr(backend, state)
        cols = [_with_period(n, period) for n in pat.col_names]
        ind, ptr = pat.indices, pat.indptr
        hit = cache[period] = {
            _with_period(name, period): [
                cols[j] for j in ind[ptr[i] : ptr[i + 1]].tolist()
            ]
            for i, name in enumerate(pat.row_names)
            if not is_cross_period(name.partition("[")[0])
        }
    return hit


def period_incidence(
    backend: Any, state: Any, period: str, cross: Any = None
) -> dict[str, list[str]]:
    """For every recorded row ``eq[idx]``, the cells ``v[idx',period]`` in its J pattern.

    Keys and columns carry the period suffix (``eq_xp[USA,agr,check]``), the format
    ``square(incidence=...)`` expects. All variable cells are columns, so structural
    zeros stay in the pattern. Rows the backend does not record are absent. With
    ``cross`` (the cross-period recording) the rows of ``period`` it records are added,
    their columns being the cells of every period they read."""
    out = dict(_single_period_incidence(backend, state, period))
    if cross is not None:
        for name, cols in _cross_pattern(cross).items():
            if name.endswith(f",{period}]") or name.endswith(f"[{period}]"):
                out[name] = cols
    return out


class ArrayIncidence:
    """``period_incidence`` as integer adjacency for ``square``, without name strings.

    ``adjacency(mp, index, rid, col_pos)`` returns, for each row id of ``rid`` (in that
    order), the sorted positions of the free columns it reads, in the order of
    ``period_incidence`` (layout order for the recorded rows, the cross pattern's order
    for the cross-period ones), each once. It is ``prep_square._adjacency`` on the names
    of ``period_incidence``, computed on the CSR pattern."""

    def __init__(self, backend: Any, state: Any, period: str, cross: Any = None):
        self.backend, self.state, self.period, self.cross = (
            backend,
            state,
            period,
            cross,
        )
        self.pat = incidence_csr(backend, state)
        self._cross_pat = _cross_pattern(cross) if cross is not None else {}

    def _tables(self, mp: MPArrays, index: Any):
        """Per run and period: csr row of every mp row (-1: not a recorded row) and
        global variable id of every layout column (-1: not in ``mp``)."""
        key = ("incidence", id(self.backend), self.period)
        hit = index.cache.get(key)
        # id() of a freed backend can be reused: keep the backend in the entry
        if hit is not None and hit[0] is self.backend:
            return hit[1]
        pat = self.pat
        g2csr = np.full(len(index.row_names), -1, dtype=np.int64)
        for fam, row in mp.rows.items():
            base = pat.family_off.get(fam)
            if base is None or is_cross_period(fam):
                continue
            where = {idx: k for k, (idx, _) in enumerate(self.backend.records[fam])}
            ids = index.period_ids("row", fam, self.period, row)
            k = np.array([where.get(row.keys[i][:-1], -1) for i in ids.tolist()])
            ok = k >= 0
            g2csr[index.row_off[fam] + ids[ok]] = base + k[ok]
        cm = np.full(len(pat.col_names), -1, dtype=np.int64)
        off = 0
        for v, doms in self.backend.var_domains.items():
            cells = itertools.product(*(self.state.sets[d] for d in doms))
            flat = {idx: k for k, idx in enumerate(cells)}
            var = mp.vars.get(v)
            if var is not None:
                ids = index.period_ids("var", v, self.period, var)
                k = np.array([flat.get(var.keys[i][:-1], -1) for i in ids.tolist()])
                ok = k >= 0
                cm[off + k[ok]] = index.var_off[v] + ids[ok]
            off += len(flat)
        index.cache[key] = (self.backend, (g2csr, cm))
        return g2csr, cm

    def _cross_table(self, index: Any) -> dict[str, list[int]]:
        """Per run: global ids of the columns of every cross row (name -> ids)."""
        key = ("cross_gids", id(self.cross))
        hit = index.cache.get(key)
        if hit is not None and hit[0] is self.cross:
            return hit[1]
        table: dict[str, list[int]] = {}
        index.cache[key] = (self.cross, table)
        return table

    def adjacency(
        self, mp: MPArrays, index: Any, rid: np.ndarray, col_pos: np.ndarray
    ) -> list[list[int]]:
        g2csr, cm = self._tables(mp, index)
        pat = self.pat
        n = len(rid)
        csr_row = g2csr[rid]
        recorded = csr_row >= 0
        r = np.where(recorded, csr_row, 0)
        starts = pat.indptr[r]
        cnt = np.where(recorded, pat.indptr[r + 1] - starts, 0)
        total = int(cnt.sum())
        first = np.cumsum(cnt) - cnt
        flat = np.repeat(starts - first, cnt) + np.arange(total, dtype=np.int64)
        gid = cm[pat.indices[flat]]
        # position of the column in the squared system (-1: dropped / not free)
        pos = np.where(gid >= 0, col_pos[np.where(gid >= 0, gid, 0)], -1)
        keep = pos >= 0
        owner = np.repeat(np.arange(n, dtype=np.int64), cnt)
        kept = np.bincount(owner[keep], minlength=n)
        vals = pos[keep].tolist()
        ends = np.cumsum(kept).tolist()
        out: list[list[int]] = []
        lo = 0
        for hi in ends:
            out.append(vals[lo:hi])
            lo = hi
        unrecorded = np.flatnonzero(~recorded).tolist()
        if unrecorded:
            table = self._cross_table(index)  # identity checked once per call
            for i in unrecorded:
                out[i] = self._cross_row(
                    mp, index, index.row_names[rid[i]], col_pos, table
                )
        return out

    def _cross_row(
        self,
        mp: MPArrays,
        index: Any,
        name: str,
        col_pos: np.ndarray,
        table: dict[str, list[int]],
    ) -> list[int]:
        p = self.period
        names = self._cross_pat.get(name)
        if names is None or not (name.endswith(f",{p}]") or name.endswith(f"[{p}]")):
            raise KeyError(f"no incidence for row {name!r}")
        gids = table.get(name)
        if gids is None:
            gids = [index.gid_of_name(c, mp) for c in names]
            gids = table[name] = [-1 if g is None else g for g in gids]
        seen: set[int] = set()
        out: list[int] = []
        for g in gids:
            j = int(col_pos[g]) if g >= 0 else -1
            if j >= 0 and j not in seen:
                seen.add(j)
                out.append(j)
        return out


# Variables of the single-period model that the multiperiod model drops together with
# their rows (the intra-period Fisher aggregates, see hybrid._pyomo_layout): a fixed
# slot at the state's level. Any other family variable missing from ``mp`` raises.
SP_ONLY_VARS = frozenset({"mfr_bs", "mfr_sb", "mfr_ss", "mfw_bs", "mfw_sb", "mfw_ss"})


def _cell_name(var: str, idx: tuple, period: str) -> str:
    return f"{var}[{','.join((*idx, period))}]"


def _flat_index(var: Any, doms: tuple[str, ...], sets: dict[str, list[str]]):
    """Flat C-order position of every cell of ``var`` in the dense array over ``doms``
    (cached on the arrays: keys never change). ``KeyError`` for a key element outside
    the sets."""
    cache = var.__dict__.setdefault("_dense_flat", {})
    # the labels themselves, not id(sets): a freed dict's id can be reused
    key = (doms, tuple(tuple(sets[d]) for d in doms))
    if key not in cache:
        pos = [{e: i for i, e in enumerate(sets[d])} for d in doms]
        shape = [len(sets[d]) for d in doms]
        flat = np.empty(len(var.keys), dtype=np.int64)
        for n, k in enumerate(var.keys):
            f = 0
            for p, size, e in zip(pos, shape, k, strict=True):
                f = f * size + p[e]
            flat[n] = f
        cache[key] = flat
    return cache[key]


class _SPTables(NamedTuple):
    cells: dict[
        str, np.ndarray | None
    ]  # var -> global id of every flat cell (-1: none)
    rows: dict[str, np.ndarray]  # family -> global row id of every record (-1: none)


def _single_period_tables(
    backend: Any, state: Any, mp: MPArrays, period: str, index: CellIndex
) -> _SPTables:
    """Per run, backend and period: where the single-period layout's cells and the
    backend's records sit in the arrays (the keys never change within a run)."""
    key = ("build_sp", id(backend), period)
    hit = index.cache.get(key)
    if hit is not None and hit[0] is backend:
        return hit[1]
    cells: dict[str, np.ndarray | None] = {}
    for v, doms in backend.var_domains.items():
        var = mp.vars.get(v)
        if var is None:
            cells[v] = None
            continue
        flat = {
            idx: k
            for k, idx in enumerate(itertools.product(*(state.sets[d] for d in doms)))
        }
        ids = index.period_ids("var", v, period, var)
        k = np.array([flat.get(var.keys[i][:-1], -1) for i in ids.tolist()], np.int64)
        g = np.full(len(flat), -1, dtype=np.int64)
        ok = k >= 0
        g[k[ok]] = index.var_off[v] + ids[ok]
        if (g < 0).any():
            missing = int(np.flatnonzero(g < 0)[0])
            raise KeyError(f"variable {v!r}: cell {missing} of the layout not in mp")
        cells[v] = g
    rows: dict[str, np.ndarray] = {}
    for fam, recs in backend.records.items():
        row = mp.rows.get(fam)
        if row is None or is_cross_period(fam):
            rows[fam] = np.full(len(recs), -1, dtype=np.int64)
            continue
        ids = index.period_ids("row", fam, period, row)
        where = {row.keys[i][:-1]: i for i in ids.tolist()}
        pos = np.array([where.get(tuple(idx), -1) for idx, _ in recs], np.int64)
        rows[fam] = np.where(pos >= 0, index.row_off[fam] + pos, -1)
    out = _SPTables(cells, rows)
    index.cache[key] = (backend, out)
    return out


class _CrossTables(NamedTuple):
    order: dict[str, tuple[np.ndarray, np.ndarray]]  # var -> (flat order, global ids)
    rows: dict[str, np.ndarray]


def _cross_period_tables(cross: Any, mp: MPArrays, index: CellIndex) -> _CrossTables:
    key = ("build_cross", id(cross))
    hit = index.cache.get(key)
    if hit is not None and hit[0] is cross:
        return hit[1]
    order = {}
    for v, doms in cross.var_domains.items():
        var = mp.var(v)
        flat = _flat_index(var, doms, cross.sets)
        gid = index.var_off[v] + np.arange(len(var.keys), dtype=np.int64)
        order[v] = (np.argsort(flat, kind="stable"), gid)
    rows = {}
    for fam, recs in cross.records.items():
        row = mp.rows.get(fam)
        if row is None:
            rows[fam] = np.full(len(recs), -1, dtype=np.int64)
            continue
        where = {k: i for i, k in enumerate(row.keys)}
        pos = np.array([where.get(tuple(idx), -1) for idx, _ in recs], np.int64)
        rows[fam] = np.where(pos >= 0, index.row_off[fam] + pos, -1)
    out = _CrossTables(order, rows)
    index.cache[key] = (cross, out)
    return out


# compiled systems kept per run, recording and part: the latest patterns, newest last
_KEPT_PATTERNS = 2


def _layout_pattern(layout: VarLayout) -> tuple:
    """What a compiled system depends on besides the values: the free cells (the slots)
    and which fixed cells are exactly 0 (``TermTable.groups`` folds their terms away)."""
    free = [
        (np.asarray(layout.slots[v]) < layout.n_cols).ravel()
        for v in layout.var_domains
    ]
    mask = np.concatenate(free) if free else np.zeros(0, dtype=bool)
    zero = np.asarray(layout.fixed_values, dtype=float) == 0.0
    return (
        tuple(layout.var_domains),
        layout.n_cols,
        len(mask),
        np.packbits(mask).tobytes(),
        len(zero),
        np.packbits(zero).tobytes(),
    )


def _kept_system(recording: Any, layout: VarLayout, index: CellIndex, part: tuple):
    """``recording.system(layout)``, compiled once per run (``index.cache``) and
    pattern (:func:`_layout_pattern`): a layout of a pattern compiled before gets that
    system with its own fixed values (a shallow copy; F and J read the fixed slots from
    ``fixed_values`` only), which is the system a compile of this layout gives."""
    import copy

    key = ("system", id(recording), part)
    sig = _layout_pattern(layout)
    hit = index.cache.get(key)
    kept = hit[1] if hit is not None and hit[0] is recording else []
    for s, system in kept:
        if s == sig:
            out = copy.copy(system)
            out.fixed_values = np.array(layout.fixed_values, dtype=float)
            return out
    system = recording.system(layout)
    kept = [*kept, (sig, system)][-_KEPT_PATTERNS:]
    index.cache[key] = (recording, kept)
    out = copy.copy(system)  # the kept one's fixed values stay those it was built with
    out.fixed_values = np.array(layout.fixed_values, dtype=float)
    return out


def _squared_rows(rows: dict[str, np.ndarray], records: dict, ids: Any) -> np.ndarray:
    """Squared row of every row of a system (family order, record order); -1: dropped."""
    gid = np.concatenate([rows[f] for f in records] or [np.zeros(0, dtype=np.int64)])
    return np.where(gid >= 0, ids.row_pos[np.maximum(gid, 0)], -1).astype(np.int64)


class SquaredSystem:
    """The squared system of one solve call, evaluated by the family backend.

    Rows are ``sq.row_names``, columns ``sq.col_names`` (paired order: row k's
    paired column is column k). The rows come from up to two recordings: the
    single-period families and, with ``cross``, the cross-period ones. With
    ``proxies=True`` a cross-period row nobody records is a timing placeholder
    (``proxy_rows``: ``F = x[k] - x0[k]`` and a unit J entry on the paired column); a
    squared column the backend has no cell for must then be paired with a proxy row
    (``build`` raises otherwise; ``proxy_cols_outside_backend`` counts them). Without
    ``proxies`` an active row with no family function raises.
    """

    def __init__(self, x0, parts, row_names, proxy_k, n_outside=0):
        self.x0 = x0
        self._n = len(x0)
        # (system, system row index of each part row, squared row index, layout column
        # -> squared column, system row -> squared row (-1: dropped))
        self._parts = parts
        self._proxy_k = proxy_k
        # set by ``build`` for ``fscale``: row name -> position, per-recording context
        self._row_pos_d: dict[str, int] | None = None
        self._row_names: list[str] = row_names
        self._fscale_ctx: list = []
        # set by ``build`` for ``set_fixed``: (system, layout, single-period part)
        self._layouts: list = []
        self._period: str | None = None
        self.proxy_rows = [row_names[k] for k in proxy_k]
        # Squared columns with no backend cell (each paired with a proxy row).
        self.proxy_cols_outside_backend = int(n_outside)
        self.proxy_families: dict[str, int] = {}
        for r in self.proxy_rows:
            f = r.partition("[")[0]
            self.proxy_families[f] = self.proxy_families.get(f, 0) + 1

    @staticmethod
    def _single_period_part(backend, state, mp, period, ids):
        """Layout of ``period``, the backend system on it, and where its columns and
        rows go in the squared system (integer maps, no name strings)."""
        tab = _single_period_tables(backend, state, mp, period, ids.index)
        col_pos = ids.col_pos
        vals_g, _ = ids.values()
        values, free, zcols = {}, {}, []
        for v, doms in backend.var_domains.items():
            shape = tuple(len(state.sets[d]) for d in doms)
            g = tab.cells[v]
            if g is not None:
                vals = vals_g[g].reshape(shape)
                cp = col_pos[g]
                mask = (cp >= 0).reshape(shape)
                zcols.append(cp[cp >= 0])
            elif v in SP_ONLY_VARS:  # a fixed slot at the state's level
                vals = np.array(state.values[v], dtype=float).reshape(shape)
                mask = np.zeros(shape, dtype=bool)
            else:
                raise KeyError(f"variable component {v!r} missing from mp")
            values[v], free[v] = vals, mask
        for p in backend.param_domains:
            values[p] = state.find_component(p).data
            free[p] = np.zeros(values[p].shape, dtype=bool)
        layout = VarLayout.from_state(
            state.sets, {**backend.var_domains, **backend.param_domains}, values, free
        )
        zperm = np.concatenate(zcols) if zcols else np.zeros(0, dtype=np.int64)
        k = _squared_rows(tab.rows, backend.records, ids)
        ctx = (backend.records, values, free, k)
        system = _kept_system(backend, layout, ids.index, ("sp", period))
        return system, zperm, k, ctx, layout

    @staticmethod
    def _cross_period_part(cross, mp, ids):
        """Layout over every period: levels and fixed flags of ``mp`` (``square``'s
        returned arrays, which carry the closure's fixings of every period); the free
        cells are the squared columns."""
        sets = cross.sets
        col_pos = ids.col_pos
        tab = _cross_period_tables(cross, mp, ids.index)
        values, free, zcols = {}, {}, []
        for v, doms in cross.var_domains.items():
            shape = tuple(len(sets[d]) for d in doms)
            vals = np.ones(shape)  # the cells no row reads (mq of other slots)
            mask = np.zeros(shape, dtype=bool)
            var = mp.var(v)
            flat = _flat_index(var, doms, sets)
            vals.flat[flat] = var.value
            mask.flat[flat] = ~var.fixed
            values[v], free[v] = vals, mask
            # layout columns of v: its free cells in flat order
            order, gid = tab.order[v]
            cp = col_pos[gid[order[~var.fixed[order]]]]
            if (cp < 0).any():
                raise KeyError(f"{v}: free cell of the layout is not a squared column")
            zcols.append(cp)
        layout = VarLayout.from_state(sets, cross.var_domains, values, free)
        zperm = np.concatenate(zcols) if zcols else np.zeros(0, dtype=np.int64)
        k = _squared_rows(tab.rows, cross.records, ids)
        ctx = (cross.records, values, free, k)
        system = _kept_system(cross, layout, ids.index, ("cross",))
        return system, zperm, k, ctx, layout

    @classmethod
    def build(
        cls,
        backend: Any,
        state: Any,
        mp: MPArrays,
        period: str,
        sq: Squared,
        cross: Any = None,
        proxies: bool = False,
    ) -> SquaredSystem:
        """Layout of ``period`` (free cells are ``sq.col_names``), then the backend
        system. Cell values are the post-squaring ones in ``sq.cells``; ``mp`` backs
        a cell ``sq`` lacks. Raises ``KeyError`` for a free column of a family
        variable that ``mp`` does not hold.

        ``cross`` is the cross-period recording (``cross_period.load_or_record``); its
        layout spans every period, so ``mp`` must be the arrays ``square`` returns
        (``return_mp=True``): other periods' levels and fixings as the closure left
        them. Rows with no family function raise unless ``proxies`` and cross-period.

        Rows, columns and values go through the integer ids ``square`` leaves on
        ``sq.ids``; a ``Squared`` without them (built from names) gets them from its
        names first."""
        n = len(sq.col_names)
        ids = sq.ids
        if (
            ids is None
            or not ids.index.fits(mp)
            or len(ids.row_gid) != len(sq.row_names)
            or len(ids.col_gid) != n
        ):
            if len(set(sq.col_names)) != n or len(sq.row_names) != n:
                raise ValueError("sq is not a square system of distinct columns")
            ids = SquaredIds.from_names(sq, mp)
        elif len(sq.row_names) != n:
            raise ValueError("sq is not a square system of distinct columns")

        raw = [cls._single_period_part(backend, state, mp, period, ids)]
        if cross is not None:
            raw.append(cls._cross_period_part(cross, mp, ids))
        x0 = np.array(ids.x0(), dtype=float)

        parts, covered, backend_cols = [], np.zeros(n, bool), np.zeros(n, bool)
        fscale_ctx = []
        layouts = []
        for i, (system, zperm, k_of_row, ctx, layout) in enumerate(raw):
            layouts.append((system, layout, i == 0))
            row_name = (
                (lambda nm, p=period: _with_period(nm, p))
                if i == 0
                else (lambda nm: nm)
            )
            fscale_ctx.append((*ctx, row_name, i == 0))
            backend_cols[zperm] = True
            fam_rows = np.flatnonzero(k_of_row >= 0).astype(np.int64)
            fam_k = k_of_row[fam_rows]
            dup = covered[fam_k]
            if dup.any():
                raise ValueError(
                    f"rows given by two recordings: {sorted(sq.row_names[k] for k in fam_k[dup])[:5]}"
                )
            covered[fam_k] = True
            row_of = np.full(system.n_rows, -1, dtype=np.int64)
            row_of[fam_rows] = fam_k
            parts.append((system, fam_rows, fam_k, zperm, row_of))
        proxy_k = np.flatnonzero(~covered)
        # A row nobody records is only legitimate as a cross-period proxy; anything
        # else active would be a fake row.
        bad = sorted(
            {
                sq.row_names[k].partition("[")[0]
                for k in proxy_k
                if not (proxies and is_cross_period(sq.row_names[k].partition("[")[0]))
            }
        )
        if bad:
            raise ValueError(f"active rows with no family: {bad}")
        # A squared column the backend has no cell for gets J entries only from a
        # proxy row's unit entry, so it must be the paired column of a proxy row.
        outside = np.flatnonzero(~backend_cols)
        unpaired = sorted(
            sq.col_names[k] for k in np.setdiff1d(outside, proxy_k, assume_unique=True)
        )
        if unpaired:
            raise ValueError(
                f"{len(unpaired)} squared columns are not backend cells and not paired "
                f"with a proxy row: {unpaired[:5]}"
            )
        ss = cls(x0, parts, sq.row_names, proxy_k, len(outside))
        ss._row_names = sq.row_names
        ss._fscale_ctx = fscale_ctx
        ss._layouts = layouts
        ss._period = period
        return ss

    def set_fixed(self, var: str, cells: Mapping[tuple, float]) -> None:
        """Write new values of fixed cells of ``var`` (``{key with period: value}``, the
        ``mp`` keys) into the fixed slots of every recording that has ``var``, in place:
        F and J then are those of a system built with these values (a parameter update
        without a rebuild, e.g. a shock sub-step). ``KeyError`` when no recording has
        ``var``; ``ValueError`` for a cell that is a column (free) or of another period
        than the system's single-period part."""
        found = False
        for system, layout, single in self._layouts:
            if var not in layout.slots:
                continue
            doms = layout.var_domains[var]
            slots = layout.slots[var]
            for key, value in cells.items():
                if single:
                    if key[-1] != self._period:
                        raise ValueError(f"{var}{key}: not of period {self._period!r}")
                    key = key[:-1]
                idx = tuple(
                    layout.sets[d].index(e) for d, e in zip(doms, key, strict=True)
                )
                slot = int(slots[idx])
                if slot < layout.n_cols:
                    raise ValueError(f"{var}{key}: a free column, not a fixed cell")
                system.fixed_values[slot - layout.n_cols] = float(value)
            found = True
        if not found:
            raise KeyError(f"{var}: no recording of the system has it")

    @property
    def _row_pos(self) -> dict[str, int]:
        """Row name -> position, built on first use (only ``fscale`` reads it)."""
        if self._row_pos_d is None:
            self._row_pos_d = {r: k for k, r in enumerate(self._row_names)}
        return self._row_pos_d

    @_row_pos.setter
    def _row_pos(self, value: dict[str, int]) -> None:
        self._row_pos_d = value

    def fscale(self) -> np.ndarray:
        """Per-row scale of the relative-residual test (``EQUILIBRIA_GTAP_TR_RELTOL``),
        aligned with ``row_names``: ``max(1, |x_j|)`` over the free cells among the
        row's recorded leaves, before zero folding (see ``family.fscale``). Proxy rows
        get 1."""
        from equilibria.family.fscale import leaf_table

        out = np.ones(self._n)
        for records, values, free, k_of_row, _row_name, _single in self._fscale_ctx:
            mag = leaf_table(records, values).fscale(values, free)
            ok = k_of_row >= 0
            out[k_of_row[ok]] = mag[ok]
        return out

    def F(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(self._n)
        for system, rows, k, zperm, _ in self._parts:
            out[k] = system.F(x[zperm])[rows]
        out[self._proxy_k] = x[self._proxy_k] - self.x0[self._proxy_k]
        return out

    def J_entries(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The entries of :meth:`J` as ``(rows, cols, data)`` (COO, duplicates not
        summed), for a caller that borders the matrix before assembling it."""
        rows, cols, data = [], [], []
        for system, _, _, zperm, row_of in self._parts:
            Jf = system.jac(x[zperm])
            r = row_of[Jf.row]
            keep = r >= 0
            rows.append(r[keep])
            cols.append(zperm[Jf.col[keep]])
            data.append(Jf.data[keep])
        rows.append(self._proxy_k)
        cols.append(self._proxy_k)
        data.append(np.ones(len(self._proxy_k)))
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)

    def J(self, x: np.ndarray) -> sp.csr_matrix:
        rows, cols, data = self.J_entries(x)
        return sp.csr_matrix((data, (rows, cols)), shape=(self._n, self._n))


__all__ = [
    "CROSS_PERIOD_FAMILIES",
    "SquaredSystem",
    "is_cross_period",
    "period_incidence",
]
