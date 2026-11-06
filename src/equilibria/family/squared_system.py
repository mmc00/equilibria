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
from typing import Any

import numpy as np
import scipy.sparse as sp

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


def _pattern(backend: Any, layout: VarLayout, period: str | None) -> dict[str, list]:
    """Row name -> columns of its J pattern. ``period`` is appended to the single-period
    names; ``None`` keeps them (the cross-period names carry their period)."""
    system = backend.system(layout)
    J = system.jac(layout.free_values).tocsr()
    cols = list(layout.col_names)
    rows = list(system.row_names)
    if period is not None:
        cols = [_with_period(n, period) for n in cols]
        rows = [_with_period(n, period) for n in rows]
    out: dict[str, list[str]] = {}
    for i, name in enumerate(rows):
        lo, hi = J.indptr[i], J.indptr[i + 1]
        out[name] = [cols[j] for j in sorted(J.indices[lo:hi])]
    return out


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


def period_incidence(
    backend: Any, state: Any, period: str, cross: Any = None
) -> dict[str, list[str]]:
    """For every recorded row ``eq[idx]``, the cells ``v[idx',period]`` in its J pattern.

    Keys and columns carry the period suffix (``eq_xp[USA,agr,check]``), the format
    ``square(incidence=...)`` expects. All variable cells are columns, so structural
    zeros stay in the pattern. Rows the backend does not record are absent. With
    ``cross`` (the cross-period recording) the rows of ``period`` it records are added,
    their columns being the cells of every period they read."""
    out = {
        k: v
        for k, v in _pattern(backend, _all_free_layout(backend, state), period).items()
        if not is_cross_period(k.partition("[")[0])
    }
    if cross is not None:
        for name, cols in _cross_pattern(cross).items():
            if name.endswith(f",{period}]") or name.endswith(f"[{period}]"):
                out[name] = cols
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
        self._row_pos: dict[str, int] = {}
        self._fscale_ctx: list = []
        self.proxy_rows = [row_names[k] for k in proxy_k]
        # Squared columns with no backend cell (each paired with a proxy row).
        self.proxy_cols_outside_backend = int(n_outside)
        self.proxy_families: dict[str, int] = {}
        for r in self.proxy_rows:
            f = r.partition("[")[0]
            self.proxy_families[f] = self.proxy_families.get(f, 0) + 1

    @staticmethod
    def _single_period_part(backend, state, mp, period, sq, col_pos):
        values, free = {}, {}
        for v, doms in backend.var_domains.items():
            shape = tuple(len(state.sets[d]) for d in doms)
            vals = np.empty(shape)
            mask = np.zeros(shape, dtype=bool)
            var = mp.vars.get(v)
            for k, idx in enumerate(itertools.product(*(state.sets[d] for d in doms))):
                name = _cell_name(v, idx, period)
                if name in sq.cells:
                    vals.flat[k] = sq.cells[name][0]
                elif var is not None:
                    vals.flat[k] = var.value[var.pos((*idx, period))]
                elif v in SP_ONLY_VARS:  # a fixed slot at the state's level
                    vals.flat[k] = np.asarray(state.values[v], dtype=float).flat[k]
                else:
                    raise KeyError(f"variable component {v!r} missing from mp")
                mask.flat[k] = name in col_pos
            values[v], free[v] = vals, mask
        for p in backend.param_domains:
            values[p] = state.find_component(p).data
            free[p] = np.zeros(values[p].shape, dtype=bool)
        layout = VarLayout.from_state(
            state.sets, {**backend.var_domains, **backend.param_domains}, values, free
        )
        names = [_with_period(n, period) for n in layout.col_names]
        ctx = (backend.records, values, free)
        return backend.system(layout), names, lambda n: _with_period(n, period), ctx

    @staticmethod
    def _cross_period_part(cross, mp):
        """Layout over every period: levels and fixed flags of ``mp`` (``square``'s
        returned arrays, which carry the closure's fixings of every period); the free
        cells are the squared columns."""
        sets = cross.sets
        values, free = {}, {}
        for v, doms in cross.var_domains.items():
            shape = tuple(len(sets[d]) for d in doms)
            vals = np.ones(shape)  # the cells no row reads (mq of other slots)
            mask = np.zeros(shape, dtype=bool)
            var = mp.var(v)
            flat = _flat_index(var, doms, sets)
            vals.flat[flat] = var.value
            mask.flat[flat] = ~var.fixed
            values[v], free[v] = vals, mask
        layout = VarLayout.from_state(sets, cross.var_domains, values, free)
        ctx = (cross.records, values, free)
        return cross.system(layout), list(layout.col_names), lambda n: n, ctx

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
        them. Rows with no family function raise unless ``proxies`` and cross-period."""
        col_pos = {c: k for k, c in enumerate(sq.col_names)}
        if len(col_pos) != len(sq.col_names) or len(sq.row_names) != len(sq.col_names):
            raise ValueError("sq is not a square system of distinct columns")
        row_pos = {r: k for k, r in enumerate(sq.row_names)}

        raw = [cls._single_period_part(backend, state, mp, period, sq, col_pos)]
        if cross is not None:
            raw.append(cls._cross_period_part(cross, mp))
        x0 = np.empty(len(sq.col_names))
        for name, k in col_pos.items():
            if name not in sq.cells:
                raise KeyError(f"{name}: free column missing from sq.cells")
            x0[k] = sq.cells[name][0]

        parts, covered, backend_cols = [], set(), np.zeros(len(sq.col_names), bool)
        fscale_ctx = []
        for i, (system, col_names, row_name, ctx) in enumerate(raw):
            fscale_ctx.append((*ctx, row_name, i == 0))
            zperm = np.array([col_pos[n] for n in col_names], dtype=np.int64)
            backend_cols[zperm] = True
            fam_rows, fam_k = [], []
            for r, name in enumerate(system.row_names):
                full = row_name(name)
                k = row_pos.get(full)
                if k is None or (i == 0 and is_cross_period(name.partition("[")[0])):
                    continue  # row inactive in this call, or not the multiperiod row
                fam_rows.append(r)
                fam_k.append(k)
            fam_rows, fam_k = np.array(fam_rows, np.int64), np.array(fam_k, np.int64)
            dup = covered & set(fam_k.tolist())
            if dup:
                raise ValueError(
                    f"rows given by two recordings: {sorted(sq.row_names[k] for k in dup)[:5]}"
                )
            covered |= set(fam_k.tolist())
            row_of = np.full(system.n_rows, -1, dtype=np.int64)
            row_of[fam_rows] = fam_k
            parts.append((system, fam_rows, fam_k, zperm, row_of))
        proxy_k = np.setdiff1d(np.arange(len(sq.row_names)), np.array(sorted(covered)))
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
        ss._row_pos = row_pos
        ss._fscale_ctx = fscale_ctx
        return ss

    def fscale(self) -> np.ndarray:
        """Per-row scale of the relative-residual test (``EQUILIBRIA_GTAP_TR_RELTOL``),
        aligned with ``row_names``: ``max(1, |x_j|)`` over the free cells among the
        row's recorded leaves, before zero folding (see ``family.fscale``). Proxy rows
        get 1."""
        from equilibria.family.fscale import record_fscale

        out = np.ones(self._n)
        for records, values, free, row_name, single in self._fscale_ctx:
            recs = {
                f: r for f, r in records.items() if not (single and is_cross_period(f))
            }
            names = record_fscale(
                recs, values, free, keep=lambda n, rn=row_name: rn(n) in self._row_pos
            )
            for n, v in names.items():
                out[self._row_pos[row_name(n)]] = v
        return out

    def F(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(self._n)
        for system, rows, k, zperm, _ in self._parts:
            out[k] = system.F(x[zperm])[rows]
        out[self._proxy_k] = x[self._proxy_k] - self.x0[self._proxy_k]
        return out

    def J(self, x: np.ndarray) -> sp.csr_matrix:
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
        return sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self._n, self._n),
        )


__all__ = [
    "CROSS_PERIOD_FAMILIES",
    "SquaredSystem",
    "is_cross_period",
    "period_incidence",
]
