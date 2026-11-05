"""Incidence of one period read from the family backend's Jacobian pattern.

``prep_square.square`` pairs rows with columns from the variables each row mentions.
Pyomo supplies that incidence as strings; here it comes from the J pattern of the
backend's recorded rows, so the period prep needs no Pyomo model.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import scipy.sparse as sp

from equilibria.family.layout import VarLayout
from equilibria.family.mp_arrays import MPArrays, Squared

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


def period_incidence(backend: Any, state: Any, period: str) -> dict[str, list[str]]:
    """For every recorded row ``eq[idx]``, the cells ``v[idx',period]`` in its J pattern.

    Keys and columns carry the period suffix (``eq_xp[USA,agr,check]``), the format
    ``square(incidence=...)`` expects. All variable cells are columns, so structural
    zeros stay in the pattern. Rows the backend does not record are absent.
    """
    layout = _all_free_layout(backend, state)
    system = backend.system(layout)
    J = system.jac(layout.free_values).tocsr()
    # Column names carry the period; layout.col_names is in column order.
    cols = [_with_period(n, period) for n in layout.col_names]
    rows = [_with_period(n, period) for n in system.row_names]
    out: dict[str, list[str]] = {}
    for i, name in enumerate(rows):
        if is_cross_period(name.partition("[")[0]):
            continue
        lo, hi = J.indptr[i], J.indptr[i + 1]
        out[name] = [cols[j] for j in sorted(J.indices[lo:hi])]
    return out


# Variables of the single-period model that the multiperiod model drops together with
# their rows (the intra-period Fisher aggregates, see hybrid._pyomo_layout): a fixed
# slot at the state's level. Any other family variable missing from ``mp`` raises.
SP_ONLY_VARS = frozenset({"mfr_bs", "mfr_sb", "mfr_ss", "mfw_bs", "mfw_sb", "mfw_ss"})


def _cell_name(var: str, idx: tuple, period: str) -> str:
    return f"{var}[{','.join((*idx, period))}]"


class SquaredSystem:
    """The squared system of one solve call, evaluated by the family backend.

    Rows are ``sq.row_names``, columns ``sq.col_names`` (paired order: row k's
    paired column is column k). Rows with no family function (``proxy_rows``: the
    cross-period families and rows the backend does not record) are a timing
    placeholder only: ``F = x[k] - x0[k]`` and a unit J entry on the paired column.
    A squared column the backend has no cell for must be paired with a proxy row
    (``build`` raises otherwise); ``proxy_cols_outside_backend`` counts them.
    """

    def __init__(
        self, x0, family_rows, family_k, zperm, system, row_names, proxy_k, n_outside=0
    ):
        self.x0 = x0
        self._n = len(x0)
        self._system = system
        self._fam_rows = family_rows  # system row index of each family row
        self._fam_k = family_k  # squared row index of each family row
        self._zperm = zperm  # layout column -> squared column
        self._proxy_k = proxy_k
        self.proxy_rows = [row_names[k] for k in proxy_k]
        # Squared columns with no backend cell (each paired with a proxy row).
        self.proxy_cols_outside_backend = int(n_outside)
        self.proxy_families: dict[str, int] = {}
        for r in self.proxy_rows:
            f = r.partition("[")[0]
            self.proxy_families[f] = self.proxy_families.get(f, 0) + 1

    @classmethod
    def build(
        cls, backend: Any, state: Any, mp: MPArrays, period: str, sq: Squared
    ) -> SquaredSystem:
        """Layout of ``period`` (free cells are ``sq.col_names``), then the backend
        system. Cell values are the post-squaring ones in ``sq.cells``; ``mp`` backs
        a cell ``sq`` lacks. Raises ``KeyError`` for a free column of a family
        variable that ``mp`` does not hold."""
        col_pos = {c: k for k, c in enumerate(sq.col_names)}
        if len(col_pos) != len(sq.col_names) or len(sq.row_names) != len(sq.col_names):
            raise ValueError("sq is not a square system of distinct columns")
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
        system = backend.system(layout)

        zperm = np.array(
            [col_pos[_with_period(n, period)] for n in layout.col_names], dtype=np.int64
        )
        x0 = np.empty(len(sq.col_names))
        for name, k in col_pos.items():
            if name not in sq.cells:
                raise KeyError(f"{name}: free column missing from sq.cells")
            x0[k] = sq.cells[name][0]

        row_pos = {r: k for k, r in enumerate(sq.row_names)}
        fam_rows, fam_k = [], []
        for i, name in enumerate(system.row_names):
            full = _with_period(name, period)
            k = row_pos.get(full)
            if k is None or is_cross_period(name.partition("[")[0]):
                continue  # row inactive in this call, or not the multiperiod row
            fam_rows.append(i)
            fam_k.append(k)
        fam_rows, fam_k = np.array(fam_rows, dtype=np.int64), np.array(fam_k, np.int64)
        proxy_k = np.setdiff1d(np.arange(len(sq.row_names)), fam_k)
        # A proxy row is only legitimate for a cross-period family (the caller supplies
        # it); an unrecorded or ifSUB-skipped row that is active would be a fake row.
        bad = sorted(
            {
                sq.row_names[k].partition("[")[0]
                for k in proxy_k
                if not is_cross_period(sq.row_names[k].partition("[")[0])
            }
        )
        if bad:
            raise ValueError(f"active rows with no family and not cross-period: {bad}")
        # A squared column the backend has no cell for gets J entries only from a
        # proxy row's unit entry, so it must be the paired column of a proxy row.
        backend_cols = np.zeros(len(sq.col_names), dtype=bool)
        backend_cols[zperm] = True
        outside = np.flatnonzero(~backend_cols)
        unpaired = sorted(
            sq.col_names[k] for k in np.setdiff1d(outside, proxy_k, assume_unique=True)
        )
        if unpaired:
            raise ValueError(
                f"{len(unpaired)} squared columns are not backend cells and not paired "
                f"with a proxy row: {unpaired[:5]}"
            )
        return cls(
            x0, fam_rows, fam_k, zperm, system, sq.row_names, proxy_k, len(outside)
        )

    def F(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(self._n)
        out[self._fam_k] = self._system.F(x[self._zperm])[self._fam_rows]
        out[self._proxy_k] = x[self._proxy_k] - self.x0[self._proxy_k]
        return out

    def J(self, x: np.ndarray) -> sp.csr_matrix:
        Jf = self._system.jac(x[self._zperm])
        # system row index -> squared row index (-1: dropped)
        row_of = np.full(self._system.n_rows, -1, dtype=np.int64)
        row_of[self._fam_rows] = self._fam_k
        r = row_of[Jf.row]
        keep = r >= 0
        rows = np.concatenate([r[keep], self._proxy_k])
        cols = np.concatenate([self._zperm[Jf.col[keep]], self._proxy_k])
        data = np.concatenate([Jf.data[keep], np.ones(len(self._proxy_k))])
        return sp.csr_matrix((data, (rows, cols)), shape=(self._n, self._n))


__all__ = [
    "CROSS_PERIOD_FAMILIES",
    "SquaredSystem",
    "is_cross_period",
    "period_incidence",
]
