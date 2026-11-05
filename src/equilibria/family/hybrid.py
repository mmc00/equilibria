"""Serve family-backend rows to a driver that holds a multiperiod PyomoNLP.

The driver solves one period ``t`` of the multiperiod Pyomo model at a time: every
other period is fixed and its rows are deactivated. ``HybridNLP`` wraps that
PyomoNLP. Each single-period family row ``eq[idx]`` maps to the Pyomo row
``eq[idx,t]`` and each family cell ``v[idx]`` to ``v[idx,t]``. Cells the NLP holds as
primals are the family's free columns; every other cell is a fixed slot with its
Pyomo value. Each family is checked against Pyomo at the seed and at a perturbed
point; families that match are served (their F and J entries replace Pyomo's), the
others stay with Pyomo and are listed in ``fallback``.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from equilibria.family.layout import VarLayout


def _split_period(name: str) -> tuple[str, str]:
    """``eq[a,b,t]`` → (``eq[a,b]``, ``t``); ``eq[t]`` → (``eq``, ``t``)."""
    head, _, inner = name.rpartition("[")
    labels, _, period = inner[:-1].rpartition(",")
    return (f"{head}[{labels}]" if labels else head), period


def active_period(nlp: Any) -> str:
    """The period index every active row of ``nlp`` carries last."""
    periods = Counter(_split_period(c.name)[1] for c in nlp.get_pyomo_constraints())
    if len(periods) != 1:
        raise ValueError(f"rows span several periods: {dict(periods)}")
    return next(iter(periods))


def _pyomo_layout(backend: Any, state: Any, m: Any, period: str, col_of: dict):
    """Layout over period ``period`` of ``m``; returns it with the Pyomo primal
    position of each free column."""
    values, free, pos = {}, {}, {}
    for v, doms in backend.var_domains.items():
        comp = m.find_component(v)
        if comp is None:
            # SP-only variable (e.g. the intra-period Fisher mf*), dropped from the
            # multiperiod model with its rows: a fixed slot at the state's level.
            values[v] = np.asarray(state.values[v], dtype=float)
            free[v] = np.zeros(values[v].shape, dtype=bool)
            pos[v] = np.full(values[v].shape, -1, dtype=np.int64)
            continue
        shape = tuple(len(state.sets[d]) for d in doms)
        vals = np.empty(shape)
        mask = np.zeros(shape, dtype=bool)
        where = np.full(shape, -1, dtype=np.int64)
        for k, idx in enumerate(itertools.product(*(state.sets[d] for d in doms))):
            key = (*idx, period) if idx else period
            # Not comp[key]: on a dense IndexedVar that would create the missing cell.
            if key not in comp:
                raise KeyError(
                    f"{v}[{','.join((*idx, period))}]: missing in the Pyomo model"
                )
            cell = comp[key]
            val = cell.value
            vals.flat[k] = np.nan if val is None else float(val)
            j = col_of.get(id(cell))
            if j is not None:
                mask.flat[k] = True
                where.flat[k] = j
        values[v], free[v], pos[v] = vals, mask, where
    for p in backend.param_domains:
        values[p] = state.find_component(p).data
        free[p] = np.zeros(values[p].shape, dtype=bool)
    layout = VarLayout.from_state(
        state.sets, {**backend.var_domains, **backend.param_domains}, values, free
    )
    zmap = np.empty(layout.n_cols, dtype=np.int64)
    for v in backend.var_domains:
        s = layout.slots[v]
        sel = free[v]
        zmap[s[sel]] = pos[v][sel]
    return layout, zmap


class HybridNLP:
    """A PyomoNLP whose verified family rows are evaluated by the family backend."""

    def __init__(self, nlp: Any, backend: Any, state: Any, *, rtol: float = 1e-10):
        self._nlp = nlp
        self.period = active_period(nlp)
        m = nlp.pyomo_model()
        col_of = {id(v): j for j, v in enumerate(nlp.get_pyomo_variables())}
        row_of = {
            _split_period(c.name)[0]: i
            for i, c in enumerate(nlp.get_pyomo_constraints())
        }
        layout, self._zmap = _pyomo_layout(backend, state, m, self.period, col_of)
        self._system = backend.system(layout)

        # Family row k → Pyomo row position (-1: the row is not active in this period).
        names = self._system.row_names
        fam_pos = np.array([row_of.get(n, -1) for n in names], dtype=np.int64)
        fam_of = np.array([f for f, _ in self._system.rows])

        x0 = nlp.get_primals().copy()
        rng = np.random.default_rng(0)
        pts = [x0, x0 * (1 + 1e-3 * rng.standard_normal(x0.size))]
        err = {f.name: 0.0 for f in self._system.families}
        for x in pts:
            nlp.set_primals(x)
            F_p = nlp.evaluate_eq_constraints()
            J_p = nlp.evaluate_jacobian_eq().tocsr()
            z = x[self._zmap]
            F_f = self._system.F(z)
            J_f = self._permuted_jac(z, fam_pos).tocsr()
            for name in err:
                k = np.flatnonzero((fam_of == name) & (fam_pos >= 0))
                if k.size == 0:
                    continue
                rows = fam_pos[k]
                fe = np.abs(F_f[k] - F_p[rows]) / np.maximum(1.0, np.abs(F_p[rows]))
                dj = (J_f[rows] - J_p[rows]).tocoo()
                scale = max(1.0, float(abs(J_p[rows]).max()) if J_p[rows].nnz else 1.0)
                je = float(np.abs(dj.data).max()) / scale if dj.nnz else 0.0
                err[name] = max(err[name], float(fe.max()), je)
        nlp.set_primals(x0)

        has_rows = {name for name in err if np.any((fam_of == name) & (fam_pos >= 0))}
        self.served = sorted(n for n in has_rows if err[n] <= rtol)
        self.fallback = {n: err[n] for n in sorted(has_rows) if err[n] > rtol}
        self.unmapped = sorted(set(err) - has_rows)

        ok = np.isin(fam_of, self.served) & (fam_pos >= 0)
        self._fam_k = np.flatnonzero(ok)
        self.family_rows = fam_pos[self._fam_k]
        self._fam_pos = np.where(ok, fam_pos, -1)
        served_row = np.zeros(nlp.n_eq_constraints(), dtype=bool)
        served_row[self.family_rows] = True
        self._served_row = served_row

    def _permuted_jac(self, z: np.ndarray, fam_pos: np.ndarray) -> sp.coo_matrix:
        """Family J with rows/cols in Pyomo order; rows not mapped are dropped."""
        J = self._system.jac(z)
        keep = fam_pos[J.row] >= 0
        return sp.coo_matrix(
            (J.data[keep], (fam_pos[J.row[keep]], self._zmap[J.col[keep]])),
            shape=(self._nlp.n_eq_constraints(), self._nlp.n_primals()),
        )

    def evaluate_eq_constraints(self, *args, **kwargs) -> np.ndarray:
        F = np.array(self._nlp.evaluate_eq_constraints(*args, **kwargs), copy=True)
        z = self._nlp.get_primals()[self._zmap]
        F[self.family_rows] = self._system.F(z)[self._fam_k]
        return F

    def evaluate_jacobian_eq(self, *args, **kwargs) -> sp.coo_matrix:
        J_p = self._nlp.evaluate_jacobian_eq(*args, **kwargs).tocoo()
        keep = ~self._served_row[J_p.row]
        J_f = self._permuted_jac(self._nlp.get_primals()[self._zmap], self._fam_pos)
        return sp.coo_matrix(
            (
                np.concatenate([J_p.data[keep], J_f.data]),
                (
                    np.concatenate([J_p.row[keep], J_f.row]),
                    np.concatenate([J_p.col[keep], J_f.col]),
                ),
            ),
            shape=J_p.shape,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._nlp, name)
