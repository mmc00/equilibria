"""Equation families evaluated as array templates (F and sparse J).

A family is one equation written once over its index sets. Its residual splits in two:

    residual[row] = row_fn(x[row_cols[row]], **row_args[row])
                    - sum_{t : term_rows[t] == row} term_fn(x[term_cols[t]], **term_args[t])

Both templates are scalar functions of a short vector of variable values; they are
``vmap``-ed over index arrays, and the Jacobian is ``vmap(jacrev(template))`` over the
same arrays. The sparsity pattern is therefore the concatenation of ``(row, col)``
index arrays, fixed at construction and independent of the evaluation point.

Columns are *slots* into ``x_all = concat(z, fixed_values)``: a slot below ``n_cols`` is
a free variable (a Jacobian column); a slot at or above ``n_cols`` is a fixed cell whose
value is data and which emits no Jacobian entry.

Index and data arrays are passed to the jitted kernels as arguments, never captured
as constants, so XLA does not fold them and one compilation serves any data of the
same shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402


def _eval_part_impl(fn, x_all, cols, args):
    return jax.vmap(lambda c, a: fn(x_all[c], **a))(cols, args)


def _jac_part_impl(fn, x_all, cols, args):
    return jax.vmap(lambda c, a: jax.jacrev(lambda xs: fn(xs, **a))(x_all[c]))(
        cols, args
    )


def _scatter_impl(values, rows, num_rows):
    return jax.ops.segment_sum(values, rows, num_segments=num_rows)


_eval_part: Any = jax.jit(_eval_part_impl, static_argnames=("fn",))
_jac_part: Any = jax.jit(_jac_part_impl, static_argnames=("fn",))
_scatter: Any = jax.jit(_scatter_impl, static_argnames=("num_rows",))


@dataclass
class Family:
    name: str
    row_fn: Callable
    row_cols: np.ndarray
    row_args: dict[str, np.ndarray]
    term_fn: Callable | None = None
    term_rows: np.ndarray | None = None
    term_cols: np.ndarray | None = None
    term_args: dict[str, np.ndarray] = field(default_factory=dict)
    # Index labels of the rows ("USA,Food"), for names only; never read by F/J.
    row_labels: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.row_cols.shape[0])

    @property
    def has_terms(self) -> bool:
        return (
            self.term_fn is not None
            and self.term_rows is not None
            and len(self.term_rows) > 0
        )


class FamilySystem:
    """Stack of families: F(z) in family order and J(z) with a fixed COO pattern."""

    def __init__(
        self,
        families: list[Family],
        n_cols: int,
        fixed_values: np.ndarray | None = None,
    ):
        self.families = families
        self.n_cols = int(n_cols)
        self.fixed_values = np.asarray(
            [] if fixed_values is None else fixed_values, dtype=float
        )
        self.rows: list[tuple[str, int]] = [
            (f.name, k) for f in families for k in range(f.n_rows)
        ]
        self.n_rows = len(self.rows)

        # Device copies of the index/data arrays, built once.
        self._dev = []
        offset = 0
        raw_rows, raw_cols = [], []
        for f in families:
            d = {
                "row_cols": jnp.asarray(f.row_cols, dtype=jnp.int64),
                "row_args": {
                    k: jnp.asarray(v, dtype=float) for k, v in f.row_args.items()
                },
            }
            k_row = f.row_cols.shape[1]
            raw_rows.append(np.repeat(np.arange(f.n_rows), k_row) + offset)
            raw_cols.append(f.row_cols.ravel())
            if f.has_terms:
                assert f.term_rows is not None and f.term_cols is not None
                d["term_rows"] = jnp.asarray(f.term_rows, dtype=jnp.int64)
                d["term_cols"] = jnp.asarray(f.term_cols, dtype=jnp.int64)
                d["term_args"] = {
                    k: jnp.asarray(v, dtype=float) for k, v in f.term_args.items()
                }
                k_term = f.term_cols.shape[1]
                raw_rows.append(np.repeat(np.asarray(f.term_rows), k_term) + offset)
                raw_cols.append(np.asarray(f.term_cols).ravel())
            self._dev.append(d)
            offset += f.n_rows

        # Fixed pattern: keep free slots, coalesce duplicate (row, col) entries.
        rr = np.concatenate(raw_rows) if raw_rows else np.zeros(0, np.int64)
        cc = np.concatenate(raw_cols) if raw_cols else np.zeros(0, np.int64)
        self._keep = cc < self.n_cols
        keys = rr[self._keep] * self.n_cols + cc[self._keep]
        uniq, self._inverse = np.unique(keys, return_inverse=True)
        self._pat_rows = uniq // self.n_cols
        self._pat_cols = uniq % self.n_cols
        self.nnz = int(len(uniq))

    @property
    def row_names(self) -> list[str]:
        """Pyomo-style row names (``family[label]``), falling back to the row number."""
        return [
            f"{f.name}[{f.row_labels[k] if f.row_labels else k}]"
            for f in self.families
            for k in range(f.n_rows)
        ]

    def _x_all(self, z):
        return jnp.concatenate(
            [jnp.asarray(z, dtype=float), jnp.asarray(self.fixed_values, dtype=float)]
        )

    def F(self, z: np.ndarray) -> np.ndarray:
        x = self._x_all(z)
        out = []
        for f, d in zip(self.families, self._dev, strict=True):
            res = _eval_part(f.row_fn, x, d["row_cols"], d["row_args"])
            if f.has_terms:
                t = _eval_part(f.term_fn, x, d["term_cols"], d["term_args"])
                res = res - _scatter(t, d["term_rows"], f.n_rows)
            out.append(res)
        return np.asarray(jnp.concatenate(out)) if out else np.zeros(0)

    def jac(self, z: np.ndarray) -> sp.coo_matrix:
        x = self._x_all(z)
        vals = []
        for f, d in zip(self.families, self._dev, strict=True):
            vals.append(_jac_part(f.row_fn, x, d["row_cols"], d["row_args"]).ravel())
            if f.has_terms:
                vals.append(
                    -_jac_part(f.term_fn, x, d["term_cols"], d["term_args"]).ravel()
                )
        v = np.asarray(jnp.concatenate(vals)) if vals else np.zeros(0)
        data = np.bincount(self._inverse, weights=v[self._keep], minlength=self.nnz)
        return sp.coo_matrix(
            (data, (self._pat_rows, self._pat_cols)), shape=(self.n_rows, self.n_cols)
        )
