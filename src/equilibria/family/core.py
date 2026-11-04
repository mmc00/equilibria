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
class TermGroup:
    """Terms of one shape: ``residual[rows[t]] += sign * fn(x[cols[t]], **args[t])``."""

    rows: np.ndarray
    fn: Callable
    cols: np.ndarray
    args: dict[str, np.ndarray]
    sign: float = 1.0


@dataclass
class CompiledFamily:
    """A family as rows plus term groups: what FamilySystem evaluates."""

    name: str
    row_labels: list[str]
    groups: list[TermGroup]

    @property
    def n_rows(self) -> int:
        return len(self.row_labels)


@dataclass
class Family:
    """Hand-written family: one row template plus one term template (Phase 0)."""

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

    def compiled(self) -> CompiledFamily:
        groups = [
            TermGroup(np.arange(self.n_rows), self.row_fn, self.row_cols, self.row_args)
        ]
        if self.has_terms:
            assert self.term_fn is not None and self.term_rows is not None
            assert self.term_cols is not None
            groups.append(
                TermGroup(
                    self.term_rows, self.term_fn, self.term_cols, self.term_args, -1.0
                )
            )
        labels = self.row_labels or [str(k) for k in range(self.n_rows)]
        return CompiledFamily(self.name, list(labels), groups)


class FamilySystem:
    """Stack of families: F(z) in family order and J(z) with a fixed COO pattern."""

    def __init__(
        self,
        families: list[Family] | list[CompiledFamily] | list[Family | CompiledFamily],
        n_cols: int,
        fixed_values: np.ndarray | None = None,
    ):
        self.families = families
        self._compiled = [
            f.compiled() if isinstance(f, Family) else f for f in families
        ]
        self.n_cols = int(n_cols)
        self.fixed_values = np.asarray(
            [] if fixed_values is None else fixed_values, dtype=float
        )
        self.rows: list[tuple[str, int]] = [
            (f.name, k) for f in self._compiled for k in range(f.n_rows)
        ]
        self.n_rows = len(self.rows)

        # Device copies of the index/data arrays, built once.
        self._dev: list[list[dict]] = []
        offset = 0
        raw_rows, raw_cols = [], []
        for f in self._compiled:
            dev = []
            for g in f.groups:
                cols = np.asarray(g.cols, dtype=np.int64).reshape(len(g.rows), -1)
                dev.append(
                    {
                        "rows": jnp.asarray(g.rows, dtype=jnp.int64),
                        "cols": jnp.asarray(cols, dtype=jnp.int64),
                        "args": {
                            k: jnp.asarray(v, dtype=float) for k, v in g.args.items()
                        },
                    }
                )
                raw_rows.append(np.repeat(np.asarray(g.rows), cols.shape[1]) + offset)
                raw_cols.append(cols.ravel())
            self._dev.append(dev)
            offset += f.n_rows

        # Fixed pattern: keep free slots, coalesce duplicate (row, col) entries.
        rr = np.concatenate(raw_rows) if raw_rows else np.zeros(0, np.int64)
        cc = np.concatenate(raw_cols) if raw_cols else np.zeros(0, np.int64)
        self._keep = cc < self.n_cols
        keys = rr[self._keep] * self.n_cols + cc[self._keep]
        uniq, self._inverse = np.unique(keys, return_inverse=True)
        self._pat_rows = uniq // max(self.n_cols, 1)
        self._pat_cols = uniq % max(self.n_cols, 1)
        self.nnz = int(len(uniq))

    @classmethod
    def from_families(cls, families, layout) -> FamilySystem:
        return cls(families, n_cols=layout.n_cols, fixed_values=layout.fixed_values)

    @property
    def row_names(self) -> list[str]:
        """Pyomo-style row names (``family[label]``)."""
        return [
            f"{f.name}[{lab}]"
            for f in self.families
            for lab in (f.row_labels or [str(k) for k in range(f.n_rows)])
        ]

    def _x_all(self, z):
        return jnp.concatenate(
            [jnp.asarray(z, dtype=float), jnp.asarray(self.fixed_values, dtype=float)]
        )

    def F(self, z: np.ndarray) -> np.ndarray:
        x = self._x_all(z)
        out = []
        for f, dev in zip(self._compiled, self._dev, strict=True):
            res = jnp.zeros(f.n_rows)
            for g, d in zip(f.groups, dev, strict=True):
                t = _eval_part(g.fn, x, d["cols"], d["args"])
                res = res + g.sign * _scatter(t, d["rows"], f.n_rows)
            out.append(res)
        return np.asarray(jnp.concatenate(out)) if out else np.zeros(0)

    def jac(self, z: np.ndarray) -> sp.coo_matrix:
        x = self._x_all(z)
        vals = []
        for f, dev in zip(self._compiled, self._dev, strict=True):
            for g, d in zip(f.groups, dev, strict=True):
                vals.append(g.sign * _jac_part(g.fn, x, d["cols"], d["args"]).ravel())
        v = np.asarray(jnp.concatenate(vals)) if vals else np.zeros(0)
        data = np.bincount(self._inverse, weights=v[self._keep], minlength=self.nnz)
        return sp.coo_matrix(
            (data, (self._pat_rows, self._pat_cols)), shape=(self.n_rows, self.n_cols)
        )
