"""Variable layout: dense slot arrays per variable over its set product.

Free cells get Jacobian columns ``0..n_cols-1`` (variables in declaration order, cells in
C order over the variable's domain). Fixed cells get slots ``n_cols + k`` whose values
live in ``fixed_values``; ``FamilySystem`` reads them as data and emits no column.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class VarLayout:
    sets: dict[str, list[str]]
    var_domains: dict[str, tuple[str, ...]]
    slots: dict[str, np.ndarray]
    n_cols: int
    free_values: np.ndarray
    fixed_values: np.ndarray

    @classmethod
    def from_state(
        cls,
        sets: dict[str, list[str]],
        var_domains: dict[str, tuple[str, ...]],
        values: dict[str, np.ndarray],
        free: dict[str, np.ndarray],
    ) -> VarLayout:
        n_free = sum(int(np.count_nonzero(free[v])) for v in var_domains)
        slots: dict[str, np.ndarray] = {}
        free_vals, fixed_vals = [], []
        next_free, next_fixed = 0, n_free
        for v in var_domains:
            mask = np.asarray(free[v], bool)
            s = np.empty(mask.shape, dtype=np.int64)
            nf = int(mask.sum())
            s[mask] = np.arange(next_free, next_free + nf)
            s[~mask] = np.arange(next_fixed, next_fixed + mask.size - nf)
            next_free += nf
            next_fixed += mask.size - nf
            slots[v] = s
            vals = np.asarray(values[v], float)
            free_vals.append(vals[mask])
            fixed_vals.append(vals[~mask])
        return cls(
            sets=sets,
            var_domains=dict(var_domains),
            slots=slots,
            n_cols=n_free,
            free_values=np.concatenate(free_vals) if free_vals else np.zeros(0),
            fixed_values=np.concatenate(fixed_vals) if fixed_vals else np.zeros(0),
        )

    def col(self, var: str) -> np.ndarray:
        """Slot of every cell of ``var`` (>= n_cols where the cell is fixed)."""
        return self.slots[var]

    @property
    def col_names(self) -> list[str]:
        """Pyomo-style names of the free columns, in column order (diagnostics/tests)."""
        names = [""] * self.n_cols
        for v, doms in self.var_domains.items():
            s = self.slots[v]
            for idx, slot in zip(
                itertools.product(*(self.sets[d] for d in doms)), s.ravel(), strict=True
            ):
                if slot < self.n_cols:
                    names[slot] = f"{v}[{','.join(idx)}]"
        return names
