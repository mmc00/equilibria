"""JaxGTAPModel — a PEDRO MCPModel backed by JAX evaluation + sparsejac Jacobian.

Wraps `build_F` (the vectorized JAX residual) and `sparsejac.jacrev(F, sparsity=pattern)` (the
sparse autodiff Jacobian) so PEDRO's `solve_mcp` runs the Newton loop with GPU-fast evaluation
and (via the cuDSS backend) GPU factorization. The GTAP squared system has all free variables
as equality rows, so lb=-inf, ub=+inf (a pure F(z)=0 system, the MCP special case).

The Jacobian is returned as scipy.sparse.csr_matrix in the (row=constraint, col=free-var)
order build_F establishes — the same order PyomoNLP uses — so it drops into the existing
colperm/cuDSS path unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sparsejac

from pedro.models.base import MCPModel
from equilibria.templates.gtap_jax.system_builder import build_F
from equilibria.templates.gtap_jax.sparsity import jacobian_pattern


class JaxGTAPModel(MCPModel):
    def __init__(self, m: Any, vectorize: bool = True):
        self._F, self._var_index, self._cons_order = build_F(m, vectorize=vectorize)
        self._pattern = jacobian_pattern(self._cons_order, self._var_index)
        self._jac = jax.jit(sparsejac.jacrev(self._F, sparsity=self._pattern))
        self.n = len(self._var_index)
        self._n_eq = len(self._cons_order)
        assert self.n == self._n_eq, f"not square: {self.n} vars vs {self._n_eq} eqs"
        self.lb = np.full(self.n, -np.inf)
        self.ub = np.full(self.n, np.inf)

        # seed: the Pyomo Var values in var_index order
        from pyomo.core.base.var import Var
        from pyomo.environ import value
        z0 = np.empty(self.n)
        for v in m.component_data_objects(Var, active=True):
            s = self._var_index.get(id(v))
            if s is not None:
                z0[s] = value(v) if v.value is not None else 1.0
        self._seed = z0

    def seed(self) -> np.ndarray:
        return self._seed.copy()

    def F(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(self._F(jnp.asarray(z, dtype=float)))

    def jacobian(self, z: np.ndarray):
        J = self._jac(jnp.asarray(z, dtype=float))  # BCOO
        idx = np.asarray(J.indices)
        data = np.asarray(J.data)
        return sp.csr_matrix(
            (data, (idx[:, 0], idx[:, 1])), shape=(self._n_eq, self.n)
        )
