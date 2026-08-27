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

from equilibria.templates.gtap_jax.system_builder import build_F
from equilibria.templates.gtap_jax.sparsity import jacobian_pattern


def _MCPModel():
    """Lazy base class: PEDRO's MCPModel if available, else a plain object. The driver's
    JAX-eval hook uses _NLPAlignedJaxEval (no PEDRO dependency); only JaxGTAPModel — used
    when solving THROUGH PEDRO's solve_mcp — needs the real base. This keeps the eval path
    usable in environments without PEDRO installed (e.g. the Kaggle gate clones only
    equilibria; PEDRO is a private repo)."""
    try:
        from pedro.models.base import MCPModel
        return MCPModel
    except Exception:
        return object


class JaxGTAPModel(_MCPModel()):
    def __init__(self, m: Any, vectorize: bool = True):
        from equilibria.templates.gtap_jax.sparsity import build_jacobian_fn
        self._F, self._var_index, self._cons_order = build_F(m, vectorize=vectorize)
        self._jac = build_jacobian_fn(self._F, self._cons_order, self._var_index)
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
        return self._jac(z)  # already a scipy.sparse.csr_matrix (family-by-family)


class _NLPAlignedJaxEval:
    """JAX F/Jacobian aligned to a PyomoNLP's EXACT variable & constraint ordering.

    The driver's newton_tr loop indexes z by `nlp.get_primals()` order and expects the
    Jacobian rows/cols in `nlp.get_pyomo_constraints()` / `get_pyomo_variables()` order. This
    builds the JAX residual + sparsejac Jacobian over those exact orderings so it is a drop-in
    replacement for `nlp.evaluate_eq_constraints()` / `evaluate_jacobian_eq()`.
    """

    def __init__(self, nlp):
        from equilibria.templates.gtap_jax.system_builder import build_F
        from equilibria.templates.gtap_jax.sparsity import build_jacobian_fn

        vars_ordered = list(nlp.get_pyomo_variables())
        cons_ordered = list(nlp.get_pyomo_constraints())
        self._var_index = {id(v): i for i, v in enumerate(vars_ordered)}
        self._cons_order = cons_ordered
        self.n = len(vars_ordered)
        self._n_eq = len(cons_ordered)

        # per-family vmap (fast + memory-bounded at scale) aligned to the NLP's exact order
        self._F, _, _ = build_F(
            None, vectorize=True, var_index=self._var_index, cons_order=cons_ordered
        )
        # family-by-family Jacobian (small graphs) — NOT one 395k-output sparsejac.jacrev
        self._jac = build_jacobian_fn(self._F, cons_ordered, self._var_index)
        # HYBRID: a few macro-closure rows have huge trees that dominate the JAX compile — they
        # are evaluated with Pyomo instead. build_F reports which global rows to Pyomo-eval.
        self._nlp = nlp
        self._pyomo_rows = np.asarray(getattr(self._F, "pyomo_rows", []), dtype=np.int64)

    def F(self, z):
        f = np.asarray(self._F(jnp.asarray(z, dtype=float)))
        if len(self._pyomo_rows):
            self._nlp.set_primals(np.asarray(z, dtype=float))
            fp = self._nlp.evaluate_eq_constraints()
            f[self._pyomo_rows] = fp[self._pyomo_rows]  # overwrite the JAX-skipped rows
        return f

    def jacobian(self, z):
        import scipy.sparse as sp
        J = self._jac(z).tocsr()  # JAX rows (Pyomo rows are empty in it)
        if len(self._pyomo_rows):
            self._nlp.set_primals(np.asarray(z, dtype=float))
            Jp = self._nlp.evaluate_jacobian_eq().tocsr()
            # replace the Pyomo rows: zero them in J, add Pyomo's rows
            mask = np.ones(J.shape[0], dtype=bool); mask[self._pyomo_rows] = False
            Jkeep = J.multiply(mask[:, None]).tocsr()
            Prows = sp.csr_matrix((J.shape), dtype=float)
            sel = sp.csr_matrix((np.ones(len(self._pyomo_rows)),
                                 (self._pyomo_rows, self._pyomo_rows)),
                                shape=(J.shape[0], J.shape[0]))
            Prows = sel @ Jp
            J = (Jkeep + Prows).tocsr()
        return J


def build_jax_eval_from_nlp(nlp):
    """Factory: a JAX F/Jacobian evaluator aligned to `nlp`'s var/constraint order."""
    return _NLPAlignedJaxEval(nlp)
