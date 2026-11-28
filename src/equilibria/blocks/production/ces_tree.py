"""A nested CES whose tree is read from a share table (ADR-0013).

One block writes every level of the tree. Which child sits under which parent
is the share table ``alpha(child, parent, owner)``: a non-zero share is an edge.
Each parent has its elasticity ``sigma(parent, owner)``, and two flags,
``additive(parent)`` (the volume-preserving CES) and ``top(parent)`` (the top
nest, whose own shifter also scales its children). The rows and their instances
are MANAGE-WB's (``data/manage/gams/inc/model.inc``):

- ``e_<prefix><quantity>(child, owner)`` (``e_xfac``, :1253), for every child
  with a share: the sum, over its parents, of
  ``alpha·q(parent)·(λ(child)·λ(top parent))^e·(P(parent)/p(child))^σ``, with
  ``e = σ - 1``, or ``-σ`` under an additive parent, and the price factor 1 when
  ``σ = 0`` (Leontief). ``P(parent)`` is the parent's price, or its exhaustion
  price under an additive parent.
- ``e_<prefix><price>(parent, owner)`` (``e_pxfac``, :1307), for every parent
  with a child: the dual price,
  ``P(parent)·λ(top) = (Σ alpha·λ(child)^e·p(child)^(1-σ))^(1/(1-σ))`` (with
  ``-σ`` for ``1-σ`` under an additive parent), or ``Σ alpha·p(child)/λ(child)``
  when ``σ = 0``.
- ``e_<prefix><index_price>(parent, owner)`` (``e_pxfacInd``, :1361), for every
  additive parent: ``p(parent) = Σ alpha·p(child)^(1-σ)·λ(child)^(-σ) ·
  P(parent)^σ``.

The variable names are fields, so one class serves several trees in the same
model. Each row is written as GAMS writes it (``lhs = rhs``), so its residual is
GAMS's. GAMS's vintage index, the twist shifter (``fpParTc`` empty) and the
non-competitive mark-up are not written: a dataset that needs them needs its own
reference (the MANAGE calibration checks they are absent).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from equilibria.blocks._levels import LEVEL_FLOOR, LevelBlock, eq
from equilibria.core.symbolic_equations import SymbolicEquation


class CESTree(LevelBlock):
    """A CES tree over ``node_set`` x ``owner_set`` read from ``alpha``."""

    name: str = Field(default="CESTree")
    # names of the variables
    quantity: str
    price: str
    index_price: str | None = None
    shifter: str | None = None
    # sets: every node (children and parents), the owners (activities); the
    # parents, a subset of the nodes (default: the nodes themselves)
    node_set: str
    owner_set: str
    parent_set: str | None = None
    # names of the parameters
    alpha: str
    sigma: str
    additive: str
    top: str
    # leaves priced by another variable family (the demand tree; plan, Task 5)
    leaf_quantity: str | None = None
    leaf_price: str | None = None
    leaf_coef: str | None = None
    prefix: str = ""
    # lower bound of every variable it declares (a model whose columns are
    # free, as MANAGE's, passes -inf)
    lower: float = LEVEL_FLOOR

    def model_post_init(self, __context: Any) -> None:
        leaves = {
            k: v
            for k, v in (
                ("leaf_quantity", self.leaf_quantity),
                ("leaf_price", self.leaf_price),
                ("leaf_coef", self.leaf_coef),
            )
            if v is not None
        }
        if leaves:
            raise NotImplementedError(
                f"CESTree {self.name}: {', '.join(leaves)} (leaves of another "
                "variable family) are written with the demand tree (plan, Task 5)"
            )
        self.required_sets = list(dict.fromkeys(self._sets()))

    def _sets(self) -> tuple[str, str, str]:
        return (self.node_set, self.parent_set or self.node_set, self.owner_set)

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        node_set, parent_set, owner_set = self._sets()
        nodes = set_manager.get(node_set).elements
        parents = set_manager.get(parent_set).elements
        owners = set_manager.get(owner_set).elements
        stray = sorted(set(parents) - set(nodes))
        if stray:
            raise ValueError(f"CESTree {self.name}: parents {stray} are not nodes")
        nn, npar, no = len(nodes), len(parents), len(owners)

        alpha = self.value(self.alpha)
        sigma = self.value(self.sigma)
        additive = self.value(self.additive) != 0
        top = self.value(self.top) != 0
        if alpha.shape != (nn, npar, no) or sigma.shape != (npar, no):
            raise ValueError(
                f"CESTree {self.name}: {self.alpha} is {alpha.shape} and "
                f"{self.sigma} {sigma.shape}, the sets give {(nn, npar, no)}"
            )
        if additive.any() and self.index_price is None:
            raise ValueError(
                f"CESTree {self.name}: additive parents need an index_price"
            )

        # the tree: a non-zero share is an edge (model.inc:1256, 1320)
        children: dict[tuple[str, str], list[str]] = {}
        parents_of: dict[tuple[str, str], list[str]] = {}
        for ic, ip, io in zip(*np.nonzero(alpha), strict=True):
            c, p, o = nodes[ic], parents[ip], owners[io]
            children.setdefault((p, o), []).append(c)
            parents_of.setdefault((c, o), []).append(p)
        sig = {(p, o): float(sigma[ip, io]) for ip, p in enumerate(parents)
               for io, o in enumerate(owners)}  # fmt: skip
        is_add = {p: bool(additive[i]) for i, p in enumerate(parents)}
        is_top = {p: bool(top[i]) for i, p in enumerate(parents)}

        self.param(parameters, self.alpha, (node_set, parent_set, owner_set))
        self.param(parameters, self.sigma, (parent_set, owner_set))
        cells = (node_set, owner_set), (nn, no)
        self.var(variables, self.quantity, *cells, lower=self.lower)
        self.var(variables, self.price, *cells, lower=self.lower)
        if self.index_price is not None:
            self.var(
                variables, self.index_price, (parent_set, owner_set), (npar, no),
                lower=self.lower,
            )  # fmt: skip
        if self.shifter is not None:
            self.var(variables, self.shifter, *cells, lower=self.lower)

        def lam(m: Any, n: str, o: str) -> Any:
            return 1.0 if self.shifter is None else getattr(m, self.shifter)[n, o]

        def parent_price(m: Any, p: str, o: str) -> Any:
            name = self.index_price if is_add[p] else self.price
            assert name is not None
            return getattr(m, name)[p, o]

        def demand(m: Any, idx: tuple[str, ...]) -> Any:
            c, o = idx
            if (c, o) not in parents_of:
                return None
            q, pr = getattr(m, self.quantity), getattr(m, self.price)
            a, s = getattr(m, self.alpha), getattr(m, self.sigma)
            terms = []
            for p in parents_of[(c, o)]:
                term = a[c, p, o] * q[p, o]
                if self.shifter is not None:
                    shift = lam(m, c, o) * (lam(m, p, o) if is_top[p] else 1.0)
                    term = term * shift ** (-s[p, o] if is_add[p] else s[p, o] - 1)
                if sig[(p, o)] != 0:  # [1 + {(P/p)**σ - 1} $ σ]
                    term = term * (parent_price(m, p, o) / pr[c, o]) ** s[p, o]
                terms.append(term)
            return q[c, o] == sum(terms)

        def dual_price(m: Any, idx: tuple[str, ...]) -> Any:
            p, o = idx
            if (p, o) not in children:
                return None
            pr = getattr(m, self.price)
            a, s = getattr(m, self.alpha), getattr(m, self.sigma)
            lhs = parent_price(m, p, o) * (lam(m, p, o) if is_top[p] else 1.0)
            kids = children[(p, o)]
            if sig[(p, o)] == 0:  # Leontief, model.inc:1342-1355
                return lhs == sum(a[c, p, o] * pr[c, o] / lam(m, c, o) for c in kids)
            power = -s[p, o] if is_add[p] else 1 - s[p, o]
            lam_power = -s[p, o] if is_add[p] else s[p, o] - 1
            terms = []
            for c in kids:
                term = a[c, p, o] * pr[c, o] ** power
                if self.shifter is not None:
                    term = term * lam(m, c, o) ** lam_power
                terms.append(term)
            return lhs == sum(terms) ** (1 / power)

        def exhaustion_price(m: Any, idx: tuple[str, ...]) -> Any:
            p, o = idx
            if not is_add[p] or (p, o) not in children:
                return None
            pr = getattr(m, self.price)
            a, s = getattr(m, self.alpha), getattr(m, self.sigma)
            terms = []
            for c in children[(p, o)]:
                term = a[c, p, o] * pr[c, o] ** (1 - s[p, o])
                if self.shifter is not None:
                    term = term * lam(m, c, o) ** (-s[p, o])
                terms.append(term)
            return pr[p, o] == sum(terms) * parent_price(m, p, o) ** s[p, o]

        e = f"e_{self.prefix}"
        rows = [
            eq(f"{e}{self.quantity}", (node_set, owner_set), demand),
            eq(f"{e}{self.price}", (parent_set, owner_set), dual_price),
        ]
        if self.index_price is not None and any(is_add[p] for p, _ in children):
            rows.append(
                eq(f"{e}{self.index_price}", (parent_set, owner_set), exhaustion_price)
            )
        return rows
