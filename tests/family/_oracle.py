"""Pyomo oracle for the family backend: the SP block model seen through PyomoNLP.

Rows and columns are matched BY NAME (``eq_facty[USA]``, ``pf[USA,Land,Food]``), so the
backend is free to order them its own way. The only thing taken from the Pyomo model
for the backend is the variable state (levels and fixed flags).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from equilibria.family.gtap_phase0 import PHASE0_VARS, build_phase0_system
from equilibria.family.layout import VarLayout
from equilibria.templates.gtap.gtap_block_model import _set_elems
from tests.family._data import load_params

FAMILIES = ("eq_pndeq", "eq_pvaeq", "eq_facty")

# Fidelity tolerance (plan Global Constraints).
RTOL = 1e-12
ATOL = 1e-14


def comp_stat_closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
    )


@dataclass
class Oracle:
    dataset: str
    params: Any
    sets: dict[str, list[str]]
    residual_region: str
    pm: Any
    nlp: Any
    con_names: list[str]
    var_names: list[str]
    x0: np.ndarray | None = None  # calibrated primals, captured before any set_primals
    model: Any = None  # the composed equilibria Model (blocks), for the family backend

    def rows_of(self, families=FAMILIES) -> np.ndarray:
        return np.array(
            [k for k, n in enumerate(self.con_names) if n.split("[")[0] in families],
            dtype=np.int64,
        )

    def state(
        self, var_domains: dict[str, tuple[str, ...]] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        values, free = {}, {}
        for v, doms in (PHASE0_VARS if var_domains is None else var_domains).items():
            comp = getattr(self.pm, v)
            shape = tuple(len(self.sets[d]) for d in doms)
            vals = np.zeros(shape)
            fr = np.zeros(shape, bool)
            for pos, idx in zip(
                itertools.product(*(range(n) for n in shape)),
                itertools.product(*(self.sets[d] for d in doms)),
                strict=True,
            ):
                key = None if not idx else idx[0] if len(idx) == 1 else idx
                if key not in comp:
                    continue  # never built: acts as a fixed 0, emits no column
                cell = comp[key]
                vals[pos] = 0.0 if cell.value is None else cell.value
                fr[pos] = not cell.fixed
            values[v], free[v] = vals, fr
        return values, free

    def layout(self) -> VarLayout:
        values, free = self.state()
        return VarLayout.from_state(self.sets, PHASE0_VARS, values, free)

    def system(self, layout: VarLayout | None = None):
        layout = self.layout() if layout is None else layout
        return layout, build_phase0_system(
            self.params, self.params.sets, layout, self.residual_region
        )

    def point(self, kind: str) -> np.ndarray:
        x = np.array(self.x0, dtype=float, copy=True)
        if kind == "perturbed":
            u = np.random.default_rng(0).uniform(-1.0, 1.0, x.size)
            x = x * (1.0 + 0.05 * u)
        return x

    def F_J(self, x: np.ndarray):
        self.nlp.set_primals(x)
        return (
            self.nlp.evaluate_eq_constraints(),
            self.nlp.evaluate_jacobian_eq().tocoo(),
        )


@cache
def oracle(dataset: str) -> Oracle:
    return build_oracle(dataset)


def build_oracle(dataset: str, prepare: Callable[[Any], None] | None = None) -> Oracle:
    """Uncached oracle; ``prepare(pm)`` runs before PyomoNLP, which freezes fixed values."""
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Objective

    from equilibria.templates.gtap.gtap_block_model import (
        build_block_single_period,
        compose_block_sp,
    )

    p = load_params(dataset)
    sets = _set_elems(p.sets)
    rr = sets["r"][-1]
    pm = build_block_single_period(p, p.sets, comp_stat_closure(), rr)
    if prepare is not None:
        prepare(pm)
    pm._family_phase0_obj = Objective(expr=0.0)
    nlp = PyomoNLP(pm)
    return Oracle(
        dataset=dataset,
        params=p,
        sets=sets,
        residual_region=rr,
        pm=pm,
        nlp=nlp,
        con_names=[c.name for c in nlp.get_pyomo_equality_constraints()],
        var_names=[v.name for v in nlp.get_pyomo_variables()],
        x0=np.asarray(nlp.get_primals(), float).copy(),
        model=compose_block_sp(p, p.sets, comp_stat_closure(), rr)[0],
    )


def backend_z(o: Oracle, layout: VarLayout, x: np.ndarray) -> np.ndarray:
    """Backend free vector from the NLP primal vector, mapped by column name.

    A free cell no active constraint references (e.g. ``pfa`` where ``af == 0``) is not
    an NLP variable; the backend gives it a column with no Jacobian entries, so it keeps
    its own level.
    """
    where = {n: k for k, n in enumerate(o.var_names)}
    z = np.array(layout.free_values, dtype=float, copy=True)
    for c, n in enumerate(layout.col_names):
        k = where.get(n)
        if k is not None:
            z[c] = x[k]
    return z


def pyomo_entries(o: Oracle, x: np.ndarray, families=FAMILIES):
    """(row name -> F) and ((row name, col name) -> J) for ``families`` (None: all)."""
    F, J = o.F_J(x)
    if families is None:
        families = {n.split("[")[0] for n in o.con_names}
    rows = set(o.rows_of(families).tolist())
    f = {o.con_names[k]: float(F[k]) for k in rows}
    j = {}
    for r, c, v in zip(J.row, J.col, J.data, strict=True):
        if r in rows:
            key = (o.con_names[r], o.var_names[c])
            j[key] = j.get(key, 0.0) + float(v)
    return f, j


def backend_entries(system, layout: VarLayout, z: np.ndarray):
    F = system.F(z)
    J = system.jac(z)
    names, cols = system.row_names, layout.col_names
    f = {names[k]: float(F[k]) for k in range(len(names))}
    j = {}
    for r, c, v in zip(J.row, J.col, J.data, strict=True):
        key = (names[r], cols[c])
        j[key] = j.get(key, 0.0) + float(v)
    return f, j


def assert_same(
    system, layout: VarLayout, o: Oracle, x: np.ndarray, families=FAMILIES
) -> None:
    """Gate 1: same rows, same (row, col) pattern and nnz, same F and J values."""
    fp, jp = pyomo_entries(o, x, families)
    fb, jb = backend_entries(system, layout, backend_z(o, layout, x))
    assert set(fb) == set(fp), (
        f"rows differ: only backend {sorted(set(fb) - set(fp))[:5]}, "
        f"only pyomo {sorted(set(fp) - set(fb))[:5]}"
    )
    assert set(jb) == set(jp), (
        f"pattern differs: only backend {sorted(set(jb) - set(jp))[:5]}, "
        f"only pyomo {sorted(set(jp) - set(jb))[:5]}"
    )
    assert system.nnz == len(jp)
    # F tolerance is relative to the row's magnitude (sum_j |J_ij x_j|): a row that is
    # ~0 at the solution but sums terms of ~1e2 carries ~1e-14 of summation-order noise.
    where = {n: k for k, n in enumerate(o.var_names)}
    scale: dict[str, float] = {}
    for (r, c), v in jp.items():
        scale[r] = scale.get(r, 0.0) + abs(v * x[where[c]])
    rk = sorted(fp)
    diff = np.array([abs(fb[k] - fp[k]) for k in rk])
    tol = np.array([RTOL * max(abs(fp[k]), scale.get(k, 0.0)) + ATOL for k in rk])
    bad = [(k, fb[k], fp[k]) for k, d, t in zip(rk, diff, tol, strict=True) if d > t]
    assert not bad, f"F differs in {len(bad)} rows, e.g. {bad[:5]}"
    jk = sorted(jp)
    np.testing.assert_allclose(
        [jb[k] for k in jk], [jp[k] for k in jk], rtol=RTOL, atol=ATOL, err_msg="J"
    )
