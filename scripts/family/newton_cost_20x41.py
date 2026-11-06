"""Cost of today's Newton step on the family squared system, per solve call.

    <python with pymumps> scripts/family/newton_cost_20x41.py
        [--dataset gtap7_20x41] [--capture DIR] [--cache-root DIR] [--out FILE]

For each solve call of a two-photo capture (settle check, check, shock): rebuild the
state from photo A, run the recipe and the squaring on arrays (rows squared from the
J pattern, cross-period rows from photo B), build ``SquaredSystem`` on the family
backend and time what one Newton iteration costs: one F+J, then MUMPS symbolic
(job 1), numeric (job 2) and solve (job 3) with today's ICNTL. The projection
multiplies by the iteration/factorization counts of today's run.

Rows with no family function (``proxy_rows``) are a timing placeholder (F = x - x0,
unit J entry); the count is reported per call.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Newton iterations and numeric factorizations of the three solve calls (settle
# check, check, shock) on 20x41, plans/paper/mediciones/fases_20x41_d44692d.log.
NIT = (3, 3, 22)
NFACT = (2, 2, 21)
# MUMPS settings of today's driver (path_capi.py, Newton-TR): no MUMPS permutation
# (the pairing is already on the diagonal), AMD ordering, iterative scaling, null
# pivot detection with the driver's CNTL(3); ICNTL(14) working-space increase.
ICNTL = {14: 100, 6: 0, 7: 0, 8: 8, 24: 1}
CNTL3 = -1e-8


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = ("pairing", "pairing_explicit_zeros", "matching", "mumps_auto")
# Order/pattern diagnostic (``measure(orderings=True)``): all with ICNTL(6)=0 and
# today's other ICNTL, the matrix handed over already in the order named.
ORDERINGS = (
    "pairing",
    "pyomo_order",
    "pyomo_J",
    "pyomo_order_pyomo_pattern",
)
# Opt-in extras of ``orderings`` (``measure(..., weighted=True)``, ``--weighted``): scipy's
# ``min_weight_full_bipartite_matching`` on the Jacobian values, which takes over 30 minutes
# per call on 20x41.
WEIGHTED = ("weighted", "weighted_free")


def _missing_diagonal(J):
    """Boolean mask of the rows whose paired diagonal (k, k) is not stored in J."""
    import numpy as np

    Jc = J.tocsr()
    return np.array(
        [
            k not in Jc.indices[Jc.indptr[k] : Jc.indptr[k + 1]]
            for k in range(Jc.shape[0])
        ]
    )


def _matching_perm(J):
    """Column order from a perfect bipartite matching on the true J pattern.

    Keeping every in-pattern pair and re-matching only the off-pattern rows is not
    always perfect (3x3: 22 of 58 unmatched), so this is a min-weight full matching
    (weight 1 on the paired diagonal, 2 elsewhere): it keeps as many in-pattern
    pairs as a perfect matching allows. Returns ``(perm, moved, moved_in_pattern)``:
    new column k is old column ``perm[k]``. Raises if no perfect matching exists.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    n = J.shape[0]
    C = J.tocoo()
    w = np.where(C.row == C.col, 1.0, 2.0)
    try:
        r, c = min_weight_full_bipartite_matching(
            csr_matrix((w, (C.row, C.col)), shape=J.shape)
        )
    except ValueError as exc:
        raise RuntimeError(f"matching not perfect: {exc}") from exc
    perm = np.empty(n, dtype=int)
    perm[r] = c
    diag_in = ~_missing_diagonal(J)
    moved = perm != np.arange(n)
    return perm, int(moved.sum()), int((moved & diag_in).sum())


def _weighted_perm(J, locked: dict[int, int]):
    """Column order from a maximum-product weighted matching on J's values.

    MC64 option 5 style: minimise the sum of ``c_ij = log(max_i |a_ij|) - log|a_ij|``
    (column maxima over the rows being matched), shifted by 1 so every weight is
    strictly positive (all perfect matchings have the same number of entries, so
    the optimum is unchanged). Exact zeros are dropped. ``locked`` maps row -> column
    (the hard GAMS pairs) and those rows/columns are left out of the matching.
    Returns ``(perm, report)``: new column k is old column ``perm[k]``; ``perm`` is
    None, with the reason in ``report["error"]``, if the rest has no perfect matching.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    n = J.shape[0]
    rows = np.array(sorted(set(range(n)) - set(locked)), dtype=int)
    cols = np.array(sorted(set(range(n)) - set(locked.values())), dtype=int)
    rep: dict = {"locked": len(locked), "matched": int(rows.size), "error": None}
    if rows.size != cols.size:
        rep["error"] = f"locked rows {len(locked)} != locked columns"
        return None, rep
    sub = J.tocsr()[rows][:, cols].tocoo()
    keep = sub.data != 0
    r, c, a = sub.row[keep], sub.col[keep], np.abs(sub.data[keep])
    perm = np.empty(n, dtype=int)
    for i, j in locked.items():
        perm[i] = j
    if rows.size:
        cmax = np.zeros(cols.size)
        np.maximum.at(cmax, c, a)
        w = np.log(cmax[c]) - np.log(a) + 1.0
        # A free row whose paired diagonal has no nonzero value would make the
        # nonzero part infeasible; the diagonal is offered at a prohibitive weight,
        # so as few rows as possible keep a zero pivot (counted in ``zero_pivots``).
        pos_r = {int(x): k for k, x in enumerate(rows)}
        pos_c = {int(x): k for k, x in enumerate(cols)}
        has = set(zip(r.tolist(), c.tolist(), strict=True))
        dr = [pos_r[i] for i in pos_r if i in pos_c and (pos_r[i], pos_c[i]) not in has]
        dc = [pos_c[rows[k]] for k in dr]
        big = float(w.sum()) + 1.0
        r = np.concatenate([r, np.array(dr, dtype=int)])
        c = np.concatenate([c, np.array(dc, dtype=int)])
        w = np.concatenate([w, np.full(len(dr), big)])
        try:
            mr, mc = min_weight_full_bipartite_matching(
                csr_matrix((w, (r, c)), shape=(rows.size, cols.size))
            )
        except ValueError as exc:
            rep["error"] = f"no perfect matching on the free part: {exc}"
            return None, rep
        perm[rows[mr]] = cols[mc]
        zero = set(zip(dr, dc, strict=True))
        rep["zero_pivots"] = int(
            sum((i, j) in zero for i, j in zip(mr.tolist(), mc.tolist(), strict=True))
        )
    return perm, rep


def icntl_for(variant: str) -> dict[int, int]:
    """The ICNTL handed to MUMPS for ``variant`` (``mumps_auto``: ICNTL(6)=7)."""
    if variant not in VARIANTS + ORDERINGS + WEIGHTED:
        raise ValueError(
            f"unknown colperm {variant!r}; use one of {VARIANTS + ORDERINGS + WEIGHTED}"
        )
    icntl = dict(ICNTL)
    if variant == "mumps_auto":
        icntl[6] = 7
    return icntl


def _mumps_step(J, F, variant: str = "pairing", keep_solution: bool = False):
    """Symbolic, numeric, solve of ``J dx = -F``; times, MUMPS codes and residual.

    A failing job is recorded (``mumps_ok`` False) with the codes, not worked around.
    ``variant`` picks the column order / ICNTL(6) (see ``VARIANTS``); an ``ORDERINGS``
    name factors ``J`` as given with ICNTL(6)=0. ``keep_solution`` adds the step
    (in ``J``'s column order) as ``solution``.
    """
    import mumps
    import numpy as np

    icntl = icntl_for(variant)
    rhs0 = -np.asarray(F, dtype=float)
    out: dict = {
        "variant": variant,
        "symbolic_s": None,
        "factor_s": None,
        "solve_s": None,
        "mumps_ok": False,
        "error": None,
        "infog1": None,
        "infog28": None,
        "infog9": None,
        "infog29": None,
        "infog12": None,
        "infog13": None,
        "residual": None,
        "nnz": int(J.nnz),
        "missing_diag": int(_missing_diagonal(J).sum()),
        "pairs_moved": 0,
        "in_pattern_pairs_dropped": 0,
        "icntl": {str(k): v for k, v in icntl.items()},  # as passed to MUMPS
    }
    A = J.tocoo()
    perm = None
    if variant == "pairing_explicit_zeros":
        miss = np.flatnonzero(_missing_diagonal(J))
        A = J.tocoo()
        from scipy.sparse import coo_matrix

        A = coo_matrix(
            (
                np.concatenate([A.data, np.zeros(miss.size)]),
                (np.concatenate([A.row, miss]), np.concatenate([A.col, miss])),
            ),
            shape=J.shape,
        )
    elif variant == "matching":
        perm, out["pairs_moved"], out["in_pattern_pairs_dropped"] = _matching_perm(J)
        A = J.tocsc()[:, perm].tocoo()
    ctx = mumps.DMumpsContext(sym=0, par=1)
    try:
        ctx.set_silent()
        for k, v in icntl.items():
            ctx.set_icntl(k, v)
        ctx.set_cntl(3, CNTL3)
        ctx.set_centralized_sparse(A)
        for job, key in ((1, "symbolic_s"), (2, "factor_s"), (3, "solve_s")):
            if job == 3:
                rhs = rhs0.copy()
                ctx.set_rhs(rhs)
            t = time.perf_counter()
            try:
                ctx.run(job=job)
            except Exception as exc:  # pymumps raises on INFOG(1) < 0
                out[key] = time.perf_counter() - t
                out["infog1"] = int(ctx.get_infog(1))
                out["error"] = f"job {job}: {exc!r}"
                return out
            out[key] = time.perf_counter() - t
            out["infog1"] = int(ctx.get_infog(1))
            if job == 1:
                out["infog9"] = int(ctx.get_infog(9))  # estimated factor entries
            if job == 2:
                out["infog28"] = int(ctx.get_infog(28))
                out["infog29"] = int(ctx.get_infog(29))  # effective factor entries
                out["infog12"] = int(ctx.get_infog(12))  # off-diagonal pivots
                out["infog13"] = int(ctx.get_infog(13))  # delayed pivots
        sol = rhs
        if perm is not None:  # column k of the permuted system is old column perm[k]
            x = np.empty_like(sol)
            x[perm] = sol
            sol = x
        out["residual"] = float(np.abs(J @ sol - rhs0).max())
        if keep_solution:
            out["solution"] = sol
        out["mumps_ok"] = True
    finally:
        ctx.destroy()
    return out


def build_pyomo_model(dataset: str):
    """The capture route's Pyomo multiperiod model, built as ``_solve_shock`` builds it
    (route closure, ``base_calibrated=True``, reference GDX, NLP mode env) and never
    solved: each call's state is loaded from the photos (``pyomo_nlp_at_b``)."""
    import os

    from equilibria.sources import find_reference
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _sibling("prep_cost_20x41")._params(dataset)
    ref = find_reference(
        dataset, "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    ) or find_reference(
        dataset, "gams", form="nlp", mode="gtap", ifsub=1, fbep="header"
    )
    prev = os.environ.get("EQUILIBRIA_GTAP_SOLVE_NLP")
    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"  # the capture runs solve_nlp=True
    try:
        m, _ = build_block_model(
            p,
            p.sets,
            _sibling("budget_20x41")._closure(),
            list(p.sets.r)[-1],
            base_calibrated=True,
            ref_gdx=None if ref is None else ref / "out.gdx",
        )
    finally:
        if prev is None:
            os.environ.pop("EQUILIBRIA_GTAP_SOLVE_NLP", None)
        else:
            os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = prev
    return m


def pyomo_nlp_at_b(m, capture: Path, a, b):
    """PyomoNLP of ``m`` in photo B's state: every cell at A2's level (A if no A2),
    B's active period on top; free cells exactly B's, active rows exactly B's."""
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Constraint, Objective, Var

    from equilibria.family.prep_capture import _key, load_a2

    base = load_a2(capture, a) or a
    n_free = 0
    for comp in m.component_objects(Var, descend_into=True):
        cells = {}
        for photo in (base, b):
            vc = photo.vars.get(comp.name)
            if vc is not None:
                cells.update(
                    zip(vc.keys, zip(vc.value, vc.fixed, strict=True), strict=True)
                )
        for idx in comp:
            vd = comp[idx]
            val, fixed = cells.get(_key(idx), (None, True))
            if val is not None:
                vd.set_value(None if val != val else float(val), skip_validation=True)
            free = (not fixed) and _key(idx)[-1:] == (b.period,)
            if free:
                vd.unfix()
                n_free += 1
            else:
                vd.fix()
    assert n_free == len(b.var_names), (n_free, len(b.var_names))
    active = set(b.con_names)
    for comp in m.component_objects(Constraint, descend_into=True):
        comp.activate()
        for idx in comp:
            cd = comp[idx]
            (cd.activate if cd.name in active else cd.deactivate)()
    for o in m.component_objects(Objective, descend_into=True):
        o.deactivate()
    if hasattr(m, "_family_cmp_obj"):
        m.del_component(m._family_cmp_obj)
    m._family_cmp_obj = Objective(expr=0.0)
    nlp = PyomoNLP(m)
    cons = [c.name for c in nlp.get_pyomo_equality_constraints()]
    assert sorted(cons) == sorted(b.con_names)
    return nlp


def _bijection(names: list[str], pos: dict[str, int], want: list[str], what: str):
    """Index of each of ``want`` in ``names``, or a report of why it cannot be done.

    Returns ``(index array | None, report)``; the report says how many names of
    ``want`` are missing from ``names``, how many of ``names`` are not in ``want``,
    duplicates, with examples. The mapping is a bijection iff both counts are 0.
    """
    import numpy as np

    missing = [w for w in want if w not in pos]
    extra = sorted(set(names) - set(want))
    dup = len(want) - len(set(want))
    rep = {
        "what": what,
        "n": len(want),
        "unmapped": len(missing),
        "not_in_other": len(extra),
        "duplicates": dup,
        "examples": (missing[:5], extra[:5]),
    }
    ok = not missing and not extra and not dup and len(want) == len(names)
    rep["bijection"] = ok
    return (np.array([pos[w] for w in want], dtype=np.int64) if ok else None), rep


def _hard_rows(mp, sq, params) -> dict[int, int]:
    """Row -> column (family indices) of the hard GAMS pairs present in ``sq``.

    ``mp`` is the recipe output ``square`` was given. ``sq.col_names[k]`` is the
    variable paired with row k, so a hard pair (e, v) sits at ``(i, i)`` with
    ``i`` the row of e; ``match_rows`` locks it, which is checked."""
    from equilibria.family.prep_square import gams_pairs

    r = {n: i for i, n in enumerate(sq.row_names)}
    out: dict[int, int] = {}
    for e, v, h in gams_pairs(mp, params=params):
        if h and e in r:
            assert sq.col_names[r[e]] == v, f"hard pair {e} -> {v} not kept"
            out[r[e]] = r[e]
    return out


def _same_step(sol, ref) -> dict:
    import numpy as np

    d = float(np.abs(sol - ref).max())
    return {
        "step_max_abs_diff": d,
        "step_rel_diff": d / max(1.0, float(np.abs(ref).max())),
    }


def _orderings(
    ss, sq, b, nlp, hard: dict[int, int] | None = None, weighted: bool = False
) -> dict:
    """Factor the squared system in the orders/patterns of ``ORDERINGS``.

    * ``pairing``: ``ss.J`` as is (the family's row/column pairing).
    * ``pyomo_order``: the same J with row k = family row of ``b.con_names[k]`` and
      column k = family column of ``b.var_names[k]`` (Pyomo's paired order).
    * ``pyomo_J``: PyomoNLP's own J (over-included zero entries kept), native order;
      needs ``nlp``.
    * ``pyomo_order_pyomo_pattern``: family values on the union of Pyomo's pattern
      (explicit zeros kept) and the family's, in Pyomo's native order; needs ``nlp``.

    With ``weighted`` it also factors the two ``WEIGHTED`` orders (maximum-product
    matching on the Jacobian values, hard GAMS pairs locked / free); the matching alone
    takes over 30 minutes per call on 20x41.

    Everything is evaluated at photo B's point when ``nlp`` is given (else ``ss.x0``).
    The cross-period rows are unit proxy rows in the family but real rows in Pyomo,
    so ``pyomo_J`` solves a different linear system: its step is reported, not equal.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    rpos = {r: k for k, r in enumerate(sq.row_names)}
    cpos = {c: k for k, c in enumerate(sq.col_names)}
    x = ss.x0
    if nlp is not None:
        cols = [v.name for v in nlp.get_pyomo_variables()]
        bval = dict(zip(cols, nlp.get_primals(), strict=True))
        x = np.array([bval.get(c, ss.x0[k]) for k, c in enumerate(sq.col_names)])
    F = ss.F(x)
    J = ss.J(x).tocsr()
    J.sort_indices()

    out: dict = {}
    base = _mumps_step(J, F, "pairing", keep_solution=True)
    ref = base.pop("solution", None)
    out["pairing"] = base

    rperm, rrep = _bijection(sq.row_names, rpos, list(b.con_names), "rows")
    cperm, crep = _bijection(sq.col_names, cpos, list(b.var_names), "cols")
    bij = {
        "rows": rrep,
        "cols": crep,
        "bijection": rperm is not None and cperm is not None,
    }
    out["bijection"] = bij
    # Column Pyomo's order gives each family row (None where there is no mapping).
    pyomo_col = None
    if bij["bijection"]:
        pyomo_col = np.empty(len(rperm), dtype=np.int64)
        pyomo_col[rperm] = cperm
    weighted_runs = (
        (("weighted", dict(hard or {})), ("weighted_free", {})) if weighted else ()
    )
    for name, locked in weighted_runs:
        t = time.perf_counter()
        wperm, wrep = _weighted_perm(J, locked)
        match_s = time.perf_counter() - t
        if wperm is None:
            out[name] = {"skipped": "matching failed", "match_s": match_s, **wrep}
            continue
        Jw = J[:, wperm].tocsr()
        Jw.sort_indices()
        res = _mumps_step(Jw, F, name, keep_solution=True)
        sol = res.pop("solution", None)
        if sol is not None and ref is not None:
            full = np.empty_like(sol)
            full[wperm] = sol  # column k of Jw is family column wperm[k]
            res.update(_same_step(full, ref))
        ident = np.arange(len(wperm))
        res.update(wrep)
        res["match_s"] = match_s
        res["perm"] = wperm.tolist()
        res["rows_differ_pairing"] = int((wperm != ident).sum())
        res["rows_differ_pyomo"] = (
            None if pyomo_col is None else int((wperm != pyomo_col).sum())
        )
        out[name] = res
    if bij["bijection"]:
        Jb = J[rperm][:, cperm].tocsr()
        Jb.sort_indices()
        res = _mumps_step(Jb, F[rperm], "pyomo_order", keep_solution=True)
        sol = res.pop("solution", None)
        if sol is not None and ref is not None:
            full = np.empty_like(sol)
            full[cperm] = sol  # column k of Jb is family column cperm[k]
            res.update(_same_step(full, ref))
        out["pyomo_order"] = res
        # Pairs of the family that differ from Pyomo's, as a diagnostic.
        out["pairs_not_pyomo"] = int(
            sum(sq.col_names[k] != c for k, c in zip(rperm, b.var_names, strict=True))
        )
    else:
        out["pyomo_order"] = {"skipped": "name mapping is not a bijection", **bij}

    skip = {"skipped": "no PyomoNLP (pyomo=False)"}
    if nlp is None:
        out["pyomo_J"] = dict(skip)
        out["pyomo_order_pyomo_pattern"] = dict(skip)
        return out
    prow = [c.name for c in nlp.get_pyomo_equality_constraints()]
    pcol = [v.name for v in nlp.get_pyomo_variables()]
    Jp = nlp.evaluate_jacobian_eq().tocoo()
    Fp = np.asarray(nlp.evaluate_eq_constraints(), dtype=float)
    # Pyomo native index -> family index (every Pyomo row/col must be in sq).
    pr, prr = _bijection(sq.row_names, rpos, prow, "pyomo rows")
    pc, pcr = _bijection(sq.col_names, cpos, pcol, "pyomo cols")
    if pr is None or pc is None:
        why = {
            "skipped": "Pyomo rows/cols are not a bijection with sq",
            "rows": prr,
            "cols": pcr,
        }
        out["pyomo_J"] = dict(why)
        out["pyomo_order_pyomo_pattern"] = dict(why)
        return out
    JP = csr_matrix((Jp.data, (Jp.row, Jp.col)), shape=Jp.shape)
    JP.sort_indices()
    res = _mumps_step(JP, Fp, "pyomo_J", keep_solution=True)
    sol = res.pop("solution", None)
    res["explicit_zero_entries"] = int((JP.data == 0).sum())
    if sol is not None and ref is not None:
        full = np.empty_like(sol)
        full[pc] = sol
        res.update(_same_step(full, ref))
    out["pyomo_J"] = res

    # Family values in Pyomo's order, on Pyomo's pattern plus the family's entries.
    Jf = J[pr][:, pc].tocoo()
    rows = np.concatenate([Jf.row, Jp.row])
    colsj = np.concatenate([Jf.col, Jp.col])
    data = np.concatenate([Jf.data, np.zeros(Jp.nnz)])
    Jd = csr_matrix((data, (rows, colsj)), shape=Jp.shape)  # sums duplicates, keeps 0s
    Jd.sort_indices()
    res = _mumps_step(Jd, F[pr], "pyomo_order_pyomo_pattern", keep_solution=True)
    sol = res.pop("solution", None)
    res["pyomo_only_entries"] = int(Jd.nnz - Jf.nnz)
    if sol is not None and ref is not None:
        full = np.empty_like(sol)
        full[pc] = sol
        res.update(_same_step(full, ref))
    out["pyomo_order_pyomo_pattern"] = res
    return out


def _same_pattern(J, prev) -> bool:
    if prev is None or J.shape != prev.shape or J.nnz != prev.nnz:
        return False
    import numpy as np

    return bool(
        np.array_equal(J.indptr, prev.indptr)
        and np.array_equal(J.indices, prev.indices)
    )


def projected(calls: list[dict]) -> float | None:
    """Σ_k [symbolic_k + NIT_k (FJ_k + solve_k) + NFACT_k factor_k]; symbolic_k = 0
    when the call has the pattern of the previous one. ``None`` if MUMPS failed."""
    if len(calls) != len(NIT) or not all(c["mumps_ok"] for c in calls):
        return None
    return float(
        sum(
            (0.0 if c["same_pattern_as_prev"] else c["symbolic_s"])
            + NIT[k] * (c["FJ_s"] + c["solve_s"])
            + NFACT[k] * c["factor_s"]
            for k, c in enumerate(calls)
        )
    )


def measure(
    dataset: str,
    capture: Path,
    cache_root: Path | None = None,
    colperm: str = "pairing",
    orderings: bool = False,
    pyomo: bool = True,
    weighted: bool = False,
) -> dict:
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.disk_cache import cache_key, enable_jax_cache
    from equilibria.family.mp_arrays import MPArrays
    from equilibria.family.prep_capture import pairs
    from equilibria.family.prep_recipe import run_recipe, sp_reference_arrays
    from equilibria.family.prep_square import ConditionalMasks, square
    from equilibria.family.squared_system import SquaredSystem, period_incidence
    from equilibria.templates.gtap import gtap_block_model as gbm

    prep, budget = _sibling("prep_cost_20x41"), _sibling("budget_20x41")
    cache_root = cache_root or Path.home() / ".cache" / "equilibria" / "family"
    enable_jax_cache(cache_root)

    p = prep._params(dataset)
    closure = prep._closure()
    rr = list(p.sets.r)[-1]
    ref = sp_reference_arrays(p, closure, rr)

    # The single-period state and the backend (recorded once, then loaded from disk).
    bclosure = budget._closure()
    model, state = gbm.compose_block_state(p, p.sets, bclosure, rr)
    key = cache_key(p, bclosure, rr)
    if key is None:
        raise RuntimeError("no cache key (uncoverable input)")
    path = cache_root / key / "records.pkl"
    if not path.exists():
        b0 = FamilyBackend()
        b0.build(model, skip=state.ifsub_skipped)
        b0.save(path)
    t = time.perf_counter()
    backend = FamilyBackend.load(path)
    backend_load_s = time.perf_counter() - t

    pyomo_model = build_pyomo_model(dataset) if orderings and pyomo else None
    masks = None
    inc_cache: dict[str, tuple[dict, float]] = {}
    calls, prev_J = [], None
    for a, b in pairs(capture):
        mp = MPArrays.from_photo(a)
        mp = run_recipe(
            mp,
            a.period,
            params=p,
            closure=closure,
            sp_ref=ref,
            mode="gtap",
            backend=backend,
            mute_welfare=True,
            seed_from_prior=False,
            base_calibrated=True,
            solve_check=True,
        )
        if masks is None:
            masks = ConditionalMasks.from_params(p, mp)
        if a.period not in inc_cache:
            t = time.perf_counter()
            pat = period_incidence(backend, state, a.period)
            inc_cache[a.period] = (pat, time.perf_counter() - t)
        pat, incidence_s = inc_cache[a.period]
        # B's incidence only for the rows the pattern does not give (cross-period).
        merged = {r: list(c) for r, c in zip(b.con_names, b.incidence, strict=True)}
        merged.update(pat)
        sq = square(
            mp,
            a.period,
            params=p,
            closure=closure,
            incidence=merged,
            exogenous=ref.exogenous,
            masks=masks,
        )

        t = time.perf_counter()
        ss = SquaredSystem.build(backend, state, mp, a.period, sq, proxies=True)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        ss.F(ss.x0)
        ss.J(ss.x0)
        fj_first_s = time.perf_counter() - t  # includes loading the compiled programs
        t = time.perf_counter()
        F = ss.F(ss.x0)
        J = ss.J(ss.x0)
        fj_s = time.perf_counter() - t
        J.sort_indices()

        m = _mumps_step(J, F, colperm)
        if orderings:
            nlp = None
            if pyomo_model is not None:
                nlp = pyomo_nlp_at_b(pyomo_model, capture, a, b)
            m["orderings"] = _orderings(ss, sq, b, nlp, _hard_rows(mp, sq, p), weighted)
        calls.append(
            {
                "period": a.period,
                "n": len(sq.row_names),
                "nnz": int(J.nnz),
                "proxy_rows": len(ss.proxy_rows),
                "proxy_families": dict(sorted(ss.proxy_families.items())),
                "proxy_cols_outside_backend": ss.proxy_cols_outside_backend,
                "incidence_s": incidence_s,
                "build_s": build_s,
                "FJ_first_s": fj_first_s,
                "FJ_s": fj_s,
                "same_pattern_as_prev": _same_pattern(J, prev_J),
                **m,
            }
        )
        prev_J = J
    return {
        "dataset": dataset,
        "colperm": colperm,
        "NIT": list(NIT),
        "NFACT": list(NFACT),
        "ICNTL": {str(k): v for k, v in icntl_for(colperm).items()},  # as passed
        "CNTL3": CNTL3,
        "backend_load_s": backend_load_s,
        "calls": calls,
        "newton_projected_s": projected(calls),
    }


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--cache-root", type=Path)
    ap.add_argument("--colperm", choices=VARIANTS, default="pairing")
    ap.add_argument(
        "--orderings",
        action="store_true",
        help="also factor the ORDERINGS (pyomo_order / pyomo_J / "
        "pyomo_order_pyomo_pattern) per call; needs PyomoNLP unless --no-pyomo",
    )
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="with --orderings: also the weighted / weighted_free orders (scipy "
        "min_weight_full_bipartite_matching; SLOW: over 30 min per call on 20x41)",
    )
    ap.add_argument(
        "--no-pyomo", action="store_true", help="with --orderings: skip PyomoNLP"
    )
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.weighted and not a.orderings:
        ap.error("--weighted needs --orderings")
    cap = (
        a.capture
        or Path.home() / ".cache/equilibria/family/reference" / a.dataset / "tol_today"
    )
    result = {
        "sha": _sha(),
        **measure(
            a.dataset,
            cap,
            a.cache_root,
            a.colperm,
            orderings=a.orderings,
            weighted=a.weighted,
            pyomo=not a.no_pyomo,
        ),
    }
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
