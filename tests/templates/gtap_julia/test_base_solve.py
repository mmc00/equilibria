"""Task 12: the assembled gtap_julia model solves the base and matches Julia."""

import pyomo.environ as pyo
import pytest

from equilibria.templates.gtap_julia.model import build_model, solve
from equilibria.templates.gtap_julia.solution import dump_solution, load_solution


@pytest.mark.slow
def test_base_solve_is_square_and_matches_julia(tmp_path):
    sol = load_solution(dump_solution(dataset="sample", out_dir=tmp_path))

    # square system (free vars == constraints)
    m = build_model(sol, rordelta=1)
    nv = sum(1 for v in m.component_data_objects(pyo.Var) if not v.fixed)
    nc = sum(1 for c in m.component_data_objects(pyo.Constraint))
    assert nv == nc, f"system not square: {nv} free vars vs {nc} constraints"

    # solve + match Julia base cell-by-cell
    res = solve(sol, rordelta=1)
    assert res["ok"], f"IPOPT did not converge: {res['status']}"
    m = res["model"]

    diffs = []
    for v in ("qo", "qxs", "pds", "qva", "qfe", "qc", "qga", "qpa", "peb", "rore"):
        pv = getattr(m, v)
        d = sol["all"].get(v, {})
        for idx in pv:
            key = tuple(str(k) for k in (idx if isinstance(idx, tuple) else (idx,)))
            ref = d.get(key)
            if ref is not None and ref == ref and abs(ref) > 1e-9:
                diffs.append(abs(pyo.value(pv[idx]) / ref - 1))
    within = sum(1 for x in diffs if x <= 0.01) / len(diffs)
    assert within >= 0.99, f"base match {within * 100:.1f}% < 99% vs Julia oracle"
