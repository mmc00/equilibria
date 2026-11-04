"""Gate 2 (family backend, Phase 0): build + one F/J evaluation vs Pyomo, per dataset.

Each dataset runs in its own subprocess so JAX start-up and compile caches do not leak
between sizes. For the Phase 0 families (eq_pndeq, eq_pvaeq, eq_facty):

- family: ``data_s`` (``_derived_params`` dicts -> arrays, shared recipe with Pyomo),
  ``layout_s`` (slot arrays), ``build_s`` (index arrays + FamilySystem), ``first_FJ_s``
  (first F + J, includes tracing/compilation), ``warm_FJ_s`` (median of 5),
  ``compile_s = first_FJ_s - warm_FJ_s``.
- pyomo: ``expr_s`` (the blocks' ``build_expression`` for these families, timed inside
  the real SP build), ``component_s`` (creating the Constraint components from those
  expressions), ``nlp_s`` (PyomoNLP with only these families active), ``FJ_s`` (one
  ``evaluate_eq_constraints`` + ``evaluate_jacobian_eq``).

Usage:
    uv run python scripts/family/gate2_scaling.py --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("gtap7_3x3", "gtap7_10x7", "gtap7_15x10")
FAMILIES = ("eq_pndeq", "eq_pvaeq", "eq_facty")


def _measure(dataset: str) -> dict:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Constraint, Objective

    from equilibria.core.equations import EquationManager
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from tests.family._data import load_params
    from tests.family._oracle import Oracle, comp_stat_closure

    p = load_params(dataset)

    # --- Pyomo: time the families' build_expression inside the real SP build ----------
    expr_s = {"t": 0.0}
    original_get = EquationManager.get

    def timed_get(self, name):
        eq = original_get(self, name)
        if name in FAMILIES and not getattr(eq, "_gate2_timed", False):
            build = eq.build_expression

            def timed(pm, idx, _build=build):
                t0 = time.perf_counter()
                out = _build(pm, idx)
                expr_s["t"] += time.perf_counter() - t0
                return out

            object.__setattr__(eq, "build_expression", timed)
            object.__setattr__(eq, "_gate2_timed", True)
        return eq

    EquationManager.get = timed_get
    try:
        from equilibria.templates.gtap.gtap_block_model import _set_elems

        sets = _set_elems(p.sets)
        rr = sets["r"][-1]
        pm = build_block_single_period(p, p.sets, comp_stat_closure(), rr)
    finally:
        EquationManager.get = original_get

    # Constraint components rebuilt from the already-built expressions.
    exprs = {n: {k: getattr(pm, n)[k].expr for k in getattr(pm, n)} for n in FAMILIES}
    t0 = time.perf_counter()
    for n, d in exprs.items():
        setattr(pm, f"_gate2_{n}", Constraint(list(d), rule=lambda m, *k, d=d: d[k if len(k) > 1 else k[0]]))
    component_s = time.perf_counter() - t0
    for n in FAMILIES:
        pm.del_component(f"_gate2_{n}")

    # PyomoNLP + one evaluation with only these families active.
    for c in pm.component_objects(Constraint, active=True):
        if c.name not in FAMILIES:
            c.deactivate()
    pm._gate2_obj = Objective(expr=0.0)
    t0 = time.perf_counter()
    nlp = PyomoNLP(pm)
    nlp_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    nlp.evaluate_eq_constraints()
    J = nlp.evaluate_jacobian_eq()
    pyomo_FJ_s = time.perf_counter() - t0
    pyomo_nnz = int(J.nnz)

    # --- Family backend ---------------------------------------------------------------
    from equilibria.family.gtap_phase0 import build_phase0_system, phase0_data
    from equilibria.family.layout import VarLayout
    from equilibria.family.gtap_phase0 import PHASE0_VARS

    o = Oracle(dataset, p, sets, rr, pm, nlp, [], [])
    values, free = o.state()  # oracle side: variable levels and fixed flags

    t0 = time.perf_counter()
    data = phase0_data(p, p.sets, rr)
    data_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    layout = VarLayout.from_state(sets, PHASE0_VARS, values, free)
    layout_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    system = build_phase0_system(p, p.sets, layout, rr, data=data)
    build_s = time.perf_counter() - t0

    z = layout.free_values
    t0 = time.perf_counter()
    system.F(z)
    system.jac(z)
    first_FJ_s = time.perf_counter() - t0
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        system.F(z)
        system.jac(z)
        warm.append(time.perf_counter() - t0)
    warm_FJ_s = statistics.median(warm)

    n_terms = sum(len(f.term_rows) for f in system.families if f.has_terms)
    return {
        "dataset": dataset,
        "rows": system.n_rows,
        "terms": int(n_terms),
        "cells": int(system.n_rows + n_terms),
        "nnz": system.nnz,
        "pyomo_nnz": pyomo_nnz,
        "family": {
            "data_s": data_s,
            "layout_s": layout_s,
            "build_s": build_s,
            "first_FJ_s": first_FJ_s,
            "warm_FJ_s": warm_FJ_s,
            "compile_s": first_FJ_s - warm_FJ_s,
        },
        "pyomo": {
            "expr_s": expr_s["t"],
            "component_s": component_s,
            "nlp_s": nlp_s,
            "FJ_s": pyomo_FJ_s,
        },
        "_np": np.__version__,
    }


def _slope(xs, ys) -> float:
    import numpy as np

    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="measure one dataset (child mode)")
    ap.add_argument("--out", type=Path, help="JSON output (driver mode)")
    args = ap.parse_args()

    if args.dataset:
        print(json.dumps(_measure(args.dataset)))
        return 0

    results = []
    for ds in DATASETS:
        out = subprocess.run(
            [sys.executable, __file__, "--dataset", ds],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(ds, "done", file=sys.stderr)

    cells = [r["cells"] for r in results]
    fam_scaling = [r["family"]["build_s"] + r["family"]["warm_FJ_s"] for r in results]
    compile_ = [r["family"]["compile_s"] for r in results]
    fam_total = [
        r["family"]["build_s"] + r["family"]["layout_s"] + r["family"]["first_FJ_s"]
        for r in results
    ]
    pyo_total = [
        r["pyomo"]["expr_s"] + r["pyomo"]["component_s"] + r["pyomo"]["nlp_s"] + r["pyomo"]["FJ_s"]
        for r in results
    ]
    summary = {
        "slope_build_plus_warm_vs_cells": _slope(cells, fam_scaling),
        "slope_pyomo_total_vs_cells": _slope(cells, pyo_total),
        "compile_ratio_15x10_over_3x3": compile_[-1] / compile_[0],
        "family_total_s": dict(zip(DATASETS, fam_total, strict=True)),
        "pyomo_total_s": dict(zip(DATASETS, pyo_total, strict=True)),
        "criteria": {
            "a_slope_le_1.2": None,
            "b_compile_ratio_le_2": None,
            "c_total_le_pyomo_at_15x10": None,
        },
    }
    summary["criteria"]["a_slope_le_1.2"] = summary["slope_build_plus_warm_vs_cells"] <= 1.2
    summary["criteria"]["b_compile_ratio_le_2"] = summary["compile_ratio_15x10_over_3x3"] <= 2.0
    summary["criteria"]["c_total_le_pyomo_at_15x10"] = fam_total[-1] <= pyo_total[-1]

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    payload = {
        "git_sha": sha,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "python": platform.python_version(),
        "datasets": {r["dataset"]: r for r in results},
        "summary": summary,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
