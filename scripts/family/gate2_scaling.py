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

Garbage collection is collected and disabled while timing (both sides): a gen-2 pause
triggered by the whole SP build was otherwise charged to these families (review of
79d32ec: 0.095 s with GC vs 0.004 s without on 15x10).

Criterion (a) is judged on a synthetic eq_pvaeq-shaped family of 1e4..1e6 terms, since the
real datasets (<4k cells) only measure dispatch overhead.

Usage:
    uv run python scripts/family/gate2_scaling.py --out <file.json>
"""

from __future__ import annotations

import argparse
import gc
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
SYNTHETIC_TERMS = (10_000, 100_000, 1_000_000)


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
    gc.collect()
    gc.disable()

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
    pyo = []
    for _ in range(5):
        t0 = time.perf_counter()
        nlp.evaluate_eq_constraints()
        J = nlp.evaluate_jacobian_eq()
        pyo.append(time.perf_counter() - t0)
    pyomo_FJ_s = statistics.median(pyo)
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

    gc.enable()
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


def _synthetic(n_terms: int) -> dict:
    """An eq_pvaeq-shaped family (10 terms per row, 2 slots per term) of ``n_terms``."""
    import numpy as np

    from equilibria.family.core import Family, FamilySystem
    from equilibria.family.gtap_phase0 import _pndeq_row, _pvaeq_term

    rng = np.random.default_rng(0)
    n_rows = n_terms // 10
    n_cols = n_rows + n_terms
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    term_rows = np.repeat(np.arange(n_rows), 10)
    fam = Family(
        name="synthetic",
        row_fn=_pndeq_row,
        row_cols=np.arange(n_rows).reshape(-1, 1),
        row_args={"e": rng.uniform(-0.4, 0.8, n_rows)},
        term_fn=_pvaeq_term,
        term_rows=term_rows,
        term_cols=np.stack(
            [n_rows + np.arange(n_terms), n_cols + np.arange(n_terms)], axis=1
        ),
        term_args={"af": rng.uniform(0.1, 1.0, n_terms), "e": rng.uniform(-0.4, 0.8, n_terms)},
    )
    system = FamilySystem([fam], n_cols=n_cols, fixed_values=rng.uniform(0.8, 1.2, n_terms))
    build_s = time.perf_counter() - t0
    z = rng.uniform(0.8, 1.2, n_cols)
    t0 = time.perf_counter()
    system.F(z)
    system.jac(z)
    first = time.perf_counter() - t0
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        system.F(z)
        system.jac(z)
        warm.append(time.perf_counter() - t0)
    gc.enable()
    w = statistics.median(warm)
    return {
        "cells": n_rows + n_terms,
        "nnz": system.nnz,
        "build_s": build_s,
        "first_FJ_s": first,
        "warm_FJ_s": w,
        "compile_s": first - w,
    }


def _slope(xs, ys) -> float:
    import numpy as np

    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="measure one dataset (child mode)")
    ap.add_argument("--synthetic", type=int, help="synthetic family of N terms (child)")
    ap.add_argument("--out", type=Path, help="JSON output (driver mode)")
    args = ap.parse_args()

    if args.dataset:
        print(json.dumps(_measure(args.dataset)))
        return 0
    if args.synthetic:
        print(json.dumps(_synthetic(args.synthetic)))
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

    synthetic = []
    for n in SYNTHETIC_TERMS:
        out = subprocess.run(
            [sys.executable, __file__, "--synthetic", str(n)],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        synthetic.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print("synthetic", n, "done", file=sys.stderr)

    real_compile = [r["family"]["compile_s"] for r in results]
    fam_total = [
        r["family"]["build_s"] + r["family"]["layout_s"] + r["family"]["first_FJ_s"]
        for r in results
    ]
    pyo_total = [
        r["pyomo"]["expr_s"] + r["pyomo"]["component_s"] + r["pyomo"]["nlp_s"] + r["pyomo"]["FJ_s"]
        for r in results
    ]
    s_cells = [x["cells"] for x in synthetic]
    s_cost = [x["build_s"] + x["warm_FJ_s"] for x in synthetic]
    # Marginal per-cell costs: Pyomo between 10x7 and 15x10; family on the two largest
    # synthetic sizes (build + one warm F/J). Fixed costs: totals minus marginal*cells.
    c_pyo = (pyo_total[-1] - pyo_total[-2]) / (results[-1]["cells"] - results[-2]["cells"])
    c_fam = (s_cost[-1] - s_cost[-2]) / (s_cells[-1] - s_cells[-2])
    fixed_pyo = pyo_total[-1] - c_pyo * results[-1]["cells"]
    fixed_fam = fam_total[-1] - c_fam * results[-1]["cells"]
    crossover = (fixed_fam - fixed_pyo) / (c_pyo - c_fam) if c_pyo > c_fam else None
    summary = {
        "slope_synthetic_build_plus_warm_vs_cells_1e5_1e6": _slope(s_cells[-2:], s_cost[-2:]),
        "slope_pyomo_total_vs_cells": _slope([r["cells"] for r in results], pyo_total),
        "compile_ratio_15x10_over_3x3": real_compile[-1] / real_compile[0],
        "compile_ratio_synthetic_1e6_over_1e4": synthetic[-1]["compile_s"] / synthetic[0]["compile_s"],
        "per_cell_us": {"family": c_fam * 1e6, "pyomo": c_pyo * 1e6},
        "fixed_s": {"family": fixed_fam, "pyomo": fixed_pyo},
        "crossover_cells": crossover,
        "family_total_s": dict(zip(DATASETS, fam_total, strict=True)),
        "pyomo_total_s": dict(zip(DATASETS, pyo_total, strict=True)),
    }
    summary["criteria"] = {
        "a_slope_le_1.2": summary["slope_synthetic_build_plus_warm_vs_cells_1e5_1e6"] <= 1.2,
        "b_compile_ratio_le_2": summary["compile_ratio_15x10_over_3x3"] <= 2.0,
        "c_total_le_pyomo_at_15x10": fam_total[-1] <= pyo_total[-1],
    }

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    payload = {
        "git_sha": sha,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "python": platform.python_version(),
        "datasets": {r["dataset"]: r for r in results},
        "synthetic": synthetic,
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
