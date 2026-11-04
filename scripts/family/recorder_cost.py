"""Recorder spike: what does reading the GTAP families from the blocks cost?

Per dataset, in its own subprocess, with GC collected and disabled while timing:

- ``compose_s``: compose the blocks into an equilibria Model (shared by both backends).
- Pyomo: ``pyomo_build_s`` (``PyomoBackend.build`` of the whole SP) and ``pyomo_nlp_s``
  (``PyomoNLP`` of it, i.e. the ``.nl`` round trip the driver pays).
- Family: ``record_s`` (every equation's ``build_expression`` on a RecordingModel),
  ``group_s`` (term groups for an all-free layout), ``first_FJ_s`` (first F + J, i.e.
  compilation) and ``warm_FJ_s`` (median of 5).
- ``groups``: term groups per family; ``errors``: families the recorder could not read.

Usage:
    uv run python scripts/family/recorder_cost.py --out <file.json> [--datasets ...]
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
DATASETS = ("gtap7_3x3", "gtap7_15x10", "gtap7_20x41")


def _timed(fn):
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        dt = time.perf_counter() - t0
        gc.enable()
    return out, dt


def _measure(dataset: str) -> dict:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.environ import Objective

    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.backends.pyomo_backend import PyomoBackend
    from equilibria.blocks.gtap import _ifsub_macros as mac
    from equilibria.family.core import FamilySystem
    from equilibria.family.layout import VarLayout
    from equilibria.templates.gtap.gtap_block_model import _set_elems, compose_block_sp
    from tests.family._data import load_params
    from tests.family._oracle import comp_stat_closure

    p = load_params(dataset)
    rr = _set_elems(p.sets)["r"][-1]
    (model, *_), compose_s = _timed(lambda: compose_block_sp(p, p.sets, comp_stat_closure(), rr))

    # --- Pyomo ---------------------------------------------------------------------
    pyo = PyomoBackend()

    def _pyomo_build():
        with mac.sharing():
            pyo.build(model)

    _, pyomo_build_s = _timed(_pyomo_build)
    pm = pyo.pyomo_model
    pm._recorder_cost_obj = Objective(expr=0.0)
    nlp, pyomo_nlp_s = _timed(lambda: PyomoNLP(pm))
    n_rows_pyomo = int(nlp.n_eq_constraints())
    del nlp, pm, pyo
    gc.collect()

    # --- Family backend: record family by family (to isolate failures) -------------
    fb = FamilyBackend()
    errors: dict[str, str] = {}
    names = list(model.equation_manager.list_equations())

    def _record_all():
        for name in names:
            try:
                fb.build(model, only=[name])
            except Exception as e:  # noqa: BLE001 — a diagnostic: report and go on
                errors[name] = f"{type(e).__name__}: {e}"[:300]

    # FamilyBackend.build resets nothing, so per-name calls accumulate records.
    _, record_s = _timed(_record_all)
    n_rows_family = sum(len(v) for v in fb.records.values())

    shapes = {v: tuple(len(fb.sets[d]) for d in doms) for v, doms in fb.var_domains.items()}
    layout = VarLayout.from_state(
        fb.sets,
        fb.var_domains,
        values={v: np.ones(s) for v, s in shapes.items()},
        free={v: np.ones(s, bool) for v, s in shapes.items()},
    )
    fams, group_s = _timed(lambda: fb.families(layout))
    system, system_s = _timed(lambda: FamilySystem.from_families(fams, layout))
    z = layout.free_values

    def _fj():
        system.F(z)
        system.jac(z)

    _, first_FJ_s = _timed(_fj)
    warm = [_timed(_fj)[1] for _ in range(5)]
    return {
        "dataset": dataset,
        "families_recorded": len(fb.records),
        "families_total": len(names),
        "rows_family": n_rows_family,
        "rows_pyomo": n_rows_pyomo,
        "nnz_family": system.nnz,
        "groups_total": sum(len(f.groups) for f in fams),
        "groups": {f.name: len(f.groups) for f in fams},
        "errors": errors,
        "compose_s": compose_s,
        "pyomo_build_s": pyomo_build_s,
        "pyomo_nlp_s": pyomo_nlp_s,
        "record_s": record_s,
        "group_s": group_s,
        "system_s": system_s,
        "first_FJ_s": first_FJ_s,
        "warm_FJ_s": statistics.median(warm),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="measure one dataset (child mode)")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.dataset:
        print(json.dumps(_measure(args.dataset)))
        return 0
    results = {}
    for ds in args.datasets:
        out = subprocess.run(
            [sys.executable, __file__, "--dataset", ds],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if out.returncode != 0:
            results[ds] = {"failed": out.stderr[-3000:]}
        else:
            results[ds] = json.loads(out.stdout.strip().splitlines()[-1])
        print(ds, "done", file=sys.stderr)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    text = json.dumps(
        {
            "git_sha": sha,
            "machine": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "datasets": results,
        },
        indent=2,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
