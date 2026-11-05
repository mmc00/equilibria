"""Bitwise parity and wall time of the array ifSUB closure against the cell-by-cell one.

    <python> scripts/family/ifsub_closure_parity.py [--dataset gtap7_20x41]

Composes the family state twice (once with each ``_apply_ifsub_closure`` form), times
only the closure call, and compares every variable's levels and fixed flags with
``np.array_equal``. Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _closure():
    from tests.family.test_ifsub_closure_vectorized import closure

    return closure()


def _compose(dataset: str, impl_name: str):
    from tests.family._data import load_params

    import equilibria.templates.gtap.gtap_block_model as gbm

    p = load_params(dataset)
    rr = list(p.sets.r)[-1]
    real = gbm._apply_ifsub_closure
    impl = getattr(gbm, impl_name)
    timing: list[tuple[float, int]] = []

    def timed(pm, params=None):
        t = time.perf_counter()
        n = impl(pm, params)
        timing.append((time.perf_counter() - t, n))
        return n

    setattr(gbm, "_apply_ifsub_closure", timed)
    try:
        _, state = gbm.compose_block_state(p, p.sets, _closure(), rr)
    finally:
        setattr(gbm, "_apply_ifsub_closure", real)
    return state, timing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    args = ap.parse_args()
    print(f"load average: {os.getloadavg()}")
    old, t_old = _compose(args.dataset, "_apply_ifsub_closure_cells")
    new, t_new = _compose(args.dataset, "_apply_ifsub_closure")
    bad = 0
    cells = 0
    for name in old.values:
        a, b = old.values[name], new.values[name]
        cells += a.size
        if not np.array_equal(a, b, equal_nan=True):
            bad += int((a != b).sum())
            print(f"MISMATCH values {name}")
        if not np.array_equal(old.fixed[name], new.fixed[name]):
            bad += 1
            print(f"MISMATCH fixed {name}")
    print(
        f"closure old {t_old[0][0]:.3f}s  new {t_new[0][0]:.3f}s  counts {t_old[0][1]} {t_new[0][1]}"
    )
    print(f"compared {cells} cells over {len(old.values)} variables; mismatches {bad}")
    return 1 if bad or t_old[0][1] != t_new[0][1] else 0


if __name__ == "__main__":
    raise SystemExit(main())
