"""Garbage-collector tuning shared by the Pyomo and the family routes.

Kept free of Pyomo imports so the array route can use it without loading Pyomo.
"""

from __future__ import annotations

import contextlib

_GC_THRESHOLD0 = 50_000


@contextlib.contextmanager
def _gc_relaxed_threshold(threshold0: int = _GC_THRESHOLD0):
    """Raise the gen-0 allocation threshold for the duration of the solve.

    Companion to _gc_frozen_permanent_graph, and it attacks the SAME cost from
    the other end. CPython triggers a gen-0 pass every `threshold0` net
    container allocations (default 700); every 10th gen-0 pass promotes into
    gen-1, and gen-1 feeds gen-2 — the expensive full pass. So the flood of
    cheap gen-0 passes is what schedules the costly ones.

    MEASURED on gtap7_20x41 (already with the freeze in place), 3 interleaved
    repetitions, 700 vs 50000:

        gen-0 passes   202,457 -> 2,772     (73x fewer)
        gen-2 passes        45 -> 18
        gen-2 time    1.92 min -> 1.42 min
        wall          7.38 min -> 6.98 min  (-0.41, wins 3/3)

    with the solution sha256 IDENTICAL across all 6 runs. This changes WHEN
    Python collects garbage, never what the model computes.

    COST: peak RSS 3.8 -> 4.8 GB (+1 GB), because garbage now lives longer
    before being reclaimed. That is the real trade-off of this lever and the
    reason it is overridable; on a memory-tight machine or a much larger
    dataset, set EQUILIBRIA_GTAP_GC_THRESHOLD=0 to keep CPython's default.
    (Neither of the write-ups this follows — Instagram's gc.freeze work and
    Close's GC tuning — reports the memory side, so this number is ours.)

    The threshold is restored on exit: it is process-global state, and leaving
    it raised would silently change the GC behaviour of everything that runs
    after us in the same process.
    """
    import gc

    if threshold0 <= 0:
        yield
        return
    prev = gc.get_threshold()
    # Only gen-0 moves; gen-1/gen-2 multipliers stay as CPython set them.
    gc.set_threshold(threshold0, prev[1], prev[2])
    try:
        yield
    finally:
        gc.set_threshold(*prev)


def _gc_threshold_from_env() -> int:
    """EQUILIBRIA_GTAP_GC_THRESHOLD: unset -> 50000, 0 -> off, N -> N.

    A non-numeric value is treated as "unset" rather than raising: a typo in an
    env var must not take down a 7-minute solve over a GC tuning knob.
    """
    import os

    raw = os.environ.get("EQUILIBRIA_GTAP_GC_THRESHOLD")
    if raw is None:
        return _GC_THRESHOLD0
    try:
        return int(raw)
    except ValueError:
        return _GC_THRESHOLD0
