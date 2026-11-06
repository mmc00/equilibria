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


@contextlib.contextmanager
def _gc_frozen_permanent_graph(enabled: bool = True):
    """Move everything alive at entry to the GC's permanent generation.

    By the time we solve, the Pyomo model is a huge, permanent object graph:
    MEASURED on gtap7_20x41, 56.2M tracked objects. Every full (gen-2) pass
    walks all of them, and CPython schedules those passes by ALLOCATION counts,
    not by how much is actually collectable — so the solve's own allocation
    churn keeps triggering full walks over a graph that never has garbage in it.

    MEASURED cost of that on gtap7_20x41: gen-2 ran 42 times at ~3.4 s each =
    2.43 min, which is 94% of all GC time and ~29% of the wall. gen-0 and gen-1
    together cost 0.07 min over 243k collections — i.e. the generational
    hypothesis is right, it is the FULL passes that hurt.

    `gc.freeze()` moves the currently-tracked objects into a permanent set that
    collections skip. It does NOT disable the GC: cyclic garbage created after
    the freeze is still collected normally (verified: a frozen 4M-object graph
    plus 600k fresh cyclic objects still reclaimed all 600k, at 32 ms instead of
    130 ms). That is the property that makes this safe where `gc.disable()`
    would not be — the solve does create cycles, and they still go away.

    MEASURED end-to-end on gtap7_20x41, 3 interleaved repetitions, PATH:
    wall 8.49 → 7.81 min median (-0.68, 8%), gen-2 2.43 → 1.89 min, with the
    solution sha256 IDENTICAL across all 6 runs. Freezing changes what the GC
    walks, never what the model computes.

    Counter-intuitive but expected: gen-2 runs MORE often frozen (42 → 73) and
    still costs less, because each pass no longer traverses the 40M permanent
    objects. Total collections are unchanged (~243k), i.e. nothing stopped
    being collected.

    The unfreeze is in a finally: leaving objects frozen would silently leak
    them for the rest of the process, which matters because callers solve
    several models per session.

    KNOWN LIMITATION: gc.unfreeze() is GLOBAL — CPython offers no way to release
    only what this call froze, so if a CALLER had frozen objects of its own
    before entering, those come back under the GC when we exit (verified: a
    pre-existing frozen set of 6325 goes to 0 after our unfreeze). Nothing in
    this repo calls gc.freeze(), so today the effect is nil; the consequence if
    someone adds one is a slower GC, never a wrong answer.
    """
    if not enabled:
        yield
        return
    import gc

    # Promote what is alive NOW to gen-2 first; freeze() only moves the
    # permanent generation, so without this the young survivors stay walkable.
    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def _gc_freeze_from_env() -> bool:
    """EQUILIBRIA_GTAP_GC_FREEZE: unset -> on, "0" -> off."""
    import os

    return os.environ.get("EQUILIBRIA_GTAP_GC_FREEZE", "1") != "0"


def _gc_freeze_now(enabled: bool = True) -> None:
    """Promote what is alive now and move it to the permanent generation.

    For objects that become long-lived AFTER the outer `_gc_frozen_permanent_graph`
    was entered (e.g. the family route loads its records after the entry point).
    Nesting is fine: the outer context's `gc.unfreeze()` releases everything.
    """
    if not enabled:
        return
    import gc

    gc.collect()
    gc.freeze()


@contextlib.contextmanager
def _gc_paused():
    """Disable the collector while bulk-loading a big object graph, then restore.

    Unpickling millions of long-lived containers makes the GC walk an ever-growing
    young generation for nothing. Restores the PREVIOUS state, so a caller that
    had the GC disabled keeps it disabled.
    """
    import gc

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
