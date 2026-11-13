"""Lightweight phase timing for the family route: ``time.perf_counter`` accumulation.

``with timer("name"):`` adds the phase's *self* time (the time of phases nested inside
it is not counted twice), so the phases of one run add up to the run's wall.
``timer.scope("check")`` prefixes the names inside it (``check.recipe``). The module
level ``CURRENT`` is the timer the pieces of one run report to; outside a run it is
``NULL``, which records nothing. No behavior change: the timer only reads the clock.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseTimer:
    def __init__(self, record: bool = True) -> None:
        self.record = record
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._stack: list[list[float]] = []  # [start, child seconds]
        self._prefix: list[str] = []

    @contextmanager
    def scope(self, name: str):
        self._prefix.append(name)
        try:
            yield
        finally:
            self._prefix.pop()

    @contextmanager
    def __call__(self, name: str):
        if not self.record:
            yield
            return
        full = ".".join([*self._prefix, name])
        frame = [time.perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            total = time.perf_counter() - frame[0]
            self._stack.pop()
            self.seconds[full] += total - frame[1]
            self.counts[full] += 1
            if self._stack:
                self._stack[-1][1] += total

    def add(self, name: str, seconds: float) -> None:
        """Account ``seconds`` to a detail entry (not part of the additive phases)."""
        if self.record:
            self.seconds[name] += seconds
            self.counts[name] += 1


NULL = PhaseTimer(record=False)
CURRENT: PhaseTimer = NULL


def phase(name: str):
    """``with phase("name"):`` on the current run's timer."""
    return CURRENT(name)


def scope(name: str):
    return CURRENT.scope(name)


def activate(timer: PhaseTimer) -> PhaseTimer:
    """Make ``timer`` the current one; returns the previous."""
    global CURRENT
    prev, CURRENT = CURRENT, timer
    return prev
