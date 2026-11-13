"""``headers_at`` builds the value headers of a run's period once.

The outputs (updated, summary, volume, decomp, post database) ask for the check and
shock headers of the same run about ten times; each build walks every cell in Python
(about 1 s per call on 20x41). The run keeps them, hands out copies, and a run made by
``dataclasses.replace`` (the decomp path's points) starts with none.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _counting(monkeypatch):
    from equilibria.family.altertax import updated

    calls = []
    real = updated._Period

    def counted(run, period):
        calls.append(period)
        return real(run, period)

    monkeypatch.setattr(updated, "_Period", counted)
    return calls


def test_second_call_reuses_the_period(altax_run, monkeypatch):
    from equilibria.family.altertax.updated import headers_at

    calls = _counting(monkeypatch)
    altax_run._headers.clear()
    first = headers_at(altax_run, "check")
    again = headers_at(altax_run, "check")
    assert calls == ["check"]
    assert first.keys() == again.keys()
    for name, h in first.items():
        assert np.array_equal(h.array, again[name].array)
    headers_at(altax_run, "shock")
    assert calls == ["check", "shock"]


def test_callers_get_copies(altax_run):
    from equilibria.family.altertax.updated import headers_at

    altax_run._headers.clear()
    first = headers_at(altax_run, "check")
    name = next(iter(first))
    keep = first[name].array.copy()
    first[name].array[...] = -1.0
    first.pop(name)
    again = headers_at(altax_run, "check")
    assert np.array_equal(again[name].array, keep)


def test_a_replaced_run_starts_empty(altax_run, monkeypatch):
    from equilibria.family.altertax.updated import headers_at

    headers_at(altax_run, "shock")
    other = dataclasses.replace(altax_run, result=altax_run.result)
    assert other._headers == {}
    calls = _counting(monkeypatch)
    headers_at(other, "shock")
    assert calls == ["shock"]
