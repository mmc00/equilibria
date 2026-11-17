"""The family parity rules count a non-finite family value as a miss, as the Pyomo
gates do (their NaN fails the comparison but stays in the denominator): dropping it
would let a run that blew up a block report 100% over fewer cells."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from tests.family import _parity


class _Var:
    def __init__(self, cells: dict):
        self._keys = list(cells)
        self.value = list(cells.values())

    def pos(self, key):
        return self._keys.index(tuple(key))


def _result(cells: dict):
    mp = SimpleNamespace(vars={"xp": _Var(cells)})
    return _parity._Run(mp, dict.fromkeys(("base", "check", "shock"), 1))


def test_family_match_counts_a_nan_cell_as_a_miss(monkeypatch):
    import _diff_core  # type: ignore

    cells = {("r1", "a1", t): 1.0 for t in ("base", "check", "shock")}
    cells[("r2", "a1", "shock")] = math.nan
    for t in ("base", "check"):
        cells[("r2", "a1", t)] = 1.0
    monkeypatch.setattr(_parity, "_solve", lambda *a, **k: (_result(cells), 1))
    monkeypatch.setattr(_diff_core, "list_populated_vars", lambda gdx: ["xp"])
    monkeypatch.setattr(
        _diff_core, "gams_levels", lambda gdx, vn: dict.fromkeys(cells, 1.0)
    )

    per = _parity.family_match("ds", 1, "pure", Path("x.gdx"))

    assert per["shock"]["total"] == 2
    assert per["shock"]["match"] == 50.0
    assert per["base"]["match"] == 100.0


def test_family_gempack_within_counts_a_nan_cell_as_a_miss(monkeypatch):
    import gempack_reference  # type: ignore

    cells = {}
    for r, shock in (("r1", 1.05), ("r2", math.nan)):
        cells[(r, "a1", "check")] = 1.0
        cells[(r, "a1", "shock")] = shock
    monkeypatch.setattr(_parity, "_solve", lambda *a, **k: (_result(cells), 1))
    monkeypatch.setattr(_parity, "gempack_reference", lambda ds: Path("sl4.har"))
    monkeypatch.setattr(gempack_reference, "Q_TO_VAR", {"qo": {"var": "xp"}})
    monkeypatch.setattr(
        gempack_reference,
        "gempack_qty_pct",
        lambda path, gvar: {("r1", "a1"): 0.05, ("r2", "a1"): 0.05},
    )

    within, _, n, code = _parity.family_gempack_within("ds", 1)

    assert (n, code) == (2, 1)
    assert within == 0.5
