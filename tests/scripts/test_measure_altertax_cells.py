"""The step-0 acceptance rule: no cell crosses out of the gates' tolerance.

A deletion of an altertax recalibration is accepted when no cell that is within
tolerance in the last accepted run ends up outside it, no stage's match % goes
down, no converged solve stops converging, and the same cells are compared.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "measure_altertax_cells.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_altertax_cells", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mac = _load()

ROW = "mcp:gtap7_3x3:ifsub0"
CELL = "pft|USA,Land|shock"
BASE = {
    ROW: {
        "codes": {"base": 1, "check": 1, "shock": 1},
        "match": {"base": 100.0, "check": 100.0, "shock": 50.0},
        "cells": {
            CELL: [1.0099, 1.0, True],
            "xf|USA,Land,Crops|shock": [2.0, 1.0, False],
        },
    }
}


def _new():
    return copy.deepcopy(BASE)


def test_identical_runs_are_accepted():
    assert mac.verdict(BASE, BASE) == []


def test_cell_crossing_out_rejects():
    new = _new()
    new[ROW]["cells"][CELL] = [1.02, 1.0, False]
    reasons = mac.verdict(BASE, new)
    assert any("crossed" in r and CELL in r for r in reasons), reasons


def test_cell_already_out_getting_worse_is_accepted():
    new = _new()
    new[ROW]["cells"]["xf|USA,Land,Crops|shock"] = [3.0, 1.0, False]
    assert mac.verdict(BASE, new) == []


def test_cell_crossing_in_is_accepted():
    new = _new()
    new[ROW]["cells"]["xf|USA,Land,Crops|shock"] = [1.0, 1.0, True]
    new[ROW]["match"]["shock"] = 100.0
    assert mac.verdict(BASE, new) == []


def test_stage_match_going_down_rejects():
    new = _new()
    new[ROW]["match"]["shock"] = 49.9
    reasons = mac.verdict(BASE, new)
    assert any("match" in r and "shock" in r for r in reasons), reasons


def test_code_one_to_other_rejects():
    new = _new()
    new[ROW]["codes"]["shock"] = 2
    reasons = mac.verdict(BASE, new)
    assert any("code" in r and "shock" in r for r in reasons), reasons


def test_row_missing_in_new_rejects():
    reasons = mac.verdict(BASE, {})
    assert any(ROW in r and "missing" in r for r in reasons), reasons


def test_cell_set_changing_rejects():
    new = _new()
    del new[ROW]["cells"]["xf|USA,Land,Crops|shock"]
    reasons = mac.verdict(BASE, new)
    assert any("cells" in r for r in reasons), reasons


def test_edge_of_tolerance_is_a_crossing():
    """rel 0.0099 -> 0.0101 crosses the 1% tolerance: that is the rule, not noise."""
    assert mac.within(1.0099, 1.0, tol=1e-2)
    assert not mac.within(1.0101, 1.0, tol=1e-2)
    new = _new()
    new[ROW]["cells"][CELL] = [1.0101, 1.0, mac.within(1.0101, 1.0, tol=1e-2)]
    assert any("crossed" in r for r in mac.verdict(BASE, new))


def test_a_new_measurement_does_not_keep_old_rows(tmp_path):
    """Rows of an earlier run left in --out would pass as measured (a skip hidden)."""
    out = tmp_path / "run.json"
    out.write_text('{"mcp:gtap7_3x3:ifsub0": {"codes": {}, "match": {}, "cells": {}}}')
    assert mac.initial_results(out, resume=False) == {}
    assert ROW in mac.initial_results(out, resume=True)
    assert mac.initial_results(tmp_path / "absent.json", resume=True) == {}


def test_a_row_that_failed_in_base_is_reported():
    """A base failure cannot be compared: say so instead of accepting in silence."""
    base = {ROW: {"error": "RuntimeError: no model captured"}}
    reasons = mac.verdict(base, BASE)
    assert any(ROW in r and "failed in base" in r for r in reasons), reasons
