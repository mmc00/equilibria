"""``b.target`` on family anchors to the check, as Pyomo's ``shock_only`` does
(MAR-251): ``quantity[shock] = target[shock] x quantity[check]``.

The reference is Pyomo's solved point: the notebook's exercise is solved on Pyomo, its
levels are written into family's arrays for every period, and family's target row at
the shock must be zero there. Before anchoring, the row still carries the BASE
quantity it folded when it was recorded; for ME9A (check != base) that is not zero,
so the test can fail. Each exercise runs in its own process (hooks last the whole
process, #214)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.examples.test_burfisher_notebooks import NB_DIR, _code

pytest.importorskip("jax")
pytestmark = [pytest.mark.integration, pytest.mark.needs_path]

ROOT = Path(__file__).resolve().parents[2]
TARGET_ROW = {"TBL94": "eq_qca_target", "ME9A": "eq_gdp_target"}
_SOLVED = "res = ex.run()\n"
_PROBE = """
import json as _json
from equilibria.family.overwrite_targets import anchor_targets, leaf_reader, node_value
from equilibria.family.run_setup import prepare_run
from tests.family._period_levels import levels as _levels

_want = _levels(ex.model)
_run, _mp = prepare_run(p, gc, "ROW", cache_root={cache!r}, jax_cache=False)
for (_name, _idx, _t), _v in _want.items():
    if _name in _mp.vars:
        _var = _mp.vars[_name]
        _var.value[_var.pos((*_idx, _t))] = _v


def _residuals():
    leaf = leaf_reader(_run.backend, _mp, "shock")
    return {{
        str(idx): node_value(res.lhs, leaf) - node_value(res.rhs, leaf)
        for idx, res in _run.backend.records[{row!r}]
    }}


_before = _residuals()
anchor_targets(_run, _mp, "shock")
_after = _residuals()
anchor_targets(_run, _mp, "shock")
with open({out!r}, "w") as _f:
    _json.dump({{"before": _before, "after": _after, "again": _residuals()}}, _f)
"""


def _probe(exp: str, tmp_path: Path) -> dict:
    code = _code(NB_DIR / f"{exp.lower()}.ipynb")
    assert code.count(_SOLVED) == 1, exp
    out = tmp_path / "residuals.json"
    probe = _PROBE.format(
        cache=str(tmp_path / "cache"), row=TARGET_ROW[exp], out=str(out)
    )
    code = code[: code.index(_SOLVED) + len(_SOLVED)] + probe
    script = tmp_path / "probe.py"
    script.write_text(code, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=NB_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    assert r.returncode == 0, f"{exp}: {r.stderr[-3000:]}"
    return json.loads(out.read_text())


@pytest.mark.parametrize("exp", ["ME9A", pytest.param("TBL94", marks=pytest.mark.slow)])
def test_target_row_is_zero_at_pyomos_shock_point(exp: str, tmp_path: Path) -> None:
    r = _probe(exp, tmp_path)
    assert r["after"], exp
    assert max(abs(v) for v in r["after"].values()) < 1e-10, r["after"]
    assert r["again"] == r["after"]  # anchoring twice is anchoring once
    if exp == "ME9A":  # its check is not its base: the base anchor is off
        assert max(abs(v) for v in r["before"].values()) > 1e-6, r["before"]


def _price_target(check: float):
    """A ``b.target`` on a price that is exactly 1.0 at the base (every GTAP price
    is), recorded as ``BlockEdit.target`` records it, on arrays whose check holds
    ``check``: ``(run, mp, records)``."""
    from types import SimpleNamespace

    import numpy as np
    from pyomo.environ import value

    from equilibria.blocks.gtap.overwrite import Target
    from equilibria.family.mp_arrays import MPArrays, RowArr, VarArr
    from equilibria.family.recorder import RecordingModel

    sets = {"r": ["USA"]}
    m = RecordingModel(
        sets,
        {},
        {"pft": (("r",), np.array([1.0])), "pf_target": (("r",), np.array([1.0]))},
    )
    res = m.pft["USA"] == m.pf_target["USA"] * float(value(m.pft["USA"]))
    records = {"eq_pf_target": [(("USA",), res)]}
    keys = [("USA", t) for t in ("base", "check", "shock")]
    vars_ = {
        "pft": VarArr(
            keys,
            np.array([1.0, check, check * 0.99]),
            *(np.zeros(3, bool), np.zeros(3), np.full(3, np.inf)),
        ),
        "pf_target": VarArr(
            keys,
            np.array([1.0, 1.0, 0.99]),
            *(np.ones(3, bool), np.zeros(3), np.full(3, np.inf)),
        ),
    }
    mp = MPArrays(
        vars_,
        {"eq_pf_target": RowArr([("USA", "shock")], np.ones(1, bool))},
        targets={"pf_target": Target(("r",), [(("USA",), lambda m, r: m.pft[r])])},
    )
    backend = SimpleNamespace(records=records, var_domains=m.var_domains, sets=sets)
    return SimpleNamespace(backend=backend), mp, records


@pytest.mark.parametrize("check", [1.05, 1.0])
def test_a_target_whose_base_quantity_is_one_anchors(check: float) -> None:
    """The recorder folds ``target x 1.0`` to ``target``: the anchor still applies,
    also when the check is exactly 1.0 again (anchoring twice)."""
    from equilibria.family.overwrite_targets import (
        anchor_targets,
        leaf_reader,
        node_value,
    )

    run, mp, records = _price_target(check)
    for _ in range(2):
        anchor_targets(run, mp, "shock")
        ((_, res),) = records["eq_pf_target"]
        leaf = leaf_reader(run.backend, mp, "shock")
        assert node_value(res.lhs, leaf) - node_value(res.rhs, leaf) == pytest.approx(
            0.0, abs=1e-15
        )
