"""The parity-ladder script on 3x3: the gate passes on the real run, fails on a bent cell."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.family._prep import _newton_mumps, capture_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _script():
    path = ROOT / "scripts" / "family" / "parity_ladder.py"
    spec = importlib.util.spec_from_file_location("parity_ladder", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linsolve():
    return "mumps" if importlib.util.find_spec("mumps") else "direct"


@pytest.fixture(scope="module")
def ran(linsolve, tmp_path_factory):
    s = _script()
    ref = capture_dir("gtap7_3x3", linsolve)
    env = {
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve,
        "EQUILIBRIA_FAMILY_CACHE": str(tmp_path_factory.mktemp("cache")),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    out_file = tmp_path_factory.mktemp("out") / "r.json"
    try:
        rc = s.main(["gtap7_3x3", "--out", str(out_file), "--reference", str(ref)])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return s, rc, json.loads(out_file.read_text()), ref


def test_gate_passes_on_3x3(ran):
    _, rc, out, _ = ran
    assert rc == 0
    assert out["gate_passed"] is True
    assert out["over_1e9"] == 0 and out["cells"] > 0
    assert out["fixed_diff"] == out["ub_diff"] == out["lb_diff"] == 0
    assert out["code"] == 1 and set(out["codes"].values()) == {1}


def test_report_has_what_the_ladder_needs(ran):
    _, _, out, _ = ran
    for key in ("nit_ours", "nit_reference", "max_rel_by_component", "wall_s"):
        assert key in out
    assert out["peak_rss_gb"] > 0 and out["sha"]
    assert len(out["worst_cells"]) <= 20
    assert set(out["nit_reference"]) == {"002_check", "004_check", "005_shock"}


def test_gate_fails_on_a_bent_cell(ran):
    import copy

    from equilibria.family.prep_capture import load

    s, _, _, ref = ran
    final = load(ref / "final.pkl")
    name = next(n for n, c in final.vars.items() if np.isfinite(c.value).any())
    c = final.vars[name]
    bent = copy.deepcopy(final)
    i = int(np.argmax(np.isfinite(c.value)))
    bent.vars[name].value = bent.vars[name].value.copy()
    bent.vars[name].value[i] += 1e-6 * max(1.0, abs(bent.vars[name].value[i]))

    class MP:  # the true values are final's own, so bending the photo is a bent cell
        vars = {n: _v(x) for n, x in final.vars.items()}

    out = s.compare(MP, bent)
    assert out["over_1e9"] == 1
    assert out["worst_cells"][0]["var"] == name
    assert out["worst_cells"][0]["key"] == list(c.keys[i])


def _v(c):
    class V:
        keys = c.keys
        value = c.value
        fixed = c.fixed
        lb = c.lb
        ub = c.ub

        def pos(self, k):
            return list(c.keys).index(k)

    return V()
