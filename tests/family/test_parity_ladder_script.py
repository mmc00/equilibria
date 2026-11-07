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


def _finite_lb_cell(final):
    for n, c in final.vars.items():
        ok = np.flatnonzero(np.isfinite(c.lb) & (np.abs(c.lb) > 0))
        if len(ok):
            return n, int(ok[0])
    raise AssertionError("no finite lower bound in the photo")


def _mp_of(final):
    class MP:
        vars = {n: _v(x) for n, x in final.vars.items()}

    return MP


def test_lower_bounds_use_the_value_tolerance(ran):
    """Relative floors inherit the ~1e-10 level difference: 1e-9 passes, 1e-8 fails."""
    import copy

    from equilibria.family.prep_capture import load

    s, _, _, ref = ran
    final = load(ref / "final.pkl")
    name, i = _finite_lb_cell(final)
    assert s.LB_RTOL == s.RTOL == 1e-9
    near = copy.deepcopy(final)
    near.vars[name].lb = near.vars[name].lb.copy()
    near.vars[name].lb[i] *= 1 + 5e-10
    assert s.compare(_mp_of(final), near)["lb_diff"] == 0
    far = copy.deepcopy(final)
    far.vars[name].lb = far.vars[name].lb.copy()
    far.vars[name].lb[i] *= 1 + 1e-8
    out = s.compare(_mp_of(final), far)
    assert out["lb_diff"] == 1
    w = out["worst_lb_cells"][0]
    assert w["var"] == name and w["key"] == list(final.vars[name].keys[i])
    assert w["rel"] > 1e-9


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


def _fake_reference(tmp_path, closure):
    s = {"n": 2, "period": "check", "code": 1, "nit": 3}
    if closure is not None:
        s["closure"] = closure
    (tmp_path / "002_S_check.json").write_text(json.dumps(s))
    return tmp_path


def test_reference_closure_defaults_to_today_when_absent(tmp_path):
    mod = _script()
    ref = _fake_reference(tmp_path, None)
    assert mod.reference_closure(ref) == {
        "ifsub": 1,
        "savf": "capFix",
        "mode": "gtap",
        "linsolve": None,
    }


def test_ladder_refuses_a_reference_of_another_closure(tmp_path):
    mod = _script()
    ref = _fake_reference(
        tmp_path,
        {"ifsub": 0, "savf": "capFix", "mode": "gtap", "linsolve": "mumps"},
    )
    with pytest.raises(ValueError, match="closure"):
        mod.run("gtap7_3x3", ref)
    with pytest.raises(ValueError, match="closure"):
        mod.run("gtap7_3x3", ref, ifsub=1)


def test_ladder_altertax_raises_until_phase_3(tmp_path):
    mod = _script()
    with pytest.raises(NotImplementedError):
        mod.run("gtap7_3x3", tmp_path, mode="altertax")


def test_reference_dir_default_and_closure():
    mod = _script()
    assert mod.reference_dir("ds").name == "tol_today"
    assert mod.reference_dir("ds", ifsub=0).name == "gtap_ifsub0_capFix"


def test_ladder_refuses_a_reference_of_another_linsolve(tmp_path, monkeypatch):
    mod = _script()
    ref = _fake_reference(
        tmp_path, {"ifsub": 1, "savf": "capFix", "mode": "gtap", "linsolve": "direct"}
    )
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "mumps")
    with pytest.raises(ValueError, match="linsolve"):
        mod.run("gtap7_3x3", ref)


def test_unknown_reference_linsolve_is_allowed(tmp_path):
    mod = _script()
    have = mod.reference_closure(_fake_reference(tmp_path, None))
    assert have["linsolve"] is None
    mod.check_linsolve(tmp_path, have, "mumps")
    mod.check_linsolve(tmp_path, {"linsolve": "mumps"}, "mumps")
    with pytest.raises(ValueError, match="linsolve"):
        mod.check_linsolve(tmp_path, {"linsolve": "direct"}, "mumps")


def _fake_reference_top_linsolve(tmp_path, top, closure=None):
    s = {"n": 2, "period": "check", "code": 1, "nit": 3, "linsolve": top}
    if closure is not None:
        s["closure"] = closure
    (tmp_path / "002_S_check.json").write_text(json.dumps(s))
    return tmp_path


def test_old_reference_reads_the_top_level_linsolve(tmp_path):
    mod = _script()
    ref = _fake_reference_top_linsolve(tmp_path, "mumps")
    assert mod.reference_closure(ref)["linsolve"] == "mumps"


def test_ladder_refuses_an_old_reference_of_another_linsolve(tmp_path, monkeypatch):
    mod = _script()
    ref = _fake_reference_top_linsolve(tmp_path, "direct")
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "mumps")
    with pytest.raises(ValueError, match="linsolve"):
        mod.run("gtap7_3x3", ref)


def test_top_level_linsolve_wins_over_the_closure_one(tmp_path):
    mod = _script()
    ref = _fake_reference_top_linsolve(
        tmp_path,
        "direct",
        {"ifsub": 1, "savf": "capFix", "mode": "gtap", "linsolve": "mumps"},
    )
    have = mod.reference_closure(ref)
    with pytest.raises(ValueError, match="linsolve"):
        mod.check_linsolve(ref, have, "mumps")
