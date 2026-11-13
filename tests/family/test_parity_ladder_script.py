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


@pytest.fixture(autouse=True)
def _keep_environ():
    # run() setdefaults the NEWTON_MUMPS settings before it may raise; without
    # this they outlive the test and switch later GTAP solves to Newton (#245).
    old = {k: os.environ.get(k) for k in _newton_mumps()}
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


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


# ── polished-root gate (user answer 8) ──────────────────────────────────────


def _weak(eps=1e-6):
    """F = A x - b, root (1, 1), A nearly singular along (1, -1)."""
    import scipy.sparse as sp

    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0 + eps]]))
    b = A @ np.ones(2)
    return (lambda x: A @ x - b), (lambda x: A)


def _polish(F, J, ours, ref, reltol=1e-6):
    s = _script()
    big = np.full(len(ours), 1e20)
    return s.polish_call(
        F,
        J,
        np.ones(len(ours)),
        np.asarray(ours, float),
        np.asarray(ref, float),
        -big,
        big,
        reltol=reltol,
        ftol=1e-7,
        linsolve="direct",
    )


def test_weak_direction_raw_differ_polished_agree_passes():
    F, J = _weak()
    v = np.array([1.0, -1.0])
    out = _polish(F, J, np.ones(2) + 1e-6 * v, np.ones(2) - 1e-6 * v)
    assert out["raw_max_rel"] > 1e-9  # the end points differ along the weak direction
    assert out["residual_ok_ours"] and out["residual_ok_reference"]
    assert out["polished_max_rel"] <= 1e-9
    assert out["passed"] is True


def test_a_raw_end_point_off_the_residual_test_fails():
    F, J = _weak()
    out = _polish(F, J, np.ones(2), np.array([1.0 + 1e-3, 1.0]))
    assert out["residual_ok_ours"] and not out["residual_ok_reference"]
    assert out["polished_max_rel"] <= 1e-9  # one step would reach the root ...
    assert out["passed"] is False  # ... but the reference stopped too early


def test_polished_points_that_differ_fail():
    import scipy.sparse as sp

    def F(x):  # two roots, x = 1 and x = -1
        return np.array([x[0] ** 2 - 1.0])

    def J(x):
        return sp.csr_matrix(np.array([[2.0 * x[0]]]))

    out = _polish(F, J, [1.0 + 1e-8], [-1.0 - 1e-8])
    assert out["residual_ok_ours"] and out["residual_ok_reference"]
    assert out["polished_max_rel"] > 1e-9
    assert out["passed"] is False


def test_polished_gate_rules_the_report():
    s = _script()
    ok = {"passed": True}
    both = {"check": ok, "shock": ok}
    base = {
        "raw_over": 3,
        "polished_over": 0,
        "fixed_diff": 0,
        "ub_diff": 0,
        "lb_diff": 0,
    }
    assert s.polished_gate(**base, structure=[], codes_ok=True, calls=both)
    assert not s.polished_gate(
        **{**base, "polished_over": 1}, structure=[], codes_ok=True, calls=both
    )
    assert not s.polished_gate(
        **base, structure=[], codes_ok=True, calls={**both, "shock": {"passed": False}}
    )
    assert not s.polished_gate(**base, structure=[], codes_ok=True, calls={})
    assert not s.polished_gate(**base, structure=[], codes_ok=False, calls=both)


def test_polished_gate_needs_exactly_check_and_shock():
    s = _script()
    ok = {"passed": True}
    base = {
        "raw_over": 0,
        "polished_over": 0,
        "fixed_diff": 0,
        "ub_diff": 0,
        "lb_diff": 0,
    }
    kw = {**base, "structure": [], "codes_ok": True}
    assert not s.polished_gate(**kw, calls={"check": ok})
    assert not s.polished_gate(**kw, calls={"check": ok, "shock": ok, "call3": ok})


def test_report_has_raw_and_polished_fields(ran):
    _, _, out, _ = ran
    for key in (
        "raw_max_rel",
        "raw_worst_cells",
        "polished_max_rel",
        "polished_worst_cells",
        "periods",
    ):
        assert key in out
    assert out["max_rel"] == out["raw_max_rel"]  # max_rel keeps its raw meaning
    for period in ("check", "shock"):
        p = out["periods"][period]
        for key in ("raw_max_rel", "polished_max_rel"):
            assert key in p
        assert p["residual_ours"] < 1e-5 and p["residual_reference"] < 1e-5
    assert "raw_max_rel" in out["periods"]["base"]  # not solved: compared raw


def test_polished_photos_take_the_post_solve_cells_of_both_sides():
    """A derived cell (``pfy``, rewritten by the post-solve) is compared at the value
    each side's polished point gives it; a cell outside the call stays raw."""
    import types

    s = _script()

    def var(vals):
        return _v(
            types.SimpleNamespace(
                keys=[("r", "shock"), ("q", "shock")],
                value=np.array(vals, dtype=float),
                fixed=np.array([True, True]),
                lb=np.full(2, -np.inf),
                ub=np.full(2, np.inf),
            )
        )

    mp = types.SimpleNamespace(vars={"pfy": var([1.0 + 1e-8, 5.0])})
    final = types.SimpleNamespace(vars={"pfy": var([1.0, 5.0])})
    assert s.compare(mp, final)["over_1e9"] == 1  # raw: the derived cell differs
    rec = types.SimpleNamespace(
        calls={"shock": {}},
        cells={"shock": [("pfy", [("r", "shock")], np.array([2.0]), np.array([2.0]))]},
    )
    ours, theirs = s.polished_photos(mp, final, rec)
    out = s.compare(ours, theirs)
    assert out["over_1e9"] == 0 and out["max_rel"] == 0.0
    assert theirs.vars["pfy"].value[1] == 5.0  # untouched cell keeps its raw value
    assert final.vars["pfy"].value[0] == 1.0  # the photos themselves are not modified


# ── fix round 1: polish steps, consistency, residual cross-check, post-solve ─


def test_polish_reports_the_step_of_each_side():
    F, J = _weak()
    v = np.array([1.0, -1.0])
    s = _script()
    big = np.full(2, 1e20)
    out = s.polish_call(
        F,
        J,
        np.ones(2),
        np.ones(2) + 1e-6 * v,
        np.ones(2) - 1e-6 * v,
        -big,
        big,
        reltol=1e-6,
        ftol=1e-7,
        linsolve="direct",
        names=["a[r,shock]", "b[r,shock]"],
    )
    assert out["step_rel_ours"] == pytest.approx(1e-6, rel=1e-3)
    assert out["step_rel_reference"] == pytest.approx(1e-6, rel=1e-3)
    w = out["step_worst_ours"][0]
    assert w["col"] in ("a[r,shock]", "b[r,shock]") and w["rel"] > 0


def test_raw_gap_explained_by_the_steps_has_no_violations():
    F, J = _weak()
    v = np.array([1.0, -1.0])
    out = _polish(F, J, np.ones(2) + 1e-6 * v, np.ones(2) - 1e-6 * v)
    assert out["consistency_violations"] == 0


def test_raw_gap_not_explained_by_the_steps_is_a_violation():
    import scipy.sparse as sp

    def F(x):
        return np.array([x[0] ** 2 - 1.0])

    def J(x):
        return sp.csr_matrix(np.array([[2.0 * x[0]]]))

    out = _polish(F, J, [1.0 + 1e-8], [-1.0 - 1e-8])
    assert out["consistency_violations"] == 1
    assert out["passed"] is False


def test_residual_cross_check_tolerance():
    s = _script()
    assert s.crosscheck_residual(3.897036984779e-8, 3.897036987554614e-08)
    assert s.crosscheck_residual(1.41016e-12, 1.41015e-12)  # noise near zero: atol
    assert not s.crosscheck_residual(3.9e-8, 3.0e-8)
    assert not s.crosscheck_residual(2e-6, 1e-6)


def test_cross_check_against_the_reference_s_file(tmp_path):
    s = _script()
    (tmp_path / "005_S_shock.json").write_text(
        json.dumps(
            {"trace_f_inf": [0.6, 3.897e-8], "trace_f_scaled_inf": [0.18, 3.896e-8]}
        )
    )
    good = {
        "passed": True,
        "residual_reference": 3.897e-8,
        "residual_scaled_reference": 3.896e-8,
    }
    out = s.cross_check_calls({"shock": dict(good)}, tmp_path)
    assert out["shock"]["crosscheck_ok"] is True and out["shock"]["passed"] is True
    assert out["shock"]["pyomo_residual"] == 3.897e-8
    bad = {**good, "residual_reference": 5e-8}
    out = s.cross_check_calls({"shock": bad}, tmp_path)
    assert out["shock"]["crosscheck_ok"] is False and out["shock"]["passed"] is False
    out = s.cross_check_calls({"check": dict(good)}, tmp_path)  # no 004 S file
    assert out["check"]["passed"] is False and "error" in out["check"]


def test_singular_jacobian_is_recorded_not_raised():
    import scipy.sparse as sp

    s = _script()
    big = np.full(2, 1e20)
    out = s.polish_call(
        lambda x: np.array([x[0] + x[1] - 2.0, x[0] + x[1] - 2.0]),
        lambda x: sp.csr_matrix(np.ones((2, 2))),
        np.ones(2),
        np.ones(2),
        np.ones(2),
        -big,
        big,
        reltol=1e-6,
        ftol=1e-7,
        linsolve="direct",
    )
    assert out["passed"] is False and out["error"]


def test_nan_residual_is_recorded_not_raised():
    F, J = _weak()
    out = _polish(lambda x: np.array([np.nan, 0.0]), J, np.ones(2), np.ones(2))
    assert out["passed"] is False and "error" in out


class _FV:
    def __init__(self, keys, value):
        self.keys = keys
        self.value = np.array(value, dtype=float)
        self.fixed = np.zeros(len(keys), dtype=bool)
        self.lb = np.full(len(keys), -np.inf)
        self.ub = np.full(len(keys), np.inf)
        self.period = np.array([k[-1] for k in keys], dtype=object)

    def pos(self, k):
        return self.keys.index(k)


class _FMP:
    def __init__(self, vars_):
        self.vars = vars_

    def copy(self):
        return _FMP(
            {n: _FV(list(v.keys), v.value.copy()) for n, v in self.vars.items()}
        )


def _fake_post(mp, period, *, x, sq=None, **kw):
    mp.vars["x"].value[0] = x[0]  # the column
    mp.vars["d"].value[0] = 2.0 * x[0]  # derived from it
    return mp


def _post_recorder(final_d):
    s = _script()
    final = _FMP(
        {"x": _FV([("r", "shock")], [1.0]), "d": _FV([("r", "shock")], [final_d])}
    )
    rec = s.PolishRecorder(final, "direct")
    rec.calls["shock"] = {
        "passed": True,
        "x_ours_polished": np.array([1.5]),
        "x_ref_polished": np.array([1.5]),
        "x_ref_raw": np.array([1.0]),
    }
    rec.columns["shock"] = ["x[r,shock]"]
    rec._pending = "shock"
    mp = _FMP({"x": _FV([("r", "shock")], [0.0]), "d": _FV([("r", "shock")], [0.0])})
    out = rec.wrap_post_solve(_fake_post)(mp, "shock", x=np.array([1.0]), sq=object())
    return rec, out


def test_wrap_post_solve_reproduces_the_reference_derived_cells():
    rec, out = _post_recorder(final_d=2.0)
    info = rec.calls["shock"]
    assert out.vars["d"].value[0] == 2.0  # the real post-solve ran on ``mp``
    assert info["post_raw_max_rel"] == 0.0 and info["post_raw_violations"] == 0
    assert info["passed"] is True
    cells = {r[0]: (r[2][0], r[3][0]) for r in rec.cells["shock"]}
    assert cells["d"] == (3.0, 3.0)  # both sides from their polished x
    assert info["n_polished_columns"] == 1 and info["n_polished_cells"] == 2


def test_wrap_post_solve_flags_a_derived_cell_off_the_reference():
    rec, _ = _post_recorder(final_d=2.0 + 1e-6)
    info = rec.calls["shock"]
    assert info["post_raw_violations"] == 1 and info["passed"] is False


def test_calls_are_tagged_by_the_driver_period_tag():
    import types

    s = _script()
    rec = s.PolishRecorder(_FMP({}), "direct")
    seen = []

    def solve_period(run, mp, period, **kw):
        seen.append(rec._tag)
        return mp, {}

    wrapped = rec.wrap_solve_period(solve_period)
    wrapped(None, None, "check", tag="settle_check")
    wrapped(None, None, "check")
    wrapped(None, None, "shock")
    assert seen == ["settle_check", "check", "shock"] and rec._tag is None
    del types


def test_missing_reference_column_is_recorded_not_raised():
    import types

    s = _script()
    rec = s.PolishRecorder(_FMP({"x": _FV([("r", "shock")], [1.0])}), "direct")

    class SS:
        def F(self, z):
            return np.zeros(2)

        def J(self, z):
            return np.eye(2)

        def fscale(self):
            return np.ones(2)

    sq = types.SimpleNamespace(
        col_names=["x[r,shock]", "gone[r,shock]"],
        cells={
            "x[r,shock]": (0, 0, -np.inf, np.inf),
            "gone[r,shock]": (0, 0, -np.inf, np.inf),
        },
    )
    res = types.SimpleNamespace(x=np.ones(2))
    rec._tag = "shock"
    rec.wrap(lambda ss, sq: (res, 0.0, 1))(SS(), sq)
    assert (
        rec.calls["shock"]["passed"] is False and "gone" in rec.calls["shock"]["error"]
    )


def test_report_without_a_recorder_fails_the_gate(ran):
    s, _, _, ref = ran
    import types

    from equilibria.family.prep_capture import load

    final = load(ref / "final.pkl")
    res = types.SimpleNamespace(
        mp=_mp_of(final),
        results={
            "base": {"code": 1},
            "check": {"code": 1, "nit": 2},
            "shock": {"code": 1, "nit": 3},
        },
        settle={"check": {"code": 1, "nit": 2}},
    )
    out = s.report(res, 1, final, ref, "gtap7_3x3", None)
    assert out["gate_passed"] is False


def test_report_json_has_the_fix_round_fields(ran):
    _, _, out, _ = ran
    for p in ("check", "shock"):
        c = out["periods"][p]
        assert c["polished"] is True
        for k in (
            "step_rel_ours",
            "step_rel_reference",
            "consistency_violations",
            "crosscheck_ok",
            "pyomo_residual",
            "post_raw_max_rel",
            "n_polished_columns",
            "n_polished_cells",
        ):
            assert k in c, k
    assert out["periods"]["settle_check"]["polished"] is False
    assert "polished_max_rel_by_component" in out
    assert out["timing"]["polish_s"] >= 0 and "family_wall_s" in out


def test_report_after_a_failed_polish_writes_the_json(ran, tmp_path):
    """A call recorded ``passed: False, error`` has no polished points: the report
    keeps its raw cells, fails the gate and the JSON is still written."""
    import types

    from equilibria.family.prep_capture import load

    s, _, _, ref = ran
    final = load(ref / "final.pkl")
    res = types.SimpleNamespace(
        mp=_mp_of(final),
        results={
            "base": {"code": 1},
            "check": {"code": 1, "nit": 2},
            "shock": {"code": 1, "nit": 3},
        },
        settle={"check": {"code": 1, "nit": 2}},
    )
    rec = s.PolishRecorder(final, "direct")
    for tag in ("check", "shock"):
        rec.calls[tag] = {"passed": False, "error": "RuntimeError: singular"}
        rec.columns[tag] = []
    out = s.report(res, 1, final, ref, "gtap7_3x3", rec)
    f = tmp_path / "r.json"
    f.write_text(json.dumps(out, indent=1, default=str))
    back = json.loads(f.read_text())
    assert back["gate_passed"] is False
    assert back["periods"]["shock"]["error"] == "RuntimeError: singular"
    assert back["polished_max_rel"] == back["raw_max_rel"]  # nothing was polished
