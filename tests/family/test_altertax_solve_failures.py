"""A failed solve never passes silently: every Newton solve of an ALTERTAX run (settle,
check, shock, the decomp path's steps) is recorded with its code, iterations, null pivots
and GMIN retries; one failed solve makes the run FAIL in ``compare_all``, the measure
script's JSON and summary, and the ``run_altertax`` command's ``run.json``.

The failure is injected: the first Newton solve of the shock period reports code 2 (as
the 20x41 profile's stalled first ratio solve did), everything else runs as usual."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from tests.family._prep import ROOT, _newton_mumps

pytest.importorskip("pyomo")
DS = "gtap7_3x3"
REF = ROOT / "data/gtap7" / DS / "gempack_altertax-rungtap_tms-rate10_s8-16-32"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"
RECORD_KEYS = {"solve", "code", "nit", "residual", "null_pivots", "gmin_retries"}


def _script(name):
    path = ROOT / "scripts" / "family" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    root = tmp_path_factory.mktemp("cache")
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(root))
    yield root
    mp.undo()


def _inject_first_shock_failure(monkeypatch):
    from equilibria.family import driver, phases

    orig = driver._newton
    seen = {"shock": 0}

    def failing(ss, sq):
        res, resid, code = orig(ss, sq)
        if list(phases.CURRENT._prefix[:1]) == ["shock"]:
            seen["shock"] += 1
            if seen["shock"] == 1:
                return res, resid, 2
        return res, resid, code

    monkeypatch.setattr(driver, "_newton", failing)
    return seen


def _ratio_run(cache, decomp_steps=None):
    from equilibria.family.altertax.run import load_altertax_params, solve_altertax
    from equilibria.family.altertax.shocks import load_tms_shock

    d = ROOT / "data/gtap7" / DS
    return solve_altertax(
        d,
        prm=REF / "parameters.prm",
        shock=load_tms_shock(REF / "tms_rate10.shk"),
        params=load_altertax_params(d, REF / "parameters.prm"),
        trade_balance="ratio",
        cache_root=cache,
        decomp_steps=decomp_steps,
    )


def test_every_solve_is_recorded(cache):
    run = _ratio_run(cache)
    recs = run.solve_records()
    assert [r["solve"] for r in recs][:2] == ["settle_check", "check"]
    assert any(r["solve"].startswith("shock") for r in recs)
    for r in recs:
        assert set(r) >= RECORD_KEYS, r
        assert r["code"] == 1
    assert run.failed_solves() == []


def test_failed_shock_solve_fails_the_run(cache, monkeypatch):
    seen = _inject_first_shock_failure(monkeypatch)
    run = _ratio_run(cache)
    assert seen["shock"] >= 1
    assert run.codes["shock"] != 1  # the final code carries the failure
    failed = run.failed_solves()
    assert len(failed) == 1 and failed[0]["solve"].startswith("shock")
    assert failed[0]["code"] == 2


def test_compare_all_fails_on_a_failed_solve(cache, monkeypatch):
    from equilibria.family.altertax import gates

    _inject_first_shock_failure(monkeypatch)
    run = _ratio_run(cache)
    cmp = gates.compare_all(run, REF)
    assert cmp["solves_pass"] is False and cmp["pass"] is False
    assert any(r["code"] == 2 for r in cmp["solves"])


def test_measure_json_and_summary_fail_on_a_failed_solve(
    cache, monkeypatch, tmp_path, capsys
):
    _inject_first_shock_failure(monkeypatch)
    m = _script("altertax_rungtap_measure")
    monkeypatch.setattr("sys.argv", ["m", DS, "--out", str(tmp_path)])
    rc = m.main()
    out = json.loads((tmp_path / f"{DS}_ratio.json").read_text())
    printed = capsys.readouterr().out
    assert rc != 0
    assert out["pass"] is False
    assert any(r["code"] == 2 for r in out["solves"])
    assert "overall FAIL" in printed
    assert "solve shock" in printed and "code 2" in printed


def test_measure_passes_and_lists_solves_when_all_converge(
    cache, monkeypatch, tmp_path, capsys
):
    m = _script("altertax_rungtap_measure")
    monkeypatch.setattr("sys.argv", ["m", DS, "--out", str(tmp_path)])
    rc = m.main()
    out = json.loads((tmp_path / f"{DS}_ratio.json").read_text())
    printed = capsys.readouterr().out
    assert rc == 0 and out["pass"] is True
    assert all(r["code"] == 1 for r in out["solves"])
    steps = out["decomp_path"]["step_records"]
    assert steps and all({"null_pivots", "code"} <= set(s) for s in steps)
    assert "overall PASS" in printed


def test_run_altertax_reports_a_failed_solve(cache, monkeypatch, tmp_path, capsys):
    _inject_first_shock_failure(monkeypatch)
    cmd = _script("run_altertax")
    out = tmp_path / "out"
    rc = cmd.main([str(nus333_dir()), "--out", str(out), "--compare", str(FIXTURES)])
    info = json.loads((out / "run.json").read_text())
    assert rc == 3
    assert info["solved"] is False
    assert any(
        r["code"] == 2 and r["solve"].startswith("shock") for r in info["solves"]
    )
    assert not (out / "compare.json").exists()


def test_path_step_with_a_non_finite_step_stops_the_run(cache, monkeypatch):
    from equilibria.family.altertax import path

    orig = path._Linear.solve

    def bad(self, b):
        return np.full_like(orig(self, b), np.nan)

    monkeypatch.setattr(path._Linear, "solve", bad)
    with pytest.raises(RuntimeError, match="sub-step"):
        _ratio_run(cache, decomp_steps=2)
