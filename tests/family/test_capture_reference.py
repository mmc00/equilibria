"""Reference capture of the Pyomo solve (P, S, final, settled): opt-in, inert by default."""

import json

import numpy as np
import pytest

from tests.family.test_prep_capture import ENV, _newton_mumps

pytest.importorskip("pyomo")


def _solve(monkeypatch):
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    # The settle must run (a cached seed would skip it and its hook).
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    return _solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True)


def _cells(m):
    from pyomo.environ import Var

    out = {}
    for comp in m.component_objects(Var, descend_into=True):
        for idx in comp:
            vd = comp[idx]
            out[(comp.name, idx)] = (vd.value, bool(vd.fixed), vd.lb, vd.ub)
    return out


def _bits(cells):
    return {
        k: (None if v[0] is None else np.float64(v[0]).tobytes(), *v[1:])
        for k, v in cells.items()
    }


@pytest.mark.slow
def test_hooks_inert_when_env_unset_and_do_not_move_the_solution(monkeypatch, tmp_path):
    from equilibria.family import prep_capture

    # 1. hooks removed altogether
    monkeypatch.delenv(ENV, raising=False)
    with monkeypatch.context() as mp:
        for name in ("capture_solve", "capture_final", "capture_settled"):
            mp.setattr(prep_capture, name, lambda *a, **k: None)
        m0, code0 = _solve(monkeypatch)
        bare = _bits(_cells(m0))

    # 2. hooks present, env unset: nothing may reach the disk
    written = []
    with monkeypatch.context() as mp:
        mp.setattr(prep_capture, "_write", lambda *a, **k: written.append(a))
        mp.setattr(prep_capture, "_write_named", lambda *a, **k: written.append(a))
        m1, code1 = _solve(monkeypatch)
    assert written == []
    assert code1 == code0 == 1
    assert _bits(_cells(m1)) == bare

    # 3. capture on: files appear and the solution is still bitwise the same
    monkeypatch.setenv(ENV, str(tmp_path))
    prep_capture.reset()
    m2, code2 = _solve(monkeypatch)
    assert code2 == 1
    assert _bits(_cells(m2)) == bare


@pytest.mark.slow
def test_reference_files_3x3(monkeypatch, tmp_path):
    from equilibria.family import prep_capture

    monkeypatch.setenv(ENV, str(tmp_path))
    prep_capture.reset()
    m, code = _solve(monkeypatch)
    assert code == 1

    s_files = sorted(tmp_path.glob("*_S_*.json"))
    assert [json.loads(f.read_text())["period"] for f in s_files] == [
        "check",
        "check",
        "shock",
    ]
    for f in s_files:
        s = json.loads(f.read_text())
        assert s["code"] == 1 and s["nit"] >= 1 and s["nfact"] >= 1
        assert len(s["trace_f_inf"]) == len(s["trace_f_scaled_inf"]) >= 1
        assert s["trace_f_inf"][-1] < 1e-5 and s["residual"] < 1e-5
        assert s["tr_env"]["EQUILIBRIA_GTAP_TR_LINSOLVE"] == "direct"
        assert s["linsolve"] == "direct"
        # same call number and period as the B photo of the same solve
        b = prep_capture.load(tmp_path / f"{s['n']:03d}_B_{s['period']}.pkl")
        p = prep_capture.load(tmp_path / f"{s['n']:03d}_P_{s['period']}.pkl")
        assert p.vars.keys() == b.vars.keys()
        assert all(k[-1] == s["period"] for c in p.vars.values() for k in c.keys if k)

    # final.pkl equals the model after the whole run
    final = prep_capture.load(tmp_path / "final.pkl")
    cells = _cells(m)
    n = 0
    for name, vc in final.vars.items():
        for k, v, fx, lb, ub in zip(
            vc.keys, vc.value, vc.fixed, vc.lb, vc.ub, strict=True
        ):
            idx = k if len(k) != 1 else k[0]
            want = cells[(name, idx if idx != () else None)]
            got = (None if np.isnan(v) else v, bool(fx))
            assert (want[0], want[1]) == got
            n += 1
    assert n == len(cells)
    assert final.rows

    # settled.pkl: the check slice calibrate_base returned
    import pickle

    settled = pickle.loads((tmp_path / "settled.pkl").read_bytes())
    assert settled and all(isinstance(v, dict) and v for v in settled.values())
