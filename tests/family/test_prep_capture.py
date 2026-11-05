"""Two-photo capture of the Pyomo period preparation (opt-in, off by default)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = "EQUILIBRIA_FAMILY_PREP_CAPTURE"


def _newton_mumps() -> dict:
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


def _solve_3x3_ifsub1(monkeypatch):
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    return _solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True)


def test_capture_off_by_default(monkeypatch, tmp_path):
    from equilibria.family import prep_capture

    monkeypatch.delenv(ENV, raising=False)
    sentinel = object()  # not a model: an active capture would fail on it
    prep_capture.capture_state(sentinel, "check")
    prep_capture.capture_squared(sentinel, [], [], 0, 0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_capture_pairs_3x3(monkeypatch, tmp_path):
    from equilibria.family import prep_capture

    monkeypatch.delenv(ENV, raising=False)
    _, code_ref = _solve_3x3_ifsub1(monkeypatch)

    monkeypatch.setenv(ENV, str(tmp_path))
    prep_capture.reset()
    _, code = _solve_3x3_ifsub1(monkeypatch)
    assert code == code_ref == 1

    pairs = prep_capture.pairs(tmp_path)
    assert [b.period for _, b in pairs] == ["check", "check", "shock"]
    for a, b in pairs:
        assert a.period == b.period
        assert len(b.con_names) == len(b.var_names) == len(b.incidence) > 0
        assert b.n_deact == 0 and b.n_fixed == 0
        free = {
            f"{v}[{','.join(k)}]" if k else v
            for v, cells in b.vars.items()
            for k, fx in zip(cells.keys, cells.fixed, strict=True)
            if not fx
        }
        assert set(b.var_names) <= free
        # A holds every period; B only the active one.
        periods_a = {k[-1] for c in a.vars.values() for k in c.keys if k}
        assert {"base", "check", "shock"} <= periods_a
        assert all(k[-1] == b.period for c in b.vars.values() for k in c.keys if k)
        assert prep_capture.load_a2(tmp_path, a) is not None
