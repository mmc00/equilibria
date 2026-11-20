"""ALTERTAX on the core driver gives what it gave on the old family driver (PR 3d).

The oracle (``tests/family/golden/pr3d_altertax_main.json.gz``) was recorded on main
(987475bd) by ``scripts/dev/record_pr3d_altertax.py`` before ``solve_altertax`` moved;
each case here runs that same recorder on this checkout, in a child process with the
same environment, and must give the same codes, every level the GEMPACK comparison
reads and every decomposition header (the path's, for the ``decomp_steps`` cases),
within 1e-9 relative + 1e-12 absolute (a run is bit-identical on one machine; the
tolerance is for another CPU). The decomposition headers get a 1e-7 absolute floor:
they are sums of level differences that cancel, so another CPU moves them more (Linux
CI vs the Mac golden: levels within 1e-9, headers up to 8e-9 absolute on values up to
1.5); a real change moves them far more.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from tests._golden import mismatches

ROOT = Path(__file__).resolve().parents[2]
with gzip.open(
    ROOT / "tests" / "family" / "golden" / "pr3d_altertax_main.json.gz", "rt"
) as _f:
    GOLDEN = json.load(_f)
pytest.importorskip("mumps")


def _recorder():
    path = ROOT / "scripts" / "dev" / "record_pr3d_altertax.py"
    spec = importlib.util.spec_from_file_location("record_pr3d_altertax", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_dict(x):
    """Lists (header arrays) as ``{index: value}``, for ``mismatches``."""
    if isinstance(x, dict):
        return {k: _as_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return {i: _as_dict(v) for i, v in enumerate(x)}
    return x


@pytest.fixture(scope="module")
def recorder():
    from equilibria._local_refs import nus333_dir

    if not (nus333_dir() / "basedata.har").exists():
        pytest.skip("nus333 data missing")
    return _recorder()


def test_the_recorder_and_the_golden_have_the_same_cases(recorder) -> None:
    assert set(recorder.CASES) == set(GOLDEN)


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_altertax_gives_what_main_gave(case: str, recorder) -> None:
    got = recorder.run_case(case, ROOT)
    want = GOLDEN[case]
    assert got["codes"] == want["codes"]
    assert mismatches(got["levels"], want["levels"], rel=1e-9, abs_tol=1e-12) == []
    assert set(got) == set(want) == {"codes", "levels", "decomp"}
    got_d, want_d = _as_dict(got["decomp"]), _as_dict(want["decomp"])
    assert mismatches(got_d, want_d, rel=1e-9, abs_tol=1e-7) == []


@pytest.fixture
def newton_env(monkeypatch, tmp_path):
    from tests.family._prep import _newton_mumps

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path / "cache"))
    return tmp_path


def test_altertax_runs_no_old_family_driver(newton_env, monkeypatch) -> None:
    """Neither the solve nor the path decomposition reaches the old orchestration."""
    import equilibria.family.driver as old
    from equilibria._local_refs import nus333_dir
    from equilibria.family.altertax.run import solve_altertax

    def boom(*a, **k):
        raise AssertionError("the old family driver was reached")

    for name in (
        "solve_gtap_family",
        "_solve_family",
        "_solve_period",
        "_period_recipe",
    ):
        monkeypatch.setattr(old, name, boom)
    run = solve_altertax(nus333_dir(), cache_root=newton_env / "c", decomp_steps=2)
    assert run.codes == {"check": 1, "shock": 1}
    assert run.decomp_path is not None


def test_a_warm_sub_step_lands_where_a_cold_one_does(newton_env) -> None:
    """``solve_shock(start=...)`` warm-starts from a previous point: the same point
    as a solve from the check, in no more Newton iterations."""
    from equilibria._local_refs import nus333_dir
    from equilibria.family.altertax.path import shock_fraction
    from equilibria.family.altertax.run import ALTAX_SHOCK, solve_altertax

    run = solve_altertax(nus333_dir(), cache_root=newton_env / "c", decomp_steps=2)
    session = run.result.session
    half, _ = session.solve_shock(shock_fraction(ALTAX_SHOCK, 0.5))
    warm, rw = session.solve_shock(ALTAX_SHOCK, start=half)
    cold, rc = session.solve_shock(ALTAX_SHOCK)
    for name in ("pm", "xp", "regy", "ev"):
        a, b = warm.vars[name], cold.vars[name]
        assert (
            mismatches(
                dict(enumerate(a.value)),
                dict(enumerate(b.value)),
                rel=1e-9,
                abs_tol=1e-12,
            )
            == []
        )
    assert rw["nit"] <= rc["nit"]


@pytest.mark.parametrize(
    "status",
    ["error: RuntimeError: MUMPS factorization error: 1", "error: ValueError: boom"],
)
def test_a_raised_solve_is_never_read_as_a_code(status: str) -> None:
    """The core turns a solver exception into an ``"error: ..."`` status; read as an
    int code it could come out 1 (solved) from the message's last word."""
    from equilibria.family.altertax.run import _int_code
    from equilibria.family.shock_session import _code

    for read in (_int_code, _code):
        with pytest.raises(RuntimeError, match="raised"):
            read(status)


def test_status_codes() -> None:
    from equilibria.family.altertax.run import _int_code

    assert [_int_code(s) for s in ("optimal", "code 2", "code -9", "not solved")] == [
        1,
        2,
        -9,
        2,
    ]


def test_a_raising_shock_solve_stops_the_run(newton_env, monkeypatch) -> None:
    """As on main, an exception from the shock's Newton solve reaches the caller; it
    does not come back as a code (here one that would read as solved)."""
    from equilibria._local_refs import nus333_dir
    from equilibria.family import period_newton, phases
    from equilibria.family.altertax.run import solve_altertax

    orig = period_newton._newton

    def raising(*args, **kw):
        if list(phases.CURRENT._prefix[:1]) == ["shock"]:
            raise RuntimeError("MUMPS factorization error: 1")
        return orig(*args, **kw)

    monkeypatch.setattr(period_newton, "_newton", raising)
    with pytest.raises(RuntimeError, match="MUMPS factorization error: 1"):
        solve_altertax(nus333_dir(), cache_root=newton_env / "c")
