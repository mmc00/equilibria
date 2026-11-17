"""path(): PATH as a recipe solver for the GTAP period model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from equilibria.solver import path_gtap
from equilibria.templates.gtap.period_model import Staged


class _Fake:
    """Stands for ``path_capi``: records each call, returns ``code``."""

    def __init__(self, code: int = 1) -> None:
        self.code = code
        self.calls: list[dict[str, Any]] = []

    def _run_path_capi_nonlinear_full(self, m: Any, params: Any, **kw: Any) -> dict:
        self.calls.append({"m": m, "params": params, **kw})
        return {"termination_code": self.code, "residual": 1e-12}


def _model(gtap_mode: bool = True, staged: bool = True) -> Any:
    closure = SimpleNamespace(name="base")
    st = Staged(closure, "params", gtap_mode, "check") if staged else None
    m = SimpleNamespace(pyomo="pyomo-model", staged=st, last_solve=None)
    m.record_solve = lambda r: setattr(m, "last_solve", r)
    return m


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _Fake:
    f = _Fake()
    monkeypatch.setattr(path_gtap, "_load_path_capi", lambda: f)
    monkeypatch.delenv("PATH_CAPI_OPTIONS", raising=False)
    return f


def test_passes_the_staged_closure_and_params(fake: _Fake) -> None:
    m = _model()
    path_gtap.path()(m)
    call = fake.calls[0]
    assert call["m"] == "pyomo-model" and call["params"] == "params"
    assert call["closure_config"] is m.staged.closure
    assert call["gtap_mode"] is True and call["equation_scaling"] is True
    assert call["enforce_post_checks"] is False and call["strict_path_capi"] is False
    assert call["solution_hint"] is None
    assert m.last_solve == {"termination_code": 1, "residual": 1e-12}


def test_options_altertax_default(fake: _Fake) -> None:
    path_gtap.path()(_model(gtap_mode=False))
    assert fake.calls[0]["path_options"] == path_gtap.ALTERTAX_PATH_OPTIONS
    assert "crash_method none" in path_gtap.ALTERTAX_PATH_OPTIONS.splitlines()


def test_options_gtap_none(fake: _Fake) -> None:
    path_gtap.path()(_model(gtap_mode=True))
    assert fake.calls[0]["path_options"] is None


def test_options_user_env_wins(fake: _Fake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "major_iteration_limit 9")
    path_gtap.path()(_model(gtap_mode=False))
    assert fake.calls[0]["path_options"] == "major_iteration_limit 9"


def test_options_argument_wins(fake: _Fake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "major_iteration_limit 9")
    path_gtap.path(options="crash_method pd")(_model(gtap_mode=False))
    assert fake.calls[0]["path_options"] == "crash_method pd"


def test_status_mapping(fake: _Fake) -> None:
    assert path_gtap.path()(_model()) == "optimal"
    fake.code = 2
    assert path_gtap.path()(_model()) == "code 2"


def test_no_stage_raises(fake: _Fake) -> None:
    with pytest.raises(RuntimeError, match="no closure staged"):
        path_gtap.path()(_model(staged=False))


def test_env_untouched(fake: _Fake) -> None:
    import os

    path_gtap.path()(_model(gtap_mode=False))
    assert "PATH_CAPI_OPTIONS" not in os.environ


def test_inside_a_run_scope_the_solve_keeps_the_run_freeze(fake: _Fake) -> None:
    """GTAP.run_scope freezes the model once for the whole run; a solve inside it
    must neither collect nor unfreeze (``gc.unfreeze`` is global), or every gap
    between solves walks the whole graph again."""
    import gc

    from equilibria.solver.gc_tuning import _gc_family_scope

    seen: list[int] = []
    run = fake._run_path_capi_nonlinear_full

    def spy(m: Any, params: Any, **kw: Any) -> dict:
        seen.append(gc.get_freeze_count())
        return run(m, params, **kw)

    fake._run_path_capi_nonlinear_full = spy  # type: ignore[method-assign]
    with _gc_family_scope():
        frozen = gc.get_freeze_count()
        assert frozen > 0
        path_gtap.path()(_model())
        path_gtap.path()(_model())
        assert gc.get_freeze_count() >= frozen  # nothing was unfrozen in between
    assert gc.get_freeze_count() == 0
    assert all(n >= frozen for n in seen)


def test_outside_a_run_scope_the_solve_freezes_for_itself(fake: _Fake) -> None:
    import gc

    seen: list[int] = []
    run = fake._run_path_capi_nonlinear_full

    def spy(m: Any, params: Any, **kw: Any) -> dict:
        seen.append(gc.get_freeze_count())
        return run(m, params, **kw)

    fake._run_path_capi_nonlinear_full = spy  # type: ignore[method-assign]
    path_gtap.path()(_model())
    assert seen[0] > 0 and gc.get_freeze_count() == 0


def test_inside_a_run_scope_the_solve_freezes_what_the_steps_built(fake: _Fake) -> None:
    """The steps build long-lived objects after the run's freeze (the per-period
    index caches, the reference model); the solve freezes them too, so the
    full passes after it do not walk them (measured on 20x41: +3.5 s a gap)."""
    import gc

    from equilibria.solver.gc_tuning import _gc_family_scope

    with _gc_family_scope():
        before = gc.get_freeze_count()
        built_by_a_step = [
            [i] for i in range(1000)
        ]  # tracked, created after the freeze
        path_gtap.path()(_model())
        assert gc.get_freeze_count() >= before + len(built_by_a_step)
    assert gc.get_freeze_count() == 0
