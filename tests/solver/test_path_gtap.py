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
    st = Staged(closure, "params", gtap_mode) if staged else None
    return SimpleNamespace(pyomo="pyomo-model", staged=st, last_solve=None)


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
