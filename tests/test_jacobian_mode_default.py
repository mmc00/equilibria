"""Default Jacobian mode when PyNumero's ASL library is not installed.

The default is ``asl``, which needs the compiled ``pynumero_ASL`` library. pip does
not install it, and ``pyomo download-extensions`` does not ship it for macOS arm64.
Without it, building the callbacks failed, the solve returned code 0 and the
README quickstart printed an EV of ~1e-6 instead of -0.108 (measured in a clean
venv, 2026-11-05). ``reverse_numeric`` gives the README's EV there.
"""

from __future__ import annotations

import pytest

from equilibria.solver import path_capi


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EQUILIBRIA_GTAP_JAC_MODE", raising=False)


def test_asl_when_available(monkeypatch: pytest.MonkeyPatch, no_env: None) -> None:
    monkeypatch.setattr(path_capi, "_asl_available", lambda: True)
    assert path_capi._default_jacobian_eval_mode() == "asl"


def test_falls_back_without_asl(
    monkeypatch: pytest.MonkeyPatch, no_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(path_capi, "_asl_available", lambda: False)
    with caplog.at_level("WARNING", logger=path_capi.logger.name):
        assert path_capi._default_jacobian_eval_mode() == "reverse_numeric"
    assert "pynumero_ASL" in caplog.text


def test_environment_variable_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_capi, "_asl_available", lambda: False)
    monkeypatch.setenv("EQUILIBRIA_GTAP_JAC_MODE", "asl")
    assert path_capi._default_jacobian_eval_mode() == "asl"
