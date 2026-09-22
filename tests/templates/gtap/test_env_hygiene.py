# tests/templates/gtap/test_env_hygiene.py
#
# Contrato de libreria: `equilibria` NO deja modificado el entorno del proceso
# que la importa. Antes, un solve dejaba PATH_CAPI_OPTIONS puesta (y la rama de
# debug NLP-via-GAMS anteponia GAMS al PATH para siempre).
#
# Los tests de aqui NO resuelven: verifican el contrato del context manager y la
# restauracion, asi que corren en segundos y sin PATH C-API.
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402


def test_env_override_restores_previous_value():
    from equilibria.solver.path_capi import _env_override

    os.environ["EQ_TEST_VAR"] = "original"
    try:
        with _env_override(EQ_TEST_VAR="temporal"):
            assert os.environ["EQ_TEST_VAR"] == "temporal"
        assert os.environ["EQ_TEST_VAR"] == "original"
    finally:
        os.environ.pop("EQ_TEST_VAR", None)


def test_env_override_removes_var_that_did_not_exist():
    from equilibria.solver.path_capi import _env_override

    os.environ.pop("EQ_TEST_ABSENT", None)
    with _env_override(EQ_TEST_ABSENT="x"):
        assert os.environ["EQ_TEST_ABSENT"] == "x"
    assert "EQ_TEST_ABSENT" not in os.environ


def test_env_override_restores_on_exception():
    """El punto del context manager: restaurar tambien cuando el bloque revienta."""
    from equilibria.solver.path_capi import _env_override

    os.environ["EQ_TEST_VAR"] = "original"
    try:
        with pytest.raises(RuntimeError), _env_override(EQ_TEST_VAR="temporal"):
            raise RuntimeError("boom")
        assert os.environ["EQ_TEST_VAR"] == "original"
    finally:
        os.environ.pop("EQ_TEST_VAR", None)


def test_solve_does_not_leak_path_capi_options(monkeypatch):
    """El wrapper publico restaura PATH_CAPI_OPTIONS aunque el solve falle.

    Se mockea el solve: lo que se prueba es la higiene del entorno, no el
    modelo — asi el test no necesita PATH C-API ni datasets.
    """
    from equilibria.templates.gtap import gtap_multiperiod_driver as driver

    def _fake(*a, **k):
        os.environ["PATH_CAPI_OPTIONS"] = "* ensuciado por el solve\n"
        return {"base": {"code": 1}}

    monkeypatch.setattr(driver, "_solve_multiperiod_inner", _fake)

    monkeypatch.setenv("PATH_CAPI_OPTIONS", "del usuario")
    driver.solve_multiperiod(None, None, None)
    assert os.environ["PATH_CAPI_OPTIONS"] == "del usuario"

    monkeypatch.delenv("PATH_CAPI_OPTIONS", raising=False)
    driver.solve_multiperiod(None, None, None)
    assert "PATH_CAPI_OPTIONS" not in os.environ


def test_solve_restores_path_capi_options_on_exception(monkeypatch):
    from equilibria.templates.gtap import gtap_multiperiod_driver as driver

    def _boom(*a, **k):
        os.environ["PATH_CAPI_OPTIONS"] = "* ensuciado antes de reventar\n"
        raise RuntimeError("solve fallido")

    monkeypatch.setattr(driver, "_solve_multiperiod_inner", _boom)
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "del usuario")

    with pytest.raises(RuntimeError):
        driver.solve_multiperiod(None, None, None)
    assert os.environ["PATH_CAPI_OPTIONS"] == "del usuario"
