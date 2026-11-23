# tests/templates/gtap/test_env_hygiene.py
#
# Contrato de libreria: `equilibria` NO deja modificado el entorno del proceso
# que la importa. Antes, un solve dejaba PATH_CAPI_OPTIONS puesta (y la rama de
# debug NLP-via-GAMS anteponia GAMS al PATH para siempre).
#
# Los de _env_override no resuelven. Los de PATH_CAPI_OPTIONS corren un settle
# 3x3 del core (un solve de PATH, segundos) con la llamada a PATH ensuciando el
# entorno: lo que se prueba es que el run lo deja como estaba.
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


def _settle_3x3() -> dict:
    """A core run with one PATH solve: the 3x3 settle (``gtap_static(settle_only)``:
    the base calibrated, the check solved) on the Pyomo backend."""
    from tests.family._data import load_params
    from tests.family.test_sp_arrays import _closure

    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    p = load_params("gtap7_3x3")
    return Exercise(GTAP(p, _closure()), gtap_static(settle_only=True)).run()


@pytest.fixture
def dirty_solve(monkeypatch):
    """The PATH call dirties PATH_CAPI_OPTIONS, then solves (``fail``: raises)."""
    capi = pytest.importorskip("path_capi_python")
    real = capi.solve_nonlinear_mcp
    state = {"fail": False}

    def dirty(*a, **k):
        os.environ["PATH_CAPI_OPTIONS"] = "* ensuciado por el solve\n"
        if state["fail"]:
            raise RuntimeError("solve fallido")
        return real(*a, **k)

    monkeypatch.setattr(capi, "solve_nonlinear_mcp", dirty)
    return state


def test_solve_does_not_leak_path_capi_options(monkeypatch, dirty_solve):
    """Un run del core restaura PATH_CAPI_OPTIONS: el valor del usuario vuelve, y
    si no habia ninguno no queda ninguno."""
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "major_iteration_limit 500")
    _settle_3x3()
    assert os.environ["PATH_CAPI_OPTIONS"] == "major_iteration_limit 500"

    monkeypatch.delenv("PATH_CAPI_OPTIONS", raising=False)
    _settle_3x3()
    assert "PATH_CAPI_OPTIONS" not in os.environ


def test_solve_restores_path_capi_options_on_exception(monkeypatch, dirty_solve):
    """El solve revienta: el core lo registra como un check no resuelto, y el
    entorno queda como estaba."""
    dirty_solve["fail"] = True
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "major_iteration_limit 500")

    res = _settle_3x3()
    assert res["check"]["code"] != "optimal"
    assert os.environ["PATH_CAPI_OPTIONS"] == "major_iteration_limit 500"
