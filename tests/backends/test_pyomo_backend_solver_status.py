"""PyomoBackend.get_solver_status no revienta con los resultados reales de un solver.

``results.solver.iterations`` no existe en los resultados de Ipopt ni de PATH via
AMPL (medido 2026-10-02 con Ipopt 3.14.19 y PATH 4.7.03): el acceso lanzaba
``AttributeError`` despues de cualquier solve. Un campo sin valor vuelve como
``UndefinedData`` y ``str()`` lo convertia en "<undefined>".
"""

from __future__ import annotations

from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from equilibria.backends import PyomoBackend


def _backend_con(results: SolverResults) -> PyomoBackend:
    backend = PyomoBackend()
    backend._solver_results = results
    return backend


def _resultados(**campos) -> SolverResults:
    r = SolverResults()
    r.solver.status = SolverStatus.ok
    r.solver.termination_condition = TerminationCondition.optimal
    for nombre, valor in campos.items():
        setattr(r.solver, nombre, valor)
    return r


def test_resultados_como_los_de_path():
    r = _resultados(
        time=0.02,
        message="Path 4.7.03\\x3a Solution found.; 50 iterations (0 for crash)",
    )
    info = _backend_con(r).get_solver_status()
    assert info["iterations"] == 50
    assert (
        info["message"] == "Path 4.7.03: Solution found.; 50 iterations (0 for crash)"
    )
    assert info["time"] == 0.02


def test_resultados_como_los_de_ipopt():
    r = _resultados(time=0.05, message="Ipopt 3.14.19\\x3a Optimal Solution Found")
    info = _backend_con(r).get_solver_status()
    assert info["iterations"] is None
    assert info["message"] == "Ipopt 3.14.19: Optimal Solution Found"


def test_sin_campos_no_hay_undefined():
    info = _backend_con(_resultados()).get_solver_status()
    assert info["message"] is None
    assert info["time"] is None
    assert info["iterations"] is None
