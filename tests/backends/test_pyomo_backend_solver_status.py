"""PyomoBackend.get_solver_status no revienta con los resultados reales de un solver.

``results.solver.iterations`` no existe en los resultados de Ipopt ni de PATH via
AMPL (medido 2026-10-02 con Ipopt 3.14.19 y PATH 4.7.03): el acceso lanzaba
``AttributeError`` despues de cualquier solve. ``time`` si viene tras un solve, pero
no en un ``.sol`` leido aparte. Un campo sin valor vuelve como
``UndefinedData`` y ``str()`` lo convertia en "<undefined>".
"""

from __future__ import annotations

from pathlib import Path

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


# .sol reales (modelo x**3 == 8 desde x=10, escritos por pathampl 4.7.03 e ipopt
# 3.14.19 el 2026-10-02) leidos con el lector .sol de Pyomo: el caso que fallaba.
SOL_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pyomo_sol"


def _leer_sol(nombre: str) -> SolverResults:
    import pyomo.environ  # noqa: F401  (registra el lector .sol)
    from pyomo.opt import ReaderFactory, ResultsFormat

    return ReaderFactory(ResultsFormat.sol)(str(SOL_DIR / nombre))


def test_sol_real_de_path():
    info = _backend_con(_leer_sol("path_4.7.03.sol")).get_solver_status()
    assert info["iterations"] == 8
    assert info["message"].startswith("Path 4.7.03: Solution found.; 8 iterations")
    assert info["time"] is None  # el .sol no trae tiempo


def test_sol_real_de_ipopt():
    info = _backend_con(_leer_sol("ipopt_3.14.19.sol")).get_solver_status()
    assert info["iterations"] is None
    assert info["message"] == "Ipopt 3.14.19: Optimal Solution Found"
