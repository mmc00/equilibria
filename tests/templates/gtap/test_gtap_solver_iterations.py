"""GTAPSolver reporta las iteraciones que dice el solver, o None si no las dice.

Antes leia ``results.solver.get("iterations", 0)``. ``results.solver`` es un
``ListContainer`` de Pyomo: ``.get`` no delega al primer elemento y devuelve el
default, asi que con PATH via AMPL (``pathampl``) el resultado decia
"Converged in 0 iterations" aunque PATH hubiera hecho 50 iteraciones mayores y
dos restarts. Medido 2026-10-02 sobre el 9x10: PATH 4.7.03 dice "50 iterations" y
PATH 5.0.05, con el mismo modelo, "68 iterations (0 for crash); 1201 pivots."; los
dos se perdian.
"""

from __future__ import annotations

from pyomo.environ import ConcreteModel, Var
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from equilibria.templates.gtap.gtap_solver import GTAPSolver


def _solver_sin_init(model) -> GTAPSolver:
    """Un GTAPSolver sin __init__: solo hace falta ``model`` para procesar resultados."""
    s = GTAPSolver.__new__(GTAPSolver)
    s.model = model
    return s


def _resultados(message: str | None) -> SolverResults:
    r = SolverResults()
    r.solver.status = SolverStatus.ok
    r.solver.termination_condition = TerminationCondition.optimal
    if message is not None:
        r.solver.message = message
    return r


def _modelo() -> ConcreteModel:
    m = ConcreteModel()
    m.walras = Var(initialize=3.5e-9)
    return m


def test_iteraciones_del_mensaje_de_path():
    msg = (
        "Path 5.0.05\\x3a Solution found.; 68 iterations (0 for crash); "
        "1201 pivots.; 325 function, 71 gradient evaluations."
    )
    res = _solver_sin_init(_modelo())._process_results(_resultados(msg), solve_time=1.0)
    assert res.iterations == 68
    assert "68 iterations" in res.message
    # el mensaje del solver ya no se pierde, y sin el escape "\\x3a" de Pyomo
    assert "Path 5.0.05: Solution found." in res.message


def test_sin_conteo_es_none_no_cero():
    res = _solver_sin_init(_modelo())._process_results(
        _resultados("Ipopt 3.14.19\\x3a Optimal Solution Found"), solve_time=1.0
    )
    assert res.iterations is None
    assert "0 iterations" not in res.message


def test_sin_mensaje_es_none():
    res = _solver_sin_init(_modelo())._process_results(
        _resultados(None), solve_time=1.0
    )
    assert res.iterations is None
    assert res.success
    # Sin mensaje, Pyomo devuelve UndefinedData (verdadero): no debe colarse
    # como "; <undefined>" en el mensaje del resultado.
    assert "undefined" not in res.message
    assert res.message == "Converged, Walras = 3.50e-09"


def test_iterations_definido_por_el_solver_tiene_prioridad():
    r = _resultados("Path 5.0.05\\x3a Solution found.; 68 iterations (0 for crash)")
    r.solver.iterations = 12
    res = _solver_sin_init(_modelo())._process_results(r, solve_time=1.0)
    assert res.iterations == 12
