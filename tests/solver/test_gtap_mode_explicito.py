# tests/solver/test_gtap_mode_explicito.py
#
# `gtap_mode` gobierna `protect_xseq` y el emparejamiento MCP HARD de
# `eq_xseq`, la fila de balance fisico que GAMS declara free-row. Apagarlo por
# accidente NO falla de forma visible: el sistema queda sobredeterminado,
# `deactivate_zero_unique_var_eqs` desactiva una ecuacion real para cuadrarlo y
# el solve aterriza en otra raiz.
#
# Por eso el modo se PASA como argumento. El marcador `model._gtap_mode` sigue
# valiendo de respaldo para los llamadores que aun no pueden pasarlo, pero ya
# no es el unico canal: con solo el `getattr(..., False)`, OLVIDAR la escritura
# era indistinguible de declarar altertax.
import ast
import inspect
import pathlib

from equilibria.solver.path_capi import (
    _resolver_gtap_mode,
    _run_path_capi_nonlinear_full,
)


class _ModeloFalso:
    """Un modelo sin marcador, como el que arma `parity_adapter`.

    El atributo se DECLARA aqui — `ty` marcaba como error colgarselo desde
    fuera, que es exactamente el antipatron que este modulo viene a corregir.
    """

    _gtap_mode: bool


def test_el_argumento_explicito_manda_sobre_el_marcador():
    m = _ModeloFalso()
    m._gtap_mode = False
    assert _resolver_gtap_mode(m, True) is True, "el argumento debe ganar"

    m._gtap_mode = True
    assert _resolver_gtap_mode(m, False) is False, "tambien cuando apaga"


def test_sin_argumento_cae_al_marcador():
    m = _ModeloFalso()
    m._gtap_mode = True
    assert _resolver_gtap_mode(m, None) is True

    m._gtap_mode = False
    assert _resolver_gtap_mode(m, None) is False


def test_sin_argumento_ni_marcador_es_altertax():
    """El default historico se conserva: quien no dice nada, no es puro-GTAP."""
    assert _resolver_gtap_mode(_ModeloFalso(), None) is False


def test_el_solver_acepta_el_modo_por_firma():
    p = inspect.signature(_run_path_capi_nonlinear_full).parameters
    assert "gtap_mode" in p, "el modo tiene que poder pasarse, no solo colgarse"
    assert p["gtap_mode"].default is None, (
        "centinela None: distingue 'no lo dije' de 'lo dije False'"
    )


def test_el_driver_pasa_el_modo_en_TODAS_sus_llamadas():
    """El invariante que sustituye al marcador.

    No enumera lineas: recorre el AST y exige que cada llamada del driver al
    solver lleve `gtap_mode=`. Si alguien añade una quinta llamada y se olvida,
    esto salta — que es justo el fallo que el `getattr` con default hacia mudo.
    """
    raiz = pathlib.Path(__file__).resolve().parents[2]
    f = raiz / "src/equilibria/templates/gtap/gtap_multiperiod_driver.py"
    arbol = ast.parse(f.read_text(encoding="utf-8"))

    sin_modo = []
    for n in ast.walk(arbol):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        nombre = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if nombre != "_run_path_capi_nonlinear_full":
            continue
        if not any(k.arg == "gtap_mode" for k in n.keywords):
            sin_modo.append(n.lineno)

    assert not sin_modo, (
        f"llamadas al solver sin `gtap_mode=` en {f.name}, lineas {sin_modo}: "
        "el modo decide el emparejamiento de eq_xseq, pasalo explicito"
    )


def test_hay_al_menos_las_cuatro_llamadas_conocidas():
    """Contraparte del anterior: que no pase por vacio si desaparecen.

    Si el driver deja de llamar al solver, el test de arriba pasaria trivialmente.
    """
    raiz = pathlib.Path(__file__).resolve().parents[2]
    f = raiz / "src/equilibria/templates/gtap/gtap_multiperiod_driver.py"
    arbol = ast.parse(f.read_text(encoding="utf-8"))
    n_llamadas = sum(
        1
        for n in ast.walk(arbol)
        if isinstance(n, ast.Call)
        and (
            n.func.attr
            if isinstance(n.func, ast.Attribute)
            else getattr(n.func, "id", "")
        )
        == "_run_path_capi_nonlinear_full"
    )
    assert n_llamadas >= 4, f"esperadas >=4 llamadas al solver, hay {n_llamadas}"
