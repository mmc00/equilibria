"""pytest configuration and shared fixtures.

Incluye los guards de solver: varios tests resuelven modelos CGE de verdad y
necesitan piezas que NO instala `uv sync` —la libreria PATH (C-API), IPOPT o
pymumps (conda-forge)—. Sin guard fallaban con un `code=2` opaco en cualquier
maquina que no las tuviera, CI incluido, en vez de saltarse.

Cada test declara SOLO lo que usa, para que un solver ausente no desactive de
paso tests que si podrian correr:

    @pytest.mark.needs_mumps     # pymumps (conda-forge)
    @pytest.mark.needs_ipopt     # ejecutable ipopt
    @pytest.mark.needs_cyipopt   # binding de Python cyipopt (NO es lo mismo)
    @pytest.mark.needs_path      # libreria PATH C-API + path-capi-python
    @pytest.mark.needs_gdxdump   # ejecutable gdxdump (GAMS)
    @pytest.mark.needs_asl       # interfaz PyNumero ASL (pynumero_ASL)

Con `EQUILIBRIA_REQUIRE_SOLVERS=1` los guards se desactivan y todo corre: util
en local para que un solver ausente se note en vez de esconderse tras un skip.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Any

import pytest

_MARCADORES = {
    "needs_mumps": "pymumps ausente (conda install -c conda-forge pymumps)",
    "needs_ipopt": "ipopt ausente en el PATH",
    "needs_cyipopt": "cyipopt ausente (pip install cyipopt)",
    "needs_path": "libreria PATH ausente (define EQUILIBRIA_PATH_CAPI_LIB_DIR)",
    "needs_gdxdump": "gdxdump ausente en el PATH (viene con GAMS)",
    "needs_asl": "interfaz PyNumero ASL ausente (pynumero_ASL)",
}


def _mumps_disponible() -> bool:
    return importlib.util.find_spec("mumps") is not None


def _ipopt_disponible() -> bool:
    """Cualquiera de las dos vias a IPOPT: el ejecutable o el binding."""
    if shutil.which("ipopt"):
        return True
    return _cyipopt_disponible()


def _cyipopt_disponible() -> bool:
    """SOLO el binding de Python.

    `needs_ipopt` acepta el ejecutable O el binding, asi que no sirve para un
    test que llama a `IPOPTSolver.solve_ipopt()`: ese metodo corta sobre
    `IPOPT_AVAILABLE`, que en pep_model_solver_ipopt.py es literalmente
    `import cyipopt`. Con el ejecutable presente y el binding ausente
    —la combinacion por defecto de `brew install ipopt` + `uv sync`—
    `needs_ipopt` dice "disponible" y el test revienta igual con ImportError.
    """
    return importlib.util.find_spec("cyipopt") is not None


def _gdxdump_disponible() -> bool:
    return shutil.which("gdxdump") is not None


def _asl_disponible() -> bool:
    """La interfaz ASL se compila aparte de pyomo; `find_spec` no alcanza."""
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface
    except Exception:
        return False
    try:
        return bool(AmplInterface.available())
    except Exception:
        return False


def _path_disponible() -> bool:
    from equilibria._local_refs import path_capi_lib_dir, path_capi_src

    return (
        path_capi_lib_dir() / "libpath50.silicon.dylib"
    ).exists() and path_capi_src().exists()


def _ausentes() -> set[str]:
    """Marcadores cuyo solver no esta disponible en esta maquina."""
    if os.environ.get("EQUILIBRIA_REQUIRE_SOLVERS") == "1":
        return set()
    faltan = set()
    if not _mumps_disponible():
        faltan.add("needs_mumps")
    if not _ipopt_disponible():
        faltan.add("needs_ipopt")
    if not _cyipopt_disponible():
        faltan.add("needs_cyipopt")
    if not _path_disponible():
        faltan.add("needs_path")
    if not _gdxdump_disponible():
        faltan.add("needs_gdxdump")
    if not _asl_disponible():
        faltan.add("needs_asl")
    return faltan


def pytest_configure(config: pytest.Config) -> None:
    for marcador, motivo in _MARCADORES.items():
        config.addinivalue_line("markers", f"{marcador}: {motivo}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    faltan = _ausentes()
    if not faltan:
        return
    for item in items:
        for marcador in faltan:
            if marcador in item.keywords:
                item.add_marker(pytest.mark.skip(reason=_MARCADORES[marcador]))
                break
