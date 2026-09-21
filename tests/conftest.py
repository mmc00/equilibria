"""pytest configuration and shared fixtures.

Incluye los guards de solver: varios tests resuelven modelos CGE de verdad y
necesitan piezas que NO instala `uv sync` —la libreria PATH (C-API), IPOPT o
pymumps (conda-forge)—. Sin guard fallaban con un `code=2` opaco en cualquier
maquina que no las tuviera, CI incluido, en vez de saltarse.

Cada test declara SOLO lo que usa, para que un solver ausente no desactive de
paso tests que si podrian correr:

    @pytest.mark.needs_mumps     # pymumps (conda-forge)
    @pytest.mark.needs_ipopt     # ejecutable ipopt
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
    "needs_path": "libreria PATH ausente (define EQUILIBRIA_PATH_CAPI_LIB_DIR)",
    "needs_gdxdump": "gdxdump ausente en el PATH (viene con GAMS)",
    "needs_asl": "interfaz PyNumero ASL ausente (pynumero_ASL)",
}


def _mumps_disponible() -> bool:
    return importlib.util.find_spec("mumps") is not None


def _ipopt_disponible() -> bool:
    if shutil.which("ipopt"):
        return True
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
    """La libreria PATH, por el nombre que tenga en ESTA plataforma.

    Comprobaba `libpath50.silicon.dylib` a secas, asi que en Linux daba False
    con la libreria instalada al lado (upstream publica `libpath50.so`) y los
    59 tests `needs_path` se saltaban en CI pasara lo que pasara.
    """
    from equilibria._local_refs import (
        path_capi_lib_dir,
        path_capi_lib_names,
        path_capi_src,
    )

    lib_dir = path_capi_lib_dir()
    tiene_lib = any((lib_dir / nombre).exists() for nombre in path_capi_lib_names())
    return tiene_lib and path_capi_src().exists()


def _ausentes() -> set[str]:
    """Marcadores cuyo solver no esta disponible en esta maquina."""
    if os.environ.get("EQUILIBRIA_REQUIRE_SOLVERS") == "1":
        return set()
    faltan = set()
    if not _mumps_disponible():
        faltan.add("needs_mumps")
    if not _ipopt_disponible():
        faltan.add("needs_ipopt")
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
