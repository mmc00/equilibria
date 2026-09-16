"""Localizacion de referencias externas (GAMS, GEMPACK, PATH) por entorno.

Estas rutas apuntan a material que NO se versiona: los .gdx/.har de referencia que
produce GAMS, los datasets NUS333, el checkout de path-capi-python y la instalacion de
Julia. Antes estaban fijas al disco del autor, asi que en cualquier otra maquina los
tests que dependen de ellas fallaban o se saltaban sin explicar que faltaba.

Salvo :func:`ref_gdx`, todas devuelven un ``Path`` exista o no —nunca ``None``— porque
quien llama suele encadenar ``.exists()`` a nivel de modulo. Comprobar la existencia y
decidir entre saltar o fallar es responsabilidad de quien llama.

Variables reconocidas:

``EQUILIBRIA_REFS_DIR``
    Raiz de los .gdx de referencia generados con GAMS (por defecto
    ``~/proyectos2/equilibria_refs``).
``EQUILIBRIA_NUS333_DIR``
    Dataset NUS333 de GTAP (por defecto ``~/Downloads/10284``).
``EQUILIBRIA_PATH_CAPI_SRC``
    Directorio ``src`` del checkout de path-capi-python (por defecto
    ``~/proyectos/path-capi-python/src``).
``EQUILIBRIA_PATH_CAPI_LIB_DIR``
    Directorio con libpath/liblusol (por defecto, la cache del proyecto).
``EQUILIBRIA_CGE_BABEL_DIR``
    Fuentes .gms del GTAP 7 estandar usadas como referencia (por defecto
    ``~/proyectos2/cge_babel``).
``EQUILIBRIA_JULIA_BIN``
    Binario de Julia (por defecto, el del PATH o el de juliaup).
``EQUILIBRIA_JULIA_PKG``
    Checkout de GlobalTradeAnalysisProjectModelV7.jl (por defecto
    ``~/proyectos/GlobalTradeAnalysisProjectModelV7.jl``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = [
    "cge_babel_dir",
    "julia_bin",
    "julia_pkg_dir",
    "nus333_dir",
    "path_capi_lib_dir",
    "path_capi_src",
    "ref_gdx",
    "refs_dir",
]


def _por_entorno(variable: str, defecto: str) -> Path:
    valor = os.environ.get(variable)
    return Path(valor).expanduser() if valor else Path(defecto).expanduser()


def refs_dir() -> Path:
    """Raiz de los .gdx de referencia de GAMS. Puede no existir."""
    return _por_entorno("EQUILIBRIA_REFS_DIR", "~/proyectos2/equilibria_refs")


def ref_gdx(*partes: str) -> Path | None:
    """Un .gdx concreto bajo :func:`refs_dir`, o ``None`` si no esta."""
    ruta = refs_dir().joinpath(*partes)
    return ruta if ruta.exists() else None


def nus333_dir() -> Path:
    """Directorio del dataset NUS333. Puede no existir."""
    return _por_entorno("EQUILIBRIA_NUS333_DIR", "~/Downloads/10284")


def path_capi_src() -> Path:
    """``src`` del checkout de path-capi-python. Puede no existir.

    Devuelve siempre un ``Path`` —nunca ``None``— porque quien llama suele
    encadenar ``.exists()`` a nivel de modulo; devolver ``None`` rompia la
    recoleccion de tests en una maquina sin el checkout.
    """
    return _por_entorno("EQUILIBRIA_PATH_CAPI_SRC", "~/proyectos/path-capi-python/src")


def cge_babel_dir() -> Path:
    """Fuentes .gms de referencia del GTAP 7 estandar. Puede no existir."""
    return _por_entorno("EQUILIBRIA_CGE_BABEL_DIR", "~/proyectos2/cge_babel")


def julia_bin() -> Path:
    """Binario de Julia. Puede no existir (p. ej. en CI).

    Se busca en ``EQUILIBRIA_JULIA_BIN``, luego en el PATH, y por ultimo en la
    instalacion de juliaup del usuario.
    """
    explicito = os.environ.get("EQUILIBRIA_JULIA_BIN")
    if explicito:
        return Path(explicito).expanduser()
    en_path = shutil.which("julia")
    if en_path:
        return Path(en_path)
    return Path("~/.juliaup/bin/julia").expanduser()


def julia_pkg_dir() -> Path:
    """Checkout de GlobalTradeAnalysisProjectModelV7.jl. Puede no existir."""
    return _por_entorno(
        "EQUILIBRIA_JULIA_PKG", "~/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
    )


def path_capi_lib_dir() -> Path:
    """Directorio con libpath/liblusol.

    Los .dylib no se versionan y viven en la cache del checkout principal, no en
    cada worktree. Se busca, en orden: EQUILIBRIA_PATH_CAPI_LIB_DIR, la cache del
    directorio actual, y la del checkout principal. Devuelve el primero que
    contenga libpath; si ninguno lo tiene, la cache del directorio actual (asi el
    mensaje de error apunta a donde el usuario esperaria ponerlo).
    """
    candidatos = []
    explicito = os.environ.get("EQUILIBRIA_PATH_CAPI_LIB_DIR")
    if explicito:
        candidatos.append(Path(explicito).expanduser())
    local = Path.cwd() / ".cache" / "path_capi"
    candidatos.extend(
        [local, Path("~/proyectos2/equilibria/.cache/path_capi").expanduser()]
    )
    for cand in candidatos:
        if (cand / "libpath50.silicon.dylib").exists():
            return cand
    return local
