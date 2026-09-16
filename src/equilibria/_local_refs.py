"""Localizacion de referencias externas (GAMS, GEMPACK, PATH) por entorno.

Estas rutas apuntan a material que NO se versiona: los .gdx/.har de referencia que
produce GAMS, los datasets NUS333 y el checkout de path-capi-python. Antes estaban
fijas al disco del autor, asi que en cualquier otra maquina los tests que dependen de
ellas se saltaban sin explicar que faltaba.

Cada funcion devuelve ``None`` cuando la referencia no esta disponible; quien llama
decide si eso es un skip o un error. El valor llega de la variable de entorno
correspondiente, y si no esta definida se usa la convencion `~/proyectos2/...`, que es
donde vive en la maquina de desarrollo original.

Variables reconocidas:

``EQUILIBRIA_REFS_DIR``
    Raiz de los .gdx de referencia generados con GAMS (por defecto
    ``~/proyectos2/equilibria_refs``).
``EQUILIBRIA_NUS333_DIR``
    Dataset NUS333 de GTAP (por defecto ``~/Downloads/10284``).
``EQUILIBRIA_PATH_CAPI_SRC``
    Directorio ``src`` del checkout de path-capi-python (por defecto
    ``~/proyectos/path-capi-python/src``).
``EQUILIBRIA_CGE_BABEL_DIR``
    Fuentes .gms del GTAP 7 estandar usadas como referencia (por defecto
    ``~/proyectos2/cge_babel``).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "cge_babel_dir",
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


def path_capi_src() -> Path | None:
    """``src`` del checkout de path-capi-python, o ``None`` si no esta."""
    ruta = _por_entorno("EQUILIBRIA_PATH_CAPI_SRC", "~/proyectos/path-capi-python/src")
    return ruta if ruta.exists() else None


def cge_babel_dir() -> Path:
    """Fuentes .gms de referencia del GTAP 7 estandar. Puede no existir."""
    return _por_entorno("EQUILIBRIA_CGE_BABEL_DIR", "~/proyectos2/cge_babel")


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
