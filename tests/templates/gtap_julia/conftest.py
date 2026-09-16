"""Guard comun: estos tests necesitan Julia y el checkout del modelo GTAPv7.

Sin este guard, los que llaman al oraculo revientan con FileNotFoundError en
cualquier maquina sin Julia —CI incluido— en lugar de saltarse. Las rutas se
configuran con EQUILIBRIA_JULIA_BIN y EQUILIBRIA_JULIA_PKG (ver
equilibria._local_refs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from equilibria._local_refs import julia_bin, julia_pkg_dir

_AQUI = str(Path(__file__).parent)


def _motivo_del_salto() -> str:
    """Devuelve por que hay que saltar, o cadena vacia si Julia esta lista."""
    binario = julia_bin()
    if not binario.exists():
        return f"julia no encontrado en {binario} (define EQUILIBRIA_JULIA_BIN)"
    pkg = julia_pkg_dir()
    if not pkg.exists():
        return (
            f"GlobalTradeAnalysisProjectModelV7.jl no encontrado en {pkg} "
            "(define EQUILIBRIA_JULIA_PKG)"
        )
    return ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    motivo = _motivo_del_salto()
    if not motivo:
        return
    salto = pytest.mark.skip(reason=motivo)
    for item in items:
        if str(item.fspath).startswith(_AQUI):
            item.add_marker(salto)
