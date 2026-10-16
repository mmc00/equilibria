"""Solution: el formato comun de las fuentes.

Niveles por ``(variable, periodo, indices)`` con los nombres DEL LADO DE ORIGEN: no
traduce nada (eso lo hace el mapeo). Las fuentes de :mod:`equilibria.compare.sources`
son la forma normal de construirla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Index = tuple[str, ...]
CellKey = tuple[str, Index]  # (periodo, indices sin el periodo)


@dataclass(frozen=True)
class Solution:
    """Niveles de una solucion.

    ``levels[var][(periodo, indices)] = nivel``; ``ndim[var]`` es el numero de
    dimensiones de ``var`` SIN contar el periodo (se conoce aunque la variable no
    tenga celdas). ``origin`` describe la fuente, solo para los mensajes.
    ``skipped[var]`` dice por que una variable de la fuente no se pudo leer (p. ej.
    un escalar sin la dimension de periodo declarada); pedirla en un mapeo es error.

    ``absent_is_zero`` (lo pone :func:`~equilibria.compare.sources.from_gdx`): la
    fuente omite las celdas con nivel 0, como hace GAMS al escribir un ``.gdx``. Al
    alinear, una celda que el otro lado tiene y esta fuente no vale 0 si la variable
    tiene algun registro en ese periodo; si no tiene ninguno, sigue sin pareja.
    """

    levels: dict[str, dict[CellKey, float]]
    ndim: dict[str, int]
    origin: str = field(default="")
    skipped: dict[str, str] = field(default_factory=dict)
    absent_is_zero: bool = False

    def periods(self) -> list[str]:
        """Periodos presentes, en el orden en que aparecen (sin ordenar)."""
        seen: dict[str, None] = {}
        for cells in self.levels.values():
            for period, _ in cells:
                seen.setdefault(period, None)
        return list(seen)
