"""Alineacion: empareja las celdas de A y B segun el mapeo.

Por cada variable y cada periodo comun: renombra periodos y elementos (por
conjunto y lado), reordena las dimensiones de B al orden de A segun ``dims`` y
empareja por tupla EXACTA. Nunca ordena etiquetas: el orden de las dimensiones es
el declarado, asi que dos conjuntos con elementos de igual nombre no se cruzan.

Celdas ausentes de un GDX: GAMS no escribe los registros con nivel 0. En un lado con
``Solution.absent_is_zero``, una celda que el otro lado tiene vale 0 si la variable
de ese lado tiene algun registro en ese periodo. El dominio de la variable se toma,
pues, de las celdas del OTRO lado (no de los conjuntos del GDX): no hace falta leer
los conjuntos y no aparecen celdas fuera del modelo del otro lado. Si la variable no
tiene ningun registro en el periodo, o el lado es un dict/Pyomo, la celda sigue sin
pareja.
"""

from __future__ import annotations

from dataclasses import dataclass

from equilibria.compare.errors import MappingError
from equilibria.compare.mapping import Mapping, VariableMap
from equilibria.compare.solution import CellKey, Index, Solution


@dataclass(frozen=True)
class Aligned:
    """Una variable alineada: parejas y celdas sin pareja (en periodos comunes)."""

    var: VariableMap
    pairs: list[tuple[CellKey, float, float]]  # ((periodo, indices), A, B)
    missing_in_a: dict[CellKey, float]  # estan en B (nivel de B), faltan en A
    missing_in_b: dict[CellKey, float]  # estan en A (nivel de A), faltan en B


def common_periods(
    a: Solution, b: Solution, mapping: Mapping
) -> tuple[list, list, list]:
    """(comunes, solo en A, solo en B), ya con nombres comunes; orden de aparicion."""
    pa = list(dict.fromkeys(mapping.period_name("a", p) for p in a.periods()))
    pb = list(dict.fromkeys(mapping.period_name("b", p) for p in b.periods()))
    return (
        [p for p in pa if p in pb],
        [p for p in pa if p not in pb],
        [p for p in pb if p not in pa],
    )


def _translate(
    sol: Solution, side: str, vm: VariableMap, mapping: Mapping, periods: set[str]
) -> dict[CellKey, float]:
    s = vm.side(side)
    if s.var in sol.skipped:
        raise MappingError(
            f"variable {vm.name!r}: {s.var!r} de {side} {sol.skipped[s.var]}"
        )
    if s.var not in sol.levels:
        raise MappingError(
            f"variable {vm.name!r}: {s.var!r} no existe en {side} ({sol.origin})"
        )
    if sol.ndim[s.var] != len(s.dims):
        raise MappingError(
            f"variable {vm.name!r}: dims de {side} {list(s.dims)} declaran "
            f"{len(s.dims)} dimensiones; {s.var!r} tiene {sol.ndim[s.var]} (sin periodo)"
        )
    order = [
        s.dims.index(d) for d in vm.a.dims
    ]  # posicion en este lado de cada dim comun
    out: dict[CellKey, float] = {}
    for (period, idx), val in sol.levels[s.var].items():
        p = mapping.period_name(side, period)
        if p not in periods:
            continue
        renamed = [
            mapping.element_name(side, d, e) for d, e in zip(s.dims, idx, strict=True)
        ]
        key: tuple[str, Index] = (p, tuple(renamed[i] for i in order))
        if key in out:
            raise MappingError(
                f"variable {vm.name!r}: dos celdas de {side} caen en {key} tras el renombre"
            )
        out[key] = val
    return out


def _fill_absent(
    own: dict[CellKey, float], other: dict[CellKey, float], sol: Solution
) -> dict[CellKey, float]:
    """Celdas de ``other`` que ``own`` no tiene, a 0 si ``sol`` omite los ceros y
    la variable tiene registros en ese periodo (ver el docstring del modulo)."""
    if not sol.absent_is_zero:
        return own
    with_records = {period for period, _ in own}
    filled = dict(own)
    for key in other:
        if key not in filled and key[0] in with_records:
            filled[key] = 0.0
    return filled


def align(
    a: Solution, b: Solution, mapping: Mapping, periods: set[str]
) -> list[Aligned]:
    """Alinea cada variable del mapeo en ``periods`` (los comunes, ya renombrados;
    ver :func:`common_periods`)."""
    result = []
    for vm in mapping.variables:
        ta = _translate(a, "a", vm, mapping, periods)
        tb = _translate(b, "b", vm, mapping, periods)
        ca = _fill_absent(ta, tb, a)
        cb = _fill_absent(tb, ta, b)
        result.append(
            Aligned(
                vm,
                [(k, va, cb[k]) for k, va in ca.items() if k in cb],
                {k: v for k, v in cb.items() if k not in ca},
                {k: v for k, v in ca.items() if k not in cb},
            )
        )
    return result
