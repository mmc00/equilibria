"""Aplicar un shock al modelo multiperiodo: escribirlo en el bloque shock.

El shock vive en el ShockBlock: cada instrumento es una Var fija por periodo, y el
shock es que su celda del periodo ``shock`` difiera de la del ``check`` (GAMS
``x.fx(..., 'shock') = v``).  Se escribe ANTES de ``solve_multiperiod``; el driver
no recibe el shock, lo lee del modelo (``shock_of``).

    apply_shock(m, {"imptx": {("ROW", "MFG", "USA"): 8.6637},
                    "aft":   {("USA", "CAPITAL"): 10.0}})

Cada numero es el % de GEMPACK y la conversion a niveles la decide el
instrumento (``GEMPACK_KIND``), sobre su valor del check:

  pct          % directo del instrumento (avaall, qe, afeall...)   x*(1+p)
  power        % de la potencia 1+t (tms, to, tpdall)              (1+t)*(1+p)-1
  power_kappa  % de la potencia 1/(1-kappaf) (tinc)                1-(1-k)/(1+p)
  power_fct    % de 1+fctts+fcttx (tfe); fcttx absorbe el cambio

Con ``levels=True`` cada numero es el valor en niveles de la celda.  Aplicar el
mismo shock dos veces da lo mismo que una: la conversion parte del check.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from equilibria.blocks.gtap.periods import SHOCK
from equilibria.templates.gtap.instruments import (
    check_instrument_cell,
    exogenous_test,
    fix_instrument_shock,
)

# Como se lee el % de GEMPACK de cada instrumento del ShockBlock.  Los objetivos
# de @overwrite (b.target) son factores sobre el check: "pct".
GEMPACK_KIND: dict[str, str] = {
    "lambdava": "pct",
    "aft": "pct",
    "imptx": "power",
    "prdtx_rai": "power",
    "fcttx": "power_fct",
    "dintx_tgt": "power",
    "mintx_tgt": "power",
    "kappaf": "power_kappa",
    "exptx": "power",
    "pop": "pct",
    "lambdaf": "pct",
    "axp": "pct",
    "lambdam": "pct",
    "lambdamg": "pct",
}


def gempack_kind(name: str) -> str:
    """Como se lee el % de GEMPACK de ``name`` (ver ``GEMPACK_KIND``)."""
    return GEMPACK_KIND.get(name, "pct")


def shocked(kind: str, chk: Any, f: Any, fs: Any = 0.0) -> Any:
    """El valor shockeado desde el de check ``chk``, con ``f`` = 1+pct/100.

    Vale con numeros y con expresiones GAMS en texto (gen_burfisher_gams.py la usa
    para escribir el mismo shock en el oraculo). ``fs``: fctts de la celda.
    """
    if kind == "pct":
        return chk * f
    if kind == "power":
        return (1 + chk) * f - 1
    if kind == "power_kappa":
        return 1 - (1 - chk) / f
    if kind == "power_fct":
        return (1 + fs + chk) * f - 1 - fs
    raise ValueError(f"unknown GEMPACK kind: {kind!r}")


def _level(m: Any, name: str, cell: tuple, pct: float) -> float:
    """Valor en niveles de ``name[*cell, 'shock']`` para el % GEMPACK ``pct``."""
    from pyomo.environ import value

    chk = float(value(getattr(m, name)[(*cell, "check")]))
    kind = gempack_kind(name)
    fs = 0.0
    if kind == "power_fct":
        # La potencia es 1+fctts+fcttx y fctts no se mueve.
        fctts = getattr(m, "_fctts", None)
        if fctts is None:
            raise ValueError(
                f"{name}{cell}: the model carries no fctts (build_block_model)"
            )
        fs = float(fctts.get(cell, 0.0))
    return float(shocked(kind, chk, 1.0 + float(pct) / 100.0, fs))


def apply_shock(
    m: Any, shock: Mapping[str, Mapping[tuple, float]], *, levels: bool = False
) -> dict[str, dict[tuple, float]]:
    """Escribe ``shock`` ({instrumento: {celda: %}}) en el periodo shock de ``m``.

    Devuelve los valores en niveles fijados.  ValueError (de
    ``fix_instrument_shock``) si un instrumento no esta registrado, si la celda no
    existe o ninguna fila viva la lee, o si el valor sale de su dominio.
    """
    out: dict[str, dict[tuple, float]] = {}
    registered = getattr(m, "_exogenous_instruments", frozenset())
    for name, cells in shock.items():
        if name not in registered:
            raise ValueError(
                f"{name!r} is not a registered instrument; registered: "
                f"{sorted(registered)}"
            )
        for cell, x in cells.items():
            cell = tuple(cell)
            check_instrument_cell(m, name, cell)  # antes de leer su check
            v = float(x) if levels else _level(m, name, cell, x)
            out.setdefault(name, {})[cell] = fix_instrument_shock(
                m, name, cell, value=v
            )
    return out


def shock_of(m: Any) -> list[str]:
    """Celdas del bloque shock cuyo 'shock' difiere de su 'check': el shock del
    modelo.  Vacia = sin shock (el driver aplica entonces el arancel +10%)."""
    from pyomo.environ import value

    out = []
    for name in sorted(getattr(m, "_exogenous_instruments", ())):
        var = getattr(m, name)
        exo = exogenous_test(m, name)
        for k in var:
            if k[-1] != SHOCK or not exo(k):
                continue  # otro periodo, o celda endogena (@overwrite): no es shock
            ck = (*k[:-1], "check")
            if ck in var and float(value(var[k])) != float(value(var[ck])):
                out.append(f"{name}{tuple(k[:-1])}")
    return out


def check_shock_entered(m: Any) -> None:
    """Tras un solve convergido: RuntimeError si un shock de ``aft`` no llego a la
    solucion.  code=1 solo no lo garantiza: si eq_xfteq se hubiera apagado, xft
    queda suelto y el solve igual converge.  Se evalua la fila
    ``xft = aft*(pft/pabs)**etaf`` del shock (este activa o no) con el aft
    shockeado; con etaf=0 es exactamente "xft se movio el factor del shock"."""
    from pyomo.environ import value

    aft = getattr(m, "aft", None)
    eq = getattr(m, "eq_xfteq", None)
    if aft is None or eq is None:
        return
    exo = exogenous_test(m, "aft")
    for k in aft:
        if k[-1] != SHOCK or not exo(k):
            continue
        ck = (*k[:-1], "check")
        if float(value(aft[k])) == float(value(aft[ck])) or k not in eq:
            continue
        row = eq[k]
        resid = float(value(row.body)) - float(value(row.upper))
        xft = float(value(m.xft[k]))
        if abs(resid) > 1e-6 * max(1.0, abs(xft)):
            raise RuntimeError(
                f"aft{tuple(k[:-1])} = {value(aft[k])} but eq_xfteq{k} is off by "
                f"{resid:.3e} (xft = {xft}): the shock did not enter the solution"
            )
