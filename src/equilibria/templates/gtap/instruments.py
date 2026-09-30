"""Aplicar un shock a un instrumento registrado (ShockBlock), solo en un periodo.

Equivale a GAMS `x.fx(..., 'shock') = v`: el instrumento es una Var fija en los
3 periodos y el driver no la libera ni la re-siembra (``_exogenous_instruments``).
"""

from __future__ import annotations

from typing import Any


def apply_shock(
    m: Any,
    name: str,
    index: tuple,
    *,
    factor: float | None = None,
    value: float | None = None,
    period: str = "shock",
) -> float:
    """Fijar ``name[*index, period]`` en ``value`` o en ``factor`` x su valor actual.

    Devuelve el valor fijado. ValueError si el nombre no es un instrumento
    registrado, si la celda no existe (nombrando los indices validos de esa
    region), si no se da exactamente uno de factor/value, o si el resultado no es
    > 0.
    """
    from pyomo.environ import value as _v

    registered = getattr(m, "_exogenous_instruments", frozenset())
    if name not in registered:
        raise ValueError(
            f"{name!r} is not a registered instrument; registered: {sorted(registered)}"
        )
    if (factor is None) == (value is None):
        raise ValueError("give exactly one of factor/value")
    var = getattr(m, name)
    idx = (*index, period)
    if idx not in var:
        valid = sorted({k[1] for k in var if k[0] == index[0] and k[-1] == period})
        raise ValueError(f"{name}{idx} does not exist. Valid for {index[0]}: {valid}")
    if value is not None:
        new = float(value)
    else:
        assert factor is not None  # exactly one of factor/value, checked above
        new = float(_v(var[idx])) * float(factor)
    if not new > 0.0:
        raise ValueError(f"{name}{idx} = {new}: must be > 0")
    var[idx].fix(new)
    return new
