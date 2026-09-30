"""Fijar un shock en un instrumento registrado (ShockBlock), solo en un periodo.

Equivale a GAMS `x.fx(..., 'shock') = v`: el instrumento es una Var fija en los
3 periodos y el driver no la libera ni la re-siembra (``_exogenous_instruments``).
No confundir con ``shocks.apply_shock``, que modifica ``params`` (la API YAML).
"""

from __future__ import annotations

from typing import Any

# Filas que leen cada instrumento. Su celda del periodo tiene que estar viva: si no,
# el shock no entra al modelo y el solve igual da code=1 (p.ej. ``aft`` de un
# factor con ``xftflag<=0``, donde eq_xfteq no se genera).
INSTRUMENT_EQS: dict[str, tuple[str, ...]] = {
    "lambdava": ("eq_va", "eq_pxeq"),
    "aft": ("eq_xfteq",),
}


def check_instrument_cell(
    m: Any, name: str, index: tuple, period: str = "shock"
) -> None:
    """ValueError si la celda ``name[*index, period]`` no puede llevar un shock.

    Falla si la celda no existe (nombrando los indices validos de esa region), si
    ninguna fila viva de ``INSTRUMENT_EQS`` la lee, o si es ``aft`` con ``xft``
    fijada (el solver desactivaria eq_xfteq). En los tres casos el shock se
    perderia con code=1.
    """
    var = getattr(m, name, None)
    idx = (*index, period)
    if var is None or idx not in var:
        valid = sorted(
            {
                k[1]
                for k in (var if var is not None else ())
                if k[0] == index[0] and k[-1] == period
            }
        )
        raise ValueError(f"{name}{idx} does not exist. Valid for {index[0]}: {valid}")
    for eq_name in INSTRUMENT_EQS.get(name, ()):
        eq = getattr(m, eq_name, None)
        if eq is None or idx not in eq or not eq[idx].active:
            live = (
                []
                if eq is None
                else sorted(
                    {
                        k[1]
                        for k in eq
                        if k[0] == index[0] and k[-1] == period and eq[k].active
                    }
                )
            )
            raise ValueError(
                f"{name}{idx}: {eq_name}{idx} does not exist or is inactive, the "
                f"shock would not enter the model. Live for {index[0]}: {live}"
            )
    if name == "aft" and m.xft[idx].fixed:
        raise ValueError(
            f"aft{idx}: xft{idx} is fixed; the solver would deactivate eq_xfteq "
            "and the shock would be lost"
        )


def fix_instrument_shock(
    m: Any,
    name: str,
    index: tuple,
    *,
    factor: float | None = None,
    value: float | None = None,
    period: str = "shock",
) -> float:
    """Fijar ``name[*index, period]`` en ``value`` o en ``factor`` x su valor de
    ``'check'`` (el benchmark: aplicar el mismo shock dos veces da lo mismo que una).

    Devuelve el valor fijado. ValueError si el nombre no es un instrumento
    registrado, si no se da exactamente uno de factor/value, si el resultado no es
    > 0, o si la celda no puede llevar el shock (``check_instrument_cell``).
    """
    from pyomo.environ import value as _v

    registered = getattr(m, "_exogenous_instruments", frozenset())
    if name not in registered:
        raise ValueError(
            f"{name!r} is not a registered instrument; registered: {sorted(registered)}"
        )
    if (factor is None) == (value is None):
        raise ValueError("give exactly one of factor/value")
    check_instrument_cell(m, name, index, period)
    var = getattr(m, name)
    idx = (*index, period)
    if value is not None:
        new = float(value)
    else:
        assert factor is not None  # exactly one of factor/value, checked above
        new = float(_v(var[(*index, "check")])) * float(factor)
    if not new > 0.0:
        raise ValueError(f"{name}{idx} = {new}: must be > 0")
    var[idx].fix(new)
    return new
