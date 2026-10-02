"""Leer los campos de ``results.solver`` de Pyomo sin sus trampas.

``results.solver`` es un ListContainer:

- ``.get()`` no delega al primer elemento y devuelve siempre el default;
- el atributo si delega, pero un campo no declarado (``iterations``, ``time``)
  lanza ``AttributeError`` y uno declarado sin valor (``message``) vuelve como
  ``UndefinedData``, que es verdadero y ``str()`` convierte en "<undefined>";
- el lector ``.sol`` escapa ":" como "\\x3a" en el mensaje.

Ningun solver AMPL llena ``iterations`` (medido con Ipopt 3.14.19 y PATH 4.7.03 /
5.0.05); PATH da el conteo en su mensaje ("50 iterations (0 for crash); ...").
"""

from __future__ import annotations

import re
from typing import Any


def solver_field(results: Any, name: str) -> Any:
    """Un campo de ``results.solver``, o None si el solver no lo dio."""
    from pyomo.opt.results.container import UndefinedData

    val = getattr(results.solver, name, None)
    return None if isinstance(val, UndefinedData) else val


def solver_message(results: Any) -> str | None:
    """El mensaje del solver, sin el escape "\\x3a" del lector ``.sol``; None si no hay."""
    msg = solver_field(results, "message")
    return None if msg is None else str(msg).replace("\\x3a", ":")


def iterations_from_message(message: str) -> int | None:
    """El conteo de un mensaje AMPL como "68 iterations (0 for crash); ..." (PATH)."""
    match = re.search(r"(\d+) iterations", message)
    return int(match.group(1)) if match else None


def solver_iterations(results: Any) -> int | None:
    """Las iteraciones que informa el solver: el campo si existe, si no su mensaje.

    None si no las informa (Ipopt via AMPL, por ejemplo): desconocido, no 0.
    """
    iterations = solver_field(results, "iterations")
    if iterations is not None:
        return iterations
    msg = solver_message(results)
    return iterations_from_message(msg) if msg else None
