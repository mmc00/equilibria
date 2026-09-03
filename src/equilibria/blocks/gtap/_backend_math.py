"""Backend-neutral math for block equation bodies.

The block bodies are written once and built by more than one backend. Pyomo and
PyOptInterface each supply their own transcendental functions and each has its own
notion of what a variable's value is before a solve, so a body that named either
one directly would only work under that backend.

These helpers dispatch on the operand instead. Under Pyomo they call straight
through to ``pyomo.environ``, so the Pyomo path — the parity oracle — is unchanged.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as _pyo


def _is_poi(obj: Any) -> bool:
    """True if ``obj`` is a PyOptInterface expression or variable handle.

    Checked by module rather than by import, so nothing here requires POI to be
    installed for the Pyomo path to work.
    """
    return type(obj).__module__.startswith("pyoptinterface")


def exp(arg: Any) -> Any:
    """Exponential, from whichever backend produced ``arg``."""
    if _is_poi(arg):
        from pyoptinterface import nl

        return nl.exp(arg)
    return _pyo.exp(arg)


def log(arg: Any) -> Any:
    """Natural logarithm, from whichever backend produced ``arg``."""
    if _is_poi(arg):
        from pyoptinterface import nl

        return nl.log(arg)
    return _pyo.log(arg)


def sqrt(arg: Any) -> Any:
    """Square root, from whichever backend produced ``arg``."""
    if _is_poi(arg):
        from pyoptinterface import nl

        return nl.sqrt(arg)
    return _pyo.sqrt(arg)


def build_value(obj: Any, default: float | None = None) -> float | None:
    """The build-time numeric value of ``obj``, or ``default`` if it has none.

    Some equations consult a variable's current value while the model is being
    built — a guard such as ``if build_value(pnd) <= 0``. Pyomo answers from the
    variable's initialization. A PyOptInterface handle has no value until a solve:
    ``PrimalStart`` is a hint for the solver and ``Value`` exists only afterwards.

    Callers must pass a ``default`` that makes the guard behave as it does under
    Pyomo, and say why at the call site. Returning ``None`` by default keeps a
    caller that forgot from silently taking a different branch.
    """
    if _is_poi(obj):
        return default
    try:
        return _pyo.value(obj)
    except (ValueError, TypeError):
        return default
