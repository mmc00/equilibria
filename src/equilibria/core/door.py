"""The door contract (ADR-0012): what every public model door ``equilibria.<model>`` exposes.

A door is one way in per model. It only re-exports the existing pieces (or wraps them:
``load`` and ``results`` are its own functions), it only adds (the internal import
paths keep working) and it hides nothing behind a do-everything call. Every door
exposes, in its ``__all__``:

- ``load(source, ...)``: the model's data;
- the template class (``GTAP``...), which satisfies ``core.exercise.Template``;
- ``ClosureConfig``: a subclass of ``contracts.ModelClosureConfig`` with at least one
  named starting point, a classmethod that takes ``**changes`` and returns a
  ``ClosureConfig`` (GTAP: ``gtap_default``);
- a recipe factory (``gtap_static``...) that returns a ``core.recipe.Recipe``;
- ``Exercise``: the same object as ``core.exercise.Exercise``;
- ``results(ex, ...)``: the view of a solved exercise.

:func:`check_door` checks what it can without data. The behaviour of ``load`` (the
lookup order and the ``FileNotFoundError`` that names the folder and the file) and of
``results`` (``RuntimeError`` before ``ex.run()``; ``tables`` and ``summary``) needs
data and a solve, so each door's own tests check it: for GTAP,
``tests/gtap/test_load.py`` and ``tests/gtap/test_results.py``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

CONTRACT_NAMES = ("load", "ClosureConfig", "Exercise", "results")


@dataclass(frozen=True)
class DoorSpec:
    """A door: its module and the names of its template, recipe factory and the
    closure's named starting point."""

    module: str
    template: str
    recipe: str
    starting_point: str


DOORS: tuple[DoorSpec, ...] = (
    DoorSpec("equilibria.gtap", "GTAP", "gtap_static", "gtap_default"),
)


def check_door(spec: DoorSpec) -> list[str]:
    """One message per broken rule of the door contract; an empty list passes."""
    from equilibria.contracts import ModelClosureConfig
    from equilibria.core.exercise import Exercise
    from equilibria.core.recipe import Recipe

    door = importlib.import_module(spec.module)
    out: list[str] = []
    exported = set(getattr(door, "__all__", ()))
    for name in (*CONTRACT_NAMES, spec.template, spec.recipe):
        if name not in exported:
            out.append(f"{spec.module}: {name!r} is not in __all__")
        if not hasattr(door, name):
            out.append(f"{spec.module}: has no {name!r}")
    if hasattr(door, "Exercise") and door.Exercise is not Exercise:
        out.append(f"{spec.module}: Exercise is not core.exercise.Exercise")
    cc = getattr(door, "ClosureConfig", None)
    if cc is not None:
        if not (isinstance(cc, type) and issubclass(cc, ModelClosureConfig)):
            out.append(
                f"{spec.module}: ClosureConfig is not a subclass of ModelClosureConfig"
            )
        start = getattr(cc, spec.starting_point, None)
        if start is None:
            out.append(f"{spec.module}: ClosureConfig has no {spec.starting_point!r}")
        elif not isinstance(start(), cc):
            out.append(
                f"{spec.module}: ClosureConfig.{spec.starting_point}() "
                "does not return a ClosureConfig"
            )
    tpl = getattr(door, spec.template, None)
    if tpl is not None:
        for method in ("build", "default_recipe"):
            if not callable(getattr(tpl, method, None)):
                out.append(f"{spec.module}: {spec.template} has no {method}()")
    factory = getattr(door, spec.recipe, None)
    if factory is not None and not isinstance(factory(), Recipe):
        out.append(f"{spec.module}: {spec.recipe}() does not return a Recipe")
    return out
