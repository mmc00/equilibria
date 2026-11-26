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

:func:`coverage_problems` keeps every template behind a door: it finds the template
classes in the source without importing it (:func:`find_templates`), and each one
must be the template of a door in :data:`DOORS` or be listed in :data:`PENDING` with
the reason it has none yet. ``tests/core/test_door_contract.py`` runs both checks in
CI, and ``scripts/dev/check_doors.py`` before a commit that touches ``src/``.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

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

# template classes without a door yet -> why (and what unblocks it)
PENDING: dict[str, str] = {
    "equilibria.templates.simple_open.template.SimpleOpenEconomy": (
        "no closure class of its own (no ModelClosureConfig subclass) yet; its "
        "door equilibria.simple_open comes after MAR-249"
    ),
}


def package_root() -> Path:
    """The folder of the installed ``equilibria`` package."""
    import equilibria

    return Path(equilibria.__file__).resolve().parent


def _names(nodes: list[ast.expr]) -> set[str]:
    out = set()
    for n in nodes:
        if isinstance(n, ast.Subscript):
            n = n.value
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _real_body(fn: ast.FunctionDef) -> bool:
    """Not a stub: not only ``...`` or ``raise NotImplementedError`` (after a docstring)."""
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]  # the docstring
    if not body:
        return False
    if len(body) == 1:
        only = body[0]
        if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant):
            return only.value.value is not Ellipsis
        if isinstance(only, ast.Raise) and only.exc is not None:
            exc = only.exc.func if isinstance(only.exc, ast.Call) else only.exc
            return "NotImplementedError" not in _names([exc])
    return True


def find_templates(root: Path, package: str = "equilibria") -> set[str]:
    """Qualified names of the template classes under ``root`` (the folder of
    ``package``), read from the source, nothing imported. A template is a class with a
    real ``default_recipe`` (not a ``Protocol`` stub nor an abstract one that raises
    ``NotImplementedError``) or a subclass of ``ModelTemplate``."""
    out: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        module = ".".join([package, *parts])
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _names(node.bases)
            if "Protocol" in bases:
                continue
            recipe = next(
                (
                    f
                    for f in node.body
                    if isinstance(f, ast.FunctionDef) and f.name == "default_recipe"
                ),
                None,
            )
            if (recipe is not None and _real_body(recipe)) or (
                recipe is None and "ModelTemplate" in bases
            ):
                out.add(f"{module}.{node.name}")
    return out


def coverage_problems(
    root: Path | None = None,
    package: str = "equilibria",
    doors: tuple[DoorSpec, ...] = DOORS,
    pending: dict[str, str] | None = None,
) -> list[str]:
    """One message per template class with no door and not in ``pending``, per
    ``pending`` entry that names no template class, and per ``pending`` entry whose
    template already has a door; an empty list passes."""
    pending = PENDING if pending is None else pending
    templates = find_templates(root or package_root(), package)
    behind: dict[str, str] = {}
    for spec in doors:
        cls = getattr(importlib.import_module(spec.module), spec.template, None)
        if cls is not None:
            behind[f"{cls.__module__}.{cls.__qualname__}"] = spec.module
    out: list[str] = []
    for t in sorted(templates):
        if t in behind and t in pending:
            out.append(f"{t} is in PENDING but has a door ({behind[t]}): remove it")
        elif t not in behind and t not in pending:
            out.append(
                f"{t} has no door: add a DoorSpec to core.door.DOORS (ADR-0012) "
                "or list it in PENDING with the reason"
            )
    out += [
        f"{p} is in PENDING but is not a template class"
        for p in sorted(pending)
        if p not in templates
    ]
    return out


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
