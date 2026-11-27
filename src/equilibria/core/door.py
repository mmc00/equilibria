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
import re
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


#: The public model doors; ``check_door`` runs on each.
DOORS: tuple[DoorSpec, ...] = (
    DoorSpec("equilibria.gtap", "GTAP", "gtap_static", "gtap_default"),
)

#: Template classes without a door yet, and why (what unblocks it).
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


MODEL_TEMPLATE = "ModelTemplate"  # equilibria.templates.base.ModelTemplate


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


def _real_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Not a stub: not only ``...``, ``pass``, ``return NotImplemented`` or ``raise
    NotImplementedError`` (after a docstring)."""
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
        if isinstance(only, ast.Pass):
            return False
        if isinstance(only, ast.Return) and "NotImplemented" in _names(
            [only.value] if only.value is not None else []
        ):
            return False
        if isinstance(only, ast.Raise) and only.exc is not None:
            exc = only.exc.func if isinstance(only.exc, ast.Call) else only.exc
            return "NotImplementedError" not in _names([exc])
    return True


def _imports(tree: ast.Module, module: str, is_pkg: bool) -> dict[str, str]:
    """Local name -> the qualified name it refers to, from the module's imports."""
    here = module.split(".") if is_pkg else module.split(".")[:-1]
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    out[top] = top
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - (node.level - 1)]
                base = base + (node.module.split(".") if node.module else [])
            else:
                base = (node.module or "").split(".")
            for a in node.names:
                out[a.asname or a.name] = ".".join([*base, a.name])
    return out


def _resolve(expr: ast.expr, imports: dict[str, str], module: str) -> str | None:
    if isinstance(expr, ast.Subscript):
        return _resolve(expr.value, imports, module)
    if isinstance(expr, ast.Name):
        return imports.get(expr.id, f"{module}.{expr.id}")
    if isinstance(expr, ast.Attribute):
        head = _resolve(expr.value, imports, module)
        return f"{head}.{expr.attr}" if head else None
    return None


def _classes(body: list[ast.stmt]):
    """The module-level classes, also inside ``if``/``try``/``with`` blocks."""
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield node
        elif isinstance(node, ast.If | ast.For | ast.While):
            yield from _classes([*node.body, *node.orelse])
        elif isinstance(node, ast.With):
            yield from _classes(node.body)
        elif isinstance(node, ast.Try):
            blocks = node.body + node.orelse + node.finalbody
            for h in node.handlers:
                blocks = blocks + h.body
            yield from _classes(blocks)


def _own_recipe(node: ast.ClassDef) -> bool | None:
    """``True``: the class defines a real ``default_recipe`` (a method or an
    assignment); ``False``: only a stub; ``None``: it defines none."""
    for f in node.body:
        if (
            isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef)
            and f.name == "default_recipe"
        ):
            if "abstractmethod" in _names(f.decorator_list):
                return False
            return _real_body(f)
        targets = (
            f.targets
            if isinstance(f, ast.Assign)
            else [f.target]
            if isinstance(f, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "default_recipe" for t in targets):
            return True
    return None


def find_templates(root: Path, package: str = "equilibria") -> set[str]:
    """Qualified names of the template classes under ``root`` (the folder of
    ``package``), read from the source, nothing imported. A template is a class with a
    real ``default_recipe`` (a method, ``async`` or not, or an assignment; not a
    ``Protocol`` stub nor an abstract one that only raises ``NotImplementedError``), a
    subclass of ``ModelTemplate``, or a subclass of another template (imports, aliases
    and re-exports are followed). Classes nested in a function or a class are not
    seen."""
    imports: dict[str, dict[str, str]] = {}
    classes: dict[str, tuple[bool | None, list[str]]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        module = ".".join([package, *parts])
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports[module] = _imports(tree, module, path.name == "__init__.py")
        for node in _classes(tree.body):
            bases = [_resolve(b, imports[module], module) or "" for b in node.bases]
            classes[f"{module}.{node.name}"] = (_own_recipe(node), bases)

    def canonical(name: str) -> str:
        for _ in range(10):  # follow re-exports (from .template import GTAP)
            mod, _, attr = name.rpartition(".")
            target = imports.get(mod, {}).get(attr)
            if target is None or target == name:
                return name
            name = target
        return name

    found = {q for q, (own, _) in classes.items() if own}
    changed = True
    while changed:
        changed = False
        for q, (own, bases) in classes.items():
            if q in found or own is False:
                continue
            canon = [canonical(b) for b in bases]
            if any(b.rpartition(".")[2] == "Protocol" for b in canon):
                continue
            if any(b.rpartition(".")[2] == MODEL_TEMPLATE or b in found for b in canon):
                found.add(q)
                changed = True
    return found


def api_page_problems(api_dir: Path, doors: tuple[DoorSpec, ...] = DOORS) -> list[str]:
    """One message per door with no page in the API reference ``api_dir``: an
    ``.. automodule:: <door>`` line (inside ``{eval-rst}``) or a MyST
    ``{automodule} <door>`` fence in one of its ``.md`` files."""
    pages = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(api_dir.glob("*.md"))
    )

    def documented(module: str) -> bool:
        m = re.escape(module)
        pattern = rf"^\s*(\.\. automodule::|[`:]{{3,}}\{{automodule\}})\s+{m}\s*$"
        return re.search(pattern, pages, re.MULTILINE) is not None

    return [
        f"{spec.module} has no API page (an `.. automodule:: {spec.module}` in {api_dir})"
        for spec in doors
        if not documented(spec.module)
    ]


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
    out: list[str] = []
    for spec in doors:
        try:
            door = importlib.import_module(spec.module)
        except ImportError as e:
            out.append(f"{spec.module}: cannot import ({e})")
            continue
        cls = getattr(door, spec.template, None)
        if cls is not None:
            behind[f"{cls.__module__}.{cls.__qualname__}"] = spec.module
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

    try:
        door = importlib.import_module(spec.module)
    except ImportError as e:
        return [f"{spec.module}: cannot import ({e})"]
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
