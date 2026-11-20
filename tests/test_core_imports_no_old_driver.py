"""No module of the package imports the old GTAP drivers (gtap-on-core PR 4).

Every ``src`` module other than the four old ones is parsed: none imports them,
by ``import``, ``from ... import`` or a helper that returns the module (the old
``_drv()`` pattern of ``pyomo_periods``). The name guard
(``tests/test_no_old_driver_names.py``) checks every file for the names; this one
checks the package's import graph, which must be clean before the files go.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
OLD = {
    "equilibria.templates.gtap.gtap_multiperiod_driver",
    "equilibria.templates.gtap.period_prep",
    "equilibria.family.driver",
    "equilibria.family.prep_recipe",
}


def _module(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts).removesuffix(
        ".__init__"
    )


def _imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_no_package_module_imports_an_old_driver() -> None:
    bad = {}
    for f in SRC.rglob("*.py"):
        mod = _module(f)
        if mod in OLD:
            continue
        hit = _imports(ast.parse(f.read_text())) & OLD
        if hit:
            bad[mod] = sorted(hit)
    assert bad == {}


def test_the_scan_sees_an_import() -> None:
    tree = ast.parse(
        "def _drv():\n"
        "    from equilibria.templates.gtap import gtap_multiperiod_driver\n"
        "    return gtap_multiperiod_driver\n"
        "from equilibria.family import driver\n"
    )
    assert _imports(tree) & OLD == {
        "equilibria.templates.gtap.gtap_multiperiod_driver",
        "equilibria.family.driver",
    }
