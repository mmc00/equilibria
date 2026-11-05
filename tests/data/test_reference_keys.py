"""Every GEMPACK matrix record is asked for by shock, steps and closure (phase 3b).

``reference()`` raises when a request matches more than one folder, but a request
that leaves a key out can still match exactly one folder today and silently read
the wrong variant tomorrow, when a sibling record is added. So a call for a
GEMPACK-side record (tool ``gempack``, ``sltoht`` or ``gemsim``) names all three
matrix keys, unless it asks for a Burfisher record (``experiment=`` or ``method=``:
a book/Gragg run, or the welfare computed from them). A call that passes
``**variant`` (a record's own variant) is not checked.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_GEMPACK_TOOLS = {"gempack", "sltoht", "gemsim"}
_MATRIX_KEYS = {"shock", "steps", "closure"}


def _incomplete_calls(source: str) -> list[int]:
    """Lines of ``reference(...)``/``find_reference(...)`` calls missing a matrix key."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("reference", "find_reference") or len(node.args) < 2:
            continue
        tool = node.args[1]
        if isinstance(tool, ast.Constant) and tool.value not in _GEMPACK_TOOLS:
            continue
        keys = {k.arg for k in node.keywords}
        if None in keys:  # **variant: a record's own variant, not checkable here
            continue
        if not keys & {"experiment", "method"} and not keys >= _MATRIX_KEYS:
            bad.append(node.lineno)
    return bad


def test_scanner_flags_a_short_matrix_request() -> None:
    src = (
        'reference("gtap7_3x3", "gempack", shock=10, closure="capFix")\n'
        'reference("gtap7_3x3", "gempack", shock=10, steps="8-16-32", closure="capFix")\n'
        'reference("nus333", "gemsim", experiment="me9c", method="gragg")\n'
        'reference("nus333", "gempack", method="gragg", output="welfare")\n'
        'reference("gtap7_3x3", "gams", form="mcp")\n'
        "find_reference(ds, tool, **variant)\n"
    )
    assert _incomplete_calls(src) == [1]


def test_reference_keys_are_complete() -> None:
    files = subprocess.run(
        ["git", "ls-files", "tests/*.py", "scripts/*.py", "examples/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hits = [
        f"{rel}:{n}"
        for rel in files
        if rel != "tests/data/test_reference_keys.py"
        for n in _incomplete_calls((ROOT / rel).read_text())
    ]
    assert not hits, "name shock, steps and closure: " + ", ".join(hits)
