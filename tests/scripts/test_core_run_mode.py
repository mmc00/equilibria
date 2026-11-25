"""solve_on_core takes no default mode (MAR-166).

The old driver defaulted to altertax and solve_on_core to gtap: a call that
leaves mode out would switch models without a word. Every caller names it.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mode_has_no_default(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "gtap"))
    sys.modules.pop("_core_run", None)
    from _core_run import solve_on_core  # ty: ignore[unresolved-import]

    mode = inspect.signature(solve_on_core).parameters["mode"]
    assert mode.kind is inspect.Parameter.KEYWORD_ONLY
    assert mode.default is inspect.Parameter.empty


def test_every_caller_names_the_mode():
    calls = []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "solve_on_core"
            ):
                calls.append((path.name, {k.arg for k in node.keywords}))
    assert calls, "no solve_on_core call found"
    assert [name for name, kws in calls if "mode" not in kws] == []
