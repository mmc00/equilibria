"""GTAP's run modules talk to the model only through GTAPPeriodModel (ADR-0011):
no Pyomo, no backend, no old driver, so the family/JAX model flies the same
steps."""

from __future__ import annotations

import ast
import pathlib

import pytest

GTAP = pathlib.Path(__file__).resolve().parents[3] / "src/equilibria/templates/gtap"
RUN_MODULES = ("period_model.py", "period_rules.py", "steps.py", "strategies.py")
BANNED = ("pyomo", "equilibria.backends", "equilibria.solver")
OLD = ("gtap_multiperiod_driver", "period_prep", "pyomo_periods", "gtap_block_model")


def _imports(path: pathlib.Path) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


@pytest.mark.parametrize("module", RUN_MODULES)
def test_gtap_run_module_imports_no_backend(module: str) -> None:
    f = GTAP / module
    assert f.exists(), f
    bad = sorted(
        m
        for m in _imports(f)
        if m == "pyomo" or m.startswith(BANNED) or m.rsplit(".", 1)[-1] in OLD
    )
    assert bad == [], f"templates/gtap/{module}: {', '.join(bad)}"
