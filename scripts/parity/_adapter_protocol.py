"""ParityAdapter Protocol + per-template registry.

A ParityAdapter is a thin wrapper around a template's existing parity pipeline
that exposes a uniform interface so the triage CLI can drive any template
generically. Each template ships exactly one adapter class.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ParityAdapter(Protocol):
    """Uniform interface for driving Python↔GAMS parity across templates."""

    def build_solved_model(self, dataset: str, scenario: str) -> Any:
        ...

    def load_gams_reference(self, gdx_path: Path) -> dict[str, dict[tuple, float]]:
        ...

    def find_py_var(self, model: Any, gams_name: str) -> tuple[Any | None, str | None]:
        ...

    def name_aliases(self) -> dict[str, str]:
        ...

    def enumerate_combinations(self) -> list[tuple[str, str]]:
        ...


class AdapterRegistry:
    """Module-level registry mapping template name → adapter class."""

    _IMPORT_PATHS: dict[str, str] = {
        "gtap": "equilibria.templates.gtap.parity_adapter:GTAPParityAdapter",
    }

    @classmethod
    def get(cls, template_name: str) -> type:
        if template_name not in cls._IMPORT_PATHS:
            raise KeyError(
                f"Unknown template '{template_name}'. "
                f"Known: {sorted(cls._IMPORT_PATHS)}"
            )
        spec = cls._IMPORT_PATHS[template_name]
        module_path, class_name = spec.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def list_templates(cls) -> list[str]:
        return sorted(cls._IMPORT_PATHS)
