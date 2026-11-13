"""The exercise (ADR-0011): one run of a template under a recipe.

An exercise carries its shock and its ``@overwrite`` blocks; nothing is
registered globally, so nothing leaks into the next exercise (#214). Each run
asks the template for a fresh model, never a shared one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from equilibria.core.driver import fly
from equilibria.core.recipe import Recipe


class Template(Protocol):
    def build(self, periods: Sequence[str], overwrite: Sequence[Any] = ()) -> Any:
        """A fresh model with ``periods``, never shared with another call."""
        ...

    def default_recipe(self) -> Recipe: ...


class Exercise:
    """``Exercise(template, recipe, shock, overwrite).run()``."""

    def __init__(
        self,
        template: Template,
        recipe: Recipe | None = None,
        shock: Mapping[str, Any] | None = None,
        overwrite: Sequence[Any] = (),
    ) -> None:
        self.template = template
        self.recipe = recipe or template.default_recipe()
        self.shock = dict(shock or {})
        self.overwrite = tuple(overwrite)
        if self.shock and not self.recipe.has_kind("shock"):
            raise ValueError(
                f"a shock was given but the recipe has no shock period "
                f"(periods: {', '.join(self.recipe.names)})"
            )
        self.model: Any = None

    def run(self) -> dict[str, dict[str, Any]]:
        """Build a fresh model and fly the recipe on it; returns
        ``{period: {"code", "residual"}}``. The model stays on ``self.model``."""
        self.model = self.template.build(self.recipe.names, self.overwrite)
        return fly(self.model, self.recipe, template=self.template, shock=self.shock)
