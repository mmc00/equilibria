"""The exercise (ADR-0011): one run of a template under a recipe.

An exercise carries its shock and its ``@overwrite`` blocks; nothing is
registered globally, so nothing leaks into the next exercise (#214). Each run
asks the template for a fresh model, never a shared one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from equilibria.core.driver import fly
from equilibria.core.period_model import PeriodModel
from equilibria.core.recipe import Recipe


class Template(Protocol):
    def build(
        self, periods: Sequence[str], overwrite: Sequence[Any] = ()
    ) -> PeriodModel:
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
        self.model: PeriodModel | None = None
        # the last run's {period: {"code", "residual"}}; None until a run finishes
        self.codes: dict[str, dict[str, Any]] | None = None

    def run(self) -> dict[str, dict[str, Any]]:
        """Build a fresh model and fly the recipe on it; returns
        ``{period: {"code", "residual"}}``. The model stays on ``self.model``.

        A template may define ``run_scope()``, a context manager entered after
        the build and held over every period (GTAP freezes its built model for
        the garbage collector there, once per run). The codes stay on
        ``self.codes`` once the run finishes (``None`` while it runs or if it raised)."""
        self.codes = None
        model = self.template.build(self.recipe.names, self.overwrite)
        self.model = model
        scope = getattr(self.template, "run_scope", None)
        with scope() if scope is not None else contextlib.nullcontext():
            codes = fly(model, self.recipe, template=self.template, shock=self.shock)
        self.codes = codes
        return codes
