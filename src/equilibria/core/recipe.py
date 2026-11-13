"""The recipe (ADR-0011): the periods, their kinds, and what to do in each kind.

A recipe lists the periods in order, each with a kind. For each kind it says the
steps before and after the solve, the solve strategy and the solver. It builds
and solves nothing itself: the driver flies it.

The core ships four kinds: ``calibrate`` (GTAP ``base``), ``replicate`` (GTAP
``check``), ``shock`` and ``step`` (one year of a recursive dynamic model). A
recipe may add a kind of its own only when none fits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

PERIOD_KINDS: tuple[str, ...] = ("calibrate", "replicate", "shock", "step")


@dataclass(frozen=True)
class Period:
    name: str
    kind: str


@dataclass
class StepContext:
    """What a step sees: the model, the template (and its data), the recipe, the
    current period and the prior one, and the exercise's shock."""

    model: Any
    template: Any
    recipe: Recipe
    period: Period
    prior: Period | None
    shock: Mapping[str, Any] = field(default_factory=dict)


# A step mutates the model in place and returns nothing; to report, it logs.
Step = Callable[[StepContext], None]
# A solver solves the model as it stands and returns a status ("optimal" when
# it converged).
Solver = Callable[[Any], str]
# A solve strategy wraps the solve (in one go, in steps, Johansen...).
Strategy = Callable[[Any, Solver], str]


def solve_once(model: Any, solver: Solver) -> str:
    """The default strategy: one call to the solver."""
    return solver(model)


@dataclass(frozen=True)
class KindPlan:
    """What the driver does in every period of one kind. ``solver=None``: the
    period is not solved (its residual is still measured)."""

    before: tuple[Step, ...] = ()
    after: tuple[Step, ...] = ()
    solver: Solver | None = None
    strategy: Strategy = solve_once


@dataclass(frozen=True)
class Recipe:
    periods: tuple[Period, ...]
    plans: Mapping[str, KindPlan]

    def __post_init__(self) -> None:
        if not self.periods:
            raise ValueError("a recipe needs at least one period")
        names = [p.name for p in self.periods]
        if len(set(names)) != len(names):
            raise ValueError(f"period names must be unique, got {names}")
        missing = sorted({p.kind for p in self.periods} - set(self.plans))
        if missing:
            raise ValueError(f"no plan for period kind(s) {missing}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.periods)

    def plan(self, period: Period) -> KindPlan:
        return self.plans[period.kind]

    def prior(self, period: Period) -> Period | None:
        i = self.periods.index(period)
        return self.periods[i - 1] if i else None

    def has_kind(self, kind: str) -> bool:
        return any(p.kind == kind for p in self.periods)
