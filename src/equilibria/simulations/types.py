"""Shared simulation API types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ShockOp = Literal["set", "scale", "add"]


class Shock(BaseModel):
    """A single policy shock instruction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    var: str
    op: ShockOp
    values: float | dict[str, float]

    @field_validator("var")
    @classmethod
    def _validate_var(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("Shock.var must be non-empty.")
        return v


class Scenario(BaseModel):
    """A named scenario containing one or more shocks.

    For PEP, ``closure`` may be provided as a plain mapping, for example::

        Scenario(
            name="government_spending",
            shocks=[Shock(var="G", op="scale", values=1.2)],
            closure={
                "fixed": ["G", "CAB", "KS", "LS", "PWM", "PWX", "CMIN", "VSTK", "TR_SELF"],
                "endogenous": ["IT", "SH", "SF", "SG", "SROW"],
                "numeraire": "e",
                "capital_mobility": "mobile",
            },
        )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    shocks: list[Shock]
    reference_slice: str = "sim1"
    closure: dict[str, object] | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("Scenario.name must be non-empty.")
        return v

    @field_validator("reference_slice")
    @classmethod
    def _normalize_reference_slice(cls, value: str) -> str:
        v = value.strip().lower()
        if not v:
            raise ValueError("Scenario.reference_slice must be non-empty.")
        return v

    @field_validator("closure", mode="before")
    @classmethod
    def _normalize_optional_closure(cls, value: object) -> dict[str, object] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("Scenario.closure must be a mapping when provided.")
        normalized = dict(value)
        return normalized or None


class ShockDefinition(BaseModel):
    """Model-specific metadata for one shockable variable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    var: str
    kind: Literal["scalar", "indexed"]
    domain: str | None
    members: tuple[str, ...] | None = None
    ops: tuple[ShockOp, ...]
    description: str
