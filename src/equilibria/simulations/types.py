"""Shared simulation API types."""

from __future__ import annotations

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
    """A named scenario containing one or more shocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    shocks: list[Shock]
    reference_slice: str = "sim1"

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


class ShockDefinition(BaseModel):
    """Model-specific metadata for one shockable variable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    var: str
    kind: Literal["scalar", "indexed"]
    domain: str | None
    members: tuple[str, ...] | None = None
    ops: tuple[ShockOp, ...]
    description: str
