"""
GDX symbol definitions using Pydantic models.

Represents GAMS data structures: Sets, Parameters, Variables, Equations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    """GAMS symbol types."""

    set = "set"
    parameter = "parameter"
    variable = "variable"
    equation = "equation"
    alias = "alias"


class SymbolBase(BaseModel):
    """Base class for all GDX symbols."""

    model_config = {"use_enum_values": True}

    name: str = Field(..., description="Symbol name")
    sym_type: SymbolType = Field(..., description="Symbol type")
    dimensions: int = Field(0, ge=0, le=20, description="Number of dimensions")
    description: str = Field("", description="Symbol description/text")
    domain: list[str] = Field(default_factory=list, description="Domain sets")


class Set(SymbolBase):
    """
    GAMS Set symbol.

    Example:
        >>> s = Set(
        ...     name="i",
        ...     sym_type="set",
        ...     dimensions=1,
        ...     description="Industries",
        ...     domain=["*"],
        ...     elements=[["agr"], ["mfg"], ["srv"]]
        ... )
    """

    sym_type: SymbolType = SymbolType.set
    elements: list[list[str]] = Field(
        default_factory=list,
        description="Set elements (each element is a list of keys)"
    )


class Parameter(SymbolBase):
    """
    GAMS Parameter symbol.

    Records are tuples of (keys, value) where keys is a list of domain elements.

    Example:
        >>> p = Parameter(
        ...     name="price",
        ...     sym_type="parameter",
        ...     dimensions=1,
        ...     description="Commodity prices",
        ...     domain=["i"],
        ...     records=[(["agr"], 1.0), (["mfg"], 1.5)]
        ... )
    """

    sym_type: SymbolType = SymbolType.parameter
    records: list[tuple[list[str], float]] = Field(
        default_factory=list,
        description="Parameter records: (keys, value)"
    )


class Variable(SymbolBase):
    """
    GAMS Variable symbol.

    Each record contains 5 values: (level, marginal, lower, upper, scale).

    Example:
        >>> v = Variable(
        ...     name="X",
        ...     sym_type="variable",
        ...     dimensions=1,
        ...     description="Output",
        ...     domain=["j"],
        ...     records=[(["agr"], (100.0, 0.0, 0.0, float("inf"), 1.0))]
        ... )
    """

    sym_type: SymbolType = SymbolType.variable
    records: list[tuple[list[str], tuple[float, float, float, float, float]]] = Field(
        default_factory=list,
        description="Variable records: (keys, (level, marginal, lower, upper, scale))"
    )


class Equation(SymbolBase):
    """
    GAMS Equation symbol.

    Each record contains 5 values: (level, marginal, lower, upper, scale).
    Same structure as Variable.
    """

    sym_type: SymbolType = SymbolType.equation
    records: list[tuple[list[str], tuple[float, float, float, float, float]]] = Field(
        default_factory=list,
        description="Equation records: (keys, (level, marginal, lower, upper, scale))"
    )
