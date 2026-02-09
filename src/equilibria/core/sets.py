"""Set definitions and indexing for CGE models.

Sets define the indices used throughout the model (sectors, commodities,
factors, regions, etc.). They support multi-dimensional indexing and
subsets.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Set(BaseModel):
    """A set definition for model indexing.

    Sets are immutable collections of elements used to index parameters,
    variables, and equations. They support multi-dimensional indexing
    through the cartesian product of sets.

    Attributes:
        name: Unique identifier for the set
        elements: List of element names
        description: Human-readable description
        domain: Optional parent set (for subsets)

    Example:
        >>> sectors = Set(name="J", elements=["agr", "mfg", "svc"],
        ...               description="Production sectors")
        >>> print(sectors)
        Set J (3 elements): agr, mfg, svc
    """

    name: str = Field(..., min_length=1, description="Set identifier")
    elements: tuple[str, ...] = Field(default_factory=tuple, description="Set elements")
    description: str = Field(default="", description="Human-readable description")
    domain: str | None = Field(default=None, description="Parent set name for subsets")

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_subset(self) -> Set:
        """Validate that subset elements are in domain."""
        if self.domain is not None and self.elements:
            # Domain validation happens at SetManager level
            pass
        return self

    def __len__(self) -> int:
        """Return number of elements in the set."""
        return len(self.elements)

    def __iter__(self) -> Iterator[str]:
        """Iterate over set elements."""
        return iter(self.elements)

    def iter_elements(self) -> Iterator[str]:
        """Iterate over set elements."""
        return iter(self.elements)

    def __contains__(self, item: str) -> bool:
        """Check if element is in set."""
        return item in self.elements

    def __getitem__(self, index: int) -> str:
        """Get element by index."""
        return self.elements[index]

    def __repr__(self) -> str:
        """String representation of the set."""
        elems = ", ".join(self.elements[:5])
        if len(self.elements) > 5:
            elems += f", ... ({len(self.elements) - 5} more)"
        domain_str = f" subset of {self.domain}" if self.domain else ""
        return f"Set {self.name}{domain_str} ({len(self.elements)} elements): {elems}"

    def index(self, element: str) -> int:
        """Get index of element in set.

        Args:
            element: Element to find

        Returns:
            Index position (0-based)

        Raises:
            ValueError: If element not in set
        """
        try:
            return self.elements.index(element)
        except ValueError as exc:
            msg = f"Element '{element}' not in set '{self.name}'"
            raise ValueError(msg) from exc

    def to_list(self) -> list[str]:
        """Return elements as a list."""
        return list(self.elements)


class SetManager:
    """Manages all sets in a model.

    The SetManager provides centralized access to all sets, handles
    subset validation, and supports multi-dimensional indexing.

    Attributes:
        sets: Dictionary of set name to Set objects

    Example:
        >>> manager = SetManager()
        >>> manager.add(Set(name="J", elements=["agr", "mfg"]))
        >>> manager.add(Set(name="I", elements=["labor", "capital"]))
        >>> # Get cartesian product of two sets
        >>> list(manager.product("J", "I"))
        [('agr', 'labor'), ('agr', 'capital'), ('mfg', 'labor'), ('mfg', 'capital')]
    """

    def __init__(self) -> None:
        """Initialize empty set manager."""
        self._sets: dict[str, Set] = {}

    def add(self, set_obj: Set) -> None:
        """Add a set to the manager.

        Args:
            set_obj: Set to add

        Raises:
            ValueError: If set with same name already exists
        """
        if set_obj.name in self._sets:
            msg = f"Set '{set_obj.name}' already exists"
            raise ValueError(msg)

        # Validate subset domain exists
        if set_obj.domain is not None and set_obj.domain not in self._sets:
            msg = f"Domain set '{set_obj.domain}' not found for subset '{set_obj.name}'"
            raise ValueError(msg)

        # Validate subset elements are in domain
        if set_obj.domain is not None:
            domain_set = self._sets[set_obj.domain]
            invalid = [e for e in set_obj.elements if e not in domain_set]
            if invalid:
                msg = f"Elements {invalid} not in domain set '{set_obj.domain}'"
                raise ValueError(msg)

        self._sets[set_obj.name] = set_obj

    def get(self, name: str) -> Set:
        """Get a set by name.

        Args:
            name: Set name

        Returns:
            The Set object

        Raises:
            KeyError: If set not found
        """
        if name not in self._sets:
            msg = f"Set '{name}' not found"
            raise KeyError(msg)
        return self._sets[name]

    def __getitem__(self, name: str) -> Set:
        """Get set by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if set exists."""
        return name in self._sets

    def __iter__(self) -> Iterator[str]:
        """Iterate over set names."""
        return iter(self._sets.keys())

    def items(self) -> Iterator[tuple[str, Set]]:
        """Iterate over (name, set) pairs."""
        return iter(self._sets.items())

    def keys(self) -> Iterator[str]:
        """Iterate over set names."""
        return iter(self._sets.keys())

    def values(self) -> Iterator[Set]:
        """Iterate over sets."""
        return iter(self._sets.values())

    def product(self, *set_names: str) -> Iterator[tuple[str, ...]]:
        """Generate cartesian product of multiple sets.

        Args:
            *set_names: Names of sets to combine

        Yields:
            Tuples of element combinations

        Example:
            >>> manager.product("J", "I")
            yields ('agr', 'labor'), ('agr', 'capital'), ...
        """
        sets = [self.get(name) for name in set_names]

        def _product(sets_list: list[Set]) -> Iterator[tuple[str, ...]]:
            if not sets_list:
                yield ()
                return
            first, *rest = sets_list
            for elem in first.iter_elements():
                for combo in _product(rest):
                    yield (elem,) + combo

        yield from _product(sets)

    def validate_index(self, set_name: str, element: str) -> bool:
        """Validate that element exists in set.

        Args:
            set_name: Name of the set
            element: Element to validate

        Returns:
            True if valid

        Raises:
            KeyError: If set not found
            ValueError: If element not in set
        """
        set_obj = self.get(set_name)
        if element not in set_obj:
            msg = f"Element '{element}' not in set '{set_name}'"
            raise ValueError(msg)
        return True

    def list_sets(self) -> list[str]:
        """Return list of all set names."""
        return list(self._sets.keys())

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of all sets."""
        return {
            "total_sets": len(self._sets),
            "sets": {
                name: {
                    "elements": len(s),
                    "description": s.description,
                    "domain": s.domain,
                }
                for name, s in self._sets.items()
            },
        }
