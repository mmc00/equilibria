"""Calibration phase definitions and registry for equilibria CGE framework.

This module provides the CalibrationPhase enum, phase registry, and
dependency validation for ordered calibration of CGE models.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from equilibria.blocks.base import Block


class CalibrationPhase(Enum):
    """Ordered calibration phases for CGE models.

        Phases are ordered such that blocks in earlier phases must be
    calibrated before blocks in later phases can access their data.

        Standard phases:
            SETS: Detect and validate sets from SAM
            PRODUCTION: Production parameters (CES shares, IO coefficients)
            TRADE: Trade parameters (import shares, export shares)
            INSTITUTIONS: Income distribution, tax rates, transfers
            DEMAND: Consumer demand parameters
            EQUILIBRIUM: Market clearing, price normalization

        Custom phases can be registered with register_phase().
    """

    SETS = (1, "sets", "Detect and validate sets from SAM")
    PRODUCTION = (
        2,
        "production",
        "Production parameters (CES shares, IO coefficients)",
    )
    TRADE = (3, "trade", "Trade parameters (import shares, export shares)")
    INSTITUTIONS = (4, "institutions", "Income distribution, tax rates, transfers")
    DEMAND = (5, "demand", "Consumer demand parameters")
    EQUILIBRIUM = (6, "equilibrium", "Market clearing, price normalization")

    def __init__(self, order: int, key: str, description: str):
        self._order = order
        self._key = key
        self._description = description

    @property
    def order(self) -> int:
        """Return the numeric order of this phase."""
        return self._order

    @property
    def key(self) -> str:
        """Return the string key of this phase."""
        return self._key

    @property
    def description(self) -> str:
        """Return the description of this phase."""
        return self._description

    def __lt__(self, other: CalibrationPhase) -> bool:
        """Enable comparison of phases by order."""
        if not isinstance(other, CalibrationPhase):
            return NotImplemented
        return self._order < other._order

    def __le__(self, other: CalibrationPhase) -> bool:
        """Enable comparison of phases by order."""
        if not isinstance(other, CalibrationPhase):
            return NotImplemented
        return self._order <= other._order

    def __gt__(self, other: CalibrationPhase) -> bool:
        """Enable comparison of phases by order."""
        if not isinstance(other, CalibrationPhase):
            return NotImplemented
        return self._order > other._order

    def __ge__(self, other: CalibrationPhase) -> bool:
        """Enable comparison of phases by order."""
        if not isinstance(other, CalibrationPhase):
            return NotImplemented
        return self._order >= other._order


class PhaseRegistry:
    """Registry for custom calibration phases.

    Allows blocks to register custom calibration phases that fit
    between the standard phases.

    Example:
        >>> # Register custom phase between PRODUCTION and TRADE
        >>> custom_phase = PhaseRegistry.register("CUSTOM", 2.5)
        >>> custom_phase.order
        2.5
    """

    _custom_phases: dict[str, CalibrationPhase] = {}
    _next_order: float = 7.0  # Start after standard phases

    @classmethod
    def register(cls, name: str, order: float | None = None) -> CalibrationPhase:
        """Register a new calibration phase.

        Args:
            name: Unique name for the phase
            order: Numeric order (if None, assigned automatically after existing phases)

        Returns:
            The registered CalibrationPhase

        Raises:
            ValueError: If phase name already exists
        """
        if name.upper() in [p.name for p in CalibrationPhase]:
            raise ValueError(f"Phase '{name}' conflicts with standard phase")

        if name in cls._custom_phases:
            raise ValueError(f"Custom phase '{name}' already registered")

        if order is None:
            order = cls._next_order
            cls._next_order += 1.0

        # Create a new enum-like object
        phase = _CustomPhase(order, name.lower(), f"Custom phase: {name}")
        cls._custom_phases[name] = phase
        return phase

    @classmethod
    def get(cls, name: str) -> CalibrationPhase | None:
        """Get a registered phase by name."""
        # Check standard phases first
        try:
            return CalibrationPhase[name.upper()]
        except KeyError:
            pass

        # Check custom phases
        return cls._custom_phases.get(name)

    @classmethod
    def list_phases(cls) -> list[CalibrationPhase]:
        """List all phases (standard and custom) in order."""
        all_phases = list(CalibrationPhase) + list(cls._custom_phases.values())
        return sorted(all_phases, key=lambda p: p.order)


class _CustomPhase:
    """Custom calibration phase (not an enum member)."""

    def __init__(self, order: float, key: str, description: str):
        self._order = order
        self._key = key
        self._description = description
        self._name = key.upper()

    @property
    def order(self) -> float:
        return self._order

    @property
    def key(self) -> str:
        return self._key

    @property
    def description(self) -> str:
        return self._description

    @property
    def name(self) -> str:
        return self._name

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (CalibrationPhase, _CustomPhase)):
            return self._order < other.order
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, (CalibrationPhase, _CustomPhase)):
            return self._order <= other.order
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (CalibrationPhase, _CustomPhase)):
            return self._order > other.order
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (CalibrationPhase, _CustomPhase)):
            return self._order >= other.order
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (CalibrationPhase, _CustomPhase)):
            return self._order == other.order
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._order)


class CalibrationDependencyError(Exception):
    """Raised when calibration dependencies are violated."""

    pass


class DependencyValidator:
    """Validates calibration dependencies between blocks.

    Ensures that blocks in later phases only access data from
    blocks in earlier phases.
    """

    def __init__(self):
        self._calibrated_blocks: dict[str, CalibrationPhase] = {}

    def register_calibration(self, block: Block, phase: CalibrationPhase) -> None:
        """Register that a block has been calibrated in a phase.

        Args:
            block: The block that was calibrated
            phase: The phase in which it was calibrated

        Raises:
            CalibrationDependencyError: If block already calibrated
        """
        block_name = block.name
        if block_name in self._calibrated_blocks:
            raise CalibrationDependencyError(
                f"Block '{block_name}' already calibrated in phase "
                f"{self._calibrated_blocks[block_name].name}"
            )
        self._calibrated_blocks[block_name] = phase

    def validate_access(
        self,
        accessor_block: Block,
        accessor_phase: CalibrationPhase,
        target_block_name: str,
    ) -> None:
        """Validate that accessor block can read from target block.

        Args:
            accessor_block: The block trying to access data
            accessor_phase: The phase of the accessor block
            target_block_name: The name of the block being accessed

        Raises:
            CalibrationDependencyError: If dependency violated
        """
        if target_block_name not in self._calibrated_blocks:
            raise CalibrationDependencyError(
                f"Block '{accessor_block.name}' in phase '{accessor_phase.name}' "
                f"cannot access block '{target_block_name}' - not yet calibrated"
            )

        target_phase = self._calibrated_blocks[target_block_name]
        if target_phase >= accessor_phase:
            raise CalibrationDependencyError(
                f"Block '{accessor_block.name}' in phase '{accessor_phase.name}' "
                f"cannot access block '{target_block_name}' in phase '{target_phase.name}' - "
                f"must be calibrated in earlier phase"
            )

    def get_calibrated_before(self, phase: CalibrationPhase) -> list[str]:
        """Get list of blocks calibrated before given phase.

        Args:
            phase: The phase to check

        Returns:
            List of block names calibrated in earlier phases
        """
        return [
            name
            for name, cal_phase in self._calibrated_blocks.items()
            if cal_phase < phase
        ]

    def reset(self) -> None:
        """Reset the validator for a new calibration run."""
        self._calibrated_blocks.clear()


# Convenience functions for phase registration
def register_calibration_phase(
    name: str, order: float | None = None
) -> CalibrationPhase:
    """Register a custom calibration phase.

    Args:
        name: Unique name for the phase
        order: Numeric order (auto-assigned if None)

    Returns:
        The registered phase
    """
    return PhaseRegistry.register(name, order)


def get_calibration_phase(name: str) -> CalibrationPhase | None:
    """Get a calibration phase by name.

    Args:
        name: Name of the phase

    Returns:
        The phase if found, None otherwise
    """
    return PhaseRegistry.get(name)


def list_calibration_phases() -> list[CalibrationPhase]:
    """List all calibration phases in order."""
    return PhaseRegistry.list_phases()
