"""Ready-to-use scenario presets for common policy experiments."""

from __future__ import annotations

import warnings
from collections.abc import Callable

from equilibria.simulations.types import Scenario, Shock


def export_tax(
    *,
    multiplier: float = 0.75,
    name: str = "export_tax",
    reference_slice: str = "sim1",
) -> Scenario:
    """Create a standard export-tax scenario: ``ttix(i) *= multiplier`` for all ``i``."""
    return Scenario(
        name=name,
        reference_slice=reference_slice,
        shocks=[Shock(var="ttix", op="scale", values={"*": float(multiplier)})],
    )


def import_price(
    *,
    commodity: str = "agr",
    multiplier: float = 1.25,
    name: str | None = None,
    reference_slice: str = "sim1",
) -> Scenario:
    """Create one-commodity import-price shock: ``PWM(commodity) *= multiplier``."""
    commodity_key = commodity.strip().lower()
    scenario_name = name if name is not None else f"import_price_{commodity_key}"
    return Scenario(
        name=scenario_name,
        reference_slice=reference_slice,
        shocks=[Shock(var="PWM", op="scale", values={commodity_key: float(multiplier)})],
    )


def import_shock(
    *,
    multiplier: float = 1.25,
    name: str = "import_shock",
    reference_slice: str = "sim1",
) -> Scenario:
    """Create all-commodities import-price shock: ``PWM(i) *= multiplier``."""
    return Scenario(
        name=name,
        reference_slice=reference_slice,
        shocks=[Shock(var="PWM", op="scale", values={"*": float(multiplier)})],
    )


def government_spending(
    *,
    multiplier: float = 1.2,
    name: str = "government_spending",
    reference_slice: str = "sim1",
) -> Scenario:
    """Create government spending shock: ``G *= multiplier``."""
    return Scenario(
        name=name,
        reference_slice=reference_slice,
        shocks=[Shock(var="G", op="scale", values=float(multiplier))],
    )


PRESET_BUILDERS: dict[str, Callable[..., Scenario]] = {
    "export_tax": export_tax,
    "import_price": import_price,
    "import_shock": import_shock,
    "government_spending": government_spending,
}


def _build_preset(name: str, **kwargs: float | str) -> Scenario:
    key = name.strip().lower()
    if key not in PRESET_BUILDERS:
        names = ", ".join(sorted(PRESET_BUILDERS))
        raise ValueError(f"Unknown preset '{name}'. Available: {names}")
    return PRESET_BUILDERS[key](**kwargs)


def _warn_deprecated_preset_api() -> None:
    warnings.warn(
        "`available_presets()` and `make_preset()` are deprecated. "
        "Use `Simulator.shock()` or construct `Scenario`/`Shock` directly.",
        DeprecationWarning,
        stacklevel=2,
    )


def available_presets() -> tuple[str, ...]:
    """Return available built-in scenario preset names."""
    _warn_deprecated_preset_api()
    return tuple(PRESET_BUILDERS.keys())


def make_preset(name: str, **kwargs: float | str) -> Scenario:
    """Build one scenario preset by name."""
    _warn_deprecated_preset_api()
    return _build_preset(name, **kwargs)
