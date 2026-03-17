"""PEP-focused convenience wrapper around the generic `Simulator` API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.simulations import presets
from equilibria.simulations.simulator import Simulator
from equilibria.simulations.types import Scenario


class PepSimulator(Simulator):
    """Convenience simulator with model fixed to ``pep``.

    Standard wrapper methods such as :meth:`run_export_tax` use the simulator's
    base contract/config. If one scenario needs a custom closure, use
    :class:`equilibria.simulations.types.Scenario` directly and pass a
    ``closure={...}`` block through :meth:`run_scenarios`.
    """

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="pep", **model_options)

    @staticmethod
    def available_presets() -> tuple[str, ...]:
        """Return built-in PEP preset names."""
        return presets.available_presets()

    @staticmethod
    def make_preset(name: str, **kwargs: float | str) -> Scenario:
        """Build one preset scenario by name."""
        return presets.make_preset(name, **kwargs)

    def run_preset(
        self,
        name: str,
        *,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
        **preset_kwargs: float | str,
    ) -> dict[str, Any]:
        """Run one preset scenario selected by name."""
        scenario = presets.make_preset(name, **preset_kwargs)
        return self.run_scenarios(
            scenarios=[scenario],
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_export_tax(
        self,
        *,
        multiplier: float = 0.75,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        """Run one standard export-tax scenario."""
        return self.run_preset(
            "export_tax",
            multiplier=multiplier,
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_import_price(
        self,
        *,
        commodity: str = "agr",
        multiplier: float = 1.25,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        """Run one-commodity import-price scenario."""
        return self.run_preset(
            "import_price",
            commodity=commodity,
            multiplier=multiplier,
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_import_shock(
        self,
        *,
        multiplier: float = 1.25,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        """Run all-commodities import-price scenario."""
        return self.run_preset(
            "import_shock",
            multiplier=multiplier,
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_government_spending(
        self,
        *,
        multiplier: float = 1.2,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        """Run one government-spending scenario."""
        return self.run_preset(
            "government_spending",
            multiplier=multiplier,
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )
