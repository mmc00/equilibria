"""PEP-focused convenience wrapper around the generic `Simulator` API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.simulations import presets
from equilibria.simulations.simulator import Simulator


class PepSimulator(Simulator):
    """Convenience simulator with model fixed to ``pep``."""

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="pep", **model_options)

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
        return self.run_scenarios(
            scenarios=[presets.export_tax(multiplier=multiplier)],
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
        return self.run_scenarios(
            scenarios=[presets.import_price(commodity=commodity, multiplier=multiplier)],
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
        return self.run_scenarios(
            scenarios=[presets.import_shock(multiplier=multiplier)],
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
        return self.run_scenarios(
            scenarios=[presets.government_spending(multiplier=multiplier)],
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )
