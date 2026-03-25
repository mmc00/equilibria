"""Convenience simulator wrapper for the PEP+CO2 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.simulations.simulator import Simulator


class PepCO2Simulator(Simulator):
    """Convenience simulator with model fixed to ``pep_co2``."""

    def __init__(self, **model_options: Any) -> None:
        super().__init__(model="pep_co2", **model_options)

    def run_carbon_tax_scale(
        self,
        *,
        multiplier: float = 2.0,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        return self.run_shock(
            var="tco2scal",
            multiplier=multiplier,
            name="carbon_tax_scale",
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_sector_carbon_tax(
        self,
        *,
        sector: str,
        multiplier: float = 2.0,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        sector_key = sector.strip().lower()
        return self.run_shock(
            var="tco2b",
            index=sector_key,
            multiplier=multiplier,
            name=f"carbon_tax_{sector_key}",
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )
