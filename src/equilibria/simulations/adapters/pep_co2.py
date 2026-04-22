"""PEP-CRI + CO2 adapter for the generic scenario simulation API.

Extends PepCRIAdapter (cross-border labor) with a sector CO2 tax layer.

Usage:
    from equilibria.simulations.adapters.pep_co2 import PepCO2Adapter

    adapter = PepCO2Adapter(
        sam_file="sam_cri_pep_icio36.xlsx",
        val_par_file="VAL_PAR.xlsx",
        dynamic_sets=True,
        co2_intensity={"agr": 0.05, "ind": 0.15, "ser": 0.03, "adm": 0.01},
        tco2b={"agr": 0.0, "ind": 0.0, "ser": 0.0, "adm": 0.0},
        tco2scal=1.0,
    )
    state = adapter.fit_base_state()
    # apply CO2 shock and solve ...
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from equilibria.simulations.adapters.pep_cri import PepCRIAdapter
from equilibria.simulations.pep_compare import compare_with_gams
from equilibria.simulations.pep_compare import key_indicators as pep_key_indicators
from equilibria.simulations.types import Scenario, Shock, ShockDefinition
from equilibria.templates.pep_co2_data import (
    get_state_co2_block,
    normalize_co2_inputs,
    set_state_co2_block,
    validate_and_fill_co2_block,
)
from equilibria.templates.pep_co2_model_solver import PEPCO2ModelSolver


class PepCO2Adapter(PepCRIAdapter):
    """PEP-CRI adapter with an extra sector-CO2 policy layer.

    Inherits from PepCRIAdapter so cross-border labor (L→ROW) is handled
    automatically.  Adds CO2 intensity, tax base and global scale factor
    on top of the standard CRI calibration.
    """

    def __init__(
        self,
        *,
        co2_data: Any | None = None,
        co2_intensity: Mapping[str, float] | None = None,
        tco2b: Mapping[str, float] | None = None,
        tco2scal: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._co2_inputs = normalize_co2_inputs(
            co2_data=co2_data,
            co2_intensity=co2_intensity,
            tco2b=tco2b,
            tco2scal=tco2scal,
        )

    def capabilities(self) -> dict[str, Any]:
        capabilities = super().capabilities()
        capabilities.update({"co2_extension": True})
        return capabilities

    def fit_base_state(self) -> Any:
        """Calibrate CRI state and attach CO2 block."""
        state = super().fit_base_state()
        block = validate_and_fill_co2_block(self._co2_inputs, state.sets.get("J", []))
        set_state_co2_block(state, block)
        return state

    def available_shocks(self) -> list[ShockDefinition]:
        catalog = list(super().available_shocks())
        j_members = tuple(self._sets.get("J", [])) or None
        catalog.extend(
            [
                ShockDefinition(
                    var="tco2b",
                    kind="indexed",
                    domain="J",
                    members=j_members,
                    ops=("set", "scale", "add"),
                    description="Base carbon-tax rate by activity/sector.",
                ),
                ShockDefinition(
                    var="co2_intensity",
                    kind="indexed",
                    domain="J",
                    members=j_members,
                    ops=("set", "scale", "add"),
                    description="CO2 intensity by activity/sector.",
                ),
                ShockDefinition(
                    var="tco2scal",
                    kind="scalar",
                    domain=None,
                    members=None,
                    ops=("set", "scale", "add"),
                    description="Global multiplier applied to all sector carbon-tax rates.",
                ),
            ]
        )
        return catalog

    def apply_shock(self, state: Any, shock: Shock) -> None:
        var = shock.var.strip().lower()
        op = shock.op.strip().lower()
        if var == "tco2scal":
            block = get_state_co2_block(state)
            block["tco2scal"] = self._apply_scalar_op(block.get("tco2scal", 1.0), op, shock.values)
            return

        if var in {"tco2b", "co2_intensity"}:
            block = get_state_co2_block(state)
            current = block.get(var, {})
            updated = self._apply_indexed_op(
                current,
                op=op,
                values=shock.values,
                domain=set(state.sets.get("J", [])),
                var_name=var,
            )
            block[var] = updated
            return

        super().apply_shock(state, shock)

    def solve_state(
        self,
        state: Any,
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        scenario: Scenario | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Solve using PEPCO2ModelSolver (CRI + CO2 equations)."""
        effective_contract = self._resolve_contract_for_scenario(scenario)
        solver = PEPCO2ModelSolver(
            calibrated_state=state,
            tolerance=self.solve_tolerance,
            max_iterations=self.max_iterations,
            init_mode=self.init_mode,
            contract=effective_contract,
            config=self.runtime_config,
            gams_results_gdx=reference_results_gdx,
            gams_results_slice=reference_slice.lower(),
            sam_file=self._runtime_sam_file,
            val_par_file=self.val_par_file,
            gdxdump_bin=self.gdxdump_bin,
            initial_vars=initial_vars,
        )
        solution = solver.solve(method=self.method)
        validation = solver.validate_solution(solution)
        return solver, solution, validation

    def compare_with_reference(
        self,
        *,
        solution_vars: Any,
        solution_params: dict[str, Any],
        reference_results_gdx: Path,
        reference_slice: str,
        abs_tol: float,
        rel_tol: float,
    ) -> dict[str, Any]:
        return compare_with_gams(
            solution_vars=solution_vars,
            solution_params=solution_params,
            gams_results_gdx=reference_results_gdx,
            gams_slice=reference_slice.lower(),
            gdxdump_bin=self.gdxdump_bin,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    def key_indicators(self, vars_obj: Any) -> dict[str, float]:
        indicators = pep_key_indicators(vars_obj)
        indicators.update(
            {
                "co2_total_emissions": float(getattr(vars_obj, "co2_total_emissions", 0.0)),
                "co2_total_tax": float(getattr(vars_obj, "co2_total_tax", 0.0)),
                "tco2scal": float(getattr(vars_obj, "tco2scal", 1.0)),
            }
        )
        return indicators
