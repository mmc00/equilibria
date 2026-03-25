"""High-level scenario simulation API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.simulations.adapters.base import BaseModelAdapter
from equilibria.simulations.adapters.gtap import GTAPAdapter
from equilibria.simulations.adapters.icio import ICIOAdapter
from equilibria.simulations.adapters.ieem import IEEMAdapter
from equilibria.simulations.adapters.pep import PepAdapter
from equilibria.simulations.adapters.pep_co2 import PepCO2Adapter
from equilibria.simulations.types import Scenario, Shock, ShockDefinition

_ADAPTER_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "pep": PepAdapter,
    "pep_co2": PepCO2Adapter,
    "ieem": IEEMAdapter,
    "gtap": GTAPAdapter,
    "icio": ICIOAdapter,
}


def register_adapter(model: str, adapter_cls: type[BaseModelAdapter]) -> None:
    """Register or replace one model adapter class."""
    key = model.strip().lower()
    if not key:
        raise ValueError("Model key must be non-empty.")
    _ADAPTER_REGISTRY[key] = adapter_cls


def available_models() -> tuple[str, ...]:
    """Return currently registered model keys."""
    return tuple(sorted(_ADAPTER_REGISTRY))


class Simulator:
    """Model-agnostic simulation runner with cached calibrated base state."""

    def __init__(self, *, model: str, **model_options: Any) -> None:
        model_key = model.strip().lower()
        if model_key not in _ADAPTER_REGISTRY:
            available = ", ".join(sorted(_ADAPTER_REGISTRY))
            raise ValueError(f"Unsupported model '{model}'. Available: {available}")

        self.model = model_key
        self.model_options = dict(model_options)
        self.adapter: BaseModelAdapter = _ADAPTER_REGISTRY[model_key](**model_options)
        self._base_state: Any | None = None

    def fit(self) -> Simulator:
        """Calibrate/build and cache base state for later scenario runs."""
        self._base_state = self.adapter.fit_base_state()
        return self

    def available_shocks(self) -> list[ShockDefinition]:
        """Return shock catalog exposed by the selected model adapter."""
        return self.adapter.available_shocks()

    def shock(
        self,
        *,
        var: str,
        index: str | None = None,
        multiplier: float | None = None,
        value: float | None = None,
        op: str = "scale",
        name: str | None = None,
        reference_slice: str = "sim1",
        closure: dict[str, object] | None = None,
    ) -> Scenario:
        """Build one scenario from a single low-level shock instruction.

        Use ``index="*"`` to target all members of an indexed variable.
        When the adapter exposes a shock catalog, this method validates
        ``var``, ``op`` and the selected ``index`` before constructing the
        scenario.
        """
        normalized_var = var.strip()
        if not normalized_var:
            raise ValueError("Shock variable must be non-empty.")

        normalized_op = op.strip().lower()
        if normalized_op not in {"set", "scale", "add"}:
            raise ValueError(f"Unsupported shock op '{op}'.")

        shock_value = self._resolve_shock_value(
            op=normalized_op,
            multiplier=multiplier,
            value=value,
        )
        definition = self._resolve_shock_definition(normalized_var)
        if definition is not None:
            allowed_ops = {item.strip().lower() for item in definition.ops}
            if normalized_op not in allowed_ops:
                choices = ", ".join(sorted(definition.ops))
                raise ValueError(
                    f"Variable '{definition.var}' does not support op '{op}'. Available: {choices}"
                )
        target_index = self._normalize_shock_index(index=index, definition=definition)
        scenario_name = name if name is not None else self._default_shock_name(
            var=normalized_var,
            index=target_index,
            op=normalized_op,
            value=shock_value,
        )

        values: float | dict[str, float]
        if definition is not None:
            normalized_var = definition.var

        if target_index is None:
            values = shock_value
        else:
            values = {target_index: shock_value}

        return Scenario(
            name=scenario_name,
            shocks=[Shock(var=normalized_var, op=normalized_op, values=values)],
            reference_slice=reference_slice,
            closure=closure,
        )

    def run_shock(
        self,
        *,
        var: str,
        index: str | None = None,
        multiplier: float | None = None,
        value: float | None = None,
        op: str = "scale",
        name: str | None = None,
        reference_slice: str = "sim1",
        closure: dict[str, object] | None = None,
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
    ) -> dict[str, Any]:
        """Build and run a single-shock scenario."""
        scenario = self.shock(
            var=var,
            index=index,
            multiplier=multiplier,
            value=value,
            op=op,
            name=name,
            reference_slice=reference_slice,
            closure=closure,
        )
        return self.run_scenarios(
            scenarios=[scenario],
            reference_results_gdx=reference_results_gdx,
            compare_abs_tol=compare_abs_tol,
            compare_rel_tol=compare_rel_tol,
            warm_start=warm_start,
            include_base=include_base,
        )

    def run_scenarios(
        self,
        *,
        scenarios: list[Scenario],
        reference_results_gdx: Path | str | None = None,
        compare_abs_tol: float = 1e-6,
        compare_rel_tol: float = 1e-6,
        warm_start: bool = True,
        include_base: bool = True,
        base_reference_slice: str = "base",
    ) -> dict[str, Any]:
        """Solve base + scenarios and optionally compare each run with a GDX reference."""
        if self._base_state is None:
            self.fit()
        assert self._base_state is not None

        self._validate_scenarios(scenarios)
        reference_gdx = (
            Path(reference_results_gdx)
            if reference_results_gdx is not None
            else None
        )

        report: dict[str, Any] = {
            "model": self.model,
            "model_options": self.model_options,
            "capabilities": self.adapter.capabilities(),
            "reference_results_gdx": str(reference_gdx) if reference_gdx else None,
            "compare_abs_tol": float(compare_abs_tol),
            "compare_rel_tol": float(compare_rel_tol),
            "warm_start": bool(warm_start),
            "base": None,
            "scenarios": [],
        }

        base_vars: Any | None = None
        last_converged_vars: Any | None = None
        if include_base:
            base_entry = self._solve_one(
                name="base",
                state=self.adapter.clone_state(self._base_state),
                initial_vars=None,
                reference_results_gdx=reference_gdx,
                reference_slice=str(base_reference_slice).strip().lower(),
                compare_abs_tol=compare_abs_tol,
                compare_rel_tol=compare_rel_tol,
                scenario=None,
            )
            report["base"] = base_entry
            base_vars = base_entry["solution_vars"]
            if bool(base_entry.get("solve", {}).get("converged")):
                last_converged_vars = base_vars

        initial_vars = last_converged_vars if warm_start else None
        for scenario in scenarios:
            scenario_state = self.adapter.clone_state(self._base_state)
            for shock in scenario.shocks:
                self.adapter.apply_shock(scenario_state, shock)

            scenario_entry = self._solve_one(
                name=scenario.name,
                state=scenario_state,
                initial_vars=initial_vars,
                reference_results_gdx=reference_gdx,
                reference_slice=scenario.reference_slice,
                compare_abs_tol=compare_abs_tol,
                compare_rel_tol=compare_rel_tol,
                scenario=scenario,
                shocks=scenario.shocks,
            )
            report["scenarios"].append(scenario_entry)
            if warm_start:
                if bool(scenario_entry.get("solve", {}).get("converged")):
                    last_converged_vars = scenario_entry["solution_vars"]
                initial_vars = last_converged_vars

        self._strip_internal_solution_refs(report)
        return report

    def _solve_one(
        self,
        *,
        name: str,
        state: Any,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
        compare_abs_tol: float,
        compare_rel_tol: float,
        scenario: Scenario | None = None,
        shocks: list[Shock] | None = None,
    ) -> dict[str, Any]:
        solver, solution, validation = self.adapter.solve_state(
            state,
            initial_vars=initial_vars,
            reference_results_gdx=reference_results_gdx,
            reference_slice=reference_slice,
            scenario=scenario,
        )
        comparison = None
        if reference_results_gdx is not None:
            comparison = self.adapter.compare_with_reference(
                solution_vars=solution.variables,
                solution_params=getattr(solver, "params", {}),
                reference_results_gdx=reference_results_gdx,
                reference_slice=reference_slice,
                abs_tol=compare_abs_tol,
                rel_tol=compare_rel_tol,
            )

        return {
            "name": name,
            "reference_slice": reference_slice,
            "shocks": [self._shock_to_dict(s) for s in (shocks or [])],
            "closure": (
                dict(scenario.closure)
                if scenario is not None and scenario.closure is not None
                else None
            ),
            "solve": {
                "converged": bool(solution.converged),
                "iterations": int(solution.iterations),
                "final_residual": float(solution.final_residual),
                "message": str(solution.message),
                "solver_stats": getattr(solution, "solver_stats", None),
                "key_indicators": self.adapter.key_indicators(solution.variables),
            },
            "validation": validation,
            "comparison": comparison,
            "solution_vars": solution.variables,
        }

    @staticmethod
    def _shock_to_dict(shock: Shock) -> dict[str, Any]:
        return {
            "var": shock.var,
            "op": shock.op,
            "values": shock.values,
        }

    @staticmethod
    def _strip_internal_solution_refs(report: dict[str, Any]) -> None:
        base = report.get("base")
        if isinstance(base, dict):
            base.pop("solution_vars", None)
        for scenario in report.get("scenarios", []):
            if isinstance(scenario, dict):
                scenario.pop("solution_vars", None)

    def _resolve_shock_definition(self, var: str) -> ShockDefinition | None:
        catalog = self.available_shocks()
        if not catalog:
            return None

        key = var.strip().lower()
        for definition in catalog:
            if definition.var.strip().lower() == key:
                return definition

        names = ", ".join(sorted(defn.var for defn in catalog))
        raise ValueError(
            f"Variable '{var}' is not shockable for model '{self.model}'. Available: {names}"
        )

    @staticmethod
    def _resolve_shock_value(
        *,
        op: str,
        multiplier: float | None,
        value: float | None,
    ) -> float:
        if multiplier is not None and value is not None:
            raise ValueError("Provide only one of `multiplier` or `value`.")

        if multiplier is not None:
            if op != "scale":
                raise ValueError("`multiplier` is only valid when `op='scale'`.")
            return float(multiplier)

        if value is not None:
            return float(value)

        if op == "scale":
            raise ValueError("Scale shocks require `multiplier=` or `value=`.")
        raise ValueError("Non-scale shocks require `value=`.")

    @staticmethod
    def _normalize_shock_index(
        *,
        index: str | None,
        definition: ShockDefinition | None,
    ) -> str | None:
        if definition is None:
            if index is None:
                return None
            target = index.strip()
            if not target:
                raise ValueError("Shock index must be non-empty when provided.")
            return target

        if definition.kind == "scalar":
            if index is not None:
                raise ValueError(
                    f"Variable '{definition.var}' is scalar and does not accept `index=`."
                )
            return None

        if definition.kind != "indexed":
            raise ValueError(f"Unsupported shock definition kind '{definition.kind}'.")

        if index is None:
            raise ValueError(
                f"Variable '{definition.var}' is indexed; provide `index=` or `index='*'`."
            )

        target = index.strip()
        if not target:
            raise ValueError("Shock index must be non-empty.")
        if target == "*":
            return target

        members = definition.members
        if not members:
            return target

        member_lookup = {member.strip().lower(): member for member in members}
        member_key = target.lower()
        if member_key not in member_lookup:
            choices = ", ".join(sorted(members))
            raise ValueError(
                f"Unknown index '{index}' for variable '{definition.var}'. Available: {choices}"
            )
        return member_lookup[member_key]

    @staticmethod
    def _default_shock_name(
        *,
        var: str,
        index: str | None,
        op: str,
        value: float,
    ) -> str:
        parts = [var.strip().lower()]
        if index is not None:
            parts.append("all" if index == "*" else index.strip().lower())
        parts.append(Simulator._format_shock_suffix(op=op, value=value))
        return "_".join(part for part in parts if part)

    @staticmethod
    def _format_shock_suffix(*, op: str, value: float) -> str:
        formatted = Simulator._format_shock_value(value)
        if op == "scale":
            return f"x{formatted}"
        if op == "set":
            return f"set_{formatted}"
        if value >= 0:
            return f"plus_{formatted}"
        return f"minus_{Simulator._format_shock_value(abs(value))}"

    @staticmethod
    def _format_shock_value(value: float) -> str:
        text = f"{float(value):g}"
        return text.replace("-", "m").replace(".", "_")

    @staticmethod
    def _validate_scenarios(scenarios: list[Scenario]) -> None:
        seen: set[str] = set()
        for scenario in scenarios:
            name = scenario.name.strip()
            if not name:
                raise ValueError("Scenario name must be non-empty.")
            key = name.lower()
            if key in seen:
                raise ValueError(f"Duplicate scenario name '{scenario.name}'.")
            seen.add(key)


def run_scenarios(
    *,
    model: str,
    scenarios: list[Scenario],
    reference_results_gdx: Path | str | None = None,
    compare_abs_tol: float = 1e-6,
    compare_rel_tol: float = 1e-6,
    warm_start: bool = True,
    include_base: bool = True,
    base_reference_slice: str = "base",
    **model_options: Any,
) -> dict[str, Any]:
    """Functional sugar over :class:`Simulator`."""
    simulator = Simulator(model=model, **model_options).fit()
    return simulator.run_scenarios(
        scenarios=scenarios,
        reference_results_gdx=reference_results_gdx,
        compare_abs_tol=compare_abs_tol,
        compare_rel_tol=compare_rel_tol,
        warm_start=warm_start,
        include_base=include_base,
        base_reference_slice=base_reference_slice,
    )
