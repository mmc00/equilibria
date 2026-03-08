"""Generic state-based adapter for models without a native solver integration yet."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from equilibria.simulations.adapters.base import BaseModelAdapter
from equilibria.simulations.types import Shock, ShockDefinition


@dataclass
class _NoSolveSolver:
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _NoSolveSolution:
    converged: bool = True
    iterations: int = 0
    final_residual: float = 0.0
    message: str = "no_solver"
    variables: dict[str, Any] = field(default_factory=dict)


class ParameterStateAdapter(BaseModelAdapter):
    """Adapter that runs scenario shocks over a generic state mapping.

    This adapter is intentionally solver-free:
    - `solve_state` returns the shocked state as a converged no-op solution.
    - Useful for early scenario plumbing in models where solve integration is pending.
    """

    def __init__(
        self,
        *,
        model_label: str,
        base_state: dict[str, Any] | None = None,
        state_loader: Callable[[], dict[str, Any]] | None = None,
        shock_definitions: list[ShockDefinition] | None = None,
    ) -> None:
        self.model_label = str(model_label)
        self._base_state = copy.deepcopy(base_state) if base_state is not None else None
        self._state_loader = state_loader
        self._shock_definitions = list(shock_definitions) if shock_definitions is not None else None
        self._resolved_shock_defs: list[ShockDefinition] = []

    def fit_base_state(self) -> dict[str, Any]:
        if self._state_loader is not None:
            state = self._state_loader()
        elif self._base_state is not None:
            state = copy.deepcopy(self._base_state)
        else:
            raise ValueError(
                f"{self.model_label}: provide `base_state` or `state_loader` "
                "to use this adapter."
            )

        if not isinstance(state, dict):
            raise TypeError(f"{self.model_label}: base state must be a dict.")

        if self._shock_definitions is None:
            self._resolved_shock_defs = self._derive_shock_catalog(state)
        else:
            self._resolved_shock_defs = list(self._shock_definitions)
        return state

    def available_shocks(self) -> list[ShockDefinition]:
        if self._resolved_shock_defs:
            return list(self._resolved_shock_defs)
        if self._shock_definitions is not None:
            return list(self._shock_definitions)
        return []

    def apply_shock(self, state: dict[str, Any], shock: Shock) -> None:
        key = self._resolve_var_key(state, shock.var)
        op = shock.op.strip().lower()
        if op not in {"set", "scale", "add"}:
            raise ValueError(f"Unsupported shock op '{shock.op}'.")

        current = state[key]
        if isinstance(current, (int, float)):
            state[key] = self._apply_scalar(float(current), op, shock.values)
            return

        if isinstance(current, dict):
            state[key] = self._apply_indexed(current, op, shock.values, var_name=key)
            return

        raise TypeError(
            f"{self.model_label}: variable '{key}' is not numeric scalar or dict."
        )

    def solve_state(
        self,
        state: dict[str, Any],
        *,
        initial_vars: Any | None,
        reference_results_gdx: Path | None,
        reference_slice: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _ = initial_vars, reference_results_gdx, reference_slice
        solver = _NoSolveSolver(params={})
        solution = _NoSolveSolution(variables=copy.deepcopy(state))
        validation = {"passed": True, "mode": "no_solver"}
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
        _ = solution_vars, solution_params, reference_results_gdx, abs_tol, rel_tol
        return {
            "passed": False,
            "reason": f"{self.model_label}: reference comparison not implemented",
            "gams_slice": reference_slice,
        }

    def key_indicators(self, vars_obj: Any) -> dict[str, float]:
        if not isinstance(vars_obj, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in vars_obj.items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
                continue
            if isinstance(value, dict):
                total = 0.0
                has_numeric = False
                for item in value.values():
                    if isinstance(item, (int, float)):
                        total += float(item)
                        has_numeric = True
                if has_numeric:
                    out[f"{key}_sum"] = total
        return out

    def _resolve_var_key(self, state: dict[str, Any], name: str) -> str:
        wanted = name.strip().lower()
        lookup = {str(k).strip().lower(): str(k) for k in state}
        if wanted not in lookup:
            keys = ", ".join(sorted(str(k) for k in state))
            raise ValueError(
                f"{self.model_label}: unknown shock variable '{name}'. "
                f"Available: {keys}"
            )
        return lookup[wanted]

    @staticmethod
    def _apply_scalar(current: float, op: str, values: float | dict[str, float]) -> float:
        if isinstance(values, dict):
            if "*" not in values:
                raise ValueError("Scalar shock with dict values requires '*' entry.")
            value = float(values["*"])
        else:
            value = float(values)

        if op == "set":
            return value
        if op == "scale":
            return current * value
        if op == "add":
            return current + value
        raise ValueError(f"Unsupported op '{op}'.")

    @staticmethod
    def _apply_indexed(
        current: dict[Any, Any],
        op: str,
        values: float | dict[str, float],
        *,
        var_name: str,
    ) -> dict[str, float]:
        src = {str(k): float(v) for k, v in current.items() if isinstance(v, (int, float))}
        if isinstance(values, dict):
            updates = {str(k): float(v) for k, v in values.items()}
            unknown = {k for k in updates if k != "*" and k not in src}
            if unknown:
                bad = ", ".join(sorted(unknown))
                raise ValueError(f"{var_name}: unknown indices in shock values: {bad}")
        else:
            updates = {"*": float(values)}

        if "*" in updates:
            wildcard = float(updates["*"])
            for key in list(src):
                src[key] = ParameterStateAdapter._merge(src[key], op, wildcard)

        for key, value in updates.items():
            if key == "*":
                continue
            src[key] = ParameterStateAdapter._merge(src[key], op, float(value))

        return src

    @staticmethod
    def _merge(current: float, op: str, value: float) -> float:
        if op == "set":
            return value
        if op == "scale":
            return current * value
        if op == "add":
            return current + value
        raise ValueError(f"Unsupported op '{op}'.")

    @staticmethod
    def _derive_shock_catalog(state: dict[str, Any]) -> list[ShockDefinition]:
        catalog: list[ShockDefinition] = []
        for key, value in state.items():
            if isinstance(value, (int, float)):
                catalog.append(
                    ShockDefinition(
                        var=str(key),
                        kind="scalar",
                        domain=None,
                        members=None,
                        ops=("set", "scale", "add"),
                        description=f"State scalar '{key}'.",
                    )
                )
                continue
            if isinstance(value, dict):
                numeric_members = [
                    str(member)
                    for member, member_value in value.items()
                    if isinstance(member_value, (int, float))
                ]
                if not numeric_members:
                    continue
                catalog.append(
                    ShockDefinition(
                        var=str(key),
                        kind="indexed",
                        domain=None,
                        members=tuple(numeric_members),
                        ops=("set", "scale", "add"),
                        description=f"State indexed mapping '{key}'.",
                    )
                )
        return catalog
