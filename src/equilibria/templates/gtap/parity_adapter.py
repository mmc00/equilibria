"""GTAP ParityAdapter — wraps GTAPModelEquations build/solve + _diff_core lookups."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_GTAP = _ROOT / "scripts" / "gtap"
if str(_SCRIPTS_GTAP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_GTAP))


class GTAPParityAdapter:
    """ParityAdapter implementation for the GTAP Standard 7 template."""

    _COMBINATIONS = [
        ("9x10", "baseline"),
        ("9x10", "shock_tm10"),
        ("9x10", "altertax"),
        ("nus333", "baseline"),
        ("nus333", "shock_tm10"),
        ("nus333", "altertax"),
    ]

    _DATASET_GDX = {
        "9x10": "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx",
        "nus333": "src/equilibria/templates/reference/gtap/data/basedata-nus333.gdx",
    }

    def enumerate_combinations(self) -> list[tuple[str, str]]:
        return list(self._COMBINATIONS)

    def name_aliases(self) -> dict[str, str]:
        from _diff_core import _NAME_ALIAS  # type: ignore[import-not-found]
        return dict(_NAME_ALIAS)

    def load_gams_reference(self, gdx_path) -> dict[str, dict[tuple, float]]:
        from _diff_core import gams_levels, list_populated_vars, split_t  # type: ignore[import-not-found]
        gdx_path = Path(gdx_path)
        out: dict[str, dict[tuple, float]] = {}
        for vname in list_populated_vars(gdx_path):
            raw = gams_levels(gdx_path, vname)
            stripped: dict[tuple, float] = {}
            for key, val in raw.items():
                body, _t = split_t(key)
                if body in stripped:
                    continue
                stripped[body] = float(val)
            if stripped:
                out[vname] = stripped
        return out

    def find_py_var(self, model: Any, gams_name: str) -> tuple[Any | None, str | None]:
        from _diff_core import find_py_var, build_derived  # type: ignore[import-not-found]
        derived = build_derived(model)
        return find_py_var(model, gams_name, derived=derived)

    def build_solved_model(self, dataset: str, scenario: str) -> Any:
        if (dataset, scenario) not in self._COMBINATIONS:
            raise ValueError(
                f"Unknown GTAP combination ({dataset!r}, {scenario!r}). "
                f"Known: {self._COMBINATIONS}"
            )

        from equilibria.templates.gtap import (
            GTAPParameters, build_gtap_contract, GTAPModelEquations,
        )
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities
        import importlib.util as _u

        gdx_rel = self._DATASET_GDX[dataset]
        gdx_path = _ROOT / gdx_rel
        params = GTAPParameters()
        params.load_from_gdx(gdx_path)

        spec = _u.spec_from_file_location(
            "run_gtap", str(_ROOT / "scripts" / "gtap" / "run_gtap.py")
        )
        run_gtap = _u.module_from_spec(spec)
        sys.modules["run_gtap"] = run_gtap
        spec.loader.exec_module(run_gtap)

        residual_region = "NAmerica" if dataset == "9x10" else "ROW"

        std_contract = build_gtap_contract({})
        base_eq = GTAPModelEquations(
            params.sets, params, residual_region=residual_region,
            closure=std_contract.closure,
        )
        m_b = base_eq.build_model()
        run_gtap._run_path_capi_nonlinear_full(
            m_b, params, enforce_post_checks=False, strict_path_capi=False,
            closure_config=std_contract.closure, equation_scaling=True,
        )

        if scenario == "baseline":
            return m_b

        if scenario == "altertax":
            shock_params = apply_altertax_elasticities(params, in_place=False)
            contract = build_gtap_contract({"closure": "altertax"})
            from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
            warm = GTAPVariableSnapshot.from_python_model(m_b)
            alt_sets = getattr(shock_params, "sets", None) or params.sets
            eq = GTAPModelEquations(
                alt_sets, shock_params, contract.closure, is_counterfactual=True,
                residual_region=residual_region, t0_snapshot=m_b,
            )
            m_alt = eq.build_model()
            run_gtap._run_path_capi_nonlinear_full(
                m_alt, shock_params,
                solver_output=False, path_license_string=None,
                enforce_post_checks=False, strict_path_capi=False,
                closure_config=contract.closure, equation_scaling=True,
                solution_hint=warm,
            )
            return m_alt

        if scenario == "shock_tm10":
            shock_params = run_gtap._apply_shock_to_params(
                params, shock_mode="tm_pct", shock_value=0.10,
            )
            contract = build_gtap_contract({"closure": "shock"})
            from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
            warm = GTAPVariableSnapshot.from_python_model(m_b)
            eq = GTAPModelEquations(
                params.sets, shock_params, contract.closure, is_counterfactual=True,
                residual_region=residual_region, t0_snapshot=m_b,
            )
            m_s = eq.build_model()
            run_gtap._run_path_capi_nonlinear_full(
                m_s, shock_params, enforce_post_checks=False, strict_path_capi=False,
                closure_config=contract.closure, equation_scaling=True,
                solution_hint=warm,
            )
            return m_s

        raise ValueError(f"Unhandled scenario {scenario!r}")
