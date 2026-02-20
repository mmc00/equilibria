"""Unit tests for solver initialization strategy dispatch."""

from __future__ import annotations

import pytest

from equilibria.templates.init_strategies import (
    CANONICAL_INIT_MODES,
    GAMSFlowInitializationStrategy,
    StrictGAMSInitializationStrategy,
    build_init_strategy,
    normalize_init_mode,
)
from equilibria.templates.pep_model_equations import PEPModelVariables


def test_build_init_strategy_modes() -> None:
    assert CANONICAL_INIT_MODES == ("gams", "excel")
    assert isinstance(build_init_strategy("gams"), StrictGAMSInitializationStrategy)
    assert isinstance(build_init_strategy("excel"), GAMSFlowInitializationStrategy)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("strict_gams", "gams"),
        ("gams_flow", "excel"),
        ("gams_levels", "excel"),
        ("equation_consistent", "excel"),
        ("gams_blockwise", "excel"),
    ],
)
def test_legacy_mode_aliases(alias: str, expected: str) -> None:
    assert normalize_init_mode(alias) == expected


def test_build_init_strategy_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_init_strategy("not_a_mode")


class _DummySolver:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sets = {"I": ["agr"], "H": ["hrp"]}

    def _ensure_strict_gams_baseline_compatibility(self) -> None:
        self.calls.append("strict_check")

    def _overlay_with_gams_levels(self, vars: PEPModelVariables) -> None:
        self.calls.append("overlay")
        vars.Q["agr"] = 10.0
        vars.C[("agr", "hrp")] = 2.0
        vars.CG["agr"] = 1.0
        vars.INV["agr"] = 3.0
        vars.VSTK["agr"] = 0.5
        vars.DIT["agr"] = 1.5
        vars.MRGN["agr"] = 0.0

    def _overlay_with_calibrated_levels(self, vars: PEPModelVariables) -> None:
        self.calls.append("overlay_calibrated")
        vars.Q["agr"] = 10.0
        vars.C[("agr", "hrp")] = 2.0
        vars.CG["agr"] = 1.0
        vars.INV["agr"] = 3.0
        vars.VSTK["agr"] = 0.5
        vars.DIT["agr"] = 1.5
        vars.MRGN["agr"] = 0.0

    def _sync_lambda_tr_from_levels(self, _vars: PEPModelVariables) -> None:
        self.calls.append("sync_lambda")

    def _sync_policy_params_from_vars(self, _vars: PEPModelVariables) -> None:
        self.calls.append("sync_policy")

    def _apply_equation_consistent_adjustments(self, _vars: PEPModelVariables) -> None:
        self.calls.append("eq_adjust")

    def _reconcile_composite_prices(self, _vars: PEPModelVariables) -> None:
        self.calls.append("reconcile_prices")

    def _reconcile_tax_identities(self, _vars: PEPModelVariables) -> None:
        self.calls.append("reconcile_taxes")

    def _recompute_gdp_aggregates(self, _vars: PEPModelVariables) -> None:
        self.calls.append("recompute_gdp")

    def _apply_trade_blockwise_flow(self, _vars: PEPModelVariables) -> None:
        self.calls.append("flow")

    def _apply_trade_blockwise_transformation(self, _vars: PEPModelVariables) -> None:
        self.calls.append("transform")

    def _apply_production_blockwise_accounting(self, _vars: PEPModelVariables) -> None:
        self.calls.append("prod")

    def _apply_commodity_balance_blockwise(self, _vars: PEPModelVariables) -> None:
        self.calls.append("commodity")

    def _apply_trade_market_clearing_blockwise(self, _vars: PEPModelVariables) -> None:
        self.calls.append("market")

    def _attempt_coupled_trade_reconciliation(self, _vars: PEPModelVariables) -> None:
        self.calls.append("coupled")

    def _apply_macro_closure_blockwise(self, _vars: PEPModelVariables) -> None:
        self.calls.append("macro")

    def _apply_gams_blockwise_presolve(self, _vars: PEPModelVariables) -> None:
        self.calls.append("blockwise_presolve")


def test_strict_strategy_calls_gate_then_overlay() -> None:
    solver = _DummySolver()
    vars = PEPModelVariables()

    build_init_strategy("gams").apply(solver, vars)

    assert solver.calls == ["strict_check", "overlay", "sync_lambda", "sync_policy"]
    assert pytest.approx(2.0) == vars.LEON


@pytest.mark.parametrize(
    "mode",
    [
        "excel",
        "gams_flow",
        "gams_levels",
        "equation_consistent",
        "gams_blockwise",
    ],
)
def test_excel_strategy_alias_sequence(mode: str) -> None:
    solver = _DummySolver()
    vars = PEPModelVariables()

    build_init_strategy(mode).apply(solver, vars)
    assert solver.calls == [
        "overlay_calibrated",
        "sync_lambda",
        "sync_policy",
    ]
    assert pytest.approx(2.0) == vars.LEON
