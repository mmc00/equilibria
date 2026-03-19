from __future__ import annotations

import pytest

from equilibria.templates.pep_closure_validator import validate_pep_closure_structure
from equilibria.templates.pep_contract import (
    PEPClosureConfig,
    PEPContract,
    build_pep_closure_config,
    build_pep_contract,
)
from equilibria.templates.pep_runtime_config import PEPRuntimeConfig, build_pep_runtime_config


def test_build_pep_contract_default_name() -> None:
    contract = build_pep_contract("pep_nlp_v1")

    assert isinstance(contract, PEPContract)
    assert contract.name == "pep_nlp_v1"
    assert contract.closure.name == "pep_default"
    assert contract.closure.numeraire == "e"
    assert contract.equations.include[0] == "EQ1"
    assert contract.equations.include[-2:] == ("EQ98", "WALRAS")
    assert "G" in contract.closure.fixed
    assert "KS" in contract.closure.fixed
    assert "LS" in contract.closure.fixed
    assert "PWX" in contract.closure.fixed
    assert "SG" in contract.closure.endogenous


def test_build_pep_contract_mapping_overrides_default() -> None:
    contract = build_pep_contract(
        {
            "closure": {
                "fixed": ["G", "KS", "LS"],
                "endogenous": ["SG"],
            },
            "bounds": {
                "free": ["SG", "LEON"],
            },
        }
    )

    assert contract.closure.fixed == ("G", "KS", "LS")
    assert contract.closure.endogenous == ("SG",)
    assert contract.bounds.free == ("SG", "LEON")
    assert contract.equations.activation_masks == "gams_parity"


def test_build_pep_closure_config_from_named_closure() -> None:
    closure = build_pep_closure_config("government_spending")

    assert isinstance(closure, PEPClosureConfig)
    assert closure.name == "government_spending"
    assert closure.label == "Government spending closure"
    assert "G" in closure.fixed
    assert "SG" in closure.endogenous


def test_build_pep_contract_applies_named_closure_before_overrides() -> None:
    contract = build_pep_contract(
        {
            "closure": {
                "name": "world_price_shock",
                "fixed": ["CAB", "PWM"],
                "endogenous": ["SG", "SROW"],
            }
        }
    )

    assert contract.closure.name == "world_price_shock"
    assert contract.closure.label == "World price shock closure"
    assert contract.closure.fixed == ("CAB", "PWM")
    assert contract.closure.endogenous == ("SG", "SROW")


def test_build_pep_contract_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        build_pep_contract(
            {
                "closure": {
                    "fixed": ["SG", "KS"],
                    "endogenous": ["SG"],
                }
            }
        )


def test_build_pep_runtime_config_default_and_parity_presets() -> None:
    default_cfg = build_pep_runtime_config("default_ipopt")
    parity_cfg = build_pep_runtime_config("parity_ipopt_gams_nlp")

    assert isinstance(default_cfg, PEPRuntimeConfig)
    assert default_cfg.problem_type == "nlp"
    assert default_cfg.jacobian_mode == "analytic"
    assert default_cfg.reference.enabled is False

    assert parity_cfg.reference.enabled is True
    assert parity_cfg.reference.source == "gams"
    assert parity_cfg.reference.model_type == "nlp"
    assert parity_cfg.reference.solver == "ipopt"
    assert parity_cfg.reference.slice == "sim1"


def test_build_pep_runtime_config_accepts_numeric_jacobian_override() -> None:
    cfg = build_pep_runtime_config({"jacobian_mode": "numeric"})

    assert isinstance(cfg, PEPRuntimeConfig)
    assert cfg.jacobian_mode == "numeric"


def test_build_pep_contract_accepts_all_active_masks() -> None:
    contract = build_pep_contract(
        {
            "equations": {
                "activation_masks": "all_active",
                "include": ["EQ41", "EQ78", "EQ98"],
            }
        }
    )

    assert contract.equations.activation_masks == "all_active"
    assert contract.equations.include == ("EQ41", "EQ78", "EQ98")


def test_validate_pep_closure_structure_square_case() -> None:
    report = validate_pep_closure_structure(
        active_equations=["EQ1", "EQ2", "EQ3"],
        free_endogenous_variables=["Q[agr]", "DD[agr]", "IM[agr]"],
        fixed_by_closure=["SG", "PWM", "e"],
        fixed_by_bounds_only=["LEON"],
        numeraire="e",
    )

    assert report.is_valid is True
    assert report.system_shape == "square"
    assert report.equation_variable_gap == 0
    assert report.numeraire_is_fixed is True


def test_validate_pep_closure_structure_flags_nonsquare_and_numeraire() -> None:
    report = validate_pep_closure_structure(
        active_equations=["EQ1", "EQ2", "EQ3", "EQ4"],
        free_endogenous_variables=["Q[agr]", "DD[agr]"],
        fixed_by_closure=["SG"],
        fixed_by_bounds_only=[],
        numeraire="e",
    )

    assert report.is_valid is False
    assert report.system_shape == "overdetermined"
    assert report.equation_variable_gap == 2
    assert report.numeraire_is_fixed is False
    assert any("non-square system" in msg for msg in report.messages)
    assert any("Numeraire" in msg for msg in report.messages)


def test_validate_pep_closure_structure_flags_unsupported_symbols() -> None:
    report = validate_pep_closure_structure(
        active_equations=["EQ1", "EQ2"],
        free_endogenous_variables=["Q[agr]", "DD[agr]"],
        fixed_by_closure=["e"],
        fixed_by_bounds_only=[],
        numeraire="e",
        unsupported_fixed_symbols=["LEON"],
        unsupported_endogenous_symbols=["LEON"],
    )

    assert report.is_valid is False
    assert report.unsupported_fixed_symbols == ("LEON",)
    assert report.unsupported_endogenous_symbols == ("LEON",)
    assert any("Unsupported fixed closure symbols" in msg for msg in report.messages)
    assert any("Unsupported endogenous closure symbols" in msg for msg in report.messages)
