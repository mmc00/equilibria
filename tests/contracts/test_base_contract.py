from __future__ import annotations

from equilibria import (
    ModelBoundsConfig as RootModelBoundsConfig,
    ModelClosureConfig as RootModelClosureConfig,
    ModelClosureValidationReport as RootModelClosureValidationReport,
    ModelContract as RootModelContract,
    ModelEquationConfig as RootModelEquationConfig,
    ModelReferenceConfig as RootModelReferenceConfig,
    ModelRuntimeConfig as RootModelRuntimeConfig,
    deep_merge_model_dicts as root_deep_merge_model_dicts,
    normalize_string_tuple as root_normalize_string_tuple,
    validate_closure_structure as root_validate_closure_structure,
)
from equilibria.contracts import (
    ModelBoundsConfig,
    ModelClosureConfig,
    ModelClosureValidationReport,
    ModelContract,
    ModelEquationConfig,
    ModelReferenceConfig,
    ModelRuntimeConfig,
    deep_merge_model_dicts,
    normalize_string_tuple,
    validate_closure_structure,
)
from equilibria.templates.pep_closure_validator import PEPClosureValidationReport
from equilibria.templates.pep_contract import (
    PEPBoundsConfig,
    PEPClosureConfig,
    PEPContract,
    PEPEquationConfig,
)
from equilibria.templates.pep_runtime_config import PEPReferenceConfig, PEPRuntimeConfig


def test_generic_contract_helpers_normalize_and_merge() -> None:
    assert normalize_string_tuple([" G ", "SG", "G", ""]) == ("G", "SG")

    merged = deep_merge_model_dicts(
        {"closure": {"fixed": ["G"]}, "bounds": {"free": ["SG"]}},
        {"closure": {"endogenous": ["SG"]}},
    )

    assert merged == {
        "closure": {"fixed": ["G"], "endogenous": ["SG"]},
        "bounds": {"free": ["SG"]},
    }


def test_generic_contract_models_validate() -> None:
    contract = ModelContract(
        name="generic_v1",
        closure=ModelClosureConfig(
            name="default",
            numeraire="pnum",
            fixed=("pnum", "G"),
            endogenous=("SG",),
        ),
        equations=ModelEquationConfig(
            name="full",
            include=("EQ1", "EQ2"),
            activation_masks="default",
        ),
        bounds=ModelBoundsConfig(name="economic", free=("SG",)),
    )
    runtime = ModelRuntimeConfig(
        name="default",
        problem_type="nlp",
        solver="ipopt",
        reference=ModelReferenceConfig(),
    )

    assert contract.closure.fixed == ("pnum", "G")
    assert runtime.reference.enabled is False


def test_generic_closure_validation_report_square_case() -> None:
    report = validate_closure_structure(
        active_equations=["EQ1", "EQ2"],
        free_endogenous_variables=["Q[agr]", "DD[agr]"],
        fixed_by_closure=["e"],
        fixed_by_bounds_only=[],
        numeraire="e",
    )

    assert isinstance(report, ModelClosureValidationReport)
    assert report.is_valid is True
    assert report.system_shape == "square"


def test_pep_contract_types_inherit_generic_base() -> None:
    assert issubclass(PEPClosureConfig, ModelClosureConfig)
    assert issubclass(PEPEquationConfig, ModelEquationConfig)
    assert issubclass(PEPBoundsConfig, ModelBoundsConfig)
    assert issubclass(PEPContract, ModelContract)
    assert issubclass(PEPReferenceConfig, ModelReferenceConfig)
    assert issubclass(PEPRuntimeConfig, ModelRuntimeConfig)
    assert issubclass(PEPClosureValidationReport, ModelClosureValidationReport)


def test_generic_contract_types_are_visible_from_package_root() -> None:
    assert RootModelContract is ModelContract
    assert RootModelClosureConfig is ModelClosureConfig
    assert RootModelEquationConfig is ModelEquationConfig
    assert RootModelBoundsConfig is ModelBoundsConfig
    assert RootModelRuntimeConfig is ModelRuntimeConfig
    assert RootModelReferenceConfig is ModelReferenceConfig
    assert RootModelClosureValidationReport is ModelClosureValidationReport
    assert root_normalize_string_tuple is normalize_string_tuple
    assert root_deep_merge_model_dicts is deep_merge_model_dicts
    assert root_validate_closure_structure is validate_closure_structure
