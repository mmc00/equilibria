"""Generic contract/runtime primitives shared across model templates."""

from equilibria.contracts.base import (
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

__all__ = [
    "ModelBoundsConfig",
    "ModelClosureConfig",
    "ModelClosureValidationReport",
    "ModelContract",
    "ModelEquationConfig",
    "ModelReferenceConfig",
    "ModelRuntimeConfig",
    "deep_merge_model_dicts",
    "normalize_string_tuple",
    "validate_closure_structure",
]
