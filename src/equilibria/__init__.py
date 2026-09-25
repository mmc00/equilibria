"""equilibria - A Modern Python Framework for CGE Modeling."""

from equilibria._logging import _install_null_handler, setup_logging
from equilibria.blocks import Block

_install_null_handler()
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
from equilibria.core import (
    Equation,
    Parameter,
    Set,
    SetManager,
    Variable,
)
from equilibria.datasets import dataset_path, list_bundled, load_bundled
from equilibria.model import Model
from equilibria.version import __version__

__all__ = [
    "__version__",
    "setup_logging",
    "dataset_path",
    "list_bundled",
    "load_bundled",
    "Model",
    "Block",
    "Set",
    "SetManager",
    "Parameter",
    "Variable",
    "Equation",
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
