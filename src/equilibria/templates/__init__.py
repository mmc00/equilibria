"""Templates module for equilibria CGE framework.

Provides pre-configured model templates for common CGE model types.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "ModelTemplate",
    "SimpleOpenEconomy",
    "SimpleOpenContract",
    "SimpleOpenClosureConfig",
    "SimpleOpenEquationConfig",
    "SimpleOpenBoundsConfig",
    "build_simple_open_closure_config",
    "build_simple_open_contract",
    "SimpleOpenRuntimeConfig",
    "SimpleOpenReferenceConfig",
    "build_simple_open_runtime_config",
    "SimpleOpenConstraintJacobianHarness",
    "SimpleOpenBenchmarkParameters",
    "SimpleOpenParitySpec",
    "build_simple_open_parity_spec",
    "PEPContract",
    "PEPClosureConfig",
    "PEPEquationConfig",
    "PEPBoundsConfig",
    "build_pep_closure_config",
    "build_pep_contract",
    "PEPRuntimeConfig",
    "PEPReferenceConfig",
    "build_pep_runtime_config",
    "PEPClosureValidationReport",
    "validate_pep_closure_structure",
    "PEPScenarioParityRunner",
    "PEPExportTaxParityRunner",
    "PEPGovernmentSpendingParityRunner",
    "PEPImportPriceParityRunner",
    "PEPImportShockParityRunner",
    "EquilibriaLevelsExtractor",
    "GAMSLevelsExtractor",
    "LevelsComparator",
    # GAMS Comparison
    "GAMSComparisonResult",
    "GAMSComparisonReport",
    "GAMSRunner",
    "SolutionComparator",
    "PEPGAMSComparator",
    "run_gams_comparison",
]


def __getattr__(name: str) -> Any:
    """Lazy-load template exports to avoid import cycles."""
    if name == "ModelTemplate":
        from equilibria.templates.base import ModelTemplate

        return ModelTemplate

    if name == "SimpleOpenEconomy":
        from equilibria.templates.simple_open import SimpleOpenEconomy

        return SimpleOpenEconomy

    if name in {
        "SimpleOpenContract",
        "SimpleOpenClosureConfig",
        "SimpleOpenEquationConfig",
        "SimpleOpenBoundsConfig",
        "build_simple_open_closure_config",
        "build_simple_open_contract",
    }:
        from equilibria.templates.simple_open_contract import (
            SimpleOpenBoundsConfig,
            SimpleOpenClosureConfig,
            SimpleOpenContract,
            SimpleOpenEquationConfig,
            build_simple_open_closure_config,
            build_simple_open_contract,
        )

        return {
            "SimpleOpenContract": SimpleOpenContract,
            "SimpleOpenClosureConfig": SimpleOpenClosureConfig,
            "SimpleOpenEquationConfig": SimpleOpenEquationConfig,
            "SimpleOpenBoundsConfig": SimpleOpenBoundsConfig,
            "build_simple_open_closure_config": build_simple_open_closure_config,
            "build_simple_open_contract": build_simple_open_contract,
        }[name]

    if name in {
        "SimpleOpenRuntimeConfig",
        "SimpleOpenReferenceConfig",
        "build_simple_open_runtime_config",
    }:
        from equilibria.templates.simple_open_runtime_config import (
            SimpleOpenReferenceConfig,
            SimpleOpenRuntimeConfig,
            build_simple_open_runtime_config,
        )

        return {
            "SimpleOpenRuntimeConfig": SimpleOpenRuntimeConfig,
            "SimpleOpenReferenceConfig": SimpleOpenReferenceConfig,
            "build_simple_open_runtime_config": build_simple_open_runtime_config,
        }[name]

    if name == "SimpleOpenConstraintJacobianHarness":
        from equilibria.templates.simple_open_constraint_jacobian import (
            SimpleOpenConstraintJacobianHarness,
        )

        return SimpleOpenConstraintJacobianHarness

    if name in {
        "SimpleOpenBenchmarkParameters",
        "SimpleOpenParitySpec",
        "build_simple_open_parity_spec",
    }:
        from equilibria.templates.simple_open_parity_spec import (
            SimpleOpenBenchmarkParameters,
            SimpleOpenParitySpec,
            build_simple_open_parity_spec,
        )

        return {
            "SimpleOpenBenchmarkParameters": SimpleOpenBenchmarkParameters,
            "SimpleOpenParitySpec": SimpleOpenParitySpec,
            "build_simple_open_parity_spec": build_simple_open_parity_spec,
        }[name]

    if name in {
        "PEPContract",
        "PEPClosureConfig",
        "PEPEquationConfig",
        "PEPBoundsConfig",
        "build_pep_closure_config",
        "build_pep_contract",
    }:
        from equilibria.templates.pep_contract import (
            PEPBoundsConfig,
            PEPClosureConfig,
            PEPContract,
            PEPEquationConfig,
            build_pep_closure_config,
            build_pep_contract,
        )

        return {
            "PEPContract": PEPContract,
            "PEPClosureConfig": PEPClosureConfig,
            "PEPEquationConfig": PEPEquationConfig,
            "PEPBoundsConfig": PEPBoundsConfig,
            "build_pep_closure_config": build_pep_closure_config,
            "build_pep_contract": build_pep_contract,
        }[name]

    if name in {
        "PEPRuntimeConfig",
        "PEPReferenceConfig",
        "build_pep_runtime_config",
    }:
        from equilibria.templates.pep_runtime_config import (
            PEPReferenceConfig,
            PEPRuntimeConfig,
            build_pep_runtime_config,
        )

        return {
            "PEPRuntimeConfig": PEPRuntimeConfig,
            "PEPReferenceConfig": PEPReferenceConfig,
            "build_pep_runtime_config": build_pep_runtime_config,
        }[name]

    if name in {"PEPClosureValidationReport", "validate_pep_closure_structure"}:
        from equilibria.templates.pep_closure_validator import (
            PEPClosureValidationReport,
            validate_pep_closure_structure,
        )

        return {
            "PEPClosureValidationReport": PEPClosureValidationReport,
            "validate_pep_closure_structure": validate_pep_closure_structure,
        }[name]

    if name == "PEPScenarioParityRunner":
        warnings.warn(
            "equilibria.templates.PEPScenarioParityRunner is deprecated. "
            "Use equilibria.simulations.PepSimulator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from equilibria.templates.pep_scenario_parity import PEPScenarioParityRunner

        return PEPScenarioParityRunner

    if name in {
        "PEPExportTaxParityRunner",
        "PEPGovernmentSpendingParityRunner",
        "PEPImportPriceParityRunner",
        "PEPImportShockParityRunner",
    }:
        warnings.warn(
            f"equilibria.templates.{name} is deprecated. "
            "Use equilibria.simulations.PepSimulator presets/methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from equilibria.templates.pep_scenario_parity import (
            PEPExportTaxParityRunner,
            PEPGovernmentSpendingParityRunner,
            PEPImportPriceParityRunner,
            PEPImportShockParityRunner,
        )

        return {
            "PEPExportTaxParityRunner": PEPExportTaxParityRunner,
            "PEPGovernmentSpendingParityRunner": PEPGovernmentSpendingParityRunner,
            "PEPImportPriceParityRunner": PEPImportPriceParityRunner,
            "PEPImportShockParityRunner": PEPImportShockParityRunner,
        }[name]

    if name in {"EquilibriaLevelsExtractor", "GAMSLevelsExtractor", "LevelsComparator"}:
        from equilibria.templates.pep_levels import (
            EquilibriaLevelsExtractor,
            GAMSLevelsExtractor,
            LevelsComparator,
        )

        return {
            "EquilibriaLevelsExtractor": EquilibriaLevelsExtractor,
            "GAMSLevelsExtractor": GAMSLevelsExtractor,
            "LevelsComparator": LevelsComparator,
        }[name]

    if name in {
        "GAMSComparisonResult",
        "GAMSComparisonReport",
        "GAMSRunner",
        "SolutionComparator",
        "PEPGAMSComparator",
        "run_gams_comparison",
    }:
        from equilibria.templates.gams_comparison import (
            GAMSComparisonReport,
            GAMSComparisonResult,
            GAMSRunner,
            PEPGAMSComparator,
            SolutionComparator,
            run_gams_comparison,
        )

        return {
            "GAMSComparisonResult": GAMSComparisonResult,
            "GAMSComparisonReport": GAMSComparisonReport,
            "GAMSRunner": GAMSRunner,
            "SolutionComparator": SolutionComparator,
            "PEPGAMSComparator": PEPGAMSComparator,
            "run_gams_comparison": run_gams_comparison,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
