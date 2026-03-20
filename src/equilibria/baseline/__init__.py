"""Baseline manifest generation and strict compatibility checks."""

from equilibria.baseline.compatibility import (
    BaselineCompatibilityReport,
    evaluate_strict_gams_baseline_compatibility,
)
from equilibria.baseline.gams_nlp_reference import (
    GAMSReferenceArtifact,
    GAMSScenarioReference,
    GAMSNLPReferenceManifest,
    build_gams_nlp_reference_manifest,
    ensure_gams_script_uses_nlp,
    extract_gams_solve_model_types,
    load_gams_nlp_reference_manifest,
)
from equilibria.baseline.manifest import (
    BaselineManifest,
    build_baseline_manifest,
    compute_state_anchors,
    file_sha256,
    load_baseline_manifest,
)
from equilibria.baseline.simple_open_gams_reference import (
    SimpleOpenClosureReference,
    SimpleOpenGAMSReferenceManifest,
    build_simple_open_gams_reference_manifest,
    load_simple_open_gams_reference_manifest,
)

__all__ = [
    "BaselineCompatibilityReport",
    "BaselineManifest",
    "GAMSReferenceArtifact",
    "GAMSScenarioReference",
    "GAMSNLPReferenceManifest",
    "SimpleOpenClosureReference",
    "SimpleOpenGAMSReferenceManifest",
    "build_baseline_manifest",
    "build_gams_nlp_reference_manifest",
    "build_simple_open_gams_reference_manifest",
    "compute_state_anchors",
    "ensure_gams_script_uses_nlp",
    "evaluate_strict_gams_baseline_compatibility",
    "extract_gams_solve_model_types",
    "file_sha256",
    "load_baseline_manifest",
    "load_gams_nlp_reference_manifest",
    "load_simple_open_gams_reference_manifest",
]
