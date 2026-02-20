"""Baseline manifest generation and strict compatibility checks."""

from equilibria.baseline.compatibility import (
    BaselineCompatibilityReport,
    evaluate_strict_gams_baseline_compatibility,
)
from equilibria.baseline.manifest import (
    BaselineManifest,
    build_baseline_manifest,
    compute_state_anchors,
    file_sha256,
    load_baseline_manifest,
)

__all__ = [
    "BaselineCompatibilityReport",
    "BaselineManifest",
    "build_baseline_manifest",
    "compute_state_anchors",
    "evaluate_strict_gams_baseline_compatibility",
    "file_sha256",
    "load_baseline_manifest",
]
