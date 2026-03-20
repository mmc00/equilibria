from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.baseline import (
    SimpleOpenClosureReference,
    SimpleOpenGAMSReferenceManifest,
    build_simple_open_gams_reference_manifest,
    load_simple_open_gams_reference_manifest,
)


def test_build_simple_open_gams_reference_manifest_roundtrip(tmp_path: Path) -> None:
    gms_script = tmp_path / "simple_open.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE simple_open_v1_benchmark USING NLP MINIMIZING OBJ;
        """
    )
    default_gdx = tmp_path / "default.gdx"
    default_gdx.write_bytes(b"default")
    flexible_gdx = tmp_path / "flexible.gdx"
    flexible_gdx.write_bytes(b"flexible")

    manifest = build_simple_open_gams_reference_manifest(
        gms_script=gms_script,
        closure_references={
            "simple_open_default": {
                "closure": "simple_open_default",
                "results_gdx": {"path": str(default_gdx), "sha256": "abc123"},
            },
            "flexible_external_balance": SimpleOpenClosureReference.model_validate(
                {
                    "closure": "flexible_external_balance",
                    "results_gdx": {"path": str(flexible_gdx), "sha256": "def456"},
                }
            ),
        },
        metadata={"anchor": "stdcge"},
    )

    assert isinstance(manifest, SimpleOpenGAMSReferenceManifest)
    assert manifest.schema_version == "simple_open_gams_reference/v1"
    assert manifest.script_model_types == ("nlp",)
    assert manifest.closure_references["simple_open_default"].results_gdx.path == str(default_gdx)

    out = tmp_path / "manifest.json"
    manifest.save_json(out)
    loaded = load_simple_open_gams_reference_manifest(out)
    assert loaded.gms_script.sha256 == manifest.gms_script.sha256
    assert loaded.closure_references["flexible_external_balance"].results_gdx.sha256 == "def456"


def test_simple_open_reference_manifest_requires_both_canonical_closures(tmp_path: Path) -> None:
    gms_script = tmp_path / "simple_open.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE simple_open_v1_benchmark USING NLP MINIMIZING OBJ;
        """
    )
    default_gdx = tmp_path / "default.gdx"
    default_gdx.write_bytes(b"default")

    with pytest.raises(ValueError, match="canonical closures"):
        build_simple_open_gams_reference_manifest(
            gms_script=gms_script,
            closure_references={
                "simple_open_default": {
                    "closure": "simple_open_default",
                    "results_gdx": {"path": str(default_gdx), "sha256": "abc123"},
                }
            },
        )
