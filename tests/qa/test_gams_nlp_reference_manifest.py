from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.baseline import (
    GAMSNLPReferenceManifest,
    build_gams_nlp_reference_manifest,
    ensure_gams_script_uses_nlp,
    extract_gams_solve_model_types,
    load_gams_nlp_reference_manifest,
)


def test_extract_gams_solve_model_types_reads_unique_model_types(tmp_path: Path) -> None:
    script = tmp_path / "reference.gms"
    script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE PEP11 USING NLP MINIMIZING OBJ;
        SOLVE PEP11 USING NLP MINIMIZING OBJ;
        """
    )

    assert extract_gams_solve_model_types(script) == ("nlp",)
    assert ensure_gams_script_uses_nlp(script) == ("nlp",)


def test_ensure_gams_script_uses_nlp_rejects_cns_only_script(tmp_path: Path) -> None:
    script = tmp_path / "reference_cns.gms"
    script.write_text(
        """
        OPTION CNS = ipopt;
        SOLVE PEP11 USING CNS;
        """
    )

    assert extract_gams_solve_model_types(script) == ("cns",)
    with pytest.raises(ValueError, match="must use NLP"):
        ensure_gams_script_uses_nlp(script)


def test_build_gams_nlp_reference_manifest_roundtrip(tmp_path: Path) -> None:
    gms_script = tmp_path / "reference_nlp.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE PEP11 USING NLP MINIMIZING OBJ;
        """
    )
    sam_file = tmp_path / "SAM.xlsx"
    sam_file.write_bytes(b"sam")
    val_par_file = tmp_path / "VAL_PAR.xlsx"
    val_par_file.write_bytes(b"val-par")
    results_gdx = tmp_path / "Results.gdx"
    results_gdx.write_bytes(b"results")
    parameters_gdx = tmp_path / "Parameters.gdx"
    parameters_gdx.write_bytes(b"parameters")
    presolve_gdx = tmp_path / "PreSolveLevels.gdx"
    presolve_gdx.write_bytes(b"presolve")

    manifest = build_gams_nlp_reference_manifest(
        gms_script=gms_script,
        sam_file=sam_file,
        val_par_file=val_par_file,
        results_gdx=results_gdx,
        parameters_gdx=parameters_gdx,
        presolve_levels_gdx=presolve_gdx,
        scenario_slices={
            "BASE": "BASE",
            "government_spending": "SIM1",
            "import_shock": "SIM2",
        },
        metadata={"sam_mode": "excel"},
    )

    assert isinstance(manifest, GAMSNLPReferenceManifest)
    assert manifest.problem_type == "nlp"
    assert manifest.solver == "ipopt"
    assert manifest.script_model_types == ("nlp",)
    assert manifest.scenario_slices["base"] == "base"
    assert manifest.scenario_slices["government_spending"] == "sim1"
    assert manifest.gms_script.sha256
    assert manifest.results_gdx.sha256

    out = tmp_path / "manifest.json"
    manifest.save_json(out)
    loaded = load_gams_nlp_reference_manifest(out)

    assert loaded.schema_version == "pep_gams_nlp_reference/v1"
    assert loaded.sam_file.sha256 == manifest.sam_file.sha256
    assert loaded.parameters_gdx.sha256 == manifest.parameters_gdx.sha256
    assert loaded.presolve_levels_gdx.sha256 == manifest.presolve_levels_gdx.sha256


def test_reference_manifest_requires_base_slice(tmp_path: Path) -> None:
    gms_script = tmp_path / "reference_nlp.gms"
    gms_script.write_text(
        """
        OPTION NLP = ipopt;
        SOLVE PEP11 USING NLP MINIMIZING OBJ;
        """
    )
    sam_file = tmp_path / "SAM.xlsx"
    sam_file.write_bytes(b"sam")
    results_gdx = tmp_path / "Results.gdx"
    results_gdx.write_bytes(b"results")

    with pytest.raises(ValueError, match="base"):
        build_gams_nlp_reference_manifest(
            gms_script=gms_script,
            sam_file=sam_file,
            results_gdx=results_gdx,
            scenario_slices={"sim1": "sim1"},
        )
