from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.baseline.compatibility import (
    evaluate_strict_gams_baseline_compatibility,
)
from equilibria.baseline.manifest import build_baseline_manifest, load_baseline_manifest
from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver


@pytest.fixture(scope="module")
def pep2_state():
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")
    calibrator = PEPModelCalibratorExcel(sam_file=sam_file, val_par_file=val_file)
    return calibrator.calibrate()


def test_baseline_manifest_roundtrip(tmp_path: Path, pep2_state) -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")
    results_gdx = Path("src/equilibria/templates/reference/pep2/scripts/Results_ipopt_excel_reference.gdx")

    manifest = build_baseline_manifest(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="base",
        sam_file=sam_file,
        val_par_file=val_file,
    )
    out = tmp_path / "baseline_manifest.json"
    manifest.save_json(out)

    loaded = load_baseline_manifest(out)
    assert loaded.schema_version == "pep_baseline_manifest/v1"
    assert loaded.gams_slice == "base"
    assert loaded.results_gdx_sha256
    assert loaded.sam_sha256
    assert loaded.val_par_sha256
    assert loaded.set_sizes["I"] >= 1
    assert loaded.state_anchors["GDP_BP"] > 0


def test_strict_baseline_compatibility_passes_with_matching_manifest(
    tmp_path: Path,
    pep2_state,
) -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")
    results_gdx = Path("src/equilibria/templates/reference/pep2/scripts/Results_ipopt_excel_reference.gdx")
    manifest_path = tmp_path / "strict_manifest.json"
    build_baseline_manifest(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="base",
        sam_file=sam_file,
        val_par_file=val_file,
    ).save_json(manifest_path)

    report = evaluate_strict_gams_baseline_compatibility(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="base",
        manifest_path=manifest_path,
        sam_file=sam_file,
        val_par_file=val_file,
        rel_tol=1e-4,
        require_manifest=True,
    )
    assert report.passed


def test_strict_baseline_compatibility_detects_slice_mismatch(
    tmp_path: Path,
    pep2_state,
) -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")
    results_gdx = Path("src/equilibria/templates/reference/pep2/scripts/Results_ipopt_excel_reference.gdx")
    manifest_path = tmp_path / "strict_manifest.json"
    build_baseline_manifest(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="var",
        sam_file=sam_file,
        val_par_file=val_file,
    ).save_json(manifest_path)

    report = evaluate_strict_gams_baseline_compatibility(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="sim1",
        manifest_path=manifest_path,
        sam_file=sam_file,
        val_par_file=val_file,
        rel_tol=1e-4,
        require_manifest=True,
    )
    failed_codes = {check.code for check in report.checks if not check.passed}
    assert not report.passed
    assert "BSL_MANIFEST_SLICE" in failed_codes


def test_strict_baseline_compatibility_allows_base_manifest_for_sim1(
    tmp_path: Path,
    pep2_state,
) -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")
    results_gdx = Path("src/equilibria/templates/reference/pep2/scripts/Results_ipopt_excel_reference.gdx")
    manifest_path = tmp_path / "strict_manifest_base.json"
    build_baseline_manifest(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="base",
        sam_file=sam_file,
        val_par_file=val_file,
    ).save_json(manifest_path)

    report = evaluate_strict_gams_baseline_compatibility(
        state=pep2_state,
        results_gdx=results_gdx,
        gams_slice="sim1",
        manifest_path=manifest_path,
        sam_file=sam_file,
        val_par_file=val_file,
        rel_tol=1e-4,
        require_manifest=True,
    )
    assert report.passed


def test_gams_solver_gate_raises_on_incompatible_baseline() -> None:
    sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx")
    val_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx")
    results_gdx = Path("src/equilibria/templates/reference/pep2/scripts/Results_ipopt_excel_reference.gdx")
    calibrator = PEPModelCalibratorExcel(sam_file=sam_file, val_par_file=val_file)
    state = calibrator.calibrate()

    solver = PEPModelSolver(
        calibrated_state=state,
        init_mode="gams",
        gams_results_gdx=results_gdx,
        gams_results_slice="sim1",
        enforce_strict_gams_baseline=True,
        sam_file=sam_file,
        val_par_file=val_file,
    )

    with pytest.raises(RuntimeError):
        solver._create_initial_guess()
