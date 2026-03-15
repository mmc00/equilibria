from __future__ import annotations

from pathlib import Path

from equilibria.qa.sam_checks import run_sam_qa_from_file
from equilibria.templates.pep_cri_transform import (
    should_apply_cri_pep_fix,
    transform_sam_to_pep_compatible,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CRI_SAM = REPO_ROOT / "src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx"


def test_should_apply_cri_pep_fix_skips_fixed_and_pep_compatible_names() -> None:
    assert should_apply_cri_pep_fix("SAM-CRI-gams.xlsx", mode="auto") is True
    assert should_apply_cri_pep_fix("SAM-CRI-gams-fixed.xlsx", mode="auto") is False
    assert should_apply_cri_pep_fix("SAM-CRI-gams-pep-compatible.xlsx", mode="auto") is False


def test_transform_sam_to_pep_compatible_passes_repo_cri_qa(tmp_path: Path) -> None:
    output = tmp_path / "sam-cri-pep-compatible.xlsx"
    report_json = tmp_path / "sam-cri-pep-compatible-report.json"

    report = transform_sam_to_pep_compatible(
        input_sam=CRI_SAM,
        output_sam=output,
        report_json=report_json,
    )

    assert output.exists()
    assert report_json.exists()
    assert report["pipeline"] == "cgebabel_balance_preserving_from_pep_layout"
    assert report["after"]["balance"]["max_row_col_abs_diff"] < 1e-6

    qa = run_sam_qa_from_file(
        sam_file=output,
        dynamic_sam=True,
        accounts={
            "gvt": "gvt",
            "row": "row",
            "td": "td",
            "ti": "ti",
            "tm": "tm",
            "tx": "tx",
            "inv": "inv",
            "vstk": "vstk",
        },
        strict_structural=False,
    )
    assert qa.passed is True
