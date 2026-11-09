"""altertax_job.py: GEMPACK altertax references, RunGTAP's button and GAMS's parameters."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from equilibria.sources import DATA_ROOT, GEMPACK_MODEL_DIR, SourceError, find_reference, reference

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
import altertax_job as job  # noqa: E402
import regenerate_altertax as regen  # noqa: E402
import run_gempack_matrix as matrix  # noqa: E402

REGIONS = ["USA", "EU_28", "ROW"]
DATASETS = regen.DATASETS
RECORDED = [*DATASETS, "9x10", "gtap7_20x41", "nus333"]  # the last 3 have no GAMS altertax
PARAMETERS = regen.PARAMETERS


def _tokens(block: str) -> set[str]:
    return set(block.replace(";", " ").lower().split())


def test_closure_is_the_standard_one_plus_the_altertax_specification() -> None:
    added = _tokens(job.altertax_closure(REGIONS)) - _tokens(matrix.EXOG_BLOCK)
    removed = _tokens(matrix.EXOG_BLOCK) - _tokens(job.altertax_closure(REGIONS))
    assert added == {'del_tbalry("usa")', 'del_tbalry("eu_28")', 'cgdslack("row")'}
    assert removed == {"cgdslack"}


def test_shock_raises_the_rate_ten_percent() -> None:
    vcif = np.array([[[100.0, 0.0], [200.0, 50.0]]])
    vmsb = np.array([[[125.0, 0.0], [200.0, 55.0]]])
    shock = job.tms_shock(vmsb, vcif)
    power = 1 + shock / 100
    assert 125.0 / 100.0 * power[0, 0, 0] == pytest.approx(1 + 0.25 * 1.1)
    assert 55.0 / 50.0 * power[0, 1, 1] == pytest.approx(1 + 0.10 * 1.1)
    assert shock[0, 0, 1] == 0  # no flow
    assert shock[0, 1, 0] == 0  # no tariff


def test_shock_text_is_one_table_per_destination() -> None:
    shock = np.arange(12.0).reshape(2, 2, 3)
    lines = job.shock_text(shock).splitlines()
    assert lines[0].startswith('2 2 3 Real SpreadSheet Header "N002"')
    assert lines[1:3] == ["0,3", "6,9"]  # commodity rows, source columns, destination 1
    assert lines[3] == "1,4"


def test_ten_jobs_with_their_folder_names() -> None:
    found = regen.jobs(DATASETS)
    assert len(found) == 10
    assert {j.folder.name for j in found} == {
        "gempack_altertax-rungtap_tms-rate10_s8-16-32",
        "gempack_altertax-gams_tms-rate10_s8-16-32",
    }


@pytest.mark.parametrize("dataset", RECORDED)
@pytest.mark.parametrize("parameters", PARAMETERS)
def test_recorded_parameters(dataset: str, parameters: str) -> None:
    folder = find_reference(dataset, "gempack", mode="altertax", parameters=parameters)
    if folder is None:
        pytest.skip("the altertax references are generated on the Windows machine")
    job.check_parameters(
        folder / job.PRM_NAME, DATA_ROOT / "gtap7" / dataset / "default.prm", parameters
    )
    worst = job.check_rates(folder / "updated.har", DATA_ROOT / "gtap7" / dataset / "basedata.har")
    assert worst < 1e-4


@pytest.mark.parametrize("dataset", DATASETS)
def test_matrix_references_stay_unique(dataset: str) -> None:
    if find_reference(dataset, "gempack", mode="altertax", parameters="gams") is None:
        pytest.skip("the altertax references are generated on the Windows machine")
    assert reference(dataset, "gempack", shock=10, steps="8-16-32", closure="capFlex")
    with pytest.raises(SourceError):
        reference(dataset, "gempack", mode="altertax")  # both sets of parameters match
