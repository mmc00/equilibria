"""The GAMS GTAP v7 model lives in data/gtap7/gams/ with a kind: model source."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from equilibria.sources import GAMS_MODEL_DIR, dataset_dir, load_source


@pytest.mark.parametrize(
    "name",
    [
        "comp.gms",
        "comp_altertax.gms",
        "model.gms",
        "cal.gms",
        "getData.gms",
        "altertax_config.gms",
        "cal_dump.gms",
    ],
)
def test_gams_model_file_present(name: str) -> None:
    assert (GAMS_MODEL_DIR / name).is_file()


def test_model_folder_is_kind_model() -> None:
    assert load_source(GAMS_MODEL_DIR).kind == "model"


@pytest.mark.parametrize("name", ["9x10Dat.gdx", "9x10Prm.gdx", "9x10Sets.gdx"])
def test_9x10_gams_inputs_in_dataset_folder(name: str) -> None:
    assert (dataset_dir("9x10") / name).is_file()


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "builder",
    [
        "build_gtap7_altertax_neos_bundle",
        "build_gtap7_pure_local_bundle",
        "build_altertax_neos_bundle",
        "build_9x10_altertax_neos_bundle",
    ],
)
def test_builders_read_the_repository_model(builder: str, monkeypatch) -> None:
    # The results must come from the model whose hash source.yaml records,
    # never from a copy outside the repository.
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "gtap"))
    spec = importlib.util.spec_from_file_location(
        builder, ROOT / "scripts" / "gtap" / f"{builder}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.UPSTREAM) == GAMS_MODEL_DIR
