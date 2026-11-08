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
    from equilibria.sources import GAMS_TOOLS_DIR

    folder = (
        GAMS_TOOLS_DIR
        if (GAMS_TOOLS_DIR / f"{builder}.py").exists()
        else ROOT / "scripts" / "gtap"
    )
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "gtap"))
    monkeypatch.syspath_prepend(str(GAMS_TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(builder, folder / f"{builder}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.UPSTREAM) == GAMS_MODEL_DIR


TOOLS = (
    "build_gtap7_pure_local_bundle.py",
    "build_gtap7_altertax_neos_bundle.py",
    "gdx_to_gams_inline.py",
    "regen_gams_reference.py",
    "regenerate_gams.py",
    "nl_standalone.py",
)


@pytest.mark.parametrize("name", TOOLS)
def test_generators_live_with_the_model(name: str) -> None:
    # every result hashes the model folder whole, so a generator inside it
    # makes the results it produced stale when it changes
    from equilibria.sources import GAMS_TOOLS_DIR

    assert GAMS_TOOLS_DIR == GAMS_MODEL_DIR / "tools"
    assert (GAMS_TOOLS_DIR / name).is_file()
    assert not (ROOT / "scripts" / "gtap" / name).exists()


def _model_hashes(folder: Path) -> dict[str, str]:
    from equilibria.sources import own_files, sha256_file

    return {p.relative_to(folder).as_posix(): sha256_file(p) for p in own_files(folder)}


def test_tools_change_the_model_hash(tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "gams"
    shutil.copytree(GAMS_MODEL_DIR, copy)
    before = _model_hashes(copy)
    # regen_gams_reference.py solves each job; the job list itself (JOB_LISTS) is
    # not a result input, see test_sources_job_lists.py
    tool = copy / "tools" / "regen_gams_reference.py"
    tool.write_text(tool.read_text() + "\n# edited\n")
    assert _model_hashes(copy) != before


def test_pycache_is_not_part_of_a_record(tmp_path: Path) -> None:
    # running a tool writes __pycache__; that must not make results stale
    import shutil

    copy = tmp_path / "gams"
    shutil.copytree(GAMS_MODEL_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = _model_hashes(copy)
    cache = copy / "tools" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "regenerate_gams.cpython-313.pyc").write_bytes(b"\x00")
    assert _model_hashes(copy) == before
