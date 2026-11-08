"""A driver that only lists the jobs is not an input of the results it runs.

regenerate_gams.py and regenerate_gempack.py enumerate the jobs and call, for
each, a command that its source.yaml records whole. Adding a dataset to the
list changes no recorded result, so it must not make every result stale. Any
other file of the model folder still is an input.
"""

from __future__ import annotations

from pathlib import Path

from equilibria.sources import JOB_LISTS, check_tree, input_files, sha256_file
from tests.data._records import data_record, result_record, write_record

DRIVER = "tools/regenerate_gams.py"
SOLVER = "tools/regen_gams_reference.py"


def _tree(root: Path) -> Path:
    """A GAMS model folder and one result whose record hashes all of it."""
    model = root / "gtap7" / "gams"
    (model / "tools").mkdir(parents=True)
    for rel, text in (("comp.gms", "model"), (DRIVER, "jobs"), (SOLVER, "solve")):
        (model / rel).write_text(text)
    hashes = {rel: sha256_file(model / rel) for rel in ("comp.gms", DRIVER, SOLVER)}
    write_record(model, data_record(dataset="gams", outputs=hashes))
    res = root / "gtap7" / "gtap7_3x3" / "gams_nlp"
    res.mkdir(parents=True)
    (res / "out.gdx").write_bytes(b"gdx")
    write_record(
        res,
        result_record(
            generated_from={"model": {"path": "gtap7/gams", "sha256": hashes}},
            outputs={"out.gdx": sha256_file(res / "out.gdx")},
        ),
    )
    write_record(
        res.parent, data_record(outputs={})
    )  # the dataset folder holds no files of its own here
    return model


def _rerecord_model(model: Path) -> None:
    """The model folder's own record follows its files (it is not the point)."""
    hashes = {
        p.relative_to(model).as_posix(): sha256_file(p)
        for p in sorted(model.rglob("*"))
        if p.is_file() and p.name != "source.yaml"
    }
    write_record(model, data_record(dataset="gams", outputs=hashes))


def test_editing_the_job_list_leaves_results_verified(tmp_path: Path) -> None:
    model = _tree(tmp_path)
    (model / DRIVER).write_text("jobs, plus gtap7_15x10")
    _rerecord_model(model)

    assert check_tree(tmp_path) == []


def test_editing_any_other_tool_still_makes_results_stale(tmp_path: Path) -> None:
    model = _tree(tmp_path)
    (model / SOLVER).write_text("solve differently")
    _rerecord_model(model)

    msgs = check_tree(tmp_path)

    assert len(msgs) == 1
    assert "regen_gams_reference.py changed after the result was generated" in msgs[0]


def test_a_new_record_does_not_hash_the_job_lists(tmp_path: Path) -> None:
    model = _tree(tmp_path)

    names = {p.relative_to(model).as_posix() for p in input_files(model, tmp_path)}

    assert names == {"comp.gms", SOLVER}


def test_the_job_lists_are_the_two_drivers() -> None:
    assert {
        "gtap7/gams/tools/regenerate_gams.py",
        "gtap7/gempack/tools/regenerate_gempack.py",
    } == JOB_LISTS
