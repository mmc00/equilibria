"""regenerate_gempack.py: the 66 GEMPACK matrix jobs, each recorded by write_source."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.data._records import data_record, result_record, write_record

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "data" / "gtap7" / "gempack" / "tools" / "regenerate_gempack.py"
G, S = Path("/opt/gp/gtapv7.exe"), Path("/opt/gp/sltoht.exe")


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQUILIBRIA_GEMPACK_WORK", str(tmp_path / "work"))


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("regenerate_gempack", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _job(m: Any, ds: str, folder_name: str) -> Any:
    return next(j for j in m.jobs([ds]) if j.folder.name == folder_name)


def test_sixty_six_jobs() -> None:
    m = _module()
    js = m.jobs(m.DATASETS)
    assert len(js) == 66
    assert sum(j.closure == "capFix" for j in js) == 6
    assert {j.folder.name for j in js if j.dataset == "gtap7_3x3"} >= {
        "gempack_tm10_s8-16-32_capflex",
        "gempack_tm0p1_s8-16-32_capflex",
        "gempack_tm10_s64_capflex",
        "gempack_tm10_s8-16-32_capfix",
    }


def test_command_records_the_variant() -> None:
    m = _module()
    job = _job(m, "gtap7_3x3", "gempack_tm10_s8-16-32_capfix")
    cmd = m.job_command(job, gtapv7=G, sltoht=S, tool_version="11.3")
    i = cmd.index("--variant")
    assert cmd[i : i + 6] == [
        "--variant", "shock=10", "--variant", 'steps="8-16-32"',
        "--variant", "closure=capFix",
    ]  # fmt: skip
    assert cmd[cmd.index("--tool-version") + 1] == "11.3"
    assert cmd[cmd.index("--out") + 1] == str(job.folder)
    assert cmd[cmd.index("--rordelta") + 1] == "0"


def test_command_records_the_generators() -> None:
    # the result hashes data/gtap7/gempack whole: its tools/ are what ran
    m = _module()
    job = _job(m, "gtap7_3x3", "gempack_tm10_s4_capflex")
    cmd = m.job_command(job, gtapv7=G, sltoht=S, tool_version="12.1.003")
    assert cmd[cmd.index("--model-file") + 1] == str(m.DATA_ROOT / "gtap7" / "gempack")
    assert str(m.MATRIX_SCRIPT).startswith(
        str(m.DATA_ROOT / "gtap7" / "gempack" / "tools")
    )


def test_command_paths_are_absolute() -> None:
    m = _module()
    job = _job(m, "gtap7_3x3", "gempack_tm10_s4_capflex")
    cmd = m.job_command(job, gtapv7=G, sltoht=S, tool_version="11.3")
    paths = [a for a in cmd if a.endswith((".py", ".exe")) or "/" in a or "\\" in a]
    assert paths and all(Path(a).is_absolute() for a in paths), paths


def _tree(tmp: Path) -> Path:
    write_record(tmp / "gtap7" / "gtap7_3x3", data_record())
    return tmp


def _exes(tmp: Path) -> list[str]:
    (tmp / "gtapv7.exe").write_text("")
    (tmp / "sltoht.exe").write_text("")
    return ["--gtapv7", str(tmp / "gtapv7.exe"), "--sltoht", str(tmp / "sltoht.exe")]


def test_resume_skips_recorded_folders(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)
    done = root / "gtap7" / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex"
    write_record(done, result_record(tool="gempack"))
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    m.main(
        [
            "--tool-version",
            "11.3",
            "--datasets",
            "gtap7_3x3",
            "--data-root",
            str(root),
            *_exes(tmp_path),
        ]
    )
    assert len(ran) == 10
    assert not any(str(done) in c for cmd in ran for c in cmd)


def test_failed_job_folder_is_removed(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)

    def fail(cmd: list[str]) -> int:
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "updated.har").write_bytes(b"half")
        return 1

    m.run_job = fail
    rc = m.main(["--tool-version", "11.3", "--datasets", "gtap7_3x3",
                 "--only", "tm10_s8-16-32_capflex", "--data-root", str(root), *_exes(tmp_path)])  # fmt: skip
    assert rc == 1
    assert not (root / "gtap7" / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex").exists()


def test_partial_folder_is_cleaned_before_run(tmp_path: Path) -> None:
    """A crash mid-run leaves files without source.yaml; they are not inputs."""
    m = _module()
    root = _tree(tmp_path)
    partial = root / "gtap7" / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex"
    partial.mkdir()
    (partial / "updated.har").write_bytes(b"half")
    seen: list[bool] = []
    m.run_job = lambda cmd: seen.append((partial / "updated.har").exists()) or 0
    m.main(["--tool-version", "11.3", "--datasets", "gtap7_3x3",
            "--only", "tm10_s8-16-32_capflex", "--data-root", str(root), *_exes(tmp_path)])  # fmt: skip
    assert seen == [False]


def test_only_filters_by_tag(capsys: pytest.CaptureFixture[str]) -> None:
    m = _module()
    assert m.main(["--dry-run", "--tool-version", "11.3", "--datasets", "gtap7_3x3",
                   "--only", "tm10_s8-16-32_capflex"]) == 0  # fmt: skip
    assert "1 job(s)" in capsys.readouterr().out


def test_tool_version_is_required() -> None:
    with pytest.raises(SystemExit) as e:
        _module().main(["--dry-run"])
    assert e.value.code == 2


def test_end_to_end_with_fake_gempack(tmp_path: Path) -> None:
    """driver -> write_source -> run_gempack_matrix --out, with fake executables:
    the folder gets a valid, verified source.yaml and the tree checks clean."""
    import shutil

    from tests.scripts.test_run_gempack_matrix_out import _fake_gtapv7, _fake_sltoht

    from equilibria.sources import (
        GEMPACK_MODEL_DIR,
        check_tree,
        dataset_dir,
        load_source,
    )

    root = tmp_path / "data"
    shutil.copytree(dataset_dir("gtap7_3x3"), root / "gtap7" / "gtap7_3x3")
    shutil.copytree(
        GEMPACK_MODEL_DIR, root / "gtap7" / "gempack",
        ignore=shutil.ignore_patterns("__pycache__"),
    )  # fmt: skip
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    m = _module()
    rc = m.main([
        "--tool-version", "12.1.003", "--datasets", "gtap7_3x3",
        "--only", "tm10_s4_capflex", "--data-root", str(root),
        "--gtapv7", str(_fake_gtapv7(bin_dir)), "--sltoht", str(_fake_sltoht(bin_dir)),
    ])  # fmt: skip
    assert rc == 0
    folder = root / "gtap7" / "gtap7_3x3" / "gempack_tm10_s4_capflex"
    rec = load_source(folder)
    assert rec.kind == "result" and rec.tool == "gempack" and rec.status == "verified"
    assert dict(rec.variant) == {"shock": 10, "steps": "4", "closure": "capFlex"}
    assert set(rec.raw["outputs"]) == {
        "updated.har", "decomp.har", "sl4dump.har", "run.cmf",
        "solve_console.txt", "sltoht_console.txt",
    }  # fmt: skip
    assert rec.raw["generated_from"]["model"]["path"] == "gtap7/gempack"
    assert check_tree(root) == []


def test_command_checks_the_release() -> None:
    m = _module()
    job = _job(m, "gtap7_3x3", "gempack_tm10_s4_capflex")
    cmd = m.job_command(job, gtapv7=G, sltoht=S, tool_version="12.1.003")
    assert cmd[cmd.index("--expect-release") + 1] == "12.1.003"


def test_truncated_record_is_rerun(tmp_path: Path) -> None:
    """A source.yaml cut short (killed mid-write) is not 'recorded': rerun it."""
    m = _module()
    root = _tree(tmp_path)
    cut = root / "gtap7" / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex"
    cut.mkdir()
    (cut / "source.yaml").write_text("kind: result\nmodel: gt")
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    m.main(["--tool-version", "11.3", "--datasets", "gtap7_3x3",
            "--only", "tm10_s8-16-32_capflex", "--data-root", str(root),
            *_exes(tmp_path)])  # fmt: skip
    assert len(ran) == 1
    assert not (cut / "source.yaml").exists()


def test_unknown_only_tag_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    m = _module()
    rc = m.main(
        ["--dry-run", "--tool-version", "11.3", "--only", "tm10_s8-16-32_capflx"]
    )
    assert rc == 2
    assert "tm10_s8-16-32_capflex" in capsys.readouterr().err


def test_missing_executables_stop_before_job_one(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    rc = m.main(["--tool-version", "11.3", "--datasets", "gtap7_3x3",
                 "--data-root", str(root), "--gtapv7", str(tmp_path / "no.exe"),
                 "--sltoht", str(tmp_path / "no2.exe")])  # fmt: skip
    assert rc == 2 and ran == []
