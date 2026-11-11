"""regenerate_paper.py: the phase 3a-2 jobs (the old corridas_paper steps 1-7 and runs folder).

Each job is recorded by write_source.py, as in 3a-1. The groups and their
counts are those of specs/data/changes/one-data-folder/plan-phase-3a2.md.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
from tests.data._records import data_record, result_record, write_record

from equilibria import sources
from equilibria.sources import DATA_ROOT, iter_sources, own_files, reference

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "data" / "gtap7" / "gempack" / "tools" / "regenerate_paper.py"
RELEASES = ["--gempack-release", "12.1.003", "--gp-release", "11.3.004"]
COUNTS = {
    "book": 39,
    "book-sltoht": 39,
    "levels": 6,
    "horridge": 3,
    "levels6": 12,
    "gragg": 45,
    "gragg-sltoht": 45,
    "swap": 11,
    "swap-sltoht": 2,
    "validation": 1,
}
COMPILED = (".gss", ".gst", ".gsc", ".exe", ".dll")


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("regenerate_paper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _exes(tmp_path: Path) -> list[str]:
    gtapdir, gp = tmp_path / "runGTAP375", tmp_path / "GP"
    for exe in (gtapdir / "gtapv7.exe", gtapdir / "gtapuv7.exe",
                gp / "tablo.exe", gp / "gemsim.exe", gp / "sltoht.exe"):  # fmt: skip
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"MZ")
    return [
        "--gtapdir",
        str(gtapdir),
        "--gp",
        str(gp),
        "--work",
        str(tmp_path / "work"),
    ]


def _tree(tmp_path: Path) -> Path:
    """A data root with the dataset folders the jobs hash, each recorded."""
    root = tmp_path / "data"
    for ds in ("nus333", "9x10", "gtap7_3x3", "gtap7_20x41"):
        folder = root / "gtap7" / ds
        folder.mkdir(parents=True)
        (folder / "basedata.har").write_bytes(b"data")
        write_record(folder, data_record(dataset=ds, outputs={}))
    gempack = root / "gtap7" / "gempack"
    shutil.copytree(DATA_ROOT / "gtap7" / "gempack", gempack)
    return root


def test_job_counts() -> None:
    m = _module()
    js = m.jobs(list(COUNTS))
    assert {g: sum(j.group == g for j in js) for g in COUNTS} == COUNTS
    assert len(js) == sum(COUNTS.values()) == 203
    assert len({j.folder for j in js}) == len(js)


def test_tools_and_releases() -> None:
    m = _module()
    tools = {j.group: j.tool for j in m.jobs(list(COUNTS))}
    assert tools == {
        "book": "gempack", "book-sltoht": "sltoht", "levels": "sltoht",
        "horridge": "gemsim", "levels6": "gemsim", "gragg": "gemsim",
        "gragg-sltoht": "sltoht", "swap": "gempack", "swap-sltoht": "sltoht",
        "validation": "gempack",
    }  # fmt: skip
    job = next(j for j in m.jobs(["levels"]) if j.dataset == "gtap7_3x3")
    cmd = m.job_command(job, gempack_release="12.1.003", gp_release="11.3.004")
    assert cmd[cmd.index("--tool-version") + 1] == "11.3.004"
    externals = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--external"]
    assert any(
        e.startswith("gtapv7.exe=") and e.endswith("@12.1.003") for e in externals
    )
    assert any(
        e.startswith("sltoht.exe=") and e.endswith("@11.3.004") for e in externals
    )


def test_matrix_references_stay_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every 3a-1 and 3a-2 record resolves to itself through reference()."""
    m = _module()
    root = tmp_path / "data"
    matrix = [
        r for r in iter_sources()
        if r.kind == "result" and r.tool == "gempack"
        and r.variant.get("closure") in ("capFlex", "capFix")
    ]  # fmt: skip
    assert len(matrix) == 66
    for r in matrix:
        write_record(root / r.path.relative_to(DATA_ROOT), dict(r.raw))
    for j in m.jobs(list(COUNTS), root=root):
        write_record(j.folder, result_record(dataset=j.dataset, tool=j.tool,
                                             variant=dict(j.variant)))  # fmt: skip
    records = list(iter_sources(root))
    # reference() itself, without reading the 265 records again for every call
    monkeypatch.setattr(sources, "iter_sources", lambda root=root: iter(records))
    for r in records:
        if r.kind == "result":
            assert reference(r.dataset, r.tool, root=root, **r.variant) == r.path  # ty: ignore[invalid-argument-type]


def test_no_compiled_gempack_files_in_data(tmp_path: Path) -> None:
    m = _module()
    assert [p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in COMPILED] == []
    work = tmp_path / "work"
    for j in m.jobs(list(COUNTS), work=work):
        for path in j.externals.values():
            assert not Path(path).resolve().is_relative_to(DATA_ROOT), (j.folder, path)
        assert j.folder.resolve().is_relative_to(DATA_ROOT)


def _absolute_one_separator(value: str) -> bool:
    """Absolute on this OS or as a Windows path, with one kind of separator."""
    one = "\\" not in value or "/" not in value
    return one and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())


def test_command_paths_are_absolute(tmp_path: Path) -> None:
    m = _module()
    for j in m.jobs(list(COUNTS), work=tmp_path / "work"):
        cmd = m.job_command(j, gempack_release="12.1.003", gp_release="11.3.004")
        for arg in cmd:
            value = arg.split("=", 1)[-1].split("@", 1)[0]
            if "/" in value or "\\" in value:
                assert _absolute_one_separator(value), (j.folder.name, arg)


def test_windows_exe_paths_off_windows() -> None:
    """On Linux/macOS Path(r"C:\runGTAP375") is a relative POSIX path, and
    joining it gives C:\runGTAP375/gtapuv7.exe. PurePosixPath reproduces that here."""
    m = _module()
    js = m.jobs(list(COUNTS), gtapdir=PurePosixPath(r"C:\runGTAP375"),
                gp=PurePosixPath(r"C:\GP"))  # fmt: skip
    exes = {str(v) for j in js for k, v in j.externals.items() if k.endswith(".exe")}
    assert exes == {
        r"C:\runGTAP375\gtapv7.exe", r"C:\runGTAP375\gtapuv7.exe",
        r"C:\GP\tablo.exe", r"C:\GP\gemsim.exe", r"C:\GP\sltoht.exe",
    }  # fmt: skip


def test_release_flags_are_required() -> None:
    with pytest.raises(SystemExit) as e:
        _module().main(["--dry-run", "--gp-release", "11.3.004"])
    assert e.value.code == 2


def test_dry_run_lists_every_job(capsys: pytest.CaptureFixture[str]) -> None:
    assert _module().main(["--dry-run", *RELEASES]) == 0
    assert "203 job(s)" in capsys.readouterr().out


def test_resume_skips_recorded_folders(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)
    done = root / "gtap7" / "nus333" / "gempack_tm10_s8-16-32_dpsaveswap"
    write_record(done, result_record(dataset="nus333", tool="gempack"))
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    rc = m.main(
        [*RELEASES, "--group", "swap", "--data-root", str(root), *_exes(tmp_path)]
    )
    assert rc == 0
    assert len(ran) == COUNTS["swap"] - 1
    assert not any(str(done) in c for cmd in ran for c in cmd)


def test_failed_nested_job_is_removed(tmp_path: Path) -> None:
    """A failed job under nus333/ leaves nothing that the dataset record would count."""
    m = _module()
    root = _tree(tmp_path)
    nus333 = root / "gtap7" / "nus333"
    before = own_files(nus333)

    def fail(cmd: list[str]) -> int:
        out = Path(cmd[cmd.index("--folder") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "solution.sl4").write_bytes(b"half")
        return 1

    m.run_job = fail
    rc = m.main([*RELEASES, "--group", "book", "--only", "me9c",
                 "--data-root", str(root), *_exes(tmp_path)])  # fmt: skip
    assert rc == 1
    assert not (nus333 / "gempack_burfisher" / "me9c").exists()
    assert own_files(nus333) == before


def test_partial_folder_is_cleaned_before_run(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)
    partial = root / "gtap7" / "nus333" / "gempack_burfisher" / "me9c"
    partial.mkdir(parents=True)
    (partial / "solution.sl4").write_bytes(b"half")
    seen: list[bool] = []
    m.run_job = lambda cmd: seen.append((partial / "solution.sl4").exists()) or 0
    m.main([*RELEASES, "--group", "book", "--only", "me9c",
            "--data-root", str(root), *_exes(tmp_path)])  # fmt: skip
    assert seen == [False]


def test_prerequisite_group_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """book-sltoht needs the book solution of the same experiment recorded first."""
    m = _module()
    root = _tree(tmp_path)
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    rc = m.main([*RELEASES, "--group", "book-sltoht", "--only", "me9c",
                 "--data-root", str(root), *_exes(tmp_path)])  # fmt: skip
    assert rc == 1 and ran == []
    assert "gempack_burfisher" in capsys.readouterr().out
    write_record(root / "gtap7" / "nus333" / "gempack_burfisher" / "me9c",
                 result_record(dataset="nus333", tool="gempack"))  # fmt: skip
    assert m.main([*RELEASES, "--group", "book-sltoht", "--only", "me9c",
                   "--data-root", str(root), *_exes(tmp_path)]) == 0  # fmt: skip
    assert len(ran) == 1


def test_unknown_only_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _module().main(["--dry-run", *RELEASES, "--group", "book", "--only", "me99"])
        == 2
    )
    assert "me99" in capsys.readouterr().err


def test_missing_executables_stop_before_job_one(tmp_path: Path) -> None:
    m = _module()
    root = _tree(tmp_path)
    ran: list[list[str]] = []
    m.run_job = lambda cmd: ran.append(cmd) or 0
    rc = m.main([*RELEASES, "--group", "book", "--data-root", str(root),
                 "--gtapdir", str(tmp_path / "nowhere"), "--gp", str(tmp_path / "nowhere")])  # fmt: skip
    assert rc == 2 and ran == []


def test_default_work_names_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """--work ends up in external paths and the command, so it is not the temp folder
    (Windows' %TEMP% is C:/Users/<name>/...): it is the clone's git-ignored
    runs/gempack_work, never under data/. The clone itself may be under $HOME
    (macOS/Linux), which the records of the Windows machine never are."""
    monkeypatch.delenv("EQUILIBRIA_GEMPACK_WORK", raising=False)
    # macOS/Linux clones live under $HOME (/Users/<name>/...): reproduce that here
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: ROOT.parent))
    work = _module()._default_work().resolve()
    assert work == (ROOT / "runs" / "gempack_work").resolve()
    assert not work.is_relative_to(Path(tempfile.gettempdir()).resolve())
    assert not work.is_relative_to(DATA_ROOT)


def test_every_command_takes_the_retry_note() -> None:
    """A job that retried a program that did not start leaves a note for the record."""
    m = _module()
    for j in m.jobs(list(COUNTS)):
        cmd = m.job_command(j, gempack_release="12.1.003", gp_release="11.3.004")
        assert cmd[cmd.index("--notes-from") + 1] == "retry_note.txt"
