"""burfisher_job.py: Burfisher's book experiments solved by RunGTAP's gtapuv7.exe.

regenerate_paper.py's ``book`` group, the input of its ``book-sltoht`` group,
recorded into data/gtap7/nus333/gempack_burfisher/<exp>/ (phase 3a-2 Task 3). As
tools/gempack-oracle/01-run-all.bat did: a header (aux files, sets, base data,
output files) plus the .exp verbatim, run from the package folder, and the
book's method kept (Johansen stays Johansen).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import DATA_ROOT, GEMPACK_MODEL_DIR

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
import burfisher_job  # noqa: E402  # ty: ignore[unresolved-import]
import gemsim_job  # noqa: E402  # ty: ignore[unresolved-import]

NUS333 = DATA_ROOT / "gtap7" / "nus333"
BANNER = " <GEMPACK Release 12.1.003   January 2020>\n"
FILES = ("solution.sl4", "updated.har", "summary.har", "decomp.har", "volume.har")


def _fake(console: str, files: tuple[str, ...] = FILES):
    seen: dict[str, object] = {}

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        seen.update(cmd=cmd, cwd=cwd)
        for f in files:
            (console_path.parent / f).write_bytes(b"x")
        console_path.write_text(console)
        return 0

    return run, seen


def _argv(tmp_path: Path, exp: str, exe: str = "gtapuv7.exe") -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    return ["--experiment", exp, "--package", str(NUS333),
            "--exe", str(tmp_path / "runGTAP375" / exe), "--work", str(tmp_path / "work"),
            "--out", str(out), "--expect-release", "12.1.003"]  # fmt: skip


def _cmf(tmp_path: Path) -> str:
    return (tmp_path / "out" / "run.cmf").read_bytes().decode("latin-1")


def test_cmf_header_then_exp_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, seen = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 0
    cmf = _cmf(tmp_path)
    exp = (NUS333 / "me9c.exp").read_bytes().decode("latin-1")
    assert cmf.endswith(exp)  # appended verbatim, CRLF and !@ lines included
    aux = str(tmp_path / "runGTAP375" / "gtapuv7")
    assert cmf.startswith(f'auxiliary files = "{aux}";')
    assert seen["cmd"][0] == str(tmp_path / "runGTAP375" / "gtapuv7.exe")


def test_gtapv7_runs_the_condensed_exe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, seen = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "gtapv7", "gtapv7.exe")) == 0
    assert _cmf(tmp_path).startswith(
        'auxiliary files = "' + str(tmp_path / "runGTAP375" / "gtapv7")
    )


def test_dir_exp_for_me8_and_tbl813(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me8")) == 0
    assert _cmf(tmp_path).endswith(
        (NUS333 / "me8-dir.exp").read_bytes().decode("latin-1")
    )


def test_book_method_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "tbl45a")) == 0
    assert "Method = Johansen;" in _cmf(tmp_path).replace("method", "Method")


def test_runs_in_a_work_copy_of_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .exp names its .prm without a path; the copy is in --work, never in data/
    or %TEMP% (the .sl4 records the current directory)."""
    run, seen = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 0
    assert Path(str(seen["cwd"])) == tmp_path / "work" / "package" / "me9c"
    assert not (tmp_path / "work" / "package" / "me9c").exists()


def test_needs_normal_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "Accuracy too low\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me4a")) == 1


def test_needs_its_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "completed without error\n", ("solution.sl4",))
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 1


def test_release_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _fake(
        " <GEMPACK Release 11.3.004   January 2015>\ncompleted without error\n"
    )
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 1


def test_scratch_files_are_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _fake(BANNER + "completed without error\n",
                   (*FILES, "solution.slc", "solution.aal", "run-GPXXX3.ud8"))  # fmt: skip
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 0
    left = {p.name for p in (tmp_path / "out").iterdir()}
    assert left == {*FILES, "run.cmf", "solve_console.txt"}


def _starts_on(attempt: int, files: tuple[str, ...] = FILES):
    """A program that does not start (empty console, no banner) until ``attempt``."""
    calls: list[int] = []

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        calls.append(len(calls) + 1)
        if len(calls) < attempt:
            console_path.write_text("")
            return -1073741502  # 0xC0000142, STATUS_DLL_INIT_FAILED
        for f in files:
            (console_path.parent / f).write_bytes(b"x")
        console_path.write_text(BANNER + "completed without error\n")
        return 0

    return run, calls


def test_retries_once_when_the_program_does_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3a-1 and me4b: now and then gtapuv7.exe never starts (no banner). One retry,
    and the record says so (the note write_source --notes-from picks up)."""
    run, calls = _starts_on(2)
    monkeypatch.setattr(burfisher_job, "run", run)
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 0)
    assert burfisher_job.main(_argv(tmp_path, "me4b")) == 0
    assert calls == [1, 2]
    note = (tmp_path / "out" / burfisher_job.NOTE_FILE).read_text()
    assert "did not start" in note and "retried once" in note


def test_a_second_failed_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, calls = _starts_on(3)
    monkeypatch.setattr(burfisher_job, "run", run)
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 0)
    assert burfisher_job.main(_argv(tmp_path, "me4b")) == 1
    assert calls == [1, 2]


def test_a_solve_that_started_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a program that never started is retried; a failed solve is a result."""
    calls: list[int] = []

    def run(cmd: list[str], cwd: Path, console_path: Path) -> int:
        calls.append(1)
        console_path.write_text(BANNER + "Accuracy too low\n")
        return 1

    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me4a")) == 1
    assert calls == [1]
