"""burfisher_job.py: Burfisher's book experiments solved by RunGTAP's gtapuv7.exe.

The input of corridas_paper.ps1 steps 1 and 6a, regenerated into
data/gtap7/nus333/gempack_burfisher/<exp>/ (phase 3a-2 Task 3). As
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
import burfisher_job  # noqa: E402

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


def test_cmf_header_then_exp_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, seen = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 0
    cmf = _cmf(tmp_path)
    exp = (NUS333 / "me9c.exp").read_bytes().decode("latin-1")
    assert cmf.endswith(exp)  # appended verbatim, CRLF and !@ lines included
    aux = str(tmp_path / "runGTAP375" / "gtapuv7")
    assert cmf.startswith(f'auxiliary files = "{aux}";')
    assert seen["cmd"][0] == str(tmp_path / "runGTAP375" / "gtapuv7.exe")


def test_gtapv7_runs_the_condensed_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, seen = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "gtapv7", "gtapv7.exe")) == 0
    assert _cmf(tmp_path).startswith('auxiliary files = "' + str(tmp_path / "runGTAP375" / "gtapv7"))


def test_dir_exp_for_me8_and_tbl813(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me8")) == 0
    assert _cmf(tmp_path).endswith((NUS333 / "me8-dir.exp").read_bytes().decode("latin-1"))


def test_book_method_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "completed without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "tbl45a")) == 0
    assert "Method = Johansen;" in _cmf(tmp_path).replace("method", "Method")


def test_runs_in_a_work_copy_of_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_release_mismatch_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(" <GEMPACK Release 11.3.004   January 2015>\ncompleted without error\n")
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 1


def test_scratch_files_are_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _fake(BANNER + "completed without error\n",
                   (*FILES, "solution.slc", "run-GPXXX3.ud8"))  # fmt: skip
    monkeypatch.setattr(burfisher_job, "run", run)
    assert burfisher_job.main(_argv(tmp_path, "me9c")) == 0
    left = {p.name for p in (tmp_path / "out").iterdir()}
    assert left == {*FILES, "run.cmf", "solve_console.txt"}
