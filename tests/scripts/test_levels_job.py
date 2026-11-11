"""levels_job.py: sltoht on GEMPACK solutions (steps 1, 2, 6, 7 and the swap levels).

``dump``    a recorded solution folder: its sl4dump.har and, with --levels, the
            levels export cross-checked against that same dump (Tasks 4 and 9).
``resolve`` re-solves a gtapv7.exe job keeping its .sl4 and exports levels
            against the RECORDED sl4dump.har: a mismatch means the re-solve is not
            the recorded solution, and nothing is recorded (Tasks 5 and 10).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import DATA_ROOT, GEMPACK_MODEL_DIR

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import gemsim_job  # noqa: E402  # ty: ignore[unresolved-import]
import levels_job  # noqa: E402  # ty: ignore[unresolved-import]
from _fake_gempack import FakeGempack  # noqa: E402

TOOLS = GEMPACK_MODEL_DIR / "tools"
MATRIX = DATA_ROOT / "gtap7" / "gtap7_3x3" / "gempack_tm10_s8-16-32_capflex"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeGempack:
    f = FakeGempack()
    monkeypatch.setattr(gemsim_job, "run", f)
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 0)
    return f


def _solution(tmp_path: Path) -> Path:
    sol = tmp_path / "data" / "gempack_burfisher" / "me9c"
    sol.mkdir(parents=True)
    (sol / "solution.sl4").write_bytes(b"sl4")
    return sol


def _dump(tmp_path: Path, *extra: str) -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    return ["dump", "--solution", str(_solution(tmp_path)), "--sltoht", "C:/GP/sltoht.exe",
            *extra, "--work", str(tmp_path / "work"), "--out", str(out),
            "--expect-release", "11.3.004"]  # fmt: skip


def _names(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir()}


def test_dump_writes_the_sl4dump(tmp_path: Path, fake: FakeGempack) -> None:
    assert levels_job.main(_dump(tmp_path)) == 0
    assert _names(tmp_path / "out") == {"sl4dump.har", "sltoht_console.txt"}
    # sltoht runs on a copy in --work: the console never names a data/ or %TEMP% path
    assert all(cwd.is_relative_to(tmp_path / "work") for _, cwd in fake.calls)


def test_dump_with_levels(tmp_path: Path, fake: FakeGempack) -> None:
    assert levels_job.main(_dump(tmp_path, "--levels")) == 0
    assert _names(tmp_path / "out") == {
        "sl4dump.har", "sltoht_console.txt", "sl4levels.har", "export_console.txt",
    }  # fmt: skip


def test_dump_refuses_another_release(tmp_path: Path, fake: FakeGempack) -> None:
    fake.releases["sltoht.exe"] = "12.1.003"
    assert levels_job.main(_dump(tmp_path)) == 1


def test_dump_retries_a_program_that_did_not_start(
    tmp_path: Path, fake: FakeGempack
) -> None:
    fake.fail_start.add("sltoht.exe")
    assert levels_job.main(_dump(tmp_path)) == 0
    assert "did not start" in (tmp_path / "out" / gemsim_job.NOTE_FILE).read_text()


def _resolve(tmp_path: Path, *how: str) -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    return ["resolve", "--dataset", "gtap7_3x3", "--shock", "10", "--steps", "8-16-32",
            *(how or ("--rordelta", "1")), "--sl4dump", str(MATRIX / "sl4dump.har"),
            "--gtapv7", "C:/runGTAP375/gtapv7.exe", "--sltoht", "C:/GP/sltoht.exe",
            "--release", "gtapv7.exe=12.1.003",
            "--work", str(tmp_path / "work"), "--out", str(out),
            "--expect-release", "11.3.004"]  # fmt: skip


def test_resolve_keeps_the_sl4(tmp_path: Path, fake: FakeGempack) -> None:
    assert levels_job.main(_resolve(tmp_path)) == 0
    assert _names(tmp_path / "out") == {
        "run.cmf", "solution.sl4", "solve_console.txt", "sl4levels.har",
        "export_console.txt",
    }  # fmt: skip
    cmf = (tmp_path / "out" / "run.cmf").read_text(encoding="latin-1")
    assert "Steps  = 8 16 32" in cmf and "Shock tm = uniform 10" in cmf


def test_resolve_cross_checks_against_the_recorded_dump(
    tmp_path: Path, fake: FakeGempack
) -> None:
    """A re-solve whose %-changes differ from the recorded sl4dump is not that solution."""
    fake.mismatch = 2
    assert levels_job.main(_resolve(tmp_path)) == 1


def test_resolve_checks_gtapv7s_own_release(tmp_path: Path, fake: FakeGempack) -> None:
    fake.releases["gtapv7.exe"] = "11.3.004"
    assert levels_job.main(_resolve(tmp_path)) == 1


def test_resolve_from_a_swap_template(tmp_path: Path, fake: FakeGempack) -> None:
    template = TOOLS / "cmf" / "nus333_tm10_dpsaveswap.cmf"
    argv = _resolve(tmp_path, "--template", str(template))
    argv[argv.index("--dataset") + 1] = "nus333"
    assert levels_job.main(argv) == 0
    assert (
        'swap dpsave("USA") = del_tbalry("USA")'
        in (tmp_path / "out" / "run.cmf").read_text()
    )
