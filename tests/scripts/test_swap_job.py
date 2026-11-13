"""swap_job.py: RunGTAP's gtapv7.exe with the retired swap closure (Tasks 10 and 11).

``swap``   runs/{nus333,9x10}_compare/rungtap/tm10.cmf, copied to tools/cmf/, with
           only the Auxiliary files line and the shock changed.
``abort``  runs/gempack_20x41_validation/tm10.cmf: GEMPACK must abort with the known
           signature (LOGE of negative in E_u, the assert-arith dump); any other
           outcome records nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.sources import GEMPACK_MODEL_DIR

sys.path.insert(0, str(GEMPACK_MODEL_DIR / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import gemsim_job  # noqa: E402  # ty: ignore[unresolved-import]
import swap_job  # noqa: E402  # ty: ignore[unresolved-import]
from _fake_gempack import FakeGempack  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CMF = GEMPACK_MODEL_DIR / "tools" / "cmf"
TRACKED = {
    "nus333": ROOT / "runs" / "nus333_compare" / "rungtap" / "tm10.cmf",
    "9x10": ROOT / "runs" / "9x10_compare" / "rungtap" / "tm10.cmf",
    "gtap7_20x41": ROOT / "runs" / "gempack_20x41_validation" / "tm10.cmf",
}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeGempack:
    f = FakeGempack()
    monkeypatch.setattr(gemsim_job, "run", f)
    monkeypatch.setattr(gemsim_job, "RETRY_PAUSE_S", 0)
    return f


def _git_bytes(path: Path) -> bytes | None:
    """The committed bytes of ``path``: what every clone gets, whatever its autocrlf."""
    import subprocess

    rel = path.relative_to(ROOT).as_posix()
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
                          capture_output=True, check=False)  # fmt: skip
    return proc.stdout if proc.returncode == 0 else None


@pytest.mark.parametrize("ds", list(TRACKED))
def test_templates_are_the_tracked_cmf(ds: str) -> None:
    """tools/cmf/ keeps the runs/ .cmf byte for byte (runs/ goes away in 3b).

    Compared with the bytes git holds for runs/, not the file on disk: on a
    Windows clone autocrlf turns runs/ into CRLF, and a copy of that disk file
    under data/ (-text) kept CRLF while macOS has LF (the bug of #186)."""
    tracked = _git_bytes(TRACKED[ds])
    if tracked is None:
        pytest.skip("runs/ already removed (phase 3b)")
    assert (CMF / f"{ds}_tm10_dpsaveswap.cmf").read_bytes() == tracked


def test_only_aux_and_shock_change() -> None:
    template = (CMF / "nus333_tm10_dpsaveswap.cmf").read_text(encoding="latin-1")
    out = swap_job.from_template(template, shock=3, aux=Path(r"C:\x\gtapv7"))
    old, new = template.splitlines(), out.splitlines()
    changed = [(a, b) for a, b in zip(old, new, strict=True) if a != b]
    assert changed == [
        (
            r"Auxiliary files = C:\runGTAP375\gtapv7 ;",
            r"Auxiliary files = C:\x\gtapv7 ;",
        ),
        ("Shock tm = uniform 10 ;", "Shock tm = uniform 3 ;"),
    ]


def test_swap_lines_kept() -> None:
    nus = (CMF / "nus333_tm10_dpsaveswap.cmf").read_text(encoding="latin-1")
    assert 'swap dpsave("USA") = del_tbalry("USA") ;' in nus

    def swaps(ds: str) -> int:  # statements, not the comments that mention the recipe
        text = (CMF / f"{ds}_tm10_dpsaveswap.cmf").read_text(encoding="latin-1")
        return sum(ln.startswith("swap dpsave(") for ln in text.splitlines())

    assert swaps("nus333") == 1
    assert swaps("9x10") == 9
    assert swaps("gtap7_20x41") == 40


def _argv(
    tmp_path: Path, mode: str, ds: str = "nus333", shock: str = "10"
) -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    return [mode, "--dataset", ds, "--shock", shock,
            "--template", str(CMF / f"{ds}_tm10_dpsaveswap.cmf"),
            "--gtapv7", "C:/runGTAP375/gtapv7.exe",
            *(("--sltoht", "C:/GP/sltoht.exe", "--release", "sltoht.exe=11.3.004")
              if mode == "swap" else ()),
            "--work", str(tmp_path / "work"), "--out", str(out),
            "--expect-release", "12.1.003"]  # fmt: skip


def test_swap_collects_the_3a1_files(tmp_path: Path, fake: FakeGempack) -> None:
    assert swap_job.main(_argv(tmp_path, "swap", shock="3")) == 0
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "updated.har", "decomp.har", "sl4dump.har", "run.cmf", "solve_console.txt",
        "sltoht_console.txt",
    }  # fmt: skip
    assert "Shock tm = uniform 3 ;" in (tmp_path / "out" / "run.cmf").read_text()


def test_expected_abort_only_on_signature(tmp_path: Path, fake: FakeGempack) -> None:
    fake.abort = True
    assert swap_job.main(_argv(tmp_path, "abort", "gtap7_20x41")) == 0
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "run.cmf",
        "solve_console.txt",
        "assert_arith_fail.har",
    }


def test_a_converged_run_is_not_the_abort(tmp_path: Path, fake: FakeGempack) -> None:
    assert swap_job.main(_argv(tmp_path, "abort", "gtap7_20x41")) == 1


def test_run_cmf_is_crlf(tmp_path: Path, fake: FakeGempack) -> None:
    """The templates keep git's LF (runs/), but every .cmf gtapv7.exe has solved here
    was CRLF: the run's .cmf is written CRLF whatever the template's line ends."""
    assert b"\r\n" not in (CMF / "nus333_tm10_dpsaveswap.cmf").read_bytes()
    assert swap_job.main(_argv(tmp_path, "swap")) == 0
    raw = (tmp_path / "out" / "run.cmf").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
