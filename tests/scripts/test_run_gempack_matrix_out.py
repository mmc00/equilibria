"""run_gempack_matrix --out: one GEMPACK job solved into one folder, fixed names.

GEMPACK runs only on Windows; fake gtapv7/sltoht executables stand in for them
and write the files the real ones write, so the collection logic runs here.
"""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "data" / "gtap7" / "gempack" / "tools" / "run_gempack_matrix.py"


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--out works under runs/gempack_matrix by default; tests stay out of the repo."""
    monkeypatch.setenv("EQUILIBRIA_GEMPACK_WORK", str(tmp_path / "work"))


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("run_gempack_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exe(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\nimport sys, pathlib\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _fake_gtapv7(tmp: Path, ok: bool = True) -> Path:
    if not ok:
        return _exe(tmp / "gtapv7", "print('Fatal error in the solve')")
    return _exe(
        tmp / "gtapv7",
        "tag = pathlib.Path(sys.argv[2]).stem\n"
        "pathlib.Path(f'updated_{tag}.har').write_bytes(b'upd')\n"
        "pathlib.Path('decomp.har').write_bytes(b'dec')\n"
        "pathlib.Path(f'{tag}.sl4').write_bytes(b'sl4')\n"
        "print('<GEMPACK Release 12.1.003 January 2020>')\n"
        "print(f'Current directory is: {pathlib.Path.cwd()}')\n"
        "print('Program completed without error.')\n"
        "print('Maximum residual ratio is 1e-9')",
    )


def _fake_sltoht(tmp: Path, ok: bool = True) -> Path:
    if not ok:
        return _exe(tmp / "sltoht", "print('sltoht: cannot open file')")
    return _exe(
        tmp / "sltoht",
        "name = pathlib.Path(sys.argv[2]).stem\n"
        "print(f'Current directory is: {pathlib.Path.cwd()}')\n"
        "pathlib.Path(f'{name}.har').write_bytes(b'dump')",
    )


def _args(tmp: Path, out: Path, gtapv7: Path, sltoht: Path | None = None) -> list[str]:
    return [
        "--datasets", "gtap7_3x3", "--shock-pct", "10", "--steps", "8 16 32",
        "--rordelta", "1", "--out", str(out),
        "--gtapv7", str(gtapv7), "--sltoht", str(sltoht or _fake_sltoht(tmp)),
    ]  # fmt: skip


def test_out_collects_fixed_names(tmp_path: Path) -> None:
    out = tmp_path / "gempack_tm10_s8-16-32_capflex"
    assert _module().main(_args(tmp_path, out, _fake_gtapv7(tmp_path))) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "decomp.har", "run.cmf", "sl4dump.har", "sltoht_console.txt",
        "solve_console.txt", "updated.har",
    ]  # fmt: skip
    assert (out / "updated.har").read_bytes() == b"upd"
    assert "8 16 32" in (out / "run.cmf").read_text()


def test_failed_solve_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "o"
    assert _module().main(_args(tmp_path, out, _fake_gtapv7(tmp_path, ok=False))) == 1
    assert not (out / "updated.har").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--datasets", "gtap7_3x3", "gtap7_3x4", "--rordelta", "1"],
        ["--grid", "--datasets", "gtap7_3x3", "--rordelta", "1"],
        ["--datasets", "gtap7_3x3"],  # no --rordelta: both closures
        ["--datasets", "gtap7_3x3", "--rordelta", "0", "--rordelta", "1"],
    ],
)
def test_out_needs_exactly_one_job(tmp_path: Path, extra: list[str]) -> None:
    assert _module().main([*extra, "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def test_failed_solve_shows_the_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The work folder is temporary: a failure must print GEMPACK's own words."""
    _module().main(_args(tmp_path, tmp_path / "o", _fake_gtapv7(tmp_path, ok=False)))
    assert "Fatal error in the solve" in capsys.readouterr().err


def test_missing_sltoht_is_refused(tmp_path: Path) -> None:
    """Without sltoht the folder would lack sl4dump.har yet be recorded verified."""
    args = _args(
        tmp_path, tmp_path / "o", _fake_gtapv7(tmp_path), tmp_path / "nope.exe"
    )
    assert _module().main(args) == 2
    assert not (tmp_path / "o").exists()


def test_failed_sltoht_exits_nonzero(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    bad = _fake_sltoht(tmp_path / "b", ok=False)
    out = tmp_path / "o"
    assert _module().main(_args(tmp_path, out, _fake_gtapv7(tmp_path), bad)) == 1


def test_consoles_carry_no_work_path(tmp_path: Path) -> None:
    """GEMPACK prints its cwd; under %TEMP% that names the Windows user."""
    out = tmp_path / "o"
    assert _module().main(_args(tmp_path, out, _fake_gtapv7(tmp_path))) == 0
    for name in ("solve_console.txt", "sltoht_console.txt"):
        text = (out / name).read_text()
        assert "Current directory is: <work>" in text, text
        assert (
            str(tmp_path) not in text
            and "/var/folders" not in text
            and "/tmp" not in text
        )


def test_cmf_uses_the_given_gtapv7(tmp_path: Path) -> None:
    gtapv7 = _fake_gtapv7(tmp_path)
    out = tmp_path / "o"
    assert _module().main(_args(tmp_path, out, gtapv7)) == 0
    assert (
        f"Auxiliary files = {gtapv7.with_suffix('')} ;" in (out / "run.cmf").read_text()
    )


def test_release_mismatch_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(tmp_path, tmp_path / "o", _fake_gtapv7(tmp_path))
    assert _module().main([*args, "--expect-release", "11.3"]) == 1
    err = capsys.readouterr().err
    assert "12.1.003" in err and "11.3" in err


def test_release_match_passes(tmp_path: Path) -> None:
    args = _args(tmp_path, tmp_path / "o", _fake_gtapv7(tmp_path))
    assert _module().main([*args, "--expect-release", "12.1.003"]) == 0


def test_cleanup_error_keeps_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file still locked when the work folder is deleted (antivirus, a lingering
    GEMPACK process on Windows) must not lose a run whose results were copied."""
    import os

    m = _module()
    real = os.unlink

    def locked(path: object, *a: object, **k: object) -> None:
        if Path(str(path)).name.startswith("updated_"):
            raise PermissionError(13, "locked by another process", str(path))
        real(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "unlink", locked)
    out = tmp_path / "o"
    assert m.main(_args(tmp_path, out, _fake_gtapv7(tmp_path))) == 0
    assert (out / "updated.har").read_bytes() == b"upd"
