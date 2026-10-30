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
SCRIPT = ROOT / "scripts" / "gtap" / "run_gempack_matrix.py"


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
        "print('Program completed without error.')\n"
        "print('Maximum residual ratio is 1e-9')",
    )


def _fake_sltoht(tmp: Path) -> Path:
    return _exe(
        tmp / "sltoht",
        "name = pathlib.Path(sys.argv[2]).stem\n"
        "pathlib.Path(f'{name}.har').write_bytes(b'dump')",
    )


def _args(tmp: Path, out: Path, gtapv7: Path) -> list[str]:
    return [
        "--datasets", "gtap7_3x3", "--shock-pct", "10", "--steps", "8 16 32",
        "--rordelta", "1", "--out", str(out),
        "--gtapv7", str(gtapv7), "--sltoht", str(_fake_sltoht(tmp)),
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
