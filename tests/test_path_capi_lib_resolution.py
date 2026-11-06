"""The GTAP PATH solve finds the PATH libraries by platform, not by a macOS-only name.

It used to default to ``libpath50.silicon.dylib`` / ``liblusol.silicon.dylib``: on
Linux or Windows the solve failed even with ``libpath50.so`` sitting in the cache
folder, while the PEP solver and ``run_gtap.py`` already resolved by platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from equilibria.solver.path_capi import _path_capi_libs


@pytest.fixture
def lib_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("PATH_CAPI_LIBPATH", raising=False)
    monkeypatch.delenv("PATH_CAPI_LIBLUSOL", raising=False)
    monkeypatch.setenv("EQUILIBRIA_PATH_CAPI_LIB_DIR", str(tmp_path))
    return tmp_path


def test_linux_names_are_found(monkeypatch: pytest.MonkeyPatch, lib_dir: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    (lib_dir / "libpath50.so").touch()
    (lib_dir / "liblusol.so").touch()

    assert _path_capi_libs() == (lib_dir / "libpath50.so", lib_dir / "liblusol.so")


def test_windows_names_are_found(
    monkeypatch: pytest.MonkeyPatch, lib_dir: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    (lib_dir / "path50.dll").touch()
    (lib_dir / "lusol.dll").touch()

    assert _path_capi_libs() == (lib_dir / "path50.dll", lib_dir / "lusol.dll")


def test_missing_lusol_is_none(monkeypatch: pytest.MonkeyPatch, lib_dir: Path) -> None:
    """LUSOL is optional for PATH; a missing file must not abort the load."""
    monkeypatch.setattr(sys, "platform", "linux")
    (lib_dir / "libpath50.so").touch()

    assert _path_capi_libs() == (lib_dir / "libpath50.so", None)


def test_environment_variables_win(
    monkeypatch: pytest.MonkeyPatch, lib_dir: Path, tmp_path: Path
) -> None:
    lib = tmp_path / "custom" / "mypath.so"
    lusol = tmp_path / "custom" / "mylusol.so"
    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(lib))
    monkeypatch.setenv("PATH_CAPI_LIBLUSOL", str(lusol))

    assert _path_capi_libs() == (lib, lusol)
