"""equilibria doctor: which dependencies a solve needs, found or not, and how to fix them."""

from __future__ import annotations

import logging
import sys

import pytest
from click.testing import CliRunner

from equilibria.cli import main
from equilibria.doctor import MISSING, OK, WARN, check_dependencies


def _by_name(checks):
    return {c.name: c for c in checks}


@pytest.fixture
def no_path_env(monkeypatch):
    for var in ("PATH_CAPI_LIBPATH", "PATH_CAPI_LIBLUSOL", "PATH_LICENSE_STRING"):
        monkeypatch.delenv(var, raising=False)


def test_a_missing_path_library_is_an_error_with_a_fix(
    monkeypatch, tmp_path, no_path_env
):
    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(tmp_path / "libpath50.dylib"))

    c = _by_name(check_dependencies())["PATH library"]

    assert c.status == MISSING
    assert str(tmp_path / "libpath50.dylib") in c.detail
    assert "pages.cs.wisc.edu/~ferris/path" in c.fix


def test_a_file_that_is_not_a_library_is_an_error(monkeypatch, tmp_path, no_path_env):
    fake = tmp_path / "libpath50.dylib"
    fake.write_text("not a shared library")
    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(fake))

    c = _by_name(check_dependencies())["PATH library"]

    assert c.status == MISSING
    assert "cannot be loaded" in c.detail


def test_without_a_license_path_is_limited_to_300_variables(no_path_env):
    c = _by_name(check_dependencies())["PATH license"]

    assert c.status == WARN
    assert "300 variables" in c.detail
    assert "PATH_LICENSE_STRING" in c.fix


def test_a_license_in_the_environment_is_found(monkeypatch, no_path_env):
    monkeypatch.setenv("PATH_LICENSE_STRING", "1234&Someone&2035&GEN&5_1_2026&...")

    assert _by_name(check_dependencies())["PATH license"].status == OK


def test_a_missing_optional_package_is_a_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "mumps", None)  # import mumps -> ImportError

    c = _by_name(check_dependencies())["pymumps"]

    assert c.status == WARN
    assert "20x41" in c.detail
    assert "conda-forge" in c.fix


def test_a_missing_required_package_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "path_capi_python", None)

    c = _by_name(check_dependencies())["path-capi-python"]

    assert c.status == MISSING
    assert "equilibria[path]" in c.fix


def test_present_packages_report_their_version():
    c = _by_name(check_dependencies())["Pyomo"]

    assert c.status == OK
    assert c.detail[0].isdigit()


def test_cli_exits_nonzero_when_something_required_is_missing(
    monkeypatch, tmp_path, no_path_env
):
    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(tmp_path / "nope.dylib"))

    r = CliRunner().invoke(main, ["doctor"])

    assert r.exit_code == 1
    assert "PATH library" in r.output


def test_every_check_goes_to_the_log(caplog, monkeypatch, tmp_path, no_path_env):
    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(tmp_path / "nope.dylib"))
    with caplog.at_level(logging.INFO, logger="equilibria"):
        checks = check_dependencies()

    logged = [r for r in caplog.records if r.name == "equilibria.doctor.dependencies"]
    assert len(logged) == len(checks)
    lib = next(r for r in logged if "PATH library" in r.getMessage())
    assert lib.levelno == logging.WARNING  # MISSING and WARN are both warnings


def test_ipopt_executable_is_checked_apart_from_cyipopt(monkeypatch, tmp_path):
    """GTAP's NLP solves call the ipopt program through Pyomo; only PEP uses cyipopt."""
    monkeypatch.setenv("PATH", str(tmp_path))  # no ipopt on the search path

    checks = _by_name(check_dependencies())

    assert checks["IPOPT program"].status == WARN
    assert "GTAP" in checks["IPOPT program"].detail
    assert "PEP" in checks["cyipopt"].detail or "PEP" in checks["cyipopt"].needed_for
