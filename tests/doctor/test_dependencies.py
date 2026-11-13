"""equilibria doctor: which dependencies a solve needs, found or not, and how to fix them."""

from __future__ import annotations

import logging
import sys

import pytest
from click.testing import CliRunner

from equilibria.cli import main
from equilibria.doctor import MISSING, OK, WARN, check_dependencies

# the developer's license, read before any test clears the environment
_LICENSE_AT_IMPORT = __import__("os").environ.get("PATH_LICENSE_STRING", "")


def _by_name(checks):
    return {c.name: c for c in checks}


@pytest.fixture
def no_path_env(monkeypatch, tmp_path):
    """No license, and no PATH library: PATH_CAPI_LIBPATH names a file that is not there.

    Unsetting the variable is not enough: the lookup falls back to the main
    checkout's cache (``_local_refs.path_capi_lib_dir``), which is the
    developer's machine.
    """
    for var in ("PATH_CAPI_LIBLUSOL", "PATH_LICENSE_STRING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(
        "PATH_CAPI_LIBPATH", str(tmp_path / "no_libs" / "libpath50.dylib")
    )
    monkeypatch.chdir(tmp_path)


def _real_path_libs():
    """The developer's PATH libraries, or skip: the license check needs the real library."""
    from equilibria._local_refs import path_capi_lib, path_capi_lusol

    lib, lusol = path_capi_lib(), path_capi_lusol()
    lusol = lusol if lusol.exists() else None
    if not lib.exists():
        pytest.skip("no PATH library on this machine")
    return lib, lusol


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


def test_a_license_that_cannot_be_verified_is_not_ok(monkeypatch, no_path_env):
    """Without a loadable PATH library the doctor cannot ask PATH, so it does not say OK."""
    monkeypatch.setenv("PATH_LICENSE_STRING", "1234&Someone&2035&GEN&5_1_2026&...")

    c = _by_name(check_dependencies())["PATH license"]

    assert c.status == WARN
    assert "not verified" in c.detail


def test_a_missing_optional_package_is_a_warning(monkeypatch, no_path_env):
    monkeypatch.setitem(sys.modules, "mumps", None)  # import mumps -> ImportError

    c = _by_name(check_dependencies())["pymumps"]

    assert c.status == WARN
    assert "20x41" in c.detail
    assert "conda-forge" in c.fix


def test_a_missing_required_package_is_an_error(monkeypatch, no_path_env):
    monkeypatch.setitem(sys.modules, "path_capi_python", None)

    c = _by_name(check_dependencies())["path-capi-python"]

    assert c.status == MISSING
    assert "equilibria[path]" in c.fix


def test_present_packages_report_their_version(no_path_env):
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
    lic = next(r for r in logged if "PATH license" in r.getMessage())
    assert lib.levelno == logging.ERROR  # MISSING: a solve cannot run
    assert lic.levelno == logging.WARNING  # WARN: it runs with a limit


def test_ipopt_executable_is_checked_apart_from_cyipopt(monkeypatch, tmp_path):
    """GTAP's NLP solves call the ipopt program through Pyomo; only PEP uses cyipopt."""
    monkeypatch.setenv("PATH", str(tmp_path))  # no ipopt on the search path

    checks = _by_name(check_dependencies())

    assert checks["IPOPT program"].status == WARN
    assert "GTAP" in checks["IPOPT program"].detail
    assert checks["cyipopt"].needed_for.startswith("PEP")


def _license_status_in_a_fresh_process(license_string: str) -> str:
    """PATH reads the license once per process, so each probe gets its own."""
    import os
    import subprocess

    lib, lusol = _real_path_libs()
    env = {
        **os.environ,
        "PATH_CAPI_LIBPATH": str(lib),
        "PATH_LICENSE_STRING": license_string,
    }
    if lusol is not None:
        env["PATH_CAPI_LIBLUSOL"] = str(lusol)
    code = (
        "from equilibria.doctor import check_dependencies; "
        "c = next(c for c in check_dependencies() if c.name == 'PATH license'); "
        "print(c.status + '|' + c.detail)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_a_license_path_rejects_is_not_ok(no_path_env):
    """Being set is not enough: PATH itself is asked whether it accepts the string."""
    status, detail = _license_status_in_a_fresh_process("garbage").split("|", 1)

    assert status == WARN
    assert "rejects" in detail


def test_the_real_license_is_accepted(no_path_env):
    if not _LICENSE_AT_IMPORT:
        pytest.skip("no PATH license on this machine")

    status, _ = _license_status_in_a_fresh_process(_LICENSE_AT_IMPORT).split("|", 1)

    assert status == OK


def test_a_lusol_named_by_env_but_absent_is_an_error(
    monkeypatch, tmp_path, no_path_env
):
    """The solve passes PATH_CAPI_LIBLUSOL through and PATHLoader raises on a missing file."""
    monkeypatch.setenv("PATH_CAPI_LIBLUSOL", str(tmp_path / "liblusol.dylib"))

    c = _by_name(check_dependencies())["LUSOL library"]

    assert c.status == MISSING


def test_a_lusol_that_does_not_load_is_an_error(monkeypatch, tmp_path, no_path_env):
    fake = tmp_path / "liblusol.dylib"
    fake.write_text("not a shared library")
    monkeypatch.setenv("PATH_CAPI_LIBLUSOL", str(fake))

    c = _by_name(check_dependencies())["LUSOL library"]

    assert c.status == MISSING
    assert "cannot be loaded" in c.detail


def test_path_capi_python_from_a_checkout_is_found(monkeypatch, tmp_path, no_path_env):
    """The solve also imports path-capi-python from EQUILIBRIA_PATH_CAPI_SRC."""
    src = tmp_path / "checkout" / "src"
    (src / "path_capi_python").mkdir(parents=True)
    (src / "path_capi_python" / "__init__.py").write_text("")
    monkeypatch.setenv("EQUILIBRIA_PATH_CAPI_SRC", str(src))
    # setitem first: the undo then removes the fake checkout's module this test
    # imports (a bare delitem records nothing when the module was not loaded)
    monkeypatch.setitem(sys.modules, "path_capi_python", None)
    monkeypatch.delitem(sys.modules, "path_capi_python")
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if "path-capi-python" not in p]
    )
    monkeypatch.setattr(
        sys, "meta_path", [f for f in sys.meta_path if "path_capi" not in repr(f)]
    )

    c = _by_name(check_dependencies())["path-capi-python"]

    assert c.status == OK
    assert str(src) in c.detail


def test_a_package_that_breaks_on_import_is_reported_not_raised(
    monkeypatch, tmp_path, no_path_env
):
    (tmp_path / "pkgs").mkdir()
    (tmp_path / "pkgs" / "mumps.py").write_text(
        "raise RuntimeError('native library missing')"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    monkeypatch.delitem(sys.modules, "mumps", raising=False)

    c = _by_name(check_dependencies())["pymumps"]

    assert c.status == WARN
    assert "native library missing" in c.detail


def test_asl_is_reported_unused_when_the_jacobian_mode_is_forced(
    monkeypatch, no_path_env
):
    monkeypatch.setenv("EQUILIBRIA_GTAP_JAC_MODE", "reverse_numeric")

    c = _by_name(check_dependencies())["PyNumero ASL"]

    assert "EQUILIBRIA_GTAP_JAC_MODE" in c.detail
