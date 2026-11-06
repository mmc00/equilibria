"""What a solve needs from the installation, and whether it is there.

A missing piece rarely fails loudly: without pymumps the 20x41 Newton route is
unavailable, without PyNumero's ASL the Jacobian falls back to a slower mode,
and without a PATH license PATH stops at 300 variables. The clean-room
replication of the paper failed for the first of these with no message. This
module names each piece, says whether it was found and how to get it.

MISSING means a solve cannot run (an MCP solve needs path-capi-python and the
PATH library); WARN means it runs with a limit or a slower path.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from equilibria.version import __version__

OK, WARN, MISSING = "ok", "warn", "missing"

_log = logging.getLogger(__name__)

_PATH_SITE = "https://pages.cs.wisc.edu/~ferris/path/julia/"


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # OK, WARN or MISSING
    detail: str
    fix: str = ""
    needed_for: str = ""


def _package(
    name: str, module: str, *, required: bool, needed_for: str, fix: str
) -> Check:
    try:
        importlib.import_module(module)
    except ImportError:
        return Check(
            name, MISSING if required else WARN, "not installed", fix, needed_for
        )
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "installed (version unknown)"
    return Check(name, OK, version, "", needed_for)


def _path_library() -> list[Check]:
    from equilibria._local_refs import path_capi_lib, path_capi_lusol

    env = os.environ.get("PATH_CAPI_LIBPATH")
    lib = Path(env).expanduser() if env else path_capi_lib()
    fix = (
        f"download it from {_PATH_SITE} and set PATH_CAPI_LIBPATH to its absolute path"
    )
    if not lib.exists():
        checks = [
            Check("PATH library", MISSING, f"not found: {lib}", fix, "MCP solves")
        ]
    else:
        try:
            ctypes.CDLL(str(lib))
            checks = [Check("PATH library", OK, str(lib), "", "MCP solves")]
        except OSError as e:
            checks = [
                Check(
                    "PATH library",
                    MISSING,
                    f"{lib} exists but cannot be loaded ({e})",
                    f"use the library for this platform ({platform.system()} {platform.machine()}) from {_PATH_SITE}",
                    "MCP solves",
                )
            ]

    env_lusol = os.environ.get("PATH_CAPI_LIBLUSOL")
    lusol = Path(env_lusol).expanduser() if env_lusol else path_capi_lusol()
    if lusol.exists():
        checks.append(
            Check("LUSOL library", OK, str(lusol), "", "PATH's basis factorization")
        )
    else:
        checks.append(
            Check(
                "LUSOL library",
                WARN,
                f"not found: {lusol} (PATH runs without it)",
                f"download it from {_PATH_SITE} and set PATH_CAPI_LIBLUSOL",
                "PATH's basis factorization",
            )
        )
    return checks


def _path_license() -> Check:
    if os.environ.get("PATH_LICENSE_STRING"):
        return Check(
            "PATH license",
            OK,
            "PATH_LICENSE_STRING is set",
            "",
            "MCP solves above 300 variables",
        )
    return Check(
        "PATH license",
        WARN,
        "not set: PATH stops at 300 variables, smaller than any GTAP dataset",
        f"set PATH_LICENSE_STRING to the free evaluation license in {_PATH_SITE}LICENSE",
        "MCP solves above 300 variables",
    )


def _asl() -> Check:
    needed = "fast Jacobian evaluation"
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface

        found = bool(AmplInterface.available())
    except ImportError:
        found = False
    if found:
        return Check("PyNumero ASL", OK, "pynumero_ASL found", "", needed)
    return Check(
        "PyNumero ASL",
        WARN,
        "not found: the Jacobian is evaluated with reverse_numeric, which is slower",
        "pyomo download-extensions && pyomo build-extensions (no prebuilt library for macOS arm64)",
        needed,
    )


def _ipopt_program() -> Check:
    import shutil

    needed = "GTAP NLP solves (Pyomo calls the ipopt program)"
    exe = shutil.which("ipopt")
    if exe:
        return Check("IPOPT program", OK, exe, "", needed)
    return Check(
        "IPOPT program",
        WARN,
        "ipopt not found on PATH",
        "install Ipopt (brew install ipopt; conda install -c conda-forge ipopt) so `ipopt` is on PATH",
        needed,
    )


def check_dependencies() -> list[Check]:
    """Every check, in order; each one is also written to the ``equilibria`` logger."""
    checks = [
        Check("Python", OK, platform.python_version()),
        Check("equilibria", OK, __version__),
        _package(
            "Pyomo",
            "pyomo",
            required=True,
            needed_for="building models",
            fix="pip install pyomo",
        ),
        _package(
            "path-capi-python",
            "path_capi_python",
            required=True,
            needed_for="MCP solves",
            fix='pip install "equilibria[path]"',
        ),
        *_path_library(),
        _path_license(),
        _asl(),
        _package(
            "pymumps",
            "mumps",
            required=False,
            needed_for="the Newton route of large datasets (20x41)",
            fix="conda install -c conda-forge pymumps",
        ),
        _ipopt_program(),
        _package(
            "cyipopt",
            "cyipopt",
            required=False,
            needed_for="PEP's IPOPT solver (GTAP uses the ipopt program instead)",
            fix='pip install "equilibria[ipopt]" (needs Ipopt on the system)',
        ),
    ]
    # a check that is not OK says what it is needed for
    checks = [
        Check(
            c.name,
            c.status,
            f"{c.detail}: needed for {c.needed_for}",
            c.fix,
            c.needed_for,
        )
        if c.status != OK and c.needed_for and c.needed_for not in c.detail
        else c
        for c in checks
    ]
    for c in checks:
        if c.status == OK:
            _log.info("%s: %s", c.name, c.detail)
        else:
            _log.warning(
                "%s: %s %s%s",
                c.name,
                c.status.upper(),
                c.detail,
                f" (fix: {c.fix})" if c.fix else "",
            )
    return checks
