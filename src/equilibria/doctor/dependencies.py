"""What a solve needs from the installation, and whether it is there.

A missing piece rarely fails loudly: without pymumps the 20x41 Newton route is
unavailable, without PyNumero's ASL the Jacobian falls back to a slower mode,
and without a PATH license PATH stops at 300 variables. The clean-room
replication of the paper failed for the first of these with no message. This
module names each piece, says whether it was found and how to get it.

MISSING means a solve cannot run (an MCP solve needs path-capi-python and the
PATH library); WARN means it runs with a limit or a slower path.

Each check looks where the solve looks, through the same helpers: the
libraries come from ``_local_refs.path_capi_libs`` (the solve's own lookup),
path-capi-python is imported with ``path_capi_src()`` on ``sys.path`` as the
solve does, and the license is checked by PATH itself (``Path_CheckLicense``),
not by whether the variable is set.

PATH reads the license once per process: after a first check with a bad
string, the good one is rejected too in that process (measured). ``equilibria
doctor`` is a fresh process; ``check_dependencies()`` called where PATH was
already loaded reports what PATH accepted then.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from equilibria.version import __version__

OK, WARN, MISSING = "ok", "warn", "missing"

_log = logging.getLogger(__name__)

_PATH_SITE = "https://pages.cs.wisc.edu/~ferris/path/julia/"

# A GTAP 3x3 has ~1,000 variables; PATH's unlicensed limit is 300.
_LICENSE_PROBE = (1000, 5000)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # OK, WARN or MISSING
    detail: str
    fix: str = ""
    needed_for: str = ""


def _import(module: str) -> tuple[ModuleType | None, str]:
    """Import ``module``; on any failure return the reason instead of raising.

    A broken package can fail with more than ImportError (a native library it
    loads, a RuntimeError at import): the solve catches ``Exception`` there,
    so the doctor does too.
    """
    try:
        return importlib.import_module(module), ""
    except ImportError:
        return None, "not installed"
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        return None, f"installed but cannot be imported ({type(e).__name__}: {e})"


def _package(
    name: str, module: str, *, required: bool, needed_for: str, fix: str
) -> Check:
    mod, why = _import(module)
    if mod is None:
        return Check(name, MISSING if required else WARN, why, fix, needed_for)
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "installed (version unknown)"
    return Check(name, OK, version, "", needed_for)


def _path_capi_python() -> tuple[Check, ModuleType | None]:
    """Imported as the solve imports it: with path_capi_src() on sys.path."""
    from equilibria._local_refs import path_capi_src

    needed = "MCP solves"
    src = path_capi_src()
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    mod, why = _import("path_capi_python")
    if mod is None:
        return Check(
            "path-capi-python",
            MISSING,
            why,
            'pip install "equilibria[path]" (or set EQUILIBRIA_PATH_CAPI_SRC to a checkout\'s src)',
            needed,
        ), None
    origin = str(getattr(mod, "__file__", "") or "")
    try:
        version = importlib.metadata.version("path-capi-python")
    except importlib.metadata.PackageNotFoundError:
        version = "version unknown"
    where = f", from {src}" if src.exists() and origin.startswith(str(src)) else ""
    return Check("path-capi-python", OK, f"{version}{where}", "", needed), mod


def _load(lib: Path) -> str:
    """'' if ctypes can load ``lib``, else why not."""
    try:
        ctypes.CDLL(str(lib))
    except OSError as e:
        return f"{lib} exists but cannot be loaded ({e})"
    return ""


def _path_libraries() -> tuple[list[Check], Path | None, Path | None]:
    """PATH and LUSOL, from the solve's own lookup; returns them if they load."""
    from equilibria._local_refs import path_capi_libs

    lib, lusol = path_capi_libs()
    get = f"download it from {_PATH_SITE}"
    arch = f"use the library for this platform ({platform.system()} {platform.machine()}) from {_PATH_SITE}"
    checks: list[Check] = []

    loaded_lib: Path | None = None
    if not lib.exists():
        checks.append(
            Check(
                "PATH library",
                MISSING,
                f"not found: {lib}",
                f"{get} and set PATH_CAPI_LIBPATH",
                "MCP solves",
            )
        )
    elif why := _load(lib):
        checks.append(Check("PATH library", MISSING, why, arch, "MCP solves"))
    else:
        checks.append(Check("PATH library", OK, str(lib), "", "MCP solves"))
        loaded_lib = lib

    needed = "PATH's basis factorization"
    loaded_lusol: Path | None = None
    if lusol is None:
        checks.append(
            Check(
                "LUSOL library",
                WARN,
                "not found (PATH runs without it)",
                f"{get} and set PATH_CAPI_LIBLUSOL",
                needed,
            )
        )
    elif not lusol.exists():
        # named by PATH_CAPI_LIBLUSOL: the solve passes it on and PATHLoader raises
        checks.append(
            Check(
                "LUSOL library",
                MISSING,
                f"PATH_CAPI_LIBLUSOL names a file that does not exist: {lusol}",
                f"{get}, or unset PATH_CAPI_LIBLUSOL",
                needed,
            )
        )
    elif why := _load(lusol):
        checks.append(Check("LUSOL library", MISSING, why, arch, needed))
    else:
        checks.append(Check("LUSOL library", OK, str(lusol), "", needed))
        loaded_lusol = lusol
    return checks, loaded_lib, loaded_lusol


def _path_license(
    pcp: ModuleType | None, lib: Path | None, lusol: Path | None
) -> Check:
    needed = "MCP solves above 300 variables"
    fix = (
        f"set PATH_LICENSE_STRING to the free evaluation license in {_PATH_SITE}LICENSE"
    )
    is_set = bool(os.environ.get("PATH_LICENSE_STRING"))
    if pcp is None or lib is None:
        # PATH cannot be asked; say only what can be seen
        if is_set:
            return Check(
                "PATH license",
                WARN,
                "set, not verified (PATH could not be loaded)",
                "",
                needed,
            )
        return Check(
            "PATH license", WARN, "not set: PATH stops at 300 variables", fix, needed
        )
    try:
        loader = pcp.PATHLoader(lib, lusol)
        accepted = loader.check_license(loader.load(), *_LICENSE_PROBE)
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        return Check(
            "PATH license",
            WARN,
            f"could not be checked ({type(e).__name__}: {e})",
            fix,
            needed,
        )
    if accepted:
        return Check(
            "PATH license",
            OK,
            f"PATH accepts it for {_LICENSE_PROBE[0]:,} variables",
            "",
            needed,
        )
    if is_set:
        return Check(
            "PATH license",
            WARN,
            "PATH_LICENSE_STRING is set but PATH rejects it (expired or malformed)",
            fix,
            needed,
        )
    return Check(
        "PATH license",
        WARN,
        "not set: PATH stops at 300 variables, smaller than any GTAP dataset",
        fix,
        needed,
    )


def _asl() -> Check:
    needed = "fast Jacobian evaluation"
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface

        found = bool(AmplInterface.available())
    except Exception:  # noqa: BLE001 - the solver treats any failure as "not available"
        found = False
    forced = os.environ.get("EQUILIBRIA_GTAP_JAC_MODE")
    note = (
        f"; not used: EQUILIBRIA_GTAP_JAC_MODE={forced} selects the mode"
        if forced
        else ""
    )
    if found:
        return Check("PyNumero ASL", OK, f"pynumero_ASL found{note}", "", needed)
    return Check(
        "PyNumero ASL",
        WARN,
        f"not found: the Jacobian is evaluated with reverse_numeric, which is slower{note}",
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
    pcp_check, pcp = _path_capi_python()
    lib_checks, lib, lusol = _path_libraries()
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
        pcp_check,
        *lib_checks,
        _path_license(pcp, lib, lusol),
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
        fix = f" (fix: {c.fix})" if c.fix else ""
        if c.status == OK:
            _log.info("%s: %s", c.name, c.detail)
        elif c.status == MISSING:
            _log.error("%s: MISSING %s%s", c.name, c.detail, fix)
        else:
            _log.warning("%s: WARN %s%s", c.name, c.detail, fix)
    return checks
