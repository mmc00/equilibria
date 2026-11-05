"""The GEMPACK release a console says it ran, checked against the one recorded.

RunGTAP's gtapv7.exe/gtapuv7.exe print 12.1.003; tablo, gemsim and sltoht of the
GEMPACK install print 11.3.004. A record's ``tool_version`` is the release of the
program that made its primary output, so a tool reads it from that program's
console and refuses a run whose release differs from the driver's argument.
"""

from __future__ import annotations

from pathlib import Path

from run_gempack_matrix import gempack_release as release

__all__ = ["ReleaseMismatch", "release", "require"]


class ReleaseMismatch(RuntimeError):
    """The console names another GEMPACK release, or none."""


def require(console: Path, expected: str) -> str:
    """The release in ``console``; raises ReleaseMismatch unless it is ``expected``."""
    found = release(console) if console.exists() else None
    if found is None:
        raise ReleaseMismatch(f"{console}: no GEMPACK Release banner")
    if found != expected:
        raise ReleaseMismatch(
            f"{console}: GEMPACK Release {found}, but the record says {expected}"
        )
    return found
