"""Diagnostics for the user's installation and, later, their model.

``check_dependencies()`` lists what a solve needs (PATH and its license, the
Python packages, the optional accelerators), found or not, with how to fix
it. ``equilibria doctor`` prints the same list.
"""

from equilibria.doctor.dependencies import MISSING, OK, WARN, Check, check_dependencies

__all__ = ["MISSING", "OK", "WARN", "Check", "check_dependencies"]
