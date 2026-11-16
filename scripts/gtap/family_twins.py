"""Which Pyomo solve cases have a family (JAX) twin, read from the coverage matrix.

A Pyomo case leaves PR CI (``pyomo_solve`` marker) only when the family route runs the
same variant against the same reference and passes at least the same floors: the
non-xfail cases of ``test_family_gams_parity`` (mode ``pure``, the larger of the mcp
and nlp floors) and ``test_family_gempack_parity``. This module imports only the
matrix and lives next to it, on the sys.path of every Pyomo gate module (and of the
scripts that import those modules by bare name), so they can mark at collection time;
``test_coverage_matrix`` checks it against the family cases themselves.
"""

from __future__ import annotations

import coverage_matrix  # type: ignore
import pytest

FAMILY_MODES = ("pure",)  # the family route runs mode="gtap" only


def twins() -> set[tuple[str, str, int, str]]:
    """``(dataset, mode, ifsub, reference)`` the family parity tests cover."""
    out = set()
    for r in coverage_matrix.mcp_rows() + coverage_matrix.nlp_rows():
        if r.mode in FAMILY_MODES:
            out.add((r.dataset, r.mode, r.ifsub, r.reference))
    return out


def pyomo_solve(dataset: str, mode: str, ifsub: int, reference: str) -> list:
    """The marks of a Pyomo case: ``pyomo_solve`` when the family covers it."""
    if (dataset, mode, ifsub, reference) in twins():
        return [pytest.mark.pyomo_solve]
    return []


def pyomo_solve_ifsub_pair(dataset: str) -> list:
    """``test_ifsub_equivalence``'s marks: its twin is the family pair ifSUB 0 and 1
    against GAMS, each at its floor (>= 99), stronger than the 90% cross-mode floor."""
    t = twins()
    if all((dataset, "pure", k, "gams") in t for k in (0, 1)):
        return [pytest.mark.pyomo_solve]
    return []
