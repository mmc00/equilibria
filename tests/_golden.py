"""Compare a recomputed record with a golden recorded on one machine."""

from __future__ import annotations

REL_TOL = 1e-12


def mismatches(
    got, want, path: tuple = (), *, rel: float = REL_TOL, abs_tol: float = 0.0
) -> list[str]:
    """Where a recomputed record leaves the golden: a key on one side only, or a value
    off by more than ``abs_tol + rel * max(|got|, |want|)``. The golden was recorded
    on one machine; another CPU or BLAS moves the last bits (~1e-15 on a settle; on a
    full solve ~1e-11 relative, and a residual near zero by ~1e-11 absolute, hence
    ``abs_tol``); a real change moves far more."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f"{(*path, k)}: only in golden" for k in want.keys() - got.keys()]
        out += [f"{(*path, k)}: not in golden" for k in got.keys() - want.keys()]
        for k in want.keys() & got.keys():
            out += mismatches(got[k], want[k], (*path, k), rel=rel, abs_tol=abs_tol)
        return out
    g, w = float(got), float(want)
    if abs(g - w) > abs_tol + rel * max(abs(g), abs(w)):
        return [f"{path}: {g!r} != {w!r}"]
    return []


def model_levels(m) -> dict[str, dict[str, str]]:
    """Every Pyomo Var level of ``m``, ``{var: {repr(index): repr(float)}}``; a cell
    with no value is left out (then a side that has one shows as a key mismatch)."""
    from pyomo.environ import Var

    out: dict[str, dict[str, str]] = {}
    for v in m.component_objects(Var, active=True):
        cells = {repr(k): repr(float(v[k].value)) for k in v if v[k].value is not None}
        if cells:
            out[v.name] = cells
    return out
