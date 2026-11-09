"""Levels of a family run in the shape ``compare_gempack_levels.model_levels`` returns."""

from __future__ import annotations

from collections.abc import Iterable

from equilibria.family.mp_arrays import MPArrays


def family_levels(
    mp: MPArrays, names: Iterable[str]
) -> dict[str, dict[tuple[str, ...], float]]:
    """``{var: {(indices..., period): level}}`` of ``names`` from ``mp`` (cell keys end
    with the period, as the Pyomo ``model_levels`` keys do)."""
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for n in names:
        v = mp.vars[n]
        out[n] = {tuple(k): float(x) for k, x in zip(v.keys, v.value, strict=True)}
    return out
