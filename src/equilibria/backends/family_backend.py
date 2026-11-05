"""Family backend: evaluate an equilibria Model's equations as array templates.

Takes the same blocks as ``PyomoBackend``. Each equation's ``build_expression`` runs
against a ``RecordingModel`` (arrays, no Pyomo), and the recorded cells of a family are
grouped by term shape into a few JAX templates (``equilibria.family``). The result is a
``FamilySystem`` giving F(z) and a sparse J(z) with a fixed pattern.

Recording does not depend on which cells are fixed; ``system(layout)`` applies the
variable layout (free columns, fixed values) afterwards.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from equilibria.family.core import CompiledFamily, FamilySystem
from equilibria.family.recorder import RecordingModel, compile_family, intrinsic

_INTRINSICS = ("sqrt", "log", "exp")


@contextmanager
def _recording_intrinsics() -> Iterator[None]:
    """Swap pyomo's sqrt/log/exp for recording versions, in pyomo.environ (for imports
    inside functions) and in every loaded equilibria module that imported them."""
    import pyomo.environ as pe

    originals = {n: getattr(pe, n) for n in _INTRINSICS}
    patched: list[tuple[Any, str, Any]] = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "equilibria" or name.startswith("equilibria.")):
            continue
        for fn in _INTRINSICS:
            if getattr(mod, fn, None) is originals[fn]:
                patched.append((mod, fn, originals[fn]))
    for fn in _INTRINSICS:
        patched.append((pe, fn, originals[fn]))
    try:
        for mod, fn, _ in patched:
            setattr(mod, fn, intrinsic(fn))
        yield
    finally:
        for mod, fn, orig in patched:
            setattr(mod, fn, orig)


class FamilyBackend:
    """Record an equilibria Model's equations once; compile them per variable layout."""

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[tuple[str, ...], Any]]] = {}
        self.var_domains: dict[str, tuple[str, ...]] = {}
        # Mutable Parameters are leaves too (always fixed slots, values from the state).
        self.param_domains: dict[str, tuple[str, ...]] = {}
        self.sets: dict[str, list[str]] = {}

    def build(
        self, model: Any, only: Iterable[str] | None = None, skip: Iterable[str] = ()
    ) -> None:
        model.check_uses()
        rec = RecordingModel.from_model(model)
        self.var_domains = dict(rec.var_domains)
        self.param_domains = dict(rec.param_domains)
        self.sets = rec.sets
        wanted = None if only is None else set(only)
        skipped = set(skip)
        em = model.equation_manager
        with _recording_intrinsics():
            for name in em.list_equations():
                if (wanted is not None and name not in wanted) or name in skipped:
                    continue
                eq = em.get(name)
                cells = []
                for idx in eq.get_indices(model.set_manager):
                    res = eq.build_expression(rec, idx)
                    if res is not None:
                        cells.append((tuple(idx), res))
                if cells:
                    self.records[name] = cells

    def layout(self, state: Any) -> Any:
        """Variable layout from a ``StateModel``: free cells are columns; fixed cells and
        mutable Parameters are fixed slots carrying their current values."""
        import numpy as np

        from equilibria.family.layout import VarLayout

        values = {v: state.values[v] for v in self.var_domains}
        free = {v: ~state.fixed[v] for v in self.var_domains}
        for p in self.param_domains:
            values[p] = state.find_component(p).data
            free[p] = np.zeros(values[p].shape, dtype=bool)
        domains = {**self.var_domains, **self.param_domains}
        return VarLayout.from_state(state.sets, domains, values, free)

    def families(self, layout: Any) -> list[CompiledFamily]:
        return [compile_family(n, recs, layout) for n, recs in self.records.items()]

    def system(self, layout: Any) -> FamilySystem:
        return FamilySystem.from_families(self.families(layout), layout)
