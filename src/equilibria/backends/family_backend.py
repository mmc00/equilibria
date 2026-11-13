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
from equilibria.family.recorder import (
    RecordingModel,
    TermTable,
    compile_family,
    intrinsic,
)

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

    _SAVED = ("records", "var_domains", "param_domains", "sets")

    def save(self, path: Any) -> None:
        """Write the recording (not the compiled code) to ``path`` with pickle."""
        import os
        import pickle
        import tempfile
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A temp name of our own: two writers sharing one would write into one inode.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump({k: getattr(self, k) for k in self._SAVED}, fh, protocol=5)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)  # never leave a half-written cache behind
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Any) -> FamilyBackend:
        import pickle

        from equilibria.solver.gc_tuning import _gc_freeze_in_scope, _gc_paused

        with open(path, "rb") as fh, _gc_paused():
            data = pickle.load(fh)
            _gc_freeze_in_scope()  # inside a family run: frozen before the GC is back
        backend = cls()
        for k in cls._SAVED:
            setattr(backend, k, data[k])
        return backend

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
        tables = self.__dict__.setdefault("_term_tables", {})
        out = []
        for n, recs in self.records.items():
            hit = tables.get(n)
            if hit is None or hit[0] is not recs:
                hit = tables[n] = (recs, TermTable(recs))
            out.append(compile_family(n, recs, layout, hit[1]))
        return out

    def system(self, layout: Any) -> FamilySystem:
        return FamilySystem.from_families(self.families(layout), layout)
