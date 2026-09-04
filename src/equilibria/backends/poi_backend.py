"""PyOptInterface backend — a sibling of :class:`PyomoBackend`.

Both backends consume the same ``EquilibriaModel``: the blocks register their sets,
parameters, variables and equations into the managers, and each backend walks those
managers to build its own representation. Sharing the input is what makes a
comparison between them meaningful — any difference in build time or Jacobian
density belongs to the backend, not to two models that were assembled differently.

The block bodies are not modified. They receive a :class:`PoiModelAdapter` in place
of the Pyomo ``ConcreteModel`` and produce POI expressions from the same source.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from equilibria.backends.poi_adapter import PoiModelAdapter

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _llvm_opt_level(level: int | None):
    """Build POI's LLVM JIT at ``level`` instead of the hardcoded maximum.

    POI constructs its target machine with ``opt=3`` inside ``LLJITCompiler``,
    exposing no way to choose. The class is swapped for the duration of the model's
    construction — the only moment the compiler is created — and restored
    afterwards, so nothing outside this call sees a patched POI.

    ``level=None`` leaves POI's own default alone.
    """
    if level is None:
        yield
        return

    from llvmlite import binding
    from pyoptinterface._src import jit_llvm

    original = jit_llvm.LLJITCompiler.__init__

    def _init(self) -> None:
        target = binding.Target.from_default_triple()
        machine = target.create_target_machine(jit=True, opt=level)
        self.lljit = binding.create_lljit_compiler(machine)
        self.rts = []
        self.source_codes = []

    jit_llvm.LLJITCompiler.__init__ = _init
    try:
        yield
    finally:
        jit_llvm.LLJITCompiler.__init__ = original


class PoiBackend:
    """Builds a POI model from an ``EquilibriaModel``.

    Attributes:
        poi_model: the underlying PyOptInterface model
        adapter: the Pyomo-surface adapter the block bodies were given
        constraints: ``{"eq_name[idx]": poi_constraint}``
        skipped: ``{eq_name: count}`` for index combinations that contributed no
            constraint — the same drops Pyomo makes, tracked so a parity gap in
            Task 3 can be attributed rather than guessed at
    """

    def __init__(self, jit: str = "LLVM", opt_level: int | None = 0) -> None:
        self.poi_model: Any = None
        self.adapter: PoiModelAdapter | None = None
        self.constraints: dict[str, Any] = {}
        self.skipped: dict[str, int] = {}
        self._model: Any = None
        # POI ships two JIT engines. LLVM is the default; TCC compiles faster but
        # dies on macOS ARM64 — a TinyCC bug, not a POI one: its ARM64 Mach-O
        # backend emitted thread-local-storage relocations that Mach-O's linker
        # never implemented, and POI's expression graph context is thread-local.
        # Fixes reached the TinyCC mailing list in Aug 2026 but have not shipped in
        # tccbox, so TCC stays Linux-only for now.
        self._jit = jit
        # LLVM optimization level for the JIT. POI hardcodes 3 (maximum), which is
        # what makes compilation dominate the build: measured on the 3x3, opt=3
        # takes 4.25s against 0.49s at opt=0 (8.7x) for identical rows, and the
        # 10x7 goes from not finishing in 10 minutes to 235s.
        #
        # Default 0 because these evaluators run inside a Newton solve where the
        # wall clock is dominated by factorization, not by evaluation. Raise it if
        # a measurement ever shows evaluation to be the bottleneck.
        self._opt_level = opt_level

    def build(self, model: Any) -> None:
        """Build the POI model, mirroring ``PyomoBackend.build``'s phases."""
        from pyoptinterface import ipopt

        self._model = model
        with _llvm_opt_level(self._opt_level):
            self.poi_model = ipopt.Model(jit=self._jit)

        sets = {
            name: list(model.set_manager.get(name).elements)
            for name in model.set_manager.list_sets()
        }
        var_specs = {
            name: tuple(model.variable_manager.get(name).domains)
            for name in model.variable_manager.list_vars()
        }

        self.adapter = PoiModelAdapter(
            self.poi_model,
            sets=sets,
            params=_ParameterView(model.parameter_manager, model.set_manager),
            var_specs=var_specs,
        )

        self._build_constraints(model)

    def _build_constraints(self, model: Any) -> None:
        """Walk every equation's index space, as the Pyomo backend does.

        Domain expansion comes from ``eq.get_indices`` rather than a private
        reimplementation, so both backends enumerate identical index tuples.
        """
        from pyomo.environ import Constraint
        from pyoptinterface import nl

        for eq_name in model.equation_manager.list_equations():
            eq = model.equation_manager.get(eq_name)
            indices_list = eq.get_indices(model.set_manager)

            if not indices_list:
                # An equation over an empty set contributes nothing. Pyomo warns
                # and moves on; matching that keeps the two row sets aligned.
                logger.warning(
                    "Equation %s resolved to zero index combinations", eq_name
                )
                continue

            for indices in indices_list:
                # One graph per constraint rather than one for the whole model.
                # POI compiles each graph into an autodiff evaluator, then
                # deduplicates identical ones, so per-constraint graphs give it
                # many small functions to share instead of one enormous vector
                # function. Measured on the 3x3: 55.9s -> 3.67s to compile, same
                # 1,110 rows, with 409 nonlinear rows collapsing to 32 compiled
                # groups. This also confirms the earlier nl.graph()-scope finding
                # in devtools, which reached the same conclusion from RAM.
                with nl.graph():
                    # Reset before each row so the handles recorded during
                    # build_expression are exactly this row's variables.
                    self.adapter._touched.clear()
                    expr = eq.build_expression(self.adapter, indices)

                    # Pyomo drops a cell for None and for Constraint.Skip. POI has
                    # to drop exactly the same ones or the name parity in
                    # test_poi_blocks_parity is meaningless.
                    if expr is None or expr is Constraint.Skip:
                        self.skipped[eq_name] = self.skipped.get(eq_name, 0) + 1
                        continue

                    key = (
                        f"{eq_name}[{','.join(map(str, indices))}]"
                        if indices
                        else eq_name
                    )
                    self.constraints[key] = self.adapter.add_constraint(key, expr)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"PoiBackend(constraints={len(self.constraints)}, "
            f"skipped={sum(self.skipped.values())})"
        )


class _LabelIndexedParam:
    """A parameter addressed by set labels, as the block bodies address it.

    Parameter values are stored as numpy arrays indexed by position, but the
    blocks write ``model.xfflag[r, f, a]`` with element names. Pyomo bridges that
    by materializing a label -> value dict, pairing ``itertools.product`` over the
    domain elements with ``np.ndindex`` over the array: leftmost domain slowest,
    matching numpy's own axis order. The same pairing is used here — reading a
    parameter cell must mean the same thing in both backends.
    """

    __slots__ = ("_name", "_values", "_scalar")

    def __init__(self, name: str, values: dict[Any, float] | None, scalar: float | None) -> None:
        self._name = name
        self._values = values
        self._scalar = scalar

    def __getitem__(self, key: Any) -> float:
        if self._values is None:
            # A scalar parameter indexed anyway: Pyomo would raise, so surface it.
            raise KeyError(f"parameter {self._name!r} is scalar and has no index {key!r}")
        k = key[0] if isinstance(key, tuple) and len(key) == 1 else key
        try:
            return self._values[k]
        except KeyError as exc:
            raise KeyError(f"parameter {self._name!r} has no cell {k!r}") from exc

    def __iter__(self):
        """Iterate the parameter's keys, like a Pyomo indexed Param.

        Some equations walk a parameter's index space to build a filtered sum
        (``for (rr, f, a) in model.fcttx if rr == r``). Without this, Python falls
        back to integer indexing through __getitem__ and fails obscurely.
        """
        return iter(() if self._values is None else self._values)

    def __contains__(self, key: Any) -> bool:
        if self._values is None:
            return False
        k = key[0] if isinstance(key, tuple) and len(key) == 1 else key
        return k in self._values

    def __len__(self) -> int:
        return 0 if self._values is None else len(self._values)

    def keys(self):
        return () if self._values is None else self._values.keys()

    def __float__(self) -> float:
        if self._scalar is None:
            raise TypeError(f"parameter {self._name!r} is indexed, not scalar")
        return self._scalar

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        n = "scalar" if self._values is None else f"{len(self._values)} cells"
        return f"_LabelIndexedParam({self._name!r}, {n})"


class _ParameterView:
    """Exposes a ``ParameterManager`` by attribute, the way blocks read params.

    Blocks write ``model.alpha[r, i]``; the manager stores ``Parameter`` objects
    keyed by name. Each is converted once, on first access, into either a float
    (scalars) or a label-indexed view — never a solver handle, because the blocks
    branch on parameter values while the model is being built.
    """

    __slots__ = ("_manager", "_sets", "_cache")

    def __init__(self, manager: Any, set_manager: Any) -> None:
        self._manager = manager
        self._sets = set_manager
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        # __slots__ attributes resolve normally; this runs only for parameters.
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]

        manager = object.__getattribute__(self, "_manager")
        try:
            param = manager.get(name)
        except (KeyError, AttributeError) as exc:
            raise AttributeError(f"{name!r} is not a parameter") from exc

        view = self._materialize(name, param)
        cache[name] = view
        return view

    def _materialize(self, name: str, param: Any) -> Any:
        import itertools

        import numpy as np

        values = getattr(param, "value", param)
        domains = tuple(getattr(param, "domains", ()) or ())
        arr = np.asarray(values)

        if not domains:
            if arr.ndim == 0 or arr.size == 1:
                return float(arr.flatten()[0])
            # Pyomo skips these with a warning: they only seed initial values and
            # are never read from a constraint body.
            logger.warning("Parameter %s has no domains — not label-indexed", name)
            return _LabelIndexedParam(name, {}, None)

        set_manager = object.__getattribute__(self, "_sets")
        elems = [list(set_manager.get(d).iter_elements()) for d in domains]
        table: dict[Any, float] = {}
        for label_tuple, np_index in zip(
            itertools.product(*elems), np.ndindex(arr.shape), strict=True
        ):
            key = label_tuple[0] if len(label_tuple) == 1 else label_tuple
            table[key] = float(arr[np_index])
        return _LabelIndexedParam(name, table, None)


def build_gtap_equilibria_model(
    params: Any,
    residual_region: str | None = None,
    closure: Any = None,
) -> Any:
    """Assemble the seven GTAP blocks into an ``EquilibriaModel``.

    This is the first half of ``build_block_single_period`` — everything up to the
    point where that function commits to Pyomo. Both backends start here, so the
    model itself is never a variable in the comparison.
    """
    from equilibria.core.sets import Set as ESet
    from equilibria.model import Model
    from equilibria.templates.gtap.gtap_block_model import (
        _block_classes,
        _mk_unit,
        _set_elems,
    )

    if_sub = bool(getattr(closure, "if_sub", False))
    savf_flag = str(getattr(closure, "savf_flag", "capFix"))

    model = Model(name="gtap_blocks_sp")
    for name, elems in _set_elems(params.sets).items():
        model.add_set(ESet(name=name, elements=elems))

    for cls in _block_classes():
        model.add_block(
            _mk_unit(
                cls,
                params.sets,
                params,
                residual_region or "ROW",
                if_sub=if_sub,
                savf_flag=savf_flag,
            )
        )

    return model
