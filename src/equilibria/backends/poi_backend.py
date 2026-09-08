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

    def __init__(
        self,
        jit: str = "LLVM",
        opt_level: int | None = 0,
        row_filter: Any = None,
    ) -> None:
        self.poi_model: Any = None
        self.adapter: PoiModelAdapter | None = None
        self.constraints: dict[str, Any] = {}
        self.skipped: dict[str, int] = {}
        self._model: Any = None
        # Optional predicate ``(eq_name, indices) -> bool`` deciding which rows to
        # build. The GTAP closure deactivates individual CELLS rather than whole
        # equations — a zero-flow trade route drops one row of eq_pfeq while its 30
        # siblings stay — so a caller matching the harness must supply the row set
        # rather than a rule. Rows rejected here are never handed to POI, which is
        # what keeps the system square: building them and discarding the handle
        # afterwards would leave them in the model.
        self._row_filter = row_filter
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
            var_init=_VariableInit(model),
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
                if self._row_filter is not None and not self._row_filter(
                    eq_name, indices
                ):
                    self.skipped[eq_name] = self.skipped.get(eq_name, 0) + 1
                    continue

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
                    # Only a row that was really created counts as "mentioning"
                    # its variables; a skipped cell must not anchor anything.
                    object.__getattribute__(
                        self.adapter, "_used_in_constraints"
                    ).update(self.adapter._touched)

    def seed_from_pyomo(self, pyomo_model: Any) -> int:
        """Copy a Pyomo model's variable values in as POI start values.

        The GTAP solve is warm-started from the benchmark, and starting anywhere
        else lands on a different equilibrium — the basin trap documented in
        equilibria-parity-debug. Returns how many variables were seeded.
        """
        from pyoptinterface import VariableAttribute
        from pyomo.environ import Var, value as pyomo_value

        adapter = self.adapter
        seeded = 0
        for var in pyomo_model.component_data_objects(Var):
            base, _, rest = var.name.partition("[")
            proxy = object.__getattribute__(adapter, "_vars").get(base)
            if proxy is None:
                continue
            key: tuple = ()
            if rest:
                key = tuple(rest.rstrip("]").split(","))
            try:
                handle = proxy[key] if key else proxy[()]
                val = float(pyomo_value(var))
            except Exception:  # noqa: BLE001 - an unseeded cell keeps POI's default
                continue
            self.poi_model.set_variable_attribute(
                handle, VariableAttribute.PrimalStart, val
            )
            seeded += 1
        return seeded

    def apply_closure_from_pyomo(
        self, pyomo_model: Any, period: str | None = None
    ) -> dict[str, int]:
        """Mirror a closed Pyomo model's fixed variables onto the POI model.

        The closure decides which variables are exogenous and which equations stay
        active — ``apply_closure`` plus ``apply_conditional_fixing``, several
        hundred lines that read benchmark flows to decide, for instance, that a
        bilateral route with no trade in the data is fixed rather than solved.
        Reimplementing that here would duplicate logic whose whole point is to
        match GAMS, and any divergence would be invisible until a parity gate
        caught it.

        So the closure is applied once, by the code that owns it, to a Pyomo model;
        this copies the outcome. A variable Pyomo fixed is bounded to its value in
        POI, which is how POI expresses the same thing.

        Constraints are NOT deactivated here: ``build`` already skips exactly the
        cells Pyomo skips, so the two row sets already match.

        ``period`` selects which period's cells to copy (the POI model is a single
        period, while the Pyomo model carries base/check/shock). The period suffix
        is stripped from the key, since POI's variables are not period-indexed.

        Read ``var.fixed`` and nothing else. A variable can be BOTH fixed here and
        listed as free by the square-system hook, because that list is captured
        before the harness applies its fixings — ``pwfact`` and ``pnum`` are fixed
        at 1.0 yet ``pwfact`` appears free. Trusting the free list instead of this
        flag leaves the numeraire unanchored, and every price then comes out scaled
        by a common factor (measured: 0.99968) while the quantities match.
        """
        from pyomo.environ import Var, value as pyomo_value

        adapter = self.adapter
        proxies = object.__getattribute__(adapter, "_vars")
        fixed = missing = 0

        for var in pyomo_model.component_data_objects(Var):
            if not var.fixed:
                continue
            base, _, rest = var.name.partition("[")
            proxy = proxies.get(base)
            if proxy is None:
                missing += 1
                continue
            key: tuple = tuple(rest.rstrip("]").split(",")) if rest else ()
            if period is not None:
                if not key or key[-1] != period:
                    continue
                key = key[:-1]
            try:
                handle = proxy[key] if key else proxy[()]
                val = float(pyomo_value(var))
            except Exception:  # noqa: BLE001 - a cell POI never built stays absent
                missing += 1
                continue
            self.poi_model.set_variable_bounds(handle, val, val)
            fixed += 1

        return {"fixed": fixed, "not_in_poi": missing}

    def pin_unconstrained(self) -> int:
        """Pin every variable that no constraint mentions, at its start value.

        The blocks skip a cell whose flag is off — ``xfflag[r,f,a] <= 0`` yields no
        equation — so that variable enters the model with nothing to anchor it and
        an optimizer is free to move it anywhere. Measured on the 3x3, that put
        1.59 into `xf[ROW,NatRes,Svces]`, a natural-resources-in-services cell that
        does not exist and that the harness holds at 0.

        The harness fixes these; matching that is what makes the two comparable.
        Returns how many variables were pinned.
        """
        from pyoptinterface import VariableAttribute

        adapter = self.adapter
        mentioned = {
            (name, key)
            for name, proxy in object.__getattribute__(adapter, "_vars").items()
            for key in proxy._cache
        }
        used = object.__getattribute__(adapter, "_used_in_constraints")

        pinned = 0
        for name, key in mentioned - used:
            proxy = object.__getattribute__(adapter, "_vars")[name]
            handle = proxy._cache[key]
            start = self.poi_model.get_variable_attribute(
                handle, VariableAttribute.PrimalStart
            )
            val = 0.0 if start is None else float(start)
            self.poi_model.set_variable_bounds(handle, val, val)
            pinned += 1
        return pinned

    def solve_min_walras(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Solve by minimising the Walras residual.

        GTAP is a square root-finding system, and POI solves optimization
        problems. Minimising ``walras**2`` — the economy-wide imbalance, which is
        zero at equilibrium — states the same problem in the form the solver
        expects: the optimum is known in advance to be zero, so the objective
        value is itself a check on whether the root was found.
        """
        import time

        from pyoptinterface import ModelAttribute

        walras = object.__getattribute__(self.adapter, "_vars").get("walras")
        if walras is None:
            raise RuntimeError("no 'walras' variable — cannot form the objective")

        # GAMS states this as `solve gtap using nlp maximizing walras`, keeping
        # eq_walras active: that equation DEFINES walras as the economy-wide excess
        # income, whose largest feasible value is zero at equilibrium, so maximizing
        # it drives the system to market clearing. Squaring and minimising instead
        # leaves eq_walras active as well, which pins walras to its defined value
        # and makes the objective a constant — the harness comment at
        # run_gtap.py:5137 spells this out.
        from pyoptinterface import ObjectiveSense

        # POI's ipopt.Model only MINIMISES: its docs say a maximization must be
        # negated by hand. Passing ObjectiveSense.Maximize is accepted and then
        # silently ignored — measured on a one-liner, `max x` over x in [0,3]
        # returned 0.0 instead of 3.0 — so the model would solve the OPPOSITE
        # problem. Negating keeps GAMS's `maximizing walras` intact.
        self.poi_model.set_objective(-walras[()], sense=ObjectiveSense.Minimize)

        for key, val in (options or {}).items():
            if isinstance(val, str):
                self.poi_model.set_raw_option_string(key, val)
            elif isinstance(val, int) and not isinstance(val, bool):
                self.poi_model.set_raw_option_int(key, val)
            else:
                self.poi_model.set_raw_option_double(key, float(val))

        t0 = time.perf_counter()
        self.poi_model.optimize()
        elapsed = time.perf_counter() - t0

        return {
            "wall_s": elapsed,
            "status": str(self.poi_model.get_model_attribute(ModelAttribute.TerminationStatus)),
            # get_obj_value() is the NEGATED objective we minimised; report walras.
            "objective": -float(self.poi_model.get_obj_value()),
            "walras": float(self.poi_model.get_value(walras[()])),
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"PoiBackend(constraints={len(self.constraints)}, "
            f"skipped={sum(self.skipped.values())})"
        )


class _VariableInit:
    """Start values and bounds for a variable cell, keyed by set labels.

    The Pyomo backend passes each variable's initial value and bounds to Pyomo when
    it declares it. Without the same information POI starts every variable at its
    own default and leaves it unbounded, which matters most for the cells the model
    switches off: a cell whose flag is zero gets no equation from the blocks, so
    nothing anchors it and the optimizer is free to move it anywhere. Measured on
    the 3x3, that put 26.96 into `xf[ROW,NatRes,Svces]` — natural resources in
    services, a combination that does not exist and that the harness keeps at 0.
    """

    __slots__ = ("_model", "_cache")

    def __init__(self, model: Any) -> None:
        self._model = model
        self._cache: dict[str, Any] = {}

    def _table(self, name: str):
        """(values, lower, upper) tables for one variable, built once."""
        import itertools

        import numpy as np

        hit = self._cache.get(name)
        if hit is not None:
            return hit

        try:
            var = self._model.variable_manager.get(name)
        except (KeyError, AttributeError):
            self._cache[name] = None
            return None

        domains = tuple(getattr(var, "domains", ()) or ())
        values = np.asarray(var.value)
        lower = np.asarray(var.lower) if var.lower is not None else None
        upper = np.asarray(var.upper) if var.upper is not None else None

        if not domains:
            table = {(): (float(values.flatten()[0]), _scalar(lower), _scalar(upper))}
        else:
            elems = [
                list(self._model.set_manager.get(d).iter_elements()) for d in domains
            ]
            table = {}
            for labels, idx in zip(
                itertools.product(*elems), np.ndindex(values.shape), strict=True
            ):
                table[labels] = (
                    float(values[idx]),
                    _cell(lower, idx),
                    _cell(upper, idx),
                )

        self._cache[name] = table
        return table

    def get(self, name: str, key: tuple) -> tuple | None:
        table = self._table(name)
        if not table:
            return None
        return table.get(key)


def _scalar(arr) -> float | None:
    if arr is None:
        return None
    import numpy as np

    flat = np.asarray(arr).flatten()
    return float(flat[0]) if flat.size else None


def _cell(arr, idx) -> float | None:
    if arr is None:
        return None
    import numpy as np

    a = np.asarray(arr)
    if a.ndim == 0 or a.size == 1:
        return float(a.flatten()[0])
    try:
        return float(a[idx])
    except (IndexError, TypeError):
        return None


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

    _overwrite_fisher_snapshot(model, params)
    return model


def _overwrite_fisher_snapshot(model: Any, params: Any) -> dict[str, float]:
    """The composer carry of ``blocks/gtap/__init__.py`` item 3, POI side.

    The CLOSURE block seeds ``xf0`` from the UN-scaled benchmark (``_vfm_init``);
    the monolith snapshots ``xf.l`` after ``apply_production_scaling``. The two are
    related exactly by ``xf_scaled = xf0 * xscale`` (verified on all 45 cells of
    gtap7_3x3), so the post-scaling snapshot is reachable here without building a
    Pyomo model first — which matters because this function runs while the model is
    still an ``EquilibriaModel``.

    ``mqfactw_bb`` goes from 3020.22 to 66.75, matching the monolith. The Pyomo path
    gets the same correction in ``build_block_single_period`` via
    ``apply_fisher_snapshot_overwrite``; both must be fixed together or the two
    backends disagree on these rows (measured: mfw_sb 66.75 vs 3021.19).

    Unlike Pyomo's immutable Params, the values live in the ParameterManager and are
    rewritten in place here, before any backend folds them into an expression.
    """
    from equilibria.templates.gtap.gtap_block_model import _set_elems

    setmap = _set_elems(params.sets)
    regions = list(setmap["r"])
    facs = list(setmap["f"])
    acts = list(setmap["a"])
    agents = list(setmap["aa"])

    xf0_p = model.get_parameter("xf0")
    pf0_p = model.get_parameter("pf0")
    xscale_p = model.get_parameter("xscale")
    if xf0_p is None or pf0_p is None or xscale_p is None:
        return {}

    xf0 = xf0_p.value
    pf0 = pf0_p.value
    xscale = xscale_p.value

    # xf0 -> the SCALED level the monolith would have snapshotted.
    for i, _r in enumerate(regions):
        for j, _f in enumerate(facs):
            for k, a in enumerate(acts):
                xf0[i, j, k] = xf0[i, j, k] * float(xscale[i, agents.index(a)])

    mqfactr: list[float] = []
    mqfactw = 0.0
    for i, _r in enumerate(regions):
        s_reg = 0.0
        for j, _f in enumerate(facs):
            for k, a in enumerate(acts):
                xs = float(xscale[i, agents.index(a)])
                if xs <= 1e-12:
                    continue
                s_reg += float(pf0[i, j, k]) * float(xf0[i, j, k]) / xs
        mqfactr.append(s_reg if s_reg > 0.0 else 1.0)
        mqfactw += s_reg

    mqr = model.get_parameter("mqfactr_bb")
    if mqr is not None:
        for i, v in enumerate(mqfactr):
            mqr.value[i] = v
    mqw = model.get_parameter("mqfactw_bb")
    if mqw is not None:
        mqw.value[0] = mqfactw if mqfactw > 0.0 else 1.0

    return {"mqfactw_bb": mqfactw, "mqfactr_bb": dict(zip(regions, mqfactr))}


def _attach_feasibility_solver() -> None:
    """Add PoiBackend.solve_feasibility — a pure root-find, no objective.

    The harness fixes `walras` at 0 and keeps eq_walras active, so that row becomes
    a constraint demanding market clearing rather than a definition with a free
    variable to optimize. With every equation an equality and every exogenous cell
    pinned, the problem is square: there is nothing to maximize, only residuals to
    drive to zero.
    """
    import time as _time
    from typing import Any as _Any

    def solve_feasibility(self, options: dict | None = None) -> dict[str, _Any]:
        from pyoptinterface import ModelAttribute

        self.poi_model.set_objective(0.0)
        for key, val in (options or {}).items():
            if isinstance(val, str):
                self.poi_model.set_raw_option_string(key, val)
            elif isinstance(val, int) and not isinstance(val, bool):
                self.poi_model.set_raw_option_int(key, val)
            else:
                self.poi_model.set_raw_option_double(key, float(val))

        t0 = _time.perf_counter()
        self.poi_model.optimize()
        elapsed = _time.perf_counter() - t0

        walras = object.__getattribute__(self.adapter, "_vars").get("walras")
        return {
            "wall_s": elapsed,
            "status": str(
                self.poi_model.get_model_attribute(ModelAttribute.TerminationStatus)
            ),
            "walras": (
                float(self.poi_model.get_value(walras[()])) if walras else float("nan")
            ),
        }

    PoiBackend.solve_feasibility = solve_feasibility


_attach_feasibility_solver()
