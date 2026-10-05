"""``@overwrite``: modificar un bloque GTAP desde el notebook de un ejercicio.

Un notebook es un ejercicio. Sus hooks se registran en la CLASE del bloque y duran
todo el proceso, asi que los ve todo lo que construye bloques: ``build_block_model``,
el modelo de un periodo que arma el driver (``_build_sp_reference``) y la calibracion.
El hook corre despues del ``setup`` original del bloque y recibe un ``BlockEdit``::

    @overwrite(ShockBlock, period="shock")
    def unemployment(b):
        b.endogenous("aft", ("USA", "LABOR"))

    @overwrite(ClosureBlock, period="shock")
    def real_wage(b):
        b.equation("eq_wreal", ("USA", "LABOR"),
                   lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r))

El cierre cambia SOLO en el shock, como el ``swap`` de GEMPACK y el de GAMS (compStat):
en base y check rige el cierre estandar (la celda fija, sin la fila nueva), asi que el
check sigue replicando el benchmark. ``period`` solo acepta ``"shock"``.

Un objetivo de cantidad (TBL94: la produccion -1%; ME9A: el PIB real) se declara con
``b.target``: una Var nueva, registrada como instrumento y fija en 1, y la fila
``quantity[shock] = target[shock] x quantity[check]``. El shock entra como factor con
``fix_instrument_shock``::

    @overwrite(ShockBlock, period="shock")
    def regulation(b):
        b.endogenous("prdtx_rai", ("USA", "MFG", "MFG"))
        b.target("qca_target", ("USA", "MFG", "MFG"),
                 quantity=lambda m, r, a, i: m.x[r, a, i], domains=("r", "a", "i"))

La regla de ``equation`` (y ``quantity``) se escribe sobre el modelo de UN periodo
(sin ``t``). Al construirse, las Vars tienen sus niveles de base (cal.gms), asi que
``value(...)`` dentro de una regla de ``equation`` es una constante de BASE; para
anclar al check (que en ME9 no reproduce la base) usar ``b.target``.

Spec: dev-tools/equilibria-tools/plans/superpowers/specs/2026-10-01-overwrite-bloques-design.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from equilibria.blocks.base import Block
from equilibria.blocks.gtap.periods import CHECK, SHOCK, as_key
from equilibria.core.symbolic_equations import SymbolicEquation

Hook = Callable[["BlockEdit"], None]

# clase del bloque -> {nombre de la funcion: hook}, en orden de registro.
_HOOKS: dict[type, dict[str, Hook]] = {}

# Orden en que se busca el set de cada etiqueta de una celda (``equation``).
_SET_ORDER = ("r", "f", "a", "i", "aa", "rp")


class Target(NamedTuple):
    """Un objetivo de ``b.target``: sus dominios y sus celdas ``(celda, quantity)``."""

    domains: tuple[str, ...]
    cells: list[tuple[tuple, Callable[..., Any]]]


def _merge_target(
    store: dict[str, Target],
    name: str,
    domains: tuple[str, ...],
    cells: list[tuple[tuple, Callable[..., Any]]],
) -> None:
    """Suma ``cells`` al objetivo ``name`` de ``store``; falla si cambian los dominios."""
    known = store.setdefault(name, Target(domains, []))
    if known.domains != domains:
        raise ValueError(f"target {name!r}: dominios {domains} != {known.domains}")
    known.cells.extend(cells)


def target_row(name: str) -> str:
    """El nombre de la fila de un objetivo de ``b.target``."""
    return f"eq_{name}"


class _Row(NamedTuple):
    """Una familia de filas de ``b.equation``: sus dominios y ``{celda: regla}``."""

    domains: tuple[str, ...]
    rules: dict[tuple, Callable[..., Any]]

    def symbolic(self, name: str) -> SymbolicEquation:
        rules = self.rules

        class _Eq(SymbolicEquation):
            def build_expression(self, pyomo_model, indices):
                rule = rules.get(tuple(indices))
                return None if rule is None else rule(pyomo_model, *indices)

        return _Eq(name=name, domains=self.domains)


class _Overwrite:
    """El decorador, mas ``clear``/``registered``/``active`` (para tests y el cache)."""

    def __call__(self, cls: type, period: str = SHOCK) -> Callable[[Hook], Hook]:
        if period != SHOCK:
            raise ValueError(
                f"@overwrite: period={period!r}; el cierre solo cambia en el "
                f"{SHOCK!r} (en base y check rige el estandar)"
            )

        def deco(fn: Hook) -> Hook:
            # Clave = nombre de la funcion: re-ejecutar la celda del notebook
            # reemplaza el hook en vez de apilarlo.
            _HOOKS.setdefault(cls, {})[getattr(fn, "__name__", repr(fn))] = fn
            return fn

        return deco

    def clear(self) -> None:
        _HOOKS.clear()

    def registered(self, cls: type) -> list[str]:
        return list(_HOOKS.get(cls, {}))

    def active(self) -> bool:
        return any(_HOOKS.values())


overwrite = _Overwrite()


def require_blocks(where: str) -> None:
    """Falla si hay hooks activos: ``where`` construye el modelo SIN bloques (el
    monolito), asi que los ignoraria en silencio."""
    if overwrite.active():
        raise RuntimeError(
            f"{where}: hay hooks @overwrite activos y el monolito no los aplica; "
            "usar el camino de bloques (sin EQUILIBRIA_GTAP_REF_MODEL=monolith)"
        )


class BlockEdit:
    """Lo que un hook puede cambiar de un bloque ya armado por su ``setup``."""

    def __init__(self, set_manager: Any, variables: dict, equations: list) -> None:
        self.set_manager = set_manager
        self.variables = variables
        self.equations = equations
        self._endogenous: dict[str, set[tuple]] = {}
        self._targets: dict[str, Target] = {}
        self._rows: dict[str, _Row] = {}

    def endogenous(self, name: str, cell: tuple) -> None:
        """La celda ``cell`` del instrumento ``name`` deja de ser exogena, solo en el
        shock: en base y check sigue fija en su benchmark."""
        from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS

        if name not in SHOCK_INSTRUMENTS:
            raise ValueError(
                f"{name!r} no es un instrumento del ShockBlock; instrumentos: "
                f"{sorted(SHOCK_INSTRUMENTS)}"
            )
        self._endogenous.setdefault(name, set()).add(tuple(cell))

    def equation(
        self,
        name: str,
        cell: tuple,
        rule: Callable[..., Any],
        domains: tuple[str, ...] | None = None,
    ) -> None:
        """Agrega la fila ``name`` sobre la celda ``cell``: ``rule(m, *cell)``, solo en
        el shock. Llamarla otra vez con el mismo nombre suma otra celda a la familia."""
        cell = tuple(cell)
        doms = tuple(domains) if domains is not None else self._domains(cell)
        row = self._rows.setdefault(name, _Row(doms, {}))
        if row.domains != doms:
            raise ValueError(f"equation {name!r}: dominios {doms} != {row.domains}")
        if cell in row.rules:
            raise ValueError(f"equation {name!r}: la celda {cell} ya tiene fila")
        row.rules[cell] = rule

    def target(
        self,
        name: str,
        cell: tuple,
        quantity: Callable[..., Any],
        domains: tuple[str, ...] | None = None,
    ) -> None:
        """Declara el objetivo ``name`` sobre ``quantity(m, *cell)``.

        ``name`` es una Var nueva, registrada como instrumento y fija en 1 en los 3
        periodos (se agrega despues de todos los bloques, ``add_targets``). La fila
        ``eq_<name>`` vive solo en el shock: ``quantity[shock] = name[shock] x
        quantity[check]`` (``shock_only``), como el ``swap`` de GEMPACK. En el
        modelo de un periodo (sin check) ancla a la base SIN escalar: la constante se
        lee al construir la fila, antes de ``apply_production_scaling``; el
        multiperiodo la reescribe."""
        from pyomo.environ import value

        cell = tuple(cell)
        doms = tuple(domains) if domains is not None else self._domains(cell)
        _merge_target(self._targets, name, doms, [(cell, quantity)])
        self.equation(
            target_row(name),
            cell,
            lambda m, *c: quantity(m, *c)
            == getattr(m, name)[c] * float(value(quantity(m, *c))),
            domains=doms,
        )

    def endogenized(self) -> dict[str, set[tuple]]:
        """Las celdas que ``endogenous`` libero, por instrumento."""
        return {k: set(v) for k, v in self._endogenous.items()}

    def declared_targets(self) -> dict[str, Target]:
        """Los objetivos que declaro ``target``."""
        return {k: Target(t.domains, list(t.cells)) for k, t in self._targets.items()}

    def row_names(self) -> set[str]:
        """Los nombres de las filas que agregaron ``equation`` y ``target``."""
        return set(self._rows)

    def row_equations(self) -> list[SymbolicEquation]:
        """Las filas de ``equation`` y ``target``, una familia por nombre."""
        return [row.symbolic(name) for name, row in self._rows.items()]

    def _domains(self, cell: tuple) -> tuple[str, ...]:
        doms = []
        for label in cell:
            for s in _SET_ORDER:
                if s in self.set_manager and label in list(self.set_manager.get(s)):
                    doms.append(s)
                    break
            else:
                raise ValueError(
                    f"la etiqueta {label!r} no esta en ningun set {_SET_ORDER}; "
                    "pasar domains= explicitamente"
                )
        return tuple(doms)


class _HookedBlock(Block):
    """Envuelve un bloque y le aplica los hooks de su clase despues del ``setup``."""

    inner: Any = None
    hooks: list[Any] = []
    endogenous: dict[str, set[tuple]] = {}
    targets: dict[str, Target] = {}
    rows: set[str] = set()

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        equations = list(self.inner.setup(set_manager, parameters, variables))
        edit = BlockEdit(set_manager, variables, equations)
        for fn in self.hooks:
            fn(edit)
        for name, cells in edit.endogenized().items():
            self.endogenous.setdefault(name, set()).update(cells)
        for name, t in edit.declared_targets().items():
            _merge_target(self.targets, name, t.domains, t.cells)
        self.rows.update(edit.row_names())
        return [*edit.equations, *edit.row_equations()]


def with_overwrites(block: Any) -> Any:
    """El bloque tal cual si su clase no tiene hooks; si no, envuelto."""
    hooks = list(_HOOKS.get(type(block), {}).values())
    if not hooks:
        return block
    return _HookedBlock(
        name=block.name,
        description=block.description,
        required_sets=list(block.required_sets),
        inner=block,
        hooks=hooks,
        endogenous={},
        targets={},
        rows=set(),
    )


def endogenous_cells(blocks: list[Any]) -> dict[str, frozenset[tuple]]:
    """Las celdas de instrumento que los hooks volvieron endogenas."""
    out: dict[str, set[tuple]] = {}
    for b in blocks:
        for name, cells in getattr(b, "endogenous", {}).items():
            out.setdefault(name, set()).update(cells)
    return {k: frozenset(v) for k, v in out.items()}


def shock_rows(blocks: list[Any]) -> frozenset[str]:
    """Los nombres de las filas que agregaron los hooks (rigen solo en el shock)."""
    return frozenset(n for b in blocks for n in getattr(b, "rows", ()))


def collect_targets(blocks: list[Any]) -> dict[str, Target]:
    """Los objetivos que declararon los hooks de todos los bloques."""
    out: dict[str, Target] = {}
    for b in blocks:
        for name, t in getattr(b, "targets", {}).items():
            _merge_target(out, name, t.domains, t.cells)
    return out


def add_targets(model: Any, targets: dict[str, Target]) -> None:
    """Agrega al ``equilibria.model.Model`` ya armado la Var de cada objetivo, en 1.

    Va despues de todos los bloques: ``add_block`` descarta en silencio una Var cuyo
    nombre ya existe (algunos bloques comparten ``ev``/``cv``/``pwfact`` a
    proposito), asi que un objetivo con nombre ya usado tiene que fallar aca.
    """
    from equilibria.blocks.gtap import _derived_params as dp
    from equilibria.core.variables import Variable

    for name, t in targets.items():
        if name in model.variable_manager:
            raise ValueError(f"target {name!r}: ya hay una variable con ese nombre")
        elems = [list(model.set_manager.get(d)) for d in t.domains]
        model.add_variable(
            Variable(
                name=name,
                value=dp.to_array({}, elems, 1.0),
                domains=t.domains,
                domain="Reals",
                lower=float("-inf"),
                upper=float("inf"),
            )
        )


class _PeriodIndex:
    """``var[k]`` de un periodo: ``var[(*k, t)]`` del modelo multiperiodo."""

    def __init__(self, var: Any, period: str) -> None:
        self._var = var
        self._t = period

    def __getitem__(self, k: Any) -> Any:
        return self._var[(*as_key(k), self._t)]


class _AtPeriod:
    """Vista de UN periodo del modelo multiperiodo: ``v[k]`` es ``v[(*k, t)]``.

    Deja evaluar una regla escrita para el modelo de un periodo (``quantity`` de
    ``b.target``) sobre las Vars de cualquier periodo. Una Var escalar del modelo de
    un periodo esta indexada solo por el periodo: se devuelve su celda. Solo traduce
    Vars: otro componente indexado (un Param o una Expression por periodo) falla, en
    vez de leerse sin el periodo."""

    def __init__(self, m: Any, period: str) -> None:
        self._m = m
        self._t = period

    def __getattr__(self, name: str) -> Any:
        from pyomo.environ import Var

        comp = getattr(self._m, name)
        if getattr(comp, "ctype", None) is Var:
            if comp.dim() == 1:
                return comp[self._t]
            return _PeriodIndex(comp, self._t)
        if callable(getattr(comp, "is_indexed", None)) and comp.is_indexed():
            raise ValueError(
                f"b.target: la regla quantity lee {name!r}, que esta indexado y no "
                "es una Var; solo se pueden leer Vars (se traducen al periodo)"
            )
        return comp


def shock_only(m: Any) -> None:
    """Deja las filas de los hooks solo en el shock y ancla las de ``b.target`` al
    check: ``quantity[shock] = target[shock] x quantity[check]``.

    Corre despues de reflejar las filas del modelo de un periodo (en todos los
    periodos o en uno, ``build_equations_intra``); es idempotente."""
    rows = getattr(m, "_shock_rows", None) or frozenset()
    for name in rows:
        con = getattr(m, name, None)
        if con is None:
            continue
        for idx in [i for i in con if i[-1] != SHOCK]:
            del con[idx]
    targets: dict[str, Target] = getattr(m, "_targets", None) or {}
    shock, check = _AtPeriod(m, SHOCK), _AtPeriod(m, CHECK)
    for name, t in targets.items():
        con = getattr(m, target_row(name), None)
        if con is None:
            continue
        for cell, quantity in t.cells:
            idx = (*cell, SHOCK)
            if idx in con:
                con[idx].set_value(
                    quantity(shock, *cell)
                    == getattr(m, name)[idx] * quantity(check, *cell)
                )


def ppriv_tornqvist(m: Any, r: str) -> Any:
    """Indice Tornqvist del consumo de hogares de ``r`` contra la base.

    ``ln T = sum_i 1/2 (s_i,0 + s_i) ln(pa_i / pa_i,0)``, con
    ``s_i = pa*xaa(hhd) / sum pa*xaa(hhd)``. Es el ``ppriv`` de GEMPACK (un Divisia)
    en un paso: en TBL65A, el Divisia sobre el camino de GAMS da +9,2576, el
    Tornqvist +9,2574 y ``ppriv`` +9,2580. Las constantes de base se leen al
    construir la fila, cuando las Vars tienen sus niveles de base.
    """
    from pyomo.environ import exp, log, value

    hh = [i for i in m.i if float(value(m.xaa[r, i, "hhd"])) > 0.0]
    pa0 = {i: float(value(m.pa[r, i, "hhd"])) for i in hh}
    e0 = {i: pa0[i] * float(value(m.xaa[r, i, "hhd"])) for i in hh}
    tot0 = sum(e0.values())
    gasto = sum(m.pa[r, i, "hhd"] * m.xaa[r, i, "hhd"] for i in hh)
    return exp(
        sum(
            0.5
            * (e0[i] / tot0 + m.pa[r, i, "hhd"] * m.xaa[r, i, "hhd"] / gasto)
            * log(m.pa[r, i, "hhd"] / pa0[i])
            for i in hh
        )
    )
