"""``@overwrite``: modificar un bloque GTAP desde el notebook de un ejercicio.

Un notebook es un ejercicio. Sus hooks se registran en la CLASE del bloque y duran
todo el proceso, asi que los ve todo lo que construye bloques: ``build_block_model``,
el modelo de un periodo que arma el driver (``_build_sp_reference``) y la calibracion.
El hook corre despues del ``setup`` original del bloque y recibe un ``BlockEdit``::

    @overwrite(ShockBlock)
    def unemployment(b):
        b.endogenous("aft", ("USA", "LABOR"))

    @overwrite(ClosureBlock)
    def real_wage(b):
        b.equation("eq_wreal", ("USA", "LABOR"),
                   lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r))

Un objetivo por periodo (p.ej. TBL94: la produccion fija en -1% en el shock) se declara
con ``b.target``: una Var nueva, registrada como instrumento y fija en
``initial(m, *cell)``; el shock entra con ``fix_instrument_shock``, solo en 'shock'::

    @overwrite(ShockBlock)
    def regulation(b):
        b.endogenous("prdtx_rai", ("USA", "MFG", "MFG"))
        b.target("qca_target", ("USA", "MFG", "MFG"),
                 initial=lambda m, r, a, i: value(m.x[r, a, i]), domains=("r", "a", "i"))

La regla de ``equation`` se escribe sobre el modelo de UN periodo (sin ``t``): la
reflexion multiperiodo la copia a base/check/shock. Al construirse, las Vars tienen sus
niveles de base (cal.gms), asi que ``value(...)`` dentro de la regla es una constante
de base.

Spec: dev-tools/equilibria-tools/plans/superpowers/specs/2026-10-01-overwrite-bloques-design.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from equilibria.blocks.base import Block
from equilibria.core.symbolic_equations import SymbolicEquation

Hook = Callable[["BlockEdit"], None]

# clase del bloque -> {nombre de la funcion: hook}, en orden de registro.
_HOOKS: dict[type, dict[str, Hook]] = {}

# Orden en que se busca el set de cada etiqueta de una celda (``equation``).
_SET_ORDER = ("r", "f", "a", "i", "aa", "rp")


class Target(NamedTuple):
    """Un objetivo de ``b.target``: sus dominios y sus celdas ``(celda, initial)``."""

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


class _Overwrite:
    """El decorador, mas ``clear``/``registered``/``active`` (para tests y el cache)."""

    def __call__(self, cls: type) -> Callable[[Hook], Hook]:
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

    def endogenous(self, name: str, cell: tuple) -> None:
        """La celda ``cell`` del instrumento ``name`` deja de ser exogena.

        Queda libre en los 3 periodos, asi que la fila que la reemplaza (``equation``)
        debe anclarse a la BASE: ``value(...)`` al construir lee los niveles de base.
        GAMS (``compStat``) ancla al CHECK. Coinciden solo si el check reproduce la
        base; en nus333 lo hace, y los tests de TBL65B y TBL94 lo verifican (el
        instrumento liberado queda en el check igual que en la base, a 1e-6).
        """
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
        """Agrega la fila ``name`` sobre la celda ``cell``: ``rule(m, *cell)``."""
        cell = tuple(cell)
        doms = domains if domains is not None else self._domains(cell)

        class _Eq(SymbolicEquation):
            def build_expression(self, pyomo_model, indices):
                if tuple(indices) != cell:
                    return None
                return rule(pyomo_model, *cell)

        self.equations.append(_Eq(name=name, domains=doms))

    def target(
        self,
        name: str,
        cell: tuple,
        initial: Callable[..., Any],
        domains: tuple[str, ...] | None = None,
    ) -> None:
        """Declara el objetivo ``name``: una Var nueva, registrada como instrumento
        (fija en los 3 periodos, el driver no la libera) y con la celda ``cell`` en
        ``initial(m, *cell)``, evaluado con los niveles de base al construir el modelo.
        Las demas celdas quedan fijas en 0 (ninguna fila las lee). La Var se agrega
        al modelo despues de todos los bloques (``add_targets``)."""
        cell = tuple(cell)
        doms = tuple(domains) if domains is not None else self._domains(cell)
        _merge_target(self._targets, name, doms, [(cell, initial)])

    def endogenized(self) -> dict[str, set[tuple]]:
        """Las celdas que ``endogenous`` libero, por instrumento."""
        return {k: set(v) for k, v in self._endogenous.items()}

    def declared_targets(self) -> dict[str, Target]:
        """Los objetivos que declaro ``target``."""
        return {k: Target(t.domains, list(t.cells)) for k, t in self._targets.items()}

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

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        equations = list(self.inner.setup(set_manager, parameters, variables))
        edit = BlockEdit(set_manager, variables, equations)
        for fn in self.hooks:
            fn(edit)
        for name, cells in edit.endogenized().items():
            self.endogenous.setdefault(name, set()).update(cells)
        for name, t in edit.declared_targets().items():
            _merge_target(self.targets, name, t.domains, t.cells)
        return edit.equations


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
    )


def endogenous_cells(blocks: list[Any]) -> dict[str, frozenset[tuple]]:
    """Las celdas de instrumento que los hooks volvieron endogenas."""
    out: dict[str, set[tuple]] = {}
    for b in blocks:
        for name, cells in getattr(b, "endogenous", {}).items():
            out.setdefault(name, set()).update(cells)
    return {k: frozenset(v) for k, v in out.items()}


def collect_targets(blocks: list[Any]) -> dict[str, Target]:
    """Los objetivos que declararon los hooks de todos los bloques."""
    out: dict[str, Target] = {}
    for b in blocks:
        for name, t in getattr(b, "targets", {}).items():
            _merge_target(out, name, t.domains, t.cells)
    return out


def add_targets(model: Any, targets: dict[str, Target]) -> None:
    """Agrega al ``equilibria.model.Model`` ya armado la Var de cada objetivo, en 0.

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
                value=dp.to_array({}, elems, 0.0),
                domains=t.domains,
                domain="Reals",
                lower=float("-inf"),
                upper=float("inf"),
            )
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
