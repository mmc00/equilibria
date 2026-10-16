"""Mapping: como se corresponden dos soluciones, declarado por el usuario.

El modulo no adivina nada: los nombres de variable de cada lado, el orden de sus
dimensiones, el renombre de elementos y de periodos, la tolerancia de cada variable
y los umbrales del resumen salen del mapeo. Lo que falta es error al cargar, no un
default escondido.

Forma (YAML o dict)::

    periods:                 # opcional; renombre de periodos por lado -> nombre comun
      b: {pre: check, post: shock}
    sets:                    # opcional; renombre de elementos POR CONJUNTO y lado
      ACT: {a: {a_Food: Food}}
    variables:
      - name: output         # nombre comun en el reporte
        a: {var: xp, dims: [REG, ACT]}
        b: {var: qo, dims: [ACT, REG]}
        tol: {rel: 0.001, abs_floor: 1.0e-6}
    exclude:
      - {var: ppriv, why: "indice Divisia: sin forma en niveles"}
      - var: factor_price    # solo algunas celdas de una variable del mapeo
        where: {FP: [NatRes], period: [shock]}
        why: "GAMS define pft solo para factores moviles"
    thresholds: [0.001, 0.005, 0.01]

Cada entrada de ``dims`` nombra el conjunto de esa dimension (sin periodo). Las de A
y B deben ser las mismas, en el orden que tenga cada fuente; el orden comun del
reporte es el de A. Si una variable repite conjunto (p. ej. origen y destino),
cada dimension necesita su propia etiqueta y su propio renombre.
Un elemento o periodo sin renombre declarado conserva su nombre.

Una exclusion sin ``where`` saca una variable entera (que no puede estar en
``variables``). Con ``where`` saca solo algunas celdas de una variable DEL MAPEO
(``var`` es su ``name``): cada clave es una de sus ``dims`` (o ``period``) y su
valor la lista de elementos (o periodos) con nombres comunes, ya renombrados. Una
celda queda excluida si cumple todas las claves. Sirve para las celdas que un lado
no define (GAMS omite del GDX lo que su modelo no tiene, igual que los ceros). El
``why`` es obligatorio siempre.
"""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from equilibria.compare.errors import MappingError
from equilibria.compare.solution import Index

SIDES = ("a", "b")


@dataclass(frozen=True)
class Tolerance:
    """``d = |A-B| / max(|ref|, abs_floor)``; la celda pasa si ``d <= rel``."""

    rel: float
    abs_floor: float


@dataclass(frozen=True)
class Side:
    """Una variable tal como la nombra una fuente."""

    var: str
    dims: tuple[str, ...]


@dataclass(frozen=True)
class VariableMap:
    name: str
    a: Side
    b: Side
    tol: Tolerance

    def side(self, s: str) -> Side:
        return self.a if s == "a" else self.b


@dataclass(frozen=True)
class CellExclusion:
    """Celdas de la variable ``variable`` (nombre comun) que no se comparan.

    ``where`` va de dim (en el orden comun, el de A) a elementos admitidos;
    ``periods`` es ``None`` si la exclusion vale para todo periodo."""

    variable: str
    where: dict[str, frozenset[str]]
    periods: frozenset[str] | None
    why: str

    def matches(self, dims: tuple[str, ...], period: str, index: Index) -> bool:
        if self.periods is not None and period not in self.periods:
            return False
        return all(
            e in self.where[d]
            for d, e in zip(dims, index, strict=True)
            if d in self.where
        )


@dataclass(frozen=True)
class Mapping:
    variables: tuple[VariableMap, ...]
    thresholds: tuple[float, ...]
    periods: dict[str, dict[str, str]]
    sets: dict[str, dict[str, dict[str, str]]]
    excluded: dict[str, str]  # variables enteras -> why
    cell_exclusions: tuple[CellExclusion, ...] = ()

    def period_name(self, side: str, period: str) -> str:
        return self.periods.get(side, {}).get(period, period)

    def element_name(self, side: str, set_name: str, element: str) -> str:
        return self.sets.get(set_name, {}).get(side, {}).get(element, element)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Mapping:
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise MappingError(f"{path}: el mapeo debe ser un diccionario YAML")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: AbcMapping[str, Any]) -> Mapping:
        unknown = set(data) - {"periods", "sets", "variables", "exclude", "thresholds"}
        if unknown:
            raise MappingError(f"claves desconocidas en el mapeo: {sorted(unknown)}")
        variables = tuple(_variable(v) for v in _required(data, "variables", "mapeo"))
        names = [v.name for v in variables]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise MappingError(f"variables repetidas en el mapeo: {sorted(dup)}")
        thresholds = tuple(
            _positive(t, "thresholds") for t in _required(data, "thresholds", "mapeo")
        )
        if not thresholds:
            raise MappingError("thresholds: hace falta al menos un umbral")
        excluded: dict[str, str] = {}
        cell_exclusions: list[CellExclusion] = []
        by_name = {v.name: v for v in variables}
        for e in data.get("exclude") or []:
            var = str(_required(e, "var", "exclude"))
            why = e.get("why")
            if not isinstance(why, str) or not why.strip():
                raise MappingError(f"exclude {var!r}: falta el motivo ('why')")
            if "where" in e:
                cell_exclusions.append(_cell_exclusion(var, e["where"], why, by_name))
            else:
                excluded[var] = why
        clash = set(excluded) & set(names)
        if clash:
            raise MappingError(
                f"variables a la vez comparadas y excluidas: {sorted(clash)}"
            )
        periods = {
            s: {str(k): str(v) for k, v in m.items()}
            for s, m in _sided(data.get("periods") or {}, "periods").items()
        }
        sets = {
            str(name): {
                s: {str(k): str(v) for k, v in m.items()}
                for s, m in _sided(per_side or {}, f"sets.{name}").items()
            }
            for name, per_side in (data.get("sets") or {}).items()
        }
        return cls(
            variables, thresholds, periods, sets, excluded, tuple(cell_exclusions)
        )


def _required(d: Any, key: str, where: str) -> Any:
    if not isinstance(d, AbcMapping) or key not in d or d[key] is None:
        raise MappingError(f"{where}: falta {key!r}")
    return d[key]


def _positive(x: Any, where: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise MappingError(f"{where}: {x!r} no es un numero") from None
    if not v > 0:
        raise MappingError(f"{where}: {v} debe ser > 0")
    return v


def _sided(d: Any, where: str) -> dict[str, AbcMapping[str, Any]]:
    if not isinstance(d, AbcMapping) or set(d) - set(SIDES):
        raise MappingError(f"{where}: solo admite los lados 'a' y 'b'")
    return {s: d[s] or {} for s in SIDES if s in d}


def _cell_exclusion(
    var: str, where: Any, why: str, by_name: dict[str, VariableMap]
) -> CellExclusion:
    at = f"exclude {var!r}"
    if var not in by_name:
        raise MappingError(
            f"{at}: una exclusion con 'where' debe nombrar una variable del mapeo"
        )
    if not isinstance(where, AbcMapping) or not where:
        raise MappingError(f"{at}: 'where' debe ser un diccionario no vacio")
    dims = by_name[var].a.dims
    if "period" in dims:
        raise MappingError(f"{at}: la dim 'period' choca con la clave de periodo")
    unknown = set(where) - set(dims) - {"period"}
    if unknown:
        raise MappingError(
            f"{at}: 'where' nombra {sorted(unknown)}, que no son dims de la "
            f"variable {list(dims)}"
        )
    sets: dict[str, frozenset[str]] = {}
    for k, elems in where.items():
        if not isinstance(elems, list) or not elems:
            raise MappingError(f"{at}: where.{k} debe ser una lista no vacia")
        sets[str(k)] = frozenset(str(x) for x in elems)
    periods = sets.pop("period", None)
    return CellExclusion(var, sets, periods, why)


def _variable(v: Any) -> VariableMap:
    name = str(_required(v, "name", "variables"))
    where = f"variable {name!r}"
    sides = {}
    for s in SIDES:
        side = _required(v, s, where)
        dims = tuple(str(x) for x in _required(side, "dims", f"{where}.{s}"))
        if len(set(dims)) != len(dims):
            raise MappingError(f"{where}.{s}: dims repetidas {list(dims)}")
        sides[s] = Side(str(_required(side, "var", f"{where}.{s}")), dims)
    if sorted(sides["a"].dims) != sorted(sides["b"].dims):
        raise MappingError(
            f"{where}: las dims de a {list(sides['a'].dims)} y b "
            f"{list(sides['b'].dims)} no son las mismas"
        )
    tol = _required(v, "tol", where)
    rel = _required(tol, "rel", f"{where}.tol")
    floor = _required(tol, "abs_floor", f"{where}.tol")
    return VariableMap(
        name,
        sides["a"],
        sides["b"],
        Tolerance(
            _positive(rel, f"{where}.tol.rel"),
            _positive(floor, f"{where}.tol.abs_floor"),
        ),
    )
