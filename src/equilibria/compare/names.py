"""Diccionario de nombres: que familia de equilibria es cual del modelo de referencia.

ADR-0009. Una fila por familia de ecuaciones o de variables, sin regla por defecto:
un nombre igual en los dos lados tambien lleva su fila. Lo que no esta en el
diccionario no se empareja (``lookup`` da ``None``).

Forma (``compare/maps/<modelo>_names.yaml``)::

    model: gtap7
    equations:
      - {gams: paeq, equilibria: eq_paa}
      - {gams: xaceq, equilibria: eq_xc, gams_fixed: {h: hhd}}
      - {gams: walraseq, equilibria: eq_walras, equilibria_extra: [t]}
      - {gams: pmuveq, why: "..."}                  # solo en GAMS
      - {gams: [axpeq, lambdandeq], why: "..."}    # varias de un lado, mismo motivo
    variables:
      - {gams: xa, equilibria: xaa}

``gams_fixed`` son los indices de GAMS fijados a un elemento (``xaceq(r,i,'hhd',t)``
es ``eq_xc(r,i,t)``); ``equilibria_extra``, los que solo tiene equilibria. Una fila
de un solo lado exige ``why``. Una familia de GAMS puede tener varias filas cuando
equilibria escribe cada rama de una ecuacion condicional como fila propia; una
familia de equilibria, una sola.
"""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from equilibria.compare.errors import MappingError

Side = Literal["gams", "equilibria"]
Kind = Literal["eq", "var"]

_SECTIONS: dict[str, Kind] = {"equations": "eq", "variables": "var"}
_KEYS = {"gams", "equilibria", "gams_fixed", "equilibria_extra", "why"}


@dataclass(frozen=True)
class NameEntry:
    """Una fila: una familia de cada lado (o de uno solo, con su motivo)."""

    kind: Kind
    gams: str | None
    equilibria: str | None
    gams_fixed: tuple[tuple[str, str], ...] = ()
    equilibria_extra: tuple[str, ...] = ()
    why: str = ""


@dataclass(frozen=True)
class NameDictionary:
    model: str
    entries: tuple[NameEntry, ...]

    def lookup_all(self, side: Side, kind: Kind, name: str) -> tuple[NameEntry, ...]:
        """Todas las filas que nombran ``name`` de ese lado."""
        return tuple(
            e for e in self.entries if e.kind == kind and getattr(e, side) == name
        )

    def lookup(self, side: Side, kind: Kind, name: str) -> NameEntry | None:
        """La primera fila que nombra ``name`` de ese lado, o ``None`` si no tiene."""
        found = self.lookup_all(side, kind, name)
        return found[0] if found else None


def load_names(model: str) -> NameDictionary:
    """El diccionario que viene con equilibria (``compare/maps/<model>_names.yaml``)."""
    path = Path(__file__).with_name("maps") / f"{model}_names.yaml"
    if not path.exists():
        raise MappingError(f"no hay diccionario de nombres para {model!r}: {path}")
    return names_from_dict(yaml.safe_load(path.read_text()))


def names_from_dict(data: AbcMapping[str, Any]) -> NameDictionary:
    unknown = set(data) - {"model", *_SECTIONS}
    if unknown:
        raise MappingError(f"claves desconocidas: {sorted(unknown)}")
    entries: list[NameEntry] = []
    for section, kind in _SECTIONS.items():
        for row in data.get(section) or []:
            entries.extend(_rows(kind, row))
    _check_unique(entries)
    return NameDictionary(model=str(data["model"]), entries=tuple(entries))


def _rows(kind: Kind, row: AbcMapping[str, Any]) -> list[NameEntry]:
    unknown = set(row) - _KEYS
    if unknown:
        raise MappingError(f"{dict(row)}: claves desconocidas {sorted(unknown)}")
    gams, eqb, why = row.get("gams"), row.get("equilibria"), row.get("why") or ""
    if gams is None and eqb is None:
        raise MappingError(f"{dict(row)}: fila sin ningun lado")
    if (gams is None or eqb is None) and not why:
        raise MappingError(f"{dict(row)}: fila de un solo lado sin 'why'")
    fixed = tuple((str(k), str(v)) for k, v in (row.get("gams_fixed") or {}).items())
    extra = tuple(str(x) for x in row.get("equilibria_extra") or ())
    if isinstance(gams, list) or isinstance(eqb, list):
        if gams is not None and eqb is not None:  # una lista y un nombre
            raise MappingError(f"{dict(row)}: una lista solo va en filas de un lado")
        if fixed or extra:
            raise MappingError(f"{dict(row)}: una lista no lleva indices")
        if isinstance(gams, list):
            return [NameEntry(kind, str(n), None, why=why) for n in gams]
        assert isinstance(eqb, list)
        return [NameEntry(kind, None, str(n), why=why) for n in eqb]
    return [
        NameEntry(
            kind=kind,
            gams=None if gams is None else str(gams),
            equilibria=None if eqb is None else str(eqb),
            gams_fixed=fixed,
            equilibria_extra=extra,
            why=why,
        )
    ]


def _check_unique(entries: list[NameEntry]) -> None:
    seen: dict[tuple[str, str], NameEntry] = {}
    for e in entries:
        if e.equilibria is not None:
            key = (e.kind, e.equilibria)
            if key in seen:
                raise MappingError(
                    f"{e.equilibria!r} tiene dos filas: {seen[key]}, {e}"
                )
            seen[key] = e
    for e in entries:
        if e.gams is not None and e.equilibria is None:
            others = [o for o in entries if o is not e and o.kind == e.kind]
            if any(o.gams == e.gams for o in others):
                raise MappingError(
                    f"{e.gams!r} esta declarada solo en GAMS y tambien emparejada"
                )
