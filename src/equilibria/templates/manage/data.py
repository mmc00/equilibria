"""MANAGE's Ghana calibration, read from the recorded GAMS run, in solve units.

The calibration is GAMS's own (design, "Decision: read GAMS's calibrated values
first", and decision 6): ``GHA_BaU.gdx`` of the reference run holds every symbol
after ``inc/cal/rescale.inc``. ``load_bau`` returns its sets, its parameters, the
2014 level of every variable (the calibrated base; GAMS never solves 2014) and
every fixed cell of both years (``lo == up``: the exogenous values and the
calibrated "parameters" that MANAGE declares as fixed variables). It never
returns a 2015 level of a free cell: those are what the 2015 solve must find,
and only ``gams_levels`` (for the parity tests) reads them.

Names keep the GDX's case, which is the case of the GAMS declaration
(``fppar``, ``xFac`` is ``xfac``), the same as ``gha2015_dict.txt``.

All values are in GAMS's solve units (GDP = 1e5): ``rescale.to_solve_units``
undoes rescale.inc record by record.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from equilibria.babel.gdx.reader import (
    get_parameters,
    get_sets,
    get_variables,
    read_gdx,
    read_parameter_values,
    read_set_elements,
    read_variable_values,
)
from equilibria.templates.manage.rescale import to_solve_units

YEARS: tuple[str, ...] = ("2014", "2015")
BASE_YEAR = "2014"
_SCALE_NAMES = ("reScale", "reEscale", "usd", "inScale")

Cells = dict[tuple[str, ...], float]


@dataclass(frozen=True)
class ManageData:
    """What the calibration may read of the GAMS run (solve units)."""

    sets: dict[str, tuple[tuple[str, ...], ...]]
    params: dict[str, Cells]
    base: dict[str, Cells]  # 2014 level, the year dropped from the key
    fixed: dict[str, Cells]  # fixed cells of both years, the year kept
    scale: dict[str, float]


class _Gdx:
    """One GDX, its sets by lower-case name, and the unit conversion."""

    def __init__(self, path: Path) -> None:
        self.data = read_gdx(path)
        self.sets = {
            s["name"]: tuple(read_set_elements(self.data, s["name"]))
            for s in get_sets(self.data)
            if s["dimension"]
        }
        self._members = {k.lower(): frozenset(v) for k, v in self.sets.items()}
        scalars = {p["name"].lower(): p["name"] for p in get_parameters(self.data)}
        self.scale = {
            n: read_parameter_values(self.data, scalars[n.lower()])[()]
            for n in _SCALE_NAMES
        }

    def member(self, set_name: str, part: tuple[str, ...]) -> bool:
        members = self._members.get(set_name.lower())
        if members is None:
            raise KeyError(f"rescale rule names set {set_name!r}, not in the GDX")
        return part in members

    def solve(self, name: str, key: tuple[str, ...], value: float) -> float:
        return to_solve_units(name, key, value, self.scale, self.member)


@cache
def _open(path: Path) -> _Gdx:
    return _Gdx(path)


def _yearly(key: tuple[str, ...]) -> bool:
    return bool(key) and key[-1] in YEARS


def load_bau(path: Path) -> ManageData:
    """The calibration of the GAMS run in ``path`` (a ``GHA_BaU.gdx``)."""
    gdx = _open(Path(path))
    params = {
        p["name"]: {
            k: gdx.solve(p["name"], k, v)
            for k, v in read_parameter_values(gdx.data, p["name"]).items()
        }
        for p in get_parameters(gdx.data)
    }
    base: dict[str, Cells] = {}
    fixed: dict[str, Cells] = {}
    for v in get_variables(gdx.data):
        name = v["name"]
        for key, rec in read_variable_values(gdx.data, name).items():
            level = gdx.solve(name, key, rec["level"])
            if rec["lower"] == rec["upper"]:
                fixed.setdefault(name, {})[key] = level
            if _yearly(key) and key[-1] == BASE_YEAR:
                base.setdefault(name, {})[key[:-1]] = level
    return ManageData(dict(gdx.sets), params, base, fixed, dict(gdx.scale))


def gams_levels(path: Path, year: str) -> dict[str, Cells]:
    """Every variable's level in ``year`` (solve units, the year dropped), free
    cells included. For the parity tests only; the calibration never calls it."""
    gdx = _open(Path(path))
    out: dict[str, Cells] = {}
    for v in get_variables(gdx.data):
        name = v["name"]
        for key, rec in read_variable_values(gdx.data, name).items():
            if _yearly(key) and key[-1] == year:
                out.setdefault(name, {})[key[:-1]] = gdx.solve(name, key, rec["level"])
    return out
