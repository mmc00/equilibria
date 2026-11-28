"""MANAGE's calibration as the blocks read it: dense arrays over GAMS's sets.

The values are GAMS's own (design, "Decision: read GAMS's calibrated values
first"): ``calibrate`` reshapes what ``load_bau`` read from ``GHA_BaU.gdx``
(solve units) into the arrays the blocks declare, in the order of the GDX's set
elements. It computes nothing that GAMS computed.

- ``values``: the parameters, and the GAMS sets a block reads as 0/1 arrays to
  select its rows (its ``$`` conditions).
- ``base``: the 2014 level of every variable.
- ``t0``: the 2014 levels an equation reads as numbers (``x.l(t0)``), as the
  parameters ``<var>_t0`` (design, "Periods and recipe").
- ``lag``: the levels of the year before (``x.l(t-1)``), as fixed variables
  ``<var>_lag``. No production row reads one.

One vintage: GAMS indexes the production tree by ``v`` (``old = v = {""}``, no
``new``, ``ifVint = 0``). Python drops that index everywhere (``domains_of``) and
``gams_key`` puts it back to compare a row with GAMS's. Today this holds the
production group (design, group A); later groups add their names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from equilibria.templates.manage.data import BASE_YEAR, Cells, ManageData

SOLVE_YEAR = "2015"

# The model's sets, by the name the blocks use (model.inc's spelling); the GDX
# keeps the declaration's case (fppar), so they are matched case-insensitively.
SETS: tuple[str, ...] = ("fpNest", "fpPar", "a", "i", "aa", "em", "aagr")

# GAMS's domain of each variable the groups declare or read ("v": the vintage,
# dropped; model.inc:272-624). pxfacInd is declared over fpNest in GAMS and
# only exists for the additive parents: Python keeps it over fpPar.
VARIABLES: dict[str, tuple[str, ...]] = {
    "xfac": ("fpNest", "a", "v"),
    "pxfac": ("fpNest", "a", "v"),
    "pxfacInd": ("fpPar", "a", "v"),
    "lambdaxfac": ("fpNest", "a", "v"),
    "xa": ("i", "aa"),
    "paf": ("i", "aa"),
    "px": ("a",),
    "pp": ("a",),
    "xp": ("a",),
    "xs": ("i",),
    "yieldLnd": ("a",),
    "tPrd": ("a",),
    "tpsb": ("a",),
    "tPhn": ("a",),
    "tEmi": ("em", "aa"),
    "emical": (),
    "chiEmiXP": ("em", "a"),
    "chixFacF": ("fpNest",),
    "chixFacA": ("a",),
    "chixFacV": ("v",),
    "chixFacFA": ("fpNest", "a"),
    "chixFacL": (),
    "chixFacK": (),
    "chixFacKL": (),
    "chixFacFP": (),
    "chixFacNRG": (),
    "chixFacAgr": ("aagr", "fpNest"),
}

# GAMS's domain of each parameter ("t": the year, read at the base year; every
# one of these is the same in 2014 and 2015 in GHA).
PARAMETERS: dict[str, tuple[str, ...]] = {
    "p_alphaXfac": ("fpNest", "fpPar", "a", "v"),
    "p_sigmaXFac": ("fpPar", "a", "v"),
    "p_chiPA": ("i",),
    "p_as": ("a", "i", "t"),
    "p_rhoEmiXP": ("em", "a"),
    "p_phiEmiXP": ("em", "a", "t"),
}

# GAMS sets a block reads as 0/1 arrays: name in the GDX -> the domain here.
INDICATORS: dict[str, tuple[str, ...]] = {
    "flagXfac": ("fpNest", "a", "v"),
    "isAdditiveCes": ("fpPar",),
    "fpTop": ("fpPar",),
    "aact": ("a",),
    "fpNestI": ("fpNest",),
    "l": ("fpNest",),
    "cap": ("fpNest",),
    "fp": ("fpNest",),
    "nrgFac": ("fpNest",),
    "actHa": ("fpNest",),
    "mapAagr": ("aagr", "a"),
    "emiFlag": ("em", "aa"),
}

# Equation families GAMS indexes by the vintage, as the last index
# (gha2015_dict.txt: e_xfac(c-pdr,a-pdr,"",2015)).
WITH_VINTAGE = frozenset(
    {"e_xfac", "e_pxfac", "e_pxfacInd", "e_pap", "e_lambdaxfac", "e_xpNew"}
)


@dataclass(frozen=True)
class Calibration:
    sets: dict[str, tuple[str, ...]]
    values: dict[str, Any]
    base: dict[str, Any]
    lag: dict[str, Any] = field(default_factory=dict)
    t0: dict[str, Any] = field(default_factory=dict)

    def block_inputs(self) -> dict[str, dict[str, Any]]:
        """``values`` and ``start`` of a level block: the parameters with the
        ``<var>_t0``, the base levels with the ``<var>_lag``."""
        return {
            "values": {**self.values, **self.t0},
            "start": {**self.base, **self.lag},
        }


def domains_of(name: str) -> tuple[str, ...]:
    """The domains of variable ``name`` in Python (GAMS's, the vintage dropped)."""
    return tuple(d for d in VARIABLES[name] if d != "v")


def gams_key(family: str, key: tuple[str, ...]) -> tuple[str, ...]:
    """The GAMS index (year dropped) of the Python row ``family(key)``."""
    return (*key, "") if family in WITH_VINTAGE else key


def _array(
    cells: Mapping[tuple[str, ...], float],
    domains: tuple[str, ...],
    sets: Mapping[str, tuple[str, ...]],
    *,
    what: str,
    year: str | None = None,
    strict: bool = True,
) -> np.ndarray:
    """Dense array of ``cells`` (GAMS keys) over ``domains`` less ``v`` and
    ``t``; ``t`` is read at ``year``. A key outside the sets raises when
    ``strict``, and is left out otherwise."""
    kept = [d for d in domains if d not in ("v", "t")]
    pos = {d: {e: k for k, e in enumerate(sets[d])} for d in kept}
    out = np.zeros(tuple(len(sets[d]) for d in kept))
    for key, value in cells.items():
        if len(key) != len(domains):
            raise ValueError(f"{what}{key}: expected the domains {domains}")
        idx: list[int] = []
        for d, e in zip(domains, key, strict=True):
            if d == "v":
                if e != "":
                    raise NotImplementedError(f"{what}{key}: one vintage only")
            elif d == "t":
                if e != year:
                    break
            elif e in pos[d]:
                idx.append(pos[d][e])
            elif strict:
                raise ValueError(f"{what}{key}: {e!r} is not in {d}")
            else:
                break
        else:
            out[tuple(idx)] = value
    return out


def _lower(table: Mapping[str, Any]) -> dict[str, Any]:
    """``table`` keyed by the lower-case name: GAMS names are case-insensitive,
    and the GDX keeps the declaration's case."""
    return {k.lower(): v for k, v in table.items()}


def to_arrays(
    levels: Mapping[str, Cells], sets: Mapping[str, tuple[str, ...]]
) -> dict[str, np.ndarray]:
    """The variables of ``VARIABLES`` in ``levels`` (``{name: {GAMS key without
    the year: level}}``, as ``load_bau(...).base`` or ``gams_levels``), dense."""
    by_name = _lower(levels)
    return {
        name: _array(by_name.get(name.lower(), {}), dom, sets, what=name)
        for name, dom in VARIABLES.items()
    }


def _check_switches(
    params: Mapping[str, Cells], sets: Mapping[str, tuple[tuple[str, ...], ...]]
) -> None:
    """What the production rows leave out must be off in the data."""
    for flag in ("ifVint", "ifEnTech", "ifDmg", "ifAirPollDmg"):
        value = params.get(flag.lower(), {}).get((), 0.0)
        if value != 0:
            raise NotImplementedError(f"{flag} = {value}: only {flag} = 0 is written")
    if sets["v"] != (("",),) or sets["old"] != (("",),) or sets.get("new"):
        raise NotImplementedError(f"vintages {sets['v']}: one vintage only")
    for name in ("fpParTc",):  # twist (model.inc:1263-1270)
        if sets.get(name.lower()):
            raise NotImplementedError(f"{name} is not empty: the twist is not written")
    if any(params.get("p_markupnoncomp", {}).values()):
        raise NotImplementedError("p_markUpNonComp is not written (model.inc:1300)")


def _fixed(data: ManageData, name: str, key: tuple[str, ...]) -> bool:
    return (*key, SOLVE_YEAR) in data.fixed.get(name, {})


def _lambda_free(
    data: ManageData,
    sets: Mapping[str, tuple[str, ...]],
    gdx_sets: Mapping[str, tuple[tuple[str, ...], ...]],
) -> np.ndarray:
    """1 where ``lambdaxfac`` is free in the solve year: the rows of
    ``e_lambdaXfac`` (``lambdaXfac.range ne 0``, model.inc:1428). iterloop.inc:88-115
    fixes a cell when every shifter that enters it is fixed; dmgAir and dmgPrd do
    not enter (ifAirPollDmg = ifDmg = 0) and no cell has ``.scale = 999``."""
    members = {
        k: {e for (e,) in gdx_sets[k]}
        for k in ("actfpnest", "l", "cap", "fp", "nrgfac")
    }
    agr: dict[str, list[str]] = {}
    for g, a in gdx_sets["mapaagr"]:
        agr.setdefault(a, []).append(g)
    nodes, acts = sets["fpNest"], sets["a"]
    out = np.zeros((len(nodes), len(acts)))
    for n, a, _v in gdx_sets["flagxfac"]:
        if n not in members["actfpnest"]:
            continue
        lab, cap = n in members["l"], n in members["cap"]
        fixed = (
            _fixed(data, "chixFacF", (n,))
            and _fixed(data, "chixFacA", (a,))
            and _fixed(data, "chixFacV", ("",))
            and _fixed(data, "chixFacFA", (n, a))
            and (not lab or _fixed(data, "chixFacL", ()))
            and (not cap or _fixed(data, "chixFacK", ()))
            and (not (cap or lab) or _fixed(data, "chixFacKL", ()))
            and (n not in members["fp"] or _fixed(data, "chixFacFP", ()))
            and (n not in members["nrgfac"] or _fixed(data, "chixFacNRG", ()))
            and all(_fixed(data, "chixFacAgr", (g, n)) for g in agr.get(a, ()))
        )
        if not fixed:
            out[nodes.index(n), acts.index(a)] = 1.0
    return out


def calibrate(data: ManageData) -> Calibration:
    """The arrays of the production group, from GAMS's calibration."""
    gdx_sets, params = _lower(data.sets), _lower(data.params)
    _check_switches(params, gdx_sets)
    sets = {s: tuple(e for (e,) in gdx_sets[s.lower()]) for s in SETS}

    values: dict[str, Any] = {}
    for name, dom in PARAMETERS.items():
        values[name] = _array(
            params.get(name.lower(), {}),
            dom,
            sets,
            what=name,
            year=BASE_YEAR,
            # p_sigmaXFac is also given for nodes that are no parent; no row reads it
            strict=name != "p_sigmaXFac",
        )
    for name, dom in INDICATORS.items():
        cells = dict.fromkeys(gdx_sets[name.lower()], 1.0)
        values[name] = _array(cells, dom, sets, what=name)
    # sam(i,a), the condition of e_xap (model.inc:1379)
    sam_ia = {
        key: v
        for key, v in params["sam"].items()
        if key[0] in sets["i"] and key[1] in sets["a"]
    }
    values["sam"] = _array(sam_ia, ("i", "a"), sets, what="sam")
    values["lambdaxfac_free"] = _lambda_free(data, sets, gdx_sets)

    base = to_arrays(data.base, sets)
    t0 = {"xfac_t0": base["xfac"], "xp_t0": base["xp"]}
    return Calibration(sets=sets, values=values, base=base, t0=t0)
