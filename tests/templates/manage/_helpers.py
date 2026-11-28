"""One MANAGE block group alone, the other groups' variables fixed, and its rows
evaluated at a GAMS point as GAMS scales them.

``ManageTemplate.create_model(groups=...)`` does not exist yet (plan, Task 9), so
the residual tests build a ``Model`` of one group here, with a ``Given`` block
that declares every variable the group uses and no block of it declares.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.base import Block
from equilibria.core.sets import Set as ESet
from equilibria.core.variables import Variable
from equilibria.model import Model
from equilibria.templates.manage.blocks import production_group
from equilibria.templates.manage.calibration import (
    Calibration,
    domains_of,
    gams_key,
)
from equilibria.templates.manage.convert import (
    ConvertRows,
    levels_by_family,
    read_dict,
    read_levels,
)
from equilibria.templates.manage.data import gams_levels
from equilibria.templates.manage.residuals import block_residuals

GROUPS: dict[str, Callable[[Calibration], tuple[Block, ...]]] = {
    "production": production_group,
}

MIN_SCALE = 1e-3  # %minScale% of the s() macro, model.inc:1206

# The floor of every residual test: GAMS's own rows at its 2015 solution are at
# most this (measured 2.79e-9, test_convert.test_solution_floor).
SOLUTION_FLOOR = 3e-9


class Given(Block):
    """Test-only block: declares the variables of other groups, fixed."""

    name: str = "Given"
    given: dict[str, Any] = {}
    domains: dict[str, tuple[str, ...]] = {}

    def setup(self, set_manager, parameters, variables):  # noqa: ANN001, ANN201
        for var, value in self.given.items():
            arr = np.asarray(value, dtype=float)
            variables[var] = Variable(
                name=var,
                value=arr if arr.ndim else np.atleast_1d(arr),
                domains=self.domains[var],
                domain="Reals",
                lower=arr if arr.ndim else float(arr),
                upper=arr if arr.ndim else float(arr),
            )
        return []


def group_model(group: str, cal: Calibration) -> Any:
    """Pyomo model of the blocks of ``group``, every variable they use and no
    block of the group declares fixed at the calibrated base."""
    blocks = GROUPS[group](cal)
    model = Model(name=f"manage_{group}")
    model.add_sets([ESet(name=k, elements=v) for k, v in cal.sets.items()])
    for block in blocks:
        model.add_block(block)
    others = sorted(
        {u for b in blocks for u in b.uses} - set(model.variable_manager.list_vars())
    )
    model.add_block(
        Given(
            given={n: cal.base[n] for n in others},
            domains={n: domains_of(n) for n in others},
        )
    )
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model


def put_levels(pm: Any, cal: Calibration, arrays: Mapping[str, np.ndarray]) -> None:
    """Set every cell of every variable of ``pm`` named in ``arrays``."""
    for name, arr in arrays.items():
        var = pm.find_component(name)
        if var is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if not domains_of(name):
            var.set_value(float(arr.reshape(-1)[0]), skip_validation=True)
            continue
        labels = [cal.sets[d] for d in domains_of(name)]
        for pos in np.ndindex(arr.shape):
            key = tuple(labels[k][i] for k, i in enumerate(pos))
            cell = var[key[0] if len(key) == 1 else key]
            # a fixed cell of another group may move to its 2015 value
            cell.set_value(float(arr[pos]), skip_validation=True)


def row_divisor(
    family: str,
    key: tuple[str, ...],
    base: Mapping[str, Mapping[tuple[str, ...], float]],
) -> float:
    """GAMS's row scale of ``family(key)`` (GAMS key, vintage included): the
    2014 level of the variable its ``s()`` names, at least 1e-3
    (``model.inc:1198-1217``; the production rows at model.inc:1253-1480, 2997)."""
    if family == "e_px":  # %s1% /max(%minScale%, xp.l(a,t-1)*px.l(a,t-1)), :1457
        (a,) = key
        return max(MIN_SCALE, base["xp"].get((a,), 0.0) * base["px"].get((a,), 0.0))
    var, cell = {
        "e_xfac": ("xfac", key),
        "e_pxfac": ("pxfac", key),
        "e_pxfacInd": ("pxfac", key),
        "e_pap": ("pxfac", key),
        "e_xap": ("xa", key),
        "e_lambdaxfac": ("lambdaxfac", key),
        "e_pp": ("pp", key),
        "e_xpNew": ("xp", key[1:2]),
        "e_yieldLnd": ("xp", key),
    }[family]
    return max(MIN_SCALE, abs(base.get(var, {}).get(cell, 0.0)))


def gams_rows(
    pm: Any,
    families: Iterable[str],
    base: Mapping[str, Mapping[tuple[str, ...], float]],
) -> dict[str, dict[tuple[str, ...], float]]:
    """``{family: {GAMS key: residual / GAMS's row scale}}``."""
    out: dict[str, dict[tuple[str, ...], float]] = {}
    for fam, rows in block_residuals(pm, families).items():
        out[fam] = {}
        for key, r in rows.items():
            gkey = gams_key(fam, tuple(str(k) for k in key))
            out[fam][gkey] = r / row_divisor(fam, gkey, base)
    return out


def convert_point(run_dir: Path, bau_path: Path, gms: str) -> dict[str, dict]:
    """The 2015 point of a CONVERT file (``gha2015_scalar.gms``: GAMS's start):
    its free columns, and every other cell at its 2015 GDX level (the cells GAMS
    held fixed, constants in the CONVERT rows)."""
    names = read_dict(run_dir / "gha2015_dict.txt")
    point = {
        k: dict(v)
        for k, v in levels_by_family(names, read_levels(run_dir / gms)).items()
    }
    free = {(f, i[:-1]) for f, i in names.cols}
    for name, cells in gams_levels(bau_path, "2015").items():
        for key, v in cells.items():
            if (name, key) not in free:
                point.setdefault(name, {})[key] = v
    return point


def convert_rows(
    run_dir: Path, gms: str, families: Iterable[str]
) -> dict[tuple[str, tuple[str, ...]], float]:
    """GAMS's own rows of ``families`` at the point of the CONVERT file ``gms``
    (``{(family, GAMS key): residual}``, GAMS's row scaling)."""
    names = read_dict(run_dir / "gha2015_dict.txt")
    rows = ConvertRows.from_gms(run_dir / gms).residuals(read_levels(run_dir / gms))
    keep = set(families)
    return {
        (f, i[:-1]): r for (f, i), r in zip(names.rows, rows, strict=True) if f in keep
    }
