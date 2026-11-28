"""MANAGE's production group (design, group A: 10 families, 5,314 rows), checked
at the GAMS points with no solve.

The tree (``CESTree``) and ``ManageProduction`` are built alone, the other
groups' variables fixed, and their rows are evaluated at:

- the 2015 GAMS solution (``gams_levels(bau, "2015")``, solve units), divided as
  GAMS divides them (``s()``, the 2014 level of the row's variable): at most the
  floor of GAMS's own rows at its solution (``test_convert.test_solution_floor``);
- the 2014 base, where no production row reads a lag (``e_xpNew`` reads the
  2014 level ``xfac.l(t0)``, a parameter): the calibration identity.

The row instance of each family must equal GAMS's (``gha2015_dict.txt``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.templates.manage._helpers import (
    convert_point,
    convert_rows,
    gams_rows,
    group_model,
    put_levels,
)
from tests.templates.manage.conftest import SOLUTION_FLOOR

from equilibria.templates.manage.blocks import PRODUCTION_FAMILIES
from equilibria.templates.manage.calibration import calibrate, to_arrays
from equilibria.templates.manage.convert import instance, read_dict
from equilibria.templates.manage.data import gams_levels, load_bau

COUNTS = {
    "e_xfac": 1995, "e_pxfac": 570, "e_pxfacInd": 65, "e_pap": 1134,
    "e_xap": 1134, "e_lambdaxfac": 255, "e_px": 51, "e_pp": 51, "e_xpNew": 51,
    "e_yieldLnd": 8,
}  # fmt: skip


@pytest.fixture(scope="module")
def data(bau_path: Path):  # noqa: ANN201
    return load_bau(bau_path)


@pytest.fixture(scope="module")
def cal(data):  # noqa: ANN001, ANN201
    return calibrate(data)


@pytest.fixture(scope="module")
def pm(cal):  # noqa: ANN001, ANN201
    return group_model("production", cal)


def test_instance_production(pm, cal, data, dict_path: Path) -> None:  # noqa: ANN001
    assert set(PRODUCTION_FAMILIES) == set(COUNTS)
    gams = instance(read_dict(dict_path))
    ours = gams_rows(pm, PRODUCTION_FAMILIES, data.base)
    assert {f: len(ours[f]) for f in COUNTS} == COUNTS
    for fam in PRODUCTION_FAMILIES:
        assert set(ours[fam]) == gams[fam], fam


def _worst(rows: dict[str, dict]) -> dict[str, float]:
    return {f: max(abs(r) for r in cells.values()) for f, cells in rows.items()}


def test_production_residuals_2015(pm, cal, data, bau_path: Path) -> None:  # noqa: ANN001
    put_levels(pm, cal, to_arrays(gams_levels(bau_path, "2015"), cal.sets))
    worst = _worst(gams_rows(pm, PRODUCTION_FAMILIES, data.base))
    assert all(w <= SOLUTION_FLOOR for w in worst.values()), worst


def test_production_residuals_2014(pm, cal, data) -> None:  # noqa: ANN001
    put_levels(pm, cal, cal.base)
    worst = _worst(gams_rows(pm, PRODUCTION_FAMILIES, data.base))
    assert all(w <= SOLUTION_FLOOR for w in worst.values()), worst


def test_production_rows_are_gams_rows_at_the_start(
    pm,
    cal,
    data,
    run_dir: Path,
    bau_path: Path,  # noqa: ANN001
) -> None:
    """Off the solution too: at GAMS's 2015 start point (resLastYear.inc; output
    and Armington demand grown by 2.18 %, e_yieldLnd off by 1.9) each row equals
    GAMS's own row (``gha2015_scalar.gms``), within the floor."""
    gms = "gha2015_scalar.gms"
    put_levels(pm, cal, to_arrays(convert_point(run_dir, bau_path, gms), cal.sets))
    ours = gams_rows(pm, PRODUCTION_FAMILIES, data.base)
    theirs = convert_rows(run_dir, gms, PRODUCTION_FAMILIES)
    assert max(abs(r) for r in theirs.values()) > 1.0  # the point is off the solution
    gap = {
        f: max(abs(r - theirs[(f, k)]) for k, r in rows.items())
        for f, rows in ours.items()
    }
    assert all(g <= SOLUTION_FLOOR for g in gap.values()), gap
