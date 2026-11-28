"""The CONVERT files of the 2015 solve: names, levels and GAMS's own rows.

The row evaluator translates each row of ``gha2015_scalar.gms`` to Python. It is
checked against GAMS itself: at the start point its largest residuals are the
ones the recorded run reports (source.yaml: 2.33 in e_paf(c-coa,a-i_s), 0.68 in
e_nshrGDP(h-hhold)), and at the solution every row is near zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.templates.manage.conftest import SOLUTION_FLOOR

from equilibria.templates.manage.convert import (
    ConvertRows,
    instance,
    read_dict,
    read_levels,
)


@pytest.fixture(scope="module")
def names(dict_path: Path):  # noqa: ANN201
    return read_dict(dict_path)


@pytest.fixture(scope="module")
def rows(run_dir: Path) -> ConvertRows:
    return ConvertRows.from_gms(run_dir / "gha2015_scalar.gms")


def test_dictionary_counts(names) -> None:  # noqa: ANN001
    assert len(names.rows) == len(names.cols) == 9452
    assert len({f for f, _ in names.rows}) == 167
    assert len({f for f, _ in names.cols}) == 136
    assert names.rows[0] == ("e_px", ("a-pdr", "2015"))
    assert names.cols[0] == ("xfac", ("c-pdr", "a-pdr", "", "2015"))
    assert names.rows[-1] == ("e_dummy", ())


def test_instance_counts(names) -> None:  # noqa: ANN001
    inst = instance(names)
    assert sum(len(v) for v in inst.values()) == 9452
    assert len(inst["e_xfac"]) == 1995 and len(inst["e_paf"]) == 1199


def test_levels_parse(run_dir: Path) -> None:
    start = read_levels(run_dir / "gha2015_scalar.gms")
    assert start[1] == pytest.approx(23.347685412907886, rel=1e-15)
    assert 9387 not in start  # a zero level is not written


def _by_row(names, res) -> dict[str, float]:  # noqa: ANN001
    return {
        f"{f}({','.join(i)})": float(r)
        for (f, i), r in zip(names.rows, res, strict=True)
    }


def test_start_point_reproduces_gams(rows: ConvertRows, names, run_dir: Path) -> None:  # noqa: ANN001
    res = _by_row(names, rows.residuals(read_levels(run_dir / "gha2015_scalar.gms")))
    assert max(abs(r) for r in res.values()) == pytest.approx(2.3294, abs=5e-5)
    # c-coa is sold to a-nmm and a-i_s at the same tPaf: the two rows tie
    assert res["e_paf(c-coa,a-i_s,2015)"] == pytest.approx(-2.3294, abs=5e-5)
    assert abs(res["e_nshrGDP(h-hhold,2015)"]) == pytest.approx(0.68, abs=0.005)


def test_solution_floor(rows: ConvertRows, run_dir: Path) -> None:
    """The floor for every residual test: GAMS's own solution, in its own rows.
    Measured 2026-11-27: 2.79e-9, in e_dCarbon(f-AEZ4,pastureLand) (a land-cover
    row whose terms are about 1e6); no row is nan."""
    res = rows.residuals(read_levels(run_dir / "gha2015_solution.gms"))
    assert max(abs(r) for r in res) <= SOLUTION_FLOOR


# Families whose GAMS row does not hold at the 2014 levels, measured 2026-11-27.
# Each reads a 2015 driver: the real GDP target (+2.18 %) and what is defined
# from real GDP; population (cohorts, births, row transfers, the labor rows of
# e_xfmSup); a lagged stock (kstock, debt); or land cover and the energy and
# emission intensities, which follow GDP and population. e_FundF (1.0 at 2014)
# is a smooth-max row on the debt-to-GDP ratio; it is examined in Task 7.
NOT_AT_2014 = {
    "e_FundF", "e_debtInterest", "e_kstock", "e_DebtStk", "e_popAge", "e_xfmSup",
    "e_Birth", "e_rowtransfers", "e_rgdpmp", "e_rgdppc", "e_rshrGDPGov",
    "e_pgdpmp", "e_rshrGDP", "e_GdpCompShr", "e_GdpCompShrM", "e_builtUpArea",
    "e_GdpCompShrX", "e_nrgInt", "e_elyInt", "e_emiInt", "e_debtGDP",
}  # fmt: skip


def test_2014_benchmark(rows: ConvertRows, names, bau_path: Path) -> None:  # noqa: ANN001
    """GAMS's 2015 rows at the 2014 levels: 146 of the 167 families hold to
    1e-8 (9,034 rows); the 21 others are the ones that read a 2015 driver."""
    from equilibria.templates.manage.data import gams_levels

    base = gams_levels(bau_path, "2014")
    x = {
        j: base.get(fam, {}).get(idx[:-1] if idx else idx, 0.0)
        for j, (fam, idx) in enumerate(names.cols, start=1)
    }
    worst: dict[str, float] = {}
    for (fam, _), r in zip(names.rows, rows.residuals(x), strict=True):
        worst[fam] = max(worst.get(fam, 0.0), abs(r))
    assert {f for f, r in worst.items() if not r <= 1e-8} == NOT_AT_2014
