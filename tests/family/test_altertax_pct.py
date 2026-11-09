"""% changes of the ALTERTAX family run against ``solution.sl4`` of GEMPACK (``altax.sl4.gz``).

Gate (user 2026-11-09): for every cell of every GEMPACK variable we map,
``|ours - gem| <= max(TOL_CHANGE * |gem|, TOL_PCT)`` and the same sign where
``|gem| > 10 * TOL_PCT``. The 13 cells below that do not pass are pinned in ``KNOWN_MISS``
with their measured cause (see the comment above it); ``test_known_misses_are_bounded``
keeps each a miss, with the sign and within 1.5x of its measured relative error."""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.sl4 import _shocks, read_sl4_changes
from equilibria.family.altertax.pct import (
    FIXED,
    gempack_changes,
    gtap7_rows,
    pct_changes,
    pct_dims,
    write_pct_har,
)
from equilibria.family.altertax.tolerances import TOL_CHANGE, TOL_PCT
from tests.family._prep import ROOT

SL4 = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax.sl4.gz"


@pytest.fixture(scope="module")
def sl4(tmp_path_factory):
    path = tmp_path_factory.mktemp("sl4") / "altax.sl4"
    with gzip.open(SL4, "rb") as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return read_sl4_changes(path)


@pytest.fixture(scope="module")
def gempack(sl4):
    return gempack_changes(sl4)


@pytest.fixture(scope="module")
def ours(altax_run):
    return pct_changes(altax_run)


def _passes(m: float, g: float) -> bool:
    if abs(m - g) > max(TOL_CHANGE * abs(g), TOL_PCT):
        return False
    return not (abs(g) > 10 * TOL_PCT and m * g <= 0)


def _rel(m: float, g: float) -> float:
    return (m - g) / max(abs(g), TOL_PCT)


# --- the .sl4 reader ---------------------------------------------------------------


def test_sl4_reader_places_every_variable_but_the_two_partly_exogenous(sl4):
    assert set(sl4.unplaced) == {"cgdslack", "del_tbal"}
    assert len(sl4.changes) == 261
    for name, ch in sl4.changes.items():
        assert ch.kind in ("p", "c"), name
        assert ch.values.shape == tuple(len(e) for e in ch.set_elements), name
        assert np.all(np.isfinite(ch.values)), name


def test_sl4_reader_puts_the_shock_in_its_cell(sl4):
    to = sl4.changes["to"]
    assert to.set_elements == [
        ["AGR", "MFG", "SER"],
        ["AGR", "MFG", "SER"],
        ["USA", "ROW"],
    ]
    assert to.values[0, 0, 0] == pytest.approx(0.0012)
    assert np.count_nonzero(to.values) == 1
    assert not np.any(sl4.changes["pop"].values)  # exogenous and not shocked


def test_shock_statements_are_parsed():
    lines = np.array(
        ['Shock to("AGR","AGR","USA") = 0.0012;', "Shock pop = -1.5e-2;", "x"]
    )
    assert _shocks(lines) == {
        "to": {("AGR", "AGR", "USA"): 0.0012},
        "pop": {(): -0.015},
    }


# --- the % changes against GEMPACK --------------------------------------------------


def test_mapped_variables_exist_in_gempack(ours, gempack):
    assert len(ours) >= 30
    assert set(ours) <= set(gempack)
    assert set(gtap7_rows()) | set(FIXED) == set(ours)


def test_every_gempack_cell_has_ours(ours, gempack):
    for name, cells in ours.items():
        assert set(gempack[name]) <= set(cells), name


def test_pct_finite(ours):
    for name, cells in ours.items():
        assert all(np.isfinite(x) for x in cells.values()), name


# Measured misses of the ALTERTAX shock at the 1.01 stand-in for the unit elasticities
# (user 2026-11-09: kept). Values are the relative error (ours - gem) / max(|gem|, TOL_PCT).
# - qes/qfe LAND in MFG and SER of USA (-1.0): the base value is 0 (EVFB(LAND,MFG|SER,USA) = 0
#   in basedata.har), so the cell has no volume: we define its change as 0 while GEMPACK
#   reports the formula value (0.0013, the change of the region's land endowment qe).
# - pfactor (both regions, +-4.52) and pgdp, qgdp: ~1e-5 pp, 1e-2 of the 0.0012 shock, a
#   price-level / numeraire residual (equilibria anchors the Fisher factor index, GEMPACK
#   the value-weighted pfactwld). Sigma sweep of altertax.prm's unit elasticities (1.01,
#   1.001, 1.0001): pfactor USA 2.50e-5, 2.91e-4, 2.95e-3 pp (grows as 2.9e-7 / (sigma - 1),
#   GEMPACK -7.1e-6): it does NOT converge to GEMPACK as sigma -> 1, so it is not the
#   stand-in alone; pgdp USA 1.93e-5, 1.99e-5, 2.00e-5 (GEMPACK 2.31e-5) and qgdp USA
#   4.02e-6, 4.07e-6, 4.07e-6 (GEMPACK 7.2e-7) are stable in sigma: a residual of the model,
#   unexplained, at the 1e-6 pp scale.
# - qpm(AGR,USA), qms(AGR,USA), qxs(AGR,ROW,USA): the small-cell trade residual of ROW->USA
#   imports of AGR (same family as Task 6's DQXS and Task 7's A23 residuals): qms and qxs
#   (identical, one flow) -4.50e-5 vs -4.75e-5 at 1.01, -5.32e-5 at 1.001, -5.39e-5 at 1.0001;
#   qpm 2.79e-5, 2.03e-5, 1.96e-5 vs 1.91e-5 (converges, so the 1.01 stand-in is its cause).
KNOWN_MISS = {
    ("qes", ("LAND", "MFG", "USA")): -1.0,
    ("qes", ("LAND", "SER", "USA")): -1.0,
    ("qfe", ("LAND", "MFG", "USA")): -1.0,
    ("qfe", ("LAND", "SER", "USA")): -1.0,
    ("pfactor", ("USA",)): +4.5248,
    ("pfactor", ("ROW",)): -4.5248,
    ("pgdp", ("USA",)): -0.1665,
    ("pgdp", ("ROW",)): +0.4180,
    ("qgdp", ("USA",)): +4.6243,
    ("qgdp", ("ROW",)): -2.4310,
    ("qpm", ("AGR", "USA")): +0.4643,
    ("qms", ("AGR", "USA")): +0.0530,
    ("qxs", ("AGR", "ROW", "USA")): +0.0530,
}


def _failures(ours, gempack):
    return {
        (n, k): _rel(ours[n][k], g)
        for n, cells in gempack.items()
        if n in ours
        for k, g in cells.items()
        if not _passes(ours[n][k], g)
    }


def test_pct_within_tolerance(ours, gempack):
    compared = sum(len(gempack[n]) for n in ours)
    assert compared >= 200, compared
    bad = {k: v for k, v in _failures(ours, gempack).items() if k not in KNOWN_MISS}
    assert not bad, bad


def test_known_misses_are_bounded(ours, gempack):
    fails = _failures(ours, gempack)
    bad = []
    for key, ref in KNOWN_MISS.items():
        if key not in fails:
            bad.append((key, "now within tolerance, remove it"))
            continue
        err = fails[key]
        if err * ref <= 0 or abs(err) > 1.5 * abs(ref):
            bad.append((key, "error changed sign or grew beyond 1.5x", err, ref))
    assert not bad, bad


def test_fixed_variables_do_not_move_but_the_shock(ours):
    assert not any(ours["pop"].values())
    assert not any(ours["pfactwld"].values())
    to = {k: v for k, v in ours["to"].items() if v != 0.0}
    assert set(to) == {("AGR", "AGR", "USA")}
    assert to[("AGR", "AGR", "USA")] == pytest.approx(0.0012, rel=1e-4)


def test_check_against_check_is_zero(altax_run):
    zero = pct_changes(altax_run, post="check")
    for name, cells in zero.items():
        assert not any(cells.values()), name


def test_pct_dims_are_gempack_sets():
    assert pct_dims("qo") == ("ACTS", "REG")
    assert pct_dims("qxs") == ("COMM", "REG", "REG")
    assert pct_dims("to") == ("COMM", "ACTS", "REG")


def test_pct_har_loads(altax_run, ours, tmp_path):
    path = tmp_path / "solution_pct.har"
    write_pct_har(altax_run, path)
    got = read_har(str(path))
    assert len(got) == len(ours)
    assert all(len(h) <= 4 for h in got)
    by_name = {h.coeff_name: h for h in got.values()}
    assert set(by_name) == set(ours)
    assert by_name["qo"].long_name == "qo"
    assert by_name["qo"].set_names == ["ACTS", "REG"]
    qo = np.asarray(by_name["qo"].array)
    assert qo.shape == (3, 2)
    assert qo[0, 0] == pytest.approx(ours["qo"][("AGR", "USA")], rel=1e-6)
    assert np.asarray(by_name["pfactwld"].array).size == 1
