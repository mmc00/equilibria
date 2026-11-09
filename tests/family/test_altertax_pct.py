"""% changes of the ALTERTAX family run against ``solution.sl4`` of GEMPACK (``altax.sl4.gz``).

Gate (user 2026-11-09): for every cell of every GEMPACK variable we map,
``|ours - gem| <= max(TOL_CHANGE * |gem|, TOL_PCT)`` and the same sign where
``|gem| > 10 * TOL_PCT``. Cells whose base flow is 0 are excluded by rule (see
``ZERO_BASE``). Of the cells compared, only 37 of 221 (one in six) are governed by the
relative bound (``|gem| >= TOL_PCT / TOL_CHANGE = 5e-5`` pp); the rest sit below it and pass
or fail on the 1e-6 pp absolute floor: the ALTERTAX shock is 0.0012 %, so most changes are
1e-5 pp or less. ``test_gate_share`` pins the split. The 9 cells that do not pass are
pinned in ``KNOWN_MISS`` with their measured cause; ``test_known_misses_are_bounded`` keeps
each a miss, with the sign and within 1.5x of its measured relative error."""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.sl4 import read_sl4_changes, read_sl4_levels
from equilibria.family.altertax.pct import (
    FIXED,
    gempack_changes,
    gtap7_rows,
    pct_changes,
    pct_dims,
    write_pct_har,
    zero_base_cells,
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


BURFISHER = ROOT / "tests/fixtures/gtap7_gempack/burfisher"
GRAGG = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"


def _read(directory, name, tmp_path):
    path = tmp_path / f"{name}.sl4"
    with gzip.open(directory / f"{name}.sl4.gz", "rb") as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return path


def _nonzero(ch):
    return {
        tuple(ch.set_elements[d][i] for d, i in enumerate(ix)): float(ch.values[ix])
        for ix in np.ndindex(ch.values.shape)
        if ch.values[ix] != 0.0
    }


def test_shocks_come_from_the_file_arrays(tmp_path):
    """Shock cells and values from SHCK/PSHK/SHOC/SHCL, checked against the command file
    of the fixtures: a ``uniform`` shock over several cells (tbl64: ``afeall`` of LABOR in
    every activity), a scalar one (gtapuv7), and lower-case labels in the command file
    (tbl66, tbl77: ``qe("labor","USA")``) whose cells are the upper-case ones of the sets."""
    r = read_sl4_changes(_read(GRAGG, "tbl64", tmp_path))
    assert _nonzero(r.changes["afeall"]) == {
        ("LABOR", a, "USA"): 10.0 for a in ("AGR", "MFG", "SER")
    }
    r = read_sl4_changes(_read(GRAGG, "gtapuv7", tmp_path))
    assert _nonzero(r.changes["pfactwld"]) == {(): 10.0}
    r = read_sl4_changes(_read(GRAGG, "tbl66", tmp_path))
    assert _nonzero(r.changes["qe"]) == {("LABOR", "USA"): 2.0}
    r = read_sl4_changes(_read(GRAGG, "tbl77", tmp_path))
    assert _nonzero(r.changes["qe"]) == {("LAND", "USA"): 10.0}
    r = read_sl4_changes(
        _read(GRAGG, "me5", tmp_path)
    )  # several statements, one variable
    assert _nonzero(r.changes["tfe"])[("LAND", "AGR", "USA")] == pytest.approx(4.2896)
    assert _nonzero(r.changes["tfe"])[("CAPITAL", "AGR", "USA")] == pytest.approx(3.27)


def test_every_fixture_reads_or_says_why(tmp_path):
    """No fixture of the Gragg folder fails to read, and a partly endogenous variable is
    reported in ``unplaced``, never read as zeros."""
    for gz in sorted(GRAGG.glob("*.sl4.gz")):
        name = gz.name[: -len(".sl4.gz")]
        r = read_sl4_changes(_read(GRAGG, name, tmp_path))
        assert len(r.changes) + len(r.unplaced) >= 261, name
        assert not set(r.changes) & set(r.unplaced), name


# read_sl4_levels is unchanged by the shared layout helper: digest of every variable of two
# fixtures (names, labels, pre and post bytes) taken from the version before this change.
LEVELS_DIGEST = {
    "gtapuv7": "979fd14af41b1274",
    "me3a": "7718a9cf21952b15",
}


@pytest.mark.parametrize("name", sorted(LEVELS_DIGEST))
def test_read_sl4_levels_is_unchanged(name, tmp_path):
    import hashlib

    levels = read_sl4_levels(_read(BURFISHER, name, tmp_path))
    h = hashlib.sha256()
    for k in sorted(levels):
        h.update(k.encode())
        v = levels[k]
        if v is None:
            h.update(b"none")
            continue
        h.update(repr(v.set_elements).encode())
        h.update(np.ascontiguousarray(v.pre).tobytes())
        h.update(np.ascontiguousarray(v.post).tobytes())
    assert h.hexdigest()[:16] == LEVELS_DIGEST[name]


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


# Cells whose base flow is 0 (EVFB(LAND, MFG|SER, USA|ROW) = 0 in basedata.har: 8 cells, qes
# and qfe of LAND in MFG and SER of both regions) are excluded by rule. A % change on a zero
# flow is undefined; we report 0, GEMPACK reports the value of its price formula: E_qes2 holds
# for every cell regardless of the weight, and the 0.0013 comes from -ETRAE * (pes - pe) on a
# zero flow (qe itself is 0 in the .sl4). GEMPACK's own volume change DQFE = qfe / 100 * BEVFB
# zeroes it. Four of the 8 would pass anyway through the absolute floor (ROW).
ZERO_BASE = {
    ("qes", ("LAND", "MFG", "USA")),
    ("qes", ("LAND", "SER", "USA")),
    ("qfe", ("LAND", "MFG", "USA")),
    ("qfe", ("LAND", "SER", "USA")),
    ("qes", ("LAND", "MFG", "ROW")),
    ("qes", ("LAND", "SER", "ROW")),
    ("qfe", ("LAND", "MFG", "ROW")),
    ("qfe", ("LAND", "SER", "ROW")),
}

# Measured misses of the ALTERTAX shock at the 1.01 stand-in for the unit elasticities
# (user 2026-11-09: kept). Values are the relative error (ours - gem) / max(|gem|, TOL_PCT).
# - pfactor (USA +4.52, ROW -4.52): ~3e-5 pp. Sigma sweep of altertax.prm's unit elasticities
#   (1.01, 1.001, 1.0001), pfactor USA 2.50e-5, 2.91e-4, 2.95e-3 pp and ROW -8.8e-6, -1.0e-4,
#   -1.0e-3. It fits a + b / (sigma - 1) with b = +2.95e-7 (USA), -1.04e-7 (ROW) and
#   a = -4.2e-6 (USA), +1.5e-6 (ROW), against GEMPACK -7.1e-6 and +2.5e-6: a deterministic
#   term of our near-unit CES formulation, independent of the solver tolerance (TR_FTOL 1e-13
#   and RELTOL 1e-12 leave it unchanged). It is NOT the Fisher/Divisia numeraire difference
#   (about 1e-12 pp here). OPEN LEAD: the origin of the b / (sigma - 1) term is not found.
# - pgdp (USA -0.17, ROW +0.42), qgdp (USA +4.62, ROW -2.43): ~1e-6 pp. Flat in sigma (pgdp
#   USA 1.93e-5, 1.99e-5, 2.00e-5 against GEMPACK 2.31e-5; qgdp USA 4.02e-6, 4.07e-6, 4.07e-6
#   against 7.2e-7: b is 5e-11 and 3e-12), so the stand-in is not their cause: a residual of
#   the model at the 1e-6 pp scale, unexplained.
# - qpm(AGR,USA), qms(AGR,USA), qxs(AGR,ROW,USA): the small-cell trade residual of ROW->USA
#   imports of AGR (same family as Task 6's DQXS and Task 7's A23 residuals): qms and qxs
#   (identical, one flow) -4.50e-5 vs -4.75e-5 at 1.01, -5.32e-5 at 1.001, -5.39e-5 at 1.0001
#   (does not converge); qpm 2.79e-5, 2.03e-5, 1.96e-5 vs 1.91e-5 (converges: the stand-in).
KNOWN_MISS = {
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


def _compared(ours, gempack):
    """``{(var, cell): (ours, gem)}`` of the mapped variables, zero-base cells excluded."""
    return {
        (n, k): (ours[n][k], g)
        for n, cells in gempack.items()
        if n in ours
        for k, g in cells.items()
        if (n, k) not in ZERO_BASE
    }


def _failures(ours, gempack):
    return {
        key: _rel(m, g)
        for key, (m, g) in _compared(ours, gempack).items()
        if not _passes(m, g)
    }


def test_zero_base_cells_are_the_eight_excluded(altax_run):
    zero = zero_base_cells(altax_run)
    assert {(n, k) for n, cells in zero.items() for k in cells} == ZERO_BASE


def test_gate_share(ours, gempack):
    """How many compared cells are governed by the relative bound and how many by the
    absolute floor (cell change below TOL_PCT / TOL_CHANGE)."""
    cells = _compared(ours, gempack)
    rel = sum(TOL_CHANGE * abs(g) >= TOL_PCT for _, g in cells.values())
    assert len(cells) == 221, len(cells)
    assert rel == 37, (rel, len(cells) - rel)


def test_pct_within_tolerance(ours, gempack):
    assert len(_compared(ours, gempack)) >= 200
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
