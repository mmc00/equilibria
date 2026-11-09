"""``summary.har`` of the ALTERTAX family run against GEMPACK's ``altax_summary.har``
and, at the check period (which reproduces the base), against the same GTAPSUM
formulas applied to the input ``basedata.har``."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.summary import (
    SUMMARY_HEADERS,
    summary_from_database,
    summary_headers,
    write_summary_har,
)
from equilibria.family.altertax.tolerances import CHECK_TOL, TOL_SUMMARY
from equilibria.family.altertax.updated import headers_at
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_summary.har"
ABS_FLOOR = 1.0e-6  # every variable of src/equilibria/compare/maps/gtap7.yaml
LEVELS = ("GDPE", "COST", "GDPI", "SALE")
EXPECTED = [
    "GDPE",
    "COST",
    "GDPI",
    "SALE",
    "VMCK",
    "VMCP",
    "TRCK",
    "TRCP",
    "VTCK",
    "VTCP",
    "MPCK",
    "MPCP",
    "VOCK",
    "VOCS",
    "VMCQ",
    "TRCQ",
    "VTCQ",
    "MPCQ",
    "VOCQ",
]


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return summary_headers(altax_run)


def _worst(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.vectorize(lambda x, y: rel_diff(x, y, y, ABS_FLOOR))(a, b)
    i = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    return float(d[i]) if d.size else 0.0, i


def _numeric(har):
    return {k: h for k, h in har.items() if np.asarray(h.array).dtype.kind == "f"}


def _basedata():
    """Numeric value headers of the input database."""
    return _numeric(read_har(str(nus333_dir() / "basedata.har")))


def test_summary_headers(headers):
    assert list(SUMMARY_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_summary_shapes_match_gempack(headers, gempack):
    for name in EXPECTED:
        a, b = headers[name], gempack[name]
        assert a.set_names == b.set_names, name
        assert a.set_elements == b.set_elements, name
        assert a.array.shape == b.array.shape, name
        assert (a.coeff_name, a.long_name) == (b.coeff_name, b.long_name), name


def test_summary_levels_match_gempack(headers, gempack):
    bad = {}
    for name in LEVELS:
        d, i = _worst(headers[name].array, gempack[name].array)
        if not d <= TOL_SUMMARY:
            bad[name] = (d, i)
    assert not bad, bad


# Accounting checks that are float32 rounding residuals of GEMPACK's own data and sums
# (VCB ~ 1e6, so one float32 ulp is 0.06-0.125 and a residual of -0.25 or -2 is 2-16 ulp).
# The cells are zero or a few ulp, so a relative metric against them is meaningless: the
# headers below are NOT matched at TOL_SUMMARY and the test below fails if one starts to be.
# Measured cause per header (max rel diff vs GEMPACK in the report):
#  * VMCK/VMCP/VTCK/VTCP (pre): GEMPACK's order of float32 accumulation of VCB, MAKEBCOM and
#    VTMUSE is not reproducible (a sequential float32 sum of the same basedata gives 3 of 6
#    VMCK cells, by chance); in float64 the data residual is a different small number.
#  * VMCQ/TRCQ/VTCQ/MPCQ (post): float32 residuals of GEMPACK's updated data against the
#    exact zero (|x| < 1e-6 %) that the float64 model gives.
NOISE_UNMATCHED = frozenset(
    ["VMCK", "VMCP", "VTCK", "VTCP", "VMCQ", "TRCQ", "VTCQ", "MPCQ"]
)
PERCENT = [
    "VMCP",
    "TRCP",
    "VTCP",
    "MPCP",
    "VOCS",
    "VMCQ",
    "TRCQ",
    "VTCQ",
    "MPCQ",
    "VOCQ",
]
NOISE_PCT = 1e-4  # percent: 1e-6 relative, about 17 float32 ulp


def _failing(headers, gempack, names):
    bad = {}
    for name in names:
        d, i = _worst(headers[name].array, gempack[name].array)
        if not d <= TOL_SUMMARY:
            bad[name] = (d, i)
    return bad


def test_summary_checks_match_gempack(headers, gempack):
    checks = [n for n in EXPECTED if n not in LEVELS]
    bad = _failing(headers, gempack, checks)
    assert set(bad) == NOISE_UNMATCHED, (set(bad) ^ NOISE_UNMATCHED, bad)


def test_unmatched_checks_are_rounding_noise(headers, gempack):
    for name in PERCENT:
        for side in (headers[name].array, gempack[name].array):
            assert np.abs(np.asarray(side, dtype=float)).max() <= NOISE_PCT, name


def test_pre_levels_are_the_summary_of_basedata(altax_run, headers):
    """The levels come from the check period, which reproduces the input data."""
    base = summary_from_database(_basedata())
    bad = {}
    for name in LEVELS:
        d, i = _worst(headers[name].array, base[name].array)
        if not d <= CHECK_TOL:
            bad[name] = (d, i)
    assert not bad, bad


def test_formulas_on_gempack_updated_data(gempack):
    """The post-sim formulas applied to GEMPACK's own updated database give GEMPACK's
    ``*Q`` headers where float32 sums allow (TRCQ, MPCQ); VMCQ and VTCQ are the same
    accumulation-order noise as VMCK and VTCK."""
    upd = read_har(
        str(ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_updated.har")
    )
    s = summary_from_database(_numeric(upd))
    assert not _failing(s, gempack, ["TRCQ", "MPCQ", "VOCQ", "GDPE"])
    assert {"VMCQ", "VTCQ"} <= set(_failing(s, gempack, ["VMCQ", "VTCQ"]))


def test_levels_carry_the_shock(altax_run, gempack):
    """Two-sided delta of the level tables on the post-update database (ours: shock minus
    check; GEMPACK: summary of altax_updated.har minus summary of basedata.har)."""
    upd = read_har(
        str(ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_updated.har")
    )
    g_post = summary_from_database(_numeric(upd))
    g_base = summary_from_database(_basedata())
    m_post = summary_from_database(headers_at(altax_run, "shock"))
    m_pre = summary_from_database(headers_at(altax_run, "check"))
    checked, bad = {}, []
    for name in LEVELS:
        g1, g0 = (np.asarray(x[name].array, dtype=np.float64) for x in (g_post, g_base))
        dg = g1 - g0
        dm = np.asarray(m_post[name].array, dtype=np.float64) - m_pre[name].array
        for ix in np.ndindex(g1.shape):
            thr = 8 * float(np.spacing(np.float32(max(abs(g1[ix]), abs(g0[ix])))))
            if max(abs(dg[ix]), abs(dm[ix])) <= thr:
                continue
            checked[name] = checked.get(name, 0) + 1
            if not (dm[ix] * dg[ix] > 0 and abs(dm[ix] - dg[ix]) <= 0.2 * abs(dg[ix])):
                bad.append((name, ix, float(dg[ix]), float(dm[ix]), thr))
    assert {"COST", "GDPI", "SALE"} <= set(checked), checked
    assert not bad, (checked, bad)


def test_outputs_finite(headers):
    for name, ha in headers.items():
        assert np.all(np.isfinite(np.asarray(ha.array, dtype=float))), name


def test_summary_loads(altax_run, tmp_path):
    written = tmp_path / "summary.har"
    write_summary_har(altax_run, written)
    got = read_har(str(written))
    assert set(got) >= set(EXPECTED)
