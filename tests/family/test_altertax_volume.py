"""``volume.har`` of the ALTERTAX family run against GEMPACK's ``altax_volume.har``.

Every header is a change (``q% / 100 * BV``) driven by a ~1e-5 shock, so GEMPACK's float32
values carry noise of a few ulp of the level ``BV`` they are a change of. Cells whose
change is above ``THRESHOLD_ULPS`` ulp32(level) on either side are compared at
``TOL_CHANGE`` (relative, same sign); the rest must agree in absolute terms within the
same bound."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.tolerances import ABS_FLOOR, TOL_CHANGE
from equilibria.family.altertax.volume import (
    VOLUME_HEADERS,
    volume_bases,
    volume_headers,
    write_volume_har,
)
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_volume.har"
THRESHOLD_ULPS = 8
EXPECTED = [
    "DQXS",
    "DQDS",
    "DQFE",
    "DQFM",
    "DQFD",
    "DQPM",
    "DQPD",
    "DTOT",
    "DQO",
    "DQC",
    "DQGD",
    "DQGM",
]


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return volume_headers(altax_run)


def _ulp32(x):
    return np.spacing(
        np.abs(np.asarray(x, dtype=np.float64)).astype(np.float32)
    ).astype(np.float64)


def _compare(name, ours, gem, level):
    """(cells above threshold, relative failures, absolute failures below it)."""
    ours = np.asarray(ours, dtype=np.float64)
    gem = np.asarray(gem, dtype=np.float64)
    thr = THRESHOLD_ULPS * _ulp32(level)
    above, rel_bad, abs_bad = 0, [], []
    for ix in np.ndindex(ours.shape):
        m, g = float(ours[ix]), float(gem[ix])
        if max(abs(m), abs(g)) > thr[ix]:
            above += 1
            ok = m * g > 0 and rel_diff(m, g, g, ABS_FLOOR) <= TOL_CHANGE
            if not ok:
                rel_bad.append((name, ix, m, g, rel_diff(m, g, g, ABS_FLOOR)))
        elif abs(m - g) > thr[ix]:
            abs_bad.append((name, ix, m, g, float(thr[ix])))
    return above, rel_bad, abs_bad


def test_volume_headers(headers):
    assert list(VOLUME_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_volume_shapes_match_gempack(headers, gempack):
    for name in EXPECTED:
        a, b = headers[name], gempack[name]
        assert a.set_names == b.set_names, name
        assert a.set_elements == b.set_elements, name
        assert a.array.shape == b.array.shape, name
        assert (a.coeff_name, a.long_name) == (b.coeff_name, b.long_name), name


def test_volume_smoke_within_ulp_bound(altax_run, headers, gempack):
    """Smoke-level bound, NOT a certification: every cell is within ``THRESHOLD_ULPS``
    float32 ulp of its level of GEMPACK's value and has its sign above that bound. The
    bound is loose (8 ulp is 6-50% of these changes), so it catches wrong signs, wrong
    cells and gross errors only; the real gate is the xfail below."""
    bases = volume_bases(altax_run)
    above, sign_bad, abs_bad = {}, [], []
    for name in EXPECTED:
        ours = np.asarray(headers[name].array, dtype=np.float64)
        gem = np.asarray(gempack[name].array, dtype=np.float64)
        thr = THRESHOLD_ULPS * _ulp32(bases[name])
        n = 0
        for ix in np.ndindex(ours.shape):
            m, g = float(ours[ix]), float(gem[ix])
            if abs(m - g) > thr[ix]:
                abs_bad.append((name, ix, m, g, float(thr[ix])))
            if max(abs(m), abs(g)) > thr[ix]:
                n += 1
                if m * g <= 0:
                    sign_bad.append((name, ix, m, g))
        above[name] = n
    assert not abs_bad, (above, abs_bad)
    assert not sign_bad, (above, sign_bad)
    assert sum(v > 0 for v in above.values()) >= 8, above


# Gate (user 2026-11-09): every cell above the 8-ulp threshold has GEMPACK's sign and a
# relative gap <= TOL_CHANGE (2e-2) on the change. Measured cause of the gaps (worst 1.26%):
# GEMPACK's volume.har equals BV * q(altax.sl4) / 100 to 5e-7, so the gap is ours: the 1.01
# stand-in for the unit elasticities of altertax.prm (linear in sigma - 1, ~0.9% of the
# change per 0.01; DQO/DQC -0.901% at 1.01, extrapolating to -3e-4 at sigma = 1), plus a
# residual of 0.17-0.36% on import/trade cells (DQXS(0,0,1), DQGM(0,1), DQFM(0,2,1),
# DQPM(0,1)) that remains at sigma -> 1 and is unexplained. Exact sigma = 1 is future work.
def test_volume_values_within_tol_change(altax_run, headers, gempack):
    bases = volume_bases(altax_run)
    rel_bad, above = [], 0
    for name in EXPECTED:
        n, r, _ = _compare(name, headers[name].array, gempack[name].array, bases[name])
        above += n
        rel_bad += r
    assert above >= 20, above
    assert not rel_bad, rel_bad


def test_dtot_is_the_sum_of_dqxs(headers):
    np.testing.assert_allclose(
        headers["DTOT"].array, headers["DQXS"].array.sum(axis=0), rtol=0, atol=1e-15
    )


def test_zero_shock_gives_exact_zeros(altax_run):
    """check vs check: no change at all, exactly zero on every header."""
    zero = volume_headers(altax_run, post="check")
    for name in EXPECTED:
        assert not np.any(np.asarray(zero[name].array)), name


def test_volume_shock_is_not_zero(headers):
    assert any(np.any(np.abs(headers[n].array) > 1e-9) for n in EXPECTED)


def test_outputs_finite(headers):
    for name, ha in headers.items():
        assert np.all(np.isfinite(np.asarray(ha.array, dtype=float))), name


def test_volume_loads(altax_run, tmp_path):
    written = tmp_path / "volume.har"
    write_volume_har(altax_run, written)
    got = read_har(str(written))
    assert set(got) >= set(EXPECTED)
