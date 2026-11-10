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
from equilibria.family.altertax.gates import (
    THRESHOLD_ULPS,
    structure_problems,
)
from equilibria.family.altertax.gates import (
    ulp32 as _ulp32,
)
from equilibria.family.altertax.gates import (
    volume_compare as _compare,
)
from equilibria.family.altertax.volume import (
    VOLUME_HEADERS,
    volume_bases,
    volume_headers,
    write_volume_har,
)
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_volume.har"
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


def test_volume_headers(headers):
    assert list(VOLUME_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_volume_shapes_match_gempack(headers, gempack):
    assert not structure_problems(headers, gempack, EXPECTED)


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
# relative gap <= TOL_CHANGE (2e-2) on the change. GEMPACK's volume.har equals
# BV * q(altax.sl4) / 100 to 5e-7, so any gap is ours. With the 1.01 stand-in the worst gap
# was 1.26 % (linear in sigma - 1, ~0.9 % of the change per 0.01); the exact sigma = 1
# Cobb-Douglas (b8fe408f, b3b161ac) and the base Fisher refresh (440d9533) close it.
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
