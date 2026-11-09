"""``volume.har`` of the ALTERTAX family run against GEMPACK's ``altax_volume.har``.

Every header is a change (``q% / 100 * BV``) driven by a ~1e-5 shock, so GEMPACK's float32
values carry noise of a few ulp of the level ``BV`` they are a change of. Cells whose
change is above ``THRESHOLD_ULPS`` ulp32(level) on either side are compared at
``TOL_VOLUME`` (relative, same sign); the rest must agree in absolute terms within the
same bound."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.tolerances import TOL_VOLUME
from equilibria.family.altertax.volume import (
    VOLUME_HEADERS,
    volume_bases,
    volume_headers,
    write_volume_har,
)
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_volume.har"
ABS_FLOOR = 1.0e-6  # every variable of src/equilibria/compare/maps/gtap7.yaml
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
            ok = m * g > 0 and rel_diff(m, g, g, ABS_FLOOR) <= TOL_VOLUME
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


def test_volume_within_float32_resolution_of_gempack(altax_run, headers, gempack):
    """Measured ceiling: every cell agrees with GEMPACK within ``THRESHOLD_ULPS`` float32
    ulp of its level (worst measured: 2.1 ulp), and above that bound the signs agree. This
    is all a float32 oracle can certify for these changes, see the xfail below."""
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
    # the comparison must really bite: most headers have cells above the noise bound
    assert sum(v > 0 for v in above.values()) >= 8, above


# The cells above the noise bound do NOT meet TOL_VOLUME (1e-4 relative). Measured cause:
# the shock is 0.0012% on one output tax, so a volume change is ~1e-5 of its level, which is
# 1e2 float32 ulp of the level (gem |change| / ulp32(level) <= 225); GEMPACK's float32
# solution therefore resolves a change to 1 ulp / change ~ 0.5-1% at best, and the
# differences are 0.3-2.1 ulp. DQO and DQC share the same 0.9011% relative difference (one
# shared xp(AGR,USA) quantity), i.e. a common-mode single-precision error, not a formula
# difference. Meeting 1e-4 would need a change of 1e4 ulp. The strict gate is kept (not
# loosened) and expected to fail; it turns into an error if it ever passes.
@pytest.mark.xfail(
    strict=True,
    reason="TOL_VOLUME=1e-4 is below the float32 resolution (~1/130 ulp) of GEMPACK changes",
)
def test_volume_values_meet_tol_volume(altax_run, headers, gempack):
    bases = volume_bases(altax_run)
    rel_bad = []
    for name in EXPECTED:
        rel_bad += _compare(
            name, headers[name].array, gempack[name].array, bases[name]
        )[1]
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
