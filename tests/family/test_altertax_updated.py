"""``updated.har`` of the ALTERTAX family run against GEMPACK's ``altax_updated.har``
and, at the check period (which reproduces the base), against the input ``basedata.har``."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.tolerances import TOL_UPDATED
from equilibria.family.altertax.updated import (
    UPDATED_HEADERS,
    headers_at,
    updated_headers,
    write_updated_har,
)
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_updated.har"
ABS_FLOOR = 1.0e-6  # every variable of src/equilibria/compare/maps/gtap7.yaml
TEXT = ("XXHS",)


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return updated_headers(altax_run)


def _worst(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.vectorize(lambda x, y: rel_diff(x, y, y, ABS_FLOOR))(a, b)
    i = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    return float(d[i]) if d.size else 0.0, i


def test_updated_headers(headers):
    assert set(headers) == set(UPDATED_HEADERS)
    assert len(UPDATED_HEADERS) == 33


def test_updated_har_shapes_match_gempack(headers, gempack):
    for name in UPDATED_HEADERS:
        a, b = headers[name], gempack[name]
        assert a.set_names == b.set_names, name
        assert a.set_elements == b.set_elements, name
        assert a.array.shape == b.array.shape, name
        assert (a.coeff_name, a.long_name) == (b.coeff_name, b.long_name), name


def test_updated_matches_gempack(headers, gempack):
    bad = {}
    for name in UPDATED_HEADERS:
        if name in TEXT:
            continue
        d, i = _worst(headers[name].array, gempack[name].array)
        if not d <= TOL_UPDATED:
            bad[name] = (d, i)
    assert not bad, bad


def test_check_period_reproduces_basedata(altax_run):
    base = read_har(str(nus333_dir() / "basedata.har"))
    chk = headers_at(altax_run, "check")
    assert chk
    bad = {}
    for name, ha in chk.items():
        d, i = _worst(ha.array, base[name].array)
        if not d <= 1e-5:
            bad[name] = (d, i)
    assert not bad, bad


def test_outputs_finite(headers):
    for name, ha in headers.items():
        if name in TEXT:
            continue
        assert np.all(np.isfinite(np.asarray(ha.array, dtype=float))), name


def test_updated_loads(altax_run, tmp_path):
    from equilibria.templates.gtap import GTAPParameters

    d = nus333_dir()
    written = tmp_path / "updated.har"
    write_updated_har(altax_run, written)
    assert set(read_har(str(written))) >= set(UPDATED_HEADERS)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=written,
        sets_path=d / "sets.har",
        default_path=d / "altertax.prm",
        baserate_path=d / "baserate.har",
    )
    assert p.benchmark.vdfb


# (header, cell) -> measured cause; cells where the two sides legitimately differ
SHOCK_DELTA_EXCLUSIONS: dict[tuple[str, tuple[int, ...]], str] = {
    ("MAKB", (0, 0, 0)): (
        "MAKB[AGR,AGR,USA], the shocked cell: GEMPACK moves -0.25 (8 float32 ulps), "
        "ours -0.405. MAKB = MAKS * (1 + prdtx_rai) holds exactly on our side "
        "(power 1.0024391 -> 1.0024511, +1.2000e-5, the 0.0012% shock); GEMPACK's own "
        "MAKB/MAKS goes 1.0024391 -> 1.0024517 (+1.26e-5, beyond float32 resolution of "
        "~1e-7), i.e. its MAKB is 5.6e-7 relative off its own MAKS and the stated shock. "
        "MAKB is a difference of two 1.2e-5 moves (price up, quantity down), so that "
        "5.6e-7 shows as 0.155 of 0.25. MAKS agrees to 1%, and the level agrees to "
        "1e-6. Not a sigma artefact: with sigma 1.02 ours moves to -0.443, away from "
        "GEMPACK."
    ),
}


def shock_delta_failures(altax_run, headers, gempack):
    """Cells where EITHER side's change is above 8 float32 ulps (GEMPACK: updated minus
    basedata; ours: shock minus check): the signs agree and ours is within 20% of
    GEMPACK's. Returns ``(checked per header, failures)``."""
    base = read_har(str(nus333_dir() / "basedata.har"))
    chk = headers_at(altax_run, "check")
    checked: dict[str, int] = {}
    bad = []
    for name, ha in headers.items():
        if name not in chk:
            continue
        g = np.asarray(gempack[name].array, dtype=np.float64)
        b = np.asarray(base[name].array, dtype=np.float64)
        dg = g - b
        dm = np.asarray(ha.array, dtype=np.float64) - chk[name].array
        for ix in np.ndindex(g.shape):
            thr = 8 * float(np.spacing(np.float32(max(abs(g[ix]), abs(b[ix])))))
            if max(abs(dg[ix]), abs(dm[ix])) <= thr:
                continue
            checked[name] = checked.get(name, 0) + 1
            if not (dm[ix] * dg[ix] > 0 and abs(dm[ix] - dg[ix]) <= 0.2 * abs(dg[ix])):
                bad.append((name, ix, float(dg[ix]), float(dm[ix]), thr))
    return checked, bad


def test_headers_carry_the_shock(altax_run, headers, gempack):
    checked, bad = shock_delta_failures(altax_run, headers, gempack)
    assert {"VTWR", "MAKS", "VDFB", "EVFP"} <= set(checked), checked
    assert all(why.strip() for why in SHOCK_DELTA_EXCLUSIONS.values())
    assert not [x for x in bad if (x[0], x[1]) not in SHOCK_DELTA_EXCLUSIONS], bad
    assert {(x[0], x[1]) for x in bad} >= set(SHOCK_DELTA_EXCLUSIONS)  # none is stale
