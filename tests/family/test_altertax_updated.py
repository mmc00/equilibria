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


def test_headers_carry_the_shock(altax_run, headers, gempack):
    """Cells where GEMPACK's change (updated minus basedata) is above ~8 float32 ulps
    and 5x our solve noise (check minus basedata): ours (shock minus check) has the same sign and is within 20% of it."""
    base = read_har(str(nus333_dir() / "basedata.har"))
    chk = headers_at(altax_run, "check")
    ulp = float(np.finfo(np.float32).eps)
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
            if abs(dg[ix]) <= 8 * ulp * max(abs(g[ix]), abs(b[ix])):
                continue
            # below 5x the solve noise (check minus base) the change is not measurable
            if abs(dg[ix]) < 5 * abs(chk[name].array[ix] - b[ix]):
                continue
            checked[name] = checked.get(name, 0) + 1
            if not (dm[ix] * dg[ix] > 0 and abs(dm[ix] - dg[ix]) <= 0.2 * abs(dg[ix])):
                bad.append((name, ix, float(dg[ix]), float(dm[ix])))
    assert {"VTWR", "MAKS", "VDFB", "EVFP"} <= set(checked), checked
    assert not bad, bad
