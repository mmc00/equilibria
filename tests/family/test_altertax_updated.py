"""``updated.har`` of the ALTERTAX family run against GEMPACK's ``altax_updated.har``
and, at the check period (which reproduces the base), against the input ``basedata.har``."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.tolerances import CHECK_TOL, TOL_UPDATED
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
        if not d <= TOL_UPDATED:
            bad[name] = (d, i)
    assert not bad, bad
    # the data are consistent with the check tolerance on the base: nothing is far off
    assert CHECK_TOL < TOL_UPDATED


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
