"""``updated.har`` of the ALTERTAX family run against GEMPACK's ``altax_updated.har``
and, at the check period (which reproduces the base), against the input ``basedata.har``."""

from __future__ import annotations

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.family.altertax.gates import (
    TEXT_HEADERS,
    structure_problems,
    updated_failures,
)
from equilibria.family.altertax.gates import (
    worst_gap as _worst,
)
from equilibria.family.altertax.updated import (
    UPDATED_HEADERS,
    headers_at,
    updated_headers,
    write_updated_har,
)
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_updated.har"
TEXT = TEXT_HEADERS


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return updated_headers(altax_run)


def test_updated_headers(headers):
    assert set(headers) == set(UPDATED_HEADERS)
    assert len(UPDATED_HEADERS) == 33


def test_updated_har_shapes_match_gempack(headers, gempack):
    assert not structure_problems(headers, gempack, UPDATED_HEADERS)


def test_updated_matches_gempack(headers, gempack):
    bad = updated_failures(headers, gempack)  # gate: TOL_UPDATED
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
        "ours -0.349. MAKB = MAKS * (1 + prdtx_rai) holds exactly on our side "
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
    """Cells where EITHER side's change is above 8 float32 ulps (GEMPACK and ours: updated
    minus the stored basedata): the signs agree and ours is within 20% of
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
        dm = (
            np.asarray(ha.array, dtype=np.float64) - b
        )  # both sides: updated minus stored base
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


def _xxhs(lines, long_name="HISTORY OF THIS HEADER ARRAY FILE"):
    from equilibria.babel.har.symbols import HeaderArray

    return HeaderArray(
        name="XXHS",
        coeff_name="XXHS",
        long_name=long_name,
        array=np.array(lines, dtype=object),
        set_names=[],
        set_elements=[],
    )


def test_xxhs_history_text_is_not_compared_as_data():
    """XXHS is history text: other lines and another line count pass; its coefficient
    name, long name and sets are still compared."""
    ours = {"XXHS": _xxhs(["a", "b", "c"])}
    gem = {"XXHS": _xxhs(["x"] * 11)}
    assert structure_problems(ours, gem, ["XXHS"]) == []
    gem_name = {"XXHS": _xxhs(["x"], long_name="OTHER")}
    assert structure_problems(ours, gem_name, ["XXHS"]) == ["XXHS"]
    assert structure_problems(ours, gem_name, ["XXHS"], labels=False) == []


def test_rebase_scales_the_stored_base_by_the_model_ratio():
    from equilibria.family.altertax.updated import rebase

    stored = np.array([100.0, 50.0])
    check = np.array([100.0004, 49.9998])  # rebuild differs from the stored base
    shock = np.array([110.00044, 25.0])
    out = rebase(shock, check, stored)
    np.testing.assert_allclose(out, stored * shock / check, rtol=1e-12)
    assert abs(out[0] - 110.0) < abs(shock[0] - 110.0)


def test_rebase_keeps_the_shock_value_where_the_check_or_stored_base_is_zero():
    from equilibria.family.altertax.updated import rebase

    stored = np.array([0.0, 5.0, 7.0])
    check = np.array([3.0, 0.0, 2.0])
    shock = np.array([4.0, 9.0, 3.0])
    out = rebase(shock, check, stored)
    assert out[0] == 4.0 and out[1] == 9.0
    assert out[2] == pytest.approx(7.0 * 3.0 / 2.0)
