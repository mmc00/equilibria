"""``decomp.har`` of the ALTERTAX family run against GEMPACK's ``altax_decomp.har``.

Two gates. ``test_formulas_*`` feeds GEMPACK's own cumulative changes (``altax.sl4``) and
updated data (``altax_updated.har``) to ``decomp_from_changes``: it certifies the
transcription of WELVIEW. ``test_decomp_values_*`` feeds the family model's changes: it
is the end-to-end gate and is an expected failure (strict) because the changes of the
family model differ from GEMPACK's in the small cells, see the comment above it.

Every welfare component is a change: ``0.01 * EVSCALFACT * (tax revenue) * (% change of a
quantity)`` of a ~1e-5 shock, so GEMPACK's float32 values carry noise relative to the scale
of the decomposition (the size of the regional EV). Cells where either side is above
``NOISE * max|WELFARE(r, .)|`` are compared relatively at ``TOL_CHANGE`` (same sign); the
rest must agree in absolute terms within the noise threshold itself (an absolute bound:
a relative one is meaningless on a number that is mostly float32 noise)."""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax.decomp import (
    CHANGE_SHAPES,
    DECOMP_HEADERS,
    decomp_from_changes,
    decomp_headers,
    write_decomp_har,
)
from equilibria.family.altertax.summary import numeric_headers
from equilibria.family.altertax.tolerances import ABS_FLOOR, TOL_CHANGE
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_decomp.har"
NOISE = 1e-6  # relative to max|EV| of the region (user 2026-11-09)
EXPECTED = [
    "A", "A1", "A2", "A21", "A211", "A22", "A221", "A222", "A22F", "A22P", "A22G",
    "A22I", "A23", "A231", "A24", "A241", "A25", "A251", "B1", "B2", "C1", "C11",
    "C12", "C13", "C14", "C15", "C16", "C17", "E1", "F1",
]  # fmt: skip
# the columns of the explanatory-factor headers that are tax rates (levels, not changes)
RATE_COLS = ("taxrateb", "taxrateu")


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return decomp_headers(altax_run)


@pytest.fixture(scope="module")
def scale(gempack):
    """max|EV| of each region: the largest |component| of its WELFARE row."""
    return np.abs(np.asarray(gempack["A"].array, dtype=np.float64)).max(axis=1)


def _region_scale(ha, ix, scale):
    """Scale of cell ``ix``: the largest scale of the regions it is indexed by."""
    pos = [i for i, n in enumerate(ha.set_names) if n == "REG"]
    return max(scale[ix[i]] for i in pos) if pos else scale.max()


def _compare(name, ours_ha, gem_ha, scale):
    """(cells above the noise threshold, relative failures, absolute failures)."""
    ours = np.asarray(ours_ha.array, dtype=np.float64)
    gem = np.asarray(gem_ha.array, dtype=np.float64)
    last = gem_ha.set_elements[-1] if gem_ha.set_names else []
    above, rel_bad, abs_bad = 0, [], []
    for ix in np.ndindex(ours.shape):
        m, g = float(ours[ix]), float(gem[ix])
        thr = NOISE * _region_scale(gem_ha, ix, scale)
        col = last[ix[-1]] if gem_ha.set_names and gem_ha.set_names[-1] == "COL" else ""
        if col in RATE_COLS:
            # a tax rate is a level: relative at TOL_CHANGE, no noise threshold
            if rel_diff(m, g, g, ABS_FLOOR) > TOL_CHANGE:
                rel_bad.append((name, ix, m, g, rel_diff(m, g, g, ABS_FLOOR)))
            above += 1
        elif max(abs(m), abs(g)) > thr:
            above += 1
            ok = m * g > 0 and rel_diff(m, g, g, ABS_FLOOR) <= TOL_CHANGE
            if not ok:
                rel_bad.append((name, ix, m, g, rel_diff(m, g, g, ABS_FLOOR)))
        elif abs(m - g) > thr:
            abs_bad.append((name, ix, m, g, thr))
    return above, rel_bad, abs_bad


def test_decomp_headers(headers):
    assert list(DECOMP_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_decomp_shapes_match_gempack(headers, gempack):
    for name in EXPECTED:
        a, b = headers[name], gempack[name]
        assert a.set_names == b.set_names, name
        assert a.set_elements == b.set_elements, name
        assert a.array.shape == b.array.shape, name
        assert (a.coeff_name, a.long_name) == (b.coeff_name, b.long_name), name


def _check_gate(headers, gempack, scale):
    rel_bad, abs_bad, above = [], [], 0
    for name in EXPECTED:
        n, r, a = _compare(name, headers[name], gempack[name], scale)
        above += n
        rel_bad += r
        abs_bad += a
    assert above >= 100, above
    assert not rel_bad, rel_bad
    assert not abs_bad, abs_bad


def _sl4_changes(path, els):
    """GEMPACK's cumulative percent changes of the variables ``decomp_from_changes``
    needs, read from the ``.sl4`` (``CUMS``, one value per cell in Fortran order)."""
    h = read_har(str(path))

    def col(n):
        return np.asarray(h[n].array).reshape(-1)

    names = [str(x).strip() for x in col("VARS")]
    nargs, setof = col("VCNA"), col("VCAR")
    first = np.concatenate(([0], np.cumsum(nargs)[:-1]))
    size, cm, pc, cums = col("SSZ"), col("CMND"), col("PCUM"), col("CUMS").astype(float)
    out = {}
    for var, axes in CHANGE_SHAPES.items():
        if var == "pop":  # exogenous, not retained in the solution: no change
            out[var] = np.zeros(len(els["REG"]))
            continue
        i = names.index(var)
        a0 = int(first[i])
        shape = tuple(int(size[int(s) - 1]) for s in setof[a0 : a0 + int(nargs[i])])
        n = int(np.prod(shape))
        assert int(cm[i]) == n and len(shape) == len(axes), var
        p = int(pc[i]) - 1
        out[var] = cums[p : p + n].reshape(shape, order="F")
    return out


@pytest.fixture(scope="module")
def sl4_headers(gempack, tmp_path_factory):
    """``decomp_from_changes`` on GEMPACK's own solution: base data, updated data and the
    cumulative changes of ``altax.sl4``."""
    from equilibria._local_refs import nus333_dir

    d = tmp_path_factory.mktemp("sl4")
    sl4 = d / "altax.sl4"
    with gzip.open(FIXTURE.parent / "altax.sl4.gz", "rb") as src, sl4.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    els = {
        "COMM": list(gempack["A211"].set_elements[0]),
        "ACTS": list(gempack["A211"].set_elements[1]),
        "REG": list(gempack["A"].set_elements[0]),
        "ENDW": list(gempack["A24"].set_elements[0]),
        "MARG": list(gempack["C16"].set_elements[0]),
    }
    pre = numeric_headers(read_har(str(nus333_dir() / "basedata.har")))
    post = numeric_headers(read_har(str(FIXTURE.parent / "altax_updated.har")))

    def arrays(har):
        return {k: np.asarray(h.array, dtype=np.float64) for k, h in har.items()}

    return decomp_from_changes(arrays(pre), arrays(post), _sl4_changes(sl4, els), els)


def test_formulas_reproduce_gempack_from_its_own_changes(sl4_headers, gempack, scale):
    """WELVIEW transcription: with GEMPACK's own changes and data every cell is within
    TOL_CHANGE of ``altax_decomp.har`` (worst measured 0.26%, the shocked output-tax
    cell: one step of the path integral against GEMPACK's multistep one)."""
    _check_gate(sl4_headers, gempack, scale)


# End-to-end gate (user 2026-11-09: TOL_CHANGE on each welfare component, noise threshold
# 1e-6 * max|EV|). EXPECTED FAILURE. Measured cause (Task 7 fix round 1): a CLOSURE
# mismatch, not the 1.01 stand-in and not small-cell quantities. GEMPACK fixes the NOMINAL
# trade balance of USA (del_tbal(USA) = -2.5e-10 in altax.sl4); the family model fixes
# savf(r) = pigbl * savf_bar(r) (blocks/gtap/demand_utility.py:627) and pigbl moves
# +7.3e-6 % while pnum stays, so savf(USA) and savf(ROW) move by the same +7.3e-6 %: a
# 0.06 M transfer ROW -> USA on a deficit of 831,458. Swapping GEMPACK's variables into
# our change vector one at a time puts ~88% of the 0.047 EV gap in prices (pfob: -0.0279
# terms of trade, -0.0035 I-S; pinv -0.0174; psave +0.0086), all quantities together only
# about -0.0057. Counterfactual (probe, savf = savf_bar, i.e. nominal, shared code
# untouched in a copy of src): EV USA/ROW 1.1592 / -1.2553 against GEMPACK 1.1704 /
# -1.2657 (from 1.2173 / -1.3142), cells within TOL_CHANGE 270 of 401 (from 151); the
# rest is the 1.01 stand-in plus the cancellation of mixed-sign tax-weighted sums. The
# formulas are exact (test above).
@pytest.mark.xfail(
    strict=True,
    reason="closure mismatch: savf = pigbl*savf_bar is real, GEMPACK fixes nominal del_tbal (see above)",
)
def test_decomp_values_within_tol_change(headers, gempack, scale):
    _check_gate(headers, gempack, scale)


def test_decomp_smoke_totals_and_signs(headers, gempack):
    """Smoke-level bound, NOT a certification: the EV of each region (sum of the WELFARE
    row, EV_ALT of line 3820) is within 5% of GEMPACK's, the allocative and terms-of-trade
    totals have its sign, and the endowment, technical and population columns vanish."""
    ev = np.asarray(headers["A"].array).sum(axis=1)
    gem = np.asarray(gempack["A"].array, dtype=np.float64)
    np.testing.assert_allclose(ev, gem.sum(axis=1), rtol=5e-2)
    ours = np.asarray(headers["A"].array)
    for col in (0, 4):  # alloc_A1, tot_E1
        assert np.all(ours[:, col] * gem[:, col] > 0), col
    assert np.abs(ours[:, [1, 2, 3, 6]]).max() < 1e-6 * np.abs(gem).max()


def test_welfare_columns_add_up(headers):
    """The WELFARE summary is the sum of its parts, as WELVIEW defines it."""
    A = np.asarray(headers["A"].array)  # (REG, COLUMN)
    a1 = np.asarray(headers["A1"].array)  # (ALLOCEFF, REG)
    a2 = np.asarray(headers["A2"].array)  # (REG, CTAX)
    b1 = np.asarray(headers["B1"].array)  # (REG, ENDW)
    b2 = np.asarray(headers["B2"].array)  # (REG,)
    c1 = np.asarray(headers["C1"].array)  # (REG, TECHTYPE)
    e1 = np.asarray(headers["E1"].array)  # (COMM, REG, PRICES, FORM)
    np.testing.assert_allclose(A[:, 0], a2.sum(axis=1), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(A[:, 1], b1.sum(axis=1) + b2, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(A[:, 2], c1.sum(axis=1), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(
        A[:, 4], e1[..., 1].sum(axis=(0, 2)), rtol=1e-9, atol=1e-18
    )
    # A1: endowments + activities + commodities are the same allocative total as A2
    np.testing.assert_allclose(A[:, 0], a1.sum(axis=0), rtol=1e-12, atol=1e-18)


def test_tax_type_components_add_up(headers):
    """A2 (tax type) is the sum of the per-instrument components of A21..A25."""
    a2 = np.asarray(headers["A2"].array)
    ctax = headers["A2"].set_elements[1]
    col = {k: i for i, k in enumerate(ctax)}
    a21 = np.asarray(headers["A21"].array)  # (ACTS, REG)
    a22 = np.asarray(headers["A22"].array)  # (COMM, ACTIVITY, REG, DIR)
    a23 = np.asarray(headers["A23"].array)  # (COMM, REG, REG, TYPE)
    a24 = np.asarray(headers["A24"].array)  # (ENDW, REG)
    a25 = np.asarray(headers["A25"].array)  # (ENDW, REG)
    act = {k: i for i, k in enumerate(headers["A22"].set_elements[1])}
    np.testing.assert_allclose(
        a2[:, col["prodtax"]], a21.sum(axis=0), rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose(
        a2[:, col["pfacttax"]], a24.sum(axis=0), rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose(
        a2[:, col["inctax"]], a25.sum(axis=0), rtol=1e-12, atol=1e-18
    )
    for tax, agent in (
        ("inputtax", "firm"), ("contax", "private"),
        ("govtax", "govt"), ("invtax", "investment"),
    ):  # fmt: skip
        np.testing.assert_allclose(
            a2[:, col[tax]], a22[:, act[agent]].sum(axis=(0, 2)), rtol=1e-12, atol=1e-18
        )
    # xtax by source (axis 1), mtax by destination (axis 2)
    np.testing.assert_allclose(
        a2[:, col["xtax"]], a23[..., 0].sum(axis=(0, 2)), rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose(
        a2[:, col["mtax"]], a23[..., 1].sum(axis=(0, 1)), rtol=1e-12, atol=1e-18
    )


def test_explanatory_welcnt_equals_summary(headers):
    """The ``welcnt`` column of every explanatory header is the summary it explains."""
    a22 = np.asarray(headers["A22"].array)
    act = {k: i for i, k in enumerate(headers["A22"].set_elements[1])}
    for name, agent in (("A22P", "private"), ("A22G", "govt"), ("A22I", "investment")):
        np.testing.assert_array_equal(
            np.asarray(headers[name].array)[..., 0], a22[:, act[agent]], err_msg=name
        )
    np.testing.assert_array_equal(
        np.asarray(headers["A22F"].array)[..., 0], np.asarray(headers["A221"].array)
    )
    # OTAX(a,r) is the sum over commodities of the A211 welcnt column
    np.testing.assert_allclose(
        np.asarray(headers["A211"].array)[..., 0].sum(axis=0),
        np.asarray(headers["A21"].array),
        rtol=1e-12,
        atol=1e-18,
    )
    np.testing.assert_array_equal(
        np.asarray(headers["A241"].array)[..., 0].sum(axis=1),
        np.asarray(headers["A24"].array),
    )


def test_unchanged_check_raises():
    """A moved exogenous variable, or a missing one, is an error, not a wrong zero."""
    from equilibria.family.altertax.decomp import _UNCHANGED
    from equilibria.family.altertax.decomp import _require_unchanged as _check

    def _require_unchanged(a, b):
        _check(a, b)  # type: ignore[arg-type]  # duck-typed stand-in for _Period

    class P:
        def __init__(self, vals, names=_UNCHANGED):
            self._vars = set(names)
            self.vals = vals

        def __getitem__(self, name):
            return {(): self.vals.get(name, 1.0)}

    _require_unchanged(P({}), P({}))
    with pytest.raises(NotImplementedError, match="aft"):
        _require_unchanged(P({}), P({"aft": 1.1}))
    with pytest.raises(KeyError, match="aft"):
        _require_unchanged(P({}, [n for n in _UNCHANGED if n != "aft"]), P({}))


def test_zero_shock_gives_exact_zeros(altax_run):
    """check vs check: no change at all; the welfare components and the volumes are
    exactly zero (the tax-rate and price-index columns are levels, not changes)."""
    zero = decomp_headers(altax_run, post="check")
    for name in ("A", "A1", "A2", "A21", "A22", "A221", "A222", "A23", "A24", "A25",
                 "B1", "B2", "C1", "E1"):  # fmt: skip
        assert not np.any(np.asarray(zero[name].array)), name
    for name in ("A211", "A22F", "A22P", "A22G", "A22I", "A231", "A241", "A251"):
        arr = np.asarray(zero[name].array)
        cols = zero[name].set_elements[-1]
        for k, c in enumerate(cols):
            if c in ("welcnt", "dvol"):
                assert not np.any(arr[..., k]), (name, c)


def test_shock_is_not_zero(headers):
    assert np.any(np.abs(np.asarray(headers["A"].array)) > 1e-3)


def test_outputs_finite(headers):
    for name, ha in headers.items():
        assert np.all(np.isfinite(np.asarray(ha.array, dtype=float))), name


def test_decomp_loads(altax_run, tmp_path):
    written = tmp_path / "decomp.har"
    write_decomp_har(altax_run, written)
    got = read_har(str(written))
    assert set(got) >= set(EXPECTED)
