"""``decomp.har`` of the ALTERTAX family run against GEMPACK's ``altax_decomp.har``.

Two gates. ``test_formulas_*`` feeds GEMPACK's own cumulative changes (``altax.sl4``) and
updated data (``altax_updated.har``) to ``decomp_from_changes``: it certifies the
transcription of WELVIEW. ``test_ev_*`` / ``test_regional_totals_*`` feed the family model's changes: the
end-to-end gate is the EV of each region and each header's regional totals at TOL_CHANGE
(the cell level is only reported), see the comment above ``KNOWN_MISS``.

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
from equilibria.family.altertax.decomp import (
    CHANGE_SHAPES,
    DECOMP_HEADERS,
    decomp_from_changes,
    decomp_headers,
    write_decomp_har,
)
from equilibria.family.altertax.gates import (
    DECOMP_KNOWN_MISS as KNOWN_MISS,
)
from equilibria.family.altertax.gates import (
    NOISE,
    decomp_scale,
    ev_gaps,
    miss_is_bounded,
    structure_problems,
)
from equilibria.family.altertax.gates import (
    decomp_compare as _compare,
)
from equilibria.family.altertax.gates import (
    total_gaps as _total_gaps,
)
from equilibria.family.altertax.summary import numeric_headers
from equilibria.family.altertax.tolerances import TOL_CHANGE
from tests.family._prep import ROOT

FIXTURE = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg/altax_decomp.har"
EXPECTED = [
    "A", "A1", "A2", "A21", "A211", "A22", "A221", "A222", "A22F", "A22P", "A22G",
    "A22I", "A23", "A231", "A24", "A241", "A25", "A251", "B1", "B2", "C1", "C11",
    "C12", "C13", "C14", "C15", "C16", "C17", "E1", "F1",
]  # fmt: skip


@pytest.fixture(scope="module")
def gempack():
    return read_har(str(FIXTURE))


@pytest.fixture(scope="module")
def headers(altax_run):
    return decomp_headers(altax_run)


@pytest.fixture(scope="module")
def scale(gempack):
    """max|EV| of each region: the largest |component| of its WELFARE row."""
    return decomp_scale(gempack)


def test_decomp_headers(headers):
    assert list(DECOMP_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_decomp_shapes_match_gempack(headers, gempack):
    assert not structure_problems(headers, gempack, EXPECTED)


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


# _regional_totals, _total_gaps and KNOWN_MISS (with the measured cause of every miss)
# live in equilibria.family.altertax.gates.


def test_ev_by_region_within_tol_change(headers, gempack):
    """EV of each region (sum of the WELFARE row, EV_ALT of line 3820) vs GEMPACK's."""
    gap = ev_gaps(headers, gempack)
    assert max(gap) <= TOL_CHANGE, gap


# End-to-end gate (user 2026-11-09 and controller ruling): TOL_CHANGE on the EV of each
# region and on each header's regional totals above the noise floor; the cell-level
# components are NOT gated (they are mixed-sign sums of tiny terms; the formulas are
# certified on GEMPACK's own changes, ``test_formulas_*``, at 0.26 %). The cell-level
# gap is reported by ``test_cell_level_report`` (-s). Closure: the nominal trade balance
# (Task 7b); before it the EV gap was 4 % (savf = pigbl * savf_bar is real).
def test_regional_totals_within_tol_change(headers, gempack, scale):
    gaps = _total_gaps(headers, gempack, scale)
    assert len(gaps) >= 40, len(gaps)
    bad = {k: v for k, v in gaps.items() if k not in KNOWN_MISS and v[2] > TOL_CHANGE}
    assert not bad, "\n".join(f"{k} {v}" for k, v in sorted(bad.items()))


def test_known_misses_are_bounded(headers, gempack, scale):
    gaps = _total_gaps(headers, gempack, scale)
    bad = []
    for k, ref in KNOWN_MISS.items():
        m, g, rel = gaps[k]
        err = (m - g) / abs(g)
        if rel <= TOL_CHANGE:
            bad.append((k, "now within TOL_CHANGE, remove it", err))
        elif not miss_is_bounded(err, ref):
            bad.append((k, "error changed sign or grew beyond 1.5x", err, ref))
    assert not bad, bad


def test_zero_components_stay_zero(headers, gempack):
    """The endowment, technical change and population columns (and the B, C headers) are
    zero in GEMPACK's real shock; so are they here (float noise below 1e-6 of EV)."""
    tol = NOISE * np.abs(np.asarray(gempack["A"].array, dtype=np.float64)).max()
    for name in ("B1", "B2", *[n for n in EXPECTED if n.startswith("C")]):
        assert np.abs(np.asarray(gempack[name].array, dtype=np.float64)).max() <= tol, (
            name
        )
        assert np.abs(np.asarray(headers[name].array)).max() <= tol, name
    ours, gem = np.asarray(headers["A"].array), np.asarray(gempack["A"].array, float)
    for col in (1, 2, 3, 6):  # ENDWB1 tech_C1 pop_D1 pref_G1
        assert np.abs(gem[:, col]).max() <= tol, col
        assert np.abs(ours[:, col]).max() <= tol, col


def test_cell_level_report(headers, gempack, scale, capsys):
    """Informational (not a gate): how many cells above the noise floor are within
    TOL_CHANGE of GEMPACK's."""
    ok = total = 0
    for name in EXPECTED:
        n, r, _ = _compare(name, headers[name], gempack[name], scale)
        total += n
        ok += n - len(r)
    with capsys.disabled():
        print(f"\ndecomp cells within TOL_CHANGE: {ok} of {total}")
    assert total >= 100


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
        _check(a, b)  # duck-typed stand-in for _Period

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
