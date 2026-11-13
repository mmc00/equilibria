"""``summary.har`` of the ALTERTAX family run against GEMPACK's ``altax_summary.har``
and, at the check period (which reproduces the base), against the same GTAPSUM
formulas applied to the input ``basedata.har``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.family.altertax.gates import (
    SUMMARY_LEVELS,
    ULP_CHECKS,
    ULP_LIMIT,
    check_ulps,
    report_summary,
    structure_problems,
    summary_failures,
)
from equilibria.family.altertax.gates import (
    ulp32 as _ulp32,
)
from equilibria.family.altertax.gates import (
    worst_gap as _worst,
)
from equilibria.family.altertax.summary import (
    SUMMARY_HEADERS,
    input_database,
    numeric_headers,
    summary_from_database,
    summary_headers,
    write_summary_har,
)
from equilibria.family.altertax.summary import (
    abs_residual as _abs_residual,
)
from equilibria.family.altertax.summary import (
    check_flows as _flows,
)
from equilibria.family.altertax.tolerances import CHECK_TOL
from equilibria.family.altertax.updated import (
    headers_at,
    post_database,
    updated_headers,
)
from tests.family._prep import ROOT, altax_gem

FIXTURE = altax_gem() / "summary.har"
LEVELS = SUMMARY_LEVELS
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


def test_summary_headers(headers):
    assert list(SUMMARY_HEADERS) == EXPECTED
    assert set(headers) == set(EXPECTED)


def test_summary_shapes_match_gempack(headers, gempack):
    assert not structure_problems(headers, gempack, EXPECTED)


def test_summary_levels_match_gempack(headers, gempack):
    bad = summary_failures(headers, gempack, LEVELS)  # gate: TOL_SUMMARY
    assert not bad, bad


# ULP_CHECKS (the headers that are float32 residuals of GEMPACK's data, with the
# measured cause of each) live in equilibria.family.altertax.gates.
PRE_RESIDUALS = [
    "VMCK",
    "VMCP",
    "VTCK",
    "VTCP",
    "TRCK",
    "TRCP",
    "MPCK",
    "MPCP",
    "VOCK",
    "VOCS",
]
Q_RESIDUALS = ["VMCQ", "TRCQ", "VTCQ", "MPCQ", "VOCQ"]


_failing = summary_failures


def test_summary_checks_match_gempack(headers, gempack):
    """The checks that are not float32 residuals match at TOL_SUMMARY."""
    checks = [n for n in EXPECTED if n not in LEVELS and n not in ULP_CHECKS]
    assert not _failing(headers, gempack, checks)


def _flows_pair(altax_run):
    return _flows(input_database(altax_run)), _flows(post_database(altax_run))


def test_ulp_checks_are_the_ten_float32_residual_headers():
    assert set(ULP_CHECKS) == {
        "MPCQ", "TRCK", "TRCP", "TRCQ", "VMCK", "VMCP", "VMCQ", "VTCK", "VTCP", "VTCQ",
    }  # fmt: skip
    assert ULP_LIMIT == 64


def test_ulp_checks_pass_for_every_reference_without_waivers(
    altax_run, headers, gempack
):
    """One rule, no reference id: every ulp-checked header within 64 ulp of F, the gate
    passes (measured max ulp per header is in the gate record)."""
    rec = report_summary(headers, gempack, flows=_flows_pair(altax_run))
    assert rec["pass"] and not rec["failing_ulps"], rec["max_ulps"]
    assert set(rec["max_ulps"]) == set(ULP_CHECKS)
    assert max(rec["max_ulps"].values()) <= ULP_LIMIT


@pytest.mark.parametrize("ulps, ok", [(63, True), (65, False)])
def test_ulp_rule_boundary(altax_run, headers, gempack, ulps, ok):
    """A cell moved off GEMPACK by 63 ulp of its flow passes, by 65 fails."""
    flows = _flows_pair(altax_run)
    name = "TRCK"  # absolute header: the gap is added to the cell as is
    ix = tuple(np.argwhere(flows[0]["TRC"] > 0)[0])
    f = flows[0]["TRC"][ix]
    moved = dict(headers)
    arr = np.array(gempack[name].array, dtype=np.float64)
    arr[ix] += ulps * float(_ulp32(f))
    moved[name] = dataclasses.replace(headers[name], array=arr)
    assert float(check_ulps(name, moved, gempack, flows[0])[ix]) == pytest.approx(ulps)
    rec = report_summary(moved, gempack, flows=flows)
    assert (name not in rec["failing_ulps"]) is ok


def test_pre_residuals_within_ulp_of_gempack(altax_run, headers, gempack):
    """Pre-sim residuals (ours from basedata) are within 2 float32 ulp of the flow F
    they check, per cell (measured worst: about 1 ulp)."""
    flows = _flows(input_database(altax_run))
    bad = {}
    for name in PRE_RESIDUALS:
        ours = _abs_residual(name, headers[name].array, flows)
        gem = _abs_residual(name, gempack[name].array, flows)
        over = np.abs(ours - gem) > 2 * _ulp32(flows[name[:3]])
        if over.any():
            bad[name] = np.argwhere(over).tolist()
    assert not bad, bad


def test_post_residuals_are_float32_noise_like_gempacks(altax_run, headers, gempack):
    """*Q: ours (rebased updated values, float32) and GEMPACK's are float32 residuals,
    both within ULP_LIMIT float32 ulp of F (GEMPACK's within 8)."""
    flows = _flows(post_database(altax_run))
    bad = {}
    for name in Q_RESIDUALS:
        f = flows[name[:3]]
        ours = np.abs(_abs_residual(name, headers[name].array, flows))
        gem = np.abs(_abs_residual(name, gempack[name].array, flows))
        o = ours > ULP_LIMIT * _ulp32(f)
        g = gem > 8 * _ulp32(f)
        if o.any() or g.any():
            bad[name] = (np.argwhere(o).tolist(), np.argwhere(g).tolist())
    assert not bad, bad


def test_model_check_period_accounting_is_exact(altax_run):
    """The model's own accounting at the check period (discarded in the output, which
    takes the input data's residuals): every residual is zero to float64, |K| <= 1e-9 F."""
    db = headers_at(altax_run, "check")
    s = summary_from_database(db)
    flows = _flows(db)
    bad = {}
    for name in PRE_RESIDUALS + Q_RESIDUALS:
        f = flows[name[:3]]
        over = np.abs(_abs_residual(name, s[name].array, flows)) > 1e-9 * np.abs(f)
        if over.any():
            bad[name] = np.argwhere(over).tolist()
    assert not bad, bad


def test_pre_levels_are_the_summary_of_basedata(altax_run, headers):
    """The levels come from the check period, which reproduces the input data."""
    base = summary_from_database(input_database(altax_run))
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
    upd = read_har(str(altax_gem() / "updated.har"))
    s = summary_from_database(numeric_headers(upd))
    assert not _failing(s, gempack, ["TRCQ", "MPCQ", "VOCQ", "GDPE"])
    assert {"VMCQ", "VTCQ"} <= set(_failing(s, gempack, ["VMCQ", "VTCQ"]))


def test_levels_carry_the_shock(altax_run, gempack):
    """Two-sided delta of the level tables on the post-update database (ours: shock minus
    check; GEMPACK: summary of altax_updated.har minus summary of basedata.har)."""
    upd = read_har(str(altax_gem() / "updated.har"))
    g_post = summary_from_database(numeric_headers(upd))
    g_base = summary_from_database(input_database(altax_run))
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


def test_post_checks_are_evaluated_on_the_rebased_updated_values(
    altax_run, headers, gempack
):
    """GEMPACK's ``*Q`` checks run on its updated data, which carries the input data's
    float32 imbalance: ours are the same formulas on the rebased updated values in
    float32, not the exact identities of the raw shock levels (which are zero)."""
    upd = updated_headers(altax_run)
    post = post_database(altax_run)
    assert set(post) <= set(upd) and all(
        np.array_equal(post[k].array, upd[k].array) for k in post
    )  # one computation shared by the updated.har and the summary
    expected = summary_from_database(post, dtype=np.float32)
    for q, p in (("VMCQ", "VMCP"), ("TRCQ", "TRCP"), ("VTCQ", "VTCP"),
                 ("MPCQ", "MPCP"), ("VOCQ", "VOCS")):  # fmt: skip
        assert np.array_equal(headers[q].array, expected[p].array), q
    raw = summary_from_database(headers_at(altax_run, "shock"))
    assert not np.array_equal(headers["TRCQ"].array, raw["TRCP"].array)
    assert np.any(np.asarray(headers["TRCQ"].array) != 0.0)
