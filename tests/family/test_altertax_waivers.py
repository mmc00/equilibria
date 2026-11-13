"""The known-miss waivers apply to the nus333 ALTAX reference only, never by key alone."""

from __future__ import annotations

import dataclasses
import shutil

import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.family.altertax import gates
from equilibria.family.altertax.decomp import decomp_headers
from equilibria.family.altertax.pct import gempack_changes, pct_changes
from equilibria.family.altertax.summary import (
    check_flows,
    input_database,
    summary_headers,
)
from equilibria.family.altertax.updated import headers_at, post_database
from tests.family._prep import altax_gem

FIXTURES = altax_gem()


@pytest.fixture(scope="module")
def sl4(tmp_path_factory):
    d = tmp_path_factory.mktemp("sl4")
    return gates.read_sl4(FIXTURES / "solution.sl4")


def test_nus333_altax_reference_is_recognised(sl4):
    assert gates.is_nus333_altax_reference(nus333_dir(), sl4)
    w = gates.waivers_for(nus333_dir(), sl4)
    assert w is gates.NUS333_ALTAX_WAIVERS and w.reference == gates.REFERENCE_ID
    assert not w.pct and not w.decomp  # empty at exact sigma = 1


def test_other_input_data_gets_no_waivers(sl4, tmp_path):
    """A dataset with the same region names but other data (3x3, 5x5, ...)."""
    shutil.copytree(nus333_dir(), tmp_path / "other", dirs_exist_ok=True)
    with (tmp_path / "other" / "basedata.har").open("ab") as f:
        f.write(b"\0")
    assert not gates.is_nus333_altax_reference(tmp_path / "other", sl4)
    assert gates.waivers_for(tmp_path / "other", sl4) is gates.NO_WAIVERS
    assert not gates.waivers_for(tmp_path / "missing", sl4).reference


def test_other_experiment_gets_no_waivers(sl4):
    to = sl4.changes["to"]
    other = dataclasses.replace(to, values=to.values * 2.0)
    changes = dict(sl4.changes) | {"to": other}
    fake = dataclasses.replace(sl4, changes=changes)
    assert gates.waivers_for(nus333_dir(), fake) is gates.NO_WAIVERS


def test_unwaived_gates_pass(altax_run, sl4):
    """At exact sigma = 1 the pct and decomp gates pass with no waivers, and the summary
    checks pass by the ulp rule that every reference shares."""
    ours_pct, gem_pct = pct_changes(altax_run), gempack_changes(sl4)
    off = gates.report_pct(altax_run, ours_pct, gem_pct, gates.NO_WAIVERS)
    assert off["pass"] and not off["failing"] and not off["known_miss"]
    assert not gates.NUS333_ALTAX_WAIVERS.pct

    ours_d = decomp_headers(altax_run)
    gem_d = read_har(str(FIXTURES / "decomp.har"))
    assert gates.report_decomp(ours_d, gem_d, gates.NO_WAIVERS)["pass"]
    assert not gates.NUS333_ALTAX_WAIVERS.decomp

    ours_s = summary_headers(altax_run)
    gem_s = read_har(str(FIXTURES / "summary.har"))
    flows = (
        check_flows(input_database(altax_run)),
        check_flows(post_database(altax_run)),
    )
    assert gates.report_summary(ours_s, gem_s, flows=flows)["pass"]  # no waivers at all


def test_a_waiver_passes_only_a_bounded_miss_of_its_own_sign(altax_run, sl4):
    """The mechanism, on a synthetic miss: one cell moved off GEMPACK fails unwaived, passes
    with its measured error, and fails again if the error grows or flips sign."""
    ours, gem = pct_changes(altax_run), gempack_changes(sl4)
    var, cell = next(
        (n, k)
        for n, cells in gem.items()
        if n in ours
        for k, g in cells.items()
        if abs(g) > 1e-3
    )
    moved = {n: dict(c) for n, c in ours.items()}
    moved[var][cell] = gem[var][cell] + 0.1 * abs(gem[var][cell])
    off = gates.report_pct(altax_run, moved, gem, gates.NO_WAIVERS)
    assert not off["pass"] and len(off["failing"]) == 1
    err = gates.pct_failures(moved, gem, gates.ZERO_BASE)[(var, cell)]

    def waived(ref):
        w = dataclasses.replace(gates.NUS333_ALTAX_WAIVERS, pct={(var, cell): ref})
        return gates.report_pct(altax_run, moved, gem, w)

    on = waived(err)
    assert on["pass"] and len(on["known_miss"]) == 1 and not on["failing"]
    assert not waived(err / 2)["pass"]  # grew beyond 1.5x
    assert not waived(-err)["pass"]  # sign flipped
