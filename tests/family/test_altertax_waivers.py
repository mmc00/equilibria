"""The known-miss waivers apply to the nus333 ALTAX reference only, never by key alone."""

from __future__ import annotations

import dataclasses
import gzip
import shutil

import pytest

from equilibria._local_refs import nus333_dir
from equilibria.babel.har.reader import read_har
from equilibria.family.altertax import gates
from equilibria.family.altertax.decomp import decomp_headers
from equilibria.family.altertax.pct import gempack_changes, pct_changes
from equilibria.family.altertax.summary import summary_headers
from tests.family._prep import ROOT

FIXTURES = ROOT / "tests/fixtures/gtap7_gempack/burfisher_gragg"


@pytest.fixture(scope="module")
def sl4(tmp_path_factory):
    d = tmp_path_factory.mktemp("sl4")
    with (
        gzip.open(FIXTURES / "altax.sl4.gz", "rb") as a,
        (d / "solution.sl4").open("wb") as b,
    ):
        shutil.copyfileobj(a, b)
    return gates.read_sl4(d / "solution.sl4")


def test_nus333_altax_reference_is_recognised(sl4):
    assert gates.is_nus333_altax_reference(nus333_dir(), sl4)
    w = gates.waivers_for(nus333_dir(), sl4)
    assert w is gates.NUS333_ALTAX_WAIVERS and w.reference == gates.REFERENCE_ID
    assert w.pct and w.decomp and w.summary


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


def test_without_waivers_the_known_misses_fail(altax_run, sl4):
    """The same outputs and the same reference: waived they pass, unwaived they fail."""
    gem_dir = FIXTURES
    ours_pct, gem_pct = pct_changes(altax_run), gempack_changes(sl4)
    on = gates.report_pct(altax_run, ours_pct, gem_pct, gates.NUS333_ALTAX_WAIVERS)
    off = gates.report_pct(altax_run, ours_pct, gem_pct, gates.NO_WAIVERS)
    assert on["pass"] and not on["failing"] and on["known_miss"]
    assert not off["pass"] and not off["known_miss"]
    assert len(off["failing"]) == len(on["known_miss"])

    ours_d = decomp_headers(altax_run)
    gem_d = read_har(str(gem_dir / "altax_decomp.har"))
    assert gates.report_decomp(ours_d, gem_d, gates.NUS333_ALTAX_WAIVERS)["pass"]
    assert not gates.report_decomp(ours_d, gem_d, gates.NO_WAIVERS)["pass"]

    ours_s = summary_headers(altax_run)
    gem_s = read_har(str(gem_dir / "altax_summary.har"))
    assert gates.report_summary(ours_s, gem_s, gates.NUS333_ALTAX_WAIVERS)["pass"]
    off_s = gates.report_summary(ours_s, gem_s, gates.NO_WAIVERS)
    assert not off_s["pass"] and not off_s["known_miss"]
