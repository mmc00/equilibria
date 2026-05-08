"""Smoke tests for the HAR → equilibria wrapper.

Skipped automatically when the GTAP Standard 7 reference dataset is not
available locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.babel.har import load_gtap_from_har

GTAP_DIR = Path("/Users/marmol/proyectos2/cge_babel/standard_gtap_7")
SUFFIX = "-9x10"

pytestmark = pytest.mark.skipif(
    not (GTAP_DIR / f"basedata{SUFFIX}.har").is_file(),
    reason="GTAP 9x10 HAR dataset not available locally",
)


def test_load_gtap_from_har_populates_sets_and_benchmark():
    p = load_gtap_from_har(GTAP_DIR, suffix=SUFFIX)
    assert len(p.sets.r) == 10
    assert len(p.sets.i) == 10
    assert len(p.sets.a) == 9
    assert p.sets.structure in {"non_diagonal", "multi_output"}
    assert len(p.sets.output_pairs) >= len(p.sets.a)


def test_load_gtap_from_har_calibrates_shares():
    p = load_gtap_from_har(GTAP_DIR, suffix=SUFFIX)
    assert len(p.shares.p_gx) > 0
    assert len(p.calibrated.gx_param) > 0
    assert all(c in p.sets.i for (_r, _a, c) in p.benchmark.makb)


def test_load_gtap_from_har_vom_indexed_by_activity():
    p = load_gtap_from_har(GTAP_DIR, suffix=SUFFIX)
    sample_keys = list(p.benchmark.vom.keys())[:5]
    for r, a in sample_keys:
        assert r in p.sets.r
        assert a in p.sets.a


def test_load_gtap_from_har_builds_pyomo_model():
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = load_gtap_from_har(GTAP_DIR, suffix=SUFFIX)
    m = GTAPModelEquations(p.sets, p).build_model()
    assert hasattr(m, "gx_param")
    assert hasattr(m, "eq_xweq")
    assert hasattr(m, "eq_pmcifeq")


def test_load_gtap_from_har_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_gtap_from_har("/nonexistent/path/zzz")
