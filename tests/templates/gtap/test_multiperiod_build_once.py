"""Equivalence + parity tests for the multi-period build-once refactor.

The HARD GATE: the model built with build_equations_all_periods must be
byte-identical (same active-Constraint count AND same sha256 over sorted
`name|str(expr)`) to the model built with the current per-period loop.
Baseline captured on gtap7_10x7: 37505 constraints, hash ec3e426d49a094cb.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pyomo.environ import Constraint

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_model_multiperiod import (
    PERIODS,
    GTAPMultiPeriodModel,
)

DATA = Path("datasets/gtap7_10x7")
BASELINE_COUNT = 37505
BASELINE_HASH = "ec3e426d49a094cb"


def _load_params():
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATA / "basedata.har",
        sets_path=DATA / "sets.har",
        default_path=DATA / "default.prm",
        baserate_path=DATA / "baserate.har",
    )
    return p


def _closure(p):
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )


def _make_mp(p):
    rr = list(p.sets.r)[-1]
    return GTAPMultiPeriodModel(p.sets, p, _closure(p), residual_region=rr)


def model_signature(m):
    cons = list(m.component_data_objects(Constraint, active=True))
    sigs = sorted(f"{c.name}|{c.expr}" for c in cons)
    h = hashlib.sha256("\n".join(sigs).encode()).hexdigest()[:16]
    return len(cons), h


def _build_current(mp):
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    return m


def _build_new(mp):
    m = mp.build_sets()
    mp.build_vars(m)
    mp.build_equations_all_periods(m)
    mp.build_equations_fisher(m)
    return m


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_current_path_matches_baseline():
    p = _load_params()
    mp = _make_mp(p)
    m = _build_current(mp)
    count, h = model_signature(m)
    assert (count, h) == (BASELINE_COUNT, BASELINE_HASH), (
        f"baseline drifted: got ({count}, {h}); "
        "if this fails on unchanged code, the equivalence gate constant is stale"
    )


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_new_path_byte_identical_to_current():
    p = _load_params()
    cur = model_signature(_build_current(_make_mp(p)))
    new = model_signature(_build_new(_make_mp(p)))
    assert new == cur, (
        f"HARD GATE FAILED: new path {new} != current {cur}. "
        "The build-once refactor changed the model. STOP."
    )
