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


def _make_block_mp(p):
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel

    rr = list(p.sets.r)[-1]
    return GTAPBlockMultiPeriodModel(p.sets, p, _closure(p), residual_region=rr)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_block_path_byte_identical():
    p = _load_params()
    cur = model_signature(_build_current(_make_block_mp(p)))
    new = model_signature(_build_new(_make_block_mp(p)))
    assert new == cur, f"HARD GATE (block): new {new} != current {cur}. STOP."


# --- Solve parity: build both ways, seed + solve, compare Python solutions --- #

_ROOT = Path(__file__).resolve().parents[3]
_GDX_10x7 = _ROOT / "tests/fixtures/gtap7/gtap7_10x7/out_gtap_shock_ifsub0.gdx"


def _seed_and_solve(build_fn):
    """Build the monolith 10x7 model via build_fn(mp), seed from the pure-gtap
    reference GDX, solve base->check->shock, return {name+index: value}."""
    import sys

    sys.path.insert(0, str(_ROOT / "scripts/gtap"))
    from pyomo.environ import Var
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_params()
    mp = _make_mp(p)
    m = build_fn(mp)
    rr = list(p.sets.r)[-1]
    m._residual_region = rr
    mp.seed_all_periods(m, str(_GDX_10x7))
    solve_multiperiod(
        m,
        p,
        _closure(p),
        ref_gdx=str(_GDX_10x7),
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode="gtap",
    )
    out = {}
    for v in m.component_objects(Var, active=True):
        for idx in v:
            try:
                out[f"{v.name}{idx}"] = float(V(v[idx]))
            except Exception:
                pass
    return out


@pytest.mark.skipif(
    not (_GDX_10x7.exists() and DATA.exists()),
    reason="gtap7_10x7 dataset or reference GDX not present",
)
def test_solve_parity_10x7():
    sol_cur = _seed_and_solve(_build_current)
    sol_new = _seed_and_solve(_build_new)
    shared = set(sol_cur) & set(sol_new)
    assert shared, "no shared var keys — build paths produced different var names"
    worst_rel, worst_key = 0.0, None
    for k in shared:
        rel = abs(sol_new[k] - sol_cur[k]) / (abs(sol_cur[k]) + 1e-12)
        if rel > worst_rel:
            worst_rel, worst_key = rel, k
    assert worst_rel < 1e-8, f"solve diverged at {worst_key}: rel={worst_rel:.2e}"
