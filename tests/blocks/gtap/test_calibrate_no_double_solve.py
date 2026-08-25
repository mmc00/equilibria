"""Lever A — eliminate calibrate_base's double solve.

HARD GATE: settle_only (cut the settle at check) must produce a settled_seed
IDENTICAL to the full base->check->shock settle. Baseline captured on gtap7_10x7.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import pytest

from equilibria.blocks.gtap.factor import FactorBlock
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

DATA = Path("datasets/gtap7_10x7")
# Full base->check->shock settle baseline (captured Task 0, gtap7_10x7).
# settle_only must reproduce this EXACTLY.
BASELINE_COUNT = 20138
BASELINE_SIG = "ded5651b6133f788"


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
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, savf_flag="capFix", numeraire="pnum",
    )


def _seed_signature(seed):
    parts = []
    for name, cells in seed.items():
        for body, val in cells.items():
            parts.append(f"{name}|{body}|{float(val):.10e}")
    parts.sort()
    h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return len(parts), h


def _calibrate(**kw):
    p = _load_params()
    rr = list(p.sets.r)[-1]
    fb = FactorBlock(sets=p.sets, params=p)
    return fb.calibrate_base(p, p.sets, _closure(p), rr, ref_gdx=None, **kw)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_full_settle_baseline():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    seed = _calibrate()
    count, h = _seed_signature(seed)
    assert count > 0, "settle produced an empty seed"
    print(f"BASELINE settled_seed: {count} cells, sig={h}")


def _solve_multiperiod_result(settle_only):
    """Build the block model + solve, return the results dict (base/check/[shock])."""
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        solve_block_model,
    )

    p = _load_params()
    rr = list(p.sets.r)[-1]
    m, mp = build_block_model(p, p.sets, _closure(p), rr)
    return solve_block_model(m, p, _closure(p), None, mode="gtap", settle_only=settle_only)


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_settle_only_skips_shock():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    res = _solve_multiperiod_result(settle_only=True)
    assert "check" in res, "settle_only must still solve the check phase"
    assert "shock" not in res, "settle_only must NOT solve the shock phase"


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_settle_only_seed_identical_to_full():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    # calibrate_base's seed must equal the full-settle baseline (Task 0 constants).
    # Before Step 3 this is trivially true (calibrate_base is still full-settle);
    # after Step 3 (calibrate_base uses settle_only) it is the byte-identical gate.
    sig = _seed_signature(_calibrate())
    assert sig == (BASELINE_COUNT, BASELINE_SIG), (
        f"HARD GATE: calibrate_base seed {sig} != full-settle baseline "
        f"({BASELINE_COUNT}, {BASELINE_SIG}). The cut changed the seed. STOP."
    )


@contextlib.contextmanager
def _monkey(obj, name, new):
    orig = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, orig)


def _sparse_solution(force_full_settle):
    from pyomo.environ import Var
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import solve_block_model as _sbm
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    p = _load_params()
    rr = list(p.sets.r)[-1]
    patch = contextlib.ExitStack()
    if force_full_settle:
        # calibrate_base imports solve_block_model LOCALLY from gtap_block_model, so
        # patch it at the source module (not factor). Force settle_only=False so the
        # settle runs the full base->check->shock stack (the pre-lever-A behavior).
        import equilibria.templates.gtap.gtap_block_model as _bm

        def _full(m, params, closure, ref_gdx, *, mode="gtap", settle_only=False):
            return _sbm(m, params, closure, ref_gdx, mode=mode, settle_only=False)

        patch.enter_context(_monkey(_bm, "solve_block_model", _full))
    with patch:
        m, mp, _ = build_sparse_model_mp(p, p.sets, _closure(p), rr, base_calibrated=True)
        solve_multiperiod(
            m, p, _closure(p), ref_gdx=None, skip_base_solve=True, mute_welfare=True,
            seed_from_prior=False, holdfix_cd=True, mode="gtap",
        )
    out = {}
    for v in m.component_objects(Var, active=True):
        for idx in v:
            with contextlib.suppress(Exception):
                out[f"{v.name}{idx}"] = float(V(v[idx]))
    return out


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_end_to_end_solve_parity():
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    sol_cut = _sparse_solution(force_full_settle=False)
    sol_full = _sparse_solution(force_full_settle=True)
    shared = set(sol_cut) & set(sol_full)
    assert shared, "no shared var keys"
    worst, key = 0.0, None
    for k in shared:
        rel = abs(sol_cut[k] - sol_full[k]) / (abs(sol_full[k]) + 1e-12)
        if rel > worst:
            worst, key = rel, k
    assert worst < 1e-8, f"final solve diverged at {key}: rel={worst:.2e}"


# ---------------------------- disk cache ---------------------------- #


def test_seed_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    from equilibria.blocks.gtap import seed_cache

    seed = {"pf": {("USA", "Land", "Food"): 1.25, ("EU", "Land", "Food"): 0.9},
            "kstock": {"USA": 42.0}}
    key = "k-abc123"
    assert seed_cache.load(key) is None
    seed_cache.save(key, seed)
    got = seed_cache.load(key)
    assert got == seed, f"roundtrip mismatch: {got} != {seed}"


def test_seed_cache_key_changes_with_input():
    from equilibria.blocks.gtap import seed_cache

    p = _load_params()
    rr = list(p.sets.r)[-1]
    c1 = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
                           fix_endowments=False, fix_taxes=False, fix_technology=False,
                           if_sub=False, savf_flag="capFix", numeraire="pnum")
    c2 = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
                           fix_endowments=False, fix_taxes=False, fix_technology=False,
                           if_sub=True, savf_flag="capFix", numeraire="pnum")  # if_sub differs
    k1 = seed_cache.cache_key("gtap7_10x7", c1, rr, p)
    k2 = seed_cache.cache_key("gtap7_10x7", c2, rr, p)
    assert k1 != k2, "cache key must change when a settle-affecting input changes"


def test_seed_cache_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    from equilibria.blocks.gtap import seed_cache

    assert seed_cache.disabled() is True
    seed_cache.save("k-x", {"pf": {("USA",): 1.0}})       # must write nothing
    assert list(tmp_path.iterdir()) == [], "disabled cache still wrote a file"
    assert seed_cache.load("k-x") is None, "disabled cache still read"


@pytest.mark.skipif(not DATA.exists(), reason="gtap7_10x7 dataset not present")
def test_cache_hit_skips_settle(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    seed1 = _calibrate()                 # miss → computes + writes
    # calibrate_base imports build_block_model LOCALLY from gtap_block_model, so
    # spy at the source module (patching factor would never see the call).
    import equilibria.templates.gtap.gtap_block_model as _bm
    called = {"n": 0}
    orig = _bm.build_block_model

    def _spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_bm, "build_block_model", _spy)
    seed2 = _calibrate()                 # hit → must NOT build/solve
    assert called["n"] == 0, "cache hit still built the model"
    assert _seed_signature(seed1) == _seed_signature(seed2), "cached seed differs"
