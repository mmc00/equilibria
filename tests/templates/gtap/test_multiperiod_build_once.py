"""Equivalence + parity tests for the multi-period build-once refactor.

The HARD GATE: the model built with build_equations_all_periods must be
byte-identical (same active-Constraint count AND same sha256 over sorted
`name|str(expr)`) to the model built with the current per-period loop.
Baseline captured on gtap7_10x7: 37505 constraints, hash ec3e426d49a094cb.
"""

from __future__ import annotations

import hashlib
import re
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
# Bumped from (37505, ec3e426d49a094cb) when build_equations_fisher started naming its
# wide cross-period sums with auxiliary variables (+133 defining rows, and the four
# Fisher rows now read those variables instead of the sums). The algebra is unchanged;
# what this constant pins is that the two BUILD PATHS agree, which
# test_new_path_byte_identical_to_current checks directly and independently of it.
BASELINE_COUNT = 37638
# Hash of the signature with float literals rounded to _SIG_FIGS (see
# model_signature). The raw full-repr hash is NOT portable: the calibrated
# coefficients come out of pow() with fractional exponents (alphad =
# (xda/xaa)*(pdp/pa)**sigma_m), and libm's pow is not correctly rounded, so it
# legitimately differs by 1-2 ULP across interpreters and platforms.
# MEASURED on this very model, all three agreeing on the model itself:
#   py3.12 + numpy 2.5.1 + pyomo 6.10.1 (mac)  -> a9b1a0e0f86e89be
#   py3.12 + numpy 2.4.6 + pyomo 6.9.5  (mac)  -> a9b1a0e0f86e89be
#   py3.11 + numpy 2.4.6 + pyomo 6.9.5  (mac)  -> 83957d87e97e95e0
#   py3.11 on CI (linux)                       -> 2822758f1a440410
# 3462 of the 37638 rows differed, every one of them by 1-2 ULP (relative
# ~2e-16 = double epsilon) — i.e. the same model, printed differently. The
# project supports python >=3.10, so a full-repr constant can only ever be
# right on one build. Rounding to 13 significant figures is stable across all
# of the above while still catching a 1e-12 relative change (verified), which
# is ~4000x finer than the noise it absorbs.
BASELINE_HASH = "e45fc91c742347f4"


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


# Significant figures kept from every float literal before hashing. 13 is the
# tightest value that is stable across the interpreters/platforms measured
# above (14 and 15 still differ); it leaves ~3 orders of magnitude of margin
# over the 1-2 ULP noise it is there to absorb.
_SIG_FIGS = 13
_FLOAT_LITERAL = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _round_float_literals(text: str, sig: int = _SIG_FIGS) -> str:
    """Round every float literal in an expression string to `sig` figures."""

    def _round(match: re.Match[str]) -> str:
        return f"{float(match.group(0)):.{sig}g}"

    return _FLOAT_LITERAL.sub(_round, text)


def model_signature(m):
    """(active constraint count, hash of the rounded expression strings).

    The rounding is what makes this comparable across machines; see the note on
    BASELINE_HASH. It is not a loosened tolerance on the MODEL: a real change to
    any coefficient, or any change in structure, still changes the hash.
    """
    cons = list(m.component_data_objects(Constraint, active=True))
    sigs = sorted(_round_float_literals(f"{c.name}|{c.expr}") for c in cons)
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


def test_rounding_absorbs_ulp_noise_but_not_real_changes():
    """The signature must ignore last-bit noise and nothing more.

    Without this, the previous full-repr baseline made the gate fail on CI for a
    model that was byte-identical in every way that matters: 3462 rows differed
    by 1-2 ULP because libm's pow() is not correctly rounded across platforms.
    The danger in the fix is the opposite one — rounding so hard that a real
    coefficient change slips through — so both directions are pinned here.
    """
    base = "eq_paa[CHN,Chem,inv,base]|0.4839302588982945*pd  ==  0.0"

    # 1 ULP apart: the real difference measured between py3.11 and py3.12.
    ulp = "eq_paa[CHN,Chem,inv,base]|0.4839302588982944*pd  ==  0.0"
    assert _round_float_literals(base) == _round_float_literals(ulp)

    # A 1e-12 relative change is ~4000x larger than that noise: must survive.
    real = "eq_paa[CHN,Chem,inv,base]|0.4839302594000000*pd  ==  0.0"
    assert _round_float_literals(base) != _round_float_literals(real)

    # And any structural change obviously must.
    extra = "eq_paa[CHN,Chem,inv,base]|0.4839302588982945*pd + 0.1*x  ==  0.0"
    assert _round_float_literals(base) != _round_float_literals(extra)


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
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
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
