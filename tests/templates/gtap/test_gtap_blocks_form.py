"""Aggregate form + domain gate: GTAP block units vs the monolith oracle (F3 Task 4).

Builds the gtap7_3x3 monolith (``if_sub=False`` comp-stat form, so every defining
equation is active) as the parity ORACLE, composes the migrated block units onto a
single equilibria Model, translates that to Pyomo, and asserts for every migrated
unit:

  * ``form_diff`` clean  — each block-built constraint cell's expanded Pyomo
    expression string-matches the monolith's (structural, cross-model — see
    ``blocks_diag._exprs_equal``).
  * ``domain_bounds_diff`` clean on the unit's OWNED vars — domain label + bounds
    tuple identical to the oracle (per-cell price floors included).

Units land incrementally; ``_MIGRATED`` lists those whose block exists and is gated.
As units 3-7 land, extend ``_MIGRATED`` (and any upstream stub they still need).

No solve — form + domain only. gd_share/ge_share drift (Blocker-C carry) is inert
on gtap7_3x3 (all omegax=inf, so gd/ge never enter an active Leontief body).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts" / "gtap"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# (module_attr, upstream stub var names it references but does not own)
_MIGRATED: list[tuple[str, list[str]]] = [
    ("TradeCETBlock", ["xds", "xs", "pd", "ps"]),
    ("ProductionSupplyBlock", ["pd", "ps", "pa", "pfa"]),
    ("FactorBlock", ["pd", "ps", "pa", "pva", "va", "pabs"]),
    (
        "ArmingtonBilateralBlock",
        [
            "pd",
            "ps",
            "pva",
            "va",
            "pabs",
            "nd",
            "pnd",
            "xc",
            "xg",
            "xi",
            "pet",
            "xet",
            "xds",
            "pnum",
        ],
    ),
    (
        "DemandUtilityBlock",
        # references but does not own: pa (ARMINGTON), pf/xf/kstock/kapEnd/arent/
        # rorc/rore (FACTOR — dedup), income vars yc/yg/yi/regy/rsav (INCOME).
        ["pa", "yc", "yg", "yi", "regy", "rsav"],
    ),
    (
        "IncomeBlock",
        # references but does not own — all owned by migrated units 1-5:
        # pf/xf/kstock (FACTOR), pi/phi/phip/uh/savf (DEMAND), p_rai/x/pd (PROD),
        # pmt/pe/xw/pmcif/pefob/pa/xaa/xda/xma (ARMINGTON). No new stubs needed.
        [],
    ),
]

# Upstream shared vars a leaf unit references but a later unit owns — stubbed so
# the block builds standalone. (domains, domain_label). Not gated for domain.
_STUBS: dict[str, tuple[tuple[str, ...], str]] = {
    "xds": (("r", "i"), "NonNegativeReals"),
    "xs": (("r", "i"), "NonNegativeReals"),
    "pd": (("r", "i"), "NonNegativeReals"),
    "ps": (("r", "i"), "NonNegativeReals"),
    "pa": (("r", "i", "aa"), "NonNegativeReals"),
    "pfa": (("r", "f", "a"), "NonNegativeReals"),
    "pva": (("r", "a"), "NonNegativeReals"),
    "va": (("r", "a"), "NonNegativeReals"),
    "pabs": (("r",), "NonNegativeReals"),
    "nd": (("r", "a"), "NonNegativeReals"),
    "pnd": (("r", "a"), "NonNegativeReals"),
    "xc": (("r", "i"), "NonNegativeReals"),
    "xg": (("r", "i"), "NonNegativeReals"),
    "xi": (("r", "i"), "NonNegativeReals"),
    "pet": (("r", "i"), "NonNegativeReals"),
    "xet": (("r", "i"), "Reals"),
    "pnum": ((), "NonNegativeReals"),
    # INCOME vars (unit 6) — Reals FREE (monolith 4722-4749).
    "yc": (("r",), "Reals"),
    "yg": (("r",), "Reals"),
    "yi": (("r",), "Reals"),
    "regy": (("r",), "Reals"),
    "rsav": (("r",), "Reals"),
}

# Per-unit domain-gate exceptions: owned-var cells whose bound differs ONLY by a
# documented post-scaling-snapshot / share-drift carry (composer owns the value),
# NOT a port error. Each entry: var name -> reason (for the trace). These cells
# are excluded from the domain assertion with the reason recorded here.
_DOMAIN_CARRY_VARS: dict[str, str] = {
    # FACTOR: kapEnd's runtime relative floor (1e-3*init) reflects the monolith's
    # POST-apply_production_scaling re-value kapEnd=(1-depr)*kstock+xiagg with
    # xiagg=yi/pi (income-side), gtap_model_equations.py:1268. The block seeds
    # from the benchmark xi_bench, so the floor differs at ~1e-8 relative on the 3
    # kapEnd cells. The composer re-values + re-floors kapEnd in Task 5 (same
    # snapshot family as pf0/xf0/mqfactr_bb). Form is 0/243 clean.
    "kapEnd": "post-apply_production_scaling xiagg=yi/pi re-value (composer carry)",
    # ARMINGTON+BILATERAL: pm/pmcif/pefob price floors (1e-3*init) reflect the
    # monolith's POST-apply_production_scaling re-value of the bilateral price vars
    # (pm≈1.05, pmcif/pefob≈1.001..1.07), gtap_model_equations.py:5383-5385. Block
    # seeds these =1.0 -> floor 1e-3. Same post-scaling snapshot family (composer
    # re-floors). Structure (index-set equality) is clean.
    "pm": "post-apply_production_scaling price re-value floor (composer carry)",
    "pmcif": "post-apply_production_scaling price re-value floor (composer carry)",
    "pefob": "post-apply_production_scaling price re-value floor (composer carry)",
    # DEMAND+UTILITY: xigbl/rorg floors (1e-3*init) reflect the monolith's POST-
    # apply_production_scaling re-value. xigbl = sum_r net-investment is re-valued
    # by _align_xi_xaa_post_scaling (xiagg->income-side) so its floor drifts ~2e-8
    # relative from the block's benchmark seed. rorg is seeded 1.0 (floor 1e-3)
    # then re-valued to ~3.9 by the kapEnd/rore rate-of-return chain (floor
    # 1e-3*3.9=0.0039 > block 1e-3). Same post-scaling snapshot family (composer
    # re-values + re-floors, checklist item 7). Form is clean; structure identical.
    "xigbl": "post-apply_production_scaling xiagg re-value floor (composer carry)",
    "rorg": "post-apply_production_scaling rate-of-return re-value floor (composer carry)",
}

# Per-unit FORM-gate carry: (unit, eq) pairs whose per-cell diff is ONLY a
# coefficient-value difference at the share-seed vs post-scaling boundary
# (Blocker C, pre-adjudicated). The cell's EXPRESSION STRUCTURE must be identical
# (verified below by stripping numeric literals) and every numeric-literal pair
# must agree within _SHARE_DRIFT_RTOL — a structural re-shape would still fail.
_FORM_CARRY_EQS: dict[tuple[str, str], str] = {
    ("ArmingtonBilateralBlock", "eq_paa"): "top-Armington alphad/alpham seed drift",
    ("ArmingtonBilateralBlock", "eq_xda"): "top-Armington alphad seed drift",
    ("ArmingtonBilateralBlock", "eq_xma"): "top-Armington alpham seed drift",
    # INCOME eq_pabs folds base_xaa (the Fisher base-period absorption quantity).
    # The 6 inv-agent xaa cells that _align_xi_xaa_post_scaling nudges (~1e-7)
    # make the base_pa*base_xaa coefficient drift from the block's benchmark seed
    # by <1e-6 relative — same Blocker-C composer-carry family (composer
    # re-snapshots base_xaa post-scaling). Structure is byte-identical.
    ("IncomeBlock", "eq_pabs"): "Fisher base_xaa post-align seed drift",
}
_SHARE_DRIFT_RTOL = 1e-6


def _oracle():
    from _parity_datasets import build_models

    m_b, _m_s = build_models("gtap7_3x3", close=False)
    return m_b


def _params_sets():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    d = ROOT / "datasets" / "gtap7_3x3"
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p, p.sets


# The gtap7_3x3 comp-stat oracle (_build_compstat_har) sets residual_region to
# the LAST region (list(p.sets.r)[-1] == "ROW"). Units 5-7 gate the residual-
# region Skip masks (eq_savf/eq_yi/eq_walras) and savf_bar rebalance against it,
# so they must be built with the same residual region as the oracle.
_RESIDUAL_REGION = "ROW"


def _mk_unit(cls, sets, params):
    """Construct a block, threading residual_region for the demand/income/closure
    units (whose Skip masks depend on it). Other units ignore the kwarg."""
    if "residual_region" in getattr(cls, "model_fields", {}):
        return cls(sets=sets, params=params, residual_region=_RESIDUAL_REGION)
    return cls(sets=sets, params=params)


def _set_elems(sets: Any) -> dict[str, list[str]]:
    agents = ["hhd", "gov", "inv", "tmg"]
    return {
        "r": list(sets.r),
        "i": list(sets.i),
        "a": list(sets.a),
        "f": list(sets.f),
        "mf": list(sets.mf),
        "sf": list(sets.sf),
        "m": list(sets.m),
        "marg": list(sets.marg),
        "aa": list(sets.a) + agents,
        "rp": list(sets.r),
        "gy": ["pt", "fc", "pc", "gc", "ic", "dt", "mt", "et", "ft", "fs"],
    }


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _is_coefficient_only_drift(block_str: str, mono_str: str, rtol: float) -> bool:
    """True if two expr strings differ ONLY in numeric literals within ``rtol``.

    Strips every numeric literal to a placeholder and requires the remaining
    (structural) token stream to be IDENTICAL, then checks each numeric-literal
    pair agrees to a relative tolerance. A genuine algebraic RE-SHAPE changes the
    non-numeric skeleton -> not a coefficient-only drift (fails, as it should).
    """
    if _NUM_RE.sub("#", block_str) != _NUM_RE.sub("#", mono_str):
        return False
    b_nums = [float(x) for x in _NUM_RE.findall(block_str)]
    m_nums = [float(x) for x in _NUM_RE.findall(mono_str)]
    if len(b_nums) != len(m_nums):
        return False
    for b, m in zip(b_nums, m_nums, strict=True):
        denom = max(abs(b), abs(m), 1e-12)
        if abs(b - m) / denom > rtol:
            return False
    return True


def _is_post_scaling_floor_carry(a_dom, b_dom, a_bnd, b_bnd) -> bool:
    """True if a domain diff is ONLY a post-scaling floor/level re-value.

    Two admissible shapes, both from the composer's post-apply_production_scaling
    re-value of a price/level var (block seeds the benchmark, floors 1e-3*seed):
      * a RAISED floor  — oracle lower >= block lower (price vars pm/pmcif/pefob,
        whose post-scaling level > 1 lifts the 1e-3*level floor above 1e-3); or
      * a tiny two-way re-VALUE drift within 1e-6 relative (kapEnd = (1-depr)*
        kstock + xiagg with xiagg=yi/pi vs the benchmark xi_bench).
    Requires same domain label and same upper bound throughout — a dropped/
    loosened bound, a domain-label change, or a large floor mismatch is a real
    port bug and still fails.
    """
    if a_dom != b_dom:
        return False
    a_lo, a_hi = a_bnd
    b_lo, b_hi = b_bnd
    if a_hi != b_hi:
        return False
    if a_lo is None or b_lo is None:
        return False
    if b_lo >= a_lo:
        return True
    denom = max(abs(a_lo), abs(b_lo), 1e-12)
    return abs(a_lo - b_lo) / denom <= 1e-6


def _build_block_model(block_classes, p, sets, stub_names):
    from equilibria.backends.pyomo_backend import PyomoBackend
    from equilibria.core.sets import Set as ESet
    from equilibria.core.variables import Variable
    from equilibria.model import Model

    setmap = _set_elems(sets)
    model = Model(name="gtap_blocks")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=elems))
    for cls in block_classes:
        model.add_block(_mk_unit(cls, sets, p))
    for n in stub_names:
        if n in model.variable_manager:
            continue
        doms, dlabel = _STUBS[n]
        shape = tuple(len(setmap[d]) for d in doms)
        # A scalar stub (empty domains) needs a length-1 array (the bridge's
        # scalar path calls len(var.value)); a 0-D np.ones(()) would break it.
        val = np.ones(shape) if shape else np.array([1.0])
        model.add_variable(
            Variable(
                name=n,
                value=val,
                domains=doms,
                domain=dlabel,
                lower=1e-3 if dlabel == "NonNegativeReals" else float("-inf"),
                upper=float("inf"),
            )
        )
    backend = PyomoBackend()
    backend.build(model)
    return backend.pyomo_model, model


@pytest.fixture(scope="module")
def _fixtures():
    import equilibria.blocks.gtap as gtap_blocks

    p, sets = _params_sets()
    oracle = _oracle()
    classes = [getattr(gtap_blocks, name) for name, _ in _MIGRATED]
    stubs = sorted({s for _, sub in _MIGRATED for s in sub})
    bm, model = _build_block_model(classes, p, sets, stubs)
    return p, sets, oracle, bm, model, gtap_blocks


@pytest.mark.parametrize("unit_name", [n for n, _ in _MIGRATED])
def test_gtap_block_form_matches_monolith(_fixtures, unit_name):
    from blocks_diag import form_diff  # ty: ignore[unresolved-import]
    from pyomo.environ import Constraint

    _p, _sets, oracle, bm, _model, gtap_blocks = _fixtures
    unit = getattr(gtap_blocks, unit_name)
    # discover the equations this unit contributes (its build_expression names)
    eqs = _mk_unit(unit, _sets, _p).setup(_ScratchSM(_sets), {}, {})
    eq_names = {e.name for e in eqs}

    bm_cons = {c.name: c for c in bm.component_objects(Constraint, active=True)}
    or_cons = {c.name: c for c in oracle.component_objects(Constraint, active=True)}

    checked = 0
    for eq in sorted(eq_names):
        con_name = f"{eq}_con"
        assert con_name in bm_cons, f"{unit_name}: {con_name} missing from block model"
        assert eq in or_cons, f"{unit_name}: {eq} missing from oracle"
        bc, oc = bm_cons[con_name], or_cons[eq]
        # Minor 1 (index-set equality): form_diff only compares the INTERSECTION
        # of block/oracle index sets, so a block that UNDER-generates cells (wrong
        # Skip mask, missing 3-D/4-D combos) would pass silently. Assert the active
        # index sets are EQUAL — zero only_block AND zero only_oracle cells.
        b_idx, o_idx = set(bc.keys()), set(oc.keys())
        only_block = b_idx - o_idx
        only_oracle = o_idx - b_idx
        assert only_oracle == set(), (
            f"{unit_name} {eq}: block UNDER-generates {len(only_oracle)} active "
            f"cell(s) the oracle has; sample {sorted(only_oracle)[:3]}"
        )
        assert only_block == set(), (
            f"{unit_name} {eq}: block OVER-generates {len(only_block)} active "
            f"cell(s) the oracle skips; sample {sorted(only_block)[:3]}"
        )
        diffs = form_diff(bc, oc)
        carry_reason = _FORM_CARRY_EQS.get((unit_name, eq))
        if carry_reason is not None:
            # Blocker-C share-drift carry: every diff on this eq must be a
            # coefficient-only difference within tolerance (structure identical).
            # A structural re-shape would NOT satisfy _is_coefficient_only_drift
            # and would fail here — so the exception cannot mask a real port bug.
            non_drift = [
                d
                for d in diffs
                if not _is_coefficient_only_drift(d[1], d[2], _SHARE_DRIFT_RTOL)
            ]
            assert non_drift == [], (
                f"{unit_name} {eq}: NON-share-drift form diff (structural or "
                f">rtol): {non_drift[0]}"
            )
        else:
            assert diffs == [], (
                f"{unit_name} {eq}: {len(diffs)} form diff(s); first: {diffs[0]}"
            )
        checked += 1
    assert checked == len(eq_names)


@pytest.mark.parametrize("unit_name, sub", _MIGRATED)
def test_gtap_block_domain_matches_monolith(_fixtures, unit_name, sub):
    from blocks_diag import domain_bounds_diff  # ty: ignore[unresolved-import]

    _p, _sets, oracle, bm, _model, gtap_blocks = _fixtures
    unit = getattr(gtap_blocks, unit_name)
    owned: dict[str, Any] = {}
    _mk_unit(unit, _sets, _p).setup(_ScratchSM(_sets), {}, owned)
    owned_names = set(owned)

    owned_diffs = [d for d in domain_bounds_diff(bm, oracle) if d[0] in owned_names]
    real: list[Any] = []
    for d in owned_diffs:
        name, _idx, a_dom, b_dom, a_bnd, b_bnd = d
        if name in _DOMAIN_CARRY_VARS and _is_post_scaling_floor_carry(
            a_dom, b_dom, a_bnd, b_bnd
        ):
            continue  # documented post-scaling floor/level carry (composer owns)
        real.append(d)
    assert real == [], f"{unit_name}: domain/bounds mismatch on owned vars: {real[:3]}"


class _ScratchSM:
    """Minimal SetManager stand-in for introspecting a block's owned symbols."""

    def __init__(self, sets: Any) -> None:
        self._m = _set_elems(sets)

    def get(self, name: str):
        return self._m[name]
