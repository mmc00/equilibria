"""Tests for the per-phase structural cache (scripts/gtap/_structural_cache.py).

Pins the CONTRACT the 20x41 wall-reduction hook depends on:
  - signature() is order-sensitive and name-based (a different active/free partition
    must hash differently; the same partition must hash identically)
  - StructuralCache only returns a hit on a byte-identical signature
  - reorder_by_name reproduces the cached matching order, or returns None on a
    name-set mismatch (never a silent partial match)
  - apply_squareness_by_name / apply_fixing_by_name reproduce the recompute path's
    model mutations (deactivate/fix) from names alone
  - snapshot_active_fixed captures exactly the active-constraint / fixed-var name sets
"""

import importlib.util
import os

import pyomo.environ as pyo
import pytest

_HELPER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "gtap", "_structural_cache.py")
)
_spec = importlib.util.spec_from_file_location("_structural_cache", _HELPER)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

signature = _mod.signature
StructuralCache = _mod.StructuralCache
reorder_by_name = _mod.reorder_by_name
apply_squareness_by_name = _mod.apply_squareness_by_name
apply_fixing_by_name = _mod.apply_fixing_by_name
snapshot_active_fixed = _mod.snapshot_active_fixed


def _toy():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1.0)
    m.cA = pyo.Constraint(expr=m.x[1] + m.x[2] == 2)
    m.cB = pyo.Constraint(expr=m.x[2] + m.x[3] == 2)
    return m


# --- signature() ---------------------------------------------------------------

def test_signature_is_order_sensitive_and_name_based():
    s1 = signature(["eqA", "eqB"], ["x", "y"])
    s2 = signature(["eqA", "eqB"], ["x", "y"])
    s3 = signature(["eqA", "eqB"], ["y", "x"])  # different free-var order
    s4 = signature(["eqB", "eqA"], ["x", "y"])  # different constraint order
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_signature_differs_on_different_name_sets():
    s1 = signature(["eqA"], ["x"])
    s2 = signature(["eqA", "eqB"], ["x"])
    assert s1 != s2


# --- StructuralCache -------------------------------------------------------------

def test_cache_hit_only_on_identical_signature():
    c = StructuralCache()
    sig = signature(["eqA"], ["x"])
    assert c.try_reuse(sig) is None  # empty cache

    c.store(
        sig,
        matched_var_names=["x"],
        squareness={"deactivated": [], "fixed_zero": []},
        fixing={"fixed": []},
    )
    hit = c.try_reuse(sig)
    assert hit == {
        "matched_var_names": ["x"],
        "squareness": {"deactivated": [], "fixed_zero": []},
        "fixing": {"fixed": []},
    }

    different_sig = signature(["eqA"], ["y"])
    assert c.try_reuse(different_sig) is None  # different sig -> miss

    c.reset()
    assert c.try_reuse(sig) is None  # reset clears the entry


def test_cache_store_replaces_single_entry():
    c = StructuralCache()
    sig1 = signature(["eqA"], ["x"])
    sig2 = signature(["eqB"], ["y"])
    c.store(sig1, matched_var_names=["x"], squareness={"deactivated": []}, fixing={"fixed": []})
    c.store(sig2, matched_var_names=["y"], squareness={"deactivated": []}, fixing={"fixed": []})
    assert c.try_reuse(sig1) is None  # the earlier entry is gone
    assert c.try_reuse(sig2) is not None


# --- reorder_by_name --------------------------------------------------------------

def test_reorder_by_name_matches_cached_order():
    m = _toy()
    fv = [m.x[1], m.x[2], m.x[3]]
    out = reorder_by_name(fv, ["x[3]", "x[1]", "x[2]"])
    assert [v.name for v in out] == ["x[3]", "x[1]", "x[2]"]


def test_reorder_by_name_mismatch_returns_none():
    m = _toy()
    assert reorder_by_name([m.x[1]], ["x[1]", "x[2]"]) is None


# --- apply_squareness_by_name / apply_fixing_by_name -------------------------------

def test_apply_squareness_by_name_deactivates_and_fixes():
    m = _toy()
    apply_squareness_by_name(m, {"deactivated": ["cB"], "fixed_zero": ["x[3]"]})
    assert not m.cB.active
    assert m.x[3].fixed and pyo.value(m.x[3]) == 0.0
    assert m.cA.active  # untouched


def test_apply_fixing_by_name_fixes_at_current_value():
    m = _toy()
    m.x[2].set_value(5.0)
    apply_fixing_by_name(m, {"fixed": ["x[2]"]})
    assert m.x[2].fixed
    assert pyo.value(m.x[2]) == 5.0


# --- snapshot_active_fixed ---------------------------------------------------------

def test_snapshot_captures_active_and_fixed():
    m = _toy()
    m.cB.deactivate()
    m.x[3].fix(0.0)
    cons, fixed = snapshot_active_fixed(m)
    assert cons == {"cA"}
    assert "x[3]" in fixed
