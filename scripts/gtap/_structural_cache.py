"""Signature-keyed reuse of the name-stable structural setup across GTAP solve phases.

`_run_path_capi_nonlinear_full` re-runs `structural_matching` + `apply_squareness_patches` +
`apply_conditional_fixing` on every phase (base, check, shock, each shock-continuation step) —
9x on the 20x41. cProfile (gate v15, 2026-08-28) showed these are 34% of a 22-min non-Newton
wall, and the 6 shock-continuation steps share an IDENTICAL active-constraint/free-var name set
(only the tariff value changes) — so their structural work is redundant 6x over.

Same mechanism as lever B2 (reuse MUMPS symbolic factorization only on a byte-identical nnz
pattern): reuse the matching/squareness/fixing decisions ONLY when the active-constraint +
free-var NAME sets are byte-identical to the previous phase. A byte-identical name-set signature
guarantees the bipartite adjacency (and therefore the matching, the squareness deactivations, and
the conditional-fixing decisions, all of which depend only on equation forms + the active/free
partition) is identical — so reuse cannot produce a different squared system. Decisions are
captured and replayed by NAME ONLY; live VarData/ConstraintData objects differ per phase.
"""
from __future__ import annotations

import hashlib

from pyomo.environ import Constraint, Var, value


def signature(constraint_names, free_var_names) -> str:
    """Order-sensitive sha256 over (active constraint names, free variable names)."""
    h = hashlib.sha256()
    h.update(b"C\x00")
    for n in constraint_names:
        h.update(n.encode())
        h.update(b"\x00")
    h.update(b"V\x00")
    for n in free_var_names:
        h.update(n.encode())
        h.update(b"\x00")
    return h.hexdigest()


class StructuralCache:
    """Single-entry (last-call) signature cache. Lives for one solve_multiperiod run."""

    def __init__(self):
        self._sig = None
        self._art = None

    def reset(self):
        self._sig = None
        self._art = None

    def try_reuse(self, sig):
        return self._art if (self._sig is not None and sig == self._sig) else None

    def store(self, sig, *, matched_var_names, squareness, fixing):
        self._sig = sig
        self._art = {
            "matched_var_names": list(matched_var_names),
            "squareness": squareness,
            "fixing": fixing,
        }


def reorder_by_name(free_variables, matched_var_names):
    """Reorder `free_variables` (live VarData) to the cached name order.

    Returns None on a name-set mismatch (caller must fall back to recompute — never
    silently proceed on a partial/stale match).
    """
    byname = {v.name: v for v in free_variables}
    if set(byname) != set(matched_var_names) or len(byname) != len(matched_var_names):
        return None
    return [byname[n] for n in matched_var_names]


def apply_squareness_by_name(model, squareness):
    """Re-apply cached squareness decisions: deactivate named constraints, fix named vars to 0."""
    for cname in squareness.get("deactivated", []):
        c = model.find_component(cname)
        if c is not None and c.active:
            c.deactivate()
    for vname in squareness.get("fixed_zero", []):
        v = model.find_component(vname)
        if v is not None:
            if v.lb is not None and float(v.lb) > 0.0:
                v.setlb(0.0)
            v.fix(0.0)


def apply_fixing_by_name(model, fixing):
    """Re-apply cached conditional-fixing decisions: fix named vars at their current value."""
    for vname in fixing.get("fixed", []):
        v = model.find_component(vname)
        if v is not None and not v.fixed:
            val = float(value(v)) if v.value is not None else 1.0
            v.fix(val)


def snapshot_active_fixed(model):
    """Return (active constraint names, fixed var names) — used to diff recompute decisions."""
    active = {
        c.name
        for c in model.component_data_objects(Constraint, active=True, descend_into=True)
    }
    fixed = {
        v.name
        for v in model.component_data_objects(Var, descend_into=True)
        if v.fixed
    }
    return active, fixed
