"""Cache Pyomo component `.name` strings by object identity.

`_run_path_capi_nonlinear_full` sorts/indexes ~395k constraints and free variables by
`.name` on EVERY one of its 9 phase invocations (base, check, shock, each shock-continuation
step). Pyomo's `.name` is NOT cached — it walks up the block hierarchy and rebuilds the full
dotted/indexed string every call (component.py:name -> getname -> component_namer.index_repr/
name_repr -> str.join). cProfile on the real 20x41 (gate v19, 2026-08-29) measured this chain
at **72 MILLION calls, 8.4 minutes of real wall-clock time** across the 9 phases.

The underlying VarData/ConstraintData Python objects are IDENTICAL across all 9 phases — the
model is built ONCE and only mutated in place (fixed/active flags change; freeze_inactive_periods
never rebuilds or reparents components). So a component's `.name` is invariant for the whole
solve_multiperiod run, and caching it by `id(obj)` is safe: no risk of staleness, because we
never recompute a name for an object whose actual name could have changed (it can't — the model
is never restructured between phases).

This is safer than the structural-matching/squareness caches (which reuse VALUE-DEPENDENT
decisions keyed on a signature): here we cache an INVARIANT STRING for an INVARIANT OBJECT,
so there is no signature to get wrong and no possibility of skipping a decision that should
have been recomputed.
"""
from __future__ import annotations

_CACHE: dict[int, str] = {}


def cached_name(obj) -> str:
    """Return obj.name, computed once per object identity and reused thereafter."""
    key = id(obj)
    name = _CACHE.get(key)
    if name is None:
        name = obj.name
        _CACHE[key] = name
    return name


def reset() -> None:
    """Clear the cache. Call between independent solve_multiperiod runs in the same
    process (e.g. tests) so a stale id() (from a garbage-collected prior model) can
    never be reused — though within one run this never happens since the model's
    components live for the run's full duration."""
    _CACHE.clear()
