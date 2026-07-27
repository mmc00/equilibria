"""Diagnostic tooling for the first B-block solve (F3).

The block-extraction framework has never solved a model, and F3 migrates all
7 units at once — so a failed 3x3 solve needs to be debuggable DIRECTED, not
blind. This module gives three read-only diagnostics, reusing/mirroring the
existing parity-cascade tools where possible:

  - ``residual_report``   — which equation is violated, worst first, at the
    model's current (optionally GDX-seeded) point.
  - ``form_diff``          — per-cell Pyomo-expression compare of a B-block's
    constraint against the monolith's (structural, not just numeric).
  - ``domain_bounds_diff`` — per-var domain/bounds mismatches between two
    models (a block that dropped a bound is invisible to a value diff).

Pure diagnostics: nothing here mutates model/economics code. Imports of the
heavy GTAP machinery (GTAPMultiPeriodModel, GDX seeding) are kept LAZY —
inside functions — so importing this module stays cheap for the test suite.

Usage (library, not a CLI):
    from blocks_diag import residual_report, domain_bounds_diff, form_diff
    rep = residual_report(model, seed=gdx_path)
    worst_eq, worst_idx, worst_res = rep[0]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "gtap") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))


def residual_report(pyomo_model, seed: Any = None) -> list[tuple[str, Any, float]]:
    """Evaluate every active Constraint's violation at the model's current point.

    Returns ``[(eq_name, index, residual), ...]`` sorted by |residual| DESCENDING
    (worst first). ``residual`` is 0.0 for a satisfied constraint.

    ``seed``: if given (a GDX path), the model is seeded first via the existing
    multi-period seeding helper (``GTAPMultiPeriodModel.seed_all_periods``,
    imported lazily) before evaluating. If ``seed is None`` the model is
    evaluated as-is — this is the toy-model / unit-test path.
    """
    from pyomo.environ import Constraint, value

    if seed is not None:
        from equilibria.templates.gtap.gtap_model_multiperiod import (
            GTAPMultiPeriodModel,
        )

        # seed_all_periods never dereferences `self` (verified: it only reads
        # its `m`/`gdx_path` args) — call it unbound so this diagnostic doesn't
        # have to construct a full GTAPMultiPeriodModel(sets, params, ...) just
        # to reach the seeding logic.
        GTAPMultiPeriodModel.seed_all_periods(None, pyomo_model, seed)

    rows: list[tuple[str, Any, float]] = []
    for con_obj in pyomo_model.component_objects(Constraint, active=True):
        for idx, con in con_obj.items():
            if not con.active:
                continue
            try:
                body_val = value(con.body)
            except Exception:
                # Undefined at this point (e.g. divide-by-zero in an unseeded
                # toy model) — surface as an infinite residual rather than crash.
                rows.append((con_obj.name, idx, float("inf")))
                continue

            residual = 0.0
            if con.equality:
                lower_val = value(con.lower)
                residual = abs(body_val - lower_val)
            else:
                lower_val = value(con.lower) if con.lower is not None else None
                upper_val = value(con.upper) if con.upper is not None else None
                if lower_val is not None and body_val < lower_val:
                    residual = lower_val - body_val
                if upper_val is not None and body_val > upper_val:
                    residual = max(residual, body_val - upper_val)

            rows.append((con_obj.name, idx, residual))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows


def _domain_name(domain) -> str:
    """Stable label for a Pyomo domain set (``Reals``, ``NonNegativeReals``, ...)."""
    name = getattr(domain, "name", None)
    if name:
        return name
    return type(domain).__name__


def domain_bounds_diff(
    model_a, model_b
) -> list[tuple[str, Any, str, str, tuple, tuple]]:
    """Compare Var domain + bounds between two models, matched by NAME.

    Returns ``[(var_name, index, a_domain, b_domain, a_bounds, b_bounds), ...]``
    for every index where the domain label or the bounds tuple differs. Vars
    present in only one model are skipped (nothing to compare) — this is a
    mismatch diagnostic, not a coverage diagnostic.
    """
    from pyomo.environ import Var

    vars_a = {vo.name: vo for vo in model_a.component_objects(Var)}
    vars_b = {vo.name: vo for vo in model_b.component_objects(Var)}

    diffs: list[tuple[str, Any, str, str, tuple, tuple]] = []
    for name in vars_a.keys() & vars_b.keys():
        vo_a, vo_b = vars_a[name], vars_b[name]
        idx_a = set(vo_a.keys()) if vo_a.is_indexed() else {None}
        idx_b = set(vo_b.keys()) if vo_b.is_indexed() else {None}
        for idx in idx_a & idx_b:
            va = vo_a[idx]
            vb = vo_b[idx]
            a_domain = _domain_name(va.domain)
            b_domain = _domain_name(vb.domain)
            a_bounds = va.bounds
            b_bounds = vb.bounds
            if a_domain != b_domain or a_bounds != b_bounds:
                diffs.append((name, idx, a_domain, b_domain, a_bounds, b_bounds))

    return diffs


def _expr_to_str(expr) -> str:
    """Canonical string form of a Pyomo expression for a textual diff fallback."""
    return str(expr)


def _exprs_equal(expr_a, expr_b) -> bool:
    """True if two Pyomo expressions are structurally equal.

    Prefers ``pyomo.core.expr.compare.compare_expressions`` (walks the
    expression tree), then falls back to a canonical string compare.

    ``compare_expressions`` matches leaf components by **object identity**, so
    two structurally identical expressions built on **different** ``ConcreteModel``
    instances (a block-built model vs a separately-built monolith — exactly the
    Task-4 form gate) return ``False`` with no exception. When that happens we
    fall back to the string compare: string-equal ⇒ structurally identical
    regardless of which model the leaf objects belong to. The fallback also
    covers a Pyomo version where the helper is unavailable (the ``except`` arm).
    """
    try:
        from pyomo.core.expr.compare import compare_expressions

        if compare_expressions(expr_a, expr_b):
            return True
    except Exception:
        pass
    return _expr_to_str(expr_a) == _expr_to_str(expr_b)


def form_diff(block, monolith_model) -> list[tuple[Any, str, str]]:
    """Per-cell expanded-Pyomo-expression diff of a B-block vs the monolith.

    ``block`` and ``monolith_model`` are each either a Pyomo Constraint
    component (indexed or scalar) OR a plain ``{index: expr_or_constraint}``
    mapping of per-cell Pyomo expressions — the two representations a B-block
    may take before/after it is wired onto a full ConcreteModel (see Task 4).

    Returns ``[(index, block_expr_str, monolith_expr_str), ...]`` for cells
    that DIFFER structurally. Cells present in only one side are skipped —
    this reports MISMATCHES, not coverage gaps (use a set-diff of indices for
    that separately).

    NOTE for Task 4: equality is treated as **string-equal after a
    canonical-form compare** (see ``_exprs_equal``). ``compare_expressions``
    matches leaf components by object identity, so a block built on its own
    ``ConcreteModel`` compared against the monolith's model would false-diff on
    every cell; the string fallback makes the cross-model block-vs-oracle
    comparison work (textually identical after a verbatim port ⇒ equal).
    ``compare_expressions`` still short-circuits the same-model case where two
    exprs are structurally different. This compares expression TREES/text, not
    numeric equivalence — two algebraically-equal-but-differently-shaped forms
    (e.g. ``a * (b + c)`` vs ``a * b + a * c``) still show up as a "diff"
    because their strings differ, which is what we WANT: it forces eyeball /
    re-transcription of a legitimate algebraic re-derivation instead of a
    silent green check.
    """

    def _cells(source):
        if hasattr(source, "items") and not isinstance(source, dict):
            # Pyomo Constraint component (indexed or scalar)
            return {idx: con.expr for idx, con in source.items()}
        if isinstance(source, dict):
            out = {}
            for idx, v in source.items():
                out[idx] = getattr(v, "expr", v)
            return out
        # Scalar constraint-like object with a single .expr
        return {None: source.expr}

    block_cells = _cells(block)
    mono_cells = _cells(monolith_model)

    diffs: list[tuple[Any, str, str]] = []
    for idx in block_cells.keys() & mono_cells.keys():
        b_expr = block_cells[idx]
        m_expr = mono_cells[idx]
        if not _exprs_equal(b_expr, m_expr):
            diffs.append((idx, _expr_to_str(b_expr), _expr_to_str(m_expr)))

    return diffs
