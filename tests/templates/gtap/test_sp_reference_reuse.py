"""solve_multiperiod must not build the same reference model twice.

WHAT THE DRIVER DOES.  For each period it builds a COMPLETE single-period
``GTAPModelEquations(...).build_model()`` — 3.4M vars on 20x41 — purely to read
which variables it leaves fixed/bounded (``_replicate_sp_fixing`` /
``_replicate_sp_bounds``), then throws the model away.

WHY IT IS WASTE.  In gtap mode the check period passes ``base_closure`` — the
SAME four arguments as the base period — so the second build is byte-identical
to the first.  MEASURED on 20x41: the three reference builds cost 1.74 min of a
7.46 min wall (23%), and yield 0.118 min of replication work.  Verified by
hashing every (name, index, value, lb, ub) of the built model: base and check
give the same digest, shock differs (it uses the shocked params).

WHAT THIS PINS.  The reuse must be CONDITIONAL.  Under altertax the check period
uses ``alt_closure``, which differs from ``base_closure`` in four fields
(name, capital_mobility, fix_taxes, fix_technology) — reusing there would
replicate the WRONG fixing pattern and silently change the solution, which is
exactly the failure mode this codebase keeps hitting (no exception, just a
different answer).
"""

import pytest

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig


def _closure(
    *,
    name: str = "base",
    closure_type: str = "MCP",
    capital_mobility: str = "sluggish",
    savf_flag: str = "capFix",
    fix_endowments: bool = False,
    fix_taxes: bool = False,
    fix_technology: bool = False,
    if_sub: bool = False,
    numeraire: str = "pnum",
) -> GTAPClosureConfig:
    """The base-period closure the driver builds, with one field overridable."""
    return GTAPClosureConfig(
        name=name,
        closure_type=closure_type,  # ty: ignore[invalid-argument-type]
        capital_mobility=capital_mobility,  # ty: ignore[invalid-argument-type]
        savf_flag=savf_flag,  # ty: ignore[invalid-argument-type]
        fix_endowments=fix_endowments,
        fix_taxes=fix_taxes,
        fix_technology=fix_technology,
        if_sub=if_sub,
        numeraire=numeraire,
    )


# ── The reuse predicate ─────────────────────────────────────────────────────────


def test_identical_closures_are_reusable():
    """Same closure object -> the built reference model is the same -> reuse."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _sp_ref_reusable

    c = _closure()
    assert _sp_ref_reusable(c, c) is True


def test_equal_but_distinct_closures_are_reusable():
    """Equality, not identity: two configs with the same fields build the same model."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _sp_ref_reusable

    assert _sp_ref_reusable(_closure(), _closure()) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "altertax"),
        ("capital_mobility", "mobile"),
        ("fix_taxes", True),
        ("fix_technology", True),
        ("fix_endowments", True),
        ("if_sub", True),
        ("numeraire", "pfact"),
        ("closure_type", "CNS"),
        ("savf_flag", "capFlex"),
    ],
)
def test_any_differing_field_blocks_reuse(field, value):
    """A closure that differs in ANY field builds a different model — never reuse.

    The altertax case is not hypothetical: alt_closure differs from base_closure
    in name, capital_mobility, fix_taxes and fix_technology.
    """
    from equilibria.templates.gtap.gtap_multiperiod_driver import _sp_ref_reusable

    assert _sp_ref_reusable(_closure(), _closure(**{field: value})) is False


def test_none_is_never_reusable():
    """No previous model to reuse -> build one."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _sp_ref_reusable

    assert _sp_ref_reusable(None, _closure()) is False
    assert _sp_ref_reusable(_closure(), None) is False


def test_altertax_pair_is_not_reusable():
    """The exact pair the driver builds under altertax must not be reused."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import _sp_ref_reusable

    base_closure = _closure()
    alt_closure = _closure(
        name="altertax",
        capital_mobility="mobile",
        fix_taxes=True,
        fix_technology=True,
    )
    assert _sp_ref_reusable(base_closure, alt_closure) is False
