"""There is ONE build_expression contract, and the bridge consumes it."""


def test_single_build_expression_signature():
    import inspect

    from equilibria.core.symbolic_equations import SymbolicEquation

    sig = inspect.signature(SymbolicEquation.build_expression)
    assert list(sig.parameters)[1:] == ["pyomo_model", "indices"]


def test_dead_dsl_removed_or_routed():
    import equilibria.core.symbolic_equations as se

    # ResidualEquation, if present, must implement the same (pyomo_model, indices) contract
    if hasattr(se, "ResidualEquation"):
        import inspect

        sig = inspect.signature(se.ResidualEquation.build_expression)
        assert list(sig.parameters)[1:] == ["pyomo_model", "indices"]

    # Default per the F3 plan is removal (YAGNI): the dead DSL must actually
    # be gone, not just untested. A passing `hasattr` check above is vacuous
    # unless we also assert the names don't exist.
    assert not hasattr(se, "ResidualEquation"), (
        "dead ResidualEquation DSL must be removed"
    )
    for helper in (
        "var",
        "param",
        "const",
        "add",
        "multiply",
        "power",
        "divide",
        "log",
        "exp",
        "sum_over",
    ):
        assert not hasattr(se, helper), (
            f"dead combinator helper {helper!r} must be removed"
        )
