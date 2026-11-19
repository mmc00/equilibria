"""``tests._golden.mismatches``: a key on one side only, or a value off by more than
``abs_tol + rel * max(|got|, |want|)``."""

from __future__ import annotations

from tests._golden import mismatches


def test_last_bits_pass_and_a_real_change_fails() -> None:
    want = {"x": {"a": "3.6512068852181603"}}
    assert mismatches({"x": {"a": "3.65120688521816"}}, want) == []
    assert mismatches({"x": {"a": "3.65120689"}}, want) != []


def test_a_missing_or_extra_cell_fails() -> None:
    assert mismatches({"x": {}}, {"x": {"a": "1.0"}}) == ["('x', 'a'): only in golden"]
    assert mismatches({"x": {"a": "1.0"}}, {"x": {}}) == ["('x', 'a'): not in golden"]


def test_a_near_zero_cell_needs_an_absolute_floor() -> None:
    """A solver residual near zero (walras ~3e-11) moves by ~2e-12 on another
    machine: relative to itself that is 7 %, absolutely it is nothing."""
    got, want = (
        {"walras": {"shock": "-3.4126e-11"}},
        {"walras": {"shock": "-3.1726e-11"}},
    )
    assert mismatches(got, want) != []
    assert mismatches(got, want, rel=1e-9, abs_tol=1e-10) == []
    assert (
        mismatches({"w": {"s": "1e-8"}}, {"w": {"s": "0.0"}}, rel=1e-9, abs_tol=1e-10)
        != []
    )
