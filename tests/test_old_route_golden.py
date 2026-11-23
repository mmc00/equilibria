"""The old-route golden comparison: 1e-9 relative for levels, a wider floor only
for walras (a residual, zero in exact arithmetic) and a wider band only where a
caller asks for it (the Johansen step)."""

from __future__ import annotations

from tests._old_route_golden import level_mismatches

W = "('walras', 'shock')"
X = "('xw', ('EU_28', 'Svces', 'ROW', 'shock'))"


def test_walras_noise_between_cpus_passes() -> None:
    # measured, Linux CI vs the Mac recording
    assert level_mismatches({W: -3.594601797008573e-11}, {W: -5.742e-12}) == []
    assert level_mismatches({"walrasshock": 2.25e-9}, {"walrasshock": 6.8e-10}) == []


def test_walras_off_by_more_than_its_floor_fails() -> None:
    assert level_mismatches({W: 1e-6}, {W: 0.0})


def test_a_level_keeps_the_tight_band() -> None:
    assert level_mismatches({X: 0.5126869643468949}, {X: 0.5126869649502608})
    assert level_mismatches({X: 0.5}, {X: 0.5 * (1 + 5e-10)}) == []


def test_a_caller_can_widen_the_band() -> None:
    got, want = {X: 0.5126869643468949}, {X: 0.5126869649502608}
    assert level_mismatches(got, want, rel=1e-7) == []
    assert level_mismatches({X: 0.5}, {X: 0.5 * (1 + 1e-6)}, rel=1e-7)


def test_tuple_keys_of_the_family_arrays() -> None:
    """``mp_mismatches`` keys cells by ``(var, field, position)``."""
    assert level_mismatches({("xw", "value", 0): 1.0}, {("xw", "value", 0): 1.0}) == []
    assert level_mismatches({("xw", "value", 0): 1.0}, {("xw", "value", 0): 1.1})
    w = ("walras", "value", 2)
    assert level_mismatches({w: 3e-11}, {w: 3e-12}) == []
