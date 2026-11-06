"""Phase timer: self time per phase, nested phases do not double count."""

import time

from equilibria.family.phases import PhaseTimer


def test_nested_phases_report_self_time():
    pt = PhaseTimer()
    with pt("outer"):
        time.sleep(0.02)
        with pt("inner"):
            time.sleep(0.03)
    assert 0.025 < pt.seconds["inner"] < 0.06
    assert 0.015 < pt.seconds["outer"] < 0.05  # inner's 0.03 is not in it
    assert pt.counts == {"outer": 1, "inner": 1}


def test_repeated_phase_accumulates_and_counts():
    pt = PhaseTimer()
    for _ in range(3):
        with pt("x"):
            time.sleep(0.005)
    assert pt.counts["x"] == 3 and pt.seconds["x"] >= 0.014


def test_exception_still_records_and_pops():
    pt = PhaseTimer()
    try:
        with pt("a"):
            raise ValueError
    except ValueError:
        pass
    with pt("b"):
        pass
    assert "a" in pt.seconds and pt.counts["b"] == 1


def test_scope_prefixes_names_and_none_timer_is_free():
    pt = PhaseTimer()
    with pt.scope("check"), pt("recipe"):
        pass
    assert "check.recipe" in pt.seconds
    from equilibria.family.phases import NULL

    with NULL("anything"), NULL.scope("s"):
        pass
    assert NULL.seconds == {}
