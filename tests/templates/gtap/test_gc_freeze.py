"""The solve freezes the permanent object graph, and always releases it.

Why this matters: the GC's full passes walk the whole built model (56.2M
tracked objects on gtap7_20x41) even though that graph never holds garbage.
Freezing it removed 0.68 min (8%) of the 20x41 wall with a byte-identical
solution. These tests pin the CONTRACT of that lever — that it engages, that
it can be switched off, and above all that it never leaks the frozen state.
"""

from __future__ import annotations

import gc

import pytest

from equilibria.templates.gtap import gtap_multiperiod_driver as driver


@pytest.fixture
def _no_frozen_state():
    """Leave the process's GC exactly as we found it.

    NOTE: the freeze count is NOT zero at rest — pytest freezes its own objects
    (measured: 375 after a plain gc.collect() under pytest). Every assertion
    below therefore compares against a baseline captured here, never against 0.
    An earlier version of this file asserted `get_freeze_count() > 0` and
    PASSED with gc.freeze() deleted from the driver, because pytest's own
    frozen objects satisfied it.
    """
    gc.unfreeze()
    gc.collect()
    yield
    gc.unfreeze()


@pytest.fixture
def spy_inner(monkeypatch):
    """Replace the solve body with a spy that records the freeze count."""
    seen = {}

    def _fake(m, params, closure, **kw):
        seen["frozen"] = gc.get_freeze_count()
        if kw.get("mode") == "boom":
            raise RuntimeError("solve exploded")
        return {"base": {"code": 1}}

    monkeypatch.setattr(driver, "_solve_multiperiod_inner", _fake)
    return seen


def test_freezes_the_graph_during_the_solve(spy_inner, _no_frozen_state):
    _junk = [[i] for i in range(10_000)]
    baseline = gc.get_freeze_count()
    result = driver.solve_multiperiod(None, None, None)
    # The 10k objects above must have been added to the frozen set.
    assert spy_inner["frozen"] >= baseline + 10_000, (
        f"the permanent graph was not frozen: {spy_inner['frozen']} vs "
        f"baseline {baseline}"
    )
    assert result == {"base": {"code": 1}}
    assert _junk  # keep alive until here


def test_unfreezes_on_normal_return(spy_inner, _no_frozen_state):
    _junk = [[i] for i in range(10_000)]
    driver.solve_multiperiod(None, None, None)
    assert gc.get_freeze_count() == 0, "the frozen set outlived the solve"
    assert _junk


def test_unfreezes_when_the_solve_raises(spy_inner, _no_frozen_state):
    """The leak that a plain ExitStack.close() at the return would NOT prevent.

    A frozen set that outlives a failed solve silently degrades every later
    solve in the same process, and nothing would point at this function.
    """
    _junk = [[i] for i in range(10_000)]
    with pytest.raises(RuntimeError, match="solve exploded"):
        driver.solve_multiperiod(None, None, None, mode="boom")
    assert gc.get_freeze_count() == 0, "a failed solve leaked the frozen set"
    assert _junk


def test_env_var_disables_the_freeze(spy_inner, monkeypatch, _no_frozen_state):
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    _junk = [[i] for i in range(10_000)]
    baseline = gc.get_freeze_count()
    driver.solve_multiperiod(None, None, None)
    assert spy_inner["frozen"] == baseline, "freeze ran despite being switched off"
    assert _junk


def test_wrapper_forwards_every_argument(monkeypatch, _no_frozen_state):
    """A dropped kwarg here would silently change the SOLVE, not just its speed."""
    got = {}

    def _fake(m, params, closure, **kw):
        got.update(kw)
        got["positional"] = (m, params, closure)
        return {}

    monkeypatch.setattr(driver, "_solve_multiperiod_inner", _fake)
    driver.solve_multiperiod(
        "MODEL",
        "PARAMS",
        "CLOSURE",
        ref_gdx="g.gdx",
        skip_base_solve=True,
        mute_welfare=False,
        seed_from_prior=True,
        holdfix_cd=False,
        mode="gtap",
        solve_check=True,
        settle_only=True,
    )
    assert got["positional"] == ("MODEL", "PARAMS", "CLOSURE")
    assert got == {
        "positional": ("MODEL", "PARAMS", "CLOSURE"),
        "ref_gdx": "g.gdx",
        "skip_base_solve": True,
        "mute_welfare": False,
        "seed_from_prior": True,
        "holdfix_cd": False,
        "mode": "gtap",
        "solve_check": True,
        "settle_only": True,
    }


def test_cyclic_garbage_is_still_collected_while_frozen(_no_frozen_state):
    """The property that separates freeze() from disable().

    If freezing also stopped collecting NEW cycles, the solve would grow
    unboundedly. It does not: frozen objects are skipped, fresh cycles are not.
    """
    permanent = [[i] for i in range(5_000)]
    gc.collect()
    gc.freeze()
    try:
        garbage = []
        for _ in range(2_000):
            a, b = {}, {}
            a["b"], b["a"] = b, a  # a reference cycle
            garbage.append(a)
        del garbage
        collected = gc.collect()
        assert collected > 0, "cycles created after freeze() were not collected"
    finally:
        gc.unfreeze()
    assert permanent
