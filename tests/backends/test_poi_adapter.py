"""The POI adapter reproduces the slice of Pyomo's ConcreteModel surface that the
GTAP blocks use.

The blocks write their equations in Pyomo syntax (``model.pf[r, f, a] * model.xf[...]``,
``for f in model.f``, ``value(model.flag[...])``). If an adapter can present that same
surface over PyOptInterface, those 5,910 lines of equations run unmodified against a
second backend. These tests pin the surface itself; Task 2 puts a real block on it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyoptinterface")


def _poi_model():
    """A POI model, skipping the test if the Ipopt shared library is unavailable.

    POI does not autoload it; the library ships with Homebrew on this machine and
    with the system package on Kaggle.
    """
    from pyoptinterface import ipopt

    if not ipopt.is_library_loaded():
        for candidate in (
            "/opt/homebrew/lib/libipopt.dylib",
            "/usr/lib/x86_64-linux-gnu/libipopt.so",
            "libipopt.so",
        ):
            try:
                if ipopt.load_library(candidate):
                    break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
        else:
            pytest.skip("Ipopt library not found; POI cannot build a model")
    return ipopt.Model()


def test_adapter_returns_the_same_handle_for_a_repeated_index():
    """``model.px['USA', 'Food']`` is one variable, however many times it is read.

    The blocks rely on Pyomo's identity semantics: an equation that mentions the same
    cell twice constrains ONE variable. An adapter that minted a fresh handle per
    access would silently split it in two and produce a different — wrongly larger —
    system.
    """
    from equilibria.backends.poi_adapter import PoiModelAdapter

    adapter = PoiModelAdapter(
        _poi_model(),
        sets={"r": ["USA", "EU"], "i": ["Food", "Mfg"]},
        params={},
        var_specs={"px": ("r", "i")},
    )

    first = adapter.px["USA", "Food"]
    assert first is not None
    assert adapter.px["USA", "Food"] is first
    assert adapter.px["EU", "Food"] is not first


def test_adapter_arithmetic_builds_a_poi_expression():
    """Operators on adapter variables produce POI expressions, not plain numbers.

    A collapse to int/float would mean the variables never entered the expression,
    leaving a constant where a constraint should be.
    """
    from equilibria.backends.poi_adapter import PoiModelAdapter

    adapter = PoiModelAdapter(
        _poi_model(),
        sets={"r": ["USA"]},
        params={},
        var_specs={"p": ("r",), "q": ("r",)},
    )

    expr = adapter.p["USA"] * adapter.q["USA"]
    assert not isinstance(expr, (int, float))


def test_adapter_iterates_a_set_like_pyomo():
    """``for f in model.f`` walks the set's elements, as the block bodies expect."""
    from equilibria.backends.poi_adapter import PoiModelAdapter

    adapter = PoiModelAdapter(
        _poi_model(), sets={"f": ["Land", "Labor"]}, params={}, var_specs={}
    )

    assert list(adapter.f) == ["Land", "Labor"]


def test_adapter_exposes_params_as_raw_values():
    """Params come back as numbers, not variable handles.

    The blocks branch on parameters at build time (``if value(model.xfflag[...])``).
    A handle would make that condition always truthy and silently change which
    constraints get built.
    """
    from equilibria.backends.poi_adapter import PoiModelAdapter

    class _Params:
        alpha = 0.35

    adapter = PoiModelAdapter(
        _poi_model(), sets={}, params=_Params(), var_specs={}
    )

    assert adapter.alpha == 0.35


def test_adapter_reports_an_unknown_name():
    """An unknown attribute raises AttributeError naming what was missing.

    A silent None here would surface much later as an inscrutable expression error.
    """
    from equilibria.backends.poi_adapter import PoiModelAdapter

    adapter = PoiModelAdapter(_poi_model(), sets={}, params={}, var_specs={})

    with pytest.raises(AttributeError, match="no_such_thing"):
        _ = adapter.no_such_thing
