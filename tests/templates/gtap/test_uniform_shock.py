"""shocks.uniform: one percent on every shock cell of an instrument that can take it.

GEMPACK writes ``Shock tm = uniform 3``; here ``{"imptx": uniform(3)}``. The cells are
the ones the +10 % default tariff writes (``_can_take_shock``), so ``uniform(10)`` on
``imptx`` must be that default exactly. No solver: gtap7_3x3 is built once.
"""

from __future__ import annotations

import pytest
from pyomo.environ import value

from equilibria.templates.gtap import shocks, uniform
from equilibria.templates.gtap.shocks import apply_default_tariff, apply_shock


@pytest.fixture(scope="module")
def built():
    from tests.templates.gtap._period_snapshot import _load_params

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p = _load_params()
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    m, _ = build_block_model(p, p.sets, gc, list(p.sets.r)[-1])
    return m


def _shock_values(m) -> dict[tuple[str, tuple], tuple[float, bool]]:
    return {
        (n, k): (float(value(getattr(m, n)[k])), bool(getattr(m, n)[k].fixed))
        for n in m._exogenous_instruments
        for k in getattr(m, n)
        if k[-1] == "shock"
    }


def _restore(m, saved) -> None:
    for (n, k), (v, fixed) in saved.items():
        var = getattr(m, n)[k]
        var.set_value(v)
        var.fixed = fixed


@pytest.fixture
def m(built):
    """The model with its ShockBlock restored after each test."""
    saved = _shock_values(built)
    yield built
    _restore(built, saved)


def _cell(m, name):
    from equilibria.templates.gtap.instruments import check_instrument_cell

    for k in getattr(m, name):
        if k[-1] != "shock":
            continue
        try:
            check_instrument_cell(m, name, k[:-1])
        except ValueError:
            continue
        return k[:-1]
    pytest.skip(f"no live {name} cell")


def test_uniform_10_equals_default_tariff(m):
    saved = _shock_values(m)
    want = apply_default_tariff(m, gtap_mode=True)
    after_default = _shock_values(m)
    _restore(m, saved)
    got = apply_shock(m, {"imptx": uniform(10)})
    assert set(got) == set(want) == {"imptx"}
    assert set(got["imptx"]) == set(want["imptx"])
    for cell, v in want["imptx"].items():
        assert got["imptx"][cell] == pytest.approx(v, rel=1e-12, abs=1e-14)
    assert _shock_values(m).keys() == after_default.keys()
    for k, (v, fixed) in after_default.items():
        gv, gfixed = _shock_values(m)[k]
        assert gv == pytest.approx(v, rel=1e-12, abs=1e-14) and gfixed == fixed


def test_uniform_3_is_power_times_1_03(m):
    saved = _shock_values(m)
    default_cells = set(apply_default_tariff(m, gtap_mode=True)["imptx"])
    _restore(m, saved)
    got = apply_shock(m, {"imptx": uniform(3)})["imptx"]
    assert set(got) == default_cells
    for cell, v in got.items():
        chk = float(value(m.imptx[(*cell, "check")]))
        assert v == pytest.approx((1 + chk) * 1.03 - 1, rel=1e-12, abs=1e-14)
        assert value(m.imptx[(*cell, "shock")]) == v


def test_uniform_no_cell_raises(m, monkeypatch):
    before = _shock_values(m)
    monkeypatch.setattr(shocks, "_can_take_shock", lambda *a: False)
    with pytest.raises(ValueError, match="uniform"):
        apply_shock(m, {"imptx": uniform(3)})
    assert _shock_values(m) == before


def test_uniform_mixed_with_cells(m):
    cell = _cell(m, "aft")
    chk = float(value(m.aft[(*cell, "check")]))
    got = apply_shock(m, {"imptx": uniform(3), "aft": {cell: 10.0}})
    assert got["aft"] == {cell: pytest.approx(chk * 1.1, rel=1e-15)}
    assert got["imptx"]


@pytest.mark.parametrize("pct", [-100.0, -150.0, 0.0])
def test_uniform_bad_percent_raises_and_fixes_nothing(m, pct):
    before = _shock_values(m)
    with pytest.raises(ValueError):
        apply_shock(m, {"imptx": uniform(pct)})
    assert _shock_values(m) == before


def test_uniform_is_a_percent_not_a_level(m):
    with pytest.raises(ValueError, match="uniform"):
        apply_shock(m, {"imptx": uniform(3)}, levels=True)


@pytest.mark.parametrize("bad", ["3", None, True])
def test_uniform_rejects_non_number(bad):
    with pytest.raises(TypeError):
        uniform(bad)


def test_uniform_refused_where_cells_are_read():
    from equilibria.family.shock_session import _write_levels

    with pytest.raises(TypeError, match="uniform"):
        _write_levels(None, {"imptx": uniform(3)})  # ty: ignore[invalid-argument-type]
