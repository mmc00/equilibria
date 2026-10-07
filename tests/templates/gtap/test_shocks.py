"""shocks.apply_shock: el % de GEMPACK escrito en el periodo shock del ShockBlock.

Sin solver: se construye gtap7_3x3 una vez y se mira que celda queda en que valor.
"""

from __future__ import annotations

import pytest
from pyomo.environ import value

from equilibria.blocks.gtap.shock import SHOCK_INSTRUMENTS
from equilibria.templates.gtap import shocks
from equilibria.templates.gtap.shocks import (
    GEMPACK_KIND,
    apply_shock,
    check_shock_entered,
    shock_of,
)


def test_every_shockblock_instrument_has_its_gempack_kind():
    assert set(GEMPACK_KIND) == set(SHOCK_INSTRUMENTS)


@pytest.mark.parametrize(
    "kind,chk,want",
    [
        ("pct", 2.0, 2.2),  # x*(1+p)
        ("power", 0.1, 1.1 * 1.1 - 1),  # (1+t)*(1+p)-1
        ("power_kappa", 0.2, 1 - 0.8 / 1.1),  # 1-(1-k)/(1+p)
        ("power_fct", 0.1, (1 + 0.05 + 0.1) * 1.1 - 1 - 0.05),  # fcttx absorbe
    ],
)
def test_conversion_by_kind(kind, chk, want):
    assert shocks.shocked(kind, chk, 1.1, 0.05) == pytest.approx(want, rel=1e-15)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        shocks.shocked("tm_pct", 0.1, 1.1)


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


@pytest.fixture
def m(built):
    """El modelo con su ShockBlock restaurado al salir de cada test."""
    saved = {
        (n, k): float(value(getattr(built, n)[k]))
        for n in built._exogenous_instruments
        for k in getattr(built, n)
        if k[-1] == "shock"
    }
    yield built
    for (n, k), v in saved.items():
        getattr(built, n)[k].fix(v)


def _cell(m, name, kind_filter=None):
    """Una celda viva del instrumento (la primera que acepta un shock)."""
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


def test_a_model_without_shock_has_none(m):
    assert shock_of(m) == []


def test_pct_moves_only_the_shock_cell(m):
    cell = _cell(m, "lambdava")
    chk = value(m.lambdava[(*cell, "check")])
    base = value(m.lambdava[(*cell, "base")])
    got = apply_shock(m, {"lambdava": {cell: 10.0}})
    assert got == {"lambdava": {cell: pytest.approx(chk * 1.1, rel=1e-15)}}
    assert value(m.lambdava[(*cell, "shock")]) == pytest.approx(chk * 1.1, rel=1e-15)
    assert value(m.lambdava[(*cell, "check")]) == chk
    assert value(m.lambdava[(*cell, "base")]) == base
    assert m.lambdava[(*cell, "shock")].fixed
    assert shock_of(m) == [f"lambdava{cell}"]


def test_a_tax_shock_is_on_the_power(m):
    cell = _cell(m, "imptx")
    t = value(m.imptx[(*cell, "check")])
    apply_shock(m, {"imptx": {cell: 10.0}})
    assert value(m.imptx[(*cell, "shock")]) == pytest.approx((1 + t) * 1.1 - 1)


def test_fcttx_reads_fctts_from_the_model(m):
    cell = _cell(m, "fcttx")
    t = value(m.fcttx[(*cell, "check")])
    fs = m._fctts.get(cell, 0.0)
    apply_shock(m, {"fcttx": {cell: 10.0}})
    assert value(m.fcttx[(*cell, "shock")]) == pytest.approx(
        (1 + fs + t) * 1.1 - 1 - fs
    )


def test_applying_twice_is_applying_once(m):
    """La conversion parte del check: repetir no compone el shock."""
    cell = _cell(m, "aft")
    once = apply_shock(m, {"aft": {cell: 10.0}})
    twice = apply_shock(m, {"aft": {cell: 10.0}})
    assert once == twice


def test_levels_writes_the_value_as_given(m):
    cell = _cell(m, "aft")
    apply_shock(m, {"aft": {cell: 1.234}}, levels=True)
    assert value(m.aft[(*cell, "shock")]) == 1.234


def test_an_unshocked_aft_needs_no_check(m):
    check_shock_entered(m)  # sin shock de aft: nada que revisar


def test_a_lost_aft_shock_is_caught(m):
    """xft que no se movio con aft: el shock no llego a la solucion."""
    cell = _cell(m, "aft")
    k = (*cell, "shock")
    xft_before = value(m.xft[k])
    apply_shock(m, {"aft": {cell: 10.0}})
    m.xft[k].set_value(value(m.xft[(*cell, "check")]))  # como si eq_xfteq no estuviera
    try:
        with pytest.raises(RuntimeError, match="did not enter"):
            check_shock_entered(m)
    finally:
        m.xft[k].set_value(xft_before)
