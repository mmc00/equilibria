"""Make-price start values of the production block, as GAMS ``cal.gms`` sets them.

``xFlag(r,a,i)`` is ``makb > 0`` (cal.gms:285-287). Where an activity makes the
commodity, ``p = ps/(1+prdtx)`` with ``prdtx = makb/maks - 1``; where it does not,
``p = 1`` (cal.gms:293-294) and ``prdtx = 0`` (cal.gms:290-291, ``maks`` is 0), so
``pp = (1+prdtx)·p = 1``. GEMPACK has no level there: ``ps`` only enters weighted by
the zero make share (``E_po``, ``E_qca``).
"""

from __future__ import annotations

import pytest

from equilibria.sources import dataset_dir, has_dataset

DATASET = "gtap7_10x7"


@pytest.fixture(scope="module")
def block():
    if not has_dataset(DATASET):
        pytest.skip(f"dataset missing: {DATASET}")
    from equilibria.blocks.gtap.production_supply import ProductionSupplyBlock
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(DATASET)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return ProductionSupplyBlock(sets=p.sets, params=p)


def _cells(block, made: bool):
    bm = block.params.benchmark
    out = [
        (r, a, i)
        for r in block.sets.r
        for a in block.sets.a
        for i in block.sets.i
        if (float(bm.makb.get((r, a, i), 0.0) or 0.0) > 0) == made
    ]
    assert out, f"no {'made' if made else 'unmade'} cells in {DATASET}"
    return out


def test_unmade_pairs_start_at_one_with_no_output_tax(block):
    bad = [
        (c, block._p_rai_init(*c), block._prdtx(*c))
        for c in _cells(block, made=False)
        if block._p_rai_init(*c) != 1.0 or block._prdtx(*c) != 0.0
    ]
    assert bad == []


def test_made_pairs_keep_the_per_commodity_tax(block):
    bm = block.params.benchmark
    for c in _cells(block, made=True):
        makb, maks = bm.makb[c], bm.maks.get(c, 0.0)
        if maks > 0:
            assert block._prdtx(*c) == pytest.approx(makb / maks - 1.0, rel=1e-15)
            assert block._p_rai_init(*c) == pytest.approx(
                1.0 / (1.0 + block._prdtx(*c)), rel=1e-15
            )


def test_the_shock_instrument_has_no_output_tax_on_unmade_pairs(block):
    """``prdtx_rai`` (the ShockBlock instrument) starts where ``_prdtx`` does."""
    from equilibria.blocks.gtap._derived_params import prdtx_rai_data

    data = prdtx_rai_data(block.params, block.sets)
    bad = [(c, v) for c, v in data.items() if v != block._prdtx(*c)]
    assert bad == []
