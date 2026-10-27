"""Que celdas de ``imptx`` pueden llevar un shock (``check_shock_cell``).

Con ifSUB=0 el arancel entra por ``eq_pmeq``.  Con ifSUB=1 esa fila se apaga y el
precio de importacion va inline (macro M_PM) en ``eq_xweq``, que lee la Var
``imptx`` de la misma celda: la celda tiene que aceptarse.  Lo que se acepta tiene
que ser una fila viva que LEA la Var, no solo una fila viva con el mismo indice.
Sin solver: se construye gtap7_3x3.
"""

from __future__ import annotations

import pytest

from equilibria.blocks.gtap.periods import SHOCK
from equilibria.templates.gtap.instruments import check_shock_cell


def _build(if_sub: bool):
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
        if_sub=if_sub,
        numeraire="pnum",
    )
    m, _ = build_block_model(p, p.sets, gc, list(p.sets.r)[-1])
    return m


def _traded_cells(m):
    return [k[:-1] for k in m.eq_xweq if k[-1] == SHOCK and m.eq_xweq[k].active]


def test_ifsub1_accepts_every_traded_imptx_cell():
    m = _build(if_sub=True)
    cells = _traded_cells(m)
    assert cells
    for cell in cells:
        check_shock_cell(m, "imptx", cell)


def test_ifsub0_still_needs_eq_pmeq():
    """eq_xweq vive pero bajo ifSUB=0 lee la Var pm, no imptx: no alcanza."""
    m = _build(if_sub=False)
    cell = _traded_cells(m)[0]
    check_shock_cell(m, "imptx", cell)
    m.eq_pmeq[(*cell, SHOCK)].deactivate()
    try:
        with pytest.raises(ValueError, match="would not enter the model"):
            check_shock_cell(m, "imptx", cell)
    finally:
        m.eq_pmeq[(*cell, SHOCK)].activate()
