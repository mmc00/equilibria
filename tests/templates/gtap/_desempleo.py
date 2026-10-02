"""Lo comun de los tests del cierre de desempleo (TBL65B/ME3C) y de ``@overwrite``.

Los dos hooks son los del notebook ``notebooks/gtap7/burfisher_exec_tbl65b.ipynb``.
"""

from __future__ import annotations

import pytest


def nus333_params():
    """Parametros nus333 con ``default.prm``; SKIP si falta el dataset."""
    from equilibria._local_refs import nus333_dir
    from equilibria.templates.gtap import GTAPParameters

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / "default.prm",
        baserate_path=har / "baserate.har",
    )
    return p


def closure():
    """El cierre de los ejercicios de Burfisher (capFlex, CAPITAL sluggish)."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFlex",
        numeraire="pnum",
    )


def register_desempleo_hooks() -> None:
    """``aft[USA,LABOR]`` endogeno + salario real ``pft = pft0 * Tornqvist``."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import (
        ClosureBlock,
        ShockBlock,
        overwrite,
        ppriv_tornqvist,
    )

    @overwrite(ShockBlock)
    def desempleo(b):
        b.endogeno("aft", ("USA", "LABOR"))

    @overwrite(ClosureBlock)
    def salario_real(b):
        b.ecuacion(
            "eq_wreal",
            ("USA", "LABOR"),
            lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r),
        )
