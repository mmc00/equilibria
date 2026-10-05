"""Los hooks del cierre de desempleo (TBL65B/ME3C), comunes a sus tests y a los de
``@overwrite``. Datos y cierre: ``_nus333``.

Los dos hooks son los del notebook ``notebooks/gtap7/burfisher_exec_tbl65b.ipynb``.
"""

from __future__ import annotations


def register_desempleo_hooks() -> None:
    """``aft[USA,LABOR]`` endogeno + salario real ``pft = pft0 * Tornqvist``."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import (
        ClosureBlock,
        ShockBlock,
        overwrite,
        ppriv_tornqvist,
    )

    @overwrite(ShockBlock, period="shock")
    def unemployment(b):
        b.endogenous("aft", ("USA", "LABOR"))

    @overwrite(ClosureBlock, period="shock")
    def real_wage(b):
        b.equation(
            "eq_wreal",
            ("USA", "LABOR"),
            lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r),
        )
