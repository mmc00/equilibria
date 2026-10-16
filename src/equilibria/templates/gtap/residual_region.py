"""residual_region="auto": la region residual la elige el modelo.

Vive aca, abajo de los constructores, para que TODA entrada que recibe una
residual la resuelva: build_block_model, build_block_single_period,
GTAPMultiPeriodModel, GTAPModelEquations, gtap_sparse/gtap_loglevels,
calibrate_base y _build_sp_reference. Un "auto" sin resolver no casa con
ninguna region: eq_walras sumaria TODAS y eq_yi no omitiria ninguna, un modelo
equivocado sin error.
"""

from __future__ import annotations

from typing import Any

AUTO = "auto"


def resolve_residual_region(
    params: Any, sets: Any, closure: Any, residual_region: str
) -> str:
    """Return the residual region, choosing it when ``residual_region == "auto"``.

    "auto" picks the region with the largest base investment. With walras free, the
    Newton step at the start of the shock loads the Walras imbalance onto yi[rres]
    (eq_yi is skipped there and eq_walras sums only rres). On gtap7_20x41 the usual
    residual, ROW, is tiny (yi=0.086): the step asks it for +4.37, ROW goes negative
    and PATH stalls; with "auto" (CHN, the largest) the shock converges in 0.5 min.

    Only under capFix: there savf = pigbl·savf_bar outside rres and the savf_bar sum
    to zero, so rres lands on pigbl·savf_bar[rres] too and the equilibrium does not
    depend on the choice (tests/templates/gtap/test_residual_region_auto.py checks it
    cell by cell). Other closures make rres absorb a different quantity.
    """
    if residual_region != AUTO:
        return residual_region
    savf_flag = str(getattr(closure, "savf_flag", "capFix"))
    if savf_flag != "capFix":
        raise ValueError(
            f'residual_region="auto" is only valid under capFix (got {savf_flag}): '
            "elsewhere the residual region changes the equilibrium; declare it."
        )
    yi = {str(r): params.benchmark.base_investment(r, sets.i) for r in sets.r}
    return max(yi, key=lambda r: yi[r])
