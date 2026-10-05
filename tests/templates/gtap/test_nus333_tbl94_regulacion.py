"""nus333 / Burfisher TBL94: regulacion de la produccion de MFG en USA, contra GAMS.

``nus333/TBL94.EXP``: ``swap qca("MFG","MFG","USA") = to("MFG","MFG","USA")`` y
``shock qca = -1``. La produccion de MFG por la actividad MFG de USA queda en -1% y el
impuesto a la produccion ``to`` se vuelve endogeno: es el impuesto "sombra" de la
regulacion (Tabla 9.4). En equilibria el cierre se arma con ``@overwrite``, solo en el
shock, igual que en el notebook del ejercicio:

- ``ShockBlock``: ``prdtx_rai[USA,MFG,MFG]`` endogeno y el objetivo ``qca_target``
  (``b.target``): ``x[shock] = qca_target x x[check]``;
- el shock: ``qca_target`` x 0,99 (``fix_instrument_shock``).

Oraculo GAMS (gams_shock/gen_gams.py, QCA: fila ``qcaeq`` emparejada con ``prdtx``,
``x = x_check*0,99``) vs GEMPACK TBL94.sl4:

    to MFG MFG USA (potencia)  GAMS +2,2495   GEMPACK +2,2500
    qo AGR/MFG/SER USA         GAMS +1,3667/-1,0000/+0,1892   GEMPACK +1,3671/-1,0/+0,1892

y los flujos bilaterales a <=0,004pp; la oferta de margenes ``qst`` a <=0,036pp (Gragg
2-4-6 en el .EXP). El lector del .sl4 desalinea ``to`` y ``qca`` (parcialmente
exogenas: CUMS guarda solo las celdas endogenas); ``to`` se leyo de su unica celda.

Todas las celdas: los niveles GAMS de check y shock estan en
``tests/fixtures/nus333_tbl94_gams_levels.json.gz`` (``_diff_core.gams_levels``) y se
comparan con ``run_burfisher.compare``.

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from tests.templates.gtap._nus333 import (
    closure,
    gams_levels,
    nus333_params,
    pct,
    run_burfisher,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ROOT / "tests" / "fixtures" / "nus333_tbl94_gams_levels.json.gz"
TOL_PP = 0.002
CELL = ("USA", "MFG", "MFG")

# GAMS capFlex, % shock/check (oraculo de arriba).
ORACLE = {
    "x": {CELL: -1.0},
    "xp": {
        ("USA", "AGR"): 1.366686,
        ("USA", "MFG"): -1.0,
        ("USA", "SER"): 0.189215,
        ("ROW", "AGR"): -0.200607,
        ("ROW", "MFG"): -0.0259,
        ("ROW", "SER"): 0.026072,
    },
    "rore": {("USA",): -0.794193, ("ROW",): -0.794193},
    "regy": {("USA",): -2.233627, ("ROW",): 1.160079},
    "pi": {("USA",): -1.12471, ("ROW",): 1.109369},
}
# El impuesto sombra: % de cambio en la potencia 1+to.
SHADOW_TAX_POWER = 2.249523


def register_tbl94_hooks() -> None:
    """Los hooks del notebook ``burfisher_exec_tbl94.ipynb``."""
    from equilibria.blocks.gtap import ShockBlock, overwrite

    @overwrite(ShockBlock, period="shock")
    def regulation(b):
        b.endogenous("prdtx_rai", CELL)
        b.target(
            "qca_target",
            CELL,
            quantity=lambda m, r, a, i: m.x[r, a, i],
            domains=("r", "a", "i"),
        )


@pytest.fixture(scope="module")
def solved():
    from pyomo.environ import value

    from equilibria.blocks.gtap import overwrite
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    p = nus333_params()
    ac = closure()
    register_tbl94_hooks()
    try:
        m, _ = cast(
            Any,
            build_block_model(
                p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
            ),
        )
        fix_instrument_shock(m, "qca_target", CELL, factor=0.99)
        res = solve_multiperiod(
            m,
            p,
            ac,
            ref_gdx=None,
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            mode="gtap",
            solve_check=True,
        )
        yield m, p, res, value
    finally:
        overwrite.clear()


def test_resuelve(solved):
    _, _, res, _ = solved
    for t in ("check", "shock"):
        assert int(res[t]["code"]) == 1, (t, res[t])


def test_impuesto_sombra(solved):
    """``to`` se ajusta para que la produccion baje 1%; en el check queda fijo en su
    base (el cierre de @overwrite rige solo en el shock)."""
    m, _, _, value = solved
    assert m.prdtx_rai[(*CELL, "check")].fixed
    t = {p: float(value(m.prdtx_rai[(*CELL, p)])) for p in ("base", "check", "shock")}
    got = 100.0 * ((1 + t["shock"]) / (1 + t["check"]) - 1)
    assert abs(got - SHADOW_TAX_POWER) < TOL_PP
    assert abs(t["check"] - t["base"]) < 1e-6


def test_iguala_a_gams(solved):
    m, _, _, value = solved
    malas = []
    for var, cells in ORACLE.items():
        for key, want in cells.items():
            got = pct(m, var, key)
            if abs(got - want) > TOL_PP:
                malas.append(f"{var}{key}: equilibria {got:+.6f} vs GAMS {want:+.6f}")
    n = sum(len(c) for c in ORACLE.values())
    assert not malas, f"{len(malas)}/{n} celdas fuera de {TOL_PP}pp:\n" + "\n".join(
        malas
    )


def test_todas_las_celdas_contra_gams(solved):
    """Check y shock, todas las celdas, contra los niveles de GAMS (tolerancia 0,1%)."""
    m, p, _, _ = solved
    r = run_burfisher().compare(m, p, gams_levels(LEVELS, "TBL94"))
    for period in ("check", "shock"):
        got = r[period]
        assert got["cells"] > 500, (period, got["cells"])
        assert got["match_pct"]["0.1%"] == 100.0, (period, got["worst"])
        assert got["instr_bad"] == [], (period, got["instr_bad"])
