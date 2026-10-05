"""nus333 / Burfisher ME9A: linea de base 2010-2050 con PIB real exogeno, contra GAMS.

``nus333/ME9A.EXP`` (``climatechange.prm``): ``swap qgdp(reg) = aoreg(reg)``, ``qgdp``
USA +109,6 / ROW +284,5, mas los shocks de ``pop`` y ``qe`` de ME9B. El PIB real queda
fijo y la productividad regional ``aoreg`` (la misma en las 3 actividades) pasa a
endogena. En equilibria el cierre se arma con ``@overwrite``, solo en el shock, igual
que en el notebook del ejercicio:

- ``ShockBlock``: ``axp[r,a]`` endogeno y el objetivo ``gdp_target`` (``b.target``):
  ``rgdpmp[shock] = gdp_target x rgdpmp[check]``;
- ``ClosureBlock``: ``eq_aoreg``, ``axp[r,a]/axp0[r,a] = axp[r,AGR]/axp0[r,AGR]``: el
  mismo % en las 3 actividades (un solo shifter por region, como ``aoreg``);
- el shock: ``gdp_target`` x 2,096 / 3,845 (``fix_instrument_shock``).

En ME9 el check NO reproduce la base (``rgdpmp`` ROW +0,030%), asi que el objetivo
tiene que anclarse al check, como en GAMS.

Oraculo GAMS (gams_shock/gen_gams.py, GDP: fila ``gdpeq`` emparejada con ``axpreg``,
``rgdpmp = rgdpmp_check*(1+%)``) vs GEMPACK ME9A.sl4:

    aoreg USA/ROW   GAMS +31,9496 / +42,4131   GEMPACK +31,9376 / +42,3080

La diferencia es la de ME9 entre GAMS y GEMPACK, no del oraculo: con el MISMO aoreg
(ME9B, 31,94 / 42,31) GAMS da rgdpmp +109,577 / +284,059 y GEMPACK qgdp +109,451 /
+284,317. GAMS ME9A es coherente con GAMS ME9B (31,94 -> 109,577; 31,950 -> 109,6).

Todas las celdas: los niveles GAMS de check y shock estan en
``tests/fixtures/nus333_me9a_gams_levels.json.gz`` (``_diff_core.gams_levels``) y se
comparan con ``run_burfisher.compare``.

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from tests.templates.gtap._nus333 import (
    ROOT,
    closure,
    gams_levels,
    nus333_params,
    pct,
    run_burfisher,
)

pytestmark = pytest.mark.integration

LEVELS = ROOT / "tests" / "fixtures" / "nus333_me9a_gams_levels.json.gz"
TOL_PP = 0.002
REGS = ("USA", "ROW")
ACTS = ("AGR", "MFG", "SER")
GDP = {"USA": 109.6, "ROW": 284.5}

# GAMS capFlex, % shock/check (oraculo de arriba).
ORACLE = {
    "rgdpmp": {("USA",): 109.6, ("ROW",): 284.5},
    "axp": {
        **{("USA", a): 31.949611 for a in ACTS},
        **{("ROW", a): 42.413121 for a in ACTS},
    },
    "xp": {
        ("USA", "AGR"): 52.712848,
        ("USA", "MFG"): 33.8334,
        ("USA", "SER"): 85.171157,
        ("ROW", "AGR"): 70.770028,
        ("ROW", "MFG"): 158.68932,
        ("ROW", "SER"): 212.975998,
    },
    "pft": {
        ("USA", "LABOR"): 5.472547,
        ("USA", "CAPITAL"): -13.766864,
        ("USA", "LAND"): 143.398993,
        ("ROW", "LABOR"): 39.09969,
        ("ROW", "CAPITAL"): -27.055608,
        ("ROW", "LAND"): 155.230698,
    },
}


def register_me9a_hooks() -> None:
    """Los hooks del notebook ``burfisher_exec_me9a.ipynb``."""
    from pyomo.environ import value

    from equilibria.blocks.gtap import ClosureBlock, ShockBlock, overwrite

    @overwrite(ShockBlock, period="shock")
    def baseline(b):
        for r in REGS:
            for a in ACTS:
                b.endogenous("axp", (r, a))
            b.target(
                "gdp_target", (r,), quantity=lambda m, r: m.rgdpmp[r], domains=("r",)
            )

    @overwrite(ClosureBlock, period="shock")
    def uniform_productivity(b):
        # El mismo % que AGR: axp/axp0 parejo (axp0 = base; en el check axp queda
        # fija en la base, asi que es el % de check a shock, como axpreg en GAMS).
        for r in REGS:
            for a in ("MFG", "SER"):
                b.equation(
                    "eq_aoreg",
                    (r, a),
                    lambda m, r, a: m.axp[r, a] / value(m.axp[r, a])
                    == m.axp[r, "AGR"] / value(m.axp[r, "AGR"]),
                    domains=("r", "a"),
                )


@pytest.fixture(scope="module")
def solved():
    from pyomo.environ import value

    from equilibria.blocks.gtap import overwrite
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    rb = run_burfisher()
    p = nus333_params("climatechange.prm")
    ac = closure()
    register_me9a_hooks()
    try:
        m, _ = cast(
            Any,
            build_block_model(
                p, p.sets, ac, "ROW", base_calibrated=False, ref_gdx=None
            ),
        )
        for r, pct in GDP.items():
            fix_instrument_shock(m, "gdp_target", (r,), factor=1 + pct / 100)
        # Los de ME9B salvo axp (que ahora es endogena).
        for name, idx, kind, pct in rb.EXERCISES["ME9B"][1]:
            if name != "axp":
                fix_instrument_shock(
                    m, name, idx, value=rb.level(m, p, name, idx, kind, pct)
                )
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


def test_productividad_fija_en_el_check(solved):
    """El cierre de @overwrite rige solo en el shock: en el check axp queda en su
    base (exogena) aunque el check no reproduzca la base."""
    m, _, _, value = solved
    for r in REGS:
        for a in ACTS:
            assert m.axp[(r, a, "check")].fixed, (r, a)
            assert float(value(m.axp[(r, a, "check")])) == pytest.approx(
                float(value(m.axp[(r, a, "base")])), rel=1e-12
            )


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
    r = run_burfisher().compare(m, p, gams_levels(LEVELS, "ME9A"))
    for period in ("check", "shock"):
        got = r[period]
        assert got["cells"] > 500, (period, got["cells"])
        assert got["match_pct"]["0.1%"] == 100.0, (period, got["worst"])
        assert got["instr_bad"] == [], (period, got["instr_bad"])
