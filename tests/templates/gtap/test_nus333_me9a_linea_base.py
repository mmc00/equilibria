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
- el shock: ``gdp_target`` +109,6% / +284,5% (``apply_shock``).

El objetivo se ancla al check, como en GAMS. Con la calibracion de betaP de GTAP
(``phiP = Σ xcshr·eh``, test_nus333_cde_betacal.py) el check es el dato.

Oraculo: GEMPACK ME9A (``tests/fixtures/gtap7_gempack/burfisher/me9a.sl4.gz``,
``tools/gempack-oracle``), todas las celdas en niveles con
``compare_burfisher_gempack``, y ``aoreg`` USA/ROW +31,9376 / +42,3080.

Ya no se compara contra GAMS: ``cal.gms`` calibra betaP solo en el periodo base y deja
el check con las participaciones crudas, asi que con ``climatechange.prm`` su check se
aparta del dato (qgd ROW 1,2 %). equilibria calibra como GTAP y se aparta de GAMS a
proposito. Medido 2026-10-22 contra GEMPACK: antes pre 87 % / post 77 % de las celdas a
0,5 %, max 1,7 %; ahora pre 100 % y post 100 % a 1 %, max 0,64 %. Los niveles de GAMS
(``tests/fixtures/nus333_me9a_gams_levels.json.gz``) quedan solo como entrada del test
de los comparadores.

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
REGS = ("USA", "ROW")
ACTS = ("AGR", "MFG", "SER")
GDP = {"USA": 109.6, "ROW": 284.5}

# GEMPACK ME9A.sl4: aoreg, % de cambio.
AOREG_GEMPACK = {"USA": 31.9376, "ROW": 42.3080}


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
    from equilibria.templates.gtap.shocks import apply_shock

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
        # El PIB objetivo y los de ME9B salvo axp (que ahora es endogena).
        me9b = [s for s in rb.EXERCISES["ME9B"][1] if s[0] != "axp"]
        apply_shock(
            m,
            {"gdp_target": {(r,): pct for r, pct in GDP.items()}, **rb.as_shock(me9b)},
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


def _gempack(solved):
    import importlib
    import sys

    m, p, _, _ = solved
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    cb: Any = importlib.import_module("compare_burfisher_gempack")
    from equilibria.blocks.gtap._derived_params import xscale_data

    return cb.run("ME9A", cb.cg.model_levels(m), dict(xscale_data(p, p.sets)))


@pytest.mark.parametrize("r", REGS)
def test_aoreg_iguala_a_gempack(solved, r):
    """La productividad que alcanza el PIB objetivo, en niveles (1 + %/100)."""
    m, _, _, _ = solved
    got = 1 + pct(m, "axp", (r, "AGR")) / 100
    want = 1 + AOREG_GEMPACK[r] / 100
    assert got == pytest.approx(want, rel=1e-3)


def test_todas_las_celdas_contra_gempack(solved):
    """pre = dato (el check sin shock); post a menos de 1 % (max medido 0,64 %)."""
    r = _gempack(solved)
    for period, tol in (("pre", 1e-5), ("post", 1e-2)):
        cells = [c for c in r.cells if c.period == period]
        assert len(cells) == 207, period
        worst = max(cells, key=lambda c: c.d)
        assert worst.d < tol, (period, worst)
    assert not r.unmatched


def test_equilibria_compare_da_lo_mismo_que_el_comparador_del_gate(solved):
    """La Tabla 3 con equilibria.compare (run_burfisher.compare_module) cuenta las
    mismas celdas y da los mismos % que el comparador de los gates, que se escribio
    antes que el modulo y hace de oraculo (#135)."""
    m, p, _, _ = solved
    levels = gams_levels(LEVELS, "ME9A")
    rb = run_burfisher()
    old = rb.compare(m, p, levels)
    new, report = rb.compare_module(m, p, levels)
    for period in ("check", "shock"):
        assert new[period]["cells"] == old[period]["cells"], period
        assert new[period]["match_pct"] == old[period]["match_pct"], period
        assert new[period]["instr_cells"] == old[period]["instr_cells"], period
    assert report.excluded
    assert all(v.lower() in rb.SKIP | rb.RF for v in report.excluded)
