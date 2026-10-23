"""Calibracion de betaP/betaG/betaS cuando la elasticidad del gasto privado no es 1.

``phiP = Σ xcshr·eh``. Con el INCPAR estandar da 1 en el dato, y entonces las beta son
las participaciones del ingreso. Con el INCPAR que el libro edita en
``climatechange.prm`` (ME9B/C/D) no da 1. Si las beta se dejan en las participaciones,
el check no es un equilibrio en el dato: el modelo se mueve hasta 1,2 % antes de
cualquier shock (``qgd`` de ROW).

GTAP (``gtapv7.tab``: UTILELAS, DPARPRIV/DPARGOV/DPARSAVE) calibra:

    phi = phiP·sp + sg + ss,   betaP = phiP·sp/phi,   betaG = sg/phi,   betaS = ss/phi

GAMS (``cal.gms``) resuelve ``betaCal`` solo en el periodo base y deja el check con
las participaciones crudas: GDX ME9C, ROW, betaP base 0,6130 / check 0,6322.

Oraculo: los valores esperados salen de la formula de GTAP aplicada al phiP y a
yc/yg/rsav/regY del periodo base del GDX de GAMS de ME9C (2026-10-22, a mano). La
prueba de comportamiento (el check reproduce el dato) usa las soluciones GEMPACK de
``tests/fixtures/gtap7_gempack/burfisher``, donde pre = dato.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params

# GDX ME9C (gen_burfisher_gams.py), periodo base: phiP, y sp/sg/ss = yc/yg/rsav ÷ regY.
GAMS_ME9C_BASE = {
    "USA": {
        "phip": 0.999178888029,
        "phi": 0.999361850437,
        "betap": 0.777034988765,
        "betag": 0.176521537774,
        "betas": 0.0464434734612,
    },
    "ROW": {
        "phip": 0.981605307677,
        "phi": 0.988371044267,
        "betap": 0.627863264344,
        "betag": 0.200087630314,
        "betas": 0.172049105342,
    },
}
SHARES_ME9C = {  # sp, sg, ss del mismo GDX
    "USA": (0.777177273789, 0.176408890632, 0.0464138355789),
    "ROW": (0.63219082597, 0.197760820118, 0.170048353912),
}


def _calib(prm: str) -> dict:
    from equilibria.blocks.gtap._derived_params import demand_income_params

    p = nus333_params(prm)
    return demand_income_params(p, p.sets, residual_region="ROW")


@pytest.mark.parametrize("r", ["USA", "ROW"])
def test_con_incpar_editado_las_beta_son_las_de_gtap(r: str) -> None:
    c = _calib("climatechange.prm")
    want = GAMS_ME9C_BASE[r]
    assert c["phip0"][(r,)] == pytest.approx(want["phip"], rel=1e-5)
    assert c["phi0"][(r,)] == pytest.approx(want["phi"], rel=1e-5)
    for b in ("betap", "betag", "betas"):
        assert c[b][(r,)] == pytest.approx(want[b], rel=1e-5), b


@pytest.mark.parametrize("r", ["USA", "ROW"])
def test_con_incpar_editado_el_gasto_privado_sigue_siendo_el_dato(r: str) -> None:
    """yc = betaP·(phi/phiP)·regY debe dar sp·regY: la calibracion no mueve el dato."""
    c = _calib("climatechange.prm")
    sp = SHARES_ME9C[r][0]
    assert c["betap"][(r,)] * c["phi0"][(r,)] / c["phip0"][(r,)] == pytest.approx(
        sp, rel=1e-5
    )


@pytest.mark.parametrize("r", ["USA", "ROW"])
def test_con_incpar_estandar_no_cambia_nada(r: str) -> None:
    """default.prm: phiP = 1 en el dato, y las beta son las participaciones."""
    c = _calib("default.prm")
    assert c["phip0"][(r,)] == pytest.approx(1.0, abs=1e-6)
    assert c["phi0"][(r,)] == pytest.approx(1.0, abs=1e-6)
    total = c["betap"][(r,)] + c["betag"][(r,)] + c["betas"][(r,)]
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
def test_el_check_de_me9_reproduce_el_dato() -> None:
    """Sin shock, con climatechange.prm, el check es el dato (pre de GEMPACK)."""
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    cb: Any = importlib.import_module("compare_burfisher_gempack")

    cb.rb.EXERCISES["CHECK_ME9"] = ("climatechange.prm", [("pop", ("USA",), 0.001)])
    try:
        levels, xscale = cb.solve("CHECK_ME9")
    finally:
        del cb.rb.EXERCISES["CHECK_ME9"]
    r = cb.run("tbl46a", levels, xscale)
    pre = [c for c in r.cells if c.period == "pre"]
    assert len(pre) == 207
    worst = max(pre, key=lambda c: c.d)
    assert worst.d < 1e-5, worst
