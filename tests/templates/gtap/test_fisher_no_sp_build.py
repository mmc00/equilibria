"""F3: build_equations_fisher no debe construir un modelo de periodo simple.

Construia uno ENTERO para leer un solo Param (`xscale`) y lo tiraba acto
seguido (`del _sp_tmp`).  En 3x3 cuesta poco; en 20x41 ese SP es el monolito
completo (3,4M celdas), y mientras exista el monolito es dependencia de
runtime.

`xscale` sale de los params sin modelo: `blocks/gtap/_derived_params.xscale_data`.
Medido identico celda a celda (104 celdas, 3 datasets, diferencia 0,0).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


def _mp(p):
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    rr = list(p.sets.r)[-1]
    return GTAPMultiPeriodModel(p.sets, p, None, residual_region=rr)


def test_build_equations_fisher_does_not_build_a_single_period_model():
    """El contrato: cero llamadas a _build_sp durante build_equations_fisher."""
    p = _load_3x3_params()
    mp = _mp(p)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)

    calls = []
    orig = type(mp)._build_sp
    try:
        type(mp)._build_sp = lambda self: (calls.append(1), orig(self))[1]
        mp.build_equations_fisher(m)
    finally:
        type(mp)._build_sp = orig

    assert calls == [], (
        f"build_equations_fisher construyo {len(calls)} modelo(s) de periodo "
        "simple; deberia leer xscale de los params (xscale_data)"
    )


def test_xscale_floats_survive_and_match_the_params():
    """El corte no puede cambiar los numeros: m._xscale_floats sigue igual.

    Es la red de este cambio — si _xscale_floats se poblara mal, las filas
    Fisher quedarian con otra escala y el seeder de seed_and_solve saltaria
    `xd` en silencio (por eso se guarda en el modelo).
    """
    from equilibria.blocks.gtap import _derived_params as dp

    p = _load_3x3_params()
    mp = _mp(p)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)

    got = getattr(m, "_xscale_floats", None)
    assert got, "build_equations_fisher debe dejar _xscale_floats en el modelo"

    want = dp.xscale_data(p, p.sets)
    for r in p.sets.r:
        for a in p.sets.a:
            exp = max(float(want.get((r, a), 1.0)), 1e-12)
            assert got[(r, a)] == pytest.approx(exp, abs=1e-12), (
                f"xscale[{r},{a}]: {got[(r, a)]} != {exp}"
            )
