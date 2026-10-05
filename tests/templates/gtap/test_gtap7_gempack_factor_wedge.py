"""Cierre gtap7_gempack: la cuna de los factores es la de los datos, en todos lados.

Los datos de GTAP cumplen EVFP = EVFB + FTRV + FBEP exacto (FBEP <= 0: el subsidio
BAJA lo que paga el productor). GAMS (cal.gms) usa fctts = -FBEP/EVFB, que lo SUBE:
1 + fcttx + fctts = (EVFB + FTRV - FBEP)/EVFB. Medido en gtap7_15x10 (2026-10-05):
la identidad de los datos cierra a 3e-8; la de GAMS falla en 0,46%.

El cierre gempack (va_subsidy_basis="gempack") solo cambiaba la valoracion del VA
(_va_wedge); el precio del factor (eq_pfaeq, via fctts), la calibracion (gx/ava) y la
recaudacion seguian con el signo de GAMS. Resultado: el check se alejaba de los datos
(peso de la tierra en el VA de EU_28 Food 10,4% contra 7,0% de EVFP en 3x3) y el
match contra GEMPACK de 15x10 quedaba en 94%. Como GTAPv7.jl (cuna neta EVFP/EVFB,
VA calibrado en EVFP), bajo gempack todo usa la cuna de los datos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets" / "gtap7_3x3"


def _params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATA / "basedata.har",
        sets_path=DATA / "sets.har",
        default_path=DATA / "default.prm",
        baserate_path=DATA / "baserate.har",
    )
    return p


def _har(header: str) -> dict:
    from equilibria.babel.har.reader import read_har

    x = read_har(str(DATA / "basedata.har"))[header]
    f, a, r = (list(e) for e in x.set_elements)
    return {
        (r[k], f[i], a[j]): float(x.array[i, j, k])
        for i in range(len(f))
        for j in range(len(a))
        for k in range(len(r))
    }


def test_cuna_gempack_reproduce_evfp():
    """1 + fcttx + fctts = EVFP/EVFB en cada celda con EVFB > 0."""
    from equilibria.templates.gtap.gtap_parameters import factor_wedge_rates

    p = _params()
    bm, evfp, evfb_har = p.benchmark, _har("EVFP"), _har("EVFB")
    celdas = 0
    for (r, f, a), evfb in bm.evfb.items():
        if float(evfb or 0.0) <= 0.0:
            continue
        fcttx, fctts = factor_wedge_rates(bm, r, f, a, "gempack")
        esperado = evfp[(r, f, a)] / evfb_har[(r, f, a)]
        assert 1.0 + fcttx + fctts == pytest.approx(esperado, rel=1e-6), (r, f, a)
        celdas += 1
    assert celdas > 0


def test_cuna_gams_sin_cambios():
    """El cierre por defecto sigue con el signo de GAMS: fctts = -FBEP/EVFB."""
    from equilibria.templates.gtap.gtap_parameters import factor_wedge_rates

    bm = _params().benchmark
    subsidiadas = [k for k, v in bm.fbep.items() if float(v or 0.0) != 0.0]
    assert subsidiadas, "3x3 trae FBEP distinto de cero"
    for r, f, a in subsidiadas:
        evfb = float(bm.evfb[(r, f, a)])
        fcttx, fctts = factor_wedge_rates(bm, r, f, a, "gams")
        assert fcttx == pytest.approx(float(bm.ftrv.get((r, f, a), 0.0)) / evfb)
        assert fctts == pytest.approx(-float(bm.fbep[(r, f, a)]) / evfb)


@pytest.mark.integration
@pytest.mark.needs_path
def test_check_gempack_reproduce_el_peso_de_los_factores():
    """Con gempack, el check resuelto tiene el peso de cada factor en el VA de EVFP."""
    sys.path.insert(0, str(ROOT / "src"))
    from pyomo.environ import value as V

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _params()
    ac = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )
    m, _ = build_block_model(
        p, p.sets, ac, list(p.sets.r)[-1], base_calibrated=True, ref_gdx=None
    )
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode="gtap",
        solve_check=True,
    )
    assert int(res["check"]["code"]) == 1

    evfp = _har("EVFP")
    pfa = cast(Any, m.pfa)
    xf = cast(Any, m.xf)
    peor = (0.0, None)
    for r in p.sets.r:
        for a in p.sets.a:
            datos = {f: evfp.get((r, f, a), 0.0) for f in p.sets.f}
            total_datos = sum(datos.values())
            if total_datos <= 0.0:
                continue
            modelo = {}
            for f in p.sets.f:
                try:
                    modelo[f] = float(V(pfa[r, f, a, "check"])) * float(
                        V(xf[r, f, a, "check"])
                    )
                except KeyError:
                    modelo[f] = 0.0
            total_modelo = sum(modelo.values())
            for f in p.sets.f:
                d = abs(modelo[f] / total_modelo - datos[f] / total_datos)
                if d > peor[0]:
                    peor = (d, (r, f, a))
    assert peor[0] < 1e-4, f"peso de factor en el VA lejos de EVFP: {peor}"
