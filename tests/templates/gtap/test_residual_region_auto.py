"""residual_region="auto": la región residual la elige el modelo (mayor inversión base).

Con walras libre, el paso de Newton del arranque del shock descarga el desbalance de
Walras en yi[rres] (eq_yi se omite en la residual; eq_walras suma sólo rres). En el
20x41 la residual declarada, ROW, es diminuta (yi=0,086): el paso le pide +4,37 (51x),
1.768 variables de ROW pasan a negativo y PATH se estanca. Con rres=USA el shock
converge en 0,4 min. Medido 2026-10-13 (dev-tools notes/cge, matriz 20x41).

Bajo capFix elegir otra residual no cambia la economía: savf[r] = pigbl·savf_bar[r]
fuera de la residual y los savf_bar suman cero en la base, así que la residual queda
en pigbl·savf_bar[rres] igual. Estos tests lo comprueban celda a celda en vez de
suponerlo.
"""

import contextlib
import io
from pathlib import Path

import pytest
from pyomo.environ import Var, value

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

ROOT = Path(__file__).resolve().parents[3]


def _params(ds):
    from equilibria.templates.gtap import GTAPParameters

    d = ROOT / "datasets" / ds
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def _closure(sf):
    return GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish", savf_flag=sf
    )


def test_auto_picks_largest_base_investment():
    from equilibria.templates.gtap.gtap_block_model import resolve_residual_region

    # 15x10: CHN tiene la mayor inversion base; la residual que declaran los
    # arneses es la ultima region (ROW), asi que "auto" la cambia de verdad.
    p = _params("gtap7_15x10")
    assert str(list(p.sets.r)[-1]) == "ROW"
    assert resolve_residual_region(p, p.sets, _closure("capFix"), "auto") == "CHN"


def test_declared_region_passes_through():
    from equilibria.templates.gtap.gtap_block_model import resolve_residual_region

    p = _params("gtap7_3x3")
    assert resolve_residual_region(p, p.sets, _closure("capFix"), "ROW") == "ROW"


@pytest.mark.parametrize("sf", ["capFlex", "capSFix", "capShrFix", "capFixDp"])
def test_auto_rejected_outside_capfix(sf):
    # Sólo bajo capFix está demostrado que la residual no cambia el equilibrio.
    from equilibria.templates.gtap.gtap_block_model import resolve_residual_region

    p = _params("gtap7_3x3")
    with pytest.raises(ValueError, match="capFix"):
        resolve_residual_region(p, p.sets, _closure(sf), "auto")


def _solution(m):
    out = {}
    for v in m.component_objects(Var, active=True):
        for i in v:
            per = i[-1] if isinstance(i, tuple) else i
            if per in ("check", "shock"):
                x = value(v[i], exception=False)
                if x is not None:
                    out[(v.name, i)] = float(x)
    return out


@pytest.mark.needs_path
@pytest.mark.parametrize(
    "ds",
    [
        "gtap7_3x3",
        "gtap7_5x5",
        "gtap7_10x7",
        pytest.param("gtap7_15x10", marks=pytest.mark.slow),
    ],
)
def test_residual_choice_same_equilibrium(ds):
    """Lo que justifica "auto": bajo capFix la residual no cambia el equilibrio.

    Se compara la declarada (última región) contra otra: la de "auto" cuando difiere
    (15x10) y si no la primera región (en 3x3/5x5/10x7 ROW ya es la de mayor
    inversión, así que "auto" no la cambia).
    """
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    from equilibria.templates.gtap.gtap_block_model import resolve_residual_region

    p = _params(ds)
    declared = str(list(p.sets.r)[-1])
    auto = resolve_residual_region(p, p.sets, _closure("capFix"), "auto")
    other = auto if auto != declared else str(list(p.sets.r)[0])
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        m_declarada, code_declarada = _solve_shock(
            ds, 1, "capFix", residual_region=declared
        )
        sol_declarada = _solution(m_declarada)
        # Si "auto" elige otra region, se pasa "auto" para probar el camino entero.
        pedida = "auto" if other == auto else other
        m_otra, code_otra = _solve_shock(ds, 1, "capFix", residual_region=pedida)
        sol_otra = _solution(m_otra)
    assert code_declarada == 1 and code_otra == 1
    assert m_otra._residual_region == other != declared
    assert sol_declarada.keys() == sol_otra.keys()
    # walras vale 0 en las dos; se compara en absoluto. El resto, relativo: 15x10
    # midió hasta 2,7e-6 (savf/chif de ROW), del orden de la tolerancia de PATH.
    malas = []
    for k, x in sol_declarada.items():
        y = sol_otra[k]
        if k[0] == "walras":
            ok = abs(x - y) <= 1e-6
        else:
            ok = abs(x - y) <= 1e-5 * max(1.0, abs(x))
        if not ok:
            malas.append((k, x, y))
    assert not malas, f"{len(malas)} celdas difieren, p.ej. {malas[:5]}"


# Toda entrada que recibe una residual resuelve "auto". Un "auto" sin resolver no
# casa con ninguna region: eq_walras sumaria todas y eq_yi no omitiria ninguna,
# un modelo equivocado sin error (hallazgo de la review de #117).


def test_multiperiod_model_resolves_auto():
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel

    p = _params("gtap7_15x10")
    mp = GTAPBlockMultiPeriodModel(
        p.sets, p, _closure("capFix"), residual_region="auto"
    )
    assert mp.residual_region == "CHN"


def test_monolith_resolves_and_rejects_auto():
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    p = _params("gtap7_15x10")
    eqs = GTAPModelEquations(p.sets, p, _closure("capFix"), residual_region="auto")
    assert eqs.residual_region == "CHN"
    with pytest.raises(ValueError, match="capFix"):
        GTAPModelEquations(p.sets, p, _closure("capFlex"), residual_region="auto")


def _sin_ruido(fn, *a, **k):
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return fn(*a, **k)


def test_single_period_skips_eq_yi_of_the_chosen_region():
    from pyomo.environ import Constraint

    from equilibria.templates.gtap.gtap_block_model import build_block_single_period

    p = _params("gtap7_15x10")
    sp = _sin_ruido(build_block_single_period, p, p.sets, _closure("capFix"), "auto")
    assert sp._residual_region == "CHN"
    con = sp.component("eq_yi")
    assert isinstance(con, Constraint)
    con_yi = {str(k[0] if isinstance(k, tuple) else k) for k in con if con[k].active}
    assert con_yi == {str(r) for r in p.sets.r} - {"CHN"}


@pytest.mark.parametrize("variante", ["sparse", "loglevels"])
def test_variant_builders_resolve_auto(variante):
    if variante == "sparse":
        from equilibria.templates.gtap_sparse.multiperiod import (
            build_sparse_model_mp as build,
        )
    else:
        from equilibria.templates.gtap_loglevels.multiperiod import (
            build_loglevels_model_mp as build,
        )

    p = _params("gtap7_15x10")
    out = _sin_ruido(build, p, p.sets, _closure("capFix"), "auto")
    m, mp = out[0], out[1]
    assert m._residual_region == mp.residual_region == "CHN"
