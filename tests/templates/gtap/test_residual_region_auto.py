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


def _base_investment(p):
    bm = p.benchmark
    return {
        str(r): sum(max(bm.get_investment_demand(r, i)[0], 1e-8) for i in p.sets.i)
        for r in p.sets.r
    }


def test_auto_picks_largest_base_investment():
    from equilibria.templates.gtap.gtap_block_model import resolve_residual_region

    p = _params("gtap7_15x10")
    yi = _base_investment(p)
    got = resolve_residual_region(p, p.sets, _closure("capFix"), "auto")
    assert got == max(yi, key=yi.get)
    # No vacuo: la elegida no es la última región, que es la que declaran los arneses.
    assert got != str(list(p.sets.r)[-1])


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
        m_d, code_d = _solve_shock(ds, 1, "capFix", residual_region=declared)
        a = _solution(m_d)
        m_o, code_o = _solve_shock(
            ds, 1, "capFix", residual_region="auto" if other == auto else other
        )
        b = _solution(m_o)
    assert code_d == 1 and code_o == 1
    assert m_o._residual_region == other != declared
    assert a.keys() == b.keys()
    # walras vale 0 en las dos; se compara en absoluto. El resto, relativo: 15x10
    # midió hasta 2,7e-6 (savf/chif de ROW), del orden de la tolerancia de PATH.
    malas = []
    for k in a:
        if k[0] == "walras":
            ok = abs(a[k] - b[k]) <= 1e-6
        else:
            ok = abs(a[k] - b[k]) <= 1e-5 * max(1.0, abs(a[k]))
        if not ok:
            malas.append((k, a[k], b[k]))
    assert not malas, f"{len(malas)} celdas difieren, p.ej. {malas[:5]}"
