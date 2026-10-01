"""nus333: el periodo base de equilibria es la base CALIBRADA de GAMS.

GAMS (compStat) no resuelve el periodo base: sus niveles son los de cal.gms y son
la referencia de los indices de Fisher del check y del shock (model.gms:1223-1320,
``sum(t0, ...)``). Si la base de equilibria arranca en otro punto, los indices
quedan corridos aunque el equilibrio real sea el mismo.

Caso medido: cal.gms:319 inicializa ``pefob = (1+exptx)*pe``; equilibria la
inicializaba en 1. En la base eso cambia ``gdpmp`` de ROW en
``sum (pefob-1)*xw = 0,148475`` (brecha medida 0,148469), y con ella
``rgdpmp``/``pgdpmp`` de ROW en el check y el shock de todos los ejercicios
(0,357pp en run_burfisher).

Oraculo: niveles del GDX de GAMS (``gams_shock/comp_shock.gms``, default.prm,
capFlex, sin shock: TBL45A en su periodo check).

LOCAL-only: SKIP si falta nus333.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration

# pefob(r,i,rp,'base') de GAMS = (1+exptx)*pe con pe=1 (cal.gms:319). Las celdas
# que no figuran valen 1.
PEFOB_BASE_GAMS = {
    ("USA", "MFG", "ROW"): 1.00293433823471,
    ("ROW", "AGR", "USA"): 0.999728263262577,
    ("ROW", "AGR", "ROW"): 0.999503576380661,
    ("ROW", "MFG", "USA"): 1.01587681523218,
    ("ROW", "MFG", "ROW"): 1.01356952214101,
}

# Periodo check de GAMS con default.prm (TBL45A_capFlex.gdx).
CHECK_GAMS = {
    "gdpmp": {"USA": 14.0617796650709, "ROW": 41.7695647044032},
    "rgdpmp": {"USA": 14.061780894911, "ROW": 41.7695622537076},
    "pgdpmp": {"USA": 0.999999912540231, "ROW": 1.0000000586718},
}
GDPMP_BASE_GAMS = {"USA": 14.0617801139496, "ROW": 41.7695611026001}

TOL_REL = 1e-6


def _params():
    from equilibria._local_refs import nus333_dir
    from equilibria.templates.gtap import GTAPParameters

    har = nus333_dir()
    if not (har / "basedata.har").exists():
        pytest.skip(f"nus333 no disponible en {har}")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / "default.prm",
        baserate_path=har / "baserate.har",
    )
    return p


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFlex",
        numeraire="pnum",
    )


@pytest.fixture(scope="module")
def solved():
    """Mismo armado que run_burfisher.solve_exercise, sin shock."""
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _params()
    gc = _closure()
    m, _ = build_block_model(p, p.sets, gc, "ROW", base_calibrated=True, ref_gdx=None)
    res = solve_multiperiod(
        m,
        p,
        gc,
        ref_gdx=None,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
        solve_check=True,
    )
    assert int(res["check"]["code"]) == 1, res["check"]
    return m, value


def test_pefob_base_es_la_calibrada_de_gams(solved):
    m, value = solved
    malas = []
    for r in m.r:
        for i in m.i:
            for rp in m.r:
                key = (r, i, rp, "base")
                if key not in m.pefob:
                    continue
                want = PEFOB_BASE_GAMS.get((r, i, rp), 1.0)
                got = float(value(m.pefob[key]))
                if abs(got - want) > TOL_REL * max(1.0, abs(want)):
                    malas.append(
                        f"pefob{key}: equilibria {got:.12g} vs GAMS {want:.12g}"
                    )
    assert not malas, "\n".join(malas)


def test_gdpmp_base_iguala_a_gams(solved):
    m, value = solved
    for r, want in GDPMP_BASE_GAMS.items():
        got = float(value(m.gdpmp[r, "base"]))
        assert abs(got / want - 1.0) < TOL_REL, (r, got, want)


def test_indices_del_pib_en_check_igualan_a_gams(solved):
    m, value = solved
    malas = []
    for var, cells in CHECK_GAMS.items():
        comp = getattr(m, var)
        for r, want in cells.items():
            got = float(value(comp[r, "check"]))
            if abs(got / want - 1.0) > TOL_REL:
                malas.append(
                    f"{var}[{r},check]: equilibria {got:.12g} vs GAMS {want:.12g}"
                )
    assert not malas, "\n".join(malas)


def test_pefob_inicial_es_la_de_cal_gms():
    """La raiz: el modelo recien construido (sin asentar ni resolver) ya trae
    pefob = (1+exptx)*pe en la base, como cal.gms:319."""
    from pyomo.environ import value

    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _params()
    m, _ = build_block_model(
        p, p.sets, _closure(), "ROW", base_calibrated=False, ref_gdx=None
    )
    pefob = cast(Any, m.pefob)
    malas = []
    for (r, i, rp), want in PEFOB_BASE_GAMS.items():
        got = float(value(pefob[r, i, rp, "base"]))
        if abs(got - want) > TOL_REL:
            malas.append(f"pefob[{r},{i},{rp},base]: {got:.12g} vs GAMS {want:.12g}")
    assert not malas, "\n".join(malas)
