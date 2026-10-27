"""El arancel +10% por defecto vive en el ShockBlock (cambio tariff-to-shock-block).

Oraculo: la solucion de GAMS de cada fixture (gtap7_3x3, los cuatro casos gtap /
altertax x ifSUB 0/1), leida con el lector GDX propio (sin GAMS instalado).

  * ``imptx[*,'shock']`` del modelo resuelto = el de GAMS. GAMS fija el arancel
    como instrumento (``tm.fx``): en gtap sobre la potencia, (1+t)*1,1-1, con la
    diagonal incluida; en altertax la tasa por 1,1.  El camino viejo dejaba la Var
    en su valor del check y metia el arancel reconstruyendo ecuaciones.
  * altertax: donde GAMS mueve ``xw`` mas de 0,1 %, equilibria lo mueve en el
    mismo sentido.  El camino viejo no metia el arancel en el solve: ``xw`` del
    shock quedaba en el del check (1e-7) y solo se parchaban ``pm`` y ``ytax``.

LOCAL: necesita el solver PATH; se salta sin el o sin el fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from equilibria._local_refs import path_capi_src

pytestmark = pytest.mark.integration

_PATH_CAPI_SRC = path_capi_src()
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

ROOT = Path(__file__).resolve().parents[3]
DATASET = "gtap7_3x3"
CASES = [("gtap", 0), ("gtap", 1), ("altertax", 0), ("altertax", 1)]


def _fixture_gdx(mode: str, ifsub: int) -> Path:
    if mode == "gtap":
        return ROOT / f"tests/fixtures/gtap7/{DATASET}/out_gtap_shock_ifsub{ifsub}.gdx"
    return (
        ROOT / f"tests/fixtures/gtap7_altertax/{DATASET}/out_altertax_ifsub{ifsub}.gdx"
    )


def _gams(gdx: Path, name: str) -> dict[tuple, float]:
    """Niveles de la Var ``name`` de GAMS, con los prefijos de set (c_, r_) fuera."""
    from equilibria.babel.gdx.reader import read_gdx, read_variable_values

    def strip(s):
        return s[2:] if len(s) > 2 and s[1] == "_" and s[0] in "acfr" else s

    raw = read_variable_values(read_gdx(gdx), name)
    return {tuple(strip(x) for x in k): v["level"] for k, v in raw.items()}


def _solve(mode: str, ifsub: int, gdx: Path):
    """El armado de test_gtap7_mcp_parity, sin shock en el modelo."""
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import PERIODS
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = ROOT / "datasets" / DATASET
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    if mode == "altertax":
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities

        pa = apply_altertax_elasticities(p, in_place=False)
        ac = GTAPClosureConfig(
            name="altertax",
            closure_type="MCP",
            capital_mobility="mobile",
            fix_endowments=False,
            fix_taxes=True,
            fix_technology=True,
            if_sub=bool(ifsub),
            numeraire="pnum",
        )
    else:
        pa = p
        ac = GTAPClosureConfig(
            name="base",
            closure_type="MCP",
            capital_mobility="sluggish",
            fix_endowments=False,
            fix_taxes=False,
            fix_technology=False,
            if_sub=bool(ifsub),
            numeraire="pnum",
        )
    rr = list(p.sets.r)[-1]
    mp = GTAPBlockMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)
    res = solve_multiperiod(
        m,
        p,
        ac,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        holdfix_cd=True,
        mode=mode,
    )
    return m, res


@pytest.fixture(scope="module", params=CASES, ids=[f"{m}-ifsub{s}" for m, s in CASES])
def solved(request):
    mode, ifsub = request.param
    gdx = _fixture_gdx(mode, ifsub)
    if not gdx.exists():
        pytest.skip(f"fixture ausente: {gdx}")
    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("PATH (path_capi_python) ausente")
    m, res = _solve(mode, ifsub, gdx)
    assert all(r["code"] == 1 for r in res.values()), res
    return mode, gdx, m


def test_the_tariff_is_the_shock_cell_of_imptx(solved):
    from pyomo.environ import value

    mode, gdx, m = solved
    gams = _gams(gdx, "imptx")
    checked = 0
    for k, want in gams.items():
        if k[-1] != "shock" or k not in m.imptx:
            continue
        assert value(m.imptx[k]) == pytest.approx(want, rel=1e-9, abs=1e-12), (
            f"{mode}: imptx{k}"
        )
        checked += 1
    assert checked, "ninguna celda de imptx comparada"


def test_altertax_moves_xw_as_gams_does(solved):
    from pyomo.environ import value

    mode, gdx, m = solved
    if mode != "altertax":
        pytest.skip("solo altertax: en gtap el arancel ya entraba al solve")
    gams = _gams(gdx, "xw")
    moved = wrong = 0
    for k, shk in gams.items():
        ck = (*k[:-1], "check")
        if k[-1] != "shock" or ck not in gams or k not in m.xw:
            continue
        g_move = (shk - gams[ck]) / abs(gams[ck]) if gams[ck] else 0.0
        if abs(g_move) <= 1e-3:
            continue
        moved += 1
        e_move = value(m.xw[k]) - value(m.xw[ck])
        if e_move * g_move <= 0.0:
            wrong += 1
    assert moved, "GAMS no mueve ninguna celda de xw: el oraculo no prueba nada"
    assert wrong == 0, f"{wrong}/{moved} celdas de xw no se mueven como en GAMS"


# --- Tabla 1 del paper: el refactor no mueve la solucion ---------------------
# Oraculo: los niveles de equilibria v0.6.1 (c096894) del 3x3 de la Tabla 1,
# medidos desde el zip de replicacion (fuente en el propio fixture). El camino
# viejo los reproduce bit a bit; el nuevo tiene que dar la misma solucion.
TABLE1_ORACLE = (
    ROOT / "tests/fixtures/gtap7_gempack/table1_levels_v061_gtap7_3x3.json.gz"
)
GEMPACK = ROOT / "tests/fixtures/gtap7_gempack"


def test_table1_3x3_keeps_its_solution():
    import gzip
    import json

    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("PATH (path_capi_python) ausente")
    sys.path.insert(0, str(ROOT / "scripts/gtap"))
    from compare_gempack_levels import run, solve  # ty: ignore[unresolved-import]

    with gzip.open(TABLE1_ORACLE, "rt") as fh:
        oracle = json.load(fh)
    want = {(v, t, tuple(i)): a for v, t, i, a in oracle["cells"]}
    levels, xscale = solve(DATASET, "capFlex", "path")
    tag = f"{DATASET}_tm10_s8-16-32"
    r = run(
        levels,
        xscale,
        GEMPACK / f"sl4levels6_{tag}.har",
        ROOT / f"datasets/{DATASET}/basedata.har",
        GEMPACK / f"updated_{tag}.har",
    )
    got = {
        (c["variable"], c["period"], tuple(c["index"])): c["a"]
        for c in json.loads(r.to_json())["cells"]
        if c["a"] is not None
    }
    assert got.keys() == want.keys()
    bad = [k for k in want if got[k] != pytest.approx(want[k], rel=1e-8, abs=1e-12)]
    assert not bad, f"{len(bad)}/{len(want)} celdas cambian, p.ej. {bad[:3]}"
    assert r.overall["pct_within"][1e-3] == oracle["overall"]["pct_within"]["0.001"]
