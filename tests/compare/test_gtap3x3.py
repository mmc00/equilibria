"""GTAP 3x3 de punta a punta: el GDX de GAMS contra equilibria (dict o modelo resuelto)."""

from __future__ import annotations

import pytest

from equilibria.compare import Mapping, compare, from_dict, from_gdx
from equilibria.sources import dataset_dir
from tests.compare import _oracle as O

pytestmark = pytest.mark.skipif(
    not O.GTAP3X3_GDX.exists(), reason="falta el GDX de gtap7_3x3"
)


def test_gams_contra_dict_de_equilibria():
    r = compare(
        from_gdx(O.GTAP3X3_GDX, period_dim=-1),
        from_dict(O.EX5_B, period_dim=-1),
        Mapping.from_dict(O.EX5_MAP),
    )
    s = r.summary["output"]
    assert s["cells"] == O.EX5_MATCHED
    for thr, pct in O.EX5_PCT_WITHIN.items():
        assert s["pct_within"][thr] == pytest.approx(pct)
    assert s["max"] == pytest.approx(O.EX5_MAX, rel=1e-9)
    assert s["max_cell"] == O.EX5_MAX_CELL
    assert r.periods_only_in_a == O.EX5_PERIODS_ONLY_IN_A
    assert r.unmatched == []
    assert r.excluded == {"pft": "ejemplo: excluida a proposito"}


MAPA = O.ROOT / "tests" / "compare" / "fixtures" / "gtap3x3_minimo.yaml"


@pytest.mark.integration
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump  # seed_all_periods lee el GDX con gdxdump
def test_gams_contra_el_modelo_de_equilibria_resuelto():
    """El oraculo es el GDX de GAMS: xp y pft de equilibria (bloques, PATH) caen
    dentro de 0,1% en todas las celdas de base, check y shock que GAMS define.

    GAMS define pft solo para los factores moviles: pft(r, NatRes) no existe en su
    modelo ni en el GDX. El mapeo lo declara como exclusion por celda (con su why);
    esas celdas (ejemplo 10: 3 reg x NatRes x 3 periodos = 9) se reportan como
    excluidas, no como fallas ni como celdas que pasan."""
    import sys

    from equilibria._local_refs import path_capi_src
    from equilibria.compare import from_pyomo
    from equilibria.templates.gtap.gtap_block_model import (
        build_block_model,
        solve_block_model,
    )
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    src = str(path_capi_src())
    if src not in sys.path:
        sys.path.insert(0, src)
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    m, mp = build_block_model(p, p.sets, gc, "ROW")
    mp.seed_all_periods(m, O.GTAP3X3_GDX)
    res = solve_block_model(m, p, gc, ref_gdx=O.GTAP3X3_GDX, mode="gtap")
    for t in ("base", "check", "shock"):
        assert res[t]["code"] == 1, (t, res[t])

    r = compare(
        from_gdx(O.GTAP3X3_GDX, period_dim=-1, variables=["xp", "pft"]),
        from_pyomo(m, period_dim=-1, variables=["xp", "pft"]),
        Mapping.from_yaml(MAPA),
        reference="a",
    )
    assert r.periods_compared == ["base", "check", "shock"]
    assert r.unmatched == []
    assert r.summary["output"]["cells"] == O.EX10_OUTPUT_CELLS
    fp = r.summary["factor_price"]
    assert fp["cells"] == O.EX10_FACTOR_PRICE_CELLS
    assert fp["excluded"] == O.EX10_NATRES_EXCLUDED
    assert [(e.variable, e.where, e.cells) for e in r.exclusions] == [
        ("factor_price", {"FP": ["NatRes"]}, O.EX10_NATRES_EXCLUDED)
    ]
    assert r.exclusions[0].why == "GAMS define pft solo para factores moviles"
    assert {x.index[-1] for x in r.excluded_cells} == {"NatRes"}
    assert all(c.index[-1] != "NatRes" for c in r.cells)
    assert all(c.passed for c in r.cells)
