"""GTAP 3x3 de punta a punta: el GDX de GAMS contra equilibria (dict o modelo resuelto)."""

from __future__ import annotations

import pytest

from equilibria.compare import Mapping, compare, from_dict, from_gdx
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
def test_gams_contra_el_modelo_de_equilibria_resuelto():
    """El oraculo es el GDX de GAMS: xp y pft de equilibria (bloques, PATH) caen
    dentro de 0,1% en todas las celdas de base, check y shock.

    GAMS define pft solo para los factores moviles (no hay NatRes en el GDX);
    equilibria tambien tiene pft[r, NatRes]. Esas 9 celdas (3 reg x 3 periodos) se
    declaran sin pareja (faltan en A), nunca como celdas que pasan."""
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

    d = O.ROOT / "datasets" / "gtap7_3x3"
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
    assert sorted({u.index[1] for u in r.unmatched}) == ["NatRes"]
    assert {(u.variable, u.missing_in) for u in r.unmatched} == {("factor_price", "a")}
    assert len(r.unmatched) == 9
    assert r.summary["output"]["cells"] == 27  # 3 reg x 3 act x 3 periodos
    assert r.summary["factor_price"]["cells"] == 36  # 3 reg x 4 factores x 3
    assert r.overall["pct_within"][1e-3] == 100.0, r.overall
