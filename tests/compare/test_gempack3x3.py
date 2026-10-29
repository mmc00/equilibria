"""GTAP 3x3 de punta a punta en NIVELES: GEMPACK (.har de niveles) contra equilibria
resuelto, deshaciendo las escalas de las cantidades. Oraculo: ejemplo 14."""

from __future__ import annotations

import sys

import pytest

from equilibria.compare import Mapping, compare, from_har, from_pyomo
from equilibria.sources import dataset_dir
from tests.compare import _oracle as O

INSCALE_WHY = "inScale: el modelo de equilibria no lo guarda como parametro"

MAP = {
    "periods": {"a": {"shock": "post"}},
    "variables": [
        {
            "name": "output",
            "a": {
                "var": "xp",
                "dims": ["REG", "ACT"],
                "divide_by": [
                    {"param": "xscale", "dims": ["REG", "ACT"]},
                    {"value": 1e-6, "why": INSCALE_WHY},
                ],
            },
            "b": {"var": "qo", "dims": ["ACT", "REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        },
        {
            "name": "dom_sales",
            "a": {
                "var": "xd",
                "dims": ["REG", "COMM"],
                "divide_by": [{"value": 1e-6, "why": INSCALE_WHY}],
            },
            "b": {"var": "qds", "dims": ["COMM", "REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        },
    ],
    "thresholds": [1e-3, 5e-3, 1e-2],
}


@pytest.mark.integration
@pytest.mark.needs_path
@pytest.mark.skipif(not O.EX14_HAR.exists(), reason="falta el .har de niveles del 3x3")
def test_gempack_contra_el_modelo_de_equilibria_resuelto_en_niveles():
    from equilibria._local_refs import path_capi_src
    from equilibria.blocks.gtap._derived_params import xscale_data
    from equilibria.templates.gtap import GTAPParameters

    src = str(path_capi_src())
    if src not in sys.path:
        sys.path.insert(0, src)
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    m, code = _solve_shock("gtap7_3x3", 0)
    assert code == 1
    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    a = from_pyomo(
        m,
        period_dim=-1,
        variables=["xp", "xd"],
        params={"xscale": xscale_data(p, p.sets)},
    )
    r = compare(a, from_har(O.EX14_HAR), Mapping.from_dict(MAP))

    assert {k: s["cells"] for k, s in r.summary.items()} == O.EX14_CELLS
    o = r.overall
    assert o["cells"] == O.EX14_TOTAL
    for thr, n in O.EX14_WITHIN.items():
        assert o["pct_within"][thr] == pytest.approx(100.0 * n / O.EX14_TOTAL)
    assert r.summary["output"]["max"] == pytest.approx(O.EX14_OUTPUT_MAX, rel=1e-2)
    assert o["max"] == pytest.approx(O.EX14_MAX, rel=1e-3)
    assert (o["max_variable"], o["max_cell"]) == O.EX14_MAX_CELL
    assert r.periods_compared == ["post"]
    assert r.unmatched == []
