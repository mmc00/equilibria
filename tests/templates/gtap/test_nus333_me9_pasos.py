"""ME9B/C/D: el shock (capital ROW +213 %, poblacion +37 %, ...) se resuelve en pasos.

Arrancando del check, PATH y Newton-TR a veces llegan y a veces se pierden, con o sin
la calibracion de betaP de GTAP (medido 2026-10-22: ME9B y ME9C fallan con uno u otro
solver segun el caso). GEMPACK nunca aplica el shock de una vez: lo parte en pasos
(Gragg). El driver hace lo mismo con los shocks del ShockBlock: sube el shock en
fracciones y cada solve arranca del anterior.

Oraculo: las soluciones GEMPACK de ``tests/fixtures/gtap7_gempack/burfisher`` (post).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from tests.templates.gtap._nus333 import ROOT, nus333_params


@pytest.mark.integration
@pytest.mark.parametrize("exp", ["ME9B", "ME9C", "ME9D"])
def test_me9_resuelve_y_coincide_con_gempack(exp: str) -> None:
    nus333_params("climatechange.prm")  # SKIP sin el dataset
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    cb: Any = importlib.import_module("compare_burfisher_gempack")
    from equilibria._local_refs import nus333_dir, path_capi_src
    from equilibria.blocks.gtap._derived_params import xscale_data

    sys.path.insert(0, str(path_capi_src()))
    m, p, codes = cb.rb.solve_exercise(exp, nus333_dir())
    assert codes == {"base": 1, "check": 1, "shock": 1}
    r = cb.run(exp, cb.cg.model_levels(m), dict(xscale_data(p, p.sets)))
    post = [c for c in r.cells if c.period == "post"]
    assert len(post) == 207
    worst = max(post, key=lambda c: c.d)
    assert worst.d < 5e-3, worst
