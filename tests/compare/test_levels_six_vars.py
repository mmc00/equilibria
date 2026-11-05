"""GEMPACK con gtapv7_levels.tab: y, qinv, rore, pinv, pva y pe ya tienen nivel.

El .har sale del paso 4 de ``scripts/windows/corridas_paper.ps1`` (gemsim sobre el TAB
parchado por ``make_levels_tab.py`` + ``export_sl4_levels.py``). Dos comprobaciones:

1. Nivel PRE contra el oraculo hecho a mano desde basedata.har,
   ``tests/compare/oracles/levb_six_vars.py``, commiteado antes que el parche.
2. El parche no cambia la solucion: el % de cambio que implican los niveles,
   ``100*(post/pre - 1)``, es el del sl4dump de la corrida SIN parche, celda por celda.
   Esto valida tambien el nivel POST, incluidos pinv/pva/pe (cuyo PRE = 1 lo fija el
   propio parche). export_sl4_levels.py hace un cruce parecido pero solo descarta e
   imprime; aqui es una puerta.

Tolerancias: el .sl4 guarda los niveles en REAL*4 (~6e-8 relativo, ver LEVEL_RTOL en
export_sl4_levels.py): 1e-6 relativo en el nivel; en el % implicito, dos niveles REAL*4
dan <= ~1,2e-7*(100+|%|) pp, ~3e-5 pp para |%| <= 22 (el mayor de los fixtures).
Medido el 2026-10-19: maximo 1,2e-5 pp (rore, 20x41). PCT_ATOL = 5e-5 pp.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.compare import from_har
from equilibria.sources import reference

ROOT = Path(__file__).resolve().parents[2]
# case -> (dataset, closure). The patched levels .har is the gemsim_levels6 record;
# the % changes come from the sl4dump of the UNPATCHED gtapv7.exe run, the gempack
# matrix record of the same closure. The capFix cases are the Table 2 rows.
CASES = {
    "gtap7_3x3": ("gtap7_3x3", "capFlex"),
    **{
        f"{ds}_capfix": (ds, "capFix")
        for ds in (
            "gtap7_3x3",
            "gtap7_3x4",
            "gtap7_5x5",
            "gtap7_10x7",
            "gtap7_15x10",
            "gtap7_20x41",
        )
    },
}


def _levels_har(case: str) -> Path:
    dataset, closure = CASES[case]
    rec = reference(
        dataset,
        "gemsim",
        tab="gtapv7_levels",
        shock=10,
        steps="8-16-32",
        closure=closure,
    )
    return rec / "sl4levels.har"


def _sl4dump(case: str) -> Path:
    dataset, closure = CASES[case]
    rec = reference(dataset, "gempack", shock=10, steps="8-16-32", closure=closure)
    return rec / "sl4dump.har"


SIX = ("y", "qinv", "rore", "pinv", "pva", "pe")
RTOL = 1e-6
PCT_ATOL = 5e-5


def _oracle() -> Any:
    path = ROOT / "tests" / "compare" / "oracles" / "levb_six_vars.py"
    spec = importlib.util.spec_from_file_location("levb_six_vars", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("case", sorted(CASES))
def test_levb_de_las_seis_variables(case: str) -> None:
    dataset, _ = CASES[case]
    sol = from_har(_levels_har(case))
    assert not set(SIX) & set(sol.no_level), sorted(set(SIX) & set(sol.no_level))
    assert sol.periods() == ["pre", "post"]

    expected = _oracle().levb(dataset)
    for var in ("y", "qinv", "rore"):
        got = {k[1][0]: x for k, x in sol.levels[var].items() if k[0] == "pre"}
        assert got.keys() == expected[var].keys(), var
        for r, x in expected[var].items():
            assert got[r] == pytest.approx(x, rel=RTOL), (var, r)
    for var in ("pinv", "pva", "pe"):
        pre = [x for k, x in sol.levels[var].items() if k[0] == "pre"]
        assert pre, var
        assert all(x == pytest.approx(1.0, rel=RTOL) for x in pre), var


@pytest.mark.parametrize("case", sorted(CASES))
def test_el_parche_no_cambia_la_solucion(case: str) -> None:
    sol = from_har(_levels_har(case))
    dump = read_har(str(_sl4dump(case)))
    by_name = {(h.long_name or "").split("#", 1)[0].strip(): h for h in dump.values()}
    for var in SIX:
        ref = by_name[var]
        pct = np.asarray(ref.array, dtype=np.float64)
        for idx in np.ndindex(pct.shape):
            key = tuple(ref.set_elements[d][i] for d, i in enumerate(idx))
            pre = sol.levels[var][("pre", key)]
            post = sol.levels[var][("post", key)]
            implied = 100.0 * (post / pre - 1.0)
            assert abs(implied - pct[idx]) <= PCT_ATOL, (var, key, implied, pct[idx])
