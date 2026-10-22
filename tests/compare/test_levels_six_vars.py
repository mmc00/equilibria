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

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "gtap7_gempack"
# caso -> (dataset, .har de niveles parchado, sl4dump (% de cambio) de la corrida SIN
# parche, gtapv7.exe). El sl4dump del 3x3 capFlex es identico (252/252 cabeceras) al
# de la corrida tm10_s8-16-32 que se parcho. Los capFix son las filas de la Tabla 2
# (corrida de Windows del 2026-10-21, paso 4).
CASES = {
    "gtap7_3x3": (
        "gtap7_3x3",
        "sl4levels6_gtap7_3x3_tm10.har",
        "sl4dump_gtap7_3x3_tm10.har",
    ),
    **{
        f"{ds}_capfix": (
            ds,
            f"sl4levels6_{ds}_tm10_s8-16-32_capfix.har",
            f"sl4dump_{ds}_tm10_s8-16-32_capfix.har",
        )
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
    dataset, name, _ = CASES[case]
    har = FIX / name
    if not har.exists():
        pytest.skip(f"falta {har.name} (paso 4 de corridas_paper.ps1, Windows)")
    sol = from_har(har)
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
    _, name, dump_name = CASES[case]
    har = FIX / name
    if not har.exists():
        pytest.skip(f"falta {har.name} (paso 4 de corridas_paper.ps1, Windows)")
    sol = from_har(har)
    dump = read_har(str(FIX / dump_name))
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
