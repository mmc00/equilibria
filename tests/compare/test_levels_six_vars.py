"""GEMPACK con gtapv7_levels.tab: y, qinv, rore, pinv, pva y pe ya tienen nivel.

El .har sale del paso 4 de ``scripts/windows/corridas_paper.ps1`` (gemsim sobre el TAB
parchado por ``make_levels_tab.py`` + ``export_sl4_levels.py``, que ya comprobo que los
% de cambio son los de la corrida sin parche). Aqui se comprueba el nivel PRE contra el
oraculo hecho a mano desde basedata.har, ``tests/compare/oracles/levb_six_vars.py``,
commiteado antes que el parche.

Tolerancia: el .sl4 guarda los niveles en REAL*4 y sltoht los imprime con ~8 cifras
(ver LEVEL_RTOL en export_sl4_levels.py), asi que 1e-6 relativo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from equilibria.compare import from_har

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "gtap7_gempack"
CASES = {
    "gtap7_3x3": FIX / "sl4levels6_gtap7_3x3_tm10.har",
    "gtap7_20x41": FIX / "sl4levels6_gtap7_20x41_tm10_s8-16-32_capfix.har",
}
SIX = ("y", "qinv", "rore", "pinv", "pva", "pe")
RTOL = 1e-6


def _oracle() -> Any:
    path = ROOT / "tests" / "compare" / "oracles" / "levb_six_vars.py"
    spec = importlib.util.spec_from_file_location("levb_six_vars", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("dataset", sorted(CASES))
def test_levb_de_las_seis_variables(dataset: str) -> None:
    har = CASES[dataset]
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
