"""Reproduce el oraculo del ejemplo 14 (tests/compare/_oracle.py) SIN equilibria.compare.

GEMPACK (``sl4levels_gtap7_3x3_tm10.har``, solo LEVA = post) contra equilibria resuelto
(``_solve_shock("gtap7_3x3", 0)``, periodo shock), deshaciendo a mano las escalas de las
cantidades de equilibria: ``xp / (xscale[r, a] * 1e-6)`` y ``xd / 1e-6``. Lee el HAR con
``babel.har.reader`` y el modelo con ``pyomo.value``; ``d = |A - B| / |B|``.

Uso (desde la raiz, con PATH): ``uv run python tests/compare/oracles/gempack3x3_levels.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from equilibria.sources import dataset_dir

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pyomo.environ import value  # noqa: E402
from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock  # noqa: E402

from equilibria.babel.har.reader import read_har  # noqa: E402
from equilibria.blocks.gtap._derived_params import xscale_data  # noqa: E402
from equilibria.templates.gtap import GTAPParameters  # noqa: E402

THRESHOLDS = (1e-3, 5e-3, 1e-2)


def main() -> None:
    m, code = _solve_shock("gtap7_3x3", 0)
    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    xscale = xscale_data(p, p.sets)
    har = read_har(
        str(ROOT / "tests/fixtures/gtap7_gempack/sl4levels_gtap7_3x3_tm10.har")
    )
    by_name = {ha.long_name.split("#")[0].strip(): ha for ha in har.values()}
    rows = []
    for gempack, var, uses_xscale in (("qo", "xp", True), ("qds", "xd", False)):
        ha = by_name[gempack]
        arr = np.asarray(ha.array, dtype=float)
        comp = getattr(m, var)
        for i, e in enumerate(ha.set_elements[0]):
            for j, r in enumerate(ha.set_elements[1]):
                scale = 1e-6 * (xscale[(r, e)] if uses_xscale else 1.0)
                a = float(value(comp[r, e, "shock"])) / scale
                b = float(arr[i, j])
                rows.append((gempack, r, e, abs(a - b) / abs(b)))
    print(f"code={code}  celdas={len(rows)}")
    for t in THRESHOLDS:
        print(f"  <= {t:g}: {sum(row[3] <= t for row in rows)}")
    for gempack in ("qo", "qds"):
        ds = [row[3] for row in rows if row[0] == gempack]
        print(f"  {gempack}: {len(ds)} celdas, max d = {max(ds):.6g}")
    worst = max(rows, key=lambda row: row[3])
    print(f"  maximo: {worst}")


if __name__ == "__main__":
    main()
