"""Genera ``eq_gtap7_<dataset>_levels.json.gz``: los niveles de equilibria resuelto
con el cierre del fixture de GEMPACK, para el mapeo GTAP (#132) sin PATH en CI.

Lee el modelo con ``pyomo.value``; no usa ``equilibria.compare``. Forma::

    {"vars": {nombre: [[[indices..., periodo], nivel], ...]},
     "params": {"xscale": [[[r, a], valor], ...]}, "code": 1}

Uso (desde la raiz, con PATH)::

    PYTHONPATH=. uv run python tests/compare/fixtures/make_eq_gtap7_levels.py gtap7_3x3
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pyomo.environ import value  # noqa: E402

from equilibria._local_refs import path_capi_src  # noqa: E402
from equilibria.blocks.gtap._derived_params import xscale_data  # noqa: E402
from equilibria.templates.gtap import GTAPParameters  # noqa: E402

# Lo que pide maps/gtap7.yaml, mas lo que usa compare_gempack_levels.py para armar
# qds, qfe, qxw, tpd, pr y VDFB.
VARS = (
    "xp",
    "rore",
    "regy",
    "pi",
    "xiagg",
    "pva",
    "xds",
    "xda",
    "pd",
    "pft",
    "pfact",
    "pwfact",
    "pgdpmp",
    "pmt",
    "pet",
    "xs",
    "xf",
    "pf",
    "xma",
    "rgdpmp",
    "xmt",
    "xw",
    "pefob",
    "dintx",
)


def main(dataset: str) -> None:
    sys.path.insert(0, str(path_capi_src()))
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    m, code = _solve_shock(dataset, 0)
    if code != 1:
        raise SystemExit(f"{dataset}: el solve termino con code={code}")
    d = ROOT / "datasets" / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    out = {
        "vars": {
            v: [
                [list(i) if isinstance(i, tuple) else [i], float(value(c))]
                for i, c in getattr(m, v).items()
            ]
            for v in VARS
        },
        "params": {
            "xscale": [[list(k), float(x)] for k, x in xscale_data(p, p.sets).items()]
        },
        "code": code,
    }
    path = Path(__file__).with_name(f"eq_{dataset}_levels.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
