"""Genera ``eq_burfisher_<exp>_levels.json.gz``: los niveles de equilibria en un
ejercicio de Burfisher (nus333), para comparar contra GEMPACK sin PATH en CI.

Resuelve con ``scripts/gtap/run_burfisher.py`` (``solve_exercise``: el ``.prm``, los
shocks y el cierre de la Tabla 3) y guarda con ``compare_gempack_levels.py``
(``model_levels`` + ``save_eq_json``); ``xscale`` sale de los parametros del ejercicio.
No usa ``equilibria.compare``.

Uso (desde la raiz, con PATH; el paquete del libro es ``data/gtap7/nus333``)::

    PYTHONPATH=src:. python tests/compare/fixtures/make_eq_burfisher_levels.py TBL46A
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "gtap")]


def main(exp: str) -> None:
    import compare_gempack_levels as cg
    import run_burfisher as rb

    from equilibria._local_refs import nus333_dir, path_capi_src
    from equilibria.blocks.gtap._derived_params import xscale_data

    sys.path.insert(0, str(path_capi_src()))
    m, p, codes = rb.solve_exercise(exp, nus333_dir())
    if set(codes.values()) != {1}:
        raise SystemExit(f"{exp}: {codes}")
    path = Path(__file__).with_name(f"eq_burfisher_{exp.lower()}_levels.json.gz")
    cg.save_eq_json(path, cg.model_levels(m), dict(xscale_data(p, p.sets)))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
