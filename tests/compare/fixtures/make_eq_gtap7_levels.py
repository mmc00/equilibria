"""Genera ``eq_<dataset>_levels.json.gz``: los niveles de equilibria resuelto con el
cierre del fixture de GEMPACK por defecto (capFlex, PATH), para el mapeo GTAP sin
PATH en CI. Es ``scripts/gtap/compare_gempack_levels.py`` (``solve`` +
``save_eq_json``), que no usa ``equilibria.compare`` para leer el modelo.

Uso (desde la raiz, con PATH)::

    PYTHONPATH=. uv run python tests/compare/fixtures/make_eq_gtap7_levels.py gtap7_3x3
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "gtap" / "compare_gempack_levels.py"


def main(dataset: str) -> None:
    spec = importlib.util.spec_from_file_location("compare_gempack_levels", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    levels, xscale = mod.solve(dataset, "capFlex", "path")
    path = Path(__file__).with_name(f"eq_{dataset}_levels.json.gz")
    mod.save_eq_json(path, levels, xscale)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
