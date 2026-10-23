"""Bienestar de GAMS en los ejercicios de Burfisher: el fixture del test de bienestar.

Lee, con ``gdxdump`` (``_diff_core.gams_levels``), los niveles de ``ev``, ``cv``, ``u``,
``ug``, ``us`` y ``uh`` de cada ``<EXP>_capFlex.gdx`` que escribe
``gen_burfisher_gams.py`` y los guarda en
``tests/fixtures/nus333_bienestar_gams_levels.json.gz``, con el formato de los otros
fixtures de niveles GAMS (``{exp: {var: [[clave, nivel], ...]}}``).

En GAMS estas variables son parte del MCP (``model.gms``: ``eveq``, ``cveq``, ``ueq``,
``ugeq``, ``useq``, ``uheq`` en ``model gtap``): el oraculo es lo que resolvio GAMS.

Uso (con GAMS en el PATH o ``GDXDUMP``)::

    PYTHONPATH=src:. python scripts/gtap/gams_welfare_fixture.py <dir de GDX>
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "gtap")]

from _diff_core import gams_levels  # noqa: E402

VARS = ("ev", "cv", "u", "ug", "us", "uh")
OUT = ROOT / "tests" / "fixtures" / "nus333_bienestar_gams_levels.json.gz"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("gams_dir", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    data: dict[str, dict[str, list]] = {}
    for gdx in sorted(a.gams_dir.glob("*_capFlex.gdx")):
        exp = gdx.name.removesuffix("_capFlex.gdx")
        data[exp] = {
            v: [[list(k), x] for k, x in sorted(gams_levels(gdx, v).items())]
            for v in VARS
        }
        if not all(data[exp].values()):
            print(f"{exp}: falta alguna variable en el GDX", file=sys.stderr)
            return 1
    a.out.write_bytes(gzip.compress(json.dumps(data, sort_keys=True).encode(), mtime=0))
    print(f"{len(data)} ejercicios -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
