"""Foto de gdxdump de TODOS los Par/Var/Equ de un GDX, para el test del lector.

Oraculo independiente del lector de equilibria: lo escribe gdxdump (GAMS), no
babel.gdx. Valores tal como los imprime gdxdump ``format=csv`` (``CSVAllFields``
para Var/Equ: Level, Marginal, Lower, Upper, Scale), como texto: los especiales
(+Inf, -Inf, Eps, NA, UNDF) y los acronimos quedan con su nombre.

Uso (desde la raiz, con GAMS):
    uv run python tests/babel/gdx/oracles/gdxdump_snapshot.py \\
        tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub0.gdx
escribe ``<gdx>.gdxdump.json.gz`` al lado.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
from _diff_core import GDXDUMP  # type: ignore  # noqa: E402

TYPES = {"Par": "parameter", "Var": "variable", "Equ": "equation"}


def symbols(gdx: Path) -> list[tuple[str, str, int]]:
    out = subprocess.run(
        [GDXDUMP, str(gdx), "symbols"], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split()
        # "  84 axp   3  Var   27  Production frontier shifter"
        if len(parts) >= 5 and parts[0].isdigit() and parts[3] in TYPES:
            rows.append((parts[1], TYPES[parts[3]], int(parts[2])))
    return rows


def dump(gdx: Path, name: str, kind: str, dim: int) -> list[list[str]]:
    args = [GDXDUMP, str(gdx), f"symb={name}", "format=csv"]
    if kind != "parameter":
        args.append("CSVAllFields")
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    if not rows:
        return []
    head, body = rows[0], rows[1:]
    nval = 1 if kind == "parameter" else 5
    assert len(head) == dim + nval, (name, head)
    return body


def main() -> None:
    gdx = Path(sys.argv[1])
    snap: dict[str, dict[str, Any]] = {
        name: {"type": kind, "dim": dim, "records": dump(gdx, name, kind, dim)}
        for name, kind, dim in symbols(gdx)
    }
    out = gdx.with_name(gdx.name + ".gdxdump.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(snap, fh, sort_keys=True)
    n = {k: sum(1 for s in snap.values() if s["type"] == k) for k in TYPES.values()}
    print(f"{out}: {n}, {sum(len(s['records']) for s in snap.values())} registros")


if __name__ == "__main__":
    main()
