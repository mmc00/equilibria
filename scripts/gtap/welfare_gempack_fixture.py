"""Bienestar de GEMPACK en los ejercicios de Burfisher: el oraculo del EV regional.

Lee los ``sl4dump_<exp>.har`` que escribe ``sltoht`` (paso 6 de
``scripts/windows/corridas_paper.ps1``) y guarda, por ejercicio, ``EV`` y ``WEV``
(cambio ordinario, $ US millones), ``u`` e ``y`` (cambio %) en
``tests/fixtures/nus333_bienestar_gempack.json.gz``.

Solo entran las corridas que terminaron bien: su log (paso 5) tiene que cerrar con
"completed without error". ME4A/B con Gragg 2-4-6 terminan en "Accuracy too low" y
dejan el ``.sl4`` igual: no son oraculo.

Uso::

    PYTHONPATH=src:. python scripts/gtap/welfare_gempack_fixture.py \
        <6_sl4dump\\gragg> <5_burfisher\\out>
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "nus333_bienestar_gempack.json.gz"
VARS = ("EV", "WEV", "u", "y")


def _by_name(har: dict) -> dict[str, tuple[list[list[str]], np.ndarray]]:
    """Cabeceras del sl4dump por nombre de variable (la primera palabra del long_name)."""
    out = {}
    for hdr in har.values():
        name = (getattr(hdr, "long_name", "") or "").split(" ")[0]
        if name in VARS and name not in out:
            out[name] = (list(getattr(hdr, "set_elements", []) or []), np.asarray(hdr.array))
    return out


def _text(path: Path) -> str:
    """El log de gemsim redirigido por PowerShell (``*>``) es UTF-16 con BOM."""
    raw = path.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return raw.decode(enc, errors="replace")


def main() -> int:
    from equilibria.babel.har.reader import read_har

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    data: dict[str, dict] = {}
    for d in sorted(p for p in a.dump_dir.iterdir() if p.is_dir()):
        stem = d.name
        log = a.run_dir / stem / f"{stem}.log"
        if not log.exists() or "completed without error" not in _text(log):
            print(f"{stem}: el log no cierra bien, fuera", file=sys.stderr)
            continue
        dump = d / f"sl4dump_{stem}.har"
        found = _by_name(read_har(str(dump)))
        if set(found) != set(VARS):
            print(f"{stem}: faltan {set(VARS) - set(found)}", file=sys.stderr)
            return 1
        exp = stem.removesuffix("-dir").upper()
        cells = {}
        for v, (sets, arr) in found.items():
            if not sets:
                cells[v] = float(arr.reshape(-1)[0])
            else:
                cells[v] = {r: float(x) for r, x in zip(sets[0], arr.reshape(-1))}
        data[exp] = cells
    a.out.write_bytes(gzip.compress(json.dumps(data, sort_keys=True).encode(), mtime=0))
    print(f"{len(data)} ejercicios -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
