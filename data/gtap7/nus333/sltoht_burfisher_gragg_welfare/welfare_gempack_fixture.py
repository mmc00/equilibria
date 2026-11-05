"""GEMPACK welfare in the Burfisher exercises: the regional EV oracle.

Reads the ``sl4dump.har`` of each Gragg record
(``data/gtap7/nus333/sltoht_burfisher_gragg/<exp>/``) and writes, per exercise,
``EV`` and ``WEV`` (ordinary change, $ US million), ``u``, ``y``, ``psave`` and
``pinv`` (% change) to a gzipped JSON, by default ``welfare.json.gz`` in the
current folder. It lives in its own record, so the record hashes it.

Only runs that finished well enter: their console
(``gemsim_burfisher_gragg/<exp>/gemsim_console.txt``) has to say "completed
without error". ME4A/B with Gragg 2-4-6 ended in "Accuracy too low" and have no
Gragg record (D1, phase 3b).

Recorded with ``scripts/dev/write_source.py``, which runs it in this folder::

    uv run python scripts/dev/write_source.py \
        --folder data/gtap7/nus333/sltoht_burfisher_gragg_welfare --kind result \
        --model gtap7 --dataset nus333 --tool sltoht \
        --variant method=gragg --variant output=welfare \
        --input dump_<exp>=data/gtap7/nus333/sltoht_burfisher_gragg/<exp> ... \
        --input run_<exp>=data/gtap7/nus333/gemsim_burfisher_gragg/<exp> ... \
        -- uv run python welfare_gempack_fixture.py --out welfare.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

VARS = ("EV", "WEV", "u", "y", "psave", "pinv")


def _by_name(har: dict) -> dict[str, tuple[list[list[str]], np.ndarray]]:
    """The sl4dump headers by variable name (the first word of the long_name)."""
    out = {}
    for hdr in har.values():
        name = (getattr(hdr, "long_name", "") or "").split(" ")[0]
        if name in VARS and name not in out:
            out[name] = (
                list(getattr(hdr, "set_elements", []) or []),
                np.asarray(hdr.array),
            )
    return out


def _text(path: Path) -> str:
    """A gemsim console redirected by PowerShell (``*>``) is UTF-16 with a BOM."""
    raw = path.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return raw.decode(enc, errors="replace")


def gragg_records() -> dict[str, tuple[Path, Path]]:
    """``{experiment: (sl4dump folder, gemsim folder)}`` of the Gragg records.

    One scan of ``data/`` (a ``reference()`` per experiment rescans it each time).
    """
    from equilibria.sources import iter_sources

    found: dict[str, dict[str, Path]] = {}
    for r in iter_sources():
        if (
            r.kind == "result"
            and r.dataset == "nus333"
            and r.tool in ("sltoht", "gemsim")
            and r.variant.get("method") == "gragg"
            and "experiment" in r.variant  # not the welfare record itself
        ):
            found.setdefault(r.variant["experiment"], {})[r.tool] = r.path
    return {e: (t["sltoht"], t["gemsim"]) for e, t in sorted(found.items())}


def main() -> int:
    from equilibria.babel.har.reader import read_har

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("welfare.json.gz"))
    a = ap.parse_args()

    data: dict[str, dict] = {}
    for exp, (dump, run) in gragg_records().items():
        if "completed without error" not in _text(run / "gemsim_console.txt"):
            print(f"{exp}: the run did not finish well, left out", file=sys.stderr)
            continue
        found = _by_name(read_har(str(dump / "sl4dump.har")))
        if set(found) != set(VARS):
            print(f"{exp}: missing {set(VARS) - set(found)}", file=sys.stderr)
            return 1
        cells = {}
        for v, (sets, arr) in found.items():
            if not sets:
                cells[v] = float(arr.reshape(-1)[0])
            else:
                cells[v] = {
                    r: float(x) for r, x in zip(sets[0], arr.reshape(-1), strict=True)
                }
        data[exp.upper()] = cells
    a.out.write_bytes(gzip.compress(json.dumps(data, sort_keys=True).encode(), mtime=0))
    print(f"{len(data)} exercises -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
