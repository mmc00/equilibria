"""Capture the saved Pyomo reference of one dataset (photos, solve stats, final state).

    <python with pymumps> scripts/family/capture_reference.py gtap7_10x7
        [--out DIR]

Same child run as ``capture_prep.py`` (``_solve_shock(ds, 1, "capFix",
solve_nlp=True)`` under ``NEWTON_MUMPS``) into
``~/.cache/equilibria/family/reference/<dataset>/tol_today/`` by default, which now
holds, per solve call: ``<n>_A/A2/B/P_<period>.pkl`` and ``<n>_S_<period>.json``
(code, nit, nfact, per-iteration residual trace, TR/MUMPS settings, GMIN, ICNTL, null
pivots), plus ``final.pkl`` and ``settled.pkl``. Prints one line per solve call with
its ``nit`` (recorded, never a gate), the disk size and the wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_prep import capture  # noqa: E402


def summary(out: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(out.glob("*_S_*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or (
        Path.home() / ".cache/equilibria/family/reference" / a.dataset / "tol_today"
    )
    out.mkdir(parents=True, exist_ok=True)
    for f in [*out.glob("*.pkl"), *out.glob("*.json")]:
        f.unlink()
    t = time.perf_counter()
    code = capture(a.dataset, out)
    wall = time.perf_counter() - t
    for s in summary(out):
        print(
            f"call {s['n']:03d} {s['period']:<6} code={s['code']} nit={s['nit']} "
            f"nfact={s['nfact']} residual={s['residual']:.2e} "
            f"null_pivots={sum(s['null_pivots'])}"
        )
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"code {code}  wall {wall / 60:.1f} min  {size / 1e9:.3f} GB  -> {out}")
    return 0 if code == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
