"""Capture the saved Pyomo reference of one dataset (photos, solve stats, final state).

    <python with pymumps> scripts/family/capture_reference.py gtap7_10x7
        [--out DIR]
        [--ifsub {0,1}] [--savf {capFix,capFlex}] [--mode gtap] [--keep {all,gate}]

Run it with ``PYTHONPATH=<worktree>/src:<worktree>`` set (without it the MUMPS Python
imports the main checkout instead of this worktree).

Same child run as ``capture_prep.py`` (``_solve_shock`` of the requested closure,
(1, capFix) by default, ``solve_nlp=True``, under ``NEWTON_MUMPS``) into
``~/.cache/equilibria/family/reference/<dataset>/<closure_tag>/`` by default
(``tol_today`` for ifsub=1 capFix gtap; ``--mode altertax`` raises until Phase 3;
``--keep gate`` drops all but ``final.pkl``, ``settled.pkl`` and the S files), which now
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
from capture_prep import capture, closure_tag  # noqa: E402


def summary(out: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(out.glob("*_S_*.json"))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ifsub", type=int, choices=(0, 1), default=1)
    ap.add_argument("--savf", choices=("capFix", "capFlex"), default="capFix")
    ap.add_argument("--mode", choices=("gtap", "altertax"), default="gtap")
    ap.add_argument("--keep", choices=("all", "gate"), default="all")
    a = ap.parse_args(argv)
    if a.mode != "gtap":
        raise NotImplementedError(f"mode={a.mode!r}: only 'gtap' until Phase 3")
    out = a.out or (
        Path.home()
        / ".cache/equilibria/family/reference"
        / a.dataset
        / closure_tag(a.ifsub, a.savf, a.mode)
    )
    out.mkdir(parents=True, exist_ok=True)
    for f in [*out.glob("*.pkl"), *out.glob("*.json")]:
        f.unlink()
    t = time.perf_counter()
    code = capture(a.dataset, out, ifsub=a.ifsub, savf=a.savf, mode=a.mode, keep=a.keep)
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
