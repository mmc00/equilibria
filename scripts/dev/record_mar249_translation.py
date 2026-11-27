"""Record the GEMPACK translation MAR-249 moves into the package, on nus333 (oracle written on main).

MAR-249 moves ``equilibria_side``, ``EQ_VARS`` and ``model_levels`` from
``scripts/gtap/compare_gempack_levels.py`` to ``equilibria.compare.gtap7_side`` and
the % change loop of ``family/altertax/pct.py`` next to them, without changing a
number. Run on ``main`` before the move, this writes, for the AlterTax nus333 run, the
``equilibria_side(...).levels`` of the family levels and the ``pct_changes(run)``.
``tests/compare/test_gtap7_side_golden.py`` compares the current checkout against it.
The case runs in a child process with the environment of the ALTERTAX tests
(Newton-TR + MUMPS, direct linear solve).

usage: python scripts/dev/record_mar249_translation.py --out FILE
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CHILD = r"""
import json, sys
sys.path[:0] = [{src!r}, {root!r}]
from equilibria._local_refs import nus333_dir
from equilibria.blocks.gtap._derived_params import xscale_data
from equilibria.family.altertax.compare import load_compare_script
from equilibria.family.altertax.levels import family_levels
from equilibria.family.altertax.pct import pct_changes
from equilibria.family.altertax.run import solve_altertax

run = solve_altertax(nus333_dir(), cache_root={cache!r})
try:
    from equilibria.compare.gtap7_side import EQ_VARS, equilibria_side
except ImportError:  # main, before MAR-249
    cg = load_compare_script()
    EQ_VARS, equilibria_side = cg.EQ_VARS, cg.equilibria_side
levels = family_levels(run.result.mp, EQ_VARS)
side = equilibria_side(levels, dict(xscale_data(run.params, run.params.sets)))
def join(k):
    flat = []
    for x in k if isinstance(k, tuple) else (k,):
        flat += list(x) if isinstance(x, tuple) else [x]
    return "|".join(map(str, flat))
out = {{
    "levels": {{v: {{join(k): float(x) for k, x in cells.items()}}
               for v, cells in side.levels.items()}},
    "pct": {{v: {{join(k): float(x) for k, x in cells.items()}}
            for v, cells in pct_changes(run).items()}},
}}
print("RESULT " + json.dumps(out))
"""


def _newton_env() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from tests.family._prep import _newton_mumps

    return {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}


def run_case(root: Path) -> dict:
    with tempfile.TemporaryDirectory() as cache:
        code = _CHILD.format(src=str(root / "src"), root=str(root), cache=cache)
        env = {**os.environ, **_newton_env(), "EQUILIBRIA_FAMILY_CACHE": cache}
        env["PYTHONPATH"] = f"{root / 'src'}:{root}"
        p = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"exit {p.returncode}\n{p.stderr[-3000:]}")
    return json.loads(line[len("RESULT ") :])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    out = run_case(a.root.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
