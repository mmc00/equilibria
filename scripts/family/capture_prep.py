"""Capture the two-photo oracle of the Pyomo period prep for one dataset.

    <python> scripts/family/capture_prep.py gtap7_10x7 [--out DIR]

Runs ``_solve_shock(ds, 1, "capFix", solve_nlp=True)`` under ``NEWTON_MUMPS`` in a
child process with ``EQUILIBRIA_FAMILY_PREP_CAPTURE`` set. Uses MUMPS when pymumps
is importable, the exact scipy LU otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
from perf_budget_20x41 import NEWTON_MUMPS  # noqa: E402

_CHILD = """
import sys
sys.path[:0] = [{src!r}, {tests!r}, {root!r}]
from templates.gtap.test_gtap7_gempack_parity import _solve_shock
_, code = _solve_shock({ds!r}, 1, "capFix", solve_nlp=True)
print("CODE", code)
"""


def capture(dataset: str, out: Path) -> int:
    try:
        import mumps  # noqa: F401

        linsolve = "mumps"
    except ImportError:
        linsolve = "direct"
    env = {
        **os.environ,
        **NEWTON_MUMPS,
        "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve,
        "EQUILIBRIA_FAMILY_PREP_CAPTURE": str(out),
    }
    env.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)
    code = _CHILD.format(
        src=str(ROOT / "src"), tests=str(ROOT / "tests"), root=str(ROOT), ds=dataset
    )
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    line = next((s for s in p.stdout.splitlines() if s.startswith("CODE ")), None)
    if line is None:
        raise RuntimeError(p.stderr[-4000:])
    return int(line.split()[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path.home() / ".cache/equilibria/family/prep_capture" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.pkl"):
        f.unlink()
    print("code", capture(a.dataset, out), "->", out)
