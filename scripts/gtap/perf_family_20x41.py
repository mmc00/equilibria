"""One cold gtap7_20x41 run on the family (array/JAX) route: wall, peak, phases.

The family twin of ``perf_budget_20x41.py``: a child process runs the shock
(capFix, ifSUB=1, base calibrated, Newton-TR + MUMPS) and prints one ``PERF``
line; the parent reads it, adds the peak memory and refuses a run that swapped.

    <python with pymumps and JAX> scripts/gtap/perf_family_20x41.py --route old

``--route old`` times ``solve_gtap_family``; its ``phases`` (self seconds per fine
phase: ``recipe``, ``square``, ``system_build``, ``newton_*``, ``post_solve``) are
the budget the core route's steps must stay within. Exit 0 = measured, 2 =
invalid (swap, no MUMPS, or the shock did not converge).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from perf_budget_20x41 import NEWTON_MUMPS, SWAP_MAX_MB, _swap_in_mb  # noqa: E402

_CHILD = """
import json, sys, time
sys.path[:0] = [{src!r}, {scripts!r}, {tests!r}, {root!r}]
from templates.gtap.test_gtap7_gempack_parity import _solve_shock
t = time.perf_counter()
res, code = _solve_shock({dataset!r}, 1, "capFix", solve_nlp=True, backend="family")
wall = (time.perf_counter() - t) / 60
phases = {{k: round(v, 2) for k, v in sorted(res.phases.items(), key=lambda kv: -kv[1])}}
print("PERF " + json.dumps({{"code": code, "wall_min": wall, "phases": phases,
                            "timing": {{k: round(v, 2) for k, v in res.timing.items()}}}}))
"""


def measure(route: str, dataset: str = "gtap7_20x41") -> tuple[dict, float]:
    if route != "old":
        raise SystemExit(f"--route {route!r}: only 'old' for now")
    code = _CHILD.format(
        dataset=dataset,
        src=str(ROOT / "src"),
        scripts=str(ROOT / "scripts" / "gtap"),
        tests=str(ROOT / "tests"),
        root=str(ROOT),
    )
    env = {**os.environ, **NEWTON_MUMPS}
    cmd = [sys.executable, "-c", code]
    darwin = platform.system() == "Darwin"
    if darwin:
        cmd = ["/usr/bin/time", "-l", *cmd]
    swap0 = _swap_in_mb()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    swap = _swap_in_mb() - swap0
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PERF ")), None)
    if line is None:
        sys.stderr.write(p.stderr[-4000:])
        return {"code": None, "wall_min": 0.0, "peak_gb": 0.0}, swap
    out = json.loads(line[5:])
    if darwin:
        peak = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
        out["peak_gb"] = int(peak.group(1)) / 1e9 if peak else 0.0
    else:
        import resource

        out["peak_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return out, swap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", choices=("old", "exercise"), default="old")
    ap.add_argument("--dataset", default="gtap7_20x41", help="a smaller one to smoke-test")
    args = ap.parse_args()
    out, swap = measure(args.route, args.dataset)
    print(
        f"measured: {out['wall_min']:.2f} min, {out['peak_gb']:.2f} GB, "
        f"swap {swap:.0f} MB, code={out['code']}"
    )
    print("phases: " + json.dumps(out.get("phases", {})))
    print("timing: " + json.dumps(out.get("timing", {})))
    if swap > SWAP_MAX_MB or out["code"] != 1:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
