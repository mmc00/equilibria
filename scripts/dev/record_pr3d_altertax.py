"""Record what ``solve_altertax`` gives on nus333 (PR 3d oracle).

PR 3d of gtap-on-core moves the family ALTERTAX runner and its path decomposition
from the old family driver to ``Exercise``. Run on ``main`` before the change, this
writes, per case, the period codes, every level the GEMPACK comparison reads
(``family_levels(run.result.mp, EQ_VARS)``) and the decomposition headers (the
path's when the case has ``decomp_steps``); ``tests/family/test_altertax_on_core.py``
compares the new route against it. Each case runs in its own child process, with
the environment of the ALTERTAX tests (Newton-TR + MUMPS, direct linear solve).

usage: python scripts/dev/record_pr3d_altertax.py --out FILE [--only CASE ...]
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

# the zero shock of tests/family/test_altertax_zero_shock.py (an exact 0 would get
# the default +10 % tariff)
ZERO = {"prdtx_rai": {("USA", "AGR", "AGR"): 1e-12}}

CASES: dict[str, dict] = {
    "nominal": {},
    "ratio": {"trade_balance": "ratio"},
    "nominal_k4": {"decomp_steps": 4},
    "ratio_k2": {"trade_balance": "ratio", "decomp_steps": 2},
    "zero": {"shock": ZERO},
    # an empty shock: the run writes the default +10 % import tariff
    "empty": {"shock": {}},
}
# the zero-shock test's solver: an absolute stop of 1e-12, no relative test
ZERO_ENV = {"EQUILIBRIA_GTAP_TR_FTOL": "1e-12", "EQUILIBRIA_GTAP_TR_RELTOL": "0"}

_CHILD = r"""
import json, sys
sys.path[:0] = [{src!r}, {root!r}]
import importlib.util
from equilibria._local_refs import nus333_dir
from equilibria.family.altertax.decomp import decomp_headers
from equilibria.family.altertax.levels import family_levels
from equilibria.family.altertax.run import solve_altertax

spec = importlib.util.spec_from_file_location(
    "cg", {root!r} + "/scripts/gtap/compare_gempack_levels.py")
cg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cg)
kw = {kw!r}
if "shock" in kw:
    kw["shock"] = {{n: {{tuple(k): v for k, v in c}} for n, c in kw["shock"].items()}}
run = solve_altertax(nus333_dir(), cache_root={cache!r}, **kw)
levels = family_levels(run.result.mp, cg.EQ_VARS)
heads = run.decomp_path.headers if run.decomp_path is not None else decomp_headers(run)
out = {{
    "codes": {{k: int(v) for k, v in run.codes.items()}},
    "levels": {{v: {{repr(k): float(x) for k, x in cells.items()}}
               for v, cells in levels.items()}},
    "decomp": {{n: h.array.tolist() for n, h in heads.items()}},
}}
print("RESULT " + json.dumps(out))
"""


def _newton_env() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from tests.family._prep import _newton_mumps

    return {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}


def run_case(name: str, root: Path) -> dict:
    kw = dict(CASES[name])
    if "shock" in kw:  # tuple keys do not survive repr -> JSON: pass as pairs
        kw["shock"] = {n: list(c.items()) for n, c in kw["shock"].items()}
    with tempfile.TemporaryDirectory() as cache:
        code = _CHILD.format(src=str(root / "src"), root=str(root), kw=kw, cache=cache)
        env = {**os.environ, **_newton_env(), "EQUILIBRIA_FAMILY_CACHE": cache}
        if name == "zero":
            env.update(ZERO_ENV)
        env["PYTHONPATH"] = f"{root / 'src'}:{root}"
        p = subprocess.run(
            [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True
        )
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"{name}: exit {p.returncode}\n{p.stderr[-3000:]}")
    return json.loads(line[len("RESULT ") :])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    names = a.only or list(CASES)
    out = {}
    for name in names:
        print(f"case {name}", flush=True)
        out[name] = run_case(name, a.root.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
