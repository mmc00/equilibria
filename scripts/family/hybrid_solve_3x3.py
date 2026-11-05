"""Solve gtap7_3x3 (ifSUB=0, capFix) base→check→shock with newton_tr, with or without
the family evaluator, in a clean child process; return the levels of every Var.

    python scripts/family/hybrid_solve_3x3.py [--family]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
from perf_budget_20x41 import NEWTON_MUMPS  # noqa: E402

# pymumps is conda-forge only; the exact scipy LU is the driver default and is
# enough at 3x3. The evaluators are compared under the same linear solver.
ENV = {**NEWTON_MUMPS, "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}

_CHILD = """
import gc, json, sys
gc.disable()
sys.path[:0] = [{root!r}, {tests!r}]
from templates.gtap.test_gtap7_gempack_parity import _solve_shock
from pyomo.environ import Var, value
m, code = _solve_shock("gtap7_3x3", 0, "capFix", solve_nlp=True)
levels = {{v.name: value(v, exception=False) for v in m.component_data_objects(Var)}}
print("RESULT " + json.dumps({{"code": code, "levels": levels}}))
"""


def run(family: bool) -> dict:
    env = {**os.environ, **ENV}
    env.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)
    if family:
        env["EQUILIBRIA_GTAP_FAMILY_EVAL"] = "1"
    else:
        env.pop("EQUILIBRIA_GTAP_FAMILY_EVAL", None)
    code = _CHILD.format(root=str(ROOT), tests=str(ROOT / "tests"))
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT
    )
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(p.stderr[-4000:])
    out = json.loads(line[len("RESULT ") :])
    out["family_eval"] = [s for s in p.stderr.splitlines() if "[family-eval]" in s]
    out["nit"] = [int(n) for n in re.findall(r"nit=(\d+)", p.stderr)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="store_true")
    r = run(ap.parse_args().family)
    print(json.dumps({k: v for k, v in r.items() if k != "levels"}, indent=1))
