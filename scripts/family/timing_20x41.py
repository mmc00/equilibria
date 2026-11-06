"""Phase timing of the family route end to end (``_solve_shock(..., backend="family")``).

usage: <python with pymumps> scripts/family/timing_20x41.py [DATASET] [--runs 3]
           [--cache-root DIR] [--out FILE] [--max-load 4.0]

Protocol: one child process per run (GC as the route sets it unless ``--gc off``), one warm-up run (fills the record and
JAX caches, not reported), then ``--runs`` measured ones; the report gives the median
and the spread (min, max) of the process wall, of the wall from the start of
``_solve_shock`` to its return, and of every phase. The machine load average and free
memory are recorded around each run; a load above ``--max-load`` before a run waits (up to
``--max-wait`` s, the average still carries the previous run) and then aborts (exit 2)
unless ``--force`` is given. The phases come from ``FamilyResult.phases``
(self time, additive); the MUMPS analysis / factor / solve times and the first F and J
call of each squared system (JAX compile-cache load) are details inside the Newton
phases, measured by wrapping the calls in the child only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CHILD = r"""
import gc, json, os, sys, time
if {gc_off!r}:
    gc.disable()
t_proc = time.perf_counter()
sys.path[:0] = [{src!r}, {root!r}]
os.environ["EQUILIBRIA_FAMILY_CACHE"] = {cache!r}
detail = {{"mumps_analysis_s": 0.0, "mumps_factor_s": 0.0, "mumps_solve_s": 0.0,
          "mumps_factor_calls": 0, "first_F_s": [], "first_J_s": []}}
mark = {{}}
t = time.perf_counter()
from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock
import equilibria.family.driver as driver
from equilibria.templates.gtap import GTAPParameters
from equilibria.family.squared_system import SquaredSystem
mark["imports_s"] = time.perf_counter() - t
mark["pyomo_imported_before"] = "pyomo" in sys.modules

def timed(cls, name, key, count=None):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            detail[key] += time.perf_counter() - t
            if count:
                detail[count] += 1
    setattr(cls, name, w)

def first_call(cls, name, key):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        first = not getattr(self, "_seen_" + name, False)
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            if first:
                self.__dict__["_seen_" + name] = True
                detail[key].append(time.perf_counter() - t)
    setattr(cls, name, w)

first_call(SquaredSystem, "F", "first_F_s")
first_call(SquaredSystem, "J", "first_J_s")
from pyomo.contrib.pynumero.linalg.mumps_interface import (
    MumpsCentralizedAssembledLinearSolver as M)
timed(M, "do_symbolic_factorization", "mumps_analysis_s")
timed(M, "do_numeric_factorization", "mumps_factor_s", "mumps_factor_calls")
timed(M, "do_back_solve", "mumps_solve_s")

orig_load = GTAPParameters.load_from_har
def load(self, *a, **k):
    mark["before_har_s"] = time.perf_counter() - t_solve
    t = time.perf_counter()
    try:
        return orig_load(self, *a, **k)
    finally:
        mark["har_s"] = time.perf_counter() - t
GTAPParameters.load_from_har = load
orig_fam = driver.solve_gtap_family
def fam(*a, **k):
    t = time.perf_counter()
    try:
        return orig_fam(*a, **k)
    finally:
        mark["family_call_s"] = time.perf_counter() - t
        mark["family_end"] = time.perf_counter()
driver.solve_gtap_family = fam

t_solve = time.perf_counter()
res, code = _solve_shock({dataset!r}, 1, "capFix", solve_nlp=True, backend="family")
t_end = time.perf_counter()
mark["solve_shock_s"] = t_end - t_solve
mark["after_family_s"] = t_end - mark.pop("family_end")
mark["pyomo_imported_after"] = "pyomo" in sys.modules
mark["pyomo_modules"] = sum(1 for m in sys.modules if m == "pyomo" or m.startswith("pyomo."))
import resource
mark["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print("RESULT " + json.dumps({{"code": code, "mark": mark, "detail": detail,
    "timing": res.timing, "phases": res.phases, "counts": res.phase_counts,
    "nit": {{k: v.get("nit") for k, v in res.results.items()}}}}))
"""


def newton_mumps() -> dict:
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


def machine() -> dict:
    """Load average and free memory now."""
    load = os.getloadavg()
    free_gb = None
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))  # type: ignore[union-attr]
        got = {
            k: int(v)
            for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", vm)
        }
        free_gb = sum(got.values()) * page / 1e9
    except Exception:
        pass
    return {"load_1m": load[0], "load_5m": load[1], "reclaimable_gb": free_gb}


def run_child(dataset: str, cache_root: Path, gc_off: bool = False) -> dict:
    code = _CHILD.format(
        src=str(ROOT / "src"),
        root=str(ROOT),
        cache=str(cache_root),
        dataset=dataset,
        gc_off=gc_off,
    )
    env = {**os.environ, **newton_mumps()}
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    t = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    wall = time.perf_counter() - t
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(p.stderr[-4000:] + p.stdout[-2000:])
    out = json.loads(line[len("RESULT ") :])
    out["process_wall_s"] = wall
    return out


def import_time(dataset: str) -> dict:
    """``python -X importtime`` of the child's imports: cumulative seconds per package."""
    code = (
        f"import sys; sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT)!r}]\n"
        "from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock\n"
        "import equilibria.family.driver\n"
        "from pyomo.contrib.pynumero.linalg.mumps_interface import "
        "MumpsCentralizedAssembledLinearSolver\n"
    )
    env = {**os.environ, **newton_mumps(), "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    p = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    tops: dict[str, float] = {}
    total = 0.0
    for ln in p.stderr.splitlines():
        m = re.match(r"import time:\s+(\d+) \|\s+(\d+) \| (\s*)(\S+)", ln)
        if not m:
            continue
        self_us, cum_us, indent, name = int(m[1]), int(m[2]), m[3], m[4]
        total += self_us / 1e6
        if not indent:  # a top-level import statement of the script
            tops[name.split(".")[0]] = tops.get(name.split(".")[0], 0.0) + cum_us / 1e6
    ranked = dict(sorted(tops.items(), key=lambda kv: -kv[1])[:12])
    return {
        "total_s": total,
        "top_level_cumulative_s": ranked,
        "pyomo_cumulative_s": tops.get("pyomo"),
        "jax_cumulative_s": tops.get("jax"),
    }


def stats(xs: list[float]) -> dict:
    return {
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
    }


def aggregate(runs: list[dict]) -> dict:
    keys = sorted({k for r in runs for k in r["phases"]})
    phases = {k: stats([r["phases"].get(k, 0.0) for r in runs]) for k in keys}
    marks = [
        "imports_s",
        "before_har_s",
        "har_s",
        "family_call_s",
        "after_family_s",
        "solve_shock_s",
    ]
    return {
        "process_wall_s": stats([r["process_wall_s"] for r in runs]),
        "marks": {k: stats([r["mark"][k] for r in runs]) for k in marks},
        "family_timing": {
            k: stats([r["timing"][k] for r in runs]) for k in runs[0]["timing"]
        },
        "phases": phases,
        "detail": {
            "mumps_analysis_s": stats([r["detail"]["mumps_analysis_s"] for r in runs]),
            "mumps_factor_s": stats([r["detail"]["mumps_factor_s"] for r in runs]),
            "mumps_factor_calls": runs[0]["detail"]["mumps_factor_calls"],
            "mumps_solve_s": stats([r["detail"]["mumps_solve_s"] for r in runs]),
            "first_F_s_per_system": [
                stats([r["detail"]["first_F_s"][i] for r in runs])
                for i in range(len(runs[0]["detail"]["first_F_s"]))
            ],
            "first_J_s_per_system": [
                stats([r["detail"]["first_J_s"][i] for r in runs])
                for i in range(len(runs[0]["detail"]["first_J_s"]))
            ],
        },
        "nit": runs[0]["nit"],
        "peak_rss_gb": max(r["mark"]["peak_rss_gb"] for r in runs),
        "pyomo_imported": runs[0]["mark"]["pyomo_imported_after"],
        "pyomo_imported_at_start": runs[0]["mark"]["pyomo_imported_before"],
        "pyomo_modules": runs[0]["mark"]["pyomo_modules"],
    }


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", nargs="?", default="gtap7_20x41")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--cache-root", type=Path, default=Path.home() / ".cache/equilibria/family"
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-load", type=float, default=4.0)
    ap.add_argument("--max-wait", type=float, default=600.0)
    ap.add_argument(
        "--gc",
        choices=("route", "off"),
        default="route",
        help="route: the garbage collector as solve_gtap_family sets it (default); "
        "off: gc.disable() in the child, for comparison",
    )
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    loads = []
    runs = []
    for i in range(a.runs + 1):  # run 0 is the warm-up
        before = machine()
        waited = 0.0
        # the 1-minute load average still carries the previous run: let it settle
        while before["load_1m"] > a.max_load and not a.force and waited < a.max_wait:
            time.sleep(15)
            waited += 15
            before = machine()
        if before["load_1m"] > a.max_load and not a.force:
            print(f"load {before['load_1m']:.2f} > {a.max_load}: stop", file=sys.stderr)
            return 2
        before["waited_s"] = waited
        r = run_child(a.dataset, a.cache_root, gc_off=a.gc == "off")
        after = machine()
        loads.append({"run": i, "warm_up": i == 0, "before": before, "after": after})
        print(
            f"run {i}{' (warm-up)' if i == 0 else ''}: process "
            f"{r['process_wall_s'] / 60:.2f} min, solve_shock "
            f"{r['mark']['solve_shock_s'] / 60:.2f} min, load "
            f"{before['load_1m']:.2f}->{after['load_1m']:.2f}",
            flush=True,
        )
        if i > 0:
            runs.append(r)
    out = {
        "dataset": a.dataset,
        "sha": _sha(),
        "gc": "disabled in the child" if a.gc == "off" else "as the route sets it",
        "runs_measured": len(runs),
        "machine": loads,
        "import_time": import_time(a.dataset),
        "individual_runs": [
            {
                "process_wall_s": r["process_wall_s"],
                "solve_shock_s": r["mark"]["solve_shock_s"],
                "code": r["code"],
            }
            for r in runs
        ],
        **aggregate(runs),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=str))
    w = out["process_wall_s"]
    print(
        f"process wall median {w['median'] / 60:.2f} min "
        f"(min {w['min'] / 60:.2f}, max {w['max'] / 60:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
