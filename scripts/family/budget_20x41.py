"""Budget of the no-Pyomo path on a GTAP dataset (ifSUB=1, capFix, gtap mode).

Times every piece of the future path that already exists, adds the measured
family Newton + prep times and projects the total against the 2-minute goal
(spec 2026-11-04-family-backend-multiperiod-design.md). Pieces not built yet are
listed in ``not_measured`` (none left: see ``scripts/family/timing_20x41.py``).

    <python with pymumps> scripts/family/budget_20x41.py [--dataset gtap7_20x41]
        [--cache-root DIR] [--out FILE]

Run it twice: the first (cold) run fills the record and JAX caches, the second
(warm) one is the measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Pyomo's 3 Newton solves (settle, check, shock) on 20x41, plans/paper/mediciones/
# fases_20x41_d44692d.json ("Newton-TR + MUMPS"). Reported as a reference only;
# it is not part of projected_s.
NEWTON_TODAY_S = 48.0
# The 3 Newton solves on the family squared system (MUMPS ICNTL(6)=7, iterations
# 3/3/22, 25 factorizations), plans/research/mediciones/
# family_newton_20x41_79260a63_mumps_auto_quiet.json ("newton_projected_s").
NEWTON_FAMILY_S = 23.2
# The period prep on arrays (recipe + squaring) of the 3 solve calls on 20x41,
# plans/research/mediciones/family_prep_20x41_7e99669c.json ("prep_per_call_total_s").
# Its single-period reference is compose_block_state, already timed here.
PERIOD_PREP_ON_ARRAYS_S = 9.3
# Once per run: the period incidence from the J pattern (8.82) plus the conditional
# masks (2.43), same file, "incidence_once_s" + "masks_once_s".
PREP_ONCE_S = 11.3
# The pieces that were "not measured", now measured end to end by
# scripts/family/timing_20x41.py (median of 3 runs after a warm-up, GC off),
# plans/research/mediciones/family_e2e_timing_20x41_1c2d00bd.json. They do not overlap
# with the projected pieces above.
MEASURED_REST_S = {
    # cross-period records: cache key (0.81) + load (0.43)
    "cross_period_rows": 1.24,
    # the settle's own calls are phases of the same recipe/square/build/Newton code
    # counted in the per-call pieces; nothing is left outside them
    "calibration_rest": 0.0,
    # process start/exit 3.13 + imports 1.04 + _solve_shock before/after the family
    # call 1.59 + family call outside its phases 2.12
    "outside_phases": 7.88,
}
# What the whole route takes in practice: wall of the child process and wall from the
# start of _solve_shock to its return. Median of 3 after a warm-up, GC as the route sets
# it (gen-0 threshold 50000 plus gc.freeze() at entry, after the records load and before
# mp_build), d664e86e (kernels from stored executables, no JAX trace), load average 3.4-4.0 at
# start (5.4-6.8 at the end of each run); the same code before the executable cache
# (e1e368a3) took 116.6 s:
# plans/research/mediciones/family_e2e_timing_20x41_d664e86e_gc_route.json
E2E_PROCESS_WALL_S = 81.0
E2E_SOLVE_SHOCK_S = 78.4
NOT_MEASURED: dict[str, str] = {}


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    # The closure tests/templates/gtap/test_gtap7_gempack_parity.py::_solve_shock uses.
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )


class _Timer:
    def __init__(self) -> None:
        self.pieces: dict[str, float | None] = {}

    def __call__(self, name: str, fn):
        t = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:
            raise RuntimeError(f"piece {name!r} failed: {exc!r}") from exc
        self.pieces[name] = time.perf_counter() - t
        return out


def measure(dataset: str, cache_root: Path) -> dict:
    """Time the pieces in this process; any failing piece raises ``RuntimeError``."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import maximum_bipartite_matching

    import equilibria.templates.gtap.gtap_block_model as gbm
    from equilibria.backends.family_backend import FamilyBackend
    from equilibria.family.disk_cache import cache_key, enable_jax_cache
    from equilibria.sources import dataset_dir
    from equilibria.templates.gtap import GTAPParameters

    cache_root = Path(cache_root)
    enable_jax_cache(cache_root)
    tm = _Timer()
    closure = _closure()

    def load_params():
        d = dataset_dir(dataset)
        p = GTAPParameters()
        p.load_from_har(
            basedata_path=d / "basedata.har",
            sets_path=d / "sets.har",
            default_path=d / "default.prm",
            baserate_path=d / "baserate.har",
        )
        return p

    p = tm("load_params", load_params)
    rr = list(p.sets.r)[-1]
    model, state = tm(
        "compose_state", lambda: gbm.compose_block_state(p, p.sets, closure, rr)
    )

    key = tm("cache_key", lambda: cache_key(p, closure, rr))
    if key is None:
        raise RuntimeError("piece 'cache_key' failed: no cache key (uncoverable input)")
    path = cache_root / key / "records.pkl"
    cold = not path.exists()
    if cold:

        def record():
            b = FamilyBackend()
            b.build(model, skip=state.ifsub_skipped)
            b.save(path)
            return b

        tm("record_and_save", record)
    backend = tm("cache_load", lambda: FamilyBackend.load(path))

    def layout_system():
        layout = backend.layout(state)
        return layout, backend.system(layout)

    layout, system = tm("layout_system", layout_system)
    z = layout.free_values

    def fj():
        F = system.F(z)
        J = system.jac(z)
        return F, J

    _, J = tm("first_FJ", fj)
    tm("warm_FJ", fj)

    def names():
        rows = np.array(system.row_names)
        cols = np.array(layout.col_names)
        return np.argsort(rows), np.argsort(cols)

    tm("names", names)
    Jc = sp.csr_matrix((np.ones(J.nnz), (J.row, J.col)), shape=J.shape)
    match = tm("matching", lambda: maximum_bipartite_matching(Jc, perm_type="column"))

    not_measured = dict(NOT_MEASURED)
    if J.shape[0] != J.shape[1]:
        not_measured["mumps_factor"] = f"single-period J is not square {J.shape}"
    else:
        try:
            import mumps
        except ImportError:
            not_measured["mumps_factor"] = "pymumps not importable"
        else:

            def factor():
                ctx = mumps.DMumpsContext()
                ctx.set_silent()
                ctx.set_centralized_sparse(J.tocoo())
                rhs = np.ones(J.shape[0])
                ctx.set_rhs(rhs)
                ctx.run(job=6)
                ctx.destroy()

            tm("mumps_factor", factor)

    pc = tm.pieces
    per_period = pc["layout_system"] + pc["first_FJ"] + pc["names"] + pc["matching"]
    projected = (
        pc["load_params"]
        + pc["compose_state"]
        + pc["cache_key"]
        + pc["cache_load"]
        + 3 * per_period
        + NEWTON_FAMILY_S
        + PERIOD_PREP_ON_ARRAYS_S
        + PREP_ONCE_S
    )
    verdict = "go" if projected <= 100 else "tight" if projected <= 120 else "no"
    with_rest = projected + sum(MEASURED_REST_S.values())
    return {
        "dataset": dataset,
        "if_sub": 1,
        "cold": cold,
        "rows": int(J.shape[0]),
        "cols": int(J.shape[1]),
        "nnz": int(J.nnz),
        "unmatched_rows": int(np.count_nonzero(match < 0)),
        "pieces": {k: float(v) for k, v in pc.items() if v is not None},
        "newton_family_s": NEWTON_FAMILY_S,
        "newton_today_s": NEWTON_TODAY_S,  # reference, not in projected_s
        "period_prep_on_arrays_s": PERIOD_PREP_ON_ARRAYS_S,
        "prep_once_s": PREP_ONCE_S,
        "projected_s": float(projected),
        "verdict": verdict,
        "verdict_conditional": bool(not_measured),
        "with_measured_rest_s": float(with_rest),
        "e2e_process_wall_s": E2E_PROCESS_WALL_S,
        "e2e_solve_shock_s": E2E_SOLVE_SHOCK_S,
        "measured_rest_s": dict(MEASURED_REST_S),
        "not_measured": not_measured,
    }


_CHILD = """
import gc, json, sys
gc.disable()
sys.path[:0] = [{src!r}, {here!r}]
from budget_20x41 import measure
print("RESULT " + json.dumps(measure({dataset!r}, {root!r})))
"""


def run(dataset: str, cache_root: Path, warm: bool) -> dict:
    """``measure`` in a clean child process with GC off; ``warm`` is recorded only."""
    code = _CHILD.format(
        src=str(ROOT / "src"),
        here=str(Path(__file__).resolve().parent),
        dataset=dataset,
        root=str(cache_root),
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(p.stderr[-4000:])
    out = json.loads(line[len("RESULT ") :])
    out["warm_requested"] = warm
    return out


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_20x41")
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "equilibria" / "family",
    )
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cold = run(a.dataset, a.cache_root, warm=False)
    warm = run(a.dataset, a.cache_root, warm=True)
    result = {"sha": _sha(), "cold": cold, "warm": warm}
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
