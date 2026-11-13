"""Golden of the Pyomo Newton-TR path on 3x3 (per-iteration trace, counts, final levels).

Recorded at the commit before ``solver/newton_tr.py`` was extracted; the refactor must
reproduce it bitwise. Floats are compared through their exact repr; the levels of every
Var cell are hashed per variable (bytes of value, fixed, lb, ub) so a mismatch names the
variable. Bitwise equality only holds on the interpreter and linear-solver build that
recorded it (``sum`` changed in 3.12; MUMPS is a different library), hence one golden per
``(linsolve, python minor)``.

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        .venv/bin/python -m tests.family._newton_golden direct         # record
    (same, with ~/proyectos/Pedro/.venv/bin/python, for mumps)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden"


def golden_path(linsolve: str) -> Path:
    return (
        GOLDEN
        / f"newton_tr_3x3_{linsolve}_py{sys.version_info.major}{sys.version_info.minor}.json"
    )


def _newton_mumps() -> dict:
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "perf_budget_20x41", root / "scripts" / "gtap" / "perf_budget_20x41.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


def _digest(vc) -> str:
    h = hashlib.sha256()
    for a in (vc.value, vc.fixed, vc.lb, vc.ub):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def run(linsolve: str) -> dict:
    """Solve 3x3 (ifSUB=1, capFix) with the capture on and return the golden record."""
    from equilibria.family import prep_capture
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    env = {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}
    saved = {k: os.environ.get(k) for k in [*env, prep_capture.ENV]}
    os.environ.update(env)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[prep_capture.ENV] = tmp
            prep_capture.reset()
            _, code = _solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True)
            calls = []
            for f in sorted(Path(tmp).glob("*_S_*.json")):
                s = json.loads(f.read_text())
                calls.append(
                    {
                        k: s[k]
                        for k in (
                            "n",
                            "period",
                            "code",
                            "converged",
                            "nit",
                            "nfact",
                            "residual",
                            "trace_f_inf",
                            "trace_f_scaled_inf",
                            "gmin",
                            "icntl",
                            "null_pivots",
                        )
                    }
                )
            final = prep_capture.load(Path(tmp) / "final.pkl")
            levels = {name: _digest(vc) for name, vc in sorted(final.vars.items())}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return {"code": code, "calls": calls, "levels": levels}


# MUMPS and the BLAS under it sum in a thread-dependent order, which moves the last
# digits of the final iterations (about 1 run in 8 at default threads). Both the
# recording and the test therefore run the solve in a child with one thread.
SINGLE_THREAD = dict.fromkeys(
    (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ),
    "1",
)


def run_in_child(linsolve: str) -> dict:
    """``run`` in a fresh single-threaded interpreter of the current python."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "golden.json"
        cmd = [sys.executable, "-m", "tests.family._newton_golden", linsolve]
        subprocess.run(
            [*cmd, "--out", str(out)],
            cwd=root,
            env={
                **os.environ,
                **SINGLE_THREAD,
                "PYTHONPATH": os.pathsep.join(sys.path),
            },
            check=True,
            capture_output=True,
        )
        return json.loads(out.read_text())


if __name__ == "__main__":
    ls = sys.argv[1]
    rec = run(ls)
    p = Path(sys.argv[3]) if sys.argv[2:3] == ["--out"] else golden_path(ls)
    p.write_text(json.dumps(rec, indent=1))
    print(p, "code", rec["code"], [(c["nit"], c["nfact"]) for c in rec["calls"]])
