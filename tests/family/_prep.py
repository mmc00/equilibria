"""Shared 3x3 ifSUB=1 two-photo capture for the period-prep tests (one solve per session)."""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _newton_mumps() -> dict:
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


@cache
def capture_dir(dataset: str = "gtap7_3x3") -> Path:
    """Capture ``dataset`` once per session into a temp dir (in-process solve)."""
    import atexit
    import os
    import shutil
    import tempfile

    from equilibria.family import prep_capture
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    out = Path(tempfile.mkdtemp(prefix=f"prep_{dataset}_"))
    # Photos run to hundreds of MB on 15x10: never leave them behind.
    atexit.register(shutil.rmtree, out, True)
    env = {
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct",
        prep_capture.ENV: str(out),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    prep_capture.reset()
    try:
        _, code = _solve_shock(dataset, 1, "capFix", solve_nlp=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert code == 1
    return out
