"""Johansen on the family route (MAR-251): two of the exercises GEMPACK solves with
Johansen (TBL45A, GTAPV7) equal the Pyomo run cell by cell, the cells the period
recomputes after each solve (``after_shock_point``) included."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo")

from equilibria._local_refs import nus333_dir  # noqa: E402
from tests.family._period_levels import levels, mismatches, solver_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Richardson combines four solves at lambda = +-h, +-2h (h = 0.01) as
# (8(f1 - f-1) - (f2 - f-2)) / 12h: a solve error e becomes up to 18e/12h = 150e in
# the slope. Measured on TBL45A: family vs Pyomo 1.6e-10 after a nonlinear shock,
# 2.0e-8 after Johansen (GTAPV7 2.6e-8). The bound is 150 x the 1e-9 of the nonlinear
# e2e tests.
REL = 1.5e-7


def _rb():
    spec = importlib.util.spec_from_file_location(
        "run_burfisher", ROOT / "scripts" / "gtap" / "run_burfisher.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve(exp: str, backend: str, cache_root=None):
    extra = {} if cache_root is None else {"EQUILIBRIA_FAMILY_CACHE": str(cache_root)}
    with solver_env(extra):
        ex = _rb().exercise(exp, nus333_dir(), shock_method="johansen", backend=backend)
        out = ex.run()
    return {k: v["code"] for k, v in out.items()}, levels(ex.model)


@pytest.mark.parametrize("exp", ["TBL45A", "GTAPV7"])
def test_family_johansen_matches_pyomo(exp, tmp_path):
    p_codes, p = _solve(exp, "pyomo")
    f_codes, f = _solve(exp, "family", tmp_path)
    assert p_codes["shock"] == f_codes["shock"] == "optimal"
    worst = max(
        (abs(f[k] - p[k]) / max(1.0, abs(p[k])) for k in f if k in p), default=0.0
    )
    print(f"{exp}: max rel family vs Pyomo {worst:.3e} over {len(f)} cells")
    bad = mismatches(p, f, REL)
    assert not bad, f"{len(bad)} of {len(f)} cells differ, first: {bad[:10]}"
