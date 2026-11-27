"""Burfisher on the family (JAX) backend (MAR-251): every exercise of
``run_burfisher.EXERCISES`` gives on family what it gives on Pyomo, and the GEMPACK
comparison of the oracle's two exercises gives the same numbers from either backend.

Johansen exercises (GEMPACK's method for them) are solved as Johansen, with a
looser bound (``REL_JOHANSEN``)."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from tests.family._period_levels import levels, mismatches, solver_env
from tests.templates.gtap._nus333 import ROOT, nus333_params

pytest.importorskip("jax")
pytestmark = pytest.mark.integration

# the exercises GEMPACK solves with Johansen (test_burfisher_gempack)
JOHANSEN = {"TBL45A", "TBL45B", "TBL45C", "TBL64", "ME4A", "ME4B", "GTAPV7"}
REL = 1e-9
# Richardson combines four solves at lambda = +-h, +-2h (h = 0.01) as
# (8(f1 - f-1) - (f2 - f-2)) / 12h: a solve error e becomes up to 18e/12h = 150e in
# the slope. Measured on TBL45A: family vs Pyomo 1.6e-10 after a nonlinear shock,
# 2.0e-8 after Johansen (GTAPV7 2.6e-8, check period 1.7e-13). The bound is 150 x REL.
REL_JOHANSEN = 1.5e-7
# TBL62B (sectorspecifick.prm) has ETRAE(CAPITAL) = -1e-5: pft(CAPITAL) enters
# eq_xfteq, the only row that pins it, with that exponent, so a solve knows it only to
# residual / |ETRAE| (measured: 1.7e-7 apart, both backends at the same residual). Its
# bound is REL / |ETRAE|.
LOOSE = {"TBL62B": {"pft": REL / 1e-5}}


def _rb() -> Any:
    nus333_params()  # SKIP without the dataset
    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    from equilibria._local_refs import path_capi_src

    sys.path.insert(0, str(path_capi_src()))
    return importlib.import_module("run_burfisher")


# One exercise per mechanism runs in PR CI (the plan-groups budget); the other ones
# are ``slow``: run them all locally with ``pytest -m slow`` or no marker filter.
IN_CI = {
    "TBL46A",  # nonlinear in steps, default.prm
    "ME9B",  # capital ROW +213 %: the one that needs the steps
    "TBL45A",  # Johansen
    "ME4A",  # Johansen, land productivity -81 %
    "TBL62B",  # sector-specific capital (ETRAE -1e-5)
    "TBL62C",  # sluggish capital
}


def _exercises() -> list[Any]:
    try:
        names = sorted(_rb().EXERCISES)
    except pytest.skip.Exception:
        return []
    return [e if e in IN_CI else pytest.param(e, marks=pytest.mark.slow) for e in names]


def _solve(exp: str, backend: str, cache_root) -> tuple[dict, Any]:
    from equilibria._local_refs import nus333_dir

    method = "johansen" if exp in JOHANSEN else "nonlinear"
    with solver_env({"EQUILIBRIA_FAMILY_CACHE": str(cache_root)}):
        ex = _rb().exercise(exp, nus333_dir(), shock_method=method, backend=backend)
        out = ex.run()
    return {k: v["code"] for k, v in out.items()}, ex


@pytest.mark.parametrize("exp", _exercises())
def test_family_equals_pyomo(exp: str, tmp_path) -> None:
    p_codes, p = _solve(exp, "pyomo", tmp_path)
    f_codes, f = _solve(exp, "family", tmp_path)
    assert f_codes == p_codes
    assert f_codes["shock"] == "optimal"
    want, got = levels(p.model), levels(f.model)
    rel = REL_JOHANSEN if exp in JOHANSEN else REL
    loose = LOOSE.get(exp, {})
    bad = mismatches(
        {k: v for k, v in got.items() if k[0] not in loose},
        {k: v for k, v in want.items() if k[0] not in loose},
        rel,
    )
    for var, bound in loose.items():
        bad += mismatches(
            {k: v for k, v in got.items() if k[0] == var},
            {k: v for k, v in want.items() if k[0] == var},
            bound,
        )
    assert not bad, f"{exp}: {len(bad)} of {len(got)} cells differ, first: {bad[:5]}"


def test_period_model_levels_read_pyomo_as_model_levels(tmp_path) -> None:
    """The backend-neutral reader gives, on a Pyomo run, what ``model_levels`` reads
    off the Pyomo model."""
    from equilibria.compare.gtap7_side import model_levels, period_model_levels

    _, ex = _solve("TBL46A", "pyomo", tmp_path)
    assert period_model_levels(ex.model) == model_levels(ex.model.pyomo)


@pytest.mark.parametrize("exp", ["tbl46a", "tbl45a"])
def test_family_against_gempack_gives_pyomos_report(exp: str) -> None:
    """``compare_burfisher_gempack`` from family levels: the same report as from
    Pyomo levels solved now (cells, within-counts, largest difference and its cell).
    (tbl45a is the book's Johansen run against a nonlinear solve: its largest
    difference is ~2%, as in the oracle.)"""
    from tests.compare import _oracle as O

    sys.path[:0] = [str(ROOT / "scripts" / "gtap")]
    cbg = importlib.import_module("compare_burfisher_gempack")
    reports = {}
    for backend in ("pyomo", "family"):
        with solver_env():
            levels_, xscale = cbg.solve(exp.upper(), backend=backend)
        reports[backend] = cbg.run(exp, levels_, xscale, O.EX18_FIX)
    p, f = reports["pyomo"], reports["family"]
    assert {k: s["cells"] for k, s in f.summary.items()} == O.EX18_CELLS
    assert f.overall["cells"] == p.overall["cells"] == O.EX18_TOTAL
    assert f.overall["pct_within"] == p.overall["pct_within"]
    assert f.overall["max"] == pytest.approx(p.overall["max"], rel=1e-5)
    assert (f.overall["max_variable"], f.overall["max_cell"]) == (
        p.overall["max_variable"],
        p.overall["max_cell"],
    )
    assert not f.unmatched
