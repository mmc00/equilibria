"""Burfisher on the family (JAX) backend (MAR-251): every exercise of
``run_burfisher.EXERCISES`` gives on family what it gives on Pyomo, and the GEMPACK
comparison of the oracle's two exercises gives the same numbers from either backend.

Johansen exercises (GEMPACK's method for them) are solved as Johansen, with the
looser bound of ``tests/family/test_family_johansen.py``."""

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


def _exercises() -> list[str]:
    try:
        return sorted(_rb().EXERCISES)
    except pytest.skip.Exception:
        return []


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
        {k: v for k, v in want.items() if k[0] not in loose},
        {k: v for k, v in got.items() if k[0] not in loose},
        rel,
    )
    for var, bound in loose.items():
        cells = {k: v for k, v in want.items() if k[0] == var}
        bad += mismatches(cells, {k: got[k] for k in cells if k in got}, bound)
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
    difference is ~2%, as in the oracle.)

    Not the oracle's ``EX18_MAX``: that was measured on the stored levels of
    ``nus333/equilibria_burfisher_levels``, solved before capFlex risk moved to
    cal.gms (tbl46a 1.4881e-4 there, 1.4897e-4 from a solve now, on both backends)."""
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
