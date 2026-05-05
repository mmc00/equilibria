"""Parity test: NUS333 (3x2) Python equilibria vs NEOS GAMS ROW LAND shock.

Runs base + 30% productivity cut on ROW LAND (re-fix xft[ROW,LAND] *= 0.7)
and asserts macro Δ% matches the NEOS reference (job 18753059, ran
comp_nus333_landshock.gms) within tolerance.

Reference values are GAMS LANDSHOCK.gdx levels at t='base' and t='shock'.
Python and GAMS produce identical values to ~4 decimal places — TOL_PP set
loose at 0.05pp to absorb solver tolerance.

Slow (~3 min). Skips when:
  - NUS333 dataset directory is unset/missing
  - PATH C-API library is unavailable (path_capi_python)
  - harpy3 (GEMPACK reader) not installed

Run manually:
    EQUILIBRIA_NUS333_DIR=/Users/marmol/Downloads/10284 \\
    uv run --with harpy3 pytest \\
        tests/templates/gtap/test_nus333_landshock_parity.py -v -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NUS333_DIR = Path(os.environ.get("EQUILIBRIA_NUS333_DIR", "/Users/marmol/Downloads/10284"))
PATH_LIB = ROOT / ".cache/path_capi/libpath50.silicon.dylib"
PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")

# NEOS reference (job 18753059, comp_nus333_landshock.gms with -30% ROW LAND).
# (base, shock) levels straight from out.gdx.
NEOS_REF = {
    "gdpmp": {"USA": (14.0617801139, 13.9713240), "ROW": (41.7695610, 40.7481620)},
    "regy":  {"USA": (12.8018441,    12.7200460), "ROW": (37.0999390, 36.1951400)},
    "u":     {"USA": (1.0,            1.0013400),  "ROW": (1.0,        0.9891750)},
}

LAND_CUT = 0.30  # cut 30% of ROW land productivity (matches GAMS landCut)
TOL_PP = 0.05    # absolute Δ% tolerance in percentage points


def _skip_reason() -> str | None:
    if not NUS333_DIR.is_dir():
        return f"NUS333 dataset not found at {NUS333_DIR} (set EQUILIBRIA_NUS333_DIR)"
    for fname in ("basedata.har", "sets.har", "default.prm", "baserate.har"):
        if not (NUS333_DIR / fname).is_file():
            return f"NUS333 missing {fname}"
    if not PATH_LIB.exists():
        return f"PATH C-API library not found at {PATH_LIB}"
    if not PATH_CAPI_SRC.is_dir():
        return f"path-capi-python source not found at {PATH_CAPI_SRC}"
    try:
        import harpy  # noqa: F401
    except ImportError:
        return "harpy not installed (uv run --with harpy3 ...)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _apply_land_productivity_shock(model, region: str, factor: float) -> None:
    """Re-fix xft[region,LAND] = base_level * factor (mirror of GAMS landshock)."""
    from pyomo.environ import value

    for idx in list(model.xft):
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        if idx_t[0] == region and idx_t[1] == "LAND":
            v = model.xft[idx]
            v.fix(float(value(v)) * factor)


@pytest.fixture(scope="module")
def landshock_results():
    """Run base + ROW LAND productivity shock once and yield extracted macros."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(PATH_CAPI_SRC))
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

    from compare_nus333_vs_neos import _solve, _copy_var_levels, _extract_key
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_solver import GTAPClosureConfig

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333_DIR / "basedata.har",
        sets_path=NUS333_DIR / "sets.har",
        default_path=NUS333_DIR / "default.prm",
        baserate_path=NUS333_DIR / "baserate.har",
    )
    closure = GTAPClosureConfig(if_sub=False)

    builder_b = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model_b = builder_b.build_model()
    _solve(model_b, params, label="base")
    base = _extract_key(model_b, params)

    builder_s = GTAPModelEquations(
        params.sets, params, residual_region="ROW", closure=closure,
        t0_snapshot=model_b,
    )
    model_s = builder_s.build_model()
    _copy_var_levels(model_b, model_s)
    _apply_land_productivity_shock(model_s, region="ROW", factor=1.0 - LAND_CUT)
    _solve(model_s, params, label="shock")
    shock = _extract_key(model_s, params)

    return base, shock


@pytest.mark.slow
@pytest.mark.parametrize("var,region", [
    ("gdpmp", "USA"), ("gdpmp", "ROW"),
    ("regy", "USA"),  ("regy", "ROW"),
    ("u", "USA"),     ("u", "ROW"),
])
def test_nus333_landshock_delta_parity(landshock_results, var, region):
    """Each macro Δ% must be within TOL_PP of the NEOS GAMS landshock reference."""
    base, shock = landshock_results
    py_b = base[var][region]
    py_s = shock[var][region]
    py_dpct = (py_s / py_b - 1.0) * 100.0

    ref_b, ref_s = NEOS_REF[var][region]
    ref_dpct = (ref_s / ref_b - 1.0) * 100.0

    diff_pp = py_dpct - ref_dpct
    assert abs(diff_pp) <= TOL_PP, (
        f"{var}[{region}] Δ% mismatch: "
        f"py={py_dpct:+.4f}% gams={ref_dpct:+.4f}% diff={diff_pp:+.4f}pp "
        f"(tol {TOL_PP}pp)"
    )
