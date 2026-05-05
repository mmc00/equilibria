"""Parity test: NUS333 (3x2) Python equilibria vs NEOS GAMS comp_nus333.gms shock.

Runs the full base + 10% import-tariff power shock chain and asserts each
(var, region) Δ% matches the NEOS reference (job 18744693) within tolerance.

The 12 structural bugs documented in
`memory/gtap_nus333_baseline_shock_bugs.md` were verified against this
test case. Tolerance is set tight (0.01pp) because parity is currently
exact at the 3-decimal level.

Slow (~3 min). Marked with @pytest.mark.slow. Skips when:
  - NUS333 dataset directory is unset/missing
  - PATH C-API library is unavailable (path_capi_python)
  - harpy3 (GEMPACK reader) not installed

Run manually:
    EQUILIBRIA_NUS333_DIR=/Users/marmol/Downloads/10284 \
    uv run --with harpy3 pytest tests/templates/gtap/test_nus333_neos_parity.py -v -s
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

# NEOS reference (job 18744693, comp_nus333.gms after tariff power shock).
NEOS_REF = {
    "gdpmp": {"USA": (14.0617801139, 14.7063426837), "ROW": (41.7695611026, 42.4668724204)},
    "regy":  {"USA": (12.8018441072, 13.3823164142), "ROW": (37.0999386636, 37.6635926896)},
    "u":     {"USA": (1.0,            1.0016493028),  "ROW": (1.0,           0.9918010965)},
}

TOL_PP = 0.01  # absolute Δ% tolerance in percentage points


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
        import harpy  # noqa: F401  (provided by the harpy3 package)
    except ImportError:
        return "harpy not installed (uv run --with harpy3 ...)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@pytest.fixture(scope="module")
def nus333_results():
    """Run the base + shock pipeline once and yield the extracted key macros."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(PATH_CAPI_SRC))
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

    from compare_nus333_vs_neos import (
        _solve,
        _apply_tariff_shock,
        _copy_var_levels,
        _extract_key,
    )
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

    _apply_tariff_shock(params, factor=1.10)
    builder_s = GTAPModelEquations(
        params.sets, params, residual_region="ROW", closure=closure,
        t0_snapshot=model_b,
    )
    model_s = builder_s.build_model()
    _copy_var_levels(model_b, model_s)
    _solve(model_s, params, label="shock")
    shock = _extract_key(model_s, params)

    return base, shock


@pytest.mark.slow
@pytest.mark.parametrize("var,region", [
    ("gdpmp", "USA"), ("gdpmp", "ROW"),
    ("regy", "USA"),  ("regy", "ROW"),
    ("u", "USA"),     ("u", "ROW"),
])
def test_nus333_neos_delta_parity(nus333_results, var, region):
    """Each macro Δ% must be within TOL_PP of the NEOS GAMS reference."""
    base, shock = nus333_results
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
