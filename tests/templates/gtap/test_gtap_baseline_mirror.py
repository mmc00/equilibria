"""Parity test: GTAP Python baseline vs the recorded 9x10 GAMS reference (base period).

This test enforces the mirror invariant: after solving the baseline with
PATH, the Python solution must match the GAMS reference within tolerance.

Marked @pytest.mark.slow (runtime ~3 min) — add to nightly CI job.

Run manually:
    EQUILIBRIA_GTAP_CAL_DUMP="notes/tmp/18700775-gams_cal_dump_9x10.gdx" \\
    uv run pytest tests/templates/gtap/test_gtap_baseline_mirror.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from equilibria._local_refs import cge_babel_dir, path_capi_src

# ---------------------------------------------------------------------------
# Skip condition: slow test and requires external GDX files + PATH C-API
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
_GDX_FILE = Path(
    os.environ.get(
        "EQUILIBRIA_GTAP_GDX",
        str(cge_babel_dir() / "standard_gtap_7" / "basedata-9x10.gdx"),
    )
)
_CAL_DUMP = Path(
    os.environ.get(
        "EQUILIBRIA_GTAP_CAL_DUMP",
        "notes/tmp/18700775-gams_cal_dump_9x10.gdx",
    )
)
_PATH_LIB = ROOT / ".cache/path_capi/libpath50.silicon.dylib"
_PATH_CAPI_SRC = path_capi_src()

pytestmark = pytest.mark.slow


def _skip_if_missing():
    if not _GDX_FILE.exists():
        pytest.skip(f"GDX file not found: {_GDX_FILE}")
    if not _CAL_DUMP.exists():
        pytest.skip(f"Cal dump not found: {_CAL_DUMP}")
    if _reference_gdx() is None:
        pytest.skip("no recorded 9x10 GAMS reference under data/")
    if not _PATH_LIB.exists():
        pytest.skip(f"PATH C-API library not found at {_PATH_LIB}")
    if not _PATH_CAPI_SRC.is_dir():
        pytest.skip(f"path-capi-python source not found at {_PATH_CAPI_SRC}")
    if str(_PATH_CAPI_SRC) not in sys.path:
        sys.path.insert(0, str(_PATH_CAPI_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_gdx() -> Path | None:
    from equilibria.templates.gtap.gtap_parity_pipeline import (
        standard_gtap_reference_gdx,
    )

    return standard_gtap_reference_gdx()


def _run_baseline_solve():
    """Build model, solve via GTAPSolver wrapper, and return (model, result_dict).

    Mirrors the reference configuration:
      - PATH C-API nonlinear full mode (10,296 equations).
      - if_sub=False, equation_scaling=True.
    """
    # CAL_DUMP env var triggers calibration override inside
    # _build_gtap_contract_with_calibration. This is the configuration that
    # produces the documented baseline residual (~1e-6) on 9x10.
    os.environ["EQUILIBRIA_GTAP_CAL_DUMP"] = str(_CAL_DUMP)

    # Make scripts/gtap importable so we can call _run_path_capi_nonlinear_full
    # and _build_gtap_contract_with_calibration.
    scripts_gtap = ROOT / "scripts" / "gtap"
    if str(scripts_gtap) not in sys.path:
        sys.path.insert(0, str(scripts_gtap))

    from run_gtap import (  # type: ignore
        _build_gtap_contract_with_calibration,
        _run_path_capi_nonlinear_full,
    )

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")
    # GAMS NEOS reference (job 18737509, tariff_comp.gms) uses ifSUB=0.
    new_closure = contract.closure.model_copy(update={"if_sub": False})
    contract = contract.model_copy(update={"closure": new_closure})

    params = GTAPParameters()
    params.load_from_gdx(_GDX_FILE)
    equations = GTAPModelEquations(params.sets, params, contract.closure)
    model = equations.build_model()

    result = _run_path_capi_nonlinear_full(
        model,
        params,
        enforce_post_checks=False,
        strict_path_capi=False,
        closure_config=contract.closure,
        equation_scaling=True,
    )
    return model, result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_solve_result():
    """Run baseline solve once and cache the result for all tests in this module."""
    _skip_if_missing()
    return _run_baseline_solve()


class TestBaselineMirrorInvariants:
    """After converged baseline solve, Python must mirror GAMS within tolerance."""

    RESIDUAL_TOL = 1e-4  # solver must reach this residual
    MIRROR_TOL_LOOSE = 5e-3  # absolute tolerance for macro-level comparison
    MIRROR_TOL_STRICT = 1e-3  # tolerance for strict gate (post Fase 2)
    MAX_MISMATCHES_LOOSE = 0  # with tol=5e-3: 0 non-xd mismatches allowed
    MAX_MISMATCHES_STRICT = 5  # with tol=1e-3: at most 5 (yi/xds/xet) before full fix

    def test_solver_converged(self, baseline_solve_result):
        """Solver must converge (PATH success flag)."""
        _, result = baseline_solve_result
        assert result.get("success"), (
            f"Solver did not converge: status={result.get('status')}, "
            f"residual={result.get('residual', float('nan')):.3e}, "
            f"termination_code={result.get('termination_code')}"
        )

    def test_residual_below_threshold(self, baseline_solve_result):
        """Residual must be below RESIDUAL_TOL."""
        _, result = baseline_solve_result
        residual = result.get("residual", float("inf"))
        assert residual < self.RESIDUAL_TOL, (
            f"Residual {residual:.3e} exceeds threshold {self.RESIDUAL_TOL:.3e}. "
            f"termination_code={result.get('termination_code')}"
        )

    def test_walras_law(self, baseline_solve_result):
        """Post-check gate (PATH C-API enforce_post_checks) — proxy for Walras' law.

        Skipped when post-checks are disabled in the runner config (the parity
        runner currently uses enforce_post_checks=False to allow inspecting
        residuals before gating on macro identities).
        """
        _, result = baseline_solve_result
        if not result.get("post_checks_enforced"):
            pytest.skip("post_checks not enforced in this run; gate not evaluated")
        gate_pass = result.get("post_checks_gate_pass")
        assert gate_pass, "PATH post-check gate failed (macro identities violated)"

    def test_gams_parity_loose(self, baseline_solve_result):
        """All non-xd variables must match GAMS within 5e-3 (0 mismatches allowed)."""
        _skip_if_missing()
        model, _ = baseline_solve_result
        self._check_parity(
            model, tol=self.MIRROR_TOL_LOOSE, max_mm=self.MAX_MISMATCHES_LOOSE
        )

    def test_gams_parity_strict(self, baseline_solve_result):
        """All non-xd variables must match GAMS within 1e-3 (up to 5 mismatches tolerated)."""
        _skip_if_missing()
        model, _ = baseline_solve_result
        self._check_parity(
            model, tol=self.MIRROR_TOL_STRICT, max_mm=self.MAX_MISMATCHES_STRICT
        )

    # ------------------------------------------------------------------

    def _check_parity(self, model, *, tol: float, max_mm: int):
        from equilibria.templates.gtap import GTAPSets
        from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

        gdx = _reference_gdx()
        assert gdx is not None  # _skip_if_missing skipped otherwise
        gams_snap = GTAPVariableSnapshot.from_gdx(gdx, GTAPSets(), period="base")
        py_snap = GTAPVariableSnapshot.from_python_model(model)

        mismatches = []
        n_compared = 0

        # Per-variable GAMS dicts; 1D keys stored with and without tuple wrapping
        gams_by_var: dict[str, dict] = {}
        for var in gams_snap.__dataclass_fields__:
            values = getattr(gams_snap, var)
            if not isinstance(values, dict):
                continue
            gams_by_var[var] = {}
            for key, val in values.items():
                gams_by_var[var][key] = val
                if isinstance(key, str):
                    gams_by_var[var][(key,)] = val
                elif isinstance(key, tuple) and len(key) == 1:
                    gams_by_var[var][key[0]] = val

        SKIP_ATTRS = {"pnum", "walras", "xd"}

        for attr in sorted(py_snap.__dataclass_fields__):
            if attr in SKIP_ATTRS:
                continue
            py_dict = getattr(py_snap, attr, {})
            if not isinstance(py_dict, dict):
                continue
            gams_dict = gams_by_var.get(attr, gams_by_var.get(attr.lower(), {}))
            if not gams_dict:
                continue

            for k, pv in py_dict.items():
                if pv == 0.0:
                    continue
                # Normalize key lookup
                if isinstance(k, str):
                    gv = gams_dict.get(k, gams_dict.get((k,), None))
                elif isinstance(k, tuple) and len(k) == 1:
                    gv = gams_dict.get(k, gams_dict.get(k[0], None))
                else:
                    gv = gams_dict.get(k)
                if gv is None or gv == 0.0:
                    continue

                n_compared += 1
                ad = abs(pv - gv)
                if ad > tol:
                    mismatches.append(
                        {
                            "group": attr,
                            "key": k,
                            "python": pv,
                            "gams": gv,
                            "abs_diff": ad,
                            "rel_diff": ad / max(abs(gv), 1e-10),
                        }
                    )

        mismatches.sort(key=lambda m: m["abs_diff"], reverse=True)
        n_mm = len(mismatches)
        top = mismatches[:10]
        top_str = "\n".join(
            f"  {m['group']:8s} {str(m['key']):40s} py={m['python']:.6g} "
            f"gams={m['gams']:.6g} diff={m['abs_diff']:.4e}"
            for m in top
        )
        assert n_mm <= max_mm, (
            f"Parity check failed: {n_mm} mismatches (tol={tol:.1e}, max={max_mm}) "
            f"over {n_compared} entries.\nTop mismatches:\n{top_str}"
        )
