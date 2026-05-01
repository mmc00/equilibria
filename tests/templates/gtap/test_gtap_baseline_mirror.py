"""Parity test: GTAP Python baseline vs GAMS COMP_generated.csv.

This test enforces the mirror invariant: after solving the baseline with
PATH, the Python solution must match the GAMS reference within tolerance.

Marked @pytest.mark.slow (runtime ~3 min) — add to nightly CI job.

Run manually:
    EQUILIBRIA_GTAP_CAL_DUMP="notes/tmp/18700775-gams_cal_dump_9x10.gdx" \\
    uv run pytest tests/templates/gtap/test_gtap_baseline_mirror.py -v -s
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip condition: slow test and requires external GDX files
# ---------------------------------------------------------------------------
_GDX_FILE = Path(
    os.environ.get(
        "EQUILIBRIA_GTAP_GDX",
        "/Users/marmol/proyectos2/cge_babel/standard_gtap_7/basedata-9x10.gdx",
    )
)
_CAL_DUMP = Path(
    os.environ.get(
        "EQUILIBRIA_GTAP_CAL_DUMP",
        "notes/tmp/18700775-gams_cal_dump_9x10.gdx",
    )
)
_COMP_CSV = Path(
    "src/equilibria/templates/reference/gtap/comp/COMP_generated.csv"
)

pytestmark = pytest.mark.slow


def _skip_if_missing():
    if not _GDX_FILE.exists():
        pytest.skip(f"GDX file not found: {_GDX_FILE}")
    if not _CAL_DUMP.exists():
        pytest.skip(f"Cal dump not found: {_CAL_DUMP}")
    if not _COMP_CSV.exists():
        pytest.skip(f"COMP_generated.csv not found: {_COMP_CSV}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gams_csv(csv_path: Path) -> dict[tuple, float]:
    """Read COMP_generated.csv into a flat dict keyed by (variable, *indices)."""
    data: dict[Any, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yr = (row.get("Year") or "").strip()
            if yr not in {"1", "1.0"}:
                continue
            var = (row.get("Variable") or "").strip().lower()
            reg = (row.get("Region") or "").strip()
            sec = (row.get("Sector") or "").strip()
            qual = (row.get("Qualifier") or "").strip()
            try:
                val = float((row.get("Value") or "0").strip())
            except (ValueError, TypeError):
                continue
            if reg and sec and qual:
                data[(var, reg, sec, qual)] = val
            elif reg and sec:
                data[(var, reg, sec)] = val
            elif reg:
                data[(var, reg)] = val
            else:
                data[(var,)] = val
    return data


def _run_baseline_solve():
    """Build model, solve, and return (model, solver_result)."""
    os.environ["EQUILIBRIA_GTAP_CAL_DUMP"] = str(_CAL_DUMP)

    from equilibria.templates.gtap import GTAPParameters, GTAPSets
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_solver import GTAPSolver
    from equilibria.templates.gtap import build_gtap_contract

    params = GTAPParameters()
    params.load_from_gdx(_GDX_FILE)
    contract = build_gtap_contract("gtap_standard7_9x10")
    equations = GTAPModelEquations(params.sets, params, contract.closure)
    model = equations.build_model()
    equations.apply_production_scaling(model)

    # Use PATH C API nonlinear solver
    from path_capi_python import PathSolverNonlinear

    result = PathSolverNonlinear(model, params).solve()
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

    RESIDUAL_TOL = 1e-4      # solver must reach this residual
    MIRROR_TOL_LOOSE = 5e-3  # absolute tolerance for macro-level comparison
    MIRROR_TOL_STRICT = 1e-3  # tolerance for strict gate (post Fase 2)
    MAX_MISMATCHES_LOOSE = 0  # with tol=5e-3: 0 non-xd mismatches allowed
    MAX_MISMATCHES_STRICT = 5  # with tol=1e-3: at most 5 (yi/xds/xet) before full fix

    def test_solver_converged(self, baseline_solve_result):
        """Solver must converge."""
        _, result = baseline_solve_result
        assert result.converged, (
            f"Solver did not converge: status={result.status}, "
            f"residual={result.residual:.3e}"
        )

    def test_residual_below_threshold(self, baseline_solve_result):
        """Residual must be below RESIDUAL_TOL after T1.1 fix."""
        _, result = baseline_solve_result
        assert result.residual < self.RESIDUAL_TOL, (
            f"Residual {result.residual:.3e} exceeds threshold {self.RESIDUAL_TOL:.3e}. "
            f"Worst eq: {result.worst_equation}"
        )

    def test_walras_law(self, baseline_solve_result):
        """Walras residual must be near zero."""
        _, result = baseline_solve_result
        walras = getattr(result, "walras", None)
        if walras is not None:
            assert abs(walras) < 1e-6, f"Walras residual {walras:.3e} is too large"

    def test_gams_parity_loose(self, baseline_solve_result):
        """All non-xd variables must match GAMS within 5e-3 (0 mismatches allowed)."""
        _skip_if_missing()
        model, _ = baseline_solve_result
        self._check_parity(model, tol=self.MIRROR_TOL_LOOSE, max_mm=self.MAX_MISMATCHES_LOOSE)

    def test_gams_parity_strict(self, baseline_solve_result):
        """All non-xd variables must match GAMS within 1e-3 (up to 5 mismatches tolerated)."""
        _skip_if_missing()
        model, _ = baseline_solve_result
        self._check_parity(model, tol=self.MIRROR_TOL_STRICT, max_mm=self.MAX_MISMATCHES_STRICT)

    # ------------------------------------------------------------------

    def _check_parity(self, model, *, tol: float, max_mm: int):
        from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot

        gams_flat = _load_gams_csv(_COMP_CSV)
        py_snap = GTAPVariableSnapshot.from_python_model(model)

        mismatches = []
        n_compared = 0

        # Build per-variable GAMS dicts
        gams_by_var: dict[str, dict] = {}
        for key, val in gams_flat.items():
            var = key[0]
            rest = key[1:]
            if var not in gams_by_var:
                gams_by_var[var] = {}
            # Store with and without tuple wrapping for 1D
            if len(rest) == 1:
                gams_by_var[var][rest[0]] = val
                gams_by_var[var][rest] = val
            else:
                gams_by_var[var][rest] = val

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
                    mismatches.append({
                        "group": attr, "key": k,
                        "python": pv, "gams": gv,
                        "abs_diff": ad, "rel_diff": ad / max(abs(gv), 1e-10),
                    })

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
